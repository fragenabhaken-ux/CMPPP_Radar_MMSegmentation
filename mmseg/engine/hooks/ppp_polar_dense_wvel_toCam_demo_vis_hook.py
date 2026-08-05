from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from aumovio.fisheye import FisheyeCamera, PinholeCamera
from mmengine.dist import is_main_process
from mmengine.hooks import Hook
from mmengine.model import is_model_wrapper
from mmseg.registry import HOOKS

import matplotlib

matplotlib.use("Agg")

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.figure import Figure


@HOOKS.register_module()
class PPPPolarDenseCameraVelocityDemoVisHook(Hook):
    """Create dense two-panel demo frames in the camera domain.

    Layout:
        left:  camera image with projected ground-truth radar points
        right: camera image with projected sampled predicted radar points

    Both point sets are colored with the same velocity color scale.

    Prediction channels:
        0: log intensity with respect to dr dtheta
        1: raw velocity mark
        2: raw RCS mark (unused here)

    GT channels:
        0: binary occupancy
        1: GT velocity
        2: GT RCS (unused here)

    Required sample metainfo:
        rad2cam: radar-to-camera transform with shape [3, 4]
        real_hw: original fisheye image size (height, width)
        fovs:     output pinhole field of view (horizontal, vertical)

    The prediction is first sampled as an inhomogeneous PPP in the polar BEV
    domain. The sampled BEV points and the occupied GT cell centres are then
    transformed identically from radar coordinates into the camera image.
    """

    def __init__(
        self,
        train_interval: int = 0,
        max_samples_train: int = 0,
        max_samples_val: int = 0,
        max_samples_test: int = 10,
        out_dir: str = "demo_frames_polar_to_camera_velocity",
        img_mean=None,
        img_std=None,
        r_min: float = 0.0,
        r_max: float = 104.0,
        theta_min: float = -1.5707963267948966,
        theta_max: float = 1.5707963267948966,
        velocity_mode: str = "signed",
        velocity_link: str = "cube",
        velocity_vmin: float = -11.0,
        velocity_vmax: float = 11.0,
        velocity_cmap: str = "coolwarm",
        gt_point_size: float = 14.0,
        pred_point_size: float | None = None,
        point_alpha: float = 0.95,
        point_edge_color: str = "black",
        point_edge_width: float = 0.35,
        rel_x_centre: float = 0.491,
        rel_y_centre: float = 0.4370,
        sampling_seed_offset: int = 1000,
        output_width_px: int = 2560,
        bottom_fraction: float = 0.105,
        panel_gap_fraction: float = 0.002,
        dpi: int = 100,
        image_ext: str = "png",
        write_all_ranks: bool = False,
    ) -> None:
        self.train_interval = int(train_interval)
        self.max_samples = {
            "train": int(max_samples_train),
            "val": int(max_samples_val),
            "test": int(max_samples_test),
        }

        self.out_dir = str(out_dir)
        self.image_ext = str(image_ext).lstrip(".")
        self.write_all_ranks = bool(write_all_ranks)

        self.r_min = float(r_min)
        self.r_max = float(r_max)
        self.theta_min = float(theta_min)
        self.theta_max = float(theta_max)

        self.velocity_mode = str(velocity_mode).lower()
        self.velocity_link = str(velocity_link).lower()
        self.velocity_vmin = float(velocity_vmin)
        self.velocity_vmax = float(velocity_vmax)
        self.velocity_cmap = str(velocity_cmap)

        self.gt_point_size = float(gt_point_size)
        self.pred_point_size = (
            float(pred_point_size)
            if pred_point_size is not None
            else float(gt_point_size)
        )
        self.point_alpha = float(point_alpha)
        self.point_edge_color = str(point_edge_color)
        self.point_edge_width = float(point_edge_width)

        self.rel_x_centre = float(rel_x_centre)
        self.rel_y_centre = float(rel_y_centre)
        self.sampling_seed_offset = int(sampling_seed_offset)

        self.output_width_px = int(output_width_px)
        self.bottom_fraction = float(bottom_fraction)
        self.panel_gap_fraction = float(panel_gap_fraction)
        self.dpi = int(dpi)

        self.img_mean = (
            None
            if img_mean is None
            else np.asarray(img_mean, dtype=np.float32)[:, None, None]
        )
        self.img_std = (
            None
            if img_std is None
            else np.asarray(img_std, dtype=np.float32)[:, None, None]
        )

        if self.r_min >= self.r_max:
            raise ValueError(
                f"r_min must be smaller than r_max, got {self.r_min} >= {self.r_max}."
            )
        if self.theta_min >= self.theta_max:
            raise ValueError(
                "theta_min must be smaller than theta_max, got "
                f"{self.theta_min} >= {self.theta_max}."
            )
        if self.velocity_mode not in {"signed", "absolute"}:
            raise ValueError(
                "velocity_mode must be 'signed' or 'absolute', "
                f"got {self.velocity_mode!r}."
            )
        if self.velocity_link not in {"identity", "cube"}:
            raise ValueError(
                "velocity_link must be 'identity' or 'cube', "
                f"got {self.velocity_link!r}."
            )
        if self.velocity_vmin >= self.velocity_vmax:
            raise ValueError(
                "velocity_vmin must be smaller than velocity_vmax, got "
                f"{self.velocity_vmin} >= {self.velocity_vmax}."
            )
        if self.output_width_px <= 0:
            raise ValueError("output_width_px must be positive.")
        if not 0.0 <= self.bottom_fraction < 0.5:
            raise ValueError("bottom_fraction must be in [0, 0.5).")
        if not 0.0 <= self.panel_gap_fraction < 0.1:
            raise ValueError("panel_gap_fraction must be in [0, 0.1).")

        self._written_in_run = {"val": 0, "test": 0}
        self._cam_cache: dict[tuple[Any, ...], PinholeCamera] = {}

    def _rank_allowed(self) -> bool:
        return self.write_all_ranks or is_main_process()

    def before_val_epoch(self, runner) -> None:
        self._written_in_run["val"] = 0

    def before_test_epoch(self, runner) -> None:
        self._written_in_run["test"] = 0

    def after_train_iter(self, runner, batch_idx, data_batch=None, outputs=None) -> None:
        del batch_idx, outputs
        if not self._rank_allowed() or not isinstance(data_batch, dict):
            return
        if self.max_samples["train"] <= 0 or self.train_interval <= 0:
            return
        if runner.iter % self.train_interval != 0:
            return

        self._run(
            runner=runner,
            data_batch=data_batch,
            phase="train",
            step=runner.iter,
        )

    def after_val_iter(self, runner, batch_idx, data_batch=None, outputs=None) -> None:
        del batch_idx, outputs
        if not self._rank_allowed() or not isinstance(data_batch, dict):
            return
        if self.max_samples["val"] <= 0:
            return
        if self._written_in_run["val"] >= self.max_samples["val"]:
            return

        self._run(
            runner=runner,
            data_batch=data_batch,
            phase="val",
            step=runner.iter,
        )

    def after_test_iter(self, runner, batch_idx, data_batch=None, outputs=None) -> None:
        del batch_idx, outputs
        if not self._rank_allowed() or not isinstance(data_batch, dict):
            return
        if self.max_samples["test"] <= 0:
            return
        if self._written_in_run["test"] >= self.max_samples["test"]:
            return

        self._run(
            runner=runner,
            data_batch=data_batch,
            phase="test",
            step=runner.iter,
        )

    def _run(self, runner, data_batch, phase: str, step: int) -> None:
        model = runner.model.module if is_model_wrapper(runner.model) else runner.model

        was_training = model.training
        model.eval()

        proc = model.data_preprocessor(data_batch)
        inputs = proc["inputs"]
        data_samples = proc["data_samples"]

        with torch.no_grad():
            feats = model.extract_feat(inputs)
            preds = model.decode_head(feats)

        if not torch.is_tensor(preds):
            raise TypeError(
                "PPPPolarDenseCameraVelocityDemoVisHook expects decode_head output "
                f"to be a tensor, got {type(preds)}."
            )
        if preds.ndim != 4 or preds.shape[1] != 3:
            raise ValueError(
                "Expected dense polar predictions with shape [B, 3, H, W], got "
                f"{tuple(preds.shape)}."
            )

        if phase == "train":
            n_vis = min(self.max_samples["train"], preds.shape[0])
            start_idx = 0
        else:
            remaining = self.max_samples[phase] - self._written_in_run[phase]
            if remaining <= 0:
                if was_training:
                    model.train()
                return
            n_vis = min(remaining, preds.shape[0])
            start_idx = self._written_in_run[phase]

        for i in range(n_vis):
            self._visualize_sample(
                runner=runner,
                step=step,
                phase=phase,
                img_tensor=inputs[i],
                pred_grid=preds[i],
                sample=data_samples[i],
                sample_idx=start_idx + i,
            )

        if phase in self._written_in_run:
            self._written_in_run[phase] += n_vis

        if was_training:
            model.train()

    def _decode_velocity(self, raw_velocity: torch.Tensor) -> np.ndarray:
        raw_velocity = raw_velocity.detach().float().cpu()
        if self.velocity_link == "identity":
            velocity = raw_velocity
        elif self.velocity_link == "cube":
            velocity = raw_velocity.pow(3)
        else:
            raise RuntimeError(f"Unsupported velocity_link={self.velocity_link!r}.")

        if self.velocity_mode == "absolute":
            velocity = velocity.abs()

        return velocity.numpy()

    def _denorm_img(self, x: np.ndarray) -> np.ndarray:
        x = x.astype(np.float32)
        if self.img_mean is not None and self.img_std is not None:
            x = x * self.img_std + self.img_mean
            x = (x + 1.0) * 127.5
        else:
            if x.min() >= -1.01 and x.max() <= 1.01:
                x = (x + 1.0) * 127.5
            elif x.max() <= 1.01:
                x = x * 255.0
        x = np.clip(x, 0, 255).astype(np.uint8)
        return np.transpose(x, (1, 2, 0))

    @staticmethod
    def _extract_gt(sample) -> torch.Tensor:
        if not hasattr(sample, "gt_sem_seg"):
            raise AttributeError("Data sample has no gt_sem_seg field.")

        gt = sample.gt_sem_seg.data
        if not torch.is_tensor(gt):
            raise TypeError(
                "Expected sample.gt_sem_seg.data to be a tensor, got "
                f"{type(gt)}."
            )

        if gt.ndim == 4:
            if gt.shape[0] != 1:
                raise ValueError(
                    "Unexpected batched GT inside one data sample: "
                    f"{tuple(gt.shape)}."
                )
            gt = gt[0]

        if gt.ndim != 3 or gt.shape[0] != 3:
            raise ValueError(
                "Expected dense polar GT with shape [3, H, W], got "
                f"{tuple(gt.shape)}."
            )

        return gt.detach().float().cpu()

    @staticmethod
    def _tagify(value: str) -> str:
        return "".join(c if c.isalnum() or c in "._-/" else "_" for c in value)

    @staticmethod
    def _meta_array(meta: dict, key: str, shape: tuple[int, ...]) -> np.ndarray:
        if key not in meta:
            raise KeyError(
                f"Missing sample.metainfo[{key!r}]. Add it in PPPPackSegInputs "
                "before using PPPPolarDenseCameraVelocityDemoVisHook."
            )
        value = meta[key]
        if torch.is_tensor(value):
            value = value.detach().cpu().numpy()
        array = np.asarray(value)
        if array.shape != shape:
            raise ValueError(
                f"Expected metainfo[{key!r}] with shape {shape}, got {array.shape}."
            )
        return array

    @staticmethod
    def _meta_pair(meta: dict, key: str, dtype) -> tuple:
        if key not in meta:
            raise KeyError(
                f"Missing sample.metainfo[{key!r}]. Add it in PPPPackSegInputs "
                "before using PPPPolarDenseCameraVelocityDemoVisHook."
            )
        value = meta[key]
        if torch.is_tensor(value):
            value = value.detach().cpu().numpy()
        array = np.asarray(value).reshape(-1)
        if array.size != 2:
            raise ValueError(
                f"Expected metainfo[{key!r}] with two values, got shape {array.shape}."
            )
        return dtype(array[0]), dtype(array[1])

    def _polar_geometry(self, h_r: int, w_theta: int) -> dict:
        dr = (self.r_max - self.r_min) / h_r
        dtheta = (self.theta_max - self.theta_min) / w_theta

        r_centres_1d = self.r_min + (
            np.arange(h_r, dtype=np.float64) + 0.5
        ) * dr
        theta_centres_1d = self.theta_min + (
            np.arange(w_theta, dtype=np.float64) + 0.5
        ) * dtheta

        rr_centres, tt_centres = np.meshgrid(
            r_centres_1d,
            theta_centres_1d,
            indexing="ij",
        )

        return {
            "dr": dr,
            "dtheta": dtheta,
            "x_centres": rr_centres * np.cos(tt_centres),
            "y_centres": rr_centres * np.sin(tt_centres),
        }

    def _sample_dense_polar_ppp(
        self,
        lam: np.ndarray,
        velocity: np.ndarray,
        geometry: dict,
        seed: int,
    ) -> dict:
        dr = float(geometry["dr"])
        dtheta = float(geometry["dtheta"])
        h_r, w_theta = lam.shape

        r_edges = np.linspace(self.r_min, self.r_max, h_r + 1)
        theta_edges = np.linspace(self.theta_min, self.theta_max, w_theta + 1)

        cell_mass = np.asarray(lam, dtype=np.float64) * dr * dtheta
        cell_mass = np.clip(cell_mass, 0.0, None)

        rng = np.random.default_rng(int(seed) + self.sampling_seed_offset)
        counts = rng.poisson(cell_mass)

        ri, ti = np.nonzero(counts > 0)
        if len(ri) == 0:
            return {
                "xy": np.zeros((0, 2), dtype=np.float64),
                "velocity": np.zeros((0,), dtype=np.float64),
                "count": 0,
            }

        reps = counts[ri, ti].astype(np.int64)
        total = int(reps.sum())
        ri_rep = np.repeat(ri, reps)
        ti_rep = np.repeat(ti, reps)

        r0 = r_edges[ri_rep]
        r1 = r_edges[ri_rep + 1]
        t0 = theta_edges[ti_rep]
        t1 = theta_edges[ti_rep + 1]

        r = r0 + rng.random(total) * (r1 - r0)
        theta = t0 + rng.random(total) * (t1 - t0)

        x_forward = r * np.cos(theta)
        y_lateral = r * np.sin(theta)

        return {
            "xy": np.stack([x_forward, y_lateral], axis=1),
            "velocity": velocity[ri_rep, ti_rep].astype(np.float64),
            "count": total,
        }

    def _get_pinhole_cam(
        self,
        image_h: int,
        image_w: int,
        real_hw: tuple[int, int],
        fovs: tuple[float, float],
    ) -> PinholeCamera:
        real_h, real_w = real_hw
        h_fov, v_fov = fovs

        key = (
            int(image_h),
            int(image_w),
            int(real_h),
            int(real_w),
            float(h_fov),
            float(v_fov),
            self.rel_x_centre,
            self.rel_y_centre,
        )

        cam = self._cam_cache.get(key)
        if cam is None:
            cam = PinholeCamera(
                height=int(image_h),
                width=int(image_w),
                hFov=float(h_fov),
                vFov=float(v_fov),
                rel_x_centre=self.rel_x_centre,
                rel_y_centre=self.rel_y_centre,
                fisheye_cam=FisheyeCamera(
                    height=int(real_h),
                    width=int(real_w),
                ),
            )
            self._cam_cache[key] = cam
        return cam

    def _project_bev_points(
        self,
        xy: np.ndarray,
        values: np.ndarray,
        rad2cam: np.ndarray,
        pinhole_cam: PinholeCamera,
        image_h: int,
        image_w: int,
    ) -> dict:
        xy = np.asarray(xy, dtype=np.float64)
        values = np.asarray(values, dtype=np.float64)

        if xy.ndim != 2 or xy.shape[1] != 2:
            raise ValueError(f"Expected xy with shape [N, 2], got {xy.shape}.")
        if values.ndim != 1 or values.shape[0] != xy.shape[0]:
            raise ValueError(
                "Expected one value per BEV point, got "
                f"xy={xy.shape}, values={values.shape}."
            )
        if len(xy) == 0:
            return {
                "x_pix": np.zeros((0,), dtype=np.float64),
                "y_pix": np.zeros((0,), dtype=np.float64),
                "values": np.zeros((0,), dtype=np.float64),
                "count": 0,
            }

        # Identical radar -> camera transform used by BuildPPPHeatmap.
        cam_pts = xy @ rad2cam[:3, :2].T + rad2cam[:3, 3:].T

        z = cam_pts[:, 2]
        finite_cam = np.all(np.isfinite(cam_pts), axis=1)
        valid_depth = finite_cam & (z > 1e-6)

        if not np.any(valid_depth):
            return {
                "x_pix": np.zeros((0,), dtype=np.float64),
                "y_pix": np.zeros((0,), dtype=np.float64),
                "values": np.zeros((0,), dtype=np.float64),
                "count": 0,
            }

        cam_pts = cam_pts[valid_depth]
        values = values[valid_depth]
        z = cam_pts[:, 2]

        u = cam_pts[:, 0] / z
        v = cam_pts[:, 1] / z
        x_pix, y_pix = pinhole_cam.real_world_to_pinhole(u, v)

        x_pix = np.asarray(x_pix, dtype=np.float64)
        y_pix = np.asarray(y_pix, dtype=np.float64)

        valid = (
            np.isfinite(x_pix)
            & np.isfinite(y_pix)
            & np.isfinite(values)
            & (x_pix >= 0.0)
            & (x_pix < float(image_w))
            & (y_pix >= 0.0)
            & (y_pix < float(image_h))
        )

        return {
            "x_pix": x_pix[valid],
            "y_pix": y_pix[valid],
            "values": values[valid],
            "count": int(valid.sum()),
        }

    @staticmethod
    def _draw_label(ax, text: str) -> None:
        ax.text(
            0.012,
            0.975,
            text,
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=11,
            color="white",
            bbox={
                "facecolor": "black",
                "edgecolor": "none",
                "alpha": 0.62,
                "pad": 4,
            },
            zorder=5,
        )

    def _plot_panel(
        self,
        ax,
        image: np.ndarray,
        projected: dict,
        point_size: float,
        label: str,
    ) -> None:
        ax.imshow(image, origin="upper", interpolation="nearest")
        if projected["count"] > 0:
            ax.scatter(
                projected["x_pix"],
                projected["y_pix"],
                s=point_size,
                c=projected["values"],
                cmap=self.velocity_cmap,
                vmin=self.velocity_vmin,
                vmax=self.velocity_vmax,
                alpha=self.point_alpha,
                edgecolors=self.point_edge_color,
                linewidths=self.point_edge_width,
                zorder=3,
            )
        # self._draw_label(ax, label)
        ax.set_xlim(0, image.shape[1])
        ax.set_ylim(image.shape[0], 0)
        ax.axis("off")

    def _make_figure(self, image_h: int, image_w: int) -> Figure:
        width_in = self.output_width_px / self.dpi
        panel_image_height_fraction = 1.0 - self.bottom_fraction
        panel_width_fraction = (1.0 - self.panel_gap_fraction) / 2.0
        image_height_px = (
            self.output_width_px
            * panel_width_fraction
            * float(image_h)
            / float(image_w)
        )
        total_height_px = image_height_px / panel_image_height_fraction
        height_in = total_height_px / self.dpi

        fig = Figure(
            figsize=(width_in, height_in),
            dpi=self.dpi,
            facecolor="black",
        )
        FigureCanvasAgg(fig)
        return fig

    def _output_path(
        self,
        runner,
        phase: str,
        video: str,
        frame_str: str,
        global_idx_str: str,
    ) -> Path:
        root = Path(self.out_dir)
        if not root.is_absolute():
            root = Path(runner.work_dir) / root

        out_dir = root / phase / video
        out_dir.mkdir(parents=True, exist_ok=True)
        return out_dir / f"frame_{frame_str}_idx_{global_idx_str}.{self.image_ext}"

    def _visualize_sample(
        self,
        runner,
        step: int,
        phase: str,
        img_tensor: torch.Tensor,
        pred_grid: torch.Tensor,
        sample,
        sample_idx: int,
    ) -> None:
        del step

        image = self._denorm_img(img_tensor.detach().cpu().numpy())
        image_h, image_w = image.shape[:2]

        pred_grid = pred_grid.detach().float().cpu()
        gt = self._extract_gt(sample)
        if pred_grid.shape != gt.shape:
            raise ValueError(
                "Prediction and GT grids must have identical shapes, got "
                f"prediction={tuple(pred_grid.shape)}, GT={tuple(gt.shape)}."
            )

        lam = torch.exp(pred_grid[0]).numpy()
        pred_velocity_grid = self._decode_velocity(pred_grid[1])

        occupancy = gt[0].numpy()
        gt_velocity_grid = gt[1].numpy()
        unique_occupancy = np.unique(occupancy)
        if not np.all(np.isin(unique_occupancy, [0.0, 1.0])):
            raise ValueError(
                "Dense polar occupancy must be binary, found "
                f"{unique_occupancy.tolist()}."
            )

        occupied = occupancy > 0.0
        gt_count = int(occupied.sum())
        if self.velocity_mode == "absolute":
            gt_velocity_grid = np.abs(gt_velocity_grid)

        h_r, w_theta = lam.shape
        geometry = self._polar_geometry(h_r=h_r, w_theta=w_theta)

        gt_xy = np.stack(
            [
                geometry["x_centres"][occupied],
                geometry["y_centres"][occupied],
            ],
            axis=1,
        )
        gt_velocity = gt_velocity_grid[occupied].astype(np.float64, copy=False)

        dr = float(geometry["dr"])
        dtheta = float(geometry["dtheta"])
        expected_count = float((lam * dr * dtheta).sum())

        meta = sample.metainfo
        rad2cam = self._meta_array(meta, "rad2cam", (3, 4)).astype(
            np.float64,
            copy=False,
        )
        real_hw = self._meta_pair(meta, "real_hw", int)
        fovs = self._meta_pair(meta, "fovs", float)
        pinhole_cam = self._get_pinhole_cam(
            image_h=image_h,
            image_w=image_w,
            real_hw=real_hw,
            fovs=fovs,
        )

        frame = meta.get("video_frame_idx")
        global_idx = meta.get("global_idx", meta.get("frame_idx"))
        seed_base = int(frame) if frame is not None else int(
            global_idx if global_idx is not None else sample_idx
        )

        pred_pack = self._sample_dense_polar_ppp(
            lam=lam,
            velocity=pred_velocity_grid,
            geometry=geometry,
            seed=seed_base,
        )

        gt_projected = self._project_bev_points(
            xy=gt_xy,
            values=gt_velocity,
            rad2cam=rad2cam,
            pinhole_cam=pinhole_cam,
            image_h=image_h,
            image_w=image_w,
        )
        pred_projected = self._project_bev_points(
            xy=pred_pack["xy"],
            values=pred_pack["velocity"],
            rad2cam=rad2cam,
            pinhole_cam=pinhole_cam,
            image_h=image_h,
            image_w=image_w,
        )

        fig = self._make_figure(image_h=image_h, image_w=image_w)
        gap = self.panel_gap_fraction
        panel_width = (1.0 - gap) / 2.0
        image_bottom = self.bottom_fraction
        image_height = 1.0 - image_bottom

        ax_gt = fig.add_axes([0.0, image_bottom, panel_width, image_height])
        ax_pred = fig.add_axes(
            [panel_width + gap, image_bottom, panel_width, image_height]
        )

        self._plot_panel(
            ax=ax_gt,
            image=image,
            projected=gt_projected,
            point_size=self.gt_point_size,
            label=(
                "Ground truth radar points"
                f"  |  projected {gt_projected['count']}/{gt_count}"
            ),
        )
        self._plot_panel(
            ax=ax_pred,
            image=image,
            projected=pred_projected,
            point_size=self.pred_point_size,
            label=(
                "Predicted radar points"
                f"  |  projected {pred_projected['count']}/{pred_pack['count']}"
                f"  |  E[N]={expected_count:.1f}"
            ),
        )

        cax = fig.add_axes([0.35, 0.024, 0.30, 0.032])
        norm = Normalize(vmin=self.velocity_vmin, vmax=self.velocity_vmax)
        mappable = ScalarMappable(norm=norm, cmap=self.velocity_cmap)
        mappable.set_array([])
        colorbar = fig.colorbar(mappable, cax=cax, orientation="horizontal")
        colorbar.ax.tick_params(
            colors="white",
            labelsize=8,
            pad=1,
            length=2,
        )
        colorbar.outline.set_edgecolor("white")
        colorbar.set_label(
            "velocity [m/s]"
            if self.velocity_mode == "signed"
            else "|velocity| [m/s]",
            color="white",
            fontsize=8,
            labelpad=1,
        )
        cax.set_facecolor("black")

        video = self._tagify(str(meta.get("video_name", "unknown_video")))
        frame_str = f"{int(frame):06d}" if frame is not None else "unknown_frame"
        global_idx_str = (
            f"{int(global_idx):08d}"
            if global_idx is not None
            else "unknown_idx"
        )
        out_path = self._output_path(
            runner=runner,
            phase=phase,
            video=video,
            frame_str=frame_str,
            global_idx_str=global_idx_str,
        )
        fig.savefig(
            out_path,
            dpi=self.dpi,
            facecolor="black",
            edgecolor="none",
            pad_inches=0,
        )
        fig.clear()