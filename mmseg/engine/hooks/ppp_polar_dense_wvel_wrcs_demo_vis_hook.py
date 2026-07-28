from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from mmengine.dist import is_main_process
from mmengine.hooks import Hook
from mmengine.model import is_model_wrapper
from mmseg.registry import HOOKS

import matplotlib

matplotlib.use("Agg")

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure


@HOOKS.register_module()
class PPPPolarDenseWVelWRcsDemoVisHook(Hook):
    """Demo visualization hook for dense polar-grid PPP prediction.

    Writes one PNG per selected frame to

        <runner.work_dir>/<out_dir>/<phase>/<video_name>/
            frame_<frame>_idx_<idx>.png

    Layout:
        Row 1:
            - Camera image spanning all three columns.

        Row 2:
            1. Predicted PPP intensity in BEV with GT occupied-cell centres.
            2. Predicted velocity field in BEV with GT velocity-colored points.
            3. Predicted RCS field in BEV with GT RCS-colored points.

        Row 3:
            1. Sampled predicted radar points in BEV, plain color.
            2. Sampled predicted radar points in BEV, colored by predicted velocity.
            3. Sampled predicted radar points in BEV, colored by predicted RCS.

    Prediction channels:
        0: log intensity with respect to dr dtheta
        1: raw velocity mark
        2: raw RCS mark

    GT channels:
        0: binary occupancy
        1: GT velocity
        2: GT RCS
    """

    def __init__(
        self,
        train_interval: int = 0,
        max_samples_train: int = 0,
        max_samples_val: int = 0,
        max_samples_test: int = 10,
        out_dir: str = "demo_frames_polar_dense_wvel_rcs",
        img_mean=None,
        img_std=None,
        r_min: float = 0.0,
        r_max: float = 104.0,
        theta_min: float = -1.5707963267948966,
        theta_max: float = 1.5707963267948966,
        bev_y_min: float = -52.0,
        bev_y_max: float = 52.0,
        intensity_gamma: float = 0.3,
        heatmap_cmap: str = "hot",
        point_size: float = 20.0,
        sample_point_size: float | None = None,
        point_color: str = "white",
        sample_point_color: str = "white",
        point_edge_color: str = "black",
        point_edge_width: float = 0.45,
        velocity_mode: str = "signed",
        velocity_link: str = "cube",
        velocity_vmin: float = -11.0,
        velocity_vmax: float = 11.0,
        velocity_cmap: str = "coolwarm",
        rcs_link: str = "cube",
        rcs_vmin: float = -60.0,
        rcs_vmax: float = 10.0,
        rcs_cmap: str = "viridis",
        sampling_seed_offset: int = 1000,
        dpi: int = 100,
        figsize=(21, 18),
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

        self.bev_y_min = float(bev_y_min)
        self.bev_y_max = float(bev_y_max)

        self.intensity_gamma = float(intensity_gamma)
        self.heatmap_cmap = str(heatmap_cmap)

        self.point_size = float(point_size)
        self.sample_point_size = (
            float(sample_point_size)
            if sample_point_size is not None
            else float(point_size)
        )
        self.point_color = str(point_color)
        self.sample_point_color = str(sample_point_color)
        self.point_edge_color = str(point_edge_color)
        self.point_edge_width = float(point_edge_width)

        self.velocity_mode = str(velocity_mode).lower()
        self.velocity_link = str(velocity_link).lower()
        self.velocity_vmin = float(velocity_vmin)
        self.velocity_vmax = float(velocity_vmax)
        self.velocity_cmap = str(velocity_cmap)

        self.rcs_link = str(rcs_link).lower()
        self.rcs_vmin = float(rcs_vmin)
        self.rcs_vmax = float(rcs_vmax)
        self.rcs_cmap = str(rcs_cmap)

        self.sampling_seed_offset = int(sampling_seed_offset)

        self.dpi = int(dpi)
        self.figsize = tuple(figsize)

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
        if self.bev_y_min >= self.bev_y_max:
            raise ValueError(
                "bev_y_min must be smaller than bev_y_max, got "
                f"{self.bev_y_min} >= {self.bev_y_max}."
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
        if self.rcs_link not in {"identity", "cube"}:
            raise ValueError(
                "rcs_link must be 'identity' or 'cube', "
                f"got {self.rcs_link!r}."
            )
        if self.rcs_vmin >= self.rcs_vmax:
            raise ValueError(
                f"rcs_vmin must be smaller than rcs_vmax, got {self.rcs_vmin} >= {self.rcs_vmax}."
            )

        self._written_in_run = {"val": 0, "test": 0}

    def _rank_allowed(self) -> bool:
        return self.write_all_ranks or is_main_process()

    def before_val_epoch(self, runner) -> None:
        self._written_in_run["val"] = 0

    def before_test_epoch(self, runner) -> None:
        self._written_in_run["test"] = 0

    def after_train_iter(self, runner, batch_idx, data_batch=None, outputs=None) -> None:
        if not self._rank_allowed() or not isinstance(data_batch, dict):
            return
        if self.max_samples["train"] <= 0 or self.train_interval <= 0:
            return
        if runner.iter % self.train_interval != 0:
            return

        self._run(runner=runner, data_batch=data_batch, phase="train", step=runner.iter)

    def after_val_iter(self, runner, batch_idx, data_batch=None, outputs=None) -> None:
        if not self._rank_allowed() or not isinstance(data_batch, dict):
            return
        if self.max_samples["val"] <= 0:
            return
        if self._written_in_run["val"] >= self.max_samples["val"]:
            return

        self._run(runner=runner, data_batch=data_batch, phase="val", step=runner.iter)

    def after_test_iter(self, runner, batch_idx, data_batch=None, outputs=None) -> None:
        if not self._rank_allowed() or not isinstance(data_batch, dict):
            return
        if self.max_samples["test"] <= 0:
            return
        if self._written_in_run["test"] >= self.max_samples["test"]:
            return

        self._run(runner=runner, data_batch=data_batch, phase="test", step=runner.iter)

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
                "PPPPolarDenseWVelWRcsDemoVisHook expects decode_head output to be a tensor, "
                f"got {type(preds)}."
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

    def _decode_rcs(self, raw_rcs: torch.Tensor) -> np.ndarray:
        raw_rcs = raw_rcs.detach().float().cpu()
        if self.rcs_link == "identity":
            rcs = raw_rcs
        elif self.rcs_link == "cube":
            rcs = raw_rcs.pow(3)
        else:
            raise RuntimeError(f"Unsupported rcs_link={self.rcs_link!r}.")
        return rcs.numpy()

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
    def _textbox(ax, text: str, x: float = 0.015, y: float = 0.975) -> None:
        ax.text(
            x,
            y,
            text,
            transform=ax.transAxes,
            fontsize=9,
            color="white",
            va="top",
            ha="left",
            bbox=dict(facecolor="black", alpha=0.60, pad=3),
        )

    def _polar_geometry(self, h_r: int, w_theta: int) -> dict:
        dr = (self.r_max - self.r_min) / h_r
        dtheta = (self.theta_max - self.theta_min) / w_theta

        r_edges = np.linspace(self.r_min, self.r_max, h_r + 1)
        theta_edges = np.linspace(self.theta_min, self.theta_max, w_theta + 1)

        r_centres_1d = self.r_min + (np.arange(h_r, dtype=np.float64) + 0.5) * dr
        theta_centres_1d = self.theta_min + (np.arange(w_theta, dtype=np.float64) + 0.5) * dtheta

        rr_edges, tt_edges = np.meshgrid(r_edges, theta_edges, indexing="ij")
        rr_centres, tt_centres = np.meshgrid(r_centres_1d, theta_centres_1d, indexing="ij")

        return dict(
            dr=dr,
            dtheta=dtheta,
            x_edges=rr_edges * np.cos(tt_edges),
            y_edges=rr_edges * np.sin(tt_edges),
            r_centres=rr_centres,
            theta_centres=tt_centres,
            x_centres=rr_centres * np.cos(tt_centres),
            y_centres=rr_centres * np.sin(tt_centres),
        )

    def _setup_bev_axes(self, ax) -> None:
        ax.set_xlim(self.bev_y_max, self.bev_y_min)
        ax.set_ylim(self.r_min, self.r_max)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("y lateral [m]")
        ax.set_ylabel("x forward [m]")
        ax.grid(True, alpha=0.25)

    def _sample_dense_polar_ppp(
        self,
        lam: np.ndarray,
        velocity: np.ndarray,
        rcs: np.ndarray,
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
                "rcs": np.zeros((0,), dtype=np.float64),
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

        sampled_velocity = velocity[ri_rep, ti_rep].astype(np.float64)
        sampled_rcs = rcs[ri_rep, ti_rep].astype(np.float64)

        return {
            "xy": np.stack([x_forward, y_lateral], axis=1).astype(np.float64),
            "velocity": sampled_velocity,
            "rcs": sampled_rcs,
            "count": total,
        }

    def _plot_heatmap_with_gt(
        self,
        ax,
        y_edges: np.ndarray,
        x_edges: np.ndarray,
        intensity_display: np.ndarray,
        gt_x: np.ndarray,
        gt_y: np.ndarray,
        expected_count: float,
        gt_count: int,
    ) -> None:
        ax.pcolormesh(
            y_edges,
            x_edges,
            intensity_display,
            shading="auto",
            cmap=self.heatmap_cmap,
        )

        if gt_count > 0:
            ax.scatter(
                gt_y,
                gt_x,
                s=self.point_size,
                c=self.point_color,
                alpha=0.97,
                edgecolors=self.point_edge_color,
                linewidths=self.point_edge_width,
                zorder=3,
            )

        self._setup_bev_axes(ax)
        ax.set_title("Predicted PPP intensity with GT occupied cells", fontsize=11, pad=3)
        self._textbox(ax, f"E[N]={expected_count:.1f}\nGT N={gt_count}\nGT dots: plain")

    def _plot_field_with_gt(
        self,
        fig: Figure,
        ax,
        y_edges: np.ndarray,
        x_edges: np.ndarray,
        field: np.ndarray,
        gt_x: np.ndarray,
        gt_y: np.ndarray,
        gt_values: np.ndarray,
        cmap: str,
        vmin: float,
        vmax: float,
        title: str,
        box_text: str,
        cbar_label: str,
    ) -> None:
        pcm = ax.pcolormesh(
            y_edges,
            x_edges,
            field,
            shading="auto",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )

        finite = np.isfinite(gt_values)
        if len(gt_x) > 0 and np.any(finite):
            ax.scatter(
                gt_y[finite],
                gt_x[finite],
                s=self.point_size,
                c=gt_values[finite],
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                alpha=0.98,
                edgecolors=self.point_edge_color,
                linewidths=self.point_edge_width,
                zorder=3,
            )

        self._setup_bev_axes(ax)
        ax.set_title(title, fontsize=11, pad=3)
        self._textbox(ax, box_text)
        cax = ax.inset_axes([0.68, 0.045, 0.28, 0.04])
        cb = fig.colorbar(pcm, cax=cax, orientation="horizontal")
        cb.ax.tick_params(labelsize=7, pad=1, length=2)
        cb.set_label(cbar_label, fontsize=7, labelpad=1)
        cax.set_facecolor((1.0, 1.0, 1.0, 0.85))

    def _plot_samples_plain(self, ax, sample_x: np.ndarray, sample_y: np.ndarray, expected_count: float, sample_count: int) -> None:
        if sample_count > 0:
            ax.scatter(
                sample_y,
                sample_x,
                s=self.sample_point_size,
                c=self.sample_point_color,
                alpha=0.97,
                edgecolors=self.point_edge_color,
                linewidths=self.point_edge_width,
                zorder=3,
            )

        self._setup_bev_axes(ax)
        ax.set_title("Sampled predicted radar points", fontsize=11, pad=3)
        self._textbox(ax, f"sampled N={sample_count}\nE[N]={expected_count:.1f}")

    def _plot_samples_colored(
        self,
        fig: Figure,
        ax,
        sample_x: np.ndarray,
        sample_y: np.ndarray,
        sample_values: np.ndarray,
        sample_count: int,
        cmap: str,
        vmin: float,
        vmax: float,
        title: str,
        box_text: str,
        cbar_label: str,
    ) -> None:
        sc = None
        finite = np.isfinite(sample_values)
        if sample_count > 0 and np.any(finite):
            sc = ax.scatter(
                sample_y[finite],
                sample_x[finite],
                s=self.sample_point_size,
                c=sample_values[finite],
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                alpha=0.98,
                edgecolors=self.point_edge_color,
                linewidths=self.point_edge_width,
                zorder=3,
            )

        self._setup_bev_axes(ax)
        ax.set_title(title, fontsize=11, pad=3)
        self._textbox(ax, box_text)
        if sc is not None:
            cax = ax.inset_axes([0.68, 0.045, 0.28, 0.04])
            cb = fig.colorbar(sc, cax=cax, orientation="horizontal")
            cb.ax.tick_params(labelsize=7, pad=1, length=2)
            cb.set_label(cbar_label, fontsize=7, labelpad=1)
            cax.set_facecolor((1.0, 1.0, 1.0, 0.85))

    def _output_path(self, runner, phase: str, video: str, frame_str: str, global_idx_str: str) -> Path:
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

        img = self._denorm_img(img_tensor.detach().cpu().numpy())
        pred_grid = pred_grid.detach().float().cpu()
        gt = self._extract_gt(sample)

        if pred_grid.shape != gt.shape:
            raise ValueError(
                "Prediction and GT grids must have identical shapes, got "
                f"prediction={tuple(pred_grid.shape)}, GT={tuple(gt.shape)}."
            )

        log_lambda = pred_grid[0]
        raw_velocity = pred_grid[1]
        raw_rcs = pred_grid[2]

        lam = torch.exp(log_lambda).numpy()
        velocity = self._decode_velocity(raw_velocity)
        rcs = self._decode_rcs(raw_rcs)

        occupancy = gt[0].numpy()
        gt_velocity_grid = gt[1].numpy()
        gt_rcs_grid = gt[2].numpy()

        unique_occupancy = np.unique(occupancy)
        if not np.all(np.isin(unique_occupancy, [0.0, 1.0])):
            raise ValueError(
                f"Dense polar occupancy must be binary, found {unique_occupancy.tolist()}."
            )

        occupied = occupancy > 0.0
        gt_count = int(occupied.sum())

        h_r, w_theta = lam.shape
        geometry = self._polar_geometry(h_r=h_r, w_theta=w_theta)

        dr = float(geometry["dr"])
        dtheta = float(geometry["dtheta"])
        cell_mass = lam * dr * dtheta
        expected_count = float(cell_mass.sum())

        x_edges = geometry["x_edges"]
        y_edges = geometry["y_edges"]
        x_centres = geometry["x_centres"]
        y_centres = geometry["y_centres"]

        gt_x = x_centres[occupied]
        gt_y = y_centres[occupied]
        gt_velocity = gt_velocity_grid[occupied].astype(np.float64, copy=False)
        gt_rcs = gt_rcs_grid[occupied].astype(np.float64, copy=False)

        if self.velocity_mode == "absolute":
            gt_velocity = np.abs(gt_velocity)

        meta = sample.metainfo
        video = self._tagify(str(meta.get("video_name", "unknown_video")))
        frame = meta.get("video_frame_idx")
        global_idx = meta.get("global_idx", meta.get("frame_idx"))

        frame_str = f"{int(frame):06d}" if frame is not None else "unknown_frame"
        global_idx_str = f"{int(global_idx):08d}" if global_idx is not None else "unknown_idx"

        seed_base = int(frame) if frame is not None else int(global_idx if global_idx is not None else sample_idx)
        sample_pack = self._sample_dense_polar_ppp(
            lam=lam,
            velocity=velocity,
            rcs=rcs,
            geometry=geometry,
            seed=seed_base,
        )
        sample_xy = sample_pack["xy"]
        sample_x = sample_xy[:, 0] if len(sample_xy) else np.zeros((0,), dtype=np.float64)
        sample_y = sample_xy[:, 1] if len(sample_xy) else np.zeros((0,), dtype=np.float64)
        sample_velocity = sample_pack["velocity"]
        sample_rcs = sample_pack["rcs"]
        sample_count = int(sample_pack["count"])

        fig = Figure(figsize=self.figsize, dpi=self.dpi)
        FigureCanvasAgg(fig)

        gs = fig.add_gridspec(
            nrows=3,
            ncols=3,
            height_ratios=(1.0, 1.0, 1.0),
            left=0.045,
            right=0.992,
            top=0.955,
            bottom=0.040,
            hspace=0.22,
            wspace=0.12,
        )

        ax_cam = fig.add_subplot(gs[0, :])
        ax_hm = fig.add_subplot(gs[1, 0])
        ax_vel = fig.add_subplot(gs[1, 1])
        ax_rcs = fig.add_subplot(gs[1, 2])
        ax_s_plain = fig.add_subplot(gs[2, 0])
        ax_s_vel = fig.add_subplot(gs[2, 1])
        ax_s_rcs = fig.add_subplot(gs[2, 2])

        fig.suptitle(
            f"{phase} | {video} | frame {frame_str} | idx {global_idx_str}",
            fontsize=13,
            y=0.985,
        )

        ax_cam.imshow(img, aspect="equal")
        ax_cam.set_title("Camera image", fontsize=12, pad=4)
        ax_cam.axis("off")

        intensity_display = np.power(lam, self.intensity_gamma)
        self._plot_heatmap_with_gt(
            ax=ax_hm,
            y_edges=y_edges,
            x_edges=x_edges,
            intensity_display=intensity_display,
            gt_x=gt_x,
            gt_y=gt_y,
            expected_count=expected_count,
            gt_count=gt_count,
        )

        vel_label = "velocity [m/s]" if self.velocity_mode == "signed" else "|velocity| [m/s]"
        self._plot_field_with_gt(
            fig=fig,
            ax=ax_vel,
            y_edges=y_edges,
            x_edges=x_edges,
            field=velocity,
            gt_x=gt_x,
            gt_y=gt_y,
            gt_values=gt_velocity,
            cmap=self.velocity_cmap,
            vmin=self.velocity_vmin,
            vmax=self.velocity_vmax,
            title="Predicted velocity with GT velocity dots",
            box_text=(
                f"link={self.velocity_link}\n"
                f"mode={self.velocity_mode}\n"
                f"GT velocity dots: N={int(np.isfinite(gt_velocity).sum())}"
            ),
            cbar_label=vel_label,
        )

        self._plot_field_with_gt(
            fig=fig,
            ax=ax_rcs,
            y_edges=y_edges,
            x_edges=x_edges,
            field=rcs,
            gt_x=gt_x,
            gt_y=gt_y,
            gt_values=gt_rcs,
            cmap=self.rcs_cmap,
            vmin=self.rcs_vmin,
            vmax=self.rcs_vmax,
            title="Predicted RCS with GT RCS dots",
            box_text=(
                f"link={self.rcs_link}\n"
                f"GT RCS dots: N={int(np.isfinite(gt_rcs).sum())}"
            ),
            cbar_label="RCS",
        )

        self._plot_samples_plain(
            ax=ax_s_plain,
            sample_x=sample_x,
            sample_y=sample_y,
            expected_count=expected_count,
            sample_count=sample_count,
        )

        self._plot_samples_colored(
            fig=fig,
            ax=ax_s_vel,
            sample_x=sample_x,
            sample_y=sample_y,
            sample_values=sample_velocity,
            sample_count=sample_count,
            cmap=self.velocity_cmap,
            vmin=self.velocity_vmin,
            vmax=self.velocity_vmax,
            title="Sampled predicted points colored by predicted velocity",
            box_text=f"sampled N={sample_count}\nmode={self.velocity_mode}",
            cbar_label=vel_label,
        )

        self._plot_samples_colored(
            fig=fig,
            ax=ax_s_rcs,
            sample_x=sample_x,
            sample_y=sample_y,
            sample_values=sample_rcs,
            sample_count=sample_count,
            cmap=self.rcs_cmap,
            vmin=self.rcs_vmin,
            vmax=self.rcs_vmax,
            title="Sampled predicted points colored by predicted RCS",
            box_text=f"sampled N={sample_count}\nlink={self.rcs_link}",
            cbar_label="RCS",
        )

        out_path = self._output_path(
            runner=runner,
            phase=phase,
            video=video,
            frame_str=frame_str,
            global_idx_str=global_idx_str,
        )
        fig.savefig(out_path, dpi=self.dpi)
        fig.clear()