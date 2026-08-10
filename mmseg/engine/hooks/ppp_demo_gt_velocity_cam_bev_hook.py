"""GT-only radar velocity visualization in camera and Cartesian BEV domains.

For each selected sample this hook writes two independent PNG files:

1. Camera view:
   Ground-truth radar detections projected into the camera frame and colored by
   ground-truth radial velocity.

2. Cartesian BEV:
   The same ground-truth detections in Cartesian coordinates, colored by
   ground-truth radial velocity.

Ground-truth channel convention:
    gt[0] = support mask
    gt[1] = forward depth [m]
    gt[2] = radial velocity [m/s]

The BEV display is intentionally mirrored left/right so that positive lateral
coordinates are shown on the left and negative lateral coordinates on the
right. With the default limits this means x = +55 m is at the left edge and
x = -55 m is at the right edge.

No model prediction or forward pass is performed.
"""

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
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.figure import Figure


def extract_gt_points_and_values(
    mask: np.ndarray,
    values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return occupied GT cell centers and values at those cells."""
    ys, xs = np.where(mask > 0)

    if xs.size == 0:
        return (
            np.zeros((0, 2), dtype=np.float64),
            np.zeros((0,), dtype=np.float64),
        )

    points = np.stack([xs + 0.5, ys + 0.5], axis=1).astype(np.float64)
    point_values = values[ys, xs].astype(np.float64)
    return points, point_values


def scale_points_to_shape(
    points: np.ndarray,
    from_hw: tuple[int, int],
    to_hw: tuple[int, int],
) -> np.ndarray:
    """Scale floating-point (x, y) coordinates between raster shapes."""
    if len(points) == 0:
        return points.astype(np.float64)

    from_h, from_w = int(from_hw[0]), int(from_hw[1])
    to_h, to_w = int(to_hw[0]), int(to_hw[1])

    if from_h == to_h and from_w == to_w:
        return points.astype(np.float64, copy=True)

    result = points.astype(np.float64, copy=True)
    result[:, 0] *= float(to_w) / float(max(from_w, 1))
    result[:, 1] *= float(to_h) / float(max(from_h, 1))
    return result


def recover_lateral_from_forward_depth(
    u_px: np.ndarray,
    forward_m: np.ndarray,
    rad2cam: np.ndarray,
    image_width: int,
    h_fov_deg: float = 110.0,
    principal_x_rel: float = 0.491,
) -> np.ndarray:
    """Recover Cartesian lateral position from image x and forward depth."""
    u_px = np.asarray(u_px, dtype=np.float64)
    forward_m = np.asarray(forward_m, dtype=np.float64)
    rad2cam = np.asarray(rad2cam, dtype=np.float64)

    if rad2cam.shape != (3, 4):
        raise ValueError(
            f"Expected rad2cam with shape (3, 4), got {rad2cam.shape}"
        )

    width = float(max(int(image_width), 1))
    cx = float(principal_x_rel) * (width - 1.0)
    fx = (width / 2.0) / np.tan(np.deg2rad(float(h_fov_deg)) / 2.0)

    qx = (u_px - cx) / fx
    qx, forward_m = np.broadcast_arrays(qx, forward_m)

    a00 = rad2cam[0, 0]
    a01 = rad2cam[0, 1]
    a20 = rad2cam[2, 0]
    a21 = rad2cam[2, 1]
    t0 = rad2cam[0, 3]
    t2 = rad2cam[2, 3]

    numerator = qx * (a20 * forward_m + t2) - (a00 * forward_m + t0)
    denominator = a01 - qx * a21

    lateral = np.full_like(numerator, np.nan, dtype=np.float64)
    np.divide(
        numerator,
        denominator,
        out=lateral,
        where=np.abs(denominator) > 1e-12,
    )
    return lateral


def backproject_pixels_to_bev(
    points_img: np.ndarray,
    depth_m: np.ndarray,
    rad2cam: np.ndarray,
    image_width: int,
    h_fov_deg: float = 110.0,
    principal_x_rel: float = 0.491,
) -> tuple[np.ndarray, np.ndarray]:
    """Back-project image x + forward depth into Cartesian (lateral, forward)."""
    points_img = np.asarray(points_img, dtype=np.float64)
    depth_m = np.asarray(depth_m, dtype=np.float64)

    num_points = min(len(points_img), len(depth_m))
    points_img = points_img[:num_points]
    depth_m = depth_m[:num_points]

    valid = (
        np.isfinite(points_img[:, 0])
        & np.isfinite(depth_m)
        & (depth_m > 0.0)
    )

    if not np.any(valid):
        return np.zeros((0, 2), dtype=np.float64), valid

    u = points_img[valid, 0]
    forward = depth_m[valid]

    lateral = recover_lateral_from_forward_depth(
        u_px=u,
        forward_m=forward,
        rad2cam=rad2cam,
        image_width=image_width,
        h_fov_deg=h_fov_deg,
        principal_x_rel=principal_x_rel,
    )

    stable = np.isfinite(lateral)
    valid_indices = np.flatnonzero(valid)
    final_valid = valid.copy()
    final_valid[valid_indices[~stable]] = False

    bev = np.stack([lateral[stable], forward[stable]], axis=1).astype(
        np.float64
    )
    return bev, final_valid


@HOOKS.register_module()
class PPPDemoGTVelocityCamBEVHook(Hook):
    """Write separate GT radial-velocity camera and Cartesian-BEV figures."""

    def __init__(
        self,
        train_interval: int = 0,
        max_samples_train: int = 0,
        max_samples_val: int = 0,
        max_samples_test: int = 10,
        out_dir: str = "demo_gt_velocity",
        img_mean=None,
        img_std=None,
        dpi: int = 120,
        camera_figsize: tuple[float, float] = (14.0, 4.8),
        bev_figsize: tuple[float, float] = (6.4, 6.8),
        point_size: float = 28.0,
        image_ext: str = "png",
        write_all_ranks: bool = False,
        h_fov_deg: float = 110.0,
        principal_x_rel: float = 0.491,
        bev_forward_max: float = 110.0,
        bev_lateral_max: float = 55.0,
        velocity_vmin: float = -11.0,
        velocity_vmax: float = 11.0,
        velocity_cmap: str = "coolwarm",
        point_edge_color: str = "black",
        point_edge_width: float = 0.45,
        show_titles: bool = True,
    ) -> None:
        self.train_interval = int(train_interval)
        self.max_samples = {
            "train": int(max_samples_train),
            "val": int(max_samples_val),
            "test": int(max_samples_test),
        }

        self.out_dir = str(out_dir)
        self.image_ext = str(image_ext).lstrip(".")
        self.dpi = int(dpi)
        self.camera_figsize = tuple(camera_figsize)
        self.bev_figsize = tuple(bev_figsize)
        self.point_size = float(point_size)
        self.write_all_ranks = bool(write_all_ranks)

        self.h_fov_deg = float(h_fov_deg)
        self.principal_x_rel = float(principal_x_rel)
        self.bev_forward_max = float(bev_forward_max)
        self.bev_lateral_max = float(bev_lateral_max)

        self.velocity_vmin = float(velocity_vmin)
        self.velocity_vmax = float(velocity_vmax)
        self.velocity_cmap = str(velocity_cmap)
        self.point_edge_color = str(point_edge_color)
        self.point_edge_width = float(point_edge_width)
        self.show_titles = bool(show_titles)

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

        self._written_in_run = {"val": 0, "test": 0}

    def _rank_allowed(self) -> bool:
        return self.write_all_ranks or is_main_process()

    def before_val_epoch(self, runner) -> None:
        self._written_in_run["val"] = 0

    def before_test_epoch(self, runner) -> None:
        self._written_in_run["test"] = 0

    def after_train_iter(
        self,
        runner,
        batch_idx,
        data_batch=None,
        outputs=None,
    ) -> None:
        del batch_idx, outputs
        if not self._rank_allowed() or not isinstance(data_batch, dict):
            return
        if self.max_samples["train"] <= 0 or self.train_interval <= 0:
            return
        if runner.iter % self.train_interval != 0:
            return
        self._run(runner, data_batch, "train")

    def after_val_iter(
        self,
        runner,
        batch_idx,
        data_batch=None,
        outputs=None,
    ) -> None:
        del batch_idx, outputs
        if not self._rank_allowed() or not isinstance(data_batch, dict):
            return
        if self.max_samples["val"] <= 0:
            return
        if self._written_in_run["val"] >= self.max_samples["val"]:
            return
        self._run(runner, data_batch, "val")

    def after_test_iter(
        self,
        runner,
        batch_idx,
        data_batch=None,
        outputs=None,
    ) -> None:
        del batch_idx, outputs
        if not self._rank_allowed() or not isinstance(data_batch, dict):
            return
        if self.max_samples["test"] <= 0:
            return
        if self._written_in_run["test"] >= self.max_samples["test"]:
            return
        self._run(runner, data_batch, "test")

    def _run(self, runner, data_batch: dict, phase: str) -> None:
        # Only the model's data preprocessor is used. There is deliberately no
        # extract_feat/decode_head/model forward call in this hook.
        model = (
            runner.model.module
            if is_model_wrapper(runner.model)
            else runner.model
        )
        processed = model.data_preprocessor(data_batch)
        inputs = processed["inputs"]
        data_samples = processed["data_samples"]

        if phase == "train":
            num_visualizations = min(
                self.max_samples["train"],
                len(data_samples),
                inputs.size(0),
            )
            start_index = 0
        else:
            remaining = self.max_samples[phase] - self._written_in_run[phase]
            if remaining <= 0:
                return
            num_visualizations = min(
                remaining,
                len(data_samples),
                inputs.size(0),
            )
            start_index = self._written_in_run[phase]

        for i in range(num_visualizations):
            self._visualize_sample(
                runner=runner,
                phase=phase,
                img_tensor=inputs[i],
                sample=data_samples[i],
                sample_index=start_index + i,
            )

        if phase in self._written_in_run:
            self._written_in_run[phase] += num_visualizations

    def _denorm_img(self, image: np.ndarray) -> np.ndarray:
        image = image.astype(np.float32, copy=False)

        if self.img_mean is not None and self.img_std is not None:
            image = image * self.img_std + self.img_mean
            image = (image + 1.0) * 127.5
        else:
            if image.min() >= -1.01 and image.max() <= 1.01:
                image = (image + 1.0) * 127.5
            elif image.max() <= 1.01:
                image = image * 255.0

        image = np.clip(image, 0, 255).astype(np.uint8)
        return np.transpose(image, (1, 2, 0))

    @staticmethod
    def _tagify(value: str) -> str:
        return "".join(
            character if character.isalnum() or character in "._-/" else "_"
            for character in value
        )

    @staticmethod
    def _to_int_or_none(value):
        if value is None:
            return None
        if isinstance(value, torch.Tensor):
            if value.numel() == 0:
                return None
            return int(value.flatten()[0].item())
        try:
            return int(value)
        except Exception:
            return None

    def _base_output_dir(
        self,
        runner,
        phase: str,
        video: str,
    ) -> Path:
        root = Path(self.out_dir)
        if not root.is_absolute():
            root = Path(runner.work_dir) / root

        output_directory = root / phase / video
        output_directory.mkdir(parents=True, exist_ok=True)
        return output_directory

    def _add_velocity_colorbar(self, figure: Figure, ax, label: str) -> None:
        scalar_mappable = ScalarMappable(
            norm=Normalize(
                vmin=self.velocity_vmin,
                vmax=self.velocity_vmax,
            ),
            cmap=self.velocity_cmap,
        )
        scalar_mappable.set_array([])
        colorbar = figure.colorbar(
            scalar_mappable,
            ax=ax,
            orientation="horizontal",
            fraction=0.055,
            pad=0.08,
            aspect=35,
        )
        colorbar.set_label(label)

    def _plot_camera(
        self,
        image: np.ndarray,
        points_img: np.ndarray,
        velocities: np.ndarray,
        output_path: Path,
    ) -> None:
        figure = Figure(figsize=self.camera_figsize, dpi=self.dpi)
        FigureCanvasAgg(figure)
        ax = figure.add_subplot(1, 1, 1)

        ax.imshow(image, aspect="auto")
        if self.show_titles:
            ax.set_title("Ground-truth radar radial velocity")
        ax.axis("off")

        valid = (
            np.isfinite(points_img[:, 0])
            & np.isfinite(points_img[:, 1])
            & np.isfinite(velocities)
        )
        if np.any(valid):
            ax.scatter(
                points_img[valid, 0],
                points_img[valid, 1],
                s=self.point_size,
                c=velocities[valid],
                cmap=self.velocity_cmap,
                vmin=self.velocity_vmin,
                vmax=self.velocity_vmax,
                alpha=0.98,
                edgecolors=self.point_edge_color,
                linewidths=self.point_edge_width,
            )

        self._add_velocity_colorbar(figure, ax, "radial velocity [m/s]")
        figure.savefig(output_path, dpi=self.dpi, bbox_inches="tight")
        figure.clear()

    def _plot_bev(
        self,
        bev_points: np.ndarray,
        velocities: np.ndarray,
        output_path: Path,
    ) -> None:
        figure = Figure(figsize=self.bev_figsize, dpi=self.dpi)
        FigureCanvasAgg(figure)
        ax = figure.add_subplot(1, 1, 1)

        if self.show_titles:
            ax.set_title("Ground-truth radar radial velocity — Cartesian BEV")

        # IMPORTANT: reversed x limits mirror the BEV across the vertical axis.
        # Therefore +x is displayed on the left and -x on the right.
        ax.set_xlim(self.bev_lateral_max, -self.bev_lateral_max)
        ax.set_ylim(0.0, self.bev_forward_max)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("lateral position [m]")
        ax.set_ylabel("forward distance [m]")
        ax.grid(True, alpha=0.20, linewidth=0.7)

        valid = (
            np.isfinite(bev_points[:, 0])
            & np.isfinite(bev_points[:, 1])
            & np.isfinite(velocities)
            & (bev_points[:, 1] >= 0.0)
            & (bev_points[:, 1] <= self.bev_forward_max)
            & (np.abs(bev_points[:, 0]) <= self.bev_lateral_max)
        )

        if np.any(valid):
            ax.scatter(
                bev_points[valid, 0],
                bev_points[valid, 1],
                s=self.point_size,
                c=velocities[valid],
                cmap=self.velocity_cmap,
                vmin=self.velocity_vmin,
                vmax=self.velocity_vmax,
                alpha=0.98,
                edgecolors=self.point_edge_color,
                linewidths=self.point_edge_width,
            )

        # Ego vehicle marker at the origin.
        ax.scatter(
            [0.0],
            [0.0],
            marker="^",
            s=90,
            c="black",
            zorder=5,
        )

        self._add_velocity_colorbar(figure, ax, "radial velocity [m/s]")
        figure.savefig(output_path, dpi=self.dpi, bbox_inches="tight")
        figure.clear()

    def _visualize_sample(
        self,
        runner,
        phase: str,
        img_tensor: torch.Tensor,
        sample,
        sample_index: int,
    ) -> None:
        gt_all = sample.gt_sem_seg.data
        if gt_all.ndim < 3 or gt_all.shape[0] < 3:
            raise ValueError(
                "Expected GT channels [support, depth, velocity], "
                f"but got shape {tuple(gt_all.shape)}."
            )

        gt_mask = gt_all[0].detach().cpu().numpy()
        gt_depth = gt_all[1].detach().cpu().numpy()
        gt_velocity = gt_all[2].detach().cpu().numpy()

        image = self._denorm_img(img_tensor.detach().cpu().numpy())
        image_h, image_w = image.shape[:2]
        gt_h, gt_w = gt_mask.shape[-2:]

        metadata = sample.metainfo
        if "rad2cam" not in metadata:
            raise KeyError(
                "Missing rad2cam in sample metainfo. "
                "Add it in PPPPackSegInputs."
            )

        rad2cam = np.asarray(metadata["rad2cam"], dtype=np.float64)
        fovs = metadata.get("fovs")
        h_fov_deg = float(fovs[0]) if fovs is not None else self.h_fov_deg

        gt_points_grid, gt_depth_values = extract_gt_points_and_values(
            gt_mask,
            gt_depth,
        )
        _, gt_velocity_values = extract_gt_points_and_values(
            gt_mask,
            gt_velocity,
        )

        # Project GT raster coordinates to the displayed camera resolution.
        gt_points_img = scale_points_to_shape(
            gt_points_grid,
            (gt_h, gt_w),
            (image_h, image_w),
        )

        # Recover Cartesian GT positions from image x and GT forward depth.
        gt_bev, gt_bev_valid = backproject_pixels_to_bev(
            points_img=gt_points_img,
            depth_m=gt_depth_values,
            rad2cam=rad2cam,
            image_width=image_w,
            h_fov_deg=h_fov_deg,
            principal_x_rel=self.principal_x_rel,
        )
        gt_velocity_bev = gt_velocity_values[: len(gt_bev_valid)][gt_bev_valid]

        video = self._tagify(
            str(metadata.get("video_name", "unknown_video"))
        )
        frame = self._to_int_or_none(metadata.get("video_frame_idx"))
        global_index = self._to_int_or_none(
            metadata.get("global_idx", metadata.get("frame_idx"))
        )

        frame_string = (
            f"{frame:06d}" if frame is not None else "unknown_frame"
        )
        global_index_string = (
            f"{global_index:08d}"
            if global_index is not None
            else f"sample_{sample_index:08d}"
        )

        output_directory = self._base_output_dir(
            runner=runner,
            phase=phase,
            video=video,
        )

        camera_path = output_directory / (
            f"frame_{frame_string}_idx_{global_index_string}_camera_velocity."
            f"{self.image_ext}"
        )
        bev_path = output_directory / (
            f"frame_{frame_string}_idx_{global_index_string}_bev_velocity."
            f"{self.image_ext}"
        )

        self._plot_camera(
            image=image,
            points_img=gt_points_img,
            velocities=gt_velocity_values,
            output_path=camera_path,
        )
        self._plot_bev(
            bev_points=gt_bev,
            velocities=gt_velocity_bev,
            output_path=bev_path,
        )