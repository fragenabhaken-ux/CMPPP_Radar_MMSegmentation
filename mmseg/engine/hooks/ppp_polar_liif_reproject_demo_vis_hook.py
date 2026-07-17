"""
Polar LIIF PPP demo visualization hook with Cartesian sampling and camera projection.

For each frame, writes a tight 2x2 layout:

    top-left:      GT radar points projected into the camera image
    top-right:     GT radar point cloud in BEV
    bottom-left:   sampled predicted radar points projected into the camera image
    bottom-right:  sampled predicted radar point cloud in BEV

The model predicts lambda(r, theta). The hook:

    1. Converts lambda(r, theta) to polar cell masses:
           lambda * dr * dtheta

    2. Bilinearly splats those masses onto a Cartesian BEV grid.

    3. Samples a Cartesian point cloud using either:
           - thinning
           - block approximation

    4. Projects GT and sampled Cartesian radar points into the camera image
       with the exact existing SYS100 radar-to-camera convention.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch

from mmengine.dist import is_main_process
from mmengine.hooks import Hook
from mmengine.model import is_model_wrapper
from mmseg.registry import HOOKS

from aumovio.fisheye import FisheyeCamera, PinholeCamera

import matplotlib

matplotlib.use("Agg")

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure


def _to_numpy(x: Any) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def sample_ppp_thinning(cell_mass: np.ndarray, rng=None):
    """Sample a point cloud from a Cartesian grid of PPP cell masses."""
    rng = np.random.default_rng() if rng is None else rng

    cell_mass = np.asarray(cell_mass, dtype=np.float64)

    if cell_mass.ndim != 2:
        raise ValueError("cell_mass must be 2D")
    if not np.all(np.isfinite(cell_mass)):
        raise ValueError("cell_mass contains NaN/Inf")
    if np.any(cell_mass < 0):
        raise ValueError("cell_mass contains negative values")

    h, w = cell_mass.shape
    max_mass = float(cell_mass.max())

    if max_mass <= 0:
        return np.zeros((0, 2), dtype=np.float64), {
            "num_points": 0,
            "num_proposals": 0,
        }

    n_proposals = int(rng.poisson(max_mass * h * w))

    if n_proposals == 0:
        return np.zeros((0, 2), dtype=np.float64), {
            "num_points": 0,
            "num_proposals": 0,
        }

    cols = rng.random(n_proposals) * w
    rows = rng.random(n_proposals) * h

    col_idx = np.clip(cols.astype(np.int64), 0, w - 1)
    row_idx = np.clip(rows.astype(np.int64), 0, h - 1)

    keep = rng.random(n_proposals) < (
        cell_mass[row_idx, col_idx] / max_mass
    )

    points = np.stack(
        [cols[keep], rows[keep]],
        axis=1,
    ).astype(np.float64)

    return points, {
        "num_points": int(points.shape[0]),
        "num_proposals": n_proposals,
    }


def sample_ppp_block(
    cell_mass: np.ndarray,
    block_size: int = 8,
    placement: str = "weighted",
    rng=None,
):
    """Sample at most one point per Cartesian BEV block."""
    rng = np.random.default_rng() if rng is None else rng

    cell_mass = np.asarray(cell_mass, dtype=np.float64)

    if cell_mass.ndim != 2:
        raise ValueError("cell_mass must be 2D")
    if not np.all(np.isfinite(cell_mass)):
        raise ValueError("cell_mass contains NaN/Inf")
    if np.any(cell_mass < 0):
        raise ValueError("cell_mass contains negative values")
    if block_size <= 0:
        raise ValueError("block_size must be > 0")
    if placement not in {"weighted", "uniform"}:
        raise ValueError("placement must be 'weighted' or 'uniform'")

    h, w = cell_mass.shape
    block_size = int(block_size)

    n_block_rows = (h + block_size - 1) // block_size
    n_block_cols = (w + block_size - 1) // block_size

    padded_h = n_block_rows * block_size
    padded_w = n_block_cols * block_size

    padded_mass = np.zeros((padded_h, padded_w), dtype=np.float64)
    padded_mass[:h, :w] = cell_mass

    blocks = padded_mass.reshape(
        n_block_rows,
        block_size,
        n_block_cols,
        block_size,
    ).transpose(0, 2, 1, 3)

    block_mass = blocks.sum(axis=(2, 3))
    activation_prob = 1.0 - np.exp(-block_mass)

    active = rng.random((n_block_rows, n_block_cols)) < activation_prob

    block_rows, block_cols = np.nonzero(active)
    n_points = len(block_rows)

    if n_points == 0:
        return np.zeros((0, 2), dtype=np.float64), {
            "num_points": 0,
            "num_active_blocks": 0,
        }

    points = np.empty((n_points, 2), dtype=np.float64)

    row_starts = block_rows * block_size
    col_starts = block_cols * block_size

    block_heights = np.minimum(block_size, h - row_starts)
    block_widths = np.minimum(block_size, w - col_starts)

    if placement == "uniform":
        points[:, 0] = (
            col_starts + rng.random(n_points) * block_widths
        )
        points[:, 1] = (
            row_starts + rng.random(n_points) * block_heights
        )

    else:
        for i, (row0, col0, block_h, block_w) in enumerate(
            zip(
                row_starts,
                col_starts,
                block_heights,
                block_widths,
            )
        ):
            block = cell_mass[
                row0:row0 + block_h,
                col0:col0 + block_w,
            ]

            total_mass = float(block.sum())

            if total_mass <= 0:
                points[i, 0] = col0 + rng.random() * block_w
                points[i, 1] = row0 + rng.random() * block_h
                continue

            flat_index = int(
                rng.choice(
                    block.size,
                    p=block.ravel() / total_mass,
                )
            )

            local_row, local_col = divmod(
                flat_index,
                int(block_w),
            )

            points[i, 0] = col0 + local_col + rng.random()
            points[i, 1] = row0 + local_row + rng.random()

    return points, {
        "num_points": int(n_points),
        "num_active_blocks": int(n_points),
    }


@HOOKS.register_module()
class PPPPolarLIIFReprojectDemoVisHook(Hook):
    """Export GT and sampled polar-PPP point clouds in camera and BEV views."""

    def __init__(
        self,
        train_interval: int = 0,
        max_samples_train: int = 0,
        max_samples_val: int = 0,
        max_samples_test: int = 10,
        out_dir: str = "demo_frames_polar_reprojected",
        img_mean=None,
        img_std=None,
        r_min: float = 0.0,
        r_max: float = 104.0,
        theta_min: float = -1.5707963267948966,
        theta_max: float = 1.5707963267948966,
        bev_x_min: float = 0.0,
        bev_x_max: float = 104.0,
        bev_y_min: float = -85.0,
        bev_y_max: float = 85.0,
        bev_resolution: float = 0.5,
        sample_method: str = "thinning",
        block_size: int = 8,
        block_placement: str = "weighted",
        point_size: float = 12.0,
        gt_point_color: str = "black",
        pred_point_color: str = "black",
        point_edge_color: str = "white",
        point_edge_width: float = 0.25,
        rel_x_centre: float = 0.491,
        rel_y_centre: float = 0.4370,
        show_polar_grid: bool = True,
        dpi: int = 100,
        figsize=(20, 8.5),
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

        self.bev_x_min = float(bev_x_min)
        self.bev_x_max = float(bev_x_max)
        self.bev_y_min = float(bev_y_min)
        self.bev_y_max = float(bev_y_max)
        self.bev_resolution = float(bev_resolution)

        self.sample_method = str(sample_method)
        self.block_size = int(block_size)
        self.block_placement = str(block_placement)

        self.point_size = float(point_size)
        self.gt_point_color = str(gt_point_color)
        self.pred_point_color = str(pred_point_color)
        self.point_edge_color = str(point_edge_color)
        self.point_edge_width = float(point_edge_width)

        self.rel_x_centre = float(rel_x_centre)
        self.rel_y_centre = float(rel_y_centre)

        self.show_polar_grid = bool(show_polar_grid)

        self.dpi = int(dpi)
        self.figsize = tuple(figsize)

        self.img_mean = (
            None
            if img_mean is None
            else np.asarray(
                img_mean,
                dtype=np.float32,
            )[:, None, None]
        )

        self.img_std = (
            None
            if img_std is None
            else np.asarray(
                img_std,
                dtype=np.float32,
            )[:, None, None]
        )

        if self.sample_method not in {"thinning", "block"}:
            raise ValueError(
                "sample_method must be 'thinning' or 'block'."
            )

        if self.bev_resolution <= 0:
            raise ValueError("bev_resolution must be > 0.")

        self._cam_cache: Dict[
            Tuple[Any, ...],
            PinholeCamera,
        ] = {}

        self._written_in_run = {
            "val": 0,
            "test": 0,
        }

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
        if not self._rank_allowed():
            return

        if not isinstance(data_batch, dict):
            return

        if self.max_samples["train"] <= 0:
            return

        if self.train_interval <= 0:
            return

        if runner.iter % self.train_interval != 0:
            return

        self._run(
            runner=runner,
            data_batch=data_batch,
            phase="train",
        )

    def after_val_iter(
        self,
        runner,
        batch_idx,
        data_batch=None,
        outputs=None,
    ) -> None:
        if not self._rank_allowed():
            return

        if not isinstance(data_batch, dict):
            return

        if self.max_samples["val"] <= 0:
            return

        if self._written_in_run["val"] >= self.max_samples["val"]:
            return

        self._run(
            runner=runner,
            data_batch=data_batch,
            phase="val",
        )

    def after_test_iter(
        self,
        runner,
        batch_idx,
        data_batch=None,
        outputs=None,
    ) -> None:
        if not self._rank_allowed():
            return

        if not isinstance(data_batch, dict):
            return

        if self.max_samples["test"] <= 0:
            return

        if self._written_in_run["test"] >= self.max_samples["test"]:
            return

        self._run(
            runner=runner,
            data_batch=data_batch,
            phase="test",
        )

    def _run(
        self,
        runner,
        data_batch,
        phase: str,
    ) -> None:
        model = (
            runner.model.module
            if is_model_wrapper(runner.model)
            else runner.model
        )

        was_training = model.training
        model.eval()

        proc = model.data_preprocessor(data_batch)

        inputs = proc["inputs"]
        data_samples = proc["data_samples"]

        with torch.no_grad():
            preds = model.decode_head(
                model.extract_feat(inputs)
            )

        if not isinstance(preds, dict):
            raise TypeError(
                "Expected decode_head output to be a dictionary."
            )

        if "grid_log_lambda" not in preds:
            raise KeyError(
                "Expected preds['grid_log_lambda']."
            )

        grid_log_lambda = preds["grid_log_lambda"]

        if phase == "train":
            n_vis = min(
                self.max_samples["train"],
                grid_log_lambda.shape[0],
            )
            start_idx = 0

        else:
            remaining = (
                self.max_samples[phase]
                - self._written_in_run[phase]
            )

            if remaining <= 0:
                if was_training:
                    model.train()
                return

            n_vis = min(
                remaining,
                grid_log_lambda.shape[0],
            )

            start_idx = self._written_in_run[phase]

        for batch_index in range(n_vis):
            self._visualize_sample(
                runner=runner,
                phase=phase,
                img_tensor=inputs[batch_index],
                grid_log_lambda=grid_log_lambda[
                    batch_index,
                    0,
                ],
                sample=data_samples[batch_index],
                iter_seed=runner.iter,
                sample_idx=start_idx + batch_index,
            )

        if phase in self._written_in_run:
            self._written_in_run[phase] += n_vis

        if was_training:
            model.train()

    def _denorm_img(self, x: np.ndarray) -> np.ndarray:
        x = x.astype(np.float32)

        if self.img_mean is not None and self.img_std is not None:
            x = x * self.img_std + self.img_mean
            x = (x + 1.0) * 127.5

        elif x.min() >= -1.01 and x.max() <= 1.01:
            x = (x + 1.0) * 127.5

        elif x.max() <= 1.01:
            x = x * 255.0

        x = np.clip(x, 0, 255).astype(np.uint8)

        return np.transpose(x, (1, 2, 0))

    @staticmethod
    def _tagify(s: str) -> str:
        return "".join(
            c if c.isalnum() or c in "._-/" else "_"
            for c in s
        )

    def _cartesian_shape(self) -> tuple[int, int]:
        n_x = int(
            round(
                (self.bev_x_max - self.bev_x_min)
                / self.bev_resolution
            )
        )

        n_y = int(
            round(
                (self.bev_y_max - self.bev_y_min)
                / self.bev_resolution
            )
        )

        if n_x <= 0 or n_y <= 0:
            raise ValueError(
                "Invalid Cartesian BEV range or resolution."
            )

        return n_x, n_y

    def _polar_mass_to_cartesian_mass(
        self,
        polar_mass: np.ndarray,
    ) -> np.ndarray:
        """Bilinearly splat polar cell masses onto a Cartesian BEV grid."""
        h_r, w_theta = polar_mass.shape

        n_x, n_y = self._cartesian_shape()

        dr = (self.r_max - self.r_min) / h_r
        dtheta = (
            (self.theta_max - self.theta_min)
            / w_theta
        )

        r = (
            self.r_min
            + (np.arange(h_r, dtype=np.float64) + 0.5) * dr
        )

        theta = (
            self.theta_min
            + (np.arange(w_theta, dtype=np.float64) + 0.5)
            * dtheta
        )

        rr, tt = np.meshgrid(
            r,
            theta,
            indexing="ij",
        )

        x_forward = rr * np.cos(tt)
        y_lateral = rr * np.sin(tt)

        row_float = (
            (x_forward - self.bev_x_min)
            / self.bev_resolution
            - 0.5
        )

        col_float = (
            (y_lateral - self.bev_y_min)
            / self.bev_resolution
            - 0.5
        )

        row_float = np.clip(
            row_float,
            0.0,
            float(n_x - 1),
        )

        col_float = np.clip(
            col_float,
            0.0,
            float(n_y - 1),
        )

        row0 = np.floor(row_float).astype(np.int64)
        col0 = np.floor(col_float).astype(np.int64)

        row1 = np.minimum(row0 + 1, n_x - 1)
        col1 = np.minimum(col0 + 1, n_y - 1)

        wr1 = row_float - row0
        wc1 = col_float - col0

        wr0 = 1.0 - wr1
        wc0 = 1.0 - wc1

        cartesian_mass = np.zeros(
            (n_x, n_y),
            dtype=np.float64,
        )

        np.add.at(
            cartesian_mass,
            (row0, col0),
            polar_mass * wr0 * wc0,
        )

        np.add.at(
            cartesian_mass,
            (row0, col1),
            polar_mass * wr0 * wc1,
        )

        np.add.at(
            cartesian_mass,
            (row1, col0),
            polar_mass * wr1 * wc0,
        )

        np.add.at(
            cartesian_mass,
            (row1, col1),
            polar_mass * wr1 * wc1,
        )

        return cartesian_mass

    def _sample_cartesian_points(
        self,
        cartesian_mass: np.ndarray,
        rng,
    ):
        if self.sample_method == "thinning":
            return sample_ppp_thinning(
                cartesian_mass,
                rng=rng,
            )

        return sample_ppp_block(
            cartesian_mass,
            block_size=self.block_size,
            placement=self.block_placement,
            rng=rng,
        )

    def _grid_points_to_xy(
        self,
        grid_points: np.ndarray,
    ) -> np.ndarray:
        """Convert sampled Cartesian grid coordinates to metric radar XY."""
        if len(grid_points) == 0:
            return np.zeros((0, 2), dtype=np.float64)

        x_forward = (
            self.bev_x_min
            + grid_points[:, 1] * self.bev_resolution
        )

        y_lateral = (
            self.bev_y_min
            + grid_points[:, 0] * self.bev_resolution
        )

        return np.stack(
            [x_forward, y_lateral],
            axis=1,
        )

    def _get_pinhole_cam(
        self,
        image_h: int,
        image_w: int,
        real_hw: np.ndarray,
        fovs,
    ) -> PinholeCamera:
        real_h = int(real_hw[0])
        real_w = int(real_hw[1])

        h_fov = float(fovs[0])
        v_fov = float(fovs[1])

        key = (
            image_h,
            image_w,
            real_h,
            real_w,
            h_fov,
            v_fov,
            self.rel_x_centre,
            self.rel_y_centre,
        )

        camera = self._cam_cache.get(key)

        if camera is None:
            camera = PinholeCamera(
                height=image_h,
                width=image_w,
                hFov=h_fov,
                vFov=v_fov,
                rel_x_centre=self.rel_x_centre,
                rel_y_centre=self.rel_y_centre,
                fisheye_cam=FisheyeCamera(
                    height=real_h,
                    width=real_w,
                ),
            )

            self._cam_cache[key] = camera

        return camera

    def _project_xy_to_image(
        self,
        xy: np.ndarray,
        sample,
        image_h: int,
        image_w: int,
    ) -> np.ndarray:
        """Project radar-frame XY points into the undistorted camera image."""
        if len(xy) == 0:
            return np.zeros((0, 2), dtype=np.float64)

        meta = sample.metainfo

        required = (
            "rad2cam",
            "real_hw",
            "fovs",
        )

        missing = [
            key
            for key in required
            if key not in meta
        ]

        if missing:
            raise KeyError(
                "Missing projection calibration in SegDataSample.metainfo: "
                f"{missing}. Apply the required patch to "
                "sys100_polar_transforms.py."
            )

        rad2cam = _to_numpy(
            meta["rad2cam"]
        ).astype(np.float64, copy=False)

        real_hw = _to_numpy(
            meta["real_hw"]
        ).astype(np.float64, copy=False)

        fovs = meta["fovs"]

        if rad2cam.shape != (3, 4):
            raise ValueError(
                "Expected rad2cam shape (3, 4), got "
                f"{rad2cam.shape}."
            )

        if real_hw.size < 2:
            raise ValueError(
                "Expected real_hw with two values."
            )

        # Exact SYS100 convention from BuildPPPHeatmap:
        # radar XY -> camera XYZ.
        camera_points = (
            xy @ rad2cam[:, :2].T
            + rad2cam[:, 3][None, :]
        )

        z = camera_points[:, 2]

        in_front = z > 1e-8

        if not np.any(in_front):
            return np.zeros((0, 2), dtype=np.float64)

        camera_points = camera_points[in_front]

        z = camera_points[:, 2]

        normalized_x = camera_points[:, 0] / z
        normalized_y = camera_points[:, 1] / z

        pinhole_cam = self._get_pinhole_cam(
            image_h=image_h,
            image_w=image_w,
            real_hw=real_hw,
            fovs=fovs,
        )

        x_pix, y_pix = pinhole_cam.real_world_to_pinhole(
            normalized_x,
            normalized_y,
        )

        x_pix = np.asarray(x_pix, dtype=np.float64)
        y_pix = np.asarray(y_pix, dtype=np.float64)

        if bool(meta.get("flip", False)):
            x_pix = (image_w - 1) - x_pix

        valid = (
            np.isfinite(x_pix)
            & np.isfinite(y_pix)
            & (x_pix >= 0.0)
            & (x_pix < image_w)
            & (y_pix >= 0.0)
            & (y_pix < image_h)
        )

        return np.stack(
            [x_pix[valid], y_pix[valid]],
            axis=1,
        )

    def _draw_polar_grid(self, ax) -> None:
        if not self.show_polar_grid:
            return

        theta = np.linspace(
            self.theta_min,
            self.theta_max,
            161,
        )

        for radius in (20.0, 40.0, 60.0, 80.0, 100.0):
            x_forward = radius * np.cos(theta)
            y_lateral = radius * np.sin(theta)

            ax.plot(
                y_lateral,
                x_forward,
                color="0.90",
                linewidth=0.7,
                zorder=0,
            )

        for theta_value in np.deg2rad(
            np.arange(-45, 46, 15)
        ):
            radius = np.linspace(
                self.r_min,
                self.r_max,
                100,
            )

            x_forward = radius * np.cos(theta_value)
            y_lateral = radius * np.sin(theta_value)

            ax.plot(
                y_lateral,
                x_forward,
                color="0.93",
                linewidth=0.6,
                zorder=0,
            )

    def _configure_bev_axes(self, ax) -> None:
        ax.set_xlim(
            self.bev_y_max,
            self.bev_y_min,
        )

        ax.set_ylim(
            self.bev_x_min,
            self.bev_x_max,
        )

        ax.set_aspect("equal", adjustable="box")

        ax.set_xlabel(
            "y lateral [m]",
            labelpad=2,
        )

        ax.set_ylabel(
            "x forward [m]",
            labelpad=2,
        )

        ax.tick_params(
            labelsize=9,
            pad=2,
        )

        self._draw_polar_grid(ax)

    @staticmethod
    def _textbox(ax, text: str) -> None:
        ax.text(
            0.015,
            0.975,
            text,
            transform=ax.transAxes,
            fontsize=9,
            color="white",
            va="top",
            ha="left",
            bbox=dict(
                facecolor="black",
                alpha=0.58,
                pad=2.5,
            ),
        )

    def _output_path(
        self,
        runner,
        phase: str,
        video: str,
        frame_str: str,
        gidx_str: str,
    ) -> Path:
        root = Path(self.out_dir)

        if not root.is_absolute():
            root = Path(runner.work_dir) / root

        out_dir = root / phase / video
        out_dir.mkdir(parents=True, exist_ok=True)

        return out_dir / (
            f"frame_{frame_str}_idx_{gidx_str}.{self.image_ext}"
        )

    def _visualize_sample(
        self,
        runner,
        phase: str,
        img_tensor: torch.Tensor,
        grid_log_lambda: torch.Tensor,
        sample,
        iter_seed: int,
        sample_idx: int,
    ) -> None:
        del sample_idx

        image = self._denorm_img(
            img_tensor.detach().cpu().numpy()
        )

        image_h, image_w = image.shape[:2]

        polar_lambda = torch.exp(
            grid_log_lambda.detach().float()
        ).cpu().numpy()

        h_r, w_theta = polar_lambda.shape

        dr = (self.r_max - self.r_min) / h_r
        dtheta = (
            (self.theta_max - self.theta_min)
            / w_theta
        )

        polar_mass = polar_lambda * dr * dtheta

        cartesian_mass = self._polar_mass_to_cartesian_mass(
            polar_mass
        )

        expected_count_polar = float(polar_mass.sum())
        expected_count_cartesian = float(cartesian_mass.sum())

        gt_xy = sample.gt_radar_points.xy.detach().cpu().numpy()

        gt_xy = gt_xy.astype(
            np.float64,
            copy=False,
        )

        meta = sample.metainfo

        video = self._tagify(
            str(meta.get("video_name", "unknown_video"))
        )

        frame = meta.get("video_frame_idx")
        gidx = meta.get(
            "global_idx",
            meta.get("frame_idx", 0),
        )

        frame_str = (
            f"{int(frame):06d}"
            if frame is not None
            else "unknown_frame"
        )

        gidx_str = (
            f"{int(gidx):08d}"
            if gidx is not None
            else "unknown_idx"
        )

        seed_base = (
            int(frame)
            if frame is not None
            else int(gidx)
        )

        rng = np.random.default_rng(
            seed_base + 1_000_003 * int(iter_seed)
        )

        predicted_grid_points, _ = self._sample_cartesian_points(
            cartesian_mass,
            rng=rng,
        )

        predicted_xy = self._grid_points_to_xy(
            predicted_grid_points
        )

        gt_image_points = self._project_xy_to_image(
            gt_xy,
            sample,
            image_h,
            image_w,
        )

        predicted_image_points = self._project_xy_to_image(
            predicted_xy,
            sample,
            image_h,
            image_w,
        )

        image_aspect = image_w / image_h

        bev_aspect = (
            (self.bev_y_max - self.bev_y_min)
            / (self.bev_x_max - self.bev_x_min)
        )

        fig = Figure(
            figsize=self.figsize,
            dpi=self.dpi,
        )

        FigureCanvasAgg(fig)

        axes = fig.subplots(
            2,
            2,
            gridspec_kw=dict(
                width_ratios=(
                    image_aspect,
                    bev_aspect,
                ),
            ),
        )

        fig.subplots_adjust(
            left=0.025,
            right=0.995,
            top=0.965,
            bottom=0.055,
            wspace=0.035,
            hspace=0.115,
        )

        # -------------------------------------------------------------
        # GT camera overlay
        # -------------------------------------------------------------
        ax = axes[0, 0]

        ax.imshow(
            image,
            aspect="equal",
        )

        if len(gt_image_points):
            ax.scatter(
                gt_image_points[:, 0],
                gt_image_points[:, 1],
                s=self.point_size,
                c=self.gt_point_color,
                edgecolors=self.point_edge_color,
                linewidths=self.point_edge_width,
                alpha=0.98,
            )

        ax.set_title(
            "GT radar points projected into camera image",
            fontsize=11,
            pad=3,
        )

        ax.axis("off")

        self._textbox(
            ax,
            f"GT N={len(gt_xy)}\nvisible={len(gt_image_points)}",
        )

        # -------------------------------------------------------------
        # GT BEV
        # -------------------------------------------------------------
        ax = axes[0, 1]

        self._configure_bev_axes(ax)

        if len(gt_xy):
            ax.scatter(
                gt_xy[:, 1],
                gt_xy[:, 0],
                s=self.point_size,
                c=self.gt_point_color,
                edgecolors=self.point_edge_color,
                linewidths=self.point_edge_width,
                alpha=0.98,
                zorder=3,
            )

        ax.set_title(
            "GT radar point cloud in BEV",
            fontsize=11,
            pad=3,
        )

        self._textbox(
            ax,
            f"GT N={len(gt_xy)}",
        )

        # -------------------------------------------------------------
        # Predicted camera overlay
        # -------------------------------------------------------------
        ax = axes[1, 0]

        ax.imshow(
            image,
            aspect="equal",
        )

        if len(predicted_image_points):
            ax.scatter(
                predicted_image_points[:, 0],
                predicted_image_points[:, 1],
                s=self.point_size,
                c=self.pred_point_color,
                edgecolors=self.point_edge_color,
                linewidths=self.point_edge_width,
                alpha=0.98,
            )

        ax.set_title(
            "Sampled predicted radar points projected into camera image",
            fontsize=11,
            pad=3,
        )

        ax.axis("off")

        self._textbox(
            ax,
            (
                f"method={self.sample_method}\n"
                f"N={len(predicted_xy)}\n"
                f"visible={len(predicted_image_points)}"
            ),
        )

        # -------------------------------------------------------------
        # Predicted BEV
        # -------------------------------------------------------------
        ax = axes[1, 1]

        self._configure_bev_axes(ax)

        if len(predicted_xy):
            ax.scatter(
                predicted_xy[:, 1],
                predicted_xy[:, 0],
                s=self.point_size,
                c=self.pred_point_color,
                edgecolors=self.point_edge_color,
                linewidths=self.point_edge_width,
                alpha=0.98,
                zorder=3,
            )

        ax.set_title(
            "Sampled predicted radar point cloud in BEV",
            fontsize=11,
            pad=3,
        )

        self._textbox(
            ax,
            (
                f"method={self.sample_method}\n"
                f"N={len(predicted_xy)}\n"
                f"E[N]={expected_count_cartesian:.1f}\n"
                f"polar E[N]={expected_count_polar:.1f}"
            ),
        )

        out_path = self._output_path(
            runner=runner,
            phase=phase,
            video=video,
            frame_str=frame_str,
            gidx_str=gidx_str,
        )

        # No bbox_inches='tight': all video frames retain identical dimensions.
        fig.savefig(
            out_path,
            dpi=self.dpi,
        )

        fig.clear()