"""
PPP demo visualization hook with three camera-domain panels and three BEV panels.

Layout
------
Top row, camera domain:
  1. Predicted PPP intensity-mass heatmap
  2. Camera image with sampled predicted PPP points colored by predicted velocity
  3. Camera image with sampled predicted PPP points colored by predicted RCS

Bottom row, BEV:
  1. Predicted PPP intensity-mass heatmap with GT radar locations overlaid
  2. Predicted velocity field with GT velocity values overlaid at GT locations
  3. Predicted RCS field with GT RCS values overlaid at GT locations

The BEV conversion uses the frame-specific ``rad2cam`` matrix and the
forward-depth prediction to invert the image-space projection.

Current four-channel convention
-------------------------------
Prediction:
    pred[0] = log intensity
    pred[1] = raw forward depth, decoded with exp
    pred[2] = raw radial velocity
    pred[3] = raw RCS

Ground truth:
    gt[0] = support mask
    gt[1] = forward depth
    gt[2] = radial velocity
    gt[3] = RCS
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
    values: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return occupied GT cell centers and values at those cells."""
    ys, xs = np.where(mask > 0)

    if xs.size == 0:
        return (
            np.zeros((0, 2), dtype=np.float64),
            np.zeros((0,), dtype=np.float64),
        )

    points = np.stack([xs + 0.5, ys + 0.5], axis=1).astype(np.float64)

    if values is None:
        point_values = np.zeros((len(points),), dtype=np.float64)
    else:
        point_values = values[ys, xs].astype(np.float64)

    return points, point_values


def sample_grid_values_at_points(
    grid: np.ndarray | None,
    points: np.ndarray,
) -> np.ndarray:
    """Sample a dense grid at floating-point ``(x, y)`` coordinates."""
    if grid is None or len(points) == 0:
        return np.zeros((0,), dtype=np.float64)

    height, width = grid.shape[-2:]
    xs = np.clip(points[:, 0].astype(np.int64), 0, width - 1)
    ys = np.clip(points[:, 1].astype(np.int64), 0, height - 1)
    return grid[ys, xs].astype(np.float64)


def scale_points_to_shape(
    points: np.ndarray,
    from_hw: tuple[int, int],
    to_hw: tuple[int, int],
) -> np.ndarray:
    """Scale points between two ``(height, width)`` grids."""
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


def sample_ppp_thinning(
    lam: np.ndarray,
    rng: np.random.Generator | None = None,
):
    """Sample a point cloud from a discrete PPP intensity by thinning."""
    rng = np.random.default_rng() if rng is None else rng
    lam = np.asarray(lam, dtype=np.float64)

    if lam.ndim != 2:
        raise ValueError("lambda grid must be 2D")
    if not np.all(np.isfinite(lam)):
        raise ValueError("lambda grid contains NaN/Inf")
    if np.any(lam < 0):
        raise ValueError("lambda grid contains negative values")

    height, width = lam.shape
    lam_max = float(lam.max())

    if lam_max <= 0:
        return np.zeros((0, 2), dtype=np.float64), {
            "num_points": 0,
            "num_proposals": 0,
        }

    num_proposals = int(rng.poisson(lam_max * height * width))
    if num_proposals == 0:
        return np.zeros((0, 2), dtype=np.float64), {
            "num_points": 0,
            "num_proposals": 0,
        }

    xs = rng.random(num_proposals) * width
    ys = rng.random(num_proposals) * height
    xi = np.clip(xs.astype(np.int64), 0, width - 1)
    yi = np.clip(ys.astype(np.int64), 0, height - 1)

    keep = rng.random(num_proposals) < (lam[yi, xi] / lam_max)
    points = np.stack([xs[keep], ys[keep]], axis=1).astype(np.float64)

    return points, {
        "num_points": int(len(points)),
        "num_proposals": int(num_proposals),
    }


def sample_ppp_block(
    lam: np.ndarray,
    block_size: int = 8,
    placement: str = "weighted",
    rng: np.random.Generator | None = None,
):
    """Sample a point cloud using block activation probabilities."""
    rng = np.random.default_rng() if rng is None else rng
    lam = np.asarray(lam, dtype=np.float64)

    if lam.ndim != 2:
        raise ValueError("lambda grid must be 2D")
    if not np.all(np.isfinite(lam)):
        raise ValueError("lambda grid contains NaN/Inf")
    if np.any(lam < 0):
        raise ValueError("lambda grid contains negative values")
    if block_size <= 0:
        raise ValueError("block_size must be > 0")
    if placement not in {"weighted", "uniform"}:
        raise ValueError("placement must be 'weighted' or 'uniform'")

    height, width = lam.shape
    block_size = int(block_size)

    num_blocks_y = (height + block_size - 1) // block_size
    num_blocks_x = (width + block_size - 1) // block_size
    padded_height = num_blocks_y * block_size
    padded_width = num_blocks_x * block_size

    lam_padded = np.zeros((padded_height, padded_width), dtype=np.float64)
    lam_padded[:height, :width] = lam

    lam_blocks = lam_padded.reshape(
        num_blocks_y,
        block_size,
        num_blocks_x,
        block_size,
    ).transpose(0, 2, 1, 3)

    block_mass = lam_blocks.sum(axis=(2, 3))
    activation_probability = 1.0 - np.exp(-block_mass)
    active = rng.random((num_blocks_y, num_blocks_x)) < activation_probability

    block_y, block_x = np.nonzero(active)
    num_points = len(block_y)

    if num_points == 0:
        return np.zeros((0, 2), dtype=np.float64), {
            "num_points": 0,
            "num_active_blocks": 0,
        }

    points = np.empty((num_points, 2), dtype=np.float64)
    x0s = block_x * block_size
    y0s = block_y * block_size
    widths = np.minimum(block_size, width - x0s)
    heights = np.minimum(block_size, height - y0s)

    if placement == "uniform":
        points[:, 0] = x0s + rng.random(num_points) * widths
        points[:, 1] = y0s + rng.random(num_points) * heights
    else:
        for index, (x0, y0, current_width, current_height) in enumerate(
            zip(x0s, y0s, widths, heights)
        ):
            block = lam[
                y0:y0 + current_height,
                x0:x0 + current_width,
            ]
            block_sum = float(block.sum())

            if block_sum <= 0:
                points[index, 0] = x0 + rng.random() * current_width
                points[index, 1] = y0 + rng.random() * current_height
                continue

            flat_index = int(
                rng.choice(block.size, p=block.ravel() / block_sum)
            )
            pixel_y, pixel_x = divmod(flat_index, int(current_width))
            points[index, 0] = x0 + pixel_x + rng.random()
            points[index, 1] = y0 + pixel_y + rng.random()

    return points, {
        "num_points": int(num_points),
        "num_active_blocks": int(num_points),
    }


def recover_lateral_from_forward_depth(
    u_px: np.ndarray,
    forward_m: np.ndarray,
    rad2cam: np.ndarray,
    image_width: int,
    h_fov_deg: float = 110.0,
    principal_x_rel: float = 0.491,
) -> np.ndarray:
    """Recover the lateral coordinate from image x and forward depth."""
    u_px = np.asarray(u_px, dtype=np.float64)
    forward_m = np.asarray(forward_m, dtype=np.float64)
    rad2cam = np.asarray(rad2cam, dtype=np.float64)

    if rad2cam.shape != (3, 4):
        raise ValueError(
            f"Expected rad2cam with shape (3, 4), got {rad2cam.shape}"
        )

    width = float(max(int(image_width), 1))
    cx = float(principal_x_rel) * (width - 1.0)
    fx = (width / 2.0) / np.tan(
        np.deg2rad(float(h_fov_deg)) / 2.0
    )

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
    lateral_sign: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Back-project image x and forward depth into Cartesian BEV."""
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

    lateral = float(lateral_sign) * lateral[stable]
    forward = forward[stable]

    bev = np.stack([lateral, forward], axis=1).astype(np.float64)
    return bev, final_valid


def project_dense_fields_to_bev(
    intensity_mass: np.ndarray,
    depth_map: np.ndarray,
    image_hw: tuple[int, int],
    rad2cam: np.ndarray,
    h_fov_deg: float,
    principal_x_rel: float,
    lateral_sign: float,
    bev_forward_max: float,
    bev_lateral_max: float,
    bev_height: int,
    bev_width: int,
    mark_maps: dict[str, np.ndarray | None],
) -> tuple[np.ndarray, dict[str, np.ndarray], np.ndarray]:
    """
    Project intensity mass and mark fields from image space into BEV.

    Intensity is accumulated by summation. Marks are aggregated by an
    intensity-mass-weighted mean, so low-intensity image cells do not dominate
    the BEV mark visualization.
    """
    intensity_mass = np.asarray(intensity_mass, dtype=np.float64)
    depth_map = np.asarray(depth_map, dtype=np.float64)

    grid_h, grid_w = intensity_mass.shape[-2:]
    image_h, image_w = int(image_hw[0]), int(image_hw[1])

    yy, xx = np.meshgrid(
        np.arange(grid_h, dtype=np.float64) + 0.5,
        np.arange(grid_w, dtype=np.float64) + 0.5,
        indexing="ij",
    )
    grid_points = np.stack([xx.ravel(), yy.ravel()], axis=1)
    image_points = scale_points_to_shape(
        grid_points,
        (grid_h, grid_w),
        (image_h, image_w),
    )

    flat_depth = depth_map.reshape(-1)
    flat_mass = intensity_mass.reshape(-1)

    bev_points, valid_mask = backproject_pixels_to_bev(
        points_img=image_points,
        depth_m=flat_depth,
        rad2cam=rad2cam,
        image_width=image_w,
        h_fov_deg=h_fov_deg,
        principal_x_rel=principal_x_rel,
        lateral_sign=lateral_sign,
    )

    mass_grid = np.zeros((bev_height, bev_width), dtype=np.float64)
    mark_numerators = {
        name: np.zeros((bev_height, bev_width), dtype=np.float64)
        for name, values in mark_maps.items()
        if values is not None
    }
    mark_denominators = {
        name: np.zeros((bev_height, bev_width), dtype=np.float64)
        for name, values in mark_maps.items()
        if values is not None
    }

    if len(bev_points) == 0:
        empty_marks = {
            name: np.full((bev_height, bev_width), np.nan, dtype=np.float64)
            for name in mark_numerators
        }
        return mass_grid, empty_marks, mass_grid.copy()

    valid_mass = flat_mass[: len(valid_mask)][valid_mask]
    valid_marks = {
        name: np.asarray(values, dtype=np.float64).reshape(-1)[: len(valid_mask)][valid_mask]
        for name, values in mark_maps.items()
        if values is not None
    }

    inside = (
        np.isfinite(bev_points[:, 0])
        & np.isfinite(bev_points[:, 1])
        & np.isfinite(valid_mass)
        & (valid_mass >= 0.0)
        & (bev_points[:, 1] >= 0.0)
        & (bev_points[:, 1] <= bev_forward_max)
        & (np.abs(bev_points[:, 0]) <= bev_lateral_max)
    )

    if not np.any(inside):
        empty_marks = {
            name: np.full((bev_height, bev_width), np.nan, dtype=np.float64)
            for name in mark_numerators
        }
        return mass_grid, empty_marks, mass_grid.copy()

    bev_points = bev_points[inside]
    valid_mass = valid_mass[inside]
    valid_marks = {
        name: values[inside]
        for name, values in valid_marks.items()
    }

    columns = np.floor(
        (bev_points[:, 0] + bev_lateral_max)
        / (2.0 * bev_lateral_max)
        * bev_width
    ).astype(np.int64)
    rows = np.floor(
        bev_points[:, 1] / bev_forward_max * bev_height
    ).astype(np.int64)

    columns = np.clip(columns, 0, bev_width - 1)
    rows = np.clip(rows, 0, bev_height - 1)

    np.add.at(mass_grid, (rows, columns), valid_mass)

    for name, mark_values in valid_marks.items():
        finite = np.isfinite(mark_values) & np.isfinite(valid_mass)
        if not np.any(finite):
            continue

        weighted_values = mark_values[finite] * valid_mass[finite]
        np.add.at(
            mark_numerators[name],
            (rows[finite], columns[finite]),
            weighted_values,
        )
        np.add.at(
            mark_denominators[name],
            (rows[finite], columns[finite]),
            valid_mass[finite],
        )

    mark_grids: dict[str, np.ndarray] = {}
    for name in mark_numerators:
        grid = np.full((bev_height, bev_width), np.nan, dtype=np.float64)
        denominator = mark_denominators[name]
        occupied = denominator > 0.0
        grid[occupied] = mark_numerators[name][occupied] / denominator[occupied]
        mark_grids[name] = grid

    return mass_grid, mark_grids, mass_grid.copy()


@HOOKS.register_module()
class PPPDemoCamToBEVWVelWRCSHook(Hook):
    """Write camera-domain and BEV PPP demo frames as PNG files."""

    def __init__(
        self,
        train_interval: int = 0,
        max_samples_train: int = 0,
        max_samples_val: int = 0,
        max_samples_test: int = 10,
        out_dir: str = "demo_bev_frames",
        img_mean=None,
        img_std=None,
        dpi: int = 100,
        figsize: tuple[float, float] = (24, 15),
        sampling_method: str = "thinning",
        sim_block_size: int = 8,
        sim_block_placement: str = "weighted",
        point_size: float = 28,
        bev_point_size: float | None = None,
        image_ext: str = "png",
        write_all_ranks: bool = False,
        gamma: float = 0.2,
        h_fov_deg: float = 110.0,
        principal_x_rel: float = 0.491,
        lateral_sign: float = 1.0,
        bev_forward_max: float = 104.0,
        bev_lateral_max: float = 52.0,
        bev_map_height: int = 256,
        bev_map_width: int = 256,
        heatmap_cmap: str = "hot",
        heatmap_interpolation: str = "bilinear",
        gt_location_color: str = "white",
        gt_location_edge_color: str = "black",
        gt_location_edge_width: float = 0.55,
        velocity_vmin: float = -11.0,
        velocity_vmax: float = 11.0,
        velocity_cmap: str = "coolwarm",
        velocity_point_edge_color: str = "black",
        velocity_point_edge_width: float = 0.45,
        rcs_link: str = "cube",
        rcs_vmin: float = -60.0,
        rcs_vmax: float = 10.0,
        rcs_cmap: str = "viridis",
        rcs_point_edge_color: str = "black",
        rcs_point_edge_width: float = 0.45,
        mark_alpha_gamma: float = 0.35,
        mark_alpha_min: float = 0.08,
        mark_alpha_max: float = 0.95,
        camera_row_height_ratio: float = 0.72,
        bev_row_height_ratio: float = 1.45,
        colorbar_height_ratio: float = 0.08,
        require_four_channels: bool = True,
    ):
        self.train_interval = int(train_interval)
        self.max_samples = {
            "train": int(max_samples_train),
            "val": int(max_samples_val),
            "test": int(max_samples_test),
        }

        self.out_dir = str(out_dir)
        self.image_ext = str(image_ext).lstrip(".")
        self.dpi = int(dpi)
        self.figsize = tuple(figsize)
        self.gamma = float(gamma)

        self.sampling_method = str(sampling_method).lower()
        if self.sampling_method not in {"thinning", "block"}:
            raise ValueError("sampling_method must be 'thinning' or 'block'")

        self.sim_block_size = int(sim_block_size)
        self.sim_block_placement = str(sim_block_placement)
        self.point_size = float(point_size)
        self.bev_point_size = (
            float(bev_point_size)
            if bev_point_size is not None
            else float(point_size)
        )
        self.write_all_ranks = bool(write_all_ranks)

        self.h_fov_deg = float(h_fov_deg)
        self.principal_x_rel = float(principal_x_rel)
        self.lateral_sign = float(lateral_sign)

        self.bev_forward_max = float(bev_forward_max)
        self.bev_lateral_max = float(bev_lateral_max)
        self.bev_map_height = int(bev_map_height)
        self.bev_map_width = int(bev_map_width)

        self.heatmap_cmap = str(heatmap_cmap)
        self.heatmap_interpolation = str(heatmap_interpolation)
        self.gt_location_color = str(gt_location_color)
        self.gt_location_edge_color = str(gt_location_edge_color)
        self.gt_location_edge_width = float(gt_location_edge_width)

        self.velocity_vmin = float(velocity_vmin)
        self.velocity_vmax = float(velocity_vmax)
        self.velocity_cmap = str(velocity_cmap)
        self.velocity_point_edge_color = str(velocity_point_edge_color)
        self.velocity_point_edge_width = float(velocity_point_edge_width)

        self.rcs_link = str(rcs_link).lower()
        if self.rcs_link not in {"identity", "cube", "linear_cube"}:
            raise ValueError(
                "rcs_link must be 'identity', 'cube', or 'linear_cube'"
            )

        self.rcs_vmin = float(rcs_vmin)
        self.rcs_vmax = float(rcs_vmax)
        self.rcs_cmap = str(rcs_cmap)
        self.rcs_point_edge_color = str(rcs_point_edge_color)
        self.rcs_point_edge_width = float(rcs_point_edge_width)

        self.mark_alpha_gamma = float(mark_alpha_gamma)
        self.mark_alpha_min = float(mark_alpha_min)
        self.mark_alpha_max = float(mark_alpha_max)

        self.camera_row_height_ratio = float(camera_row_height_ratio)
        self.bev_row_height_ratio = float(bev_row_height_ratio)
        self.colorbar_height_ratio = float(colorbar_height_ratio)
        self.require_four_channels = bool(require_four_channels)

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

    def before_val_epoch(self, runner):
        self._written_in_run["val"] = 0

    def before_test_epoch(self, runner):
        self._written_in_run["test"] = 0

    def after_train_iter(self, runner, batch_idx, data_batch=None, outputs=None):
        if not self._rank_allowed() or not isinstance(data_batch, dict):
            return
        if self.max_samples["train"] <= 0 or self.train_interval <= 0:
            return
        if runner.iter % self.train_interval != 0:
            return
        self._run(runner, data_batch, "train", runner.iter)

    def after_val_iter(self, runner, batch_idx, data_batch=None, outputs=None):
        if not self._rank_allowed() or not isinstance(data_batch, dict):
            return
        if self.max_samples["val"] <= 0:
            return
        if self._written_in_run["val"] >= self.max_samples["val"]:
            return
        self._run(runner, data_batch, "val", runner.iter)

    def after_test_iter(self, runner, batch_idx, data_batch=None, outputs=None):
        if not self._rank_allowed() or not isinstance(data_batch, dict):
            return
        if self.max_samples["test"] <= 0:
            return
        if self._written_in_run["test"] >= self.max_samples["test"]:
            return
        self._run(runner, data_batch, "test", runner.iter)

    def _run(self, runner, data_batch, phase, step):
        model = runner.model.module if is_model_wrapper(runner.model) else runner.model
        was_training = model.training
        model.eval()

        velocity_mode = getattr(model.decode_head, "velocity_mode", "signed")
        if velocity_mode not in {"signed", "absolute"}:
            raise ValueError(
                f"velocity_mode must be 'signed' or 'absolute', got {velocity_mode!r}"
            )

        proc = model.data_preprocessor(data_batch)
        inputs = proc["inputs"]
        data_samples = proc["data_samples"]

        with torch.no_grad():
            predictions = model.decode_head(model.extract_feat(inputs))
            if isinstance(predictions, (tuple, list)):
                predictions = predictions[0]

        if self.require_four_channels and predictions.size(1) < 4:
            if was_training:
                model.train()
            raise ValueError(
                "PPPDemoBEVVisHook requires at least four output channels: "
                "log intensity, depth, velocity, and RCS. "
                f"Received {predictions.size(1)} channels."
            )

        if phase == "train":
            num_visualized = min(self.max_samples["train"], predictions.size(0))
            start_index = 0
        else:
            remaining = self.max_samples[phase] - self._written_in_run[phase]
            if remaining <= 0:
                if was_training:
                    model.train()
                return
            num_visualized = min(remaining, predictions.size(0))
            start_index = self._written_in_run[phase]

        for index in range(num_visualized):
            self._visualize_sample(
                runner=runner,
                phase=phase,
                img_tensor=inputs[index],
                prediction=predictions[index],
                sample=data_samples[index],
                iteration_seed=getattr(runner, "iter", 0),
                sample_index=start_index + index,
                velocity_mode=velocity_mode,
            )

        if phase in self._written_in_run:
            self._written_in_run[phase] += num_visualized

        if was_training:
            model.train()

    def _decode_velocity(
        self,
        raw_velocity: torch.Tensor,
        velocity_mode: str,
    ) -> np.ndarray:
        if velocity_mode == "signed":
            velocity = raw_velocity.pow(3)
        elif velocity_mode == "absolute":
            velocity = torch.abs(raw_velocity)
        else:
            raise RuntimeError(f"Unsupported velocity_mode={velocity_mode!r}")
        return velocity.detach().cpu().numpy()

    def _decode_rcs(self, raw_rcs: torch.Tensor) -> np.ndarray:
        if self.rcs_link == "identity":
            rcs = raw_rcs
        elif self.rcs_link == "cube":
            rcs = raw_rcs.pow(3)
        elif self.rcs_link == "linear_cube":
            rcs = raw_rcs + raw_rcs.pow(3)
        else:
            raise RuntimeError(f"Unsupported rcs_link={self.rcs_link!r}")
        return rcs.detach().cpu().numpy()

    def _denorm_img(self, image: np.ndarray) -> np.ndarray:
        image = image.astype(np.float32)

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

    @staticmethod
    def _textbox(ax, text: str, dark: bool = True):
        ax.text(
            0.02,
            0.98,
            text,
            transform=ax.transAxes,
            fontsize=9,
            color="white" if dark else "black",
            va="top",
            ha="left",
            bbox=dict(
                facecolor="black" if dark else "white",
                alpha=0.62,
                pad=3,
            ),
            zorder=10,
        )

    def _sample_prediction(self, lam_mass: np.ndarray, seed: int):
        if self.sampling_method == "thinning":
            return sample_ppp_thinning(
                lam_mass,
                rng=np.random.default_rng(seed + 11),
            )

        return sample_ppp_block(
            lam_mass,
            block_size=self.sim_block_size,
            placement=self.sim_block_placement,
            rng=np.random.default_rng(seed + 29),
        )

    def _plot_camera_heatmap(self, ax, heatmap: np.ndarray, expected_count: float):
        display = np.power(np.maximum(heatmap, 0.0), self.gamma)
        ax.imshow(
            display,
            cmap=self.heatmap_cmap,
            aspect="auto",
            interpolation=self.heatmap_interpolation,
        )
        ax.set_title("Camera domain: predicted PPP heatmap", fontsize=11, pad=3)
        ax.axis("off")
        self._textbox(
            ax,
            f"E[N]={expected_count:.2f}\ngamma={self.gamma:g}",
            dark=True,
        )

    def _plot_camera_mark_points(
        self,
        ax,
        image: np.ndarray,
        points_img: np.ndarray,
        values: np.ndarray,
        title: str,
        cmap: str,
        vmin: float,
        vmax: float,
        edge_color: str,
        edge_width: float,
        textbox: str,
    ):
        ax.imshow(image, aspect="auto")
        ax.set_title(title, fontsize=11, pad=3)
        ax.axis("off")

        if len(points_img) and len(values):
            count = min(len(points_img), len(values))
            points = points_img[:count]
            point_values = values[:count]

            valid = (
                np.isfinite(points[:, 0])
                & np.isfinite(points[:, 1])
                & np.isfinite(point_values)
            )

            if np.any(valid):
                ax.scatter(
                    points[valid, 0],
                    points[valid, 1],
                    s=self.point_size,
                    c=point_values[valid],
                    cmap=cmap,
                    vmin=vmin,
                    vmax=vmax,
                    alpha=0.98,
                    edgecolors=edge_color,
                    linewidths=edge_width,
                )

        self._textbox(ax, textbox, dark=True)

    def _setup_bev_axis(self, ax, title: str):
        ax.set_title(title, fontsize=11, pad=4)
        ax.set_facecolor("black")
        ax.set_xlim(self.bev_lateral_max, -self.bev_lateral_max)
        ax.set_ylim(0.0, self.bev_forward_max)
        ax.set_aspect("equal", adjustable="box")
        ax.set_box_aspect(1)
        ax.set_xlabel("lateral position [m]", labelpad=2)
        ax.set_ylabel("forward distance [m]", labelpad=2)
        ax.grid(True, color="white", alpha=0.15, linewidth=0.7)
        ax.tick_params(labelsize=8, pad=2)

        ax.scatter(
            [0.0],
            [0.0],
            marker="^",
            s=90,
            c="white",
            edgecolors="black",
            linewidths=0.8,
            zorder=5,
        )

    def _plot_bev_intensity(
        self,
        ax,
        intensity_grid: np.ndarray,
        gt_bev: np.ndarray,
        expected_count: float,
    ):
        self._setup_bev_axis(ax, "BEV: predicted PPP heatmap + GT locations")

        display = np.power(np.maximum(intensity_grid, 0.0), self.gamma)
        ax.imshow(
            display,
            origin="lower",
            extent=[
                -self.bev_lateral_max,
                self.bev_lateral_max,
                0.0,
                self.bev_forward_max,
            ],
            cmap=self.heatmap_cmap,
            aspect="equal",
            interpolation=self.heatmap_interpolation,
            zorder=1,
        )

        if len(gt_bev):
            valid = (
                np.isfinite(gt_bev[:, 0])
                & np.isfinite(gt_bev[:, 1])
                & (gt_bev[:, 1] >= 0.0)
                & (gt_bev[:, 1] <= self.bev_forward_max)
                & (np.abs(gt_bev[:, 0]) <= self.bev_lateral_max)
            )
            if np.any(valid):
                ax.scatter(
                    gt_bev[valid, 0],
                    gt_bev[valid, 1],
                    s=self.bev_point_size,
                    c=self.gt_location_color,
                    edgecolors=self.gt_location_edge_color,
                    linewidths=self.gt_location_edge_width,
                    alpha=0.98,
                    zorder=4,
                )

        self._textbox(
            ax,
            f"background=pred intensity\ndots=GT locations\nE[N]={expected_count:.2f}",
            dark=True,
        )

    def _mass_to_alpha(self, mass_grid: np.ndarray) -> np.ndarray:
        mass = np.maximum(np.asarray(mass_grid, dtype=np.float64), 0.0)
        maximum = float(np.nanmax(mass)) if mass.size else 0.0

        if not np.isfinite(maximum) or maximum <= 0.0:
            return np.zeros_like(mass, dtype=np.float64)

        normalized = np.clip(mass / maximum, 0.0, 1.0)
        alpha = np.power(normalized, self.mark_alpha_gamma)
        alpha = self.mark_alpha_min + (
            self.mark_alpha_max - self.mark_alpha_min
        ) * alpha
        alpha[mass <= 0.0] = 0.0
        return alpha

    def _plot_bev_mark(
        self,
        ax,
        mark_grid: np.ndarray | None,
        mass_grid: np.ndarray,
        gt_bev: np.ndarray,
        gt_values: np.ndarray,
        title: str,
        cmap: str,
        vmin: float,
        vmax: float,
        edge_color: str,
        edge_width: float,
        textbox: str,
    ):
        self._setup_bev_axis(ax, title)

        if mark_grid is not None:
            alpha = self._mass_to_alpha(mass_grid)
            valid_mark = np.isfinite(mark_grid)
            alpha = np.where(valid_mark, alpha, 0.0)
            display = np.where(valid_mark, mark_grid, 0.0)

            ax.imshow(
                display,
                origin="lower",
                extent=[
                    -self.bev_lateral_max,
                    self.bev_lateral_max,
                    0.0,
                    self.bev_forward_max,
                ],
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                alpha=alpha,
                aspect="equal",
                zorder=1,
            )

        if len(gt_bev) and len(gt_values):
            count = min(len(gt_bev), len(gt_values))
            points = gt_bev[:count]
            values = gt_values[:count]

            valid = (
                np.isfinite(points[:, 0])
                & np.isfinite(points[:, 1])
                & np.isfinite(values)
                & (points[:, 1] >= 0.0)
                & (points[:, 1] <= self.bev_forward_max)
                & (np.abs(points[:, 0]) <= self.bev_lateral_max)
            )

            if np.any(valid):
                ax.scatter(
                    points[valid, 0],
                    points[valid, 1],
                    s=self.bev_point_size,
                    c=values[valid],
                    cmap=cmap,
                    vmin=vmin,
                    vmax=vmax,
                    alpha=1.0,
                    edgecolors=edge_color,
                    linewidths=edge_width,
                    zorder=4,
                )

        self._textbox(ax, textbox, dark=True)

    @staticmethod
    def _add_colorbar(
        fig,
        cax,
        cmap: str,
        vmin: float,
        vmax: float,
        label: str,
    ):
        scalar_mappable = ScalarMappable(
            norm=Normalize(vmin=vmin, vmax=vmax),
            cmap=cmap,
        )
        scalar_mappable.set_array([])
        colorbar = fig.colorbar(
            scalar_mappable,
            cax=cax,
            orientation="horizontal",
        )
        colorbar.set_label(label, fontsize=9)
        colorbar.ax.tick_params(labelsize=8)
        colorbar.outline.set_linewidth(0.8)

    def _output_path(
        self,
        runner,
        phase: str,
        video: str,
        frame_str: str,
        global_index_str: str,
    ) -> Path:
        root = Path(self.out_dir)
        if not root.is_absolute():
            root = Path(runner.work_dir) / root

        output_directory = root / phase / video
        output_directory.mkdir(parents=True, exist_ok=True)
        return output_directory / (
            f"frame_{frame_str}_idx_{global_index_str}.{self.image_ext}"
        )

    def _visualize_sample(
        self,
        runner,
        phase: str,
        img_tensor: torch.Tensor,
        prediction: torch.Tensor,
        sample,
        iteration_seed: int,
        sample_index: int,
        velocity_mode: str,
    ):
        gt_all = sample.gt_sem_seg.data.to(prediction.device)

        gt_mask_tensor = gt_all[0]
        gt_depth_tensor = gt_all[1] if gt_all.size(0) > 1 else None
        gt_velocity_tensor = gt_all[2] if gt_all.size(0) > 2 else None
        gt_rcs_tensor = gt_all[3] if gt_all.size(0) > 3 else None

        lam = torch.exp(prediction[0]).detach().cpu().numpy()
        pred_depth = torch.exp(prediction[1]).detach().cpu().numpy()
        pred_velocity = self._decode_velocity(prediction[2], velocity_mode)
        pred_rcs = self._decode_rcs(prediction[3])

        gt_mask = gt_mask_tensor.detach().cpu().numpy()
        gt_depth = (
            gt_depth_tensor.detach().cpu().numpy()
            if gt_depth_tensor is not None
            else None
        )
        gt_velocity = (
            gt_velocity_tensor.detach().cpu().numpy()
            if gt_velocity_tensor is not None
            else None
        )
        gt_rcs = (
            gt_rcs_tensor.detach().cpu().numpy()
            if gt_rcs_tensor is not None
            else None
        )

        if velocity_mode == "absolute" and gt_velocity is not None:
            gt_velocity = np.abs(gt_velocity)

        image = self._denorm_img(img_tensor.detach().cpu().numpy())
        image_h, image_w = image.shape[:2]
        pred_h, pred_w = lam.shape[-2:]
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

        video = self._tagify(
            str(metadata.get("video_name", "unknown_video"))
        )
        frame = self._to_int_or_none(metadata.get("video_frame_idx"))
        global_index = self._to_int_or_none(
            metadata.get("global_idx", metadata.get("frame_idx"))
        )

        frame_str = f"{frame:06d}" if frame is not None else "unknown_frame"
        global_index_str = (
            f"{global_index:08d}"
            if global_index is not None
            else "unknown_idx"
        )

        original_shape = metadata.get("ori_shape")
        if (
            isinstance(original_shape, (tuple, list))
            and len(original_shape) >= 2
        ):
            norm_h, norm_w = int(original_shape[0]), int(original_shape[1])
        else:
            norm_h, norm_w = pred_h, pred_w

        lam_mass = lam / float(max(norm_h * norm_w, 1))
        expected_count = float(lam_mass.sum())

        gt_points_grid, gt_depth_values = extract_gt_points_and_values(
            gt_mask,
            gt_depth,
        )
        _, gt_velocity_values = extract_gt_points_and_values(
            gt_mask,
            gt_velocity,
        )
        _, gt_rcs_values = extract_gt_points_and_values(
            gt_mask,
            gt_rcs,
        )

        gt_points_img = scale_points_to_shape(
            gt_points_grid,
            (gt_h, gt_w),
            (image_h, image_w),
        )

        seed_base = (
            frame
            if frame is not None
            else global_index
            if global_index is not None
            else sample_index
        )
        seed = int(seed_base) + 1000 * int(iteration_seed)

        pred_points_grid, pred_stats = self._sample_prediction(lam_mass, seed)
        pred_points_img = scale_points_to_shape(
            pred_points_grid,
            (pred_h, pred_w),
            (image_h, image_w),
        )

        pred_velocity_values = sample_grid_values_at_points(
            pred_velocity,
            pred_points_grid,
        )
        pred_rcs_values = sample_grid_values_at_points(
            pred_rcs,
            pred_points_grid,
        )

        gt_bev, gt_valid = backproject_pixels_to_bev(
            points_img=gt_points_img,
            depth_m=gt_depth_values,
            rad2cam=rad2cam,
            image_width=image_w,
            h_fov_deg=h_fov_deg,
            principal_x_rel=self.principal_x_rel,
            lateral_sign=self.lateral_sign,
        )

        gt_velocity_bev = (
            gt_velocity_values[: len(gt_valid)][gt_valid]
            if len(gt_valid)
            else np.zeros((0,), dtype=np.float64)
        )
        gt_rcs_bev = (
            gt_rcs_values[: len(gt_valid)][gt_valid]
            if len(gt_valid)
            else np.zeros((0,), dtype=np.float64)
        )

        bev_mass, bev_marks, bev_mark_mass = project_dense_fields_to_bev(
            intensity_mass=lam_mass,
            depth_map=pred_depth,
            image_hw=(image_h, image_w),
            rad2cam=rad2cam,
            h_fov_deg=h_fov_deg,
            principal_x_rel=self.principal_x_rel,
            lateral_sign=self.lateral_sign,
            bev_forward_max=self.bev_forward_max,
            bev_lateral_max=self.bev_lateral_max,
            bev_height=self.bev_map_height,
            bev_width=self.bev_map_width,
            mark_maps={
                "velocity": pred_velocity,
                "rcs": pred_rcs,
            },
        )

        figure = Figure(figsize=self.figsize, dpi=self.dpi)
        FigureCanvasAgg(figure)

        grid = figure.add_gridspec(
            nrows=3,
            ncols=3,
            height_ratios=[
                self.camera_row_height_ratio,
                self.bev_row_height_ratio,
                self.colorbar_height_ratio,
            ],
            left=0.025,
            right=0.992,
            top=0.95,
            bottom=0.04,
            wspace=0.065,
            hspace=0.13,
        )

        ax_camera_heatmap = figure.add_subplot(grid[0, 0])
        ax_camera_velocity = figure.add_subplot(grid[0, 1])
        ax_camera_rcs = figure.add_subplot(grid[0, 2])

        ax_bev_heatmap = figure.add_subplot(grid[1, 0])
        ax_bev_velocity = figure.add_subplot(grid[1, 1])
        ax_bev_rcs = figure.add_subplot(grid[1, 2])

        ax_empty_colorbar = figure.add_subplot(grid[2, 0])
        ax_velocity_colorbar = figure.add_subplot(grid[2, 1])
        ax_rcs_colorbar = figure.add_subplot(grid[2, 2])
        ax_empty_colorbar.axis("off")

        figure.suptitle(
            f"{phase} | {video} | frame {frame_str} | idx {global_index_str}",
            fontsize=13,
            y=0.982,
        )

        self._plot_camera_heatmap(
            ax_camera_heatmap,
            heatmap=lam_mass,
            expected_count=expected_count,
        )

        sample_count = pred_stats.get("num_points", len(pred_points_img))

        self._plot_camera_mark_points(
            ax_camera_velocity,
            image=image,
            points_img=pred_points_img,
            values=pred_velocity_values,
            title="Camera domain: predicted velocity",
            cmap=self.velocity_cmap,
            vmin=self.velocity_vmin,
            vmax=self.velocity_vmax,
            edge_color=self.velocity_point_edge_color,
            edge_width=self.velocity_point_edge_width,
            textbox=(
                f"sample={self.sampling_method}\n"
                f"N={sample_count}\n"
                f"mode={velocity_mode}"
            ),
        )

        self._plot_camera_mark_points(
            ax_camera_rcs,
            image=image,
            points_img=pred_points_img,
            values=pred_rcs_values,
            title="Camera domain: predicted RCS",
            cmap=self.rcs_cmap,
            vmin=self.rcs_vmin,
            vmax=self.rcs_vmax,
            edge_color=self.rcs_point_edge_color,
            edge_width=self.rcs_point_edge_width,
            textbox=(
                f"sample={self.sampling_method}\n"
                f"N={sample_count}\n"
                f"link={self.rcs_link}"
            ),
        )

        self._plot_bev_intensity(
            ax_bev_heatmap,
            intensity_grid=bev_mass,
            gt_bev=gt_bev,
            expected_count=expected_count,
        )

        self._plot_bev_mark(
            ax_bev_velocity,
            mark_grid=bev_marks.get("velocity"),
            mass_grid=bev_mark_mass,
            gt_bev=gt_bev,
            gt_values=gt_velocity_bev,
            title="BEV: predicted velocity + GT velocity",
            cmap=self.velocity_cmap,
            vmin=self.velocity_vmin,
            vmax=self.velocity_vmax,
            edge_color=self.velocity_point_edge_color,
            edge_width=self.velocity_point_edge_width,
            textbox=(
                "background=pred velocity\n"
                "dots=GT velocity"
            ),
        )

        self._plot_bev_mark(
            ax_bev_rcs,
            mark_grid=bev_marks.get("rcs"),
            mass_grid=bev_mark_mass,
            gt_bev=gt_bev,
            gt_values=gt_rcs_bev,
            title="BEV: predicted RCS + GT RCS",
            cmap=self.rcs_cmap,
            vmin=self.rcs_vmin,
            vmax=self.rcs_vmax,
            edge_color=self.rcs_point_edge_color,
            edge_width=self.rcs_point_edge_width,
            textbox=(
                "background=pred RCS\n"
                "dots=GT RCS"
            ),
        )

        self._add_colorbar(
            figure,
            ax_velocity_colorbar,
            cmap=self.velocity_cmap,
            vmin=self.velocity_vmin,
            vmax=self.velocity_vmax,
            label="radial velocity [m/s]",
        )
        self._add_colorbar(
            figure,
            ax_rcs_colorbar,
            cmap=self.rcs_cmap,
            vmin=self.rcs_vmin,
            vmax=self.rcs_vmax,
            label="RCS",
        )

        output_path = self._output_path(
            runner=runner,
            phase=phase,
            video=video,
            frame_str=frame_str,
            global_index_str=global_index_str,
        )

        figure.savefig(output_path, dpi=self.dpi, bbox_inches="tight")
        figure.clear()