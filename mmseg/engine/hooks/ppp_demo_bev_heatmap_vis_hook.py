"""
PPP demo BEV heatmap visualization hook.

Layout:
  1. top left:  camera image with GT radar points, colored by GT depth
  2. top right: predicted PPP heatmap in pixel space
  3. bottom left: GT points back-projected into BEV, colored by GT depth
  4. bottom right: predicted PPP heatmap projected into BEV via predicted depth

Important correction compared with the first BEV heatmap version:
    The BEV heatmap is now projected with mass-preserving area-style splatting.

    Each image pixel mass is split into subpixel samples.
    Each subpixel is back-projected with bilinear depth interpolation.
    Each subpixel mass is then bilinearly splatted to the four nearest BEV bins.

    This means a compact bright image-space region that corresponds to a large
    physical BEV area is spread over more BEV bins, instead of being collapsed
    into isolated points or stripes.

By default the BEV window is lateral [-50, 50] m and forward [0, 104] m.
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
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.figure import Figure


def make_synthetic_depth_cmap(name: str = "synthetic_depth_clash"):
    return LinearSegmentedColormap.from_list(
        name,
        [
            "#ff00ff",
            "#7a00ff",
            "#003cff",
            "#00e5ff",
            "#00ff66",
            "#ffff00",
            "#ff6a00",
            "#ff0000",
        ],
        N=256,
    )


def extract_gt_points_and_depths(
    mask: np.ndarray,
    depth: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return GT point coordinates and GT depth values."""
    ys, xs = np.where(mask > 0)

    if xs.size == 0:
        return (
            np.zeros((0, 2), dtype=np.float64),
            np.zeros((0,), dtype=np.float64),
        )

    pts = np.stack([xs + 0.5, ys + 0.5], axis=1).astype(np.float64)

    if depth is None:
        depths = np.zeros((len(pts),), dtype=np.float64)
    else:
        depths = depth[ys, xs].astype(np.float64)

    return pts, depths


def scale_points_to_shape(
    pts: np.ndarray,
    from_hw: tuple[int, int],
    to_hw: tuple[int, int],
) -> np.ndarray:
    """Scale points from one H,W grid to another H,W grid."""
    if len(pts) == 0:
        return pts.astype(np.float64)

    from_h, from_w = int(from_hw[0]), int(from_hw[1])
    to_h, to_w = int(to_hw[0]), int(to_hw[1])

    if from_h == to_h and from_w == to_w:
        return pts.astype(np.float64, copy=True)

    out = pts.astype(np.float64, copy=True)
    out[:, 0] *= float(to_w) / float(max(from_w, 1))
    out[:, 1] *= float(to_h) / float(max(from_h, 1))
    return out


def backproject_pixels_to_bev(
    pts_img: np.ndarray,
    depth_m: np.ndarray,
    image_width: int,
    h_fov_deg: float = 110.0,
    principal_x_rel: float = 0.5,
    lateral_sign: float = 1.0,
    depth_is_forward: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Back-project image x-coordinate plus depth into BEV coordinates.

    Returns:
        bev: shape (N_valid, 2), columns are (lateral_m, forward_m)
        valid: boolean mask into the original input arrays
    """
    pts_img = np.asarray(pts_img, dtype=np.float64)
    depth_m = np.asarray(depth_m, dtype=np.float64)

    if len(pts_img) == 0 or len(depth_m) == 0:
        return np.zeros((0, 2), dtype=np.float64), np.zeros((len(pts_img),), dtype=bool)

    n = min(len(pts_img), len(depth_m))
    pts_img = pts_img[:n]
    depth_m = depth_m[:n]

    valid = np.isfinite(pts_img[:, 0]) & np.isfinite(depth_m) & (depth_m > 0.0)
    if not np.any(valid):
        return np.zeros((0, 2), dtype=np.float64), valid

    w = float(max(int(image_width), 1))
    cx = float(principal_x_rel) * (w - 1.0)
    fx = (w / 2.0) / np.tan(np.deg2rad(float(h_fov_deg)) / 2.0)

    u = pts_img[valid, 0]
    d = depth_m[valid]
    theta = np.arctan((u - cx) / fx)

    if depth_is_forward:
        forward = d
        lateral = d * np.tan(theta)
    else:
        forward = d * np.cos(theta)
        lateral = d * np.sin(theta)

    lateral = float(lateral_sign) * lateral
    bev = np.stack([lateral, forward], axis=1).astype(np.float64)
    return bev, valid


def _gaussian_kernel1d(sigma: float) -> np.ndarray:
    sigma = float(sigma)
    if sigma <= 0.0:
        return np.array([1.0], dtype=np.float64)

    radius = max(1, int(np.ceil(3.0 * sigma)))
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    k = np.exp(-0.5 * (x / sigma) ** 2)
    k /= k.sum()
    return k


def gaussian_smooth2d(arr: np.ndarray, sigma: float) -> np.ndarray:
    """Lightweight dependency-free Gaussian smoothing."""
    sigma = float(sigma)
    if sigma <= 0.0:
        return arr

    arr = np.asarray(arr, dtype=np.float64)
    k = _gaussian_kernel1d(sigma)
    r = len(k) // 2

    padded_x = np.pad(arr, ((0, 0), (r, r)), mode="edge")
    tmp = np.empty_like(arr, dtype=np.float64)
    for j in range(arr.shape[1]):
        tmp[:, j] = np.sum(padded_x[:, j:j + len(k)] * k[None, :], axis=1)

    padded_y = np.pad(tmp, ((r, r), (0, 0)), mode="edge")
    out = np.empty_like(arr, dtype=np.float64)
    for i in range(arr.shape[0]):
        out[i, :] = np.sum(padded_y[i:i + len(k), :] * k[:, None], axis=0)

    return out


def _bilinear_splat_to_bev(
    bev: np.ndarray,
    lateral: np.ndarray,
    forward: np.ndarray,
    mass: np.ndarray,
    *,
    bev_forward_max: float,
    bev_lateral_max: float,
) -> None:
    """Bilinearly splat mass into the BEV grid in-place."""
    bev_h, bev_w = bev.shape

    x = ((lateral + float(bev_lateral_max)) / (2.0 * float(bev_lateral_max))) * (bev_w - 1)
    y = (forward / float(bev_forward_max)) * (bev_h - 1)

    valid = np.isfinite(x)
    valid &= np.isfinite(y)
    valid &= np.isfinite(mass)
    valid &= mass > 0.0
    valid &= x >= 0.0
    valid &= x <= (bev_w - 1)
    valid &= y >= 0.0
    valid &= y <= (bev_h - 1)

    if not np.any(valid):
        return

    x = x[valid]
    y = y[valid]
    mass = mass[valid]

    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    x1 = np.clip(x0 + 1, 0, bev_w - 1)
    y1 = np.clip(y0 + 1, 0, bev_h - 1)

    wx = x - x0
    wy = y - y0

    w00 = (1.0 - wx) * (1.0 - wy)
    w01 = wx * (1.0 - wy)
    w10 = (1.0 - wx) * wy
    w11 = wx * wy

    np.add.at(bev, (y0, x0), mass * w00)
    np.add.at(bev, (y0, x1), mass * w01)
    np.add.at(bev, (y1, x0), mass * w10)
    np.add.at(bev, (y1, x1), mass * w11)


def project_mass_heatmap_to_bev(
    lam_mass: np.ndarray,
    depth_m: np.ndarray,
    *,
    bev_h: int,
    bev_w: int,
    bev_forward_max: float,
    bev_lateral_max: float,
    h_fov_deg: float,
    principal_x_rel: float,
    lateral_sign: float,
    depth_is_forward: bool,
    subpixel_factor: int = 3,
    chunk_rows: int = 32,
    min_mass: float = 0.0,
    smoothing_sigma: float = 0.0,
) -> np.ndarray:
    """
    Project image-space PPP mass heatmap into a BEV mass heatmap.

    This is mass-preserving area-style splatting:
      1. Each image pixel mass is split into subpixel_factor^2 sub-samples.
      2. For every sub-sample, depth is bilinearly interpolated.
      3. The sub-sample is projected to continuous BEV coordinates.
      4. Its mass is bilinearly splatted to the four nearest BEV cells.
    """
    lam_mass = np.asarray(lam_mass, dtype=np.float64)
    depth_m = np.asarray(depth_m, dtype=np.float64)

    if lam_mass.ndim != 2 or depth_m.ndim != 2:
        raise ValueError("lam_mass and depth_m must be 2D arrays")
    if lam_mass.shape != depth_m.shape:
        raise ValueError("lam_mass and depth_m must have the same shape")

    h, w = lam_mass.shape
    bev = np.zeros((int(bev_h), int(bev_w)), dtype=np.float64)

    if h == 0 or w == 0:
        return bev

    subpixel_factor = max(1, int(subpixel_factor))
    n_sub = subpixel_factor * subpixel_factor
    chunk_rows = max(1, int(chunk_rows))
    min_mass = float(min_mass)

    x_base = np.arange(w, dtype=np.float64)
    x_idx = x_base.astype(np.int64)
    x_right = np.clip(x_idx + 1, 0, w - 1)

    cx = float(principal_x_rel) * (w - 1.0)
    fx = (w / 2.0) / np.tan(np.deg2rad(float(h_fov_deg)) / 2.0)

    offsets = (np.arange(subpixel_factor, dtype=np.float64) + 0.5) / subpixel_factor

    for row_start in range(0, h, chunk_rows):
        row_end = min(h, row_start + chunk_rows)
        rows = np.arange(row_start, row_end, dtype=np.int64)
        rows_down = np.clip(rows + 1, 0, h - 1)

        mass_base = lam_mass[row_start:row_end, :]
        if min_mass > 0.0:
            mass_base = np.where(mass_base >= min_mass, mass_base, 0.0)

        if not np.any(mass_base > 0.0):
            continue

        mass_each = mass_base / float(n_sub)

        depth00 = depth_m[rows[:, None], x_idx[None, :]]
        depth01 = depth_m[rows[:, None], x_right[None, :]]
        depth10 = depth_m[rows_down[:, None], x_idx[None, :]]
        depth11 = depth_m[rows_down[:, None], x_right[None, :]]

        for oy in offsets:
            for ox in offsets:
                depth_top = (1.0 - ox) * depth00 + ox * depth01
                depth_bottom = (1.0 - ox) * depth10 + ox * depth11
                depth_sub = (1.0 - oy) * depth_top + oy * depth_bottom

                u_sub = x_base + ox
                theta = np.arctan((u_sub - cx) / fx)

                if depth_is_forward:
                    forward = depth_sub
                    lateral = depth_sub * np.tan(theta[None, :])
                else:
                    forward = depth_sub * np.cos(theta[None, :])
                    lateral = depth_sub * np.sin(theta[None, :])

                lateral = float(lateral_sign) * lateral

                valid = np.isfinite(mass_each)
                valid &= np.isfinite(forward)
                valid &= np.isfinite(lateral)
                valid &= mass_each > 0.0
                valid &= forward > 0.0
                valid &= forward <= float(bev_forward_max)
                valid &= lateral >= -float(bev_lateral_max)
                valid &= lateral <= float(bev_lateral_max)

                if not np.any(valid):
                    continue

                _bilinear_splat_to_bev(
                    bev,
                    lateral=lateral[valid],
                    forward=forward[valid],
                    mass=mass_each[valid],
                    bev_forward_max=bev_forward_max,
                    bev_lateral_max=bev_lateral_max,
                )

    if smoothing_sigma > 0.0:
        bev = gaussian_smooth2d(bev, smoothing_sigma)

    return bev


@HOOKS.register_module()
class PPPDemoBEVHeatmapVisHook(Hook):
    """Write PPP GT-point plus predicted-heatmap BEV visualizations as PNG files."""

    def __init__(
        self,
        train_interval=0,
        max_samples_train=0,
        max_samples_val=0,
        max_samples_test=10,
        out_dir="demo_frames",
        img_mean=None,
        img_std=None,
        dpi=110,
        figsize=(24, 18),
        image_ext="png",
        write_all_ranks=False,
        h_fov_deg=110.0,
        principal_x_rel=0.5,
        lateral_sign=1.0,
        depth_is_forward=True,
        bev_forward_max=104.0,
        bev_lateral_max=50.0,
        bev_grid_shape=(416, 400),
        distance_vmin=0.0,
        distance_vmax=None,
        depth_cmap="synthetic",
        depth_point_edge_color="black",
        depth_point_edge_width=0.45,
        point_size=28,
        bev_point_size=None,
        heatmap_cmap="hot",
        heatmap_gamma=0.45,
        heatmap_vmin=0.0,
        heatmap_vmax=None,
        heatmap_percentile=99.7,
        heatmap_interpolation="nearest",
        bev_heatmap_interpolation="bilinear",
        bev_row_height_ratio=2.5,
        colorbar_height_ratio=0.09,
        bev_projection_subpixel_factor=3,
        bev_projection_chunk_rows=32,
        bev_projection_min_mass=0.0,
        bev_heatmap_smoothing_sigma=0.7,
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
        self.write_all_ranks = bool(write_all_ranks)

        self.h_fov_deg = float(h_fov_deg)
        self.principal_x_rel = float(principal_x_rel)
        self.lateral_sign = float(lateral_sign)
        self.depth_is_forward = bool(depth_is_forward)

        self.bev_forward_max = float(bev_forward_max)
        self.bev_lateral_max = float(bev_lateral_max)
        self.bev_grid_shape = tuple(int(x) for x in bev_grid_shape)

        self.distance_vmin = float(distance_vmin) if distance_vmin is not None else None
        if distance_vmax is None:
            self.distance_vmax = self.bev_forward_max
        else:
            self.distance_vmax = float(distance_vmax)

        if depth_cmap == "synthetic":
            self.depth_cmap = make_synthetic_depth_cmap()
        else:
            self.depth_cmap = depth_cmap

        self.depth_point_edge_color = str(depth_point_edge_color)
        self.depth_point_edge_width = float(depth_point_edge_width)
        self.point_size = float(point_size)
        self.bev_point_size = float(bev_point_size) if bev_point_size is not None else float(point_size)

        self.heatmap_cmap = str(heatmap_cmap)
        self.heatmap_gamma = float(heatmap_gamma)
        self.heatmap_vmin = float(heatmap_vmin)
        self.heatmap_vmax = None if heatmap_vmax is None else float(heatmap_vmax)
        self.heatmap_percentile = float(heatmap_percentile)
        self.heatmap_interpolation = str(heatmap_interpolation)
        self.bev_heatmap_interpolation = str(bev_heatmap_interpolation)

        self.bev_row_height_ratio = float(bev_row_height_ratio)
        self.colorbar_height_ratio = float(colorbar_height_ratio)

        self.bev_projection_subpixel_factor = int(bev_projection_subpixel_factor)
        self.bev_projection_chunk_rows = int(bev_projection_chunk_rows)
        self.bev_projection_min_mass = float(bev_projection_min_mass)
        self.bev_heatmap_smoothing_sigma = float(bev_heatmap_smoothing_sigma)

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

        proc = model.data_preprocessor(data_batch)
        inputs = proc["inputs"]
        data_samples = proc["data_samples"]

        with torch.no_grad():
            preds = model.decode_head(model.extract_feat(inputs))
            if isinstance(preds, (tuple, list)):
                preds = preds[0]

        if phase == "train":
            n_vis = min(self.max_samples["train"], preds.size(0))
            start_idx = 0
        else:
            remaining = self.max_samples[phase] - self._written_in_run[phase]
            if remaining <= 0:
                if was_training:
                    model.train()
                return
            n_vis = min(remaining, preds.size(0))
            start_idx = self._written_in_run[phase]

        for i in range(n_vis):
            self._visualize_sample(
                runner=runner,
                step=step,
                phase=phase,
                img_tensor=inputs[i],
                pred=preds[i],
                sample=data_samples[i],
                sample_idx=start_idx + i,
            )

        if phase in self._written_in_run:
            self._written_in_run[phase] += n_vis

        if was_training:
            model.train()

    def _denorm_img(self, x):
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
    def _tagify(s: str) -> str:
        return "".join(c if c.isalnum() or c in "._-/" else "_" for c in s)

    @staticmethod
    def _to_int_or_none(x):
        if x is None:
            return None
        if isinstance(x, torch.Tensor):
            if x.numel() == 0:
                return None
            return int(x.flatten()[0].item())
        try:
            return int(x)
        except Exception:
            return None

    @staticmethod
    def _textbox(ax, text, dark=True, loc="upper_left"):
        if loc == "upper_right":
            x, y, ha = 0.98, 0.98, "right"
        else:
            x, y, ha = 0.02, 0.98, "left"

        ax.text(
            x,
            y,
            text,
            transform=ax.transAxes,
            fontsize=10,
            color="white" if dark else "black",
            va="top",
            ha=ha,
            bbox=dict(facecolor="black" if dark else "white", alpha=0.62, pad=3),
        )

    def _display_heatmap(self, arr: np.ndarray) -> np.ndarray:
        arr = np.asarray(arr, dtype=np.float64)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        arr = np.clip(arr, 0.0, None)

        if self.heatmap_gamma == 1.0:
            return arr
        return arr ** self.heatmap_gamma

    def _heatmap_scale(self, *arrays: np.ndarray) -> tuple[float, float]:
        if self.heatmap_gamma == 1.0:
            vmin = self.heatmap_vmin
            vmax_cfg = self.heatmap_vmax
        else:
            vmin = max(self.heatmap_vmin, 0.0) ** self.heatmap_gamma
            vmax_cfg = None if self.heatmap_vmax is None else max(self.heatmap_vmax, 0.0) ** self.heatmap_gamma

        if vmax_cfg is not None:
            vmax = float(vmax_cfg)
            return float(vmin), max(vmax, float(vmin) + 1e-12)

        vals = []
        for a in arrays:
            x = np.asarray(a, dtype=np.float64)
            x = x[np.isfinite(x)]
            x = x[x > 0.0]
            if x.size:
                vals.append(x)

        if not vals:
            return float(vmin), 1.0

        all_vals = np.concatenate(vals)
        vmax = np.percentile(all_vals, self.heatmap_percentile)
        vmax = float(max(vmax, all_vals.max() * 0.2, 1e-12))
        return float(vmin), vmax

    def _overlay_depth_points(self, ax, img, pts, depth_values, title, textbox):
        ax.imshow(img, aspect="auto")
        ax.set_title(title, fontsize=11, pad=3)
        ax.axis("off")

        if len(pts) and len(depth_values):
            values = np.asarray(depth_values, dtype=np.float64)
            n = min(len(pts), len(values))
            pts = pts[:n]
            values = values[:n]

            valid = np.isfinite(pts[:, 0]) & np.isfinite(pts[:, 1])
            valid &= np.isfinite(values) & (values > 0.0)

            if np.any(valid):
                ax.scatter(
                    pts[valid, 0],
                    pts[valid, 1],
                    s=self.point_size,
                    c=values[valid],
                    cmap=self.depth_cmap,
                    vmin=self.distance_vmin,
                    vmax=self.distance_vmax,
                    alpha=0.98,
                    edgecolors=self.depth_point_edge_color,
                    linewidths=self.depth_point_edge_width,
                )

        self._textbox(ax, textbox, dark=True)

    def _plot_bev_points(self, ax, bev, depth_values, title, textbox):
        ax.set_title(title, fontsize=12, color="black", pad=4)
        ax.set_facecolor("black")
        ax.set_xlim(-self.bev_lateral_max, self.bev_lateral_max)
        ax.set_ylim(0.0, self.bev_forward_max)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("lateral position [m]", labelpad=2)
        ax.set_ylabel("forward distance [m]", labelpad=2)
        ax.grid(True, color="white", alpha=0.16, linewidth=0.8)
        ax.tick_params(colors="black", labelsize=9, pad=2)

        for spine in ax.spines.values():
            spine.set_color("black")

        ax.scatter(
            [0.0],
            [0.0],
            marker="^",
            s=105,
            c="white",
            edgecolors="black",
            linewidths=0.8,
            zorder=4,
        )

        if len(bev) and len(depth_values):
            depth_values = np.asarray(depth_values, dtype=np.float64)
            n = min(len(bev), len(depth_values))
            bev = bev[:n]
            depth_values = depth_values[:n]

            valid = np.isfinite(bev[:, 0]) & np.isfinite(bev[:, 1])
            valid &= np.isfinite(depth_values) & (depth_values > 0.0)
            valid &= bev[:, 1] >= 0.0
            valid &= bev[:, 1] <= self.bev_forward_max
            valid &= np.abs(bev[:, 0]) <= self.bev_lateral_max

            if np.any(valid):
                ax.scatter(
                    bev[valid, 0],
                    bev[valid, 1],
                    s=self.bev_point_size,
                    c=depth_values[valid],
                    cmap=self.depth_cmap,
                    vmin=self.distance_vmin,
                    vmax=self.distance_vmax,
                    alpha=0.98,
                    edgecolors="white",
                    linewidths=0.35,
                    zorder=3,
                )

        self._textbox(ax, textbox, dark=True)

    def _plot_pixel_heatmap(self, ax, heatmap_display, title, textbox, vmin, vmax):
        ax.imshow(
            heatmap_display,
            cmap=self.heatmap_cmap,
            vmin=vmin,
            vmax=vmax,
            aspect="auto",
            interpolation=self.heatmap_interpolation,
        )
        ax.set_title(title, fontsize=11, pad=3)
        ax.axis("off")
        self._textbox(ax, textbox, dark=True)

    def _plot_bev_heatmap(self, ax, bev_display, title, textbox, vmin, vmax):
        ax.imshow(
            bev_display,
            cmap=self.heatmap_cmap,
            vmin=vmin,
            vmax=vmax,
            origin="lower",
            extent=[-self.bev_lateral_max, self.bev_lateral_max, 0.0, self.bev_forward_max],
            aspect="equal",
            interpolation=self.bev_heatmap_interpolation,
        )
        ax.set_title(title, fontsize=12, color="black", pad=4)
        ax.set_facecolor("black")
        ax.set_xlabel("lateral position [m]", labelpad=2)
        ax.set_ylabel("forward distance [m]", labelpad=2)
        ax.grid(True, color="white", alpha=0.10, linewidth=0.8)
        ax.tick_params(colors="black", labelsize=9, pad=2)

        for spine in ax.spines.values():
            spine.set_color("black")

        ax.scatter(
            [0.0],
            [0.0],
            marker="^",
            s=105,
            c="white",
            edgecolors="black",
            linewidths=0.8,
            zorder=4,
        )

        self._textbox(ax, textbox, dark=True)

    def _add_depth_colorbar(self, fig, cax):
        norm = Normalize(vmin=self.distance_vmin, vmax=self.distance_vmax)
        sm = ScalarMappable(norm=norm, cmap=self.depth_cmap)
        sm.set_array([])

        cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
        cb.set_label("GT depth [m]", fontsize=10)
        cb.ax.tick_params(labelsize=9)
        cb.outline.set_linewidth(0.8)
        cax.set_title("shared depth color scale", fontsize=9, pad=2)
        return cb

    def _add_heatmap_colorbar(self, fig, cax, vmin, vmax):
        norm = Normalize(vmin=vmin, vmax=vmax)
        sm = ScalarMappable(norm=norm, cmap=self.heatmap_cmap)
        sm.set_array([])

        label = "predicted λ mass"
        if self.heatmap_gamma != 1.0:
            label = f"predicted λ mass^γ, γ={self.heatmap_gamma:g}"

        cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
        cb.set_label(label, fontsize=10)
        cb.ax.tick_params(labelsize=9)
        cb.outline.set_linewidth(0.8)
        cax.set_title("shared heatmap color scale", fontsize=9, pad=2)
        return cb

    def _output_path(self, runner, phase: str, video: str, frame_str: str, gidx_str: str) -> Path:
        root = Path(self.out_dir)
        if not root.is_absolute():
            root = Path(runner.work_dir) / root

        out = root / phase / video
        out.mkdir(parents=True, exist_ok=True)

        return out / f"frame_{frame_str}_idx_{gidx_str}.{self.image_ext}"

    def _visualize_sample(self, runner, step, phase, img_tensor, pred, sample, sample_idx):
        gt_all = sample.gt_sem_seg.data.to(pred.device)

        gt_mask_tensor = gt_all[0]
        gt_depth_tensor = gt_all[1] if gt_all.size(0) > 1 else None

        lam = torch.exp(pred[0]).detach().cpu().numpy()
        pred_depth = torch.exp(pred[1]).detach().cpu().numpy() if pred.size(0) > 1 else None

        gt_mask = gt_mask_tensor.detach().cpu().numpy()
        gt_depth = gt_depth_tensor.detach().cpu().numpy() if gt_depth_tensor is not None else None

        img = self._denorm_img(img_tensor.detach().cpu().numpy())
        img_h, img_w = img.shape[:2]
        lam_h, lam_w = lam.shape[-2:]
        gt_h, gt_w = gt_mask.shape[-2:]

        meta = sample.metainfo
        video = self._tagify(str(meta.get("video_name", "unknown_video")))
        frame = self._to_int_or_none(meta.get("video_frame_idx"))
        gidx = self._to_int_or_none(meta.get("global_idx", meta.get("frame_idx")))

        frame_str = f"{frame:06d}" if frame is not None else "unknown_frame"
        gidx_str = f"{gidx:08d}" if gidx is not None else "unknown_idx"

        ori_shape = meta.get("ori_shape")
        if isinstance(ori_shape, (tuple, list)) and len(ori_shape) >= 2:
            norm_h, norm_w = int(ori_shape[0]), int(ori_shape[1])
        else:
            norm_h, norm_w = lam_h, lam_w

        lam_mass = lam / float(max(norm_h * norm_w, 1))
        exp_n = float(lam_mass.sum())
        gt_n = float(gt_mask.sum())

        gt_pts_grid, gt_depths = extract_gt_points_and_depths(gt_mask, gt_depth)
        gt_pts_img = scale_points_to_shape(gt_pts_grid, (gt_h, gt_w), (img_h, img_w))

        gt_bev, gt_valid = backproject_pixels_to_bev(
            gt_pts_img,
            gt_depths,
            image_width=img_w,
            h_fov_deg=self.h_fov_deg,
            principal_x_rel=self.principal_x_rel,
            lateral_sign=self.lateral_sign,
            depth_is_forward=self.depth_is_forward,
        )
        gt_depths_bev = gt_depths[: len(gt_valid)][gt_valid] if len(gt_valid) else np.zeros((0,), dtype=np.float64)

        gt_visible = (
            int(
                np.sum(
                    (gt_bev[:, 1] >= 0.0)
                    & (gt_bev[:, 1] <= self.bev_forward_max)
                    & (np.abs(gt_bev[:, 0]) <= self.bev_lateral_max)
                )
            )
            if len(gt_bev)
            else 0
        )

        bev_h, bev_w = self.bev_grid_shape

        if pred_depth is not None:
            bev_heat = project_mass_heatmap_to_bev(
                lam_mass,
                pred_depth,
                bev_h=bev_h,
                bev_w=bev_w,
                bev_forward_max=self.bev_forward_max,
                bev_lateral_max=self.bev_lateral_max,
                h_fov_deg=self.h_fov_deg,
                principal_x_rel=self.principal_x_rel,
                lateral_sign=self.lateral_sign,
                depth_is_forward=self.depth_is_forward,
                subpixel_factor=self.bev_projection_subpixel_factor,
                chunk_rows=self.bev_projection_chunk_rows,
                min_mass=self.bev_projection_min_mass,
                smoothing_sigma=self.bev_heatmap_smoothing_sigma,
            )
            pixel_heat_display = self._display_heatmap(lam_mass)
            bev_heat_display = self._display_heatmap(bev_heat)
            h_vmin, h_vmax = self._heatmap_scale(pixel_heat_display, bev_heat_display)
            bev_mass_visible = float(bev_heat.sum())
        else:
            bev_heat = None
            pixel_heat_display = self._display_heatmap(lam_mass)
            bev_heat_display = None
            h_vmin, h_vmax = self._heatmap_scale(pixel_heat_display)
            bev_mass_visible = 0.0

        fig = Figure(figsize=self.figsize, dpi=self.dpi)
        FigureCanvasAgg(fig)
        gs = fig.add_gridspec(
            nrows=3,
            ncols=2,
            height_ratios=[1.0, self.bev_row_height_ratio, self.colorbar_height_ratio],
            left=0.025,
            right=0.992,
            top=0.955,
            bottom=0.035,
            wspace=0.035,
            hspace=0.075,
        )
        ax_cam_gt = fig.add_subplot(gs[0, 0])
        ax_pix_heat = fig.add_subplot(gs[0, 1])
        ax_bev_gt = fig.add_subplot(gs[1, 0])
        ax_bev_heat = fig.add_subplot(gs[1, 1])
        ax_cbar_left = fig.add_subplot(gs[2, 0])
        ax_cbar_right = fig.add_subplot(gs[2, 1])

        fig.suptitle(
            f"{phase} | {video} | frame {frame_str} | idx {gidx_str} | "
            f"GT points and area-splatted predicted heatmaps",
            fontsize=13,
            y=0.988,
        )

        self._overlay_depth_points(
            ax_cam_gt,
            img=img,
            pts=gt_pts_img,
            depth_values=gt_depths,
            title="Camera: GT radar points",
            textbox=(f"GT N={len(gt_pts_img)}\ncolor: GT depth"),
        )

        self._plot_bev_points(
            ax_bev_gt,
            bev=gt_bev,
            depth_values=gt_depths_bev,
            title="BEV: GT back-projection",
            textbox=(
                f"visible={gt_visible}/{len(gt_bev)}\n"
                f"x lateral: ±{self.bev_lateral_max:g} m\n"
                f"y forward: 0..{self.bev_forward_max:g} m"
            ),
        )

        self._plot_pixel_heatmap(
            ax_pix_heat,
            heatmap_display=pixel_heat_display,
            title="Pixel space: predicted λ heatmap",
            textbox=(f"E[N]={exp_n:.2f}\nGT count={gt_n:.0f}\ndisplay γ={self.heatmap_gamma:g}"),
            vmin=h_vmin,
            vmax=h_vmax,
        )

        if bev_heat_display is not None:
            self._plot_bev_heatmap(
                ax_bev_heat,
                bev_display=bev_heat_display,
                title="BEV: area-splatted predicted λ heatmap",
                textbox=(
                    f"mass in BEV window={bev_mass_visible:.2f}\n"
                    f"subpixel={self.bev_projection_subpixel_factor}x{self.bev_projection_subpixel_factor}, "
                    f"smooth={self.bev_heatmap_smoothing_sigma:g}\n"
                    f"x lateral: ±{self.bev_lateral_max:g} m\n"
                    f"y forward: 0..{self.bev_forward_max:g} m"
                ),
                vmin=h_vmin,
                vmax=h_vmax,
            )
        else:
            ax_bev_heat.set_facecolor("black")
            ax_bev_heat.set_title("BEV: area-splatted predicted λ heatmap", fontsize=12, color="black", pad=4)
            ax_bev_heat.set_xlabel("lateral position [m]", labelpad=2)
            ax_bev_heat.set_ylabel("forward distance [m]", labelpad=2)
            ax_bev_heat.set_xlim(-self.bev_lateral_max, self.bev_lateral_max)
            ax_bev_heat.set_ylim(0.0, self.bev_forward_max)
            ax_bev_heat.set_aspect("equal", adjustable="box")
            ax_bev_heat.grid(True, color="white", alpha=0.10, linewidth=0.8)
            self._textbox(ax_bev_heat, "missing predicted depth channel", dark=True)

        self._add_depth_colorbar(fig, ax_cbar_left)
        self._add_heatmap_colorbar(fig, ax_cbar_right, h_vmin, h_vmax)

        out_path = self._output_path(
            runner=runner,
            phase=phase,
            video=video,
            frame_str=frame_str,
            gidx_str=gidx_str,
        )

        fig.savefig(out_path, dpi=self.dpi)
        fig.clear()
