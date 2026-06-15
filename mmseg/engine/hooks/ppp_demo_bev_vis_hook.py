"""
PPP demo BEV visualization hook.

Produces one PNG per visualized sample with four panels:

  1. Camera image with GT radar points, colored by GT distance/depth
  2. Camera image with sampled predicted PPP points, colored by predicted distance/depth
  3. GT points back-projected into BEV
  4. Predicted sampled points back-projected into BEV

The BEV conversion is intentionally consistent for GT and prediction. It does
not use the SYS100 radar-grid reconstruction. It uses only horizontal image
position and distance/depth:

    lateral = depth * tan(horizontal_angle)
    forward = depth

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
    """
    Synthetic neon colormap for distance/depth.

    The colors are intentionally artificial and high contrast so that points
    remain visible on top of natural camera images and black BEV panels.
    """
    return LinearSegmentedColormap.from_list(
        name,
        [
            "#ff00ff",  # neon magenta
            "#7a00ff",  # violet
            "#003cff",  # electric blue
            "#00e5ff",  # cyan
            "#00ff66",  # neon green
            "#ffff00",  # yellow
            "#ff6a00",  # orange
            "#ff0000",  # red
        ],
        N=256,
    )


def extract_gt_points_depth_velocity(
    mask: np.ndarray,
    depth: np.ndarray | None,
    velocity: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Return GT point coordinates and marks from dense GT maps."""
    ys, xs = np.where(mask > 0)

    if xs.size == 0:
        return (
            np.zeros((0, 2), dtype=np.float64),
            np.zeros((0,), dtype=np.float64),
            None if velocity is None else np.zeros((0,), dtype=np.float64),
        )

    pts = np.stack([xs + 0.5, ys + 0.5], axis=1).astype(np.float64)

    if depth is None:
        depths = np.zeros((len(pts),), dtype=np.float64)
    else:
        depths = depth[ys, xs].astype(np.float64)

    velocities = None
    if velocity is not None:
        velocities = velocity[ys, xs].astype(np.float64)

    return pts, depths, velocities


def sample_grid_values_at_points(grid: np.ndarray | None, pts: np.ndarray) -> np.ndarray:
    """
    Sample grid values at floating point coordinates.

    Uses nearest lower pixel index, clipped to the valid image region.
    Points are expected as (x, y).
    """
    if grid is None or len(pts) == 0:
        return np.zeros((0,), dtype=np.float64)

    h, w = grid.shape[-2:]
    xs = np.clip(pts[:, 0].astype(np.int64), 0, w - 1)
    ys = np.clip(pts[:, 1].astype(np.int64), 0, h - 1)

    return grid[ys, xs].astype(np.float64)


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


def sample_ppp_thinning(lam: np.ndarray, rng=None):
    """Sample a point cloud from a discrete PPP intensity by thinning."""
    rng = np.random.default_rng() if rng is None else rng

    lam = np.asarray(lam, dtype=np.float64)
    if lam.ndim != 2:
        raise ValueError("lambda grid must be 2D")
    if not np.all(np.isfinite(lam)):
        raise ValueError("lambda grid contains NaN/Inf")
    if np.any(lam < 0):
        raise ValueError("lambda grid contains negative values")

    h, w = lam.shape
    lam_max = float(lam.max())

    if lam_max <= 0:
        return np.zeros((0, 2), dtype=np.float64), {
            "num_points": 0,
            "num_proposals": 0,
        }

    n_prop = int(rng.poisson(lam_max * h * w))
    if n_prop == 0:
        return np.zeros((0, 2), dtype=np.float64), {
            "num_points": 0,
            "num_proposals": 0,
        }

    xs = rng.random(n_prop) * w
    ys = rng.random(n_prop) * h
    xi = np.clip(xs.astype(int), 0, w - 1)
    yi = np.clip(ys.astype(int), 0, h - 1)

    keep = rng.random(n_prop) < (lam[yi, xi] / lam_max)
    pts = np.stack([xs[keep], ys[keep]], axis=1).astype(np.float64)

    return pts, {
        "num_points": int(len(pts)),
        "num_proposals": int(n_prop),
    }


def sample_ppp_block(lam: np.ndarray, block_size=8, placement="weighted", rng=None):
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

    h, w = lam.shape
    bs = int(block_size)

    nby = (h + bs - 1) // bs
    nbx = (w + bs - 1) // bs
    hp, wp = nby * bs, nbx * bs

    lam_pad = np.zeros((hp, wp), dtype=np.float64)
    lam_pad[:h, :w] = lam

    lam_blocks = lam_pad.reshape(nby, bs, nbx, bs).transpose(0, 2, 1, 3)
    block_mass = lam_blocks.sum(axis=(2, 3))
    p = 1.0 - np.exp(-block_mass)

    active = rng.random((nby, nbx)) < p
    by, bx = np.nonzero(active)
    n = len(by)

    if n == 0:
        return np.zeros((0, 2), dtype=np.float64), {
            "num_points": 0,
            "num_active_blocks": 0,
        }

    pts = np.empty((n, 2), dtype=np.float64)
    x0s = bx * bs
    y0s = by * bs
    wbs = np.minimum(bs, w - x0s)
    hbs = np.minimum(bs, h - y0s)

    if placement == "uniform":
        pts[:, 0] = x0s + rng.random(n) * wbs
        pts[:, 1] = y0s + rng.random(n) * hbs
    else:
        for i, (x0, y0, wb, hb) in enumerate(zip(x0s, y0s, wbs, hbs)):
            block = lam[y0:y0 + hb, x0:x0 + wb]
            s = float(block.sum())

            if s <= 0:
                pts[i, 0] = x0 + rng.random() * wb
                pts[i, 1] = y0 + rng.random() * hb
                continue

            idx = int(rng.choice(block.size, p=block.ravel() / s))
            py, px = divmod(idx, int(wb))
            pts[i, 0] = x0 + px + rng.random()
            pts[i, 1] = y0 + py + rng.random()

    return pts, {
        "num_points": int(n),
        "num_active_blocks": int(n),
    }


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

    If depth_is_forward=True, depth_m is interpreted as vehicle-forward distance.
    If depth_is_forward=False, depth_m is interpreted as radial ray distance.
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


@HOOKS.register_module()
class PPPDemoBEVVisHook(Hook):
    """Write PPP camera-plus-BEV test visualizations as PNG files."""

    def __init__(
        self,
        train_interval=0,
        max_samples_train=0,
        max_samples_val=0,
        max_samples_test=10,
        out_dir="demo_bev_frames",
        img_mean=None,
        img_std=None,
        dpi=100,
        figsize=(22, 16),
        sampling_method="thinning",
        sim_block_size=8,
        sim_block_placement="weighted",
        point_size=28,
        bev_point_size=None,
        image_ext="png",
        write_all_ranks=False,
        h_fov_deg=110.0,
        principal_x_rel=0.5,
        lateral_sign=1.0,
        depth_is_forward=True,
        bev_forward_max=104.0,
        bev_lateral_max=50.0,
        distance_vmin=0.0,
        distance_vmax=None,
        depth_cmap="synthetic",
        depth_point_edge_color="black",
        depth_point_edge_width=0.45,
        show_velocity=False,
        velocity_as_color=False,
        velocity_vmin=-20.0,
        velocity_vmax=20.0,
        velocity_cmap="coolwarm",
        bev_row_height_ratio=2.5,
        colorbar_height_ratio=0.075,
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
        self.bev_row_height_ratio = float(bev_row_height_ratio)
        self.colorbar_height_ratio = float(colorbar_height_ratio)

        self.sampling_method = str(sampling_method).lower()
        if self.sampling_method not in {"thinning", "block"}:
            raise ValueError("sampling_method must be 'thinning' or 'block'")

        self.sim_block_size = int(sim_block_size)
        self.sim_block_placement = str(sim_block_placement)
        self.point_size = float(point_size)
        self.bev_point_size = float(bev_point_size) if bev_point_size is not None else float(point_size)
        self.write_all_ranks = bool(write_all_ranks)

        self.h_fov_deg = float(h_fov_deg)
        self.principal_x_rel = float(principal_x_rel)
        self.lateral_sign = float(lateral_sign)
        self.depth_is_forward = bool(depth_is_forward)
        self.bev_forward_max = float(bev_forward_max)
        self.bev_lateral_max = float(bev_lateral_max)

        self.distance_vmin = float(distance_vmin) if distance_vmin is not None else None
        if distance_vmax is None:
            self.distance_vmax = self.bev_forward_max
        else:
            self.distance_vmax = float(distance_vmax)

        self.depth_point_edge_color = str(depth_point_edge_color)
        self.depth_point_edge_width = float(depth_point_edge_width)

        if depth_cmap == "synthetic":
            self.depth_cmap = make_synthetic_depth_cmap()
        else:
            self.depth_cmap = depth_cmap

        self.show_velocity = bool(show_velocity)
        self.velocity_as_color = bool(velocity_as_color)
        self.velocity_vmin = float(velocity_vmin)
        self.velocity_vmax = float(velocity_vmax)
        self.velocity_cmap = velocity_cmap

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
                iter_seed=getattr(runner, "iter", 0),
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

    def _sample_prediction(self, lam_mass: np.ndarray, seed: int):
        if self.sampling_method == "thinning":
            return sample_ppp_thinning(lam_mass, rng=np.random.default_rng(seed + 11))

        return sample_ppp_block(
            lam_mass,
            block_size=self.sim_block_size,
            placement=self.sim_block_placement,
            rng=np.random.default_rng(seed + 29),
        )

    def _color_setup(self, use_velocity_color: bool):
        """Return the shared color mapping used by all four panels."""
        if use_velocity_color:
            return (
                self.velocity_cmap,
                self.velocity_vmin,
                self.velocity_vmax,
                "velocity [model units]",
                "velocity",
            )

        return (
            self.depth_cmap,
            self.distance_vmin,
            self.distance_vmax,
            "distance / forward depth [m]",
            "distance",
        )

    def _point_color_values(
        self,
        distances: np.ndarray,
        velocities: np.ndarray | None,
        use_velocity_color: bool,
    ) -> np.ndarray:
        if use_velocity_color and velocities is not None:
            return velocities.astype(np.float64)
        return distances.astype(np.float64)

    def _add_shared_colorbar(self, fig, cax, use_velocity_color: bool):
        """Add one shared colorbar spanning the full figure width."""
        cmap, vmin, vmax, label, color_mode = self._color_setup(use_velocity_color)
        norm = Normalize(vmin=vmin, vmax=vmax)
        sm = ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])

        cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
        cb.set_label(f"shared color scale: {label}", fontsize=10)
        cb.ax.tick_params(labelsize=9)
        cb.outline.set_linewidth(0.8)
        cax.set_title(f"All dots use the same {color_mode} color scale", fontsize=9, pad=2)
        return cb

    def _overlay_camera_points(
        self,
        ax,
        img,
        pts_img,
        distances,
        velocities,
        title,
        textbox,
        use_velocity_color: bool,
    ):
        ax.imshow(img, aspect="auto")
        ax.set_title(title, fontsize=11, pad=3)
        ax.axis("off")

        color_values = self._point_color_values(
            distances,
            velocities,
            use_velocity_color,
        )
        cmap, vmin, vmax, _, _ = self._color_setup(use_velocity_color)

        if len(pts_img) and len(distances):
            n = min(len(pts_img), len(distances), len(color_values))
            pts_plot = pts_img[:n]
            distances_plot = distances[:n]
            color_values_plot = color_values[:n]

            valid = np.isfinite(pts_plot[:, 0]) & np.isfinite(pts_plot[:, 1])
            valid &= np.isfinite(distances_plot) & (distances_plot > 0.0)
            valid &= np.isfinite(color_values_plot)

            if np.any(valid):
                ax.scatter(
                    pts_plot[valid, 0],
                    pts_plot[valid, 1],
                    s=self.point_size,
                    c=color_values_plot[valid],
                    cmap=cmap,
                    vmin=vmin,
                    vmax=vmax,
                    alpha=0.98,
                    edgecolors=self.depth_point_edge_color,
                    linewidths=self.depth_point_edge_width,
                )

        self._textbox(ax, textbox, dark=True)

    def _plot_bev(
        self,
        ax,
        bev,
        distances,
        velocities,
        title,
        textbox,
        use_velocity_color: bool,
    ):
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

        color_values = self._point_color_values(
            distances,
            velocities,
            use_velocity_color,
        )
        cmap, vmin, vmax, _, _ = self._color_setup(use_velocity_color)

        if len(bev) and len(distances):
            n = min(len(bev), len(distances), len(color_values))
            bev_plot = bev[:n]
            distances_plot = distances[:n]
            color_values_plot = color_values[:n]

            valid = np.isfinite(bev_plot[:, 0]) & np.isfinite(bev_plot[:, 1])
            valid &= np.isfinite(distances_plot) & (distances_plot > 0.0)
            valid &= np.isfinite(color_values_plot)
            valid &= bev_plot[:, 1] >= 0.0
            valid &= bev_plot[:, 1] <= self.bev_forward_max
            valid &= np.abs(bev_plot[:, 0]) <= self.bev_lateral_max

            if np.any(valid):
                ax.scatter(
                    bev_plot[valid, 0],
                    bev_plot[valid, 1],
                    s=self.bev_point_size,
                    c=color_values_plot[valid],
                    cmap=cmap,
                    vmin=vmin,
                    vmax=vmax,
                    alpha=0.98,
                    edgecolors="white",
                    linewidths=0.35,
                    zorder=3,
                )

        self._textbox(ax, textbox, dark=True)

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

        out = root / phase / video
        out.mkdir(parents=True, exist_ok=True)
        return out / f"frame_{frame_str}_idx_{gidx_str}.{self.image_ext}"

    def _velocity_stats_text(self, velocities: np.ndarray | None) -> str:
        if not self.show_velocity or velocities is None or len(velocities) == 0:
            return ""
        finite = velocities[np.isfinite(velocities)]
        if finite.size == 0:
            return ""
        return f"\nvel mean={finite.mean():.2f}, min={finite.min():.2f}, max={finite.max():.2f}"

    def _visualize_sample(
        self,
        runner,
        step,
        phase,
        img_tensor,
        pred,
        sample,
        iter_seed,
        sample_idx,
    ):
        gt_all = sample.gt_sem_seg.data.to(pred.device)

        gt_mask_tensor = gt_all[0]
        gt_depth_tensor = gt_all[1] if gt_all.size(0) > 1 else None
        gt_velocity_tensor = gt_all[2] if gt_all.size(0) > 2 else None

        lam = torch.exp(pred[0]).detach().cpu().numpy()
        pred_depth = torch.exp(pred[1]).detach().cpu().numpy() if pred.size(0) > 1 else None
        pred_velocity = pred[2].detach().cpu().numpy() if pred.size(0) > 2 else None

        gt_mask = gt_mask_tensor.detach().cpu().numpy()
        gt_depth = (
            gt_depth_tensor.detach().cpu().numpy()
            if gt_depth_tensor is not None
            else None
        )
        gt_velocity = (
            gt_velocity_tensor.detach().cpu().numpy()
            if gt_velocity_tensor is not None and self.show_velocity
            else None
        )

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

        gt_pts_grid, gt_depths, gt_velocities = extract_gt_points_depth_velocity(
            gt_mask,
            gt_depth,
            gt_velocity,
        )
        gt_pts_img = scale_points_to_shape(gt_pts_grid, (gt_h, gt_w), (img_h, img_w))

        seed_base = frame if frame is not None else (gidx if gidx is not None else sample_idx)
        seed0 = int(seed_base) + 1000 * int(iter_seed)

        pred_pts_grid, pred_stats = self._sample_prediction(lam_mass, seed0)
        pred_pts_img = scale_points_to_shape(pred_pts_grid, (lam_h, lam_w), (img_h, img_w))

        if pred_depth is not None:
            pred_depths = sample_grid_values_at_points(pred_depth, pred_pts_grid)
        else:
            pred_depths = np.zeros((len(pred_pts_grid),), dtype=np.float64)

        pred_velocities = None
        if self.show_velocity and pred_velocity is not None:
            pred_velocities = sample_grid_values_at_points(pred_velocity, pred_pts_grid)

        gt_bev, gt_valid = backproject_pixels_to_bev(
            gt_pts_img,
            gt_depths,
            image_width=img_w,
            h_fov_deg=self.h_fov_deg,
            principal_x_rel=self.principal_x_rel,
            lateral_sign=self.lateral_sign,
            depth_is_forward=self.depth_is_forward,
        )
        pred_bev, pred_valid = backproject_pixels_to_bev(
            pred_pts_img,
            pred_depths,
            image_width=img_w,
            h_fov_deg=self.h_fov_deg,
            principal_x_rel=self.principal_x_rel,
            lateral_sign=self.lateral_sign,
            depth_is_forward=self.depth_is_forward,
        )

        gt_depths_bev = gt_depths[: len(gt_valid)][gt_valid] if len(gt_valid) else np.zeros((0,), dtype=np.float64)
        pred_depths_bev = pred_depths[: len(pred_valid)][pred_valid] if len(pred_valid) else np.zeros((0,), dtype=np.float64)

        gt_velocities_bev = None
        if gt_velocities is not None and len(gt_valid):
            gt_velocities_bev = gt_velocities[: len(gt_valid)][gt_valid]

        pred_velocities_bev = None
        if pred_velocities is not None and len(pred_valid):
            pred_velocities_bev = pred_velocities[: len(pred_valid)][pred_valid]

        gt_visible = int(
            np.sum(
                (gt_bev[:, 1] >= 0.0)
                & (gt_bev[:, 1] <= self.bev_forward_max)
                & (np.abs(gt_bev[:, 0]) <= self.bev_lateral_max)
            )
        ) if len(gt_bev) else 0
        pred_visible = int(
            np.sum(
                (pred_bev[:, 1] >= 0.0)
                & (pred_bev[:, 1] <= self.bev_forward_max)
                & (np.abs(pred_bev[:, 0]) <= self.bev_lateral_max)
            )
        ) if len(pred_bev) else 0

        use_velocity_color = (
            self.show_velocity
            and self.velocity_as_color
            and gt_velocities is not None
            and pred_velocities is not None
        )

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
        ax_cam_pred = fig.add_subplot(gs[0, 1])
        ax_bev_gt = fig.add_subplot(gs[1, 0])
        ax_bev_pred = fig.add_subplot(gs[1, 1])
        ax_cbar = fig.add_subplot(gs[2, :])

        fig.suptitle(
            f"{phase} | {video} | frame {frame_str} | idx {gidx_str} | "
            f"BEV from image x + depth | hFOV={self.h_fov_deg:g}°",
            fontsize=13,
            y=0.988,
        )

        gt_text = (
            f"GT N={len(gt_pts_img)}\n"
            f"BEV valid={len(gt_bev)}, visible={gt_visible}\n"
            f"distance color, y image ignored"
            f"{self._velocity_stats_text(gt_velocities)}"
        )
        pred_text = (
            f"sample={self.sampling_method}\n"
            f"N={pred_stats.get('num_points', len(pred_pts_img))}, E[N]={exp_n:.2f}\n"
            f"BEV valid={len(pred_bev)}, visible={pred_visible}"
            f"{self._velocity_stats_text(pred_velocities)}"
        )

        self._overlay_camera_points(
            ax_cam_gt,
            img,
            gt_pts_img,
            gt_depths,
            gt_velocities,
            title="Camera: GT radar points",
            textbox=gt_text,
            use_velocity_color=use_velocity_color,
        )
        self._overlay_camera_points(
            ax_cam_pred,
            img,
            pred_pts_img,
            pred_depths,
            pred_velocities,
            title=f"Camera: predicted PPP sample ({self.sampling_method})",
            textbox=pred_text,
            use_velocity_color=use_velocity_color,
        )
        self._plot_bev(
            ax_bev_gt,
            gt_bev,
            gt_depths_bev,
            gt_velocities_bev,
            title="BEV: GT back-projection",
            textbox=(
                f"x lateral: ±{self.bev_lateral_max:g} m\n"
                f"y forward: 0..{self.bev_forward_max:g} m\n"
                f"visible={gt_visible}/{len(gt_bev)}"
                f"{self._velocity_stats_text(gt_velocities_bev)}"
            ),
            use_velocity_color=use_velocity_color,
        )
        self._plot_bev(
            ax_bev_pred,
            pred_bev,
            pred_depths_bev,
            pred_velocities_bev,
            title="BEV: predicted sample back-projection",
            textbox=(
                f"sample={self.sampling_method}\n"
                f"E[N]={exp_n:.2f}\n"
                f"visible={pred_visible}/{len(pred_bev)}"
                f"{self._velocity_stats_text(pred_velocities_bev)}"
            ),
            use_velocity_color=use_velocity_color,
        )

        self._add_shared_colorbar(fig, ax_cbar, use_velocity_color=use_velocity_color)

        out_path = self._output_path(
            runner=runner,
            phase=phase,
            video=video,
            frame_str=frame_str,
            gidx_str=gidx_str,
        )
        fig.savefig(out_path, dpi=self.dpi, bbox_inches="tight")
        fig.clear()