"""
PPP demo visualization hook with predicted depth and velocity panels.

Reduced visualization hook for producing demo frames as image files.

This variant adds panels for:
  1. GT points projected on the camera image, colored by GT depth
  2. PPP sample by thinning projected on the camera image, colored by predicted depth
  3. PPP sample by block approximation projected on the camera image, colored by predicted depth
  4. Predicted PPP intensity mass heatmap
  5. Gamma-adjusted predicted PPP intensity mass heatmap
  6. Predicted depth heatmap with GT depths overlaid as dots
  7. Predicted velocity heatmap with GT velocities overlaid as dots

For two-depth heads, depth uses channel 1, i.e. exp(raw_d0).
For two-depth heads, velocity uses channel 3.
For normal heads, velocity uses channel 2.

It writes PNG files directly to:

    <runner.work_dir>/<out_dir>/<phase>/<video_name>/frame_<frame>_idx_<idx>.png

For distributed test runs, set write_all_ranks=True.
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
from matplotlib.figure import Figure
from matplotlib.colors import LinearSegmentedColormap


def make_synthetic_depth_cmap(name: str = "synthetic_depth_clash"):
    """
    Synthetic neon colormap for depth.

    The colors are intentionally artificial and high contrast so that points
    remain visible on top of natural camera images.
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


def extract_gt_points(mask: np.ndarray) -> np.ndarray:
    """Return point coordinates at positive GT mask cells as (x, y)."""
    ys, xs = np.where(mask > 0)
    if xs.size == 0:
        return np.zeros((0, 2), dtype=np.float64)
    return np.stack([xs + 0.5, ys + 0.5], axis=1).astype(np.float64)


def extract_gt_points_and_values(
    mask: np.ndarray,
    values: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return GT point coordinates and values.

    Coordinates are returned as (x, y).
    Values are taken from the same pixel positions.
    """
    ys, xs = np.where(mask > 0)

    if xs.size == 0:
        return (
            np.zeros((0, 2), dtype=np.float64),
            np.zeros((0,), dtype=np.float64),
        )

    pts = np.stack([xs + 0.5, ys + 0.5], axis=1).astype(np.float64)

    if values is None:
        point_values = np.zeros((len(pts),), dtype=np.float64)
    else:
        point_values = values[ys, xs].astype(np.float64)

    return pts, point_values


def sample_grid_values_at_points(grid: np.ndarray, pts: np.ndarray) -> np.ndarray:
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


@HOOKS.register_module()
class PPPDemoWDepthWVelVisHook(Hook):
    """Write reduced PPP demo visualizations with predicted depth and velocity."""

    def __init__(
        self,
        train_interval=0,
        max_samples_train=0,
        max_samples_val=0,
        max_samples_test=10,
        out_dir="demo_frames",
        img_mean=None,
        img_std=None,
        dpi=90,
        figsize=(14, 28),
        gamma=0.9,
        sim_block_size=8,
        sim_block_placement="weighted",
        point_size=28,
        image_ext="png",
        write_all_ranks=False,
        depth_vmin=None,
        depth_vmax=None,
        depth_cmap="synthetic",
        depth_point_edge_color="black",
        depth_point_edge_width=0.45,
        vel_vmin=None,
        vel_vmax=None,
        vel_cmap="coolwarm",
        vel_point_edge_color="black",
        vel_point_edge_width=0.45,
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
        self.sim_block_size = int(sim_block_size)
        self.sim_block_placement = str(sim_block_placement)
        self.point_size = float(point_size)
        self.write_all_ranks = bool(write_all_ranks)

        self.depth_vmin = depth_vmin
        self.depth_vmax = depth_vmax
        self.depth_point_edge_color = str(depth_point_edge_color)
        self.depth_point_edge_width = float(depth_point_edge_width)

        self.vel_vmin = vel_vmin
        self.vel_vmax = vel_vmax
        self.vel_cmap = str(vel_cmap)
        self.vel_point_edge_color = str(vel_point_edge_color)
        self.vel_point_edge_width = float(vel_point_edge_width)

        if depth_cmap == "synthetic":
            self.depth_cmap = make_synthetic_depth_cmap()
        else:
            self.depth_cmap = depth_cmap

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
                f"velocity_mode must be 'signed' or 'absolute', got {velocity_mode!r}."
            )

        proc = model.data_preprocessor(data_batch)
        inputs = proc["inputs"]
        data_samples = proc["data_samples"]

        with torch.no_grad():
            preds = model.decode_head(model.extract_feat(inputs))

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
                iter_seed=runner.iter,
                sample_idx=start_idx + i,
                velocity_mode=velocity_mode,
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
    def _label(ax, text):
        ax.text(
            -0.015,
            0.5,
            text,
            transform=ax.transAxes,
            rotation=90,
            fontsize=13,
            va="center",
            ha="center",
        )

    @staticmethod
    def _textbox(ax, text):
        ax.text(
            0.02,
            0.98,
            text,
            transform=ax.transAxes,
            fontsize=11,
            color="white",
            va="top",
            ha="left",
            bbox=dict(facecolor="black", alpha=0.55, pad=3),
        )

    def _imshow(
        self,
        ax,
        arr,
        ylabel,
        cmap=None,
        vmin=None,
        vmax=None,
        fig=None,
        cbar=False,
    ):
        im = ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
        ax.axis("off")
        self._label(ax, ylabel)

        if cbar and fig is not None:
            cax = ax.inset_axes([0.0, -0.06, 1.0, 0.04])
            cb = fig.colorbar(im, cax=cax, orientation="horizontal")
            cb.ax.tick_params(labelsize=10)

        return im

    def _overlay_depth_points(
        self,
        ax,
        img,
        pts,
        depth_values,
        ylabel,
        n,
        textbox_extra=None,
    ):
        ax.imshow(img, aspect="auto")

        if len(pts) and len(depth_values):
            values = np.asarray(depth_values, dtype=np.float64)
            finite = np.isfinite(values)

            if np.any(finite):
                pts_plot = pts[finite]
                values_plot = values[finite]

                ax.scatter(
                    pts_plot[:, 0],
                    pts_plot[:, 1],
                    s=self.point_size,
                    c=values_plot,
                    cmap=self.depth_cmap,
                    vmin=self.depth_vmin,
                    vmax=self.depth_vmax,
                    alpha=0.98,
                    edgecolors=self.depth_point_edge_color,
                    linewidths=self.depth_point_edge_width,
                )

        ax.axis("off")
        self._label(ax, ylabel)

        text = f"N={n}"
        if textbox_extra:
            text += f"\n{textbox_extra}"
        self._textbox(ax, text)

    def _overlay_gt_depth_dots(
        self,
        ax,
        gt_pts: np.ndarray,
        gt_depths: np.ndarray,
    ):
        if len(gt_pts) == 0 or len(gt_depths) == 0:
            return

        values = np.asarray(gt_depths, dtype=np.float64)
        finite = np.isfinite(values)

        if not np.any(finite):
            return

        pts = gt_pts[finite]
        values = values[finite]

        ax.scatter(
            pts[:, 0],
            pts[:, 1],
            s=self.point_size,
            c=values,
            cmap=self.depth_cmap,
            vmin=self.depth_vmin,
            vmax=self.depth_vmax,
            alpha=0.98,
            edgecolors=self.depth_point_edge_color,
            linewidths=self.depth_point_edge_width,
        )

    def _overlay_gt_velocity_dots(
        self,
        ax,
        gt_pts: np.ndarray,
        gt_vels: np.ndarray,
    ):
        if len(gt_pts) == 0 or len(gt_vels) == 0:
            return

        values = np.asarray(gt_vels, dtype=np.float64)
        finite = np.isfinite(values)

        if not np.any(finite):
            return

        pts = gt_pts[finite]
        values = values[finite]

        ax.scatter(
            pts[:, 0],
            pts[:, 1],
            s=self.point_size,
            c=values,
            cmap=self.vel_cmap,
            vmin=self.vel_vmin,
            vmax=self.vel_vmax,
            alpha=0.98,
            edgecolors=self.vel_point_edge_color,
            linewidths=self.vel_point_edge_width,
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

        out = root / phase / video
        out.mkdir(parents=True, exist_ok=True)

        return out / f"frame_{frame_str}_idx_{gidx_str}.{self.image_ext}"

    def _decode_velocity(
        self,
        pred: torch.Tensor,
        velocity_mode: str,
    ) -> tuple[np.ndarray | None, str]:
        raw_vel = None

        if pred.size(0) == 4:
            raw_vel = pred[3]
        elif pred.size(0) > 2:
            raw_vel = pred[2]

        if raw_vel is None:
            return None, "Pred vel [m/s]"

        if velocity_mode == "signed":
            vel = (raw_vel ** 3).detach().cpu().numpy()
            return vel, "Pred vel [m/s]"

        if velocity_mode == "absolute":
            vel = torch.abs(raw_vel).detach().cpu().numpy()
            return vel, "Pred |vel| [m/s]"

        raise RuntimeError(f"Unsupported velocity_mode={velocity_mode!r}")

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
        velocity_mode,
    ):
        gt_all = sample.gt_sem_seg.data.to(pred.device)

        gt_mask_tensor = gt_all[0]
        gt_depth_tensor = gt_all[1] if gt_all.size(0) > 1 else None
        gt_vel_tensor = gt_all[2] if gt_all.size(0) > 2 else None

        lam = torch.exp(pred[0]).detach().cpu().numpy()
        pred_depth = torch.exp(pred[1]).detach().cpu().numpy() if pred.size(0) > 1 else None
        pred_vel, pred_vel_label = self._decode_velocity(pred, velocity_mode)

        gt_mask = gt_mask_tensor.detach().cpu().numpy()
        gt_depth = (
            gt_depth_tensor.detach().cpu().numpy()
            if gt_depth_tensor is not None
            else None
        )
        gt_vel = (
            gt_vel_tensor.detach().cpu().numpy()
            if gt_vel_tensor is not None
            else None
        )

        img = self._denorm_img(img_tensor.detach().cpu().numpy())

        meta = sample.metainfo
        video = self._tagify(str(meta.get("video_name", "unknown_video")))
        frame = meta.get("video_frame_idx")
        gidx = meta.get("global_idx", meta.get("frame_idx"))

        frame_str = f"{int(frame):06d}" if frame is not None else "unknown_frame"
        gidx_str = f"{int(gidx):08d}" if gidx is not None else "unknown_idx"

        ori_shape = meta.get("ori_shape")
        if isinstance(ori_shape, (tuple, list)) and len(ori_shape) >= 2:
            h, w = int(ori_shape[0]), int(ori_shape[1])
        else:
            h, w = lam.shape[-2:]

        lam_mass = lam / float(h * w)
        exp_n = float(lam_mass.sum())
        gt_n = float(gt_mask.sum())

        gt_pts, gt_depths = extract_gt_points_and_values(gt_mask, gt_depth)
        _, gt_vels = extract_gt_points_and_values(gt_mask, gt_vel)

        if velocity_mode == "absolute":
            gt_vels = np.abs(gt_vels)

        seed_base = int(frame) if frame is not None else int(meta.get("frame_idx", 0))
        seed0 = seed_base + 1000 * int(iter_seed)

        sim1_pts, sim1_stats = sample_ppp_thinning(
            lam_mass,
            rng=np.random.default_rng(seed0 + 11),
        )
        sim2_pts, sim2_stats = sample_ppp_block(
            lam_mass,
            block_size=self.sim_block_size,
            placement=self.sim_block_placement,
            rng=np.random.default_rng(seed0 + 29),
        )

        if pred_depth is not None:
            sim1_depths = sample_grid_values_at_points(pred_depth, sim1_pts)
            sim2_depths = sample_grid_values_at_points(pred_depth, sim2_pts)
        else:
            sim1_depths = np.zeros((len(sim1_pts),), dtype=np.float64)
            sim2_depths = np.zeros((len(sim2_pts),), dtype=np.float64)

        fig = Figure(figsize=self.figsize, dpi=self.dpi)
        FigureCanvasAgg(fig)
        axs = fig.subplots(7, 1)
        fig.subplots_adjust(
            left=0.025,
            right=0.99,
            top=0.95,
            bottom=0.015,
            hspace=0.09,
        )
        fig.suptitle(
            f"{phase} | {video} | frame {frame_str} | idx {gidx_str}",
            fontsize=14,
        )

        self._overlay_depth_points(
            axs[0],
            img=img,
            pts=gt_pts,
            depth_values=gt_depths,
            ylabel="GT points",
            n=len(gt_pts),
            textbox_extra="color: GT depth",
        )

        self._overlay_depth_points(
            axs[1],
            img=img,
            pts=sim1_pts,
            depth_values=sim1_depths,
            ylabel="Sample thinning",
            n=sim1_stats["num_points"],
            textbox_extra="color: pred depth",
        )

        self._overlay_depth_points(
            axs[2],
            img=img,
            pts=sim2_pts,
            depth_values=sim2_depths,
            ylabel="Sample block",
            n=sim2_stats["num_points"],
            textbox_extra="color: pred depth",
        )

        self._imshow(
            axs[3],
            lam_mass,
            "Pred λ mass",
            cmap="hot",
            fig=fig,
            cbar=True,
        )
        self._textbox(
            axs[3],
            f"E[N]={exp_n:.2f}\nGT={gt_n:.0f}\nΔ={exp_n - gt_n:+.2f}",
        )

        self._imshow(
            axs[4],
            lam_mass ** self.gamma,
            "Pred λ^γ",
            cmap="hot",
            fig=fig,
            cbar=True,
        )
        self._textbox(axs[4], f"γ={self.gamma}")

        if pred_depth is not None:
            self._imshow(
                axs[5],
                pred_depth,
                "Pred depth [m]",
                cmap=self.depth_cmap,
                vmin=self.depth_vmin,
                vmax=self.depth_vmax,
                fig=fig,
                cbar=True,
            )
            self._overlay_gt_depth_dots(
                axs[5],
                gt_pts=gt_pts,
                gt_depths=gt_depths,
            )
            self._textbox(
                axs[5],
                f"GT depth dots: N={len(gt_depths)}",
            )
        else:
            axs[5].axis("off")
            self._label(axs[5], "Pred depth")
            self._textbox(axs[5], "missing channel")

        if pred_vel is not None:
            self._imshow(
                axs[6],
                pred_vel,
                pred_vel_label,
                cmap=self.vel_cmap,
                vmin=self.vel_vmin,
                vmax=self.vel_vmax,
                fig=fig,
                cbar=True,
            )
            self._overlay_gt_velocity_dots(
                axs[6],
                gt_pts=gt_pts,
                gt_vels=gt_vels,
            )
            self._textbox(
                axs[6],
                f"mode={velocity_mode}\nGT vel dots: N={len(gt_vels)}",
            )
        else:
            axs[6].axis("off")
            self._label(axs[6], "Pred vel")
            self._textbox(axs[6], "missing channel")

        out_path = self._output_path(
            runner=runner,
            phase=phase,
            video=video,
            frame_str=frame_str,
            gidx_str=gidx_str,
        )

        fig.savefig(out_path, dpi=self.dpi, bbox_inches="tight")
        fig.clear()