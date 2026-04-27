"""
PPP demo visualization hook.

Reduced visualization hook for producing demo frames as image files.

It keeps only:
  1. GT points projected on the camera image
  2. PPP sample by thinning projected on the camera image
  3. PPP sample by block approximation projected on the camera image
  4. Predicted PPP intensity mass heatmap
  5. Gamma-adjusted predicted PPP intensity mass heatmap

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


def extract_gt_points(mask: np.ndarray) -> np.ndarray:
    """Return point coordinates at positive GT mask cells as (x, y)."""
    ys, xs = np.where(mask > 0)
    if xs.size == 0:
        return np.zeros((0, 2), dtype=np.float64)
    return np.stack([xs + 0.5, ys + 0.5], axis=1).astype(np.float64)


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
class PPPDemoVisHook(Hook):
    """Write reduced PPP demo visualizations as PNG files."""

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
        figsize=(14, 20),
        gamma=0.9,
        sim_block_size=8,
        sim_block_placement="weighted",
        point_size=28,
        image_ext="png",
        write_all_ranks=False,
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

    def _imshow(self, ax, arr, ylabel, cmap=None, vmin=None, vmax=None, fig=None, cbar=False):
        im = ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
        ax.axis("off")
        self._label(ax, ylabel)

        if cbar and fig is not None:
            cax = ax.inset_axes([0.0, -0.06, 1.0, 0.04])
            cb = fig.colorbar(im, cax=cax, orientation="horizontal")
            cb.ax.tick_params(labelsize=10)

        return im

    def _overlay(self, ax, img, pts, ylabel, color, n):
        ax.imshow(img, aspect="auto")

        if len(pts):
            ax.scatter(
                pts[:, 0],
                pts[:, 1],
                s=self.point_size,
                c=color,
                alpha=0.95,
                edgecolors="none",
            )

        ax.axis("off")
        self._label(ax, ylabel)
        self._textbox(ax, f"N={n}")

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

        # Frame-based filename:
        # In distributed test runs, samplers can pad a few samples.
        # If a padded duplicate is written twice, this overwrites the same frame
        # instead of creating duplicate video frames.
        return out / f"frame_{frame_str}_idx_{gidx_str}.{self.image_ext}"

    def _visualize_sample(self, runner, step, phase, img_tensor, pred, sample, iter_seed, sample_idx):
        gt = sample.gt_sem_seg.data.to(pred.device)[0]

        lam = torch.exp(pred[0]).detach().cpu().numpy()
        gt_mask = gt.detach().cpu().numpy()
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

        # Existing convention from the original hook:
        # exp(pred[0]) is divided by image area before sampling/count display.
        lam_mass = lam / float(h * w)
        exp_n = float(lam_mass.sum())
        gt_n = float(gt_mask.sum())
        gt_pts = extract_gt_points(gt_mask)

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

        fig = Figure(figsize=self.figsize, dpi=self.dpi)
        FigureCanvasAgg(fig)
        axs = fig.subplots(5, 1)
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

        self._overlay(axs[0], img, gt_pts, "GT points", "lime", len(gt_pts))
        self._overlay(
            axs[1],
            img,
            sim1_pts,
            "Sample thinning",
            "red",
            sim1_stats["num_points"],
        )
        self._overlay(
            axs[2],
            img,
            sim2_pts,
            "Sample block",
            "orange",
            sim2_stats["num_points"],
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

        out_path = self._output_path(
            runner=runner,
            phase=phase,
            video=video,
            frame_str=frame_str,
            gidx_str=gidx_str,
        )

        fig.savefig(out_path, dpi=self.dpi, bbox_inches="tight")
        fig.clear()