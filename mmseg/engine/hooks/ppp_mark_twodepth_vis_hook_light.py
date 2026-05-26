"""
Visualization hook for PPP-based models with two-depth prediction.

Expected prediction channels:
    pred[0]: intensity logits
    pred[1]: raw_d0
    pred[2]: raw_delta
    pred[3]: raw_velocity

Depth reconstruction:
    d0      = exp(raw_d0)
    delta   = softplus(raw_delta)
    d_near  = clamp(d0 - delta, min=depth_eps)
    d_far   = d0 + delta
    velocity = raw_velocity ** 3
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from mmengine.dist import is_main_process
from mmengine.hooks import Hook
from mmengine.model import is_model_wrapper
from mmseg.registry import HOOKS

import matplotlib
matplotlib.use("Agg")
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure


def extract_gt_points(mask: np.ndarray) -> np.ndarray:
    ys, xs = np.where(mask > 0)
    if xs.size == 0:
        return np.zeros((0, 2), dtype=np.float64)
    return np.stack([xs + 0.5, ys + 0.5], axis=1).astype(np.float64)


def sample_ppp_thinning(lam: np.ndarray, rng=None):
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


def fig_to_rgb(fig: Figure) -> np.ndarray:
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    return np.asarray(canvas.buffer_rgba(), dtype=np.uint8)[..., :3].copy()


def two_depth_from_pred(pred: torch.Tensor, depth_eps: float):
    if pred.size(0) < 4:
        raise ValueError(
            f"Two-depth visualization expects 4 prediction channels, got {pred.size(0)}"
        )

    raw_d0 = pred[1]
    raw_delta = pred[2]
    raw_vel = pred[3]

    d0 = torch.exp(raw_d0)
    delta = F.softplus(raw_delta)
    d_near = torch.clamp(d0 - delta, min=float(depth_eps))
    d_far = d0 + delta
    vel = raw_vel ** 3

    return d0, delta, d_near, d_far, vel


@HOOKS.register_module()
class PPPMarkTwoDepthVisHookLight(Hook):
    def __init__(
        self,
        train_interval=500,
        max_samples_train=1,
        max_samples_val=10,
        max_samples_test=10,
        img_mean=None,
        img_std=None,
        dpi=90,
        figsize=(14, 46),
        depth_vmin=None,
        depth_vmax=None,
        delta_vmin=None,
        delta_vmax=None,
        vel_vmin=None,
        vel_vmax=None,
        gamma=0.9,
        sim_block_size=8,
        sim_block_placement="weighted",
        point_size=28,
        depth_eps=1e-3,
    ):
        self.train_interval = int(train_interval)
        self.max_samples = {
            "train": int(max_samples_train),
            "val": int(max_samples_val),
            "test": int(max_samples_test),
        }

        self.dpi = int(dpi)
        self.figsize = tuple(figsize)

        self.depth_vmin = depth_vmin
        self.depth_vmax = depth_vmax
        self.delta_vmin = delta_vmin
        self.delta_vmax = delta_vmax
        self.vel_vmin = vel_vmin
        self.vel_vmax = vel_vmax

        self.gamma = float(gamma)
        self.sim_block_size = int(sim_block_size)
        self.sim_block_placement = str(sim_block_placement)
        self.point_size = float(point_size)
        self.depth_eps = float(depth_eps)

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

        self._written_in_run = {
            "val": 0,
            "test": 0,
        }

    def before_val_epoch(self, runner):
        self._written_in_run["val"] = 0

    def before_test_epoch(self, runner):
        self._written_in_run["test"] = 0

    def after_train_iter(self, runner, batch_idx, data_batch=None, outputs=None):
        if not is_main_process():
            return
        if not isinstance(data_batch, dict):
            return
        if self.max_samples["train"] <= 0:
            return
        if self.train_interval <= 0:
            return
        if runner.iter % self.train_interval != 0:
            return

        self._run(runner, data_batch, "train", runner.iter)

    def after_val_iter(self, runner, batch_idx, data_batch=None, outputs=None):
        if not is_main_process():
            return
        if not isinstance(data_batch, dict):
            return
        if self.max_samples["val"] <= 0:
            return
        if self._written_in_run["val"] >= self.max_samples["val"]:
            return

        self._run(runner, data_batch, "val", runner.iter)

    def after_test_iter(self, runner, batch_idx, data_batch=None, outputs=None):
        if not is_main_process():
            return
        if not isinstance(data_batch, dict):
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

    @staticmethod
    def _masked_map(arr: np.ndarray, mask: np.ndarray) -> np.ndarray:
        out = arr.copy()
        out[mask <= 0] = np.nan
        return out

    def _visualize_sample(self, runner, step, phase, img_tensor, pred, sample, iter_seed, sample_idx):
        gt_all = sample.gt_sem_seg.data.to(pred.device)

        gt_mask_t = gt_all[0]
        gt_depth_t = gt_all[1] if gt_all.size(0) > 1 else None
        gt_vel_t = gt_all[2] if gt_all.size(0) > 2 else None

        lam = torch.exp(pred[0]).detach().cpu().numpy()
        d0_t, delta_t, d_near_t, d_far_t, vel_t = two_depth_from_pred(
            pred,
            depth_eps=self.depth_eps,
        )

        d0 = d0_t.detach().cpu().numpy()
        delta = delta_t.detach().cpu().numpy()
        d_near = d_near_t.detach().cpu().numpy()
        d_far = d_far_t.detach().cpu().numpy()
        vel = vel_t.detach().cpu().numpy()

        gt_mask = gt_mask_t.detach().cpu().numpy()
        gt_depth = (
            gt_depth_t.detach().cpu().numpy()
            if gt_depth_t is not None
            else None
        )
        gt_vel = (
            gt_vel_t.detach().cpu().numpy()
            if gt_vel_t is not None
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
        axs = fig.subplots(12, 1)
        fig.subplots_adjust(
            left=0.025,
            right=0.99,
            top=0.975,
            bottom=0.01,
            hspace=0.08,
        )
        fig.suptitle(
            f"{phase} | {video} | frame {frame_str} | idx {gidx_str}",
            fontsize=14,
        )

        self._imshow(axs[0], img, "Image")
        self._imshow(axs[1], gt_mask, "GT support", cmap="gray")
        self._overlay(axs[2], img, gt_pts, "GT points", "lime", len(gt_pts))
        self._overlay(
            axs[3],
            img,
            sim1_pts,
            "Sim1 thinning",
            "red",
            sim1_stats["num_points"],
        )
        self._overlay(
            axs[4],
            img,
            sim2_pts,
            "Sim2 block",
            "orange",
            sim2_stats["num_points"],
        )

        self._imshow(
            axs[5],
            lam_mass,
            "Pred λ mass",
            cmap="hot",
            fig=fig,
            cbar=True,
        )
        self._textbox(
            axs[5],
            f"E[N]={exp_n:.2f}\nGT={gt_n:.0f}\nΔ={exp_n - gt_n:+.2f}",
        )

        self._imshow(
            axs[6],
            lam_mass ** self.gamma,
            "Pred λ^γ",
            cmap="hot",
            fig=fig,
            cbar=True,
        )
        self._textbox(axs[6], f"γ={self.gamma}")

        if gt_depth is not None:
            self._imshow(
                axs[7],
                self._masked_map(gt_depth, gt_mask),
                "GT depth [m]",
                cmap="viridis",
                vmin=self.depth_vmin,
                vmax=self.depth_vmax,
                fig=fig,
                cbar=True,
            )
        else:
            axs[7].axis("off")
            self._label(axs[7], "GT depth [m]")

        self._imshow(
            axs[8],
            d_near,
            "Pred d_near [m]",
            cmap="viridis",
            vmin=self.depth_vmin,
            vmax=self.depth_vmax,
            fig=fig,
            cbar=True,
        )

        self._imshow(
            axs[9],
            d_far,
            "Pred d_far [m]",
            cmap="viridis",
            vmin=self.depth_vmin,
            vmax=self.depth_vmax,
            fig=fig,
            cbar=True,
        )

        self._imshow(
            axs[10],
            delta,
            "Pred delta [m]",
            cmap="magma",
            vmin=self.delta_vmin,
            vmax=self.delta_vmax,
            fig=fig,
            cbar=True,
        )
        self._textbox(
            axs[10],
            f"d0 median={float(np.nanmedian(d0)):.2f}\n"
            f"delta median={float(np.nanmedian(delta)):.2f}",
        )

        self._imshow(
            axs[11],
            vel,
            "Pred vel [m/s]",
            cmap="coolwarm",
            vmin=self.vel_vmin,
            vmax=self.vel_vmax,
            fig=fig,
            cbar=True,
        )

        if gt_vel is not None:
            self._textbox(
                axs[11],
                "velocity shown from pred[3]\n"
                "GT velocity exists but is not plotted here",
            )

        tag = f"ppp_vis/{phase}/{video}_f{frame_str}_idx{gidx_str}_sample{sample_idx}"
        runner.visualizer.add_image(tag, fig_to_rgb(fig), step=step)
        fig.clear()