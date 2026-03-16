import os
import time
import numpy as np
import torch
from mmengine.dist import is_main_process
from mmengine.model import is_model_wrapper

# Remedy "Tcl_AsyncDelete: async handler deleted by the wrong thread"
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mmengine.hooks import Hook
from mmseg.registry import HOOKS


def extract_gt_points_from_mask(mask: np.ndarray) -> np.ndarray:
    """Convert binary/support mask to point centers in image coordinates."""
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return np.zeros((0, 2), dtype=np.float64)
    return np.stack([xs + 0.5, ys + 0.5], axis=1).astype(np.float64)


def sample_ppp_by_thinning(
    lambda_grid: np.ndarray,
    *,
    pixel_area: float | None = None,
    rng: np.random.Generator | None = None,
):
    """
    Sample an inhomogeneous PPP using thinning from a homogeneous PPP.

    Parameters
    ----------
    lambda_grid : (H, W) array
        Intensity lambda(x) evaluated on a pixel grid (nonnegative).
        Assumed to already represent PPP intensity MASS per pixel.
    pixel_area : float
        Kept for API compatibility, not used.
    rng : np.random.Generator
        Random generator.

    Returns
    -------
    points_xy : (N, 2) array
        Accepted points in image coordinates (x, y).
    N : int
        Number of accepted points.
    lambda_max : float
        Max intensity used for thinning.
    N_prop : int
        Number of proposed homogeneous PPP points.
    """
    if rng is None:
        rng = np.random.default_rng()

    lam = np.asarray(lambda_grid, dtype=np.float64)
    assert np.all(lam >= 0)

    H, W = lam.shape

    lambda_max = lam.max()
    if lambda_max == 0:
        return np.zeros((0, 2), dtype=np.float64), 0, 0.0, 0

    area_total = H * W
    Lambda_hom = lambda_max * area_total

    N_prop = rng.poisson(Lambda_hom)
    if N_prop == 0:
        return np.zeros((0, 2), dtype=np.float64), 0, float(lambda_max), 0

    xs = rng.random(N_prop) * W
    ys = rng.random(N_prop) * H

    xi = np.clip(xs.astype(int), 0, W - 1)
    yi = np.clip(ys.astype(int), 0, H - 1)

    lam_x = lam[yi, xi]
    accept_prob = lam_x / lambda_max
    keep = rng.random(N_prop) < accept_prob

    points_xy = np.stack([xs[keep], ys[keep]], axis=1).astype(np.float64)
    return points_xy, int(points_xy.shape[0]), float(lambda_max), int(N_prop)


def sample_points_block_bernoulli(
    lambda_grid: np.ndarray,
    *,
    block_size: int = 8,
    pixel_area: float = 1.0,
    placement: str = "weighted",
    rng: np.random.Generator | None = None,
):
    """
    Block/Bernoulli approximation of a PPP:
      For each block A:
        Λ_A = sum_A λ
        Activate with p_A = 1 - exp(-Λ_A)
      If active, place exactly one point in A.

    Notes
    -----
    - Vectorized over blocks.
    - Only final point placement for active blocks is done in Python.
    - If lambda_grid already stores per-pixel mass, keep pixel_area=1.0.
    """
    if rng is None:
        rng = np.random.default_rng()

    lam = np.asarray(lambda_grid, dtype=np.float64)
    if lam.ndim != 2:
        raise ValueError("lambda_grid must be a 2D array.")
    if not np.all(np.isfinite(lam)):
        raise ValueError("lambda_grid contains NaN/Inf.")
    if np.any(lam < 0):
        raise ValueError("lambda_grid contains negative values.")

    bs = int(block_size)
    if bs <= 0:
        raise ValueError("block_size must be > 0.")
    if placement not in {"weighted", "uniform"}:
        raise ValueError("placement must be 'weighted' or 'uniform'")

    H, W = lam.shape

    nby = (H + bs - 1) // bs
    nbx = (W + bs - 1) // bs
    num_blocks = nby * nbx

    Hp = nby * bs
    Wp = nbx * bs
    if Hp != H or Wp != W:
        lam_pad = np.zeros((Hp, Wp), dtype=np.float64)
        lam_pad[:H, :W] = lam
    else:
        lam_pad = lam

    lam_blocks = lam_pad.reshape(nby, bs, nbx, bs).transpose(0, 2, 1, 3)
    block_sums = lam_blocks.sum(axis=(2, 3)) * pixel_area

    p_active = 1.0 - np.exp(-block_sums)
    expected_blocks_active = float(p_active.sum())

    active_mask = rng.random(size=(nby, nbx)) < p_active
    active_by, active_bx = np.nonzero(active_mask)
    num_active = active_by.size

    if num_active == 0:
        stats = dict(
            block_size=bs,
            placement=placement,
            num_blocks=num_blocks,
            num_active_blocks=0,
            expected_active_blocks=expected_blocks_active,
            num_points=0,
        )
        return np.zeros((0, 2), dtype=np.float64), stats

    points_xy = np.empty((num_active, 2), dtype=np.float64)

    x0s = active_bx * bs
    y0s = active_by * bs

    wBs = np.minimum(bs, W - x0s)
    hBs = np.minimum(bs, H - y0s)

    if placement == "uniform":
        points_xy[:, 0] = x0s + rng.random(num_active) * wBs
        points_xy[:, 1] = y0s + rng.random(num_active) * hBs
    else:
        for i, (by, bx, x0, y0, wB, hB) in enumerate(
            zip(active_by, active_bx, x0s, y0s, wBs, hBs)
        ):
            block = lam[y0:y0 + hB, x0:x0 + wB]
            s = block.sum()

            if s <= 0:
                points_xy[i, 0] = x0 + rng.random() * wB
                points_xy[i, 1] = y0 + rng.random() * hB
                continue

            probs = block.ravel() / s
            idx = rng.choice(probs.size, p=probs)
            py, px = divmod(idx, wB)

            points_xy[i, 0] = x0 + px + rng.random()
            points_xy[i, 1] = y0 + py + rng.random()

    stats = dict(
        block_size=bs,
        placement=placement,
        num_blocks=num_blocks,
        num_active_blocks=int(num_active),
        expected_active_blocks=expected_blocks_active,
        num_points=int(num_active),
    )
    return points_xy, stats


@HOOKS.register_module()
class PPPMarkVisHookLight(Hook):
    """
    Lightweight visualization hook for PPP marked models.

    Writes ONLY:
      - one PNG per visualized sample

    Notes:
      - Runs on main process only.
      - Adds an extra forward pass for visualization
        (extract_feat + decode_head).
      - No JSON output.
      - Uses the extended plot layout from PPPMarkVisualizationHook.
      - Keeps the light-hook suptitle format:
            f"{phase} | {video_str} | frame {vframe_str} | idx {gidx_str}"
    """

    def __init__(
        self,
        interval: int = 500,
        max_samples_train=1,
        max_samples_val=10,
        max_samples_test=10,
        out_subdir: str = "vis_data",
        img_mean=None,
        img_std=None,
        dpi: int = 90,
        pad_inches: float = 0.02,
        figsize=(14, 34),
        depth_vmin: float | None = None,
        depth_vmax: float | None = None,
        vel_vmin: float | None = None,
        vel_vmax: float | None = None,
        gamma: float = 0.9,
        sim_block_size: int = 8,
        sim_block_placement: str = "weighted",
        point_size: float = 28,
    ):
        self.interval = int(interval)
        self.max_samples_train = int(max_samples_train)
        self.max_samples_val = int(max_samples_val)
        self.max_samples_test = int(max_samples_test)
        self.out_subdir = str(out_subdir)

        self.dpi = int(dpi)
        self.pad_inches = float(pad_inches)
        self.figsize = tuple(figsize)

        self.depth_vmin = depth_vmin
        self.depth_vmax = depth_vmax
        self.vel_vmin = vel_vmin
        self.vel_vmax = vel_vmax

        self.gamma = float(gamma)
        self.sim_block_size = int(sim_block_size)
        self.sim_block_placement = str(sim_block_placement)
        self.point_size = float(point_size)

        self.img_mean = (
            np.array(img_mean, dtype=np.float32)[:, None, None]
            if img_mean is not None
            else None
        )
        self.img_std = (
            np.array(img_std, dtype=np.float32)[:, None, None]
            if img_std is not None
            else None
        )

    # -------------------------
    # Hook entry points
    # -------------------------
    def after_train_iter(self, runner, batch_idx, data_batch=None, outputs=None):
        if not self._should_run(runner, batch_idx, data_batch, phase="train"):
            return
        self._run_vis(runner, data_batch=data_batch, outputs=outputs, phase="train")

    def after_val_iter(self, runner, batch_idx, data_batch=None, outputs=None):
        if not self._should_run(runner, batch_idx, data_batch, phase="val"):
            return
        self._run_vis(
            runner, data_batch=data_batch, outputs=None, phase="val", batch_idx=batch_idx
        )

    def after_test_iter(self, runner, batch_idx, data_batch=None, outputs=None):
        if not self._should_run(runner, batch_idx, data_batch, phase="test"):
            return
        self._run_vis(
            runner, data_batch=data_batch, outputs=None, phase="test", batch_idx=batch_idx
        )

    # -------------------------
    # Internal helpers
    # -------------------------
    def _get_max_samples_for_phase(self, phase: str, batch_size: int) -> int:
        if phase == "train" and self.max_samples_train is not None:
            return min(self.max_samples_train, batch_size)
        if phase == "val" and self.max_samples_val is not None:
            return min(self.max_samples_val, batch_size)
        if phase == "test" and self.max_samples_test is not None:
            return min(self.max_samples_test, batch_size)
        return batch_size

    def _should_run(self, runner, batch_idx, data_batch, phase: str) -> bool:
        if not is_main_process():
            return False
        if not isinstance(data_batch, dict):
            return False

        if phase == "train":
            return (runner.iter % self.interval) == 0

        return (int(batch_idx) % self.interval) == 0

    def _get_vis_dir(self, runner, phase: str) -> str:
        out_dir = os.path.join(runner.log_dir, self.out_subdir, phase)
        os.makedirs(out_dir, exist_ok=True)
        return out_dir

    def _run_vis(
        self,
        runner,
        *,
        data_batch: dict,
        outputs: dict | None,
        phase: str,
        batch_idx: int | None = None,
    ):
        model = runner.model
        if is_model_wrapper(model):
            model = model.module

        was_training = model.training
        model.eval()

        proc = model.data_preprocessor(data_batch)
        inputs: torch.Tensor = proc["inputs"]
        data_samples = proc["data_samples"]

        with torch.no_grad():
            feats = model.extract_feat(inputs)
            preds = model.decode_head(feats)

        self._visualize_batch(
            runner=runner,
            inputs=inputs,
            preds=preds,
            data_samples=data_samples,
            outputs=outputs,
            phase=phase,
            batch_idx=batch_idx,
        )

        if was_training:
            model.train()

    def _denorm_img(self, img_chw: np.ndarray) -> np.ndarray:
        img = img_chw.astype(np.float32)
        if self.img_mean is not None and self.img_std is not None:
            img = img * self.img_std + self.img_mean
            img = (img + 1.0) * 127.5
        else:
            if img.min() >= -1.01 and img.max() <= 1.01:
                img = (img + 1.0) * 127.5
            elif img.max() <= 1.01:
                img = img * 255.0

        img = np.clip(img, 0, 255).astype(np.uint8)
        return np.transpose(img, (1, 2, 0))

    def _sanitize_for_filename(self, s: str) -> str:
        return "".join(c if c.isalnum() or c in "._-" else "_" for c in s)

    def _draw_points_overlay(
        self,
        ax,
        img: np.ndarray,
        points_xy: np.ndarray,
        *,
        ylabel: str | None = None,
        n_text: str,
        color: str,
        point_size: float = 30,
        alpha: float = 0.95,
    ):
        ax.imshow(img, aspect="auto")
        if points_xy is not None and len(points_xy) > 0:
            ax.scatter(
                points_xy[:, 0],
                points_xy[:, 1],
                s=point_size,
                c=color,
                alpha=alpha,
                edgecolors="none",
            )

        if ylabel:
            ax.set_ylabel(ylabel, fontsize=13)
        ax.axis("off")

        ax.text(
            0.02,
            0.98,
            n_text,
            transform=ax.transAxes,
            fontsize=11,
            color="white",
            va="top",
            ha="left",
            bbox=dict(facecolor="black", alpha=0.55, pad=3),
        )

    def _visualize_batch(
        self,
        *,
        runner,
        inputs: torch.Tensor,
        preds: torch.Tensor,
        data_samples: list,
        outputs: dict | None,
        phase: str,
        batch_idx: int | None,
    ):
        num_vis = self._get_max_samples_for_phase(phase, preds.size(0))
        device = inputs.device
        out_dir = self._get_vis_dir(runner, phase)

        for i in range(num_vis):
            pred = preds[i]
            gt = data_samples[i].gt_sem_seg.data.to(device)

            pred_log_intensity = pred[0]
            pred_lambda_theta = torch.exp(pred_log_intensity)

            pred_depth = pred[1] if pred.size(0) > 1 else None
            pred_velocity = pred[2] if pred.size(0) > 2 else None

            gt_support = gt[0]
            gt_depth = gt[1] if gt.size(0) > 1 else None
            gt_velocity = gt[2] if gt.size(0) > 2 else None

            img = self._denorm_img(inputs[i].detach().cpu().numpy())

            pred_lambda_theta_np = pred_lambda_theta.detach().cpu().numpy()
            gt_support_np = gt_support.detach().cpu().numpy()

            meta = data_samples[i].metainfo

            video_name = meta.get("video_name", None)
            video_frame_idx = meta.get("video_frame_idx", None)
            global_idx = meta.get("global_idx", meta.get("frame_idx", None))

            video_str = (
                self._sanitize_for_filename(str(video_name))
                if video_name is not None
                else "unknown_video"
            )
            vframe_str = (
                f"{int(video_frame_idx):06d}"
                if video_frame_idx is not None
                else "unknown_frame"
            )
            gidx_str = (
                f"{int(global_idx):08d}"
                if global_idx is not None
                else "unknown_idx"
            )

            ori_shape = meta.get("ori_shape", None)
            if isinstance(ori_shape, (tuple, list)) and len(ori_shape) >= 2:
                H, W = int(ori_shape[0]), int(ori_shape[1])
            else:
                H, W = int(pred_lambda_theta.shape[-2]), int(pred_lambda_theta.shape[-1])

            pixel_area = float(H * W)
            pred_lambda_mass = pred_lambda_theta_np / pixel_area
            expected_count = float(pred_lambda_mass.sum())
            gt_count = float(gt_support_np.sum())

            gt_points_xy = extract_gt_points_from_mask(gt_support_np)

            seed_base = int(meta.get("frame_idx", 0)) + 1000 * int(runner.iter)
            rng_thin = np.random.default_rng(seed_base + 11)
            rng_block = np.random.default_rng(seed_base + 29)

            sim1_points_xy, sim1_N, sim1_lambda_max, sim1_N_prop = sample_ppp_by_thinning(
                pred_lambda_mass,
                rng=rng_thin,
            )

            sim2_points_xy, sim2_stats = sample_points_block_bernoulli(
                pred_lambda_mass,
                block_size=self.sim_block_size,
                pixel_area=1.0,
                placement=self.sim_block_placement,
                rng=rng_block,
            )
            sim2_N = int(sim2_stats["num_points"])

            prefix = (
                f"iter_{runner.iter:07d}_{phase}_"
                f"{video_str}_f{vframe_str}_idx{gidx_str}_s{i}"
            )

            fig, axs = plt.subplots(
                9,
                1,
                figsize=self.figsize,
            )

            plt.subplots_adjust(
                left=0.01,
                right=0.99,
                top=0.97,
                bottom=0.01,
                hspace=0.08,
            )

            fig.suptitle(
                f"{phase} | {video_str} | frame {vframe_str} | idx {gidx_str}",
                fontsize=14,
            )

            def _cbar_under(fig_, ax_, im_, height=0.04, pad=0.02):
                cax = ax_.inset_axes([0.0, -(height + pad), 1.0, height])
                cb = fig_.colorbar(im_, cax=cax, orientation="horizontal")
                cb.ax.tick_params(labelsize=10)
                return cb

            def _ylabel(ax_, text):
                ax_.text(
                    -0.015,
                    0.5,
                    text,
                    transform=ax_.transAxes,
                    rotation=90,
                    fontsize=13,
                    va="center",
                    ha="center",
                )

            row = 0

            # 1) Image
            axs[row].imshow(img, aspect="auto")
            axs[row].axis("off")
            _ylabel(axs[row], "Image")
            row += 1

            # 2) GT support
            axs[row].imshow(gt_support_np, cmap="gray", aspect="auto")
            axs[row].axis("off")
            _ylabel(axs[row], "GT support")
            row += 1

            # 3) GT points
            self._draw_points_overlay(
                axs[row],
                img,
                gt_points_xy,
                n_text=f"N={len(gt_points_xy)}",
                color="lime",
                point_size=self.point_size,
                alpha=0.95,
            )
            _ylabel(axs[row], "GT points")
            row += 1

            # 4) Sim1 thinning
            self._draw_points_overlay(
                axs[row],
                img,
                sim1_points_xy,
                n_text=f"N={sim1_N}",
                color="red",
                point_size=self.point_size,
                alpha=0.90,
            )
            _ylabel(axs[row], "Sim1 thinning")
            row += 1

            # 5) Sim2 block
            self._draw_points_overlay(
                axs[row],
                img,
                sim2_points_xy,
                n_text=f"N={sim2_N}",
                color="orange",
                point_size=self.point_size,
                alpha=0.90,
            )
            _ylabel(axs[row], "Sim2 block")
            row += 1

            # 6) Pred λ mass
            im0 = axs[row].imshow(pred_lambda_mass, cmap="hot", aspect="auto")
            axs[row].axis("off")
            axs[row].text(
                0.02,
                0.98,
                f"E[N]={expected_count:.2f}\nGT={gt_count:.0f}\nΔ={expected_count-gt_count:+.2f}",
                transform=axs[row].transAxes,
                fontsize=11,
                color="white",
                va="top",
                ha="left",
                bbox=dict(facecolor="black", alpha=0.55, pad=3),
            )
            _ylabel(axs[row], "Pred λ mass")
            _cbar_under(fig, axs[row], im0)
            row += 1

            # 7) Pred λ^γ
            gamma = self.gamma
            lam_vis = pred_lambda_mass ** gamma

            im1 = axs[row].imshow(lam_vis, cmap="hot", aspect="auto")
            axs[row].axis("off")
            axs[row].text(
                0.02,
                0.98,
                f"γ={gamma}",
                transform=axs[row].transAxes,
                fontsize=11,
                color="white",
                va="top",
                ha="left",
                bbox=dict(facecolor="black", alpha=0.55, pad=3),
            )
            _ylabel(axs[row], "Pred λ^γ")
            _cbar_under(fig, axs[row], im1)
            row += 1

            # 8) Depth
            if pred_depth is not None:
                depth_np = pred_depth.detach().cpu().numpy()
                im2 = axs[row].imshow(
                    depth_np,
                    cmap="viridis",
                    vmin=self.depth_vmin,
                    vmax=self.depth_vmax,
                    aspect="auto",
                )
                axs[row].axis("off")
                _cbar_under(fig, axs[row], im2)
                _ylabel(axs[row], "Pred depth")
            else:
                axs[row].axis("off")
            row += 1

            # 9) Velocity
            if pred_velocity is not None:
                vel_np = pred_velocity.detach().cpu().numpy()
                im3 = axs[row].imshow(
                    vel_np,
                    cmap="coolwarm",
                    vmin=self.vel_vmin,
                    vmax=self.vel_vmax,
                    aspect="auto",
                )
                axs[row].axis("off")
                _cbar_under(fig, axs[row], im3)
                _ylabel(axs[row], "Pred vel")
            else:
                axs[row].axis("off")

            png_path = os.path.join(out_dir, f"{prefix}.png")
            fig.savefig(
                png_path,
                dpi=self.dpi,
                bbox_inches="tight",
                pad_inches=self.pad_inches,
            )
            plt.close(fig)


# import os
# import json
# import numpy as np
# import torch
# from mmengine.dist import is_main_process
# from mmengine.model import is_model_wrapper

# # Remedy "Tcl_AsyncDelete: async handler deleted by the wrong thread"
# import matplotlib
# matplotlib.use("Agg")
# import matplotlib.pyplot as plt

# from mmengine.hooks import Hook
# from mmseg.registry import HOOKS


# @HOOKS.register_module()
# class PPPMarkVisHookLight(Hook):
#     """
#     Lightweight visualization hook for PPP marked models.

#     Writes ONLY:
#       - one PNG per visualized sample
#       - one JSON (meta + small diagnostics) per visualized sample

#     Notes:
#       - Runs on main process only.
#       - Adds an extra forward pass for visualization (extract_feat + decode_head),
#         since training outputs do not reliably expose dense predictions.
#       - Uses tight saving (bbox_inches="tight") and tiny padding to reduce whitespace.
#     """

#     def __init__(
#         self,
#         interval: int = 500,
#         max_samples_train=1,
#         max_samples_val=10,
#         max_samples_test=10,
#         out_subdir: str = "vis_data",
#         img_mean=None,
#         img_std=None,
#         # Plot controls
#         dpi: int = 200,
#         pad_inches: float = 0.02,
#         figsize=(12, 7),
#         # Optional fixed scales for better comparability across frames
#         depth_vmin: float | None = None,
#         depth_vmax: float | None = None,
#         vel_vmin: float | None = None,
#         vel_vmax: float | None = None,
#     ):
#         self.interval = int(interval)
#         self.max_samples_train = int(max_samples_train)
#         self.max_samples_val = int(max_samples_val)
#         self.max_samples_test = int(max_samples_test)
#         self.out_subdir = str(out_subdir)

#         self.dpi = int(dpi)
#         self.pad_inches = float(pad_inches)
#         self.figsize = tuple(figsize)

#         self.depth_vmin = depth_vmin
#         self.depth_vmax = depth_vmax
#         self.vel_vmin = vel_vmin
#         self.vel_vmax = vel_vmax

#         self.img_mean = (
#             np.array(img_mean, dtype=np.float32)[:, None, None]
#             if img_mean is not None
#             else None
#         )
#         self.img_std = (
#             np.array(img_std, dtype=np.float32)[:, None, None]
#             if img_std is not None
#             else None
#         )

#     # -------------------------
#     # Hook entry points
#     # -------------------------
#     def after_train_iter(self, runner, batch_idx, data_batch=None, outputs=None):
#         if not self._should_run(runner, batch_idx, data_batch, phase="train"):
#             return
#         self._run_vis(runner, data_batch=data_batch, outputs=outputs, phase="train")

#     def after_val_iter(self, runner, batch_idx, data_batch=None, outputs=None):
#         if not self._should_run(runner, batch_idx, data_batch, phase="val"):
#             return
#         self._run_vis(
#             runner, data_batch=data_batch, outputs=None, phase="val", batch_idx=batch_idx
#         )

#     def after_test_iter(self, runner, batch_idx, data_batch=None, outputs=None):
#         if not self._should_run(runner, batch_idx, data_batch, phase="test"):
#             return
#         self._run_vis(
#             runner, data_batch=data_batch, outputs=None, phase="test", batch_idx=batch_idx
#         )

#     # -------------------------
#     # Internal helpers
#     # -------------------------
#     def _get_max_samples_for_phase(self, phase: str, batch_size: int) -> int:
#         if phase == "train" and self.max_samples_train is not None:
#             return min(self.max_samples_train, batch_size)
#         if phase == "val" and self.max_samples_val is not None:
#             return min(self.max_samples_val, batch_size)
#         if phase == "test" and self.max_samples_test is not None:
#             return min(self.max_samples_test, batch_size)
#         return batch_size
    
#     def _should_run(self, runner, batch_idx, data_batch, phase: str) -> bool:
#         if not is_main_process():
#             return False
#         if not isinstance(data_batch, dict):
#             return False

#         # Train: gate by global iteration
#         if phase == "train":
#             return (runner.iter % self.interval) == 0

#         # Val/Test: gate by batch_idx (more intuitive than runner.iter)
#         return (int(batch_idx) % self.interval) == 0

#     def _get_vis_dir(self, runner, phase: str) -> str:
#         out_dir = os.path.join(runner.log_dir, self.out_subdir, phase)
#         os.makedirs(out_dir, exist_ok=True)
#         return out_dir

#     def _run_vis(
#         self,
#         runner,
#         *,
#         data_batch: dict,
#         outputs: dict | None,
#         phase: str,
#         batch_idx: int | None = None,
#     ):
#         model = runner.model
#         if is_model_wrapper(model):
#             model = model.module

#         was_training = model.training
#         model.eval()

#         proc = model.data_preprocessor(data_batch)
#         inputs: torch.Tensor = proc["inputs"]
#         data_samples = proc["data_samples"]

#         with torch.no_grad():
#             feats = model.extract_feat(inputs)
#             preds = model.decode_head(feats)

#         self._visualize_batch(
#             runner=runner,
#             inputs=inputs,
#             preds=preds,
#             data_samples=data_samples,
#             outputs=outputs,
#             phase=phase,
#             batch_idx=batch_idx,
#         )

#         if was_training:
#             model.train()

#     def _denorm_img(self, img_chw: np.ndarray) -> np.ndarray:
#         img = img_chw.astype(np.float32)
#         if self.img_mean is not None and self.img_std is not None:
#             img = img * self.img_std + self.img_mean
#         # If image is in [-1, 1], map it back to [0, 255] for visualization
#         img = (img + 1.0) * 127.5
#         img = np.clip(img, 0, 255).astype(np.uint8)
#         return np.transpose(img, (1, 2, 0))

#     def _sanitize_for_filename(self, s: str) -> str:
#         # Keep filenames safe on cluster FS.
#         return "".join(c if c.isalnum() or c in "._-" else "_" for c in s)

#     def _visualize_batch(
#         self,
#         *,
#         runner,
#         inputs: torch.Tensor,
#         preds: torch.Tensor,
#         data_samples: list,
#         outputs: dict | None,
#         phase: str,
#         batch_idx: int | None,
#     ):
#         num_vis = self._get_max_samples_for_phase(phase, preds.size(0))
#         device = inputs.device
#         out_dir = self._get_vis_dir(runner, phase)

#         for i in range(num_vis):
#             pred = preds[i]  # (C,H,W)
#             gt = data_samples[i].gt_sem_seg.data.to(device)  # (C_gt,H,W)

#             # --- Split channels (assumes channel order: log_intensity, depth, velocity) ---
#             pred_log_intensity = pred[0]
#             pred_lambda = torch.exp(pred_log_intensity)

#             pred_depth = pred[1] if pred.size(0) > 1 else None
#             pred_vel = pred[2] if pred.size(0) > 2 else None

#             gt_support = gt[0]

#             # --- Image ---
#             img = self._denorm_img(inputs[i].detach().cpu().numpy())

#             # --- Metainfo (now includes video_name + video_frame_idx) ---
#             meta = data_samples[i].metainfo

#             video_name = meta.get("video_name", None)
#             video_frame_idx = meta.get("video_frame_idx", None)
#             global_idx = meta.get("global_idx", meta.get("frame_idx", None))
#             sample_id = meta.get("sample_id", None)

#             video_str = self._sanitize_for_filename(str(video_name)) if video_name else "unknown_video"
#             vframe_str = f"{int(video_frame_idx):06d}" if video_frame_idx is not None else "unknown_frame"
#             gidx_str = f"{int(global_idx):08d}" if global_idx is not None else "unknown_idx"

#             # --- PPP count diagnostics (match PPPLoss-style normalization) ---
#             ori_shape = meta.get("ori_shape", None)
#             if isinstance(ori_shape, (tuple, list)) and len(ori_shape) >= 2:
#                 H, W = int(ori_shape[0]), int(ori_shape[1])
#             else:
#                 H, W = int(pred_lambda.shape[-2]), int(pred_lambda.shape[-1])

#             pixel_area = float(H * W)
#             pred_lambda_mass = (pred_lambda / pixel_area).detach().cpu().numpy()
#             expected_count = float(pred_lambda_mass.sum())
#             gt_count = float(gt_support.detach().cpu().numpy().sum())

#             # --- filenames / prefix ---
#             prefix = (
#                 f"iter_{runner.iter:07d}_{phase}_"
#                 f"{video_str}_f{vframe_str}_idx{gidx_str}_s{i}"
#             )

#             # --- JSON ---
#             json_dump = {
#                 "phase": phase,
#                 "iter": int(runner.iter),
#                 "epoch": int(getattr(runner, "epoch", 0)),
#                 "batch_idx": None if batch_idx is None else int(batch_idx),
#                 "sample_in_batch": int(i),
#                 "sample_id": sample_id,
#                 "global_idx": None if global_idx is None else int(global_idx),
#                 "video_name": video_name,
#                 "video_frame_idx": None if video_frame_idx is None else int(video_frame_idx),
#                 "img_shape": meta.get("img_shape", None),
#                 "ori_shape": meta.get("ori_shape", None),
#                 "ppp_diagnostics": {
#                     "H": int(H),
#                     "W": int(W),
#                     "img_area": int(H * W),
#                     "expected_count": expected_count,
#                     "gt_count": gt_count,
#                     "count_error": expected_count - gt_count,
#                 },
#                 "losses": (
#                     {
#                         k: float(v)
#                         for k, v in (outputs or {}).items()
#                         if torch.is_tensor(v) and v.numel() == 1
#                     }
#                     if phase == "train"
#                     else {}
#                 ),
#             }

#             with open(os.path.join(out_dir, f"{prefix}.json"), "w") as f:
#                 json.dump(json_dump, f, indent=2)

#             # --- Plot: 5 rows x 1 col, FULL screen, no whitespace ---

#             fig, axs = plt.subplots(
#                 5,
#                 1,
#                 figsize=(14, 20),   # large panels
#             )

#             # remove ALL padding
#             plt.subplots_adjust(
#                 left=0.01,
#                 right=0.99,
#                 top=0.97,
#                 bottom=0.01,
#                 hspace=0.08,
#             )

#             fig.suptitle(
#                 f"{phase} | {video_str} | frame {vframe_str} | idx {gidx_str}",
#                 fontsize=14,
#             )

#             def _cbar_under(fig, ax, im, height=0.04, pad=0.02):
#                 cax = ax.inset_axes([0.0, -(height + pad), 1.0, height])
#                 cb = fig.colorbar(im, cax=cax, orientation="horizontal")
#                 cb.ax.tick_params(labelsize=10)
#                 return cb

#             row = 0

#             # 1) Image
#             axs[row].imshow(img, aspect="auto")
#             axs[row].set_title("Image", fontsize=13)
#             axs[row].axis("off")
#             row += 1

#             # 2) Pred λ mass
#             im0 = axs[row].imshow(pred_lambda_mass, cmap="hot", aspect="auto")
#             axs[row].set_title("Pred λ mass", fontsize=13)
#             axs[row].axis("off")

#             axs[row].text(
#                 0.02,
#                 0.98,
#                 f"E[N]={expected_count:.2f}\nGT={gt_count:.0f}\nΔ={expected_count-gt_count:+.2f}",
#                 transform=axs[row].transAxes,
#                 fontsize=11,
#                 color="white",
#                 va="top",
#                 ha="left",
#                 bbox=dict(facecolor="black", alpha=0.55, pad=3),
#             )

#             _cbar_under(fig, axs[row], im0)
#             row += 1

#             # 3) GT support
#             axs[row].imshow(gt_support.detach().cpu().numpy(), cmap="gray", aspect="auto")
#             axs[row].set_title("GT support", fontsize=13)
#             axs[row].axis("off")
#             row += 1

#             # 4) Depth
#             if pred_depth is not None:
#                 depth_np = pred_depth.detach().cpu().numpy()
#                 im1 = axs[row].imshow(
#                     depth_np,
#                     cmap="viridis",
#                     vmin=self.depth_vmin,
#                     vmax=self.depth_vmax,
#                     aspect="auto",
#                 )
#                 axs[row].set_title("Pred depth", fontsize=13)
#                 axs[row].axis("off")
#                 _cbar_under(fig, axs[row], im1)
#             else:
#                 axs[row].axis("off")
#             row += 1

#             # 5) Velocity
#             if pred_vel is not None:
#                 vel_np = pred_vel.detach().cpu().numpy()
#                 im2 = axs[row].imshow(
#                     vel_np,
#                     cmap="coolwarm",
#                     vmin=self.vel_vmin,
#                     vmax=self.vel_vmax,
#                     aspect="auto",
#                 )
#                 axs[row].set_title("Pred vel", fontsize=13)
#                 axs[row].axis("off")
#                 _cbar_under(fig, axs[row], im2)
#             else:
#                 axs[row].axis("off")

#             png_path = os.path.join(out_dir, f"{prefix}.png")

#             fig.savefig(
#                 png_path,
#                 dpi=90,           # keeps file size reasonable
#                 bbox_inches="tight",
#             )

#             plt.close(fig)