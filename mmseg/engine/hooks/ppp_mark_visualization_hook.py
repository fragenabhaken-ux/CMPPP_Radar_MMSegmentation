import os
import json
import numpy as np
import torch
from mmengine.dist import is_main_process
from mmengine.model import is_model_wrapper

# to remedy "Tcl_AsyncDelete: async handler deleted by the wrong thread"
import matplotlib
matplotlib.use('Agg')

import time

import matplotlib.pyplot as plt
import torch.nn.functional as F

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
        NOTE: lambda_grid is assumed to already represent PPP intensity MASS per pixel.
    pixel_area : float
        Area represented by one pixel (kept for API compatibility, not used).
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
    """
    if rng is None:
        rng = np.random.default_rng()

    lam = np.asarray(lambda_grid, dtype=np.float64)
    assert np.all(lam >= 0)

    H, W = lam.shape

    lambda_max = lam.max()
    if lambda_max == 0:
        return np.zeros((0, 2)), 0, 0.0, 0

    # Total intensity of homogeneous PPP
    # lambda_grid already represents intensity mass per pixel,
    # so the domain area is H * W (NOT multiplied by pixel_area).
    area_total = H * W
    Lambda_hom = lambda_max * area_total

    # 1) sample homogeneous PPP
    # N_prop is the number of proposal points in the homogeneous Poisson point process
    # with intensity lambda_max over the whole image domain.
    N_prop = rng.poisson(Lambda_hom)

    if N_prop == 0:
        return np.zeros((0, 2)), 0, lambda_max, 0

    # uniform proposal locations over image
    xs = rng.random(N_prop) * W
    ys = rng.random(N_prop) * H

    # Converts continuous x-coordinates to integers, Equivalent to floor(xs) for
    # nonnegative values, Maps point → pixel index
    xi = np.clip(xs.astype(int), 0, W - 1)
    yi = np.clip(ys.astype(int), 0, H - 1)

    # 2) thinning step
    # For each proposed point (xi,yi), we evaluate the predicted intensity at its location
    # by looking up the corresponding pixel in lambda_grid.
    lam_x = lam[yi, xi]
    accept_prob = lam_x / lambda_max

    # this performs independent Bernoulli trials for all proposal points:
    # Each proposal point is kept if its random number is smaller than its acceptance probability
    keep = rng.random(N_prop) < accept_prob

    # Finally, we collect the coordinates of all accepted points.
    points_xy = np.stack([xs[keep], ys[keep]], axis=1)

    return points_xy, points_xy.shape[0], float(lambda_max), N_prop


def sample_ppp_by_thinning_torch(
    lambda_grid: "torch.Tensor",
    *,
    generator: "torch.Generator | None" = None,
) -> "tuple[torch.Tensor, int, float, int]":
    """
    Fully-vectorised PyTorch implementation of tiled inhomogeneous PPP sampling
    via thinning.  All proposals across all 8×8 blocks are batched into a single
    flat tensor of length tot_proposed, so memory is O(tot_proposed) rather than
    O(n_blocks × max_proposals_per_block).

    lambda_grid: (H, W) torch.Tensor of PPP intensity mass per pixel.
    Returns (points_xy, n_kept, lambda_max_global, tot_proposed) where points_xy
    is a (n_kept, 2) tensor on the same device as lambda_grid.
    """

    lam = lambda_grid.double()
    assert (lam >= 0).all(), "lambda_grid must be non-negative"
    H, W = lam.shape
    device = lam.device

    # Pad to the next multiple of 8 so no points land in the padded area
    Hp = (H + 7) // 8 * 8
    Wp = (W + 7) // 8 * 8
    if Hp != H or Wp != W:
        lam_pad = lam.new_zeros(Hp, Wp)
        lam_pad[:H, :W] = lam
        lam = lam_pad

    H8, W8 = Hp // 8, Wp // 8

    # Per-block max intensity: (H8, W8)
    lamb = lam.reshape(H8, 8, W8, 8).permute(0, 2, 1, 3).reshape(H8, W8, 64)
    lambda_max = lamb.max(dim=2).values  # (H8, W8)
    lam_max_global = float(lambda_max.max())

    # Per-block proposal counts ~ Poisson(lambda_max * 64); flatten to (H8*W8,)
    N_flat = torch.poisson(lambda_max.flatten() * 64, generator=generator).long()
    tot_sampled = int(N_flat.sum())

    if tot_sampled == 0:
        return lam.new_zeros(0, 2), 0, lam_max_global, 0

    # For each proposal, record its block index and derive block row/col
    block_ids = torch.repeat_interleave(
        torch.arange(H8 * W8, device=device), N_flat
    )  # (tot_sampled,)
    block_i = block_ids // W8  # (tot_sampled,)
    block_j = block_ids % W8

    # Sample uniform offsets within each 8×8 block
    offs_x = torch.rand(tot_sampled, dtype=torch.float64, device=device, generator=generator)
    offs_y = torch.rand(tot_sampled, dtype=torch.float64, device=device, generator=generator)
    xs = offs_x * 8 + block_j.double() * 8  # absolute x  (tot_sampled,)
    ys = offs_y * 8 + block_i.double() * 8  # absolute y

    xi = xs.long()
    yi = ys.long()

    # Thinning: accept with probability lam[pixel] / lambda_max[block]
    lam_x = lam[yi, xi]
    lam_block = lambda_max.flatten()[block_ids]
    u = torch.rand(tot_sampled, dtype=torch.float64, device=device, generator=generator)
    keep = u < (lam_x / lam_block)

    points_xy = torch.stack([xs[keep], ys[keep]], dim=1)
    return points_xy, int(keep.sum()), lam_max_global, tot_sampled


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
    - This implementation is vectorized over blocks.
    - Only the final point placement for active blocks is done in Python.
    - `pixel_area` is included in Λ_A here. If your lambda_grid already stores
      per-pixel mass instead of intensity density, set pixel_area=1.0.
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

    # Number of blocks along each axis
    nby = (H + bs - 1) // bs
    nbx = (W + bs - 1) // bs
    num_blocks = nby * nbx

    # Pad once so every block is bs x bs
    Hp = nby * bs
    Wp = nbx * bs
    if Hp != H or Wp != W:
        lam_pad = np.zeros((Hp, Wp), dtype=np.float64)
        lam_pad[:H, :W] = lam
    else:
        lam_pad = lam

    # Reshape into blocks: (nby, bs, nbx, bs) -> sum over inner block dims
    lam_blocks = lam_pad.reshape(nby, bs, nbx, bs).transpose(0, 2, 1, 3)
    block_sums = lam_blocks.sum(axis=(2, 3)) * pixel_area  # shape: (nby, nbx)

    # Activation probabilities
    p_active = 1.0 - np.exp(-block_sums)
    expected_blocks_active = float(p_active.sum())

    # Sample active blocks in one shot
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

    # Prepare output
    points_xy = np.empty((num_active, 2), dtype=np.float64)

    # Top-left pixel coordinates of active blocks
    x0s = active_bx * bs
    y0s = active_by * bs

    # Actual block sizes (important at image borders)
    wBs = np.minimum(bs, W - x0s)
    hBs = np.minimum(bs, H - y0s)

    if placement == "uniform":
        points_xy[:, 0] = x0s + rng.random(num_active) * wBs
        points_xy[:, 1] = y0s + rng.random(num_active) * hBs

    else:  # placement == "weighted"
        # Loop only over active blocks
        for i, (by, bx, x0, y0, wB, hB) in enumerate(
            zip(active_by, active_bx, x0s, y0s, wBs, hBs)
        ):
            block = lam[y0:y0 + hB, x0:x0 + wB]
            s = block.sum()

            # Fallback for degenerate border/padded cases
            if s <= 0:
                points_xy[i, 0] = x0 + rng.random() * wB
                points_xy[i, 1] = y0 + rng.random() * hB
                continue

            probs = (block.ravel() / s)
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
class PPPMarkVisualizationHook(Hook):
    """
    Visualization + analytics hook for marked Poisson Point Process models.

    Visualizes:
      - RGB image
      - predicted intensity λ(x)
      - GT intensity
      - predicted depth
      - predicted velocity

    Saves:
      - raw numpy arrays (pred + GT)
      - meta information
      - loss values
    """

    def __init__(
        self,
        interval=200,
        max_samples=1,
        img_mean=None,
        img_std=None,
        # Plot controls
        dpi: int = 90,
        figsize=(14, 34),
        # Optional fixed scales for better comparability across frames
        depth_vmin: float | None = None,
        depth_vmax: float | None = None,
        vel_vmin: float | None = None,
        vel_vmax: float | None = None,
        gamma=0.9,
        sim_block_size=8,
        sim_block_placement="weighted",
        point_size=28,
    ):
        self.interval = interval # this is for train. For test/val we display all (but filter using dataset stride)
        self.max_samples = max_samples

        self.img_mean = (
            np.array(img_mean, dtype=np.float32)[:, None, None]
            if img_mean is not None else None
        )
        self.img_std = (
            np.array(img_std, dtype=np.float32)[:, None, None]
            if img_std is not None else None
        )

        self.gamma=gamma
        self.sim_block_size = sim_block_size
        self.sim_block_placement = sim_block_placement
        self.point_size = point_size
        self.point_size = point_size

        self.dpi = int(dpi)
        self.figsize = tuple(figsize)

        self.depth_vmin = depth_vmin
        self.depth_vmax = depth_vmax
        self.vel_vmin = vel_vmin
        self.vel_vmax = vel_vmax

    def after_train_iter(self, runner, batch_idx, data_batch=None, outputs=None):
        if not is_main_process():
            return
        if runner.iter % self.interval != 0:
            return
        if not isinstance(data_batch, dict):
            return

        model = runner.model
        if is_model_wrapper(model):
            model = model.module
        was_training = model.training
        model.eval() # this switches the model into eval mode (at the end we reset it to training mode)

        velocity_mode = getattr(model.decode_head, "velocity_mode", "signed")
        if velocity_mode not in {"signed", "absolute"}:
            raise ValueError(
                f"velocity_mode must be 'signed' or 'absolute', got {velocity_mode!r}."
            )

        # ---------------------------------------------------------
        # Apply data preprocesso
        # ---------------------------------------------------------
        proc = model.data_preprocessor(data_batch)
        inputs = proc['inputs']
        data_samples = proc['data_samples']

        with torch.no_grad():
            feats = model.extract_feat(inputs)
            preds = model.decode_head(feats)   # (N,C,H,W)

        self.visualize_ppp_batch(
            runner=runner,
            inputs=inputs,
            preds=preds,
            data_samples=data_samples,
            outputs=outputs,
            phase="train",
            velocity_mode=velocity_mode,
        )

        if was_training:
            model.train()

    def after_test_iter(self, runner, batch_idx, data_batch=None, outputs=None):
        """
        Visualization during testing / evaluation.
        Runs for every test batch (or you can subsample via batch_idx).
        """
        if not is_main_process():
            return
        if runner.iter % self.interval != 0:
            return
        if not isinstance(data_batch, dict):
            return

        model = runner.model
        if is_model_wrapper(model):
            model = model.module
        model.eval()  # ensure eval mode (already true in test loop, but explicit is fine)

        velocity_mode = getattr(model.decode_head, "velocity_mode", "signed")
        if velocity_mode not in {"signed", "absolute"}:
            raise ValueError(
                f"velocity_mode must be 'signed' or 'absolute', got {velocity_mode!r}."
            )

        proc = model.data_preprocessor(data_batch)
        inputs = proc["inputs"]
        data_samples = proc["data_samples"]

        with torch.no_grad():
            feats = model.extract_feat(inputs)
            preds = model.decode_head(feats)

        self.visualize_ppp_batch(
            runner=runner,
            inputs=inputs,
            preds=preds,
            data_samples=data_samples,
            outputs=None,      # no losses during test
            phase="test",
            velocity_mode=velocity_mode,
        )

    def after_val_iter(self, runner, batch_idx, data_batch=None, outputs=None):
        """
        Visualization during validation.
        """
        if not is_main_process():
            return
        if runner.iter % self.interval != 0:
            return
        if not isinstance(data_batch, dict):
            return

        model = runner.model
        if is_model_wrapper(model):
            model = model.module
        model.eval()  # val loop already uses eval, explicit is fine

        velocity_mode = getattr(model.decode_head, "velocity_mode", "signed")
        if velocity_mode not in {"signed", "absolute"}:
            raise ValueError(
                f"velocity_mode must be 'signed' or 'absolute', got {velocity_mode!r}."
            )

        proc = model.data_preprocessor(data_batch)
        inputs = proc["inputs"]
        data_samples = proc["data_samples"]

        with torch.no_grad():
            feats = model.extract_feat(inputs)
            preds = model.decode_head(feats)

        self.visualize_ppp_batch(
            runner=runner,
            inputs=inputs,
            preds=preds,
            data_samples=data_samples,
            outputs=None,   # losses are not passed here either
            phase="val",
            velocity_mode=velocity_mode,
        )

    def _get_vis_dir(self, runner, phase: str) -> str:
        vis_dir = os.path.join(
            runner.log_dir,
            "vis_data",
            phase
        )
        os.makedirs(vis_dir, exist_ok=True)
        return vis_dir

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

    def _sanitize_for_filename(self, s: str) -> str:
        # Keep filenames safe on cluster FS.
        return "".join(c if c.isalnum() or c in "._-" else "_" for c in s)

    def visualize_ppp_batch(
        self,
        *,
        runner,
        inputs: torch.Tensor,              # (N, 3, H, W)
        preds: torch.Tensor,               # (N, C, H, W)
        data_samples: list,                # list[SegDataSample]
        outputs: dict | None = None,        # losses (train only)
        phase: str = "train",               # "train" | "test"
        velocity_mode: str = "signed",
    ):

        num_vis = min(self.max_samples, preds.size(0))
        device = inputs.device
        out_dir = self._get_vis_dir(runner, phase)

        # ---------------------------------------------------------
        # Loop over first few samples only
        # ---------------------------------------------------------
        for i in range(num_vis):

            pred = preds[i]
            gt = data_samples[i].gt_sem_seg.data.to(device)

            # ---------------------------------------------
            # Split channels
            # ---------------------------------------------
            pred_log_intensity = pred[0] # logits
            pred_lambda_theta = torch.exp(pred_log_intensity) # pred_intensity

            pred_depth = torch.exp(pred[1]) if pred.size(0) > 1 else None

            raw_velocity = None
            pred_velocity_channel = None

            if pred.size(0) == 4:
                raw_velocity = pred[3]
                pred_velocity_channel = 3
            elif pred.size(0) > 2:
                raw_velocity = pred[2]
                pred_velocity_channel = 2

            if raw_velocity is not None:
                if velocity_mode == "signed":
                    pred_velocity = raw_velocity ** 3
                    pred_velocity_label = "Pred vel [m/s]"
                elif velocity_mode == "absolute":
                    pred_velocity = torch.abs(raw_velocity)
                    pred_velocity_label = "Pred |vel| [m/s]"
                else:
                    raise RuntimeError(f"Unsupported velocity_mode={velocity_mode!r}")
            else:
                pred_velocity = None
                pred_velocity_label = "Pred vel [m/s]"

            gt_support = gt[0]
            gt_depth = gt[1] if gt.size(0) > 1 else None

            if gt.size(0) > 2:
                if velocity_mode == "signed":
                    gt_velocity = gt[2]
                elif velocity_mode == "absolute":
                    gt_velocity = torch.abs(gt[2])
                else:
                    raise RuntimeError(f"Unsupported velocity_mode={velocity_mode!r}")
            else:
                gt_velocity = None

            # ---------------------------------------------
            # Image (de-normalize)
            # ---------------------------------------------
            img = inputs[i].cpu().numpy()

            if self.img_mean is not None and self.img_std is not None:
                img = img * self.img_std + self.img_mean
                # If image is in [-1, 1], map it back to [0, 255] for visualization
                img = (img + 1.0) * 127.5

            img = img.clip(0, 255).astype(np.uint8)
            img = np.transpose(img, (1, 2, 0))

            # ---------------------------------------------
            # Convert to numpy
            # ---------------------------------------------
            pred_lambda_theta_np = pred_lambda_theta.cpu().numpy()
            gt_support_np = gt_support.cpu().numpy()

            depth_np = pred_depth.cpu().numpy() if pred_depth is not None else None
            vel_np = pred_velocity.cpu().numpy() if pred_velocity is not None else None

            gt_depth_np = gt_depth.cpu().numpy() if gt_depth is not None else None
            gt_vel_np = gt_velocity.cpu().numpy() if gt_velocity is not None else None

            # ---------------------------------------------
            # Get metrics - counts (Convert to numpy)
            # ---------------------------------------------

            # H, W = pred_lambda_theta_np.shape
            H, W = data_samples[i].metainfo["ori_shape"]
            pixel_area = H * W
            pred_lambda_mass = pred_lambda_theta_np / pixel_area      # match PPPLoss normalization
            expected_count = pred_lambda_mass.sum()

            gt_count = float(gt_support_np.sum())

            # Torch version
            pred_lambda_mass_torch = pred_lambda_theta.detach() / pixel_area

            # ---------------------------------------------
            # Meta information
            # ---------------------------------------------
            meta = data_samples[i].metainfo

            frame_idx = int(meta["frame_idx"])
            prefix = (
                f"iter_{runner.iter:07d}_"
                f"{phase}_"
                f"frame_{frame_idx:06d}_"
                f"s{i}"
            )

            video_name = meta.get("video_name", None)
            video_frame_idx = meta.get("video_frame_idx", None)
            global_idx = meta.get("global_idx", meta.get("frame_idx", None))
            sample_id = meta.get("sample_id", None)

            video_str = self._sanitize_for_filename(str(video_name)) if video_name else "unknown_video"
            vframe_str = f"{int(video_frame_idx):06d}" if video_frame_idx is not None else "unknown_frame"
            gidx_str = f"{int(global_idx):08d}" if global_idx is not None else "unknown_idx"

            # ---------------------------------------------
            # GT points + simulated point clouds
            # ---------------------------------------------
            gt_points_xy = extract_gt_points_from_mask(gt_support_np)

            # deterministic per sample if desired
            seed_base = int(meta.get("frame_idx", 0)) + 1000 * int(runner.iter)
            rng_thin = np.random.default_rng(seed_base + 11)
            rng_block = np.random.default_rng(seed_base + 29)

            # --- NumPy thinning timing ---
            t0 = time.perf_counter()
            sim1_points_xy, sim1_N, sim1_lambda_max, sim1_N_prop = sample_ppp_by_thinning(
                pred_lambda_mass,
                # pixel_area=1.0,
                rng=rng_thin,
            )
            t1 = time.perf_counter()
            sim1_time_ms = 1000.0 * (t1 - t0)

            print(sim1_N_prop)

            t2 = time.perf_counter()
            sim2_points_xy, sim2_stats = sample_points_block_bernoulli(
                pred_lambda_mass,
                block_size=self.sim_block_size,
                pixel_area=1.0,
                placement=self.sim_block_placement,
                rng=rng_block,
            )
            sim2_N = int(sim2_stats["num_points"])

            t3 = time.perf_counter()
            sim2_time_ms = 1000.0 * (t3 - t2)

            print(
                f"[PPP timing] frame={meta.get('frame_idx')} | "
                f"numpy={sim1_time_ms:.3f} ms | "
                f"block_bernoulli={sim2_time_ms:.3f} ms | "
                f"N_numpy={sim1_N} | N_block_bernoulli={sim2_N}"
            )

            # ---------------------------------------------
            # Save raw arrays
            # ---------------------------------------------
            np.save(os.path.join(out_dir, f"{prefix}_img.npy"), img)
            np.save(os.path.join(out_dir, f"{prefix}_pred_log_intensity.npy"), pred_log_intensity.detach().cpu().numpy())
            np.save(os.path.join(out_dir, f"{prefix}_pred_lambda_mass.npy"), pred_lambda_mass)
            np.save(os.path.join(out_dir, f"{prefix}_gt_support.npy"), gt_support_np)

            if depth_np is not None:
                np.save(os.path.join(out_dir, f"{prefix}_pred_depth.npy"), depth_np)
                np.save(os.path.join(out_dir, f"{prefix}_gt_depth.npy"), gt_depth_np)

            if vel_np is not None:
                np.save(os.path.join(out_dir, f"{prefix}_pred_velocity.npy"), vel_np)
                np.save(os.path.join(out_dir, f"{prefix}_gt_velocity.npy"), gt_vel_np)

            np.save(os.path.join(out_dir, f"{prefix}_gt_points_xy.npy"), gt_points_xy)
            np.save(os.path.join(out_dir, f"{prefix}_sim1_points_xy.npy"), sim1_points_xy)
            np.save(os.path.join(out_dir, f"{prefix}_sim2_points_xy.npy"), sim2_points_xy)

            # ---------------------------------------------
            # Save meta + losses
            # ---------------------------------------------
            meta_dump = {
                "iter": int(runner.iter),
                "frame_idx": int(meta["frame_idx"]),
                "sample_in_batch": int(i),
                "img_shape": meta.get("img_shape", None),
                "ori_shape": meta.get("ori_shape", None),
                "velocity_mode": velocity_mode,
                "pred_velocity_channel": (
                    int(pred_velocity_channel)
                    if pred_velocity_channel is not None
                    else None
                ),
                "losses": {
                    k: float(v) for k, v in (outputs or {}).items()
                    if torch.is_tensor(v)
                },
                "ppp_diagnostics": {
                    "H": int(H),
                    "W": int(W),
                    "img_area": int(pixel_area),
                    "expected_count": float(expected_count),
                    "gt_count": float(gt_count),
                    "count_error": float(expected_count - gt_count),
                },
                "sampling_diagnostics": {
                    "gt_num_points": int(len(gt_points_xy)),
                    "sim1_thinning": {
                        "num_points": int(sim1_N),
                        "lambda_max": float(sim1_lambda_max),
                        "num_proposals": int(sim1_N_prop),
                        "acceptance_rate": float(sim1_N / max(sim1_N_prop, 1)),
                    },
                    "sim2_block_bernoulli": {
                        **sim2_stats
                    }
                }
            }

            with open(os.path.join(out_dir, f"{prefix}_meta.json"), "w") as f:
                json.dump(meta_dump, f, indent=2)

            # ---------------------------------------------
            # Visualization
            # ---------------------------------------------
            fig, axs = plt.subplots(
                9,
                1,
                figsize=self.figsize,   # large panels
            )

            # remove ALL padding
            plt.subplots_adjust(
                left=0.01,
                right=0.99,
                top=0.97,
                bottom=0.01,
                hspace=0.08,
            )

            fig.suptitle(
                f"{phase} | iter {int(runner.iter)} | {video_str} | frame {vframe_str} | idx {gidx_str}",
                fontsize=14,
            )

            def _cbar_under(fig, ax, im, height=0.04, pad=0.02):
                cax = ax.inset_axes([0.0, -(height + pad), 1.0, height])
                cb = fig.colorbar(im, cax=cax, orientation="horizontal")
                cb.ax.tick_params(labelsize=10)
                return cb

            def _ylabel(ax, text):
                ax.text(
                    -0.015, 0.5, text,
                    transform=ax.transAxes,
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
            _ylabel(axs[row], "GT support")
            axs[row].axis("off")
            row += 1

            # 3) GT points overlay (thick green dots)
            self._draw_points_overlay(
                axs[row],
                img,
                gt_points_xy,
                # ylabel="GT points",
                n_text=f"N={len(gt_points_xy)}",
                color="lime",
                point_size=self.point_size,
                alpha=0.95,
            )
            _ylabel(axs[row], "GT points")
            row += 1

            # 4) Sim1: thinning PPP
            self._draw_points_overlay(
                axs[row],
                img,
                sim1_points_xy,
                # ylabel="Sim1 thinning",
                n_text=f"N={sim1_N}",
                color="red",
                point_size=self.point_size,
                alpha=0.90,
            )
            _ylabel(axs[row], "Sim1 thinning")
            row += 1

            # 5) Sim2: block/Bernoulli PPP
            self._draw_points_overlay(
                axs[row],
                img,
                sim2_points_xy,
                # ylabel="Sim2 block",
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

            # 7) Pred λ^gamma visualization
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
                _ylabel(axs[row], "Pred depth [m]")
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
                _ylabel(axs[row], pred_velocity_label)
            else:
                axs[row].axis("off")

            png_path = os.path.join(out_dir, f"{prefix}.png")

            fig.savefig(
                png_path,
                dpi=self.dpi,           # keeps file size reasonable
                bbox_inches="tight",
            )

            plt.close(fig)