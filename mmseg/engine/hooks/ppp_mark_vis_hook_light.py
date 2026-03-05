import os
import json
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


@HOOKS.register_module()
class PPPMarkVisHookLight(Hook):
    """
    Lightweight visualization hook for PPP marked models.

    Writes ONLY:
      - one PNG per visualized sample
      - one JSON (meta + small diagnostics) per visualized sample

    Notes:
      - Runs on main process only.
      - Adds an extra forward pass for visualization (extract_feat + decode_head),
        since training outputs do not reliably expose dense predictions.
      - Uses tight saving (bbox_inches="tight") and tiny padding to reduce whitespace.
    """

    def __init__(
        self,
        interval: int = 500,
        max_samples: int = 1,
        out_subdir: str = "vis_light",
        img_mean=None,
        img_std=None,
        # Plot controls
        dpi: int = 200,
        pad_inches: float = 0.02,
        figsize=(12, 7),
        # Optional fixed scales for better comparability across frames
        depth_vmin: float | None = None,
        depth_vmax: float | None = None,
        vel_vmin: float | None = None,
        vel_vmax: float | None = None,
    ):
        self.interval = int(interval)
        self.max_samples = int(max_samples)
        self.out_subdir = str(out_subdir)

        self.dpi = int(dpi)
        self.pad_inches = float(pad_inches)
        self.figsize = tuple(figsize)

        self.depth_vmin = depth_vmin
        self.depth_vmax = depth_vmax
        self.vel_vmin = vel_vmin
        self.vel_vmax = vel_vmax

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
    def _should_run(self, runner, batch_idx, data_batch, phase: str) -> bool:
        if not is_main_process():
            return False
        if not isinstance(data_batch, dict):
            return False

        # Train: gate by global iteration
        if phase == "train":
            return (runner.iter % self.interval) == 0

        # Val/Test: gate by batch_idx (more intuitive than runner.iter)
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
        img = np.clip(img, 0, 255).astype(np.uint8)
        return np.transpose(img, (1, 2, 0))

    def _sanitize_for_filename(self, s: str) -> str:
        # Keep filenames safe on cluster FS.
        return "".join(c if c.isalnum() or c in "._-" else "_" for c in s)

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
        num_vis = min(self.max_samples, preds.size(0))
        device = inputs.device
        out_dir = self._get_vis_dir(runner, phase)

        for i in range(num_vis):
            pred = preds[i]  # (C,H,W)
            gt = data_samples[i].gt_sem_seg.data.to(device)  # (C_gt,H,W)

            # --- Split channels (assumes channel order: log_intensity, depth, velocity) ---
            pred_log_intensity = pred[0]
            pred_lambda = torch.exp(pred_log_intensity)

            pred_depth = pred[1] if pred.size(0) > 1 else None
            pred_vel = pred[2] if pred.size(0) > 2 else None

            gt_support = gt[0]

            # --- Image ---
            img = self._denorm_img(inputs[i].detach().cpu().numpy())

            # --- Metainfo (now includes video_name + video_frame_idx) ---
            meta = data_samples[i].metainfo

            video_name = meta.get("video_name", None)
            video_frame_idx = meta.get("video_frame_idx", None)
            global_idx = meta.get("global_idx", meta.get("frame_idx", None))
            sample_id = meta.get("sample_id", None)

            video_str = self._sanitize_for_filename(str(video_name)) if video_name else "unknown_video"
            vframe_str = f"{int(video_frame_idx):06d}" if video_frame_idx is not None else "unknown_frame"
            gidx_str = f"{int(global_idx):08d}" if global_idx is not None else "unknown_idx"

            # --- PPP count diagnostics (match PPPLoss-style normalization) ---
            ori_shape = meta.get("ori_shape", None)
            if isinstance(ori_shape, (tuple, list)) and len(ori_shape) >= 2:
                H, W = int(ori_shape[0]), int(ori_shape[1])
            else:
                H, W = int(pred_lambda.shape[-2]), int(pred_lambda.shape[-1])

            pixel_area = float(H * W)
            pred_lambda_mass = (pred_lambda / pixel_area).detach().cpu().numpy()
            expected_count = float(pred_lambda_mass.sum())
            gt_count = float(gt_support.detach().cpu().numpy().sum())

            # --- filenames / prefix ---
            if phase == "train":
                prefix = (
                    f"iter_{runner.iter:07d}_{phase}_"
                    f"{video_str}_f{vframe_str}_idx{gidx_str}_s{i}"
                )
            else:
                bidx = int(batch_idx) if batch_idx is not None else -1
                prefix = (
                    f"epoch_{runner.epoch:04d}_{phase}_b{bidx:05d}_"
                    f"{video_str}_f{vframe_str}_idx{gidx_str}_s{i}"
                )

            # --- JSON (small + useful) ---
            json_dump = {
                "phase": phase,
                "iter": int(runner.iter),
                "epoch": int(getattr(runner, "epoch", 0)),
                "batch_idx": None if batch_idx is None else int(batch_idx),
                "sample_in_batch": int(i),
                "sample_id": sample_id,
                "global_idx": None if global_idx is None else int(global_idx),
                "video_name": video_name,
                "video_frame_idx": None if video_frame_idx is None else int(video_frame_idx),
                "img_shape": meta.get("img_shape", None),
                "ori_shape": meta.get("ori_shape", None),
                "ppp_diagnostics": {
                    "H": int(H),
                    "W": int(W),
                    "img_area": int(H * W),
                    "expected_count": expected_count,
                    "gt_count": gt_count,
                    "count_error": expected_count - gt_count,
                },
                "losses": (
                    {
                        k: float(v)
                        for k, v in (outputs or {}).items()
                        if torch.is_tensor(v) and v.numel() == 1
                    }
                    if phase == "train"
                    else {}
                ),
            }

            with open(os.path.join(out_dir, f"{prefix}.json"), "w") as f:
                json.dump(json_dump, f, indent=2)

            # --- Plot (tight, low whitespace) ---
            fig, axs = plt.subplots(2, 3, figsize=self.figsize, constrained_layout=True)

            fig.suptitle(
                f"{phase} | {video_str} | frame {vframe_str} | idx {gidx_str}",
                fontsize=11,
            )

            # 1) Image
            axs[0, 0].imshow(img)
            axs[0, 0].set_title("Image", fontsize=10)
            axs[0, 0].axis("off")

            # 2) Pred intensity (mass)
            im0 = axs[0, 1].imshow(pred_lambda_mass, cmap="hot")
            axs[0, 1].set_title("Pred λ mass", fontsize=10)
            axs[0, 1].axis("off")
            axs[0, 1].text(
                0.02,
                0.98,
                f"E[N]={expected_count:.2f}\nGT={gt_count:.0f}\nΔ={expected_count-gt_count:+.2f}",
                transform=axs[0, 1].transAxes,
                fontsize=9,
                color="white",
                va="top",
                ha="left",
                bbox=dict(facecolor="black", alpha=0.45, pad=2),
            )
            fig.colorbar(im0, ax=axs[0, 1], fraction=0.045, pad=0.01)

            # 3) GT support
            axs[0, 2].imshow(gt_support.detach().cpu().numpy(), cmap="gray")
            axs[0, 2].set_title("GT support", fontsize=10)
            axs[0, 2].axis("off")

            # 4) Depth
            if pred_depth is not None:
                depth_np = pred_depth.detach().cpu().numpy()
                im1 = axs[1, 0].imshow(
                    depth_np,
                    cmap="viridis",
                    vmin=self.depth_vmin,
                    vmax=self.depth_vmax,
                )
                axs[1, 0].set_title("Pred depth", fontsize=10)
                axs[1, 0].axis("off")
                fig.colorbar(im1, ax=axs[1, 0], fraction=0.045, pad=0.01)
            else:
                axs[1, 0].axis("off")

            # 5) Velocity
            if pred_vel is not None:
                vel_np = pred_vel.detach().cpu().numpy()
                im2 = axs[1, 1].imshow(
                    vel_np,
                    cmap="coolwarm",
                    vmin=self.vel_vmin,
                    vmax=self.vel_vmax,
                )
                axs[1, 1].set_title("Pred vel", fontsize=10)
                axs[1, 1].axis("off")
                fig.colorbar(im2, ax=axs[1, 1], fraction=0.045, pad=0.01)
            else:
                axs[1, 1].axis("off")

            axs[1, 2].axis("off")

            # Save tightly
            png_path = os.path.join(out_dir, f"{prefix}.png")
            fig.savefig(
                png_path,
                dpi=self.dpi,
                bbox_inches="tight",
                pad_inches=self.pad_inches,
            )
            plt.close(fig)