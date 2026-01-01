import os
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F

from mmengine.hooks import Hook
from mmseg.registry import HOOKS


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
        out_dir='work_dirs/ppp_mark_vis',
        max_samples=1,
    ):
        self.interval = interval
        self.out_dir = out_dir
        self.max_samples = max_samples
        os.makedirs(out_dir, exist_ok=True)

    def after_train_iter(self, runner, batch_idx, data_batch=None, outputs=None):

        if (runner.iter + 1) % self.interval != 0:
            return
        if not isinstance(data_batch, dict):
            return

        model = runner.model
        model.eval()

        # ---------------------------------------------------------
        # Apply data preprocesso
        # ---------------------------------------------------------
        proc = model.data_preprocessor(data_batch)
        inputs = proc['inputs']
        data_samples = proc['data_samples']

        device = inputs.device

        with torch.no_grad():
            feats = model.extract_feat(inputs)
            preds = model.decode_head(feats)   # (N,C,H,W)

        # ---------------------------------------------------------
        # Loop over first few samples only
        # ---------------------------------------------------------
        num_vis = min(self.max_samples, preds.size(0))

        for i in range(num_vis):

            pred = preds[i]
            gt = data_samples[i].gt_sem_seg.data.to(device)

            # ---------------------------------------------
            # Split channels
            # ---------------------------------------------
            pred_intensity_logits = pred[0]
            pred_intensity = torch.exp(pred_intensity_logits)

            pred_depth = pred[1] if pred.size(0) > 1 else None
            pred_velocity = pred[2] if pred.size(0) > 2 else None

            gt_intensity = gt[0]
            gt_depth = gt[1] if gt.size(0) > 1 else None
            gt_velocity = gt[2] if gt.size(0) > 2 else None

            # ---------------------------------------------
            # Image (de-normalize)
            # ---------------------------------------------
            img = inputs[i].cpu().numpy()
            mean = np.array([123.675, 116.28, 103.53])[:, None, None]
            std = np.array([58.395, 57.12, 57.375])[:, None, None]
            img = (img * std + mean).clip(0, 255).astype(np.uint8)
            img = np.transpose(img, (1, 2, 0))

            # ---------------------------------------------
            # Convert to numpy
            # ---------------------------------------------
            lam_np = pred_intensity.cpu().numpy()
            gt_lam_np = gt_intensity.cpu().numpy()

            depth_np = pred_depth.cpu().numpy() if pred_depth is not None else None
            vel_np = pred_velocity.cpu().numpy() if pred_velocity is not None else None

            gt_depth_np = gt_depth.cpu().numpy() if gt_depth is not None else None
            gt_vel_np = gt_velocity.cpu().numpy() if gt_velocity is not None else None

            # ---------------------------------------------
            # Get metrics - counts (Convert to numpy)
            # ---------------------------------------------

            H, W = lam_np.shape
            pixel_area = H * W
            lam_norm = lam_np / pixel_area      # match PPPLoss normalization
            expected_count = lam_norm.sum()

            gt_count = float(gt_lam_np.sum())


            # ---------------------------------------------
            # Meta information
            # ---------------------------------------------
            meta = data_samples[i].metainfo
            sample_id = meta.get('img_path', f'sample_{i}')
            sample_id = os.path.splitext(os.path.basename(sample_id))[0]
            prefix = f"iter_{runner.iter:07d}_{sample_id}"

            # ---------------------------------------------
            # Save raw arrays
            # ---------------------------------------------
            np.save(os.path.join(self.out_dir, f"{prefix}_img.npy"), img)
            np.save(os.path.join(self.out_dir, f"{prefix}_lambda_pred.npy"), lam_np)
            np.save(os.path.join(self.out_dir, f"{prefix}_lambda_gt.npy"), gt_lam_np)

            if depth_np is not None:
                np.save(os.path.join(self.out_dir, f"{prefix}_depth_pred.npy"), depth_np)
                np.save(os.path.join(self.out_dir, f"{prefix}_depth_gt.npy"), gt_depth_np)

            if vel_np is not None:
                np.save(os.path.join(self.out_dir, f"{prefix}_velocity_pred.npy"), vel_np)
                np.save(os.path.join(self.out_dir, f"{prefix}_velocity_gt.npy"), gt_vel_np)

            # ---------------------------------------------
            # Save meta + losses
            # ---------------------------------------------
            meta_dump = {
                "iter": runner.iter,
                "img_path": meta.get("img_path", ""),
                "img_shape": meta.get("img_shape", None),
                "ori_shape": meta.get("ori_shape", None),
                "scale_factor": meta.get("scale_factor", None),
                "crop_bbox": meta.get("img_crop_bbox", None),
                "losses": {
                    k: float(v) for k, v in (outputs or {}).items()
                    if torch.is_tensor(v)
                },
                "ppp_diagnostics": {
                    "expected_count_full": float(expected_count),
                    "gt_count_full": int(gt_count),
                    "count_error": float(expected_count - gt_count)
                }
            }

            with open(os.path.join(self.out_dir, f"{prefix}_meta.json"), "w") as f:
                json.dump(meta_dump, f, indent=2)

            # ---------------------------------------------
            # Visualization
            # ---------------------------------------------
            fig, axs = plt.subplots(2, 3, figsize=(18, 10))

            axs[0, 0].imshow(img)
            axs[0, 0].set_title("Image")

            axs[0, 1].imshow(lam_np, cmap='hot')
            axs[0, 1].set_title("Predicted λ(x)")

            axs[0, 1].text(
                0.02, 0.95,
                "PPP Count Diagnostic\n"
                f"∫λ dx = {expected_count:.2f}\n"
                f"GT = {gt_count}\n"
                f"Δ = {expected_count - gt_count:+.2f}",
                transform=axs[0, 1].transAxes,
                fontsize=11,
                color="white",
                verticalalignment="top",
                bbox=dict(facecolor="black", alpha=0.5, pad=4)
            )

            axs[0, 2].imshow(gt_lam_np, cmap='gray')
            axs[0, 2].set_title("GT intensity")

            if depth_np is not None:
                im_depth = axs[1, 0].imshow(depth_np, cmap='viridis')
                # # uncomment to fix the scale image to image
                # im_depth = axs[1, 0].imshow(
                #     depth_np,
                #     cmap='viridis',
                #     vmin=0.0,
                #     vmax=80.0   # or max sensor range
                # )
                axs[1, 0].set_title("Predicted depth")
                cbar = fig.colorbar(im_depth, ax=axs[1, 0], fraction=0.046, pad=0.04)
                cbar.set_label("Depth (meters)")
            else:
                axs[1, 0].axis("off")

            if vel_np is not None:
                # coolwarm :Blue - negative values, White / light - values near 0, Red - positive values
                # Note: matplotlip autoscales the scale (so color != a certail vel value)
                im_vel = axs[1, 1].imshow(vel_np, cmap='coolwarm')
                # # uncomment to fix the scale image to image
                # im_vel = axs[1, 1].imshow(
                #     vel_np,
                #     cmap='coolwarm',
                #     vmin=-20.0,
                #     vmax=20.0   # doppler range
                # )
                axs[1, 1].set_title("Predicted velocity")
                cbar = fig.colorbar(im_vel, ax=axs[1, 1], fraction=0.046, pad=0.04)
                cbar.set_label("Velocity (m/s)")
            else:
                axs[1, 1].axis("off")

            axs[1, 2].axis("off")

            for ax in axs.flat:
                ax.axis("off")

            plt.tight_layout()
            plt.savefig(os.path.join(self.out_dir, f"{prefix}_vis.png"))
            plt.close()

        model.train()
