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
        # Apply data preprocessor (IMPORTANT)
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

            axs[0, 2].imshow(gt_lam_np, cmap='gray')
            axs[0, 2].set_title("GT intensity")

            if depth_np is not None:
                axs[1, 0].imshow(depth_np, cmap='viridis')
                axs[1, 0].set_title("Predicted depth")
            else:
                axs[1, 0].axis("off")

            if vel_np is not None:
                axs[1, 1].imshow(vel_np, cmap='coolwarm')
                axs[1, 1].set_title("Predicted velocity")
            else:
                axs[1, 1].axis("off")

            axs[1, 2].axis("off")

            for ax in axs.flat:
                ax.axis("off")

            plt.tight_layout()
            plt.savefig(os.path.join(self.out_dir, f"{prefix}_vis.png"))
            plt.close()

        model.train()
