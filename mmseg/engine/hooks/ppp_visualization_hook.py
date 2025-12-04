import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch.nn.functional as F
from mmengine.hooks import Hook
from mmengine.registry import HOOKS


THING_CLASSES = {24,25,26,27,28,31,32,33}


def extract_centers_from_instanceIds(instance_map):
    """Same logic as PPPHead."""
    centers = []
    unique_ids = np.unique(instance_map)

    for inst_id in unique_ids:
        if inst_id == 0:
            continue

        class_id = inst_id // 1000
        if class_id not in THING_CLASSES:
            continue

        ys, xs = np.where(instance_map == inst_id)
        if len(xs) == 0:
            continue

        cx = float(xs.mean())
        cy = float(ys.mean())
        centers.append((cx, cy))

    return centers


@HOOKS.register_module()
class PPPVisualizationHook(Hook):

    def __init__(self, interval=500, out_dir='work_dirs/ppp_vis'):
        self.interval = interval
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)

    def after_train_iter(self, runner, batch_idx, data_batch=None, outputs=None):

        if data_batch is None or batch_idx % self.interval != 0:
            return
        if not isinstance(data_batch, dict):
            return

        inputs = data_batch["inputs"]
        samples = data_batch["data_samples"]

        if isinstance(inputs, list):
            inputs = inputs[0]
        if inputs.dim() == 3:
            inputs = inputs.unsqueeze(0)

        device = next(runner.model.parameters()).device
        inputs = inputs.to(device)
        samples = [s.to(device) for s in samples]

        # --- Preprocess ---
        processed = runner.model.data_preprocessor(
            dict(inputs=inputs, data_samples=samples)
        )
        inputs_pp = processed["inputs"]
        samples_pp = processed["data_samples"]

        # --- Predict ---
        with torch.no_grad():
            preds = runner.model.predict(inputs_pp, samples_pp)

        pred = preds[0]
        seglog = pred.seg_logits
        if hasattr(seglog, "data"):
            seglog = seglog.data

        heatmap_resized = seglog.cpu()  # (1,H_res,W_res)

        # ----------------------------------------------------------------------
        # 1) Lade ORIGINAL-BILD und ORIGINAL-INSTANCE-IDs
        # ----------------------------------------------------------------------
        meta = samples_pp[0].metainfo

        img_path = meta["img_path"]       # original leftImg8bit
        label_path = meta["seg_map_path"] # original instanceIds

        # ---> Lade original RGB
        img_full = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img_full = cv2.cvtColor(img_full, cv2.COLOR_BGR2RGB)

        H_full, W_full = img_full.shape[:2]

        # ---> Lade original instanceIds
        inst_full = cv2.imread(label_path, cv2.IMREAD_UNCHANGED)
        centers_full = extract_centers_from_instanceIds(inst_full)

        # ----------------------------------------------------------------------
        # 2) Upsample PPP-Heatmap zurück auf ORIGINALGRÖSSE
        # ----------------------------------------------------------------------
        heat_up = F.interpolate(
            heatmap_resized.unsqueeze(0),  # (1,1,h,w)
            size=(H_full, W_full),
            mode="bilinear",
            align_corners=False
        )[0,0].numpy()

        # ----------------------------------------------------------------------
        # 3) Plotten
        # ----------------------------------------------------------------------
        out_path = os.path.join(self.out_dir, f"vis_full_{runner.iter}.png")
        plt.figure(figsize=(16,6))

        # Original mit Zentren
        plt.subplot(1,2,1)
        plt.imshow(img_full)
        plt.title(f"FULL IMAGE + GT CENTERS\n{os.path.basename(img_path)}")
        for (cx, cy) in centers_full:
            plt.plot(cx, cy, 'rx', markersize=10, markeredgewidth=2)
        plt.axis("off")

        # Heatmap + Zentren
        plt.subplot(1,2,2)
        plt.imshow(img_full, alpha=0.4)
        plt.imshow(heat_up, cmap='hot', alpha=0.6)
        plt.title("PPP Heatmap (upsampled) + GT")
        for (cx, cy) in centers_full:
            plt.plot(cx, cy, 'x', color='cyan', markersize=10, markeredgewidth=2)
        plt.axis("off")

        plt.tight_layout()
        plt.savefig(out_path)
        plt.close()

        runner.logger.info(f"[PPPVisualizationHook] Saved FULL visualization: {out_path}")
