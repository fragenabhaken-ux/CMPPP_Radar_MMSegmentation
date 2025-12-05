import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch.nn.functional as F
from mmengine.hooks import Hook
from mmengine.registry import HOOKS


# THING_CLASSES = {24,25,26,27,28,31,32,33}
THING_CLASSES = {26,27,28}


def extract_centers_from_instanceIds(instance_map):
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

        centers.append((float(xs.mean()), float(ys.mean())))
    return centers


@HOOKS.register_module()
class PPPVisualizationHook(Hook):

    def __init__(self, interval=500, out_dir='work_dirs/ppp_vis'):
        self.interval = interval
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)

    def after_train_iter(self, runner, batch_idx, data_batch=None, outputs=None):

        if batch_idx % self.interval != 0:
            return
        if data_batch is None or not isinstance(data_batch, dict):
            return

        # --------------------------------------------
        # 1. Extract raw inputs + samples
        # --------------------------------------------
        inputs = data_batch["inputs"]
        samples = data_batch["data_samples"]

        if isinstance(inputs, list):
            inputs = inputs[0]
        if inputs.dim() == 3:
            inputs = inputs.unsqueeze(0)

        device = next(runner.model.parameters()).device
        inputs = inputs.to(device)
        samples = [s.to(device) for s in samples]

        # --------------------------------------------
        # 2. mmseg preprocessing: apply resize/crop/flip/normalize
        # --------------------------------------------
        processed = runner.model.data_preprocessor(
            dict(inputs=inputs, data_samples=samples)
        )
        inputs_pp = processed["inputs"]            # (N,3,Hc,Wc)
        samples_pp = processed["data_samples"]     # cropped + resized GT

        # --------------------------------------------
        # 3. Predict heatmap logits
        # --------------------------------------------
        with torch.no_grad():
            preds = runner.model.predict(inputs_pp, samples_pp)

        pred = preds[0]
        seglog = pred.seg_logits
        if hasattr(seglog, "data"):
            seglog = seglog.data

        heat_logits = seglog[0]    # shape (Hh,Wh) on decoder scale

        # --------------------------------------------
        # 4. Get cropped GT instance mask (aligned!)
        # --------------------------------------------
        inst_pp = samples_pp[0].gt_sem_seg.data[0].cpu().numpy()
        centers_pp = extract_centers_from_instanceIds(inst_pp)

        # --------------------------------------------
        # 5. Convert normalized image back to RGB for viz
        # --------------------------------------------
        img_pp = inputs_pp[0].cpu().numpy()  # (3,Hc,Wc)

        mean = np.array([123.675, 116.28, 103.53]).reshape(3,1,1)
        std  = np.array([58.395, 57.12, 57.375]).reshape(3,1,1)

        img_vis = (img_pp * std + mean).clip(0,255).astype(np.uint8)
        img_vis = np.transpose(img_vis, (1,2,0))  # → (Hc,Wc,3), RGB

        Hc, Wc = img_vis.shape[:2]

        # --------------------------------------------
        # 6. Resize decoder heatmap to cropped image size Hc,Wc
        # --------------------------------------------
        heat_up = F.interpolate(
            heat_logits.unsqueeze(0).unsqueeze(0),  # (1,1,Hh,Wh)
            size=(Hc, Wc),
            mode="bilinear",
            align_corners=False
        )[0,0].cpu().numpy()

        # --------------------------------------------
        # 7. File naming
        # --------------------------------------------
        meta = samples_pp[0].metainfo
        sample_id = os.path.splitext(os.path.basename(meta["img_path"]))[0]
        prefix = f"{runner.iter:08d}_{sample_id}"

        out_img_path = os.path.join(self.out_dir, f"{prefix}_vis.png")
        npy_path     = os.path.join(self.out_dir, f"{prefix}_heatmap.npy")
        txt_path     = os.path.join(self.out_dir, f"{prefix}_heatmap.txt")

        np.save(npy_path, heat_up)
        np.savetxt(txt_path, heat_up, fmt="%.6f")

        # --------------------------------------------
        # 8. Extract PPP losses
        # --------------------------------------------
        loss_lines = []

        if isinstance(outputs, dict):
            for key, value in outputs.items():
                if key.startswith("ppp_"):    # only PPP metrics
                    val = float(value.item() if isinstance(value, torch.Tensor) else value)
                    loss_lines.append(f"{key}: {val:.6f}")

        if not loss_lines:
            loss_lines = ["No PPP loss values found"]

        loss_text = "\n".join(loss_lines)

        # --------------------------------------------
        # 9. Plot three panels
        # --------------------------------------------
        plt.figure(figsize=(28, 8))

        # (1) Cropped GT image + centers
        plt.subplot(1,3,1)
        plt.imshow(img_vis)
        plt.title(f"Cropped GT centers (n={len(centers_pp)})")

        for (cx, cy) in centers_pp:
            plt.plot(cx, cy, "rx", markersize=10, markeredgewidth=2)
        plt.axis("off")

        # (2) Overlay: image + heatmap
        plt.subplot(1,3,2)
        plt.imshow(img_vis, alpha=0.4)
        plt.imshow(heat_up, cmap="hot", alpha=0.6)
        for (cx, cy) in centers_pp:
            plt.plot(cx, cy, "cx", markersize=10, markeredgewidth=2)

        plt.title("Heatmap overlay")
        plt.axis("off")

        # (3) Pure heatmap + loss text
        plt.subplot(1,3,3)
        plt.imshow(heat_up, cmap="hot")
        plt.title("Heatmap (pure)")
        plt.text(10,20, loss_text, color="white", fontsize=12,
                 bbox=dict(facecolor="black", alpha=0.5))
        plt.axis("off")

        plt.tight_layout()
        plt.savefig(out_img_path)
        plt.close()

        runner.logger.info(f"[PPPVisualizationHook] Saved: {out_img_path}")
        runner.logger.info(f"[PPPVisualizationHook] Saved heatmap arrays: {npy_path}, {txt_path}")
