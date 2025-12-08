import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch.nn.functional as F
from mmengine.hooks import Hook
from mmengine.registry import HOOKS


THING_CLASSES = {
            24,  # person
            25,  # rider
            26,  # car
            27,  # truck
            28,  # bus
            31,  # train
            32,  # motorcycle
            33   # bicycle
        }


# -----------------------------------------------------------
# Utility 1: Extract thing-instance centers
# -----------------------------------------------------------
def extract_centers(instance_map):
    centers = []
    for inst_id in np.unique(instance_map):
        if inst_id == 0:
            continue
        class_id = inst_id // 1000
        if class_id not in THING_CLASSES:
            continue
        ys, xs = np.where(instance_map == inst_id)
        if len(xs) > 0:
            centers.append((float(xs.mean()), float(ys.mean())))
    return centers


# -----------------------------------------------------------
# Utility 2: Un-normalize image
# -----------------------------------------------------------
def to_rgb(img_chw):
    mean = np.array([123.675, 116.28, 103.53]).reshape(3,1,1)
    std  = np.array([58.395, 57.12, 57.375]).reshape(3,1,1)
    img = (img_chw * std + mean).clip(0,255).astype(np.uint8)
    return np.transpose(img, (1,2,0))  # (H,W,3)


# -----------------------------------------------------------
# Utility 3: Upscale heatmap
# -----------------------------------------------------------
def upscale_heatmap(heat_logits, H, W):
    return F.interpolate(
        heat_logits[None, None],
        size=(H, W),
        mode="bilinear",
        align_corners=False
    )[0,0].cpu().numpy()


# -----------------------------------------------------------
# Utility 4: Plotting
# -----------------------------------------------------------
def plot_visualization(img, lambd, centers, loss_text, save_path):
    plt.figure(figsize=(28, 8))

    # (1) GT centers
    plt.subplot(1,3,1)
    plt.imshow(img)
    plt.title(f"GT centers (n={len(centers)})")
    for (cx, cy) in centers:
        plt.plot(cx, cy, "rx", markersize=10, markeredgewidth=2)
    plt.axis("off")

    # (2) Overlay
    plt.subplot(1,3,2)
    plt.imshow(img, alpha=0.4)
    plt.imshow(lambd, cmap="hot", alpha=0.6)
    for (cx, cy) in centers:
        plt.plot(cx, cy, "cx", markersize=10, markeredgewidth=2)
    plt.title("Intensity overlay")
    plt.axis("off")

    # (3) Pure heatmap
    plt.subplot(1,3,3)
    plt.imshow(lambd, cmap="hot")
    plt.title("Intensity")
    plt.text(10,20, loss_text, color="white", fontsize=12,
            bbox=dict(facecolor="black", alpha=0.5))
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


# =====================================================================
# CLEAN VISUALIZER HOOK
# =====================================================================
@HOOKS.register_module()
class PPPVisualizationHook(Hook):

    def __init__(self, interval=500, out_dir="work_dirs/ppp_vis"):
        self.interval = interval
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)

    def after_train_iter(self, runner, batch_idx, data_batch=None, outputs=None):

        if batch_idx % self.interval != 0:
            return
        if not isinstance(data_batch, dict):
            return

        # -----------------------------------------------------------
        # Extract inputs
        # -----------------------------------------------------------
        inputs = data_batch["inputs"]
        samples = data_batch["data_samples"]

        if isinstance(inputs, list):
            inputs = inputs[0]
        if inputs.dim() == 3:
            inputs = inputs[0][None]

        device = next(runner.model.parameters()).device
        inputs = inputs.to(device)
        samples = [s.to(device) for s in samples]

        # Run data preprocessor (crop + resize + norm)
        proc = runner.model.data_preprocessor(
            dict(inputs=inputs, data_samples=samples)
        )
        inputs_pp = proc["inputs"]
        samples_pp = proc["data_samples"]

        # -----------------------------------------------------------
        # Heatmap (SECOND FORWARD)
        # -----------------------------------------------------------
        with torch.no_grad():
            feats = runner.model.extract_feat(inputs_pp)
            seg_logits = runner.model.decode_head(feats)
        heat_logits = seg_logits[0,0].detach().cpu()

        # -----------------------------------------------------------
        # GT instance mask + centers
        # -----------------------------------------------------------
        inst_map = samples_pp[0].gt_sem_seg.data[0].cpu().numpy()
        centers = extract_centers(inst_map)

        # -----------------------------------------------------------
        # Recover visualization image
        # -----------------------------------------------------------
        img_pp = inputs_pp[0].cpu().numpy()
        img_rgb = to_rgb(img_pp)
        Hc, Wc = img_rgb.shape[:2]

        # -----------------------------------------------------------
        # Upscale heatmap
        # -----------------------------------------------------------
        heat_up = upscale_heatmap(heat_logits, Hc, Wc)

        # --- Convert logits f(x) → PPP intensity lambda(x)
        pixel_area = Hc * Wc
        lambda_map = np.exp(heat_up) / pixel_area


        # -----------------------------------------------------------
        # Filename setup
        # -----------------------------------------------------------
        meta = samples_pp[0].metainfo
        sample_id = os.path.splitext(os.path.basename(meta["img_path"]))[0]
        prefix = f"{runner.iter:08d}_{sample_id}"

        vis_path = os.path.join(self.out_dir, f"{prefix}_vis.png")
        
        # ---- 1) Save heatmap and intensity ----
        np.save(os.path.join(self.out_dir, f"{prefix}_heatmap.npy"), heat_up)
        np.save(os.path.join(self.out_dir, f"{prefix}_lambda.npy"), lambda_map)

        # ---- 2) Save cropped RGB image ----
        np.save(os.path.join(self.out_dir, f"{prefix}_img.npy"), img_rgb)

        # ---- 3) Save GT instance map ----
        np.save(os.path.join(self.out_dir, f"{prefix}_gt_inst.npy"), inst_map)

        # ---- 4) Save centers ----
        np.save(os.path.join(self.out_dir, f"{prefix}_centers.npy"), np.array(centers, dtype=np.float32))

        # ---- _gt_inst.npy Data Instance numpy mask
        np.save(os.path.join(self.out_dir, f"{prefix}_gt_inst.npy"), inst_map)

        # ---- 5) Save meta-information ----
        meta_dict = {
            "iter": runner.iter,
            "img_path": meta["img_path"],
            "crop_shape": [int(Hc), int(Wc)],
            "original_shape": list(meta.get("ori_shape", [])),
            "scale_factor": list(meta.get("scale_factor", [])),
            "crop_box": list(meta.get("img_crop_bbox", [])),
        }
        import json
        with open(os.path.join(self.out_dir, f"{prefix}_meta.json"), "w") as f:
            json.dump(meta_dict, f, indent=2)


        # -----------------------------------------------------------
        # Loss text
        # -----------------------------------------------------------
        if outputs and "decode.loss_ppp" in outputs:
            loss_val = outputs["decode.loss_ppp"].item()
            loss_text = f"decode.loss_ppp: {loss_val:.6f}"
        else:
            loss_text = "(no PPP loss found)"

        # -----------------------------------------------------------
        # Save visualization plot
        # -----------------------------------------------------------
        plot_visualization(
            img=img_rgb,
            lambd=lambda_map,
            centers=centers,
            loss_text=loss_text,
            save_path=vis_path
        )

        runner.logger.info(f"[PPPVisualizationHook] saved → {vis_path}")
