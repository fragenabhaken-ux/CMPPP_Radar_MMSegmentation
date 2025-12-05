import os
import torch
import numpy as np
import cv2
import torch.nn.functional as F
from mmengine.hooks import Hook
from mmengine.registry import HOOKS


@HOOKS.register_module()
class PPPDebugHook(Hook):

    def __init__(self, interval=200, out_dir='work_dirs/debug_ppp'):
        self.interval = interval
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)

        # Cityscapes "thing" classes
        self.thing_classes = {24, 25, 26, 27, 28, 31, 32, 33}
        # self.thing_classes = {26, 27, 28}

    # -------------------------------------------------------------------------
    # Center extraction (same logic as PPPHead.extract_centers_from_label)
    # -------------------------------------------------------------------------
    def extract_centers(self, inst_mask):
        centers = []
        unique_ids = np.unique(inst_mask)

        for inst_id in unique_ids:
            if inst_id == 0:
                continue

            class_id = inst_id // 1000
            if class_id not in self.thing_classes:
                continue

            ys, xs = np.where(inst_mask == inst_id)
            if len(xs) == 0:
                continue

            cx, cy = xs.mean(), ys.mean()
            centers.append([cx, cy])

        return np.array(centers, dtype=np.float32)

    # -------------------------------------------------------------------------
    # Convert centers to downsampled mask (PPPHead.centers_to_mask equivalent)
    # -------------------------------------------------------------------------
    def make_center_mask(self, centers, H_in, W_in, H_out, W_out):
        mask = np.zeros((H_out, W_out), dtype=np.float32)

        if centers.shape[0] == 0:
            return mask

        for cx, cy in centers:
            cx_d = int(round(cx * (W_out / W_in)))
            cy_d = int(round(cy * (H_out / H_in)))
            if 0 <= cx_d < W_out and 0 <= cy_d < H_out:
                mask[cy_d, cx_d] = 1.0

        return mask

    # -------------------------------------------------------------------------
    # Main Debug Hook
    # -------------------------------------------------------------------------
    def after_train_iter(self, runner, batch_idx=None, data_batch=None, outputs=None):
        cur_iter = runner.iter
        if cur_iter % self.interval != 0:
            return

        # ------------------------------------------------------------------
        # 1) Use REAL mmseg preprocessor (handles dtype + mean/std + device)
        # ------------------------------------------------------------------
        processed = runner.model.data_preprocessor(data_batch, training=True)
        imgs_tensor = processed['inputs']           # (N,3,H,W) float32 normalized
        samples = processed['data_samples']         # SegDataSamples
        device = imgs_tensor.device

        # ------------------------------------------------------------------
        # 2) Forward through backbone + decoder
        # ------------------------------------------------------------------
        with torch.no_grad():
            feats = runner.model.extract_feat(imgs_tensor)
            seg_logits = runner.model.decode_head.forward(feats)  # (N,1,h,w)

        seg_logits_np = seg_logits.detach().cpu().numpy()[:, 0]     # (N,h,w)
        imgs_np = imgs_tensor.detach().cpu().numpy()                 # normalized image

        # Undo normalization for visualization
        mean = np.array([123.675, 116.28, 103.53]).reshape(1, 3, 1, 1)
        std = np.array([58.395, 57.12, 57.375]).reshape(1, 3, 1, 1)
        imgs_vis = imgs_np * std + mean    # (N,3,H,W)
        imgs_vis = np.clip(imgs_vis, 0, 255).astype(np.uint8)

        for idx, sample in enumerate(samples):

            # -------------------------------
            # 1. Original image
            # -------------------------------
            img = imgs_vis[idx].transpose(1, 2, 0)
            img_bgr = img[..., ::-1]
            H_in, W_in = img_bgr.shape[:2]

            # -------------------------------
            # 2. Instance label
            # -------------------------------
            inst_mask = sample.gt_sem_seg.data.cpu().numpy()[0].astype(np.float32)

            lab_norm = 255 * (inst_mask - inst_mask.min()) / (inst_mask.ptp() + 1e-6)
            lab_norm = lab_norm.astype(np.uint8)
            label_color = cv2.applyColorMap(lab_norm, cv2.COLORMAP_PARULA)

            # -------------------------------
            # 3. PPP center mask
            # -------------------------------
            centers = self.extract_centers(inst_mask)   # [K,2]

            H_out, W_out = seg_logits_np.shape[1:]
            center_mask = self.make_center_mask(centers, H_in, W_in, H_out, W_out)

            center_mask_color = cv2.applyColorMap(
                (center_mask * 255).astype(np.uint8),
                cv2.COLORMAP_BONE
            )

            # -------------------------------
            # 4. Predicted heatmap
            # -------------------------------
            heat = seg_logits_np[idx]           # (h,w)
            heat = F.softplus(torch.from_numpy(heat)).numpy()

            heat_norm = 255 * (heat - heat.min()) / (heat.ptp() + 1e-6)
            heat_norm = heat_norm.astype(np.uint8)
            heat_color = cv2.applyColorMap(heat_norm, cv2.COLORMAP_JET)

            # -------------------------------
            # Resize all maps to match input resolution
            # -------------------------------
            label_color_r = cv2.resize(label_color, (W_in, H_in))
            center_mask_r = cv2.resize(center_mask_color, (W_in, H_in))
            heat_color_r = cv2.resize(heat_color, (W_in, H_in))

            # -------------------------------
            # Stack 4 panels vertically
            # -------------------------------
            combined = np.vstack([
                img_bgr,
                label_color_r,
                center_mask_r,
                heat_color_r
            ])

            out_file = f"{self.out_dir}/iter{cur_iter:05d}_sample{idx}_combined.png"
            cv2.imwrite(out_file, combined)
