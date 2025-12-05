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

        # self.thing_classes = {24, 25, 26, 27, 28, 31, 32, 33}
        self.thing_classes = {26, 27, 28}

    # -----------------------------------------------------------
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
            centers.append([xs.mean(), ys.mean()])

        return np.array(centers, dtype=np.float32)

    # -----------------------------------------------------------
    def make_center_mask(self, centers, H_in, W_in, H_out, W_out):
        mask = np.zeros((H_out, W_out), dtype=np.float32)
        if len(centers) == 0:
            return mask

        for cx, cy in centers:
            cx_d = int(round(cx * (W_out / W_in)))
            cy_d = int(round(cy * (H_out / H_in)))
            if 0 <= cx_d < W_out and 0 <= cy_d < H_out:
                mask[cy_d, cx_d] = 1.0

        return mask

    # -----------------------------------------------------------
    def after_train_iter(self, runner, batch_idx=None, data_batch=None, outputs=None):

        cur_iter = runner.iter
        if cur_iter % self.interval != 0:
            return

        processed = runner.model.data_preprocessor(data_batch, training=True)
        imgs_tensor = processed['inputs']
        samples = processed['data_samples']

        # forward model
        with torch.no_grad():
            feats = runner.model.extract_feat(imgs_tensor)
            seg_logits = runner.model.decode_head.forward(feats)

        seg_logits_np = seg_logits.detach().cpu().numpy()[:, 0]
        imgs_np = imgs_tensor.detach().cpu().numpy()

        # Undo normalization
        mean = np.array([123.675, 116.28, 103.53]).reshape(1, 3, 1, 1)
        std = np.array([58.395, 57.12, 57.375]).reshape(1, 3, 1, 1)
        imgs_vis = np.clip(imgs_np * std + mean, 0, 255).astype(np.uint8)

        # =====================================================
        # Process each sample
        # =====================================================
        for idx, sample in enumerate(samples):

            # ---- 1. Sample ID ----
            sample_id = os.path.splitext(
                os.path.basename(sample.metainfo['img_path'])
            )[0]

            prefix = f"{self.out_dir}/iter{cur_iter:05d}_{idx}_{sample_id}"

            # ---- 2. Original image ----
            img = imgs_vis[idx].transpose(1, 2, 0)
            img_bgr = img[..., ::-1]
            H_in, W_in = img_bgr.shape[:2]

            # ---- 3. instance mask ----
            inst_mask = sample.gt_sem_seg.data.cpu().numpy()[0].astype(np.float32)

            # colorized label
            lab_norm = 255 * (inst_mask - inst_mask.min()) / (inst_mask.ptp() + 1e-6)
            label_color = cv2.applyColorMap(lab_norm.astype(np.uint8), cv2.COLORMAP_PARULA)

            # ---- 4. centers + center mask ----
            centers = self.extract_centers(inst_mask)
            H_out, W_out = seg_logits_np.shape[1:]
            center_mask = self.make_center_mask(centers, H_in, W_in, H_out, W_out)

            center_mask_color = cv2.applyColorMap(
                (center_mask * 255).astype(np.uint8),
                cv2.COLORMAP_BONE
            )

            # ---- 5. PPP heatmap ----
            heat = F.softplus(torch.from_numpy(seg_logits_np[idx])).numpy()
            heat_norm = 255 * (heat - heat.min()) / (heat.ptp() + 1e-6)
            heat_color = cv2.applyColorMap(heat_norm.astype(np.uint8), cv2.COLORMAP_JET)

            # resize overlays to input img size
            label_color_r = cv2.resize(label_color, (W_in, H_in))
            center_mask_r = cv2.resize(center_mask_color, (W_in, H_in))
            heat_color_r = cv2.resize(heat_color, (W_in, H_in))

            # ---- 6. Stack 4 panels ----
            combined = np.vstack([
                img_bgr,
                label_color_r,
                center_mask_r,
                heat_color_r
            ])

            # ---- 7. Save combined image ----
            cv2.imwrite(prefix + "_combined.png", combined)

            # ---- 8. Save heatmap arrays ----
            np.save(prefix + "_heatmap.npy", heat)
            np.savetxt(prefix + "_heatmap.txt", heat, fmt="%.6f")

            # ---- 9. Save metadata text ----
            info_text = [
                f"Iteration: {cur_iter}",
                f"Sample: {sample_id}",
                f"GT centers: {len(centers)}",
                f"Input shape: {H_in} x {W_in}",
                f"Heatmap shape: {heat.shape}",
            ]
            with open(prefix + "_info.txt", "w") as f:
                f.write("\n".join(info_text))
