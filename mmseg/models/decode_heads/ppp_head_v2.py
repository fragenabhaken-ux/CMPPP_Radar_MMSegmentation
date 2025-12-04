import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2

from mmseg.registry import MODELS
from .decode_head import BaseDecodeHead


# ==============================================================
#   Residual Block
# ==============================================================

class ResBlock(nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=1):
        super().__init__()
        pad = (kernel_size // 2) * dilation
        self.conv1 = nn.Conv2d(channels, channels, kernel_size,
                               padding=pad, dilation=dilation)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size,
                               padding=pad, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)


# ==============================================================
#   Squeeze-and-Excitation Attention Block
# ==============================================================

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)

    def forward(self, x):
        N, C, H, W = x.size()
        s = x.mean(dim=[2,3])               # global avg
        s = F.relu(self.fc1(s))
        s = torch.sigmoid(self.fc2(s))
        s = s.view(N, C, 1, 1)
        return x * s


# ==============================================================
#   PPP Head V3
# ==============================================================

@MODELS.register_module()
class PPPHead_V2(BaseDecodeHead):

    def __init__(self,
                 num_resblocks=3,          # number of ResBlocks
                 kernel_size=3,
                 dilation=1,
                 concat_input=True,
                 dropout_ratio=0.1,
                 loss_decode=dict(type='PPPLoss_V2', loss_weight=1.0),
                 **kwargs):

        super().__init__(**kwargs)

        self.concat_input = concat_input
        self.loss_decode = MODELS.build(loss_decode)

        # ------------------------------
        # Initial conv: in_channels → channels
        # ------------------------------
        pad = (kernel_size // 2) * dilation
        self.conv_initial = nn.Sequential(
            nn.Conv2d(self.in_channels, self.channels,
                      kernel_size=kernel_size,
                      padding=pad,
                      dilation=dilation),
            nn.BatchNorm2d(self.channels),
            nn.ReLU(inplace=True)
        )

        # ------------------------------
        # Residual Blocks (adds depth)
        # ------------------------------
        resblocks = []
        for _ in range(num_resblocks):
            resblocks.append(ResBlock(self.channels,
                                      kernel_size=kernel_size,
                                      dilation=dilation))
        self.resblocks = nn.Sequential(*resblocks)

        # ------------------------------
        # Dilated context block (Atrous)
        # ------------------------------
        self.context = nn.Sequential(
            nn.Conv2d(self.channels, self.channels, 3, padding=2, dilation=2),
            nn.BatchNorm2d(self.channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(self.channels, self.channels, 3, padding=4, dilation=4),
            nn.BatchNorm2d(self.channels),
            nn.ReLU(inplace=True)
        )

        # ------------------------------
        # SE Attention
        # ------------------------------
        self.se = SEBlock(self.channels)

        # ------------------------------
        # Concat refinement (if enabled)
        # ------------------------------
        if concat_input:
            self.conv_cat = nn.Sequential(
                nn.Conv2d(self.in_channels + self.channels, self.channels,
                          kernel_size=kernel_size,
                          padding=kernel_size // 2),
                nn.BatchNorm2d(self.channels),
                nn.ReLU(inplace=True),
                nn.Dropout2d(dropout_ratio)
            )

        # Thing classes for Cityscapes
        self.thing_classes = {24, 25, 26, 27, 28, 31, 32, 33}


    # ----------------------------------------------------------
    # Forward feature computation
    # ----------------------------------------------------------

    def _forward_feature(self, inputs):
        x = self._transform_inputs(inputs)

        feats = self.conv_initial(x)
        feats = self.resblocks(feats)
        feats = self.context(feats)
        feats = self.se(feats)

        if self.concat_input:
            feats = self.conv_cat(torch.cat([x, feats], dim=1))

        return feats


    def forward(self, inputs):
        feats = self._forward_feature(inputs)
        return self.cls_seg(feats)   # (N, 1, H, W)


    # ----------------------------------------------------------
    # PPP LOSS
    # ----------------------------------------------------------

    def loss(self, inputs, batch_data_samples, train_cfg=None):
        seg_logits = self.forward(inputs)

        labels = []
        for sample in batch_data_samples:
            data = sample.gt_sem_seg.data
            if isinstance(data, torch.Tensor):
                labels.append(data)
            elif isinstance(data, (tuple, list)):
                for x in data:
                    if isinstance(x, torch.Tensor):
                        labels.append(x); break
                else:
                    raise TypeError("gt_sem_seg.data contains no tensor")
            else:
                raise TypeError("Unsupported gt_sem_seg.data type")

        seg_labels = torch.stack(labels).to(seg_logits.device)
        centers_gt = self.extract_centers_from_label(seg_labels)

        print("LOSS CENTERS:", [c.shape[0] for c in centers_gt])

        heatmap_logits = seg_logits[:, 0:1, :, :]

        center_mask = self.centers_to_mask(
            centers_gt,
            heatmap_logits[:, 0],
            meta=batch_data_samples[0].metainfo
        ).unsqueeze(1)

        loss_decode = self.loss_decode(heatmap_logits, center_mask)
        return dict(loss_decode=loss_decode)


    # ----------------------------------------------------------
    # Center extraction
    # ----------------------------------------------------------

    def extract_centers_from_label(self, seg_labels):
        batch_centers = []
        for b in range(seg_labels.size(0)):
            inst_mask = seg_labels[b, 0].cpu().numpy()
            centers = []
            unique_ids = np.unique(inst_mask)

            for inst_id in unique_ids:
                if inst_id == 0: continue
                class_id = inst_id // 1000
                if class_id not in self.thing_classes: continue

                ys, xs = np.where(inst_mask == inst_id)
                if len(xs) == 0: continue

                centers.append([float(xs.mean()), float(ys.mean())])

            batch_centers.append(torch.tensor(
                centers, dtype=torch.float32, device=seg_labels.device
            ))

        return batch_centers


    # ----------------------------------------------------------
    # Mask generator
    # ----------------------------------------------------------

    def centers_to_mask(self, batch_centers, heatmap_logits, meta):
        N, H, W = heatmap_logits.shape
        mask = torch.zeros_like(heatmap_logits)

        H_ori, W_ori = meta['ori_shape'][:2]

        for b in range(N):
            centers = batch_centers[b]
            if centers.numel() == 0:
                continue

            for j in range(centers.size(0)):
                cx, cy = centers[j]
                cx = int(round(cx.item() * (W / W_ori)))
                cy = int(round(cy.item() * (H / H_ori)))

                if 0 <= cx < W and 0 <= cy < H:
                    mask[b, cy, cx] = 1.0

        return mask
