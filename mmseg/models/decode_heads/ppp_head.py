# Copyright (c) OpenMMLab.
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2

from mmseg.registry import MODELS
from .decode_head import BaseDecodeHead


@MODELS.register_module()
class PPPHead(BaseDecodeHead):
    """
    PPPHead: Features -> Heatmap-Logits -> PPP-Loss

    - Input: Encoder-Features
    - Output: 1-Kanal-Heatmap-Logits (Intensität λ-Logits)
    - GT:    Cityscapes-InstanceIds -> Zentren -> Binärmaske der Zentren
    - Loss:  PPPLoss(heatmap_logits, center_mask)
    """

    def __init__(self,
                 num_convs=2,
                 kernel_size=3,
                 concat_input=True,
                 dilation=1,
                 loss_ppp=dict(type='PPPLoss', loss_weight=1.0),
                 **kwargs):
        super().__init__(**kwargs)

        self.num_convs = num_convs
        self.concat_input = concat_input
        self.kernel_size = kernel_size
        self.loss_ppp = MODELS.build(loss_ppp)

        self.final_conv = nn.Sequential(
            nn.Conv2d(self.channels, self.channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.up1 = nn.ConvTranspose2d(self.channels, 128, 4, stride=2, padding=1)
        self.up2 = nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1)
        self.up3 = nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1)   

        conv_padding = (kernel_size // 2) * dilation
        convs = []
        convs.append(
            nn.Sequential(
                nn.Conv2d(self.in_channels, self.channels,
                          kernel_size=kernel_size,
                          padding=conv_padding,
                          dilation=dilation),
                nn.ReLU(inplace=True)
            )
        )
        for i in range(num_convs - 1):
            convs.append(
                nn.Sequential(
                    nn.Conv2d(self.channels, self.channels,
                              kernel_size=kernel_size,
                              padding=conv_padding,
                              dilation=dilation),
                    nn.ReLU(inplace=True)
                )
            )
        self.convs = nn.Sequential(*convs)

        if self.concat_input:
            self.conv_cat = nn.Sequential(
                nn.Conv2d(self.in_channels + self.channels, self.channels,
                          kernel_size=kernel_size, padding=kernel_size // 2),
                nn.ReLU(inplace=True)
            )

        # Cityscapes "thing classes" (die mit Instanzen)
        self.thing_classes = {
            # 24,  # person
            # 25,  # rider
            26,  # car
            27,  # truck
            28,  # bus
            # 31,  # train
            # 32,  # motorcycle
            # 33   # bicycle
        }

    def _forward_feature(self, inputs):
        x = self._transform_inputs(inputs)
        feats = self.convs(x)
        if self.concat_input:
            feats = self.conv_cat(torch.cat([x, feats], dim=1))
        return feats

    def forward(self, inputs):
        """Produce heatmap logits (N, 1, H, W)."""
        feats = self._forward_feature(inputs)
        feats = self.final_conv(feats)
        feats = F.relu(self.up1(feats))  # (N, 128, 64, 128)
        feats = F.relu(self.up2(feats))  # (N, 64, 128, 256)
        feats = F.relu(self.up3(feats))  # (N, 32, 256, 512)  
        seg_logits = self.cls_seg(feats)  # num_classes sollte 1 sein
        return seg_logits


    def loss(self, inputs, batch_data_samples, train_cfg=None):
        seg_logits = self.forward(inputs)

        # collect GT labels
        seg_labels = torch.stack(
            [s.gt_sem_seg.data for s in batch_data_samples]
        ).to(seg_logits.device)

        centers_gt = self.extract_centers_from_label(seg_labels)

        center_mask = self.centers_to_mask(
            centers_gt,
            seg_logits[:, 0],
            meta=batch_data_samples[0].metainfo
        )

        # print("PPP centers per image:", [c.shape[0] for c in centers_gt])
        # print("Max in center_mask:", center_mask.max().item())

        loss_ppp = self.loss_ppp(seg_logits, center_mask, avg_factor=len(centers_gt))

        return {'loss_ppp': loss_ppp}


    def extract_centers_from_label(self, seg_labels):
        """
        seg_labels: (N, 1, H, W) instanceId mask
        Returns: List[tensor(K_i, 2)] mit (cx, cy) pro Bild.
        """
        batch_centers = []

        for b in range(seg_labels.size(0)):
            inst_mask = seg_labels[b, 0].cpu().numpy()

            centers = []
            unique_ids = np.unique(inst_mask)

            for inst_id in unique_ids:
                if inst_id == 0:
                    continue  # background

                # Cityscapes encoding: classId * 1000 + instanceNumber
                class_id = inst_id // 1000
                if class_id not in self.thing_classes:
                    continue

                ys, xs = np.where(inst_mask == inst_id)
                if len(xs) == 0:
                    continue

                cx = float(xs.mean())
                cy = float(ys.mean())
                centers.append([cx, cy])

            batch_centers.append(
                torch.tensor(centers, dtype=torch.float32,
                             device=seg_labels.device)
            )
        return batch_centers
    

    def centers_to_mask(self, batch_centers, heatmap_logits, meta):
        """
        batch_centers: List[tensor(K_i, 2)] in coords of gt_sem_seg
        heatmap_logits: (N, H_out, W_out)
        meta: metainfo of that sample
        """
        N, H_out, W_out = heatmap_logits.shape
        mask = torch.zeros_like(heatmap_logits)

        H_src, W_src = meta['img_shape'][:2]

        for b in range(N):
            centers = batch_centers[b]
            if centers.numel() == 0:
                continue

            for j in range(centers.size(0)):
                cx, cy = centers[j]

                # scale from label/img space → heatmap space
                cx = cx * (W_out / W_src)
                cy = cy * (H_out / H_src)

                cx = int(round(cx.item()))
                cy = int(round(cy.item()))

                if 0 <= cx < W_out and 0 <= cy < H_out:
                    mask[b, cy, cx] = 1.0

        return mask.unsqueeze(1)
