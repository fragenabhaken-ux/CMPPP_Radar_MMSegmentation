# Copyright (c) OpenMMLab.
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmseg.registry import MODELS
from .decode_head import BaseDecodeHead


@MODELS.register_module()
class PPPMarkHead(BaseDecodeHead):
    """Marked PPP decode head with conv tower + upsampling.

    Output channels:
      0: intensity logits f(x)
      1: depth mark (regression)
      2: velocity mark (regression)

    Expects GT in data_samples[i].gt_sem_seg.data with shape (3, H, W):
      channel 0: intensity mask {0,1}
      channel 1: depth
      channel 2: velocity
    """

    def __init__(
        self,
        # ---- your conv head args ----
        num_convs: int = 2,
        kernel_size: int = 3,
        concat_input: bool = True,
        dilation: int = 1,

        # ---- losses ----
        loss_ppp: dict = dict(type="PPPLoss", loss_weight=1.0),
        # loss_depth: dict = dict(type="SmoothL1Loss", loss_weight=1.0),
        # loss_velocity: dict = dict(type="SmoothL1Loss", loss_weight=1.0),

        # ---- BaseDecodeHead args (make them EXPLICIT!) ----
        in_channels: int = 512,
        channels: int = 32,
        num_classes: int = 3,
        in_index: int = -1,
        input_transform=None,
        align_corners: bool = False,
        dropout_ratio: float = 0.1,
        conv_cfg=None,
        norm_cfg=None,
        act_cfg=dict(type="ReLU"),
        init_cfg=None,
        **kwargs,
    ):
        # IMPORTANT:
        # Do NOT pass num_convs / kernel_size / concat_input / dilation into BaseDecodeHead.
        super().__init__(
            in_channels=in_channels,
            channels=channels,
            num_classes=num_classes,
            in_index=in_index,
            input_transform=input_transform,
            align_corners=align_corners,
            dropout_ratio=dropout_ratio,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            init_cfg=init_cfg,
            **kwargs,
        )

        assert self.num_classes == 3, f"Expected num_classes=3, got {self.num_classes}"

        self.num_convs = int(num_convs)
        self.kernel_size = int(kernel_size)
        self.concat_input = bool(concat_input)

        self.loss_ppp = MODELS.build(loss_ppp)
        # self.loss_depth = MODELS.build(loss_depth)
        # self.loss_velocity = MODELS.build(loss_velocity)

        conv_padding = (self.kernel_size // 2) * dilation

        convs = []
        convs.append(
            nn.Sequential(
                nn.Conv2d(self.in_channels, self.channels,
                          kernel_size=self.kernel_size,
                          padding=conv_padding,
                          dilation=dilation),
                nn.ReLU(inplace=True),
            )
        )
        for _ in range(self.num_convs - 1):
            convs.append(
                nn.Sequential(
                    nn.Conv2d(self.channels, self.channels,
                              kernel_size=self.kernel_size,
                              padding=conv_padding,
                              dilation=dilation),
                    nn.ReLU(inplace=True),
                )
            )
        self.convs = nn.Sequential(*convs)

        if self.concat_input:
            self.conv_cat = nn.Sequential(
                nn.Conv2d(self.in_channels + self.channels, self.channels,
                          kernel_size=self.kernel_size,
                          padding=self.kernel_size // 2),
                nn.ReLU(inplace=True),
            )

        # keep your “final conv + upsampling” tower
        self.final_conv = nn.Sequential(
            nn.Conv2d(self.channels, self.channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.up1 = nn.ConvTranspose2d(self.channels, 128, 4, stride=2, padding=1)
        self.up2 = nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1)
        self.up3 = nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1)

        # NOTE: cls_seg is created by BaseDecodeHead and maps -> num_classes (=3)

    def _forward_feature(self, inputs):
        x = self._transform_inputs(inputs)
        feats = self.convs(x)
        if self.concat_input:
            feats = self.conv_cat(torch.cat([x, feats], dim=1))
        return feats

    def forward(self, inputs):
        feats = self._forward_feature(inputs)
        feats = self.final_conv(feats)
        feats = F.relu(self.up1(feats))
        feats = F.relu(self.up2(feats))
        feats = F.relu(self.up3(feats))
        out = self.cls_seg(feats)  # (N,3,H,W)
        return out

    def loss(self, inputs, batch_data_samples, train_cfg=None):
        preds = self.forward(inputs)  # (N,3,H,W)

        gt = torch.stack([s.gt_sem_seg.data for s in batch_data_samples]).to(preds.device)  # (N,3,H,W)

        gt_int = gt[:, 0:1]         # (N,1,H,W) 0/1
        gt_depth = gt[:, 1]         # (N,H,W)
        gt_vel = gt[:, 2]           # (N,H,W)

        pred_int = preds[:, 0:1]    # (N,1,H,W) logits
        pred_depth = preds[:, 1]    # (N,H,W)
        pred_vel = preds[:, 2]      # (N,H,W)

        # PPP loss only on intensity channel
        avg_factor = max(float(gt_int.sum().item()), 1.0)
        loss_ppp = self.loss_ppp(pred_int, gt_int, avg_factor=avg_factor)

        # masked mark regression at point locations only
        mask = gt_int.squeeze(1)  # (N,H,W)
        normalizer = mask.sum().clamp(min=1.0)

        # loss_depth = (
        #     self.loss_depth(pred_depth, gt_depth, reduction="none") * mask
        # ).sum() / normalizer

        # loss_velocity = (
        #     self.loss_velocity(pred_vel, gt_vel, reduction="none") * mask
        # ).sum() / normalizer

        loss_depth = (
            F.smooth_l1_loss(pred_depth, gt_depth, reduction="none") * mask
        ).sum() / normalizer

        loss_velocity = (
            F.smooth_l1_loss(pred_vel, gt_vel, reduction="none") * mask
        ).sum() / normalizer

        return dict(loss_ppp=loss_ppp, loss_depth=loss_depth, loss_velocity=loss_velocity)


# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# from mmseg.registry import MODELS
# from mmseg.models.decode_heads.decode_head import BaseDecodeHead


# @MODELS.register_module()
# class PPPMarkHead(BaseDecodeHead):
#     """
#     Decode head for a marked Poisson Point Process.

#     Output channels:
#         0: intensity logits f(x)
#         1: depth mark
#         2: velocity mark

#     Losses:
#         - PPPLoss on channel 0
#         - masked regression losses on channels 1,2
#     """

#     def __init__(
#         self,
#         loss_ppp=dict(type='PPPLoss', loss_weight=1.0),
#         loss_depth=dict(type='L1Loss', loss_weight=1.0),
#         loss_velocity=dict(type='L1Loss', loss_weight=1.0),
#         **kwargs
#     ):
#         # num_classes MUST equal number of output channels
#         super().__init__(**kwargs)

#         # losses
#         self.loss_ppp = MODELS.build(loss_ppp)
#         self.loss_depth = MODELS.build(loss_depth)
#         self.loss_velocity = MODELS.build(loss_velocity)

#     def forward(self, inputs):
#         """
#         Returns:
#             Tensor (N, 3, H, W)
#         """
#         x = self._transform_inputs(inputs)
#         x = self.cls_seg(x)
#         return x

#     def loss(self, inputs, batch_data_samples, train_cfg=None):
#         """
#         batch_data_samples[i].gt_sem_seg.data is expected to be:
#             (3, H, W):
#                 channel 0: intensity mask (0/1)
#                 channel 1: depth
#                 channel 2: velocity
#         """
#         preds = self.forward(inputs)

#         # collect GT
#         gt = torch.stack(
#             [sample.gt_sem_seg.data for sample in batch_data_samples]
#         ).to(preds.device)

#         # split predictions
#         pred_intensity = preds[:, 0:1]   # (N,1,H,W)
#         pred_depth = preds[:, 1]
#         pred_velocity = preds[:, 2]

#         # split GT
#         gt_intensity = gt[:, 0:1]         # (N,1,H,W)
#         gt_depth = gt[:, 1]
#         gt_velocity = gt[:, 2]

#         # ----------------------------
#         # PPP loss (intensity only)
#         # ----------------------------
#         avg_factor = max(gt_intensity.sum().item(), 1.0)
#         loss_ppp = self.loss_ppp(
#             pred_intensity,
#             gt_intensity,
#             avg_factor=avg_factor
#         )

#         # ----------------------------
#         # Masked mark losses
#         # ----------------------------
#         mask = gt_intensity.squeeze(1)  # (N,H,W)

#         # avoid division by zero
#         normalizer = mask.sum().clamp(min=1.0)

#         loss_depth = (
#             self.loss_depth(
#                 pred_depth,
#                 gt_depth,
#                 reduction='none'
#             ) * mask
#         ).sum() / normalizer

#         loss_velocity = (
#             self.loss_velocity(
#                 pred_velocity,
#                 gt_velocity,
#                 reduction='none'
#             ) * mask
#         ).sum() / normalizer

#         return dict(
#             loss_ppp=loss_ppp,
#             loss_depth=loss_depth,
#             loss_velocity=loss_velocity,
#         )
