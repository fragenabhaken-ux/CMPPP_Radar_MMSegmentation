from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule

from mmseg.registry import MODELS
from .decode_head import BaseDecodeHead


@MODELS.register_module()
class PPPMarkUNetHead(BaseDecodeHead):
    """PPP head for MMSeg UNet.

    This head follows the FCNHead structure used with MMSeg's UNet, but replaces
    the segmentation loss with the PPP loss formulation.

    Output channels:
      0: intensity logits
      1: depth mark (regression, transformed with exp in loss)
      2: velocity mark (regression, transformed with cube in loss)

    Expects GT in data_samples[i].gt_sem_seg.data with shape (3, H, W):
      channel 0: intensity mask {0,1}
      channel 1: depth
      channel 2: velocity
    """

    def __init__(
        self,
        num_convs: int = 1,
        kernel_size: int = 3,
        dilation: int = 1,
        ppp_weight: float = 1.0,
        depth_weight: float = 1.0,
        velocity_weight: float = 1.0,
        loss_ppp: dict = dict(type="PPPLoss"),
        **kwargs,
    ):
        assert num_convs >= 0 and dilation > 0 and isinstance(dilation, int)

        self.num_convs = int(num_convs)
        self.kernel_size = int(kernel_size)

        super().__init__(**kwargs)

        if self.num_classes != 3:
            raise ValueError(f"Expected num_classes=3, got {self.num_classes}")
        if self.num_convs == 0 and self.in_channels != self.channels:
            raise ValueError(
                "If num_convs=0, in_channels must equal channels. "
                f"Got in_channels={self.in_channels}, channels={self.channels}."
            )

        self.ppp_weight = float(ppp_weight)
        self.depth_weight = float(depth_weight)
        self.velocity_weight = float(velocity_weight)
        self.loss_ppp = MODELS.build(loss_ppp)

        conv_padding = (self.kernel_size // 2) * dilation

        if self.num_convs == 0:
            self.convs = nn.Identity()
        else:
            convs = [
                ConvModule(
                    self.in_channels,
                    self.channels,
                    kernel_size=self.kernel_size,
                    padding=conv_padding,
                    dilation=dilation,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg,
                )
            ]
            for _ in range(self.num_convs - 1):
                convs.append(
                    ConvModule(
                        self.channels,
                        self.channels,
                        kernel_size=self.kernel_size,
                        padding=conv_padding,
                        dilation=dilation,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg,
                    )
                )
            self.convs = nn.Sequential(*convs)

    def _forward_feature(self, inputs):
        """Forward features before cls_seg."""
        x = self._transform_inputs(inputs)
        feats = self.convs(x)
        return feats

    def forward(self, inputs):
        """Forward function."""
        feats = self._forward_feature(inputs)
        out = self.cls_seg(feats)
        return out

    def loss(self, inputs, batch_data_samples, train_cfg=None):
        preds = self.forward(inputs)

        gt = torch.stack(
            [sample.gt_sem_seg.data for sample in batch_data_samples]
        ).to(preds.device)

        if gt.ndim != 4:
            raise ValueError(f"GT must be (N,C,H,W), got {tuple(gt.shape)}")
        if gt.shape[1] != 3:
            raise ValueError(f"Expected 3 GT channels, got {gt.shape[1]}")
        if gt.shape[0] != preds.shape[0]:
            raise ValueError("Batch size mismatch between preds and GT")

        if preds.shape[2:] != gt.shape[2:]:
            preds = F.interpolate(
                preds,
                size=gt.shape[2:],
                mode="bilinear",
                align_corners=self.align_corners,
            )

        gt_int = gt[:, 0:1]
        gt_depth = gt[:, 1]
        gt_vel = gt[:, 2]

        unique_vals = torch.unique(gt_int)
        if not torch.all((unique_vals == 0) | (unique_vals == 1)):
            raise ValueError(f"Intensity channel is not binary, values={unique_vals}")

        pred_int = preds[:, 0:1]
        raw_depth = preds[:, 1]
        raw_vel = preds[:, 2]

        pred_depth = torch.exp(raw_depth)
        pred_vel = raw_vel ** 3

        loss_ppp = self.loss_ppp(pred_int, gt_int)

        mask = gt_int.squeeze(1)
        nbr_points = mask.sum(dim=(1, 2)).clamp_min(1)

        loss_depth = (
            (F.l1_loss(pred_depth, gt_depth, reduction="none") * mask).sum(dim=(1, 2))
            / nbr_points
        ).mean()

        loss_velocity = (
            (F.l1_loss(pred_vel, gt_vel, reduction="none") * mask).sum(dim=(1, 2))
            / nbr_points
        ).mean()

        return dict(
            loss_ppp=self.ppp_weight * loss_ppp,
            loss_depth=self.depth_weight * loss_depth,
            loss_velocity=self.velocity_weight * loss_velocity,
        )