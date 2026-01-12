# Copyright (c) OpenMMLab.
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmseg.registry import MODELS
from .decode_head import BaseDecodeHead

import logging

LOGGER_NAME = "ppp_mark_head"
LOG_LEVEL = logging.INFO  # change to logging.INFO, logging.DEBUG 

logger = logging.getLogger(__name__)
logger.setLevel(LOG_LEVEL)

if not logger.hasHandlers():
    handler = logging.StreamHandler()
    handler.setLevel(LOG_LEVEL)
    logger.addHandler(handler)


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

        # ---- loss weights for marks ----
        ppp_weight: float = 1,
        depth_weight: float = 1,
        velocity_weight: float = 1,

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

        self.ppp_weight = float(ppp_weight)
        self.depth_weight = float(depth_weight)
        self.velocity_weight = float(velocity_weight)


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
        out = self.cls_seg(feats)  # Input (N, channels, H, W), output (N,3,H,W)
        return out

    def loss(self, inputs, batch_data_samples, train_cfg=None):
        preds = self.forward(inputs)  # (N,3,H,W)

        gt = torch.stack([s.gt_sem_seg.data for s in batch_data_samples]).to(preds.device)  # (N,3,H,W)

        # GT sanity checks
        assert gt.ndim == 4, f"GT must be (N,C,H,W), got {gt.shape}"
        assert gt.shape[1] == 3, f"Expected 3 GT channels, got {gt.shape[1]}"
        assert gt.shape[0] == preds.shape[0], "Batch size mismatch GT vs preds"

        if logger.isEnabledFor(logging.DEBUG):
            for i in range(gt.shape[0]):
                gt_i = gt[i]
                int_i = gt_i[0]
                depth_i = gt_i[1]
                vel_i = gt_i[2]

                logger.debug(
                    "PPPMarkHead | sample=%d | PPP_points=%d | "
                    "depth[min,max]=[%.3f, %.3f] | vel[min,max]=[%.3f, %.3f]",
                    i,
                    int(int_i.sum().item()),
                    depth_i.min().item(), depth_i.max().item(),
                    vel_i.min().item(), vel_i.max().item(),
                )

        if torch.isnan(gt).any():
            raise ValueError("GT contains NaNs")

        if torch.isinf(gt).any():
            raise ValueError("GT contains Infs")

        logger.debug(
            "PPPMarkHead | GT stats: "
            "shape=%s | int_sum=%d | depth[min,max]=[%.3f, %.3f] | vel[min,max]=[%.3f, %.3f]",
            tuple(gt.shape),
            int(gt[:, 0].sum().item()),
            gt[:, 1].min().item(), gt[:, 1].max().item(),
            gt[:, 2].min().item(), gt[:, 2].max().item(),
        )

        gt_int = gt[:, 0:1]         # (N,1,H,W) 0/1
        gt_depth = gt[:, 1]         # (N,H,W)
        gt_vel = gt[:, 2]           # (N,H,W)

        # intensity must be binary {0,1}
        unique_vals = torch.unique(gt_int)
        assert torch.all((unique_vals == 0) | (unique_vals == 1)), (
            f"Intensity channel not binary, values={unique_vals}"
        )

        assert gt_int.max() <= 1, "Intensity GT corrupted (values > 1)"
        assert gt_int.min() >= 0, "Intensity GT corrupted (values < 0)"

        pred_int = preds[:, 0:1]    # (N,1,H,W) logits
        raw_depth = preds[:, 1]    # (N,H,W)
        raw_vel = preds[:, 2]      # (N,H,W)

        # apply link to depth and velocity
        pred_depth = torch.exp(raw_depth)  # depth > 0
        pred_vel = raw_vel ** 3   # signed

        assert torch.isfinite(pred_int).all(), "pred_int has NaN/Inf"
        assert torch.isfinite(pred_depth).all(), "pred_depth has NaN/Inf"
        assert torch.isfinite(pred_vel).all(), "pred_vel has NaN/Inf"

        assert (pred_depth > 0).all(), "pred_depth must be strictly positive"

        # PPP loss only on intensity channel
        # avg_factor = max(float(gt_int.sum().item()), 1.0)
        avg_factor = 1 # Could this be the saling bug??? Normalization by gt_count??)
        loss_ppp = self.ppp_weight * self.loss_ppp(pred_int, gt_int, avg_factor=avg_factor)

        # masked mark regression at point locations only
        mask = gt_int.squeeze(1)  # (N,H,W)
        # # We normalize the loss to make the gradient scale independent of how 
        # # many points happen to be in an image.
        # normalizer = mask.sum().clamp(min=1.0)

        if mask.sum() == 0:
            logger.debug(
                "PPPMarkHead | empty PPP mask in batch (no points), skipping mark supervision"
            )

        # loss_depth = (
        #     self.loss_depth(pred_depth, gt_depth, reduction="none") * mask
        # ).sum() / normalizer

        # loss_velocity = (
        #     self.loss_velocity(pred_vel, gt_vel, reduction="none") * mask
        # ).sum() / normalizer

        # Note: While the intensity is trained everywhere, the marks are only trained
        # on the masked points. The prediction is to be understood as E[m|point], i.e.
        # if PPP has an event here, that is what the mark would be (conditional preds).
        loss_depth = self.depth_weight * (
            F.l1_loss(pred_depth, gt_depth, reduction="none") * mask
        ).sum()

        loss_velocity = self.velocity_weight * (
            F.l1_loss(pred_vel, gt_vel, reduction="none") * mask
        ).sum()

        logger.debug(
            "PPPMarkHead | batch=%d | PPP_points=%d | loss_ppp=%.4f | loss_d=%.4f | loss_v=%.4f",
            preds.shape[0],
            int(gt_int.sum().item()),
            loss_ppp.item(),
            loss_depth.item(),
            loss_velocity.item(),
        )

        return dict(loss_ppp=loss_ppp, loss_depth=loss_depth, loss_velocity=loss_velocity)