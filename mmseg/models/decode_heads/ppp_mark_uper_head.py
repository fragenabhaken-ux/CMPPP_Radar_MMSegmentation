from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv.cnn import ConvModule
from mmseg.registry import MODELS
from mmseg.models.utils import resize
from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmseg.models.decode_heads.psp_head import PPM


@MODELS.register_module()
class PPPMarkUperHead(BaseDecodeHead):
    """
    UPerNet-style head with PPP outputs.

    Architecture:
      - UPerNet multi-scale fusion (PSP + FPN) is kept.
      - Final prediction is explicitly upsampled back to the input resolution.

    Output channels:
      0: intensity logits
      1: depth mark (regression, transformed with exp in loss)
      2: velocity mark (regression, transformed with cube in loss)
    """

    def __init__(
        self,
        pool_scales=(1, 2, 3, 6),

        # PPP-specific
        ppp_weight=1.0,
        depth_weight=1.0,
        velocity_weight=1.0,
        loss_ppp=dict(type="PPPLoss"),

        # BaseDecodeHead args
        in_channels=(256, 512, 1024, 2048),
        channels=512,
        num_classes=3,
        in_index=(0, 1, 2, 3),
        input_transform='multiple_select',
        align_corners=False,
        dropout_ratio=0.1,
        conv_cfg=None,
        norm_cfg=None,
        act_cfg=dict(type="ReLU"),
        init_cfg=None,
        **kwargs,
    ):
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

        # PPP loss weights / builder
        self.ppp_weight = float(ppp_weight)
        self.depth_weight = float(depth_weight)
        self.velocity_weight = float(velocity_weight)
        self.loss_ppp = MODELS.build(loss_ppp)

        # -----------------------------
        # PSP module on deepest feature
        # -----------------------------
        self.psp_modules = PPM(
            pool_scales,
            self.in_channels[-1],
            self.channels,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg,
            align_corners=self.align_corners,
        )

        self.bottleneck = ConvModule(
            self.in_channels[-1] + len(pool_scales) * self.channels,
            self.channels,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg,
        )

        # -----------------------------
        # FPN lateral and refinement convs
        # -----------------------------
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        for in_ch in self.in_channels[:-1]:
            self.lateral_convs.append(
                ConvModule(
                    in_ch,
                    self.channels,
                    1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg,
                    inplace=False,
                )
            )
            self.fpn_convs.append(
                ConvModule(
                    self.channels,
                    self.channels,
                    3,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg,
                    inplace=False,
                )
            )

        self.fpn_bottleneck = ConvModule(
            len(self.in_channels) * self.channels,
            self.channels,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg,
        )

    # =========================================================
    # UPer feature fusion
    # =========================================================

    def psp_forward(self, inputs):
        """PSP on deepest feature map."""
        x = inputs[-1]
        psp_outs = [x]
        psp_outs.extend(self.psp_modules(x))
        psp_outs = torch.cat(psp_outs, dim=1)
        return self.bottleneck(psp_outs)

    def _forward_feature(self, inputs):
        """
        Returns fused UPer features at the highest backbone resolution
        (i.e. resolution of stage 1 / inputs[0]).
        """
        inputs = self._transform_inputs(inputs)

        # lateral projections for lower three stages
        laterals = [
            self.lateral_convs[i](inputs[i])
            for i in range(len(self.lateral_convs))
        ]

        # append top feature processed by PSP
        laterals.append(self.psp_forward(inputs))

        # top-down fusion
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            laterals[i - 1] = laterals[i - 1] + resize(
                input=laterals[i],
                size=prev_shape,
                mode='bilinear',
                align_corners=self.align_corners,
            )

        # per-level refinement
        fpn_outs = [
            self.fpn_convs[i](laterals[i])
            for i in range(used_backbone_levels - 1)
        ]
        fpn_outs.append(laterals[-1])  # PSP output for deepest level

        # resize all to highest backbone resolution
        for i in range(used_backbone_levels - 1, 0, -1):
            fpn_outs[i] = resize(
                input=fpn_outs[i],
                size=fpn_outs[0].shape[2:],
                mode='bilinear',
                align_corners=self.align_corners,
            )

        feats = torch.cat(fpn_outs, dim=1)
        feats = self.fpn_bottleneck(feats)
        return feats

    def forward(self, inputs):
        """
        Forward pass.

        inputs is the list of backbone stage outputs.
        For a standard ResNet backbone, inputs[0] has spatial size H/4 x W/4.
        Therefore we explicitly upsample by factor 4 back to the input resolution.
        """
        # save target output size before _transform_inputs changes the structure
        out_h = inputs[0].shape[2] * 4
        out_w = inputs[0].shape[3] * 4

        feats = self._forward_feature(inputs)
        out = self.cls_seg(feats)  # (N, 3, H/4, W/4)

        # explicit full-resolution output
        out = resize(
            input=out,
            size=(out_h, out_w),
            mode='bilinear',
            align_corners=self.align_corners,
        )
        return out

    # =========================================================
    # PPP loss
    # =========================================================

    def loss(self, inputs, batch_data_samples, train_cfg=None):
        preds = self.forward(inputs)  # intended to be full resolution

        gt = torch.stack(
            [s.gt_sem_seg.data for s in batch_data_samples]
        ).to(preds.device)  # (N, 3, H, W)

        # safeguard: if sizes ever differ, align preds to GT
        if preds.shape[2:] != gt.shape[2:]:
            preds = resize(
                input=preds,
                size=gt.shape[2:],
                mode='bilinear',
                align_corners=self.align_corners,
            )

        # GT channels
        gt_int = gt[:, 0:1]   # (N,1,H,W)
        gt_depth = gt[:, 1]   # (N,H,W)
        gt_vel = gt[:, 2]     # (N,H,W)

        # prediction channels
        pred_int = preds[:, 0:1]
        raw_depth = preds[:, 1]
        raw_vel = preds[:, 2]

        # links
        pred_depth = torch.exp(raw_depth)
        pred_vel = raw_vel ** 3

        # PPP loss
        loss_ppp = self.loss_ppp(pred_int, gt_int)

        # masked mark regression at point locations only
        mask = gt_int.squeeze(1)  # (N,H,W)
        nbr_points = mask.sum(dim=(1, 2)).clamp_min(1)

        loss_depth = (
            (F.l1_loss(pred_depth, gt_depth, reduction="none") * mask)
            .sum(dim=(1, 2)) / nbr_points
        ).mean()

        loss_velocity = (
            (F.l1_loss(pred_vel, gt_vel, reduction="none") * mask)
            .sum(dim=(1, 2)) / nbr_points
        ).mean()

        return dict(
            loss_ppp=self.ppp_weight * loss_ppp,
            loss_depth=self.depth_weight * loss_depth,
            loss_velocity=self.velocity_weight * loss_velocity,
        )