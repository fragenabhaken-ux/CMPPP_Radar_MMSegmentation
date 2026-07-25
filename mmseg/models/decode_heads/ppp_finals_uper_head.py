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
class PPPfinalsUperHead(BaseDecodeHead):
    """
    UPerNet-style head with PPP outputs.

    Architecture:
      - UPerNet multi-scale fusion (PSP + FPN) is kept.
      - Final prediction is explicitly upsampled back to the input resolution.

    Output channels:
      0: intensity logits
      1: depth mark, transformed with exp in the loss
      2: velocity mark, transformed with cube in the loss
      3: RCS mark, transformed with exp in the loss

    Expects GT in data_samples[i].gt_sem_seg.data with shape (4, H, W):
      0: intensity mask {0, 1}
      1: depth in meters
      2: wrapped dynamic radial velocity
      3: RCS

    For signed velocity supervision, the loss uses the shortest distance on the
    Doppler circle. Each sample must provide a positive ``doppler_period`` in
    ``data_sample.metainfo``.
    """

    def __init__(
        self,
        pool_scales=(1, 2, 3, 6),

        # PPP-specific
        ppp_weight=1.0,
        depth_weight=1.0,
        velocity_weight=1.0,
        rcs_weight=1.0,
        velocity_mode: str = "signed",
        loss_ppp=dict(type="PPPLoss"),

        # BaseDecodeHead args
        in_channels=(256, 512, 1024, 2048),
        channels=512,
        num_classes=4,
        in_index=(0, 1, 2, 3),
        input_transform="multiple_select",
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

        if self.num_classes != 4:
            raise ValueError(f"Expected num_classes=4, got {self.num_classes}")

        self.ppp_weight = float(ppp_weight)
        self.depth_weight = float(depth_weight)
        self.velocity_weight = float(velocity_weight)
        self.rcs_weight = float(rcs_weight)

        self.velocity_mode = str(velocity_mode).lower()
        if self.velocity_mode not in {"signed", "absolute"}:
            raise ValueError(
                f"velocity_mode must be 'signed' or 'absolute', got {velocity_mode!r}."
            )

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
        """Apply the PPM to the deepest feature map."""
        x = inputs[-1]
        psp_outs = [x]
        psp_outs.extend(self.psp_modules(x))
        psp_outs = torch.cat(psp_outs, dim=1)
        return self.bottleneck(psp_outs)

    def _forward_feature(self, inputs):
        """Return fused UPer features at the highest backbone resolution."""
        inputs = self._transform_inputs(inputs)

        laterals = [
            self.lateral_convs[i](inputs[i])
            for i in range(len(self.lateral_convs))
        ]
        laterals.append(self.psp_forward(inputs))

        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            laterals[i - 1] = laterals[i - 1] + resize(
                input=laterals[i],
                size=prev_shape,
                mode="bilinear",
                align_corners=self.align_corners,
            )

        fpn_outs = [
            self.fpn_convs[i](laterals[i])
            for i in range(used_backbone_levels - 1)
        ]
        fpn_outs.append(laterals[-1])

        for i in range(used_backbone_levels - 1, 0, -1):
            fpn_outs[i] = resize(
                input=fpn_outs[i],
                size=fpn_outs[0].shape[2:],
                mode="bilinear",
                align_corners=self.align_corners,
            )

        feats = torch.cat(fpn_outs, dim=1)
        return self.fpn_bottleneck(feats)

    def forward(self, inputs):
        """Predict the four full-resolution PPP output channels."""
        out_h = inputs[0].shape[2] * 4
        out_w = inputs[0].shape[3] * 4

        feats = self._forward_feature(inputs)
        out = self.cls_seg(feats)

        return resize(
            input=out,
            size=(out_h, out_w),
            mode="bilinear",
            align_corners=self.align_corners,
        )

    # =========================================================
    # Mark losses
    # =========================================================

    @staticmethod
    def _doppler_periods(
        batch_data_samples,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Return one positive Doppler period per sample as [N, 1, 1]."""
        periods = []
        for sample_idx, sample in enumerate(batch_data_samples):
            metainfo = sample.metainfo
            if "doppler_period" not in metainfo:
                raise KeyError(
                    "Missing 'doppler_period' in data sample "
                    f"{sample_idx}. Ensure the transform packs dopplerRange."
                )

            period = float(metainfo["doppler_period"])
            period_tensor = torch.tensor(period, dtype=torch.float64)
            if not torch.isfinite(period_tensor).item() or period <= 0:
                raise ValueError(
                    f"Invalid Doppler period for sample {sample_idx}: {period}"
                )
            periods.append(period)

        return torch.tensor(
            periods,
            device=device,
            dtype=dtype,
        ).view(-1, 1, 1)

    @staticmethod
    def _circular_velocity_error(
        pred_vel: torch.Tensor,
        target_vel: torch.Tensor,
        doppler_period: torch.Tensor,
    ) -> torch.Tensor:
        """Return the signed shortest velocity difference modulo the period."""
        return torch.remainder(
            pred_vel - target_vel + 0.5 * doppler_period,
            doppler_period,
        ) - 0.5 * doppler_period

    # =========================================================
    # PPP loss
    # =========================================================

    def loss(self, inputs, batch_data_samples, train_cfg=None):
        preds = self.forward(inputs)

        gt = torch.stack(
            [sample.gt_sem_seg.data for sample in batch_data_samples]
        ).to(device=preds.device, dtype=preds.dtype)

        if gt.ndim != 4:
            raise ValueError(f"GT must be (N,C,H,W), got {tuple(gt.shape)}")
        if gt.shape[1] != 4:
            raise ValueError(f"Expected 4 GT channels, got {gt.shape[1]}")
        if gt.shape[0] != preds.shape[0]:
            raise ValueError("Batch size mismatch between preds and GT")

        if preds.shape[2:] != gt.shape[2:]:
            preds = resize(
                input=preds,
                size=gt.shape[2:],
                mode="bilinear",
                align_corners=self.align_corners,
            )

        # GT channels
        gt_int = gt[:, 0:1]
        gt_depth = gt[:, 1]
        gt_vel = gt[:, 2]
        gt_rcs = gt[:, 3]

        unique_vals = torch.unique(gt_int)
        if not torch.all((unique_vals == 0) | (unique_vals == 1)):
            raise ValueError(f"Intensity channel is not binary, values={unique_vals}")

        # Prediction channels
        pred_int = preds[:, 0:1]
        raw_depth = preds[:, 1]
        raw_vel = preds[:, 2]
        raw_rcs = preds[:, 3]

        # Mark links
        pred_depth = torch.exp(raw_depth)
        pred_rcs = raw_rcs**3

        if self.velocity_mode == "signed":
            pred_vel = raw_vel**3
            doppler_period = self._doppler_periods(
                batch_data_samples,
                device=pred_vel.device,
                dtype=pred_vel.dtype,
            )
            velocity_error = self._circular_velocity_error(
                pred_vel,
                gt_vel,
                doppler_period,
            )
            loss_velocity_per_pixel = torch.abs(velocity_error)
        elif self.velocity_mode == "absolute":
            pred_vel = torch.abs(raw_vel)
            loss_velocity_per_pixel = F.l1_loss(
                pred_vel,
                torch.abs(gt_vel),
                reduction="none",
            )
        else:
            raise RuntimeError(f"Unsupported velocity_mode={self.velocity_mode!r}")

        loss_ppp = self.loss_ppp(pred_int, gt_int)

        mask = gt_int.squeeze(1)
        nbr_points = mask.sum(dim=(1, 2)).clamp_min(1)

        gt_log_depth = torch.log(torch.clamp(gt_depth, min=1e-3))
        loss_depth_per_pixel = F.l1_loss(
            raw_depth,
            gt_log_depth,
            reduction="none",
        )
        loss_depth = (
            (loss_depth_per_pixel * mask).sum(dim=(1, 2))
            / nbr_points
        ).mean()

        loss_velocity = (
            (loss_velocity_per_pixel * mask).sum(dim=(1, 2))
            / nbr_points
        ).mean()

        loss_rcs = (
            (F.l1_loss(pred_rcs, gt_rcs, reduction="none") * mask)
            .sum(dim=(1, 2))
            / nbr_points
        ).mean()

        return dict(
            loss_ppp=self.ppp_weight * loss_ppp,
            loss_depth=self.depth_weight * loss_depth,
            loss_velocity=self.velocity_weight * loss_velocity,
            loss_rcs=self.rcs_weight * loss_rcs,
        )