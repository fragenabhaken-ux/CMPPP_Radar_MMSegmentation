from __future__ import annotations

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmseg.registry import MODELS
from .decode_head import BaseDecodeHead


LOGGER_NAME = "ppp_mark_head"
LOG_LEVEL = logging.INFO  # change to logging.DEBUG for more verbose output

logger = logging.getLogger(__name__)
logger.setLevel(LOG_LEVEL)

if not logger.hasHandlers():
    handler = logging.StreamHandler()
    handler.setLevel(LOG_LEVEL)
    logger.addHandler(handler)


@MODELS.register_module()
class PPPMarkHeadWoBottleneck(BaseDecodeHead):
    """Marked PPP decode head with gradual channel reduction and upsampling.

    Output channels:
      0: intensity logits f(x)
      1: depth mark (regression)
      2: velocity mark (regression)

    Expects GT in data_samples[i].gt_sem_seg.data with shape (3, H, W):
      channel 0: intensity mask {0,1}
      channel 1: depth
      channel 2: velocity

    Revised decoder structure (i.e. no bottleneck):
        512 -> 256 -> 128 -> 64 -> 32 -> 3
    """

    def __init__(
        self,
        # ---- refinement block args ----
        num_convs: int = 2,
        kernel_size: int = 3,
        concat_input: bool = True,
        dilation: int = 1,

        # ---- decoder channel plan ----
        refine_channels: int = 256,
        up_channels: tuple[int, int, int] = (128, 64, 32),

        # ---- loss weights ----
        ppp_weight: float = 1.0,
        depth_weight: float = 1.0,
        velocity_weight: float = 1.0,

        # ---- losses ----
        loss_ppp: dict = dict(type="PPPLoss"),

        # ---- BaseDecodeHead args ----
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
        """
        Notes
        -----
        `channels` is kept as the final feature width before `cls_seg`, so it
        should match `up_channels[-1]`. By default:
            channels = 32
            up_channels = (128, 64, 32)
        """
        if channels != up_channels[-1]:
            raise ValueError(
                f"`channels` must equal the last entry of `up_channels`. "
                f"Got channels={channels}, up_channels[-1]={up_channels[-1]}."
            )

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

        if self.num_classes != 3:
            raise ValueError(f"Expected num_classes=3, got {self.num_classes}")

        self.num_convs = int(num_convs)
        self.kernel_size = int(kernel_size)
        self.concat_input = bool(concat_input)
        self.refine_channels = int(refine_channels)
        self.up_channels = tuple(int(c) for c in up_channels)

        self.ppp_weight = float(ppp_weight)
        self.depth_weight = float(depth_weight)
        self.velocity_weight = float(velocity_weight)

        self.loss_ppp = MODELS.build(loss_ppp)

        conv_padding = (self.kernel_size // 2) * dilation

        # ------------------------------------------------------------------
        # Feature refinement block
        # Gradual first step: 512 -> 256 instead of 512 -> 32
        # ------------------------------------------------------------------
        convs = [
            nn.Sequential(
                nn.Conv2d(
                    self.in_channels,
                    self.refine_channels,
                    kernel_size=self.kernel_size,
                    padding=conv_padding,
                    dilation=dilation,
                ),
                nn.ReLU(inplace=True),
            )
        ]

        for _ in range(self.num_convs - 1):
            convs.append(
                nn.Sequential(
                    nn.Conv2d(
                        self.refine_channels,
                        self.refine_channels,
                        kernel_size=self.kernel_size,
                        padding=conv_padding,
                        dilation=dilation,
                    ),
                    nn.ReLU(inplace=True),
                )
            )

        self.convs = nn.Sequential(*convs)

        if self.concat_input:
            self.conv_cat = nn.Sequential(
                nn.Conv2d(
                    self.in_channels + self.refine_channels,
                    self.refine_channels,
                    kernel_size=self.kernel_size,
                    padding=self.kernel_size // 2,
                ),
                nn.ReLU(inplace=True),
            )

        # Optional extra processing before transposed-conv upsampling
        self.final_conv = nn.Sequential(
            nn.Conv2d(self.refine_channels, self.refine_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        # ------------------------------------------------------------------
        # Progressive upsampling path:
        # 256 -> 128 -> 64 -> 32
        # ------------------------------------------------------------------
        self.up1 = nn.ConvTranspose2d(
            self.refine_channels, self.up_channels[0], kernel_size=4, stride=2, padding=1
        )
        self.up2 = nn.ConvTranspose2d(
            self.up_channels[0], self.up_channels[1], kernel_size=4, stride=2, padding=1
        )
        self.up3 = nn.ConvTranspose2d(
            self.up_channels[1], self.up_channels[2], kernel_size=4, stride=2, padding=1
        )

        # cls_seg is created by BaseDecodeHead and maps:
        #   channels (= 32) -> num_classes (= 3)

    def _forward_feature(self, inputs):
        x = self._transform_inputs(inputs)
        feats = self.convs(x)
        if self.concat_input:
            feats = self.conv_cat(torch.cat([x, feats], dim=1))
        return feats

    def forward(self, inputs):
        feats = self._forward_feature(inputs)   # (N, 256, H, W) by default
        feats = self.final_conv(feats)          # (N, 256, H, W)
        feats = F.relu(self.up1(feats))         # (N, 128, 2H, 2W)
        feats = F.relu(self.up2(feats))         # (N,  64, 4H, 4W)
        feats = F.relu(self.up3(feats))         # (N,  32, 8H, 8W)
        out = self.cls_seg(feats)               # (N,   3, 8H, 8W)
        return out

    def loss(self, inputs, batch_data_samples, train_cfg=None):
        preds = self.forward(inputs)  # (N, 3, H, W)

        assert torch.isfinite(preds).all(), (
            f"preds {tuple(preds.shape)} has "
            f"{(~torch.isfinite(preds)).sum().item()} NaN/Inf values"
        )

        gt = torch.stack([s.gt_sem_seg.data for s in batch_data_samples]).to(preds.device)

        # ------------------------------------------------------------------
        # GT sanity checks
        # ------------------------------------------------------------------
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
                    depth_i.min().item(),
                    depth_i.max().item(),
                    vel_i.min().item(),
                    vel_i.max().item(),
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
            gt[:, 1].min().item(),
            gt[:, 1].max().item(),
            gt[:, 2].min().item(),
            gt[:, 2].max().item(),
        )

        gt_int = gt[:, 0:1]   # (N,1,H,W)
        gt_depth = gt[:, 1]   # (N,H,W)
        gt_vel = gt[:, 2]     # (N,H,W)

        unique_vals = torch.unique(gt_int)
        assert torch.all((unique_vals == 0) | (unique_vals == 1)), (
            f"Intensity channel not binary, values={unique_vals}"
        )
        assert gt_int.max() <= 1, "Intensity GT corrupted (values > 1)"
        assert gt_int.min() >= 0, "Intensity GT corrupted (values < 0)"

        pred_int = preds[:, 0:1]   # logits for PPP intensity
        raw_depth = preds[:, 1]
        raw_vel = preds[:, 2]

        # Link functions
        pred_depth = torch.exp(raw_depth)   # strictly positive
        pred_vel = raw_vel ** 3             # signed

        assert torch.isfinite(pred_int).all(), (
            f"pred_int {tuple(pred_int.shape)} has "
            f"{(~torch.isfinite(pred_int)).sum().item()} NaN/Inf"
        )
        assert torch.isfinite(pred_depth).all(), "pred_depth has NaN/Inf"
        assert torch.isfinite(pred_vel).all(), "pred_vel has NaN/Inf"
        assert (pred_depth > 0).all(), "pred_depth must be strictly positive"

        # ------------------------------------------------------------------
        # PPP loss on intensity channel
        # ------------------------------------------------------------------
        loss_ppp = self.loss_ppp(pred_int, gt_int)

        # ------------------------------------------------------------------
        # Mark regression only at point locations
        # ------------------------------------------------------------------
        mask = gt_int.squeeze(1)  # (N,H,W)
        nbr_points = mask.sum(dim=(1, 2)).clamp_min(1)  # (N,)

        if mask.sum() == 0:
            logger.debug(
                "PPPMarkHead | empty PPP mask in batch (no points), skipping mark supervision"
            )

        # Per-image normalization so samples with many points do not dominate
        loss_depth = (
            (F.l1_loss(pred_depth, gt_depth, reduction="none") * mask).sum(dim=(1, 2))
            / nbr_points
        ).mean()

        loss_velocity = (
            (F.l1_loss(pred_vel, gt_vel, reduction="none") * mask).sum(dim=(1, 2))
            / nbr_points
        ).mean()

        logger.debug(
            "PPPMarkHead | batch=%d | PPP_points=%d | "
            "loss_ppp=%.1e * %.4f | loss_d=%.1e * %.4f | loss_v=%.1e * %.4f",
            preds.shape[0],
            int(gt_int.sum().item()),
            self.ppp_weight,
            loss_ppp.item(),
            self.depth_weight,
            loss_depth.item(),
            self.velocity_weight,
            loss_velocity.item(),
        )

        return dict(
            loss_ppp=self.ppp_weight * loss_ppp,
            loss_depth=self.depth_weight * loss_depth,
            loss_velocity=self.velocity_weight * loss_velocity,
        )