from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule

from mmseg.registry import MODELS
from .decode_head import BaseDecodeHead


class InterpConvBlock(nn.Module):
    """Bilinear upsampling followed by ConvModule."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        conv_cfg=None,
        norm_cfg=None,
        act_cfg=dict(type="ReLU"),
        align_corners: bool = False,
    ):
        super().__init__()
        self.align_corners = align_corners
        self.conv = ConvModule(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(
            x,
            scale_factor=2,
            mode="bilinear",
            align_corners=self.align_corners,
        )
        return self.conv(x)


@MODELS.register_module()
class PPPMarkViTUpsampleHead(BaseDecodeHead):
    """PPP decode head for ViT patch features.

    This head consumes a low-resolution spatial ViT feature map, upsamples it
    convolutionally, and predicts three output channels:

        channel 0: intensity logits
        channel 1: depth mark, transformed with exp in the loss
        channel 2: velocity mark, transformed with cube in the loss

    The loss logic intentionally mirrors PPPMarkUNetHead.
    """

    def __init__(
        self,
        num_convs: int = 1,
        num_upsample_layers: int = 4,
        upsample_channels: tuple[int, ...] = (256, 128, 64, 64),
        kernel_size: int = 3,
        dilation: int = 1,
        ppp_weight: float = 1.0,
        depth_weight: float = 1.0,
        velocity_weight: float = 1.0,
        velocity_mode: str = "signed",
        loss_ppp: dict = dict(type="PPPLoss"),
        **kwargs,
    ):
        assert num_convs >= 0
        assert num_upsample_layers >= 0
        assert dilation > 0 and isinstance(dilation, int)

        self.num_convs = int(num_convs)
        self.num_upsample_layers = int(num_upsample_layers)
        self.upsample_channels = tuple(int(c) for c in upsample_channels)
        self.kernel_size = int(kernel_size)

        if len(self.upsample_channels) != self.num_upsample_layers:
            raise ValueError(
                "len(upsample_channels) must equal num_upsample_layers. "
                f"Got {len(self.upsample_channels)} and "
                f"{self.num_upsample_layers}."
            )

        super().__init__(**kwargs)

        if self.num_classes != 3:
            raise ValueError(f"Expected num_classes=3, got {self.num_classes}")

        self.ppp_weight = float(ppp_weight)
        self.depth_weight = float(depth_weight)
        self.velocity_weight = float(velocity_weight)

        self.velocity_mode = str(velocity_mode).lower()
        if self.velocity_mode not in {"signed", "absolute"}:
            raise ValueError(
                f"velocity_mode must be 'signed' or 'absolute', got {velocity_mode!r}."
            )

        self.loss_ppp = MODELS.build(loss_ppp)

        conv_padding = (self.kernel_size // 2) * dilation

        if self.num_convs == 0:
            if self.in_channels != self.channels:
                raise ValueError(
                    "If num_convs=0, in_channels must equal channels. "
                    f"Got in_channels={self.in_channels}, "
                    f"channels={self.channels}."
                )
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

        up_layers = []
        current_channels = self.channels

        for out_channels in self.upsample_channels:
            up_layers.append(
                InterpConvBlock(
                    current_channels,
                    out_channels,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg,
                    align_corners=self.align_corners,
                )
            )
            current_channels = out_channels

        self.upsample_layers = nn.Sequential(*up_layers)

        # BaseDecodeHead creates cls_seg with input channels = self.channels.
        # After upsampling, the number of channels may have changed, so we
        # replace cls_seg accordingly.
        if self.num_upsample_layers > 0:
            self.conv_seg = nn.Conv2d(
                current_channels,
                self.num_classes,
                kernel_size=1,
            )

    def _forward_feature(self, inputs):
        x = self._transform_inputs(inputs)
        feats = self.convs(x)
        feats = self.upsample_layers(feats)
        return feats

    def forward(self, inputs):
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

        if self.velocity_mode == "signed":
            pred_vel = raw_vel ** 3
            target_vel = gt_vel
        elif self.velocity_mode == "absolute":
            pred_vel = torch.abs(raw_vel)
            target_vel = torch.abs(gt_vel)
        else:
            raise RuntimeError(f"Unsupported velocity_mode={self.velocity_mode!r}")

        loss_ppp = self.loss_ppp(pred_int, gt_int)

        mask = gt_int.squeeze(1)
        nbr_points = mask.sum(dim=(1, 2)).clamp_min(1)

        loss_depth = (
            (F.l1_loss(pred_depth, gt_depth, reduction="none") * mask).sum(dim=(1, 2))
            / nbr_points
        ).mean()

        loss_velocity = (
            (F.l1_loss(pred_vel, target_vel, reduction="none") * mask).sum(dim=(1, 2))
            / nbr_points
        ).mean()

        return dict(
            loss_ppp=self.ppp_weight * loss_ppp,
            loss_depth=self.depth_weight * loss_depth,
            loss_velocity=self.velocity_weight * loss_velocity,
        )