from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from mmseg.registry import MODELS
from .decode_head import BaseDecodeHead


@MODELS.register_module()
class PPPMarkViTPatchRearrangeHead(BaseDecodeHead):
    """PPP decode head for ViT patch features using patch-to-pixel rearrangement.

    This head consumes a low-resolution spatial ViT feature map and converts it
    into a dense PPP prediction by first increasing the channel dimension and
    then rearranging channels into spatial sub-pixels.

    For example, with patch_size=(16, 16) and num_classes=3:

        B x 384 x 16 x 32
        -> B x (3 * 16 * 16) x 16 x 32
        -> B x 3 x 256 x 512

    Output channels:

        channel 0: intensity logits
        channel 1: depth mark, transformed with exp in the loss
        channel 2: velocity mark, transformed with cube in the loss
    """

    def __init__(
        self,
        patch_size: tuple[int, int] = (16, 16),
        ppp_weight: float = 1.0,
        depth_weight: float = 1.0,
        velocity_weight: float = 1.0,
        loss_ppp: dict = dict(type="PPPLoss"),

        # Kept only so old configs do not immediately break.
        # These arguments are intentionally ignored in this head.
        num_convs: int | None = None,
        num_upsample_layers: int | None = None,
        upsample_channels: tuple[int, ...] | None = None,
        kernel_size: int | None = None,
        dilation: int | None = None,

        **kwargs,
    ):
        self.patch_size = tuple(int(v) for v in patch_size)

        if len(self.patch_size) != 2:
            raise ValueError(f"patch_size must have length 2, got {self.patch_size}")

        self.patch_h, self.patch_w = self.patch_size

        super().__init__(**kwargs)

        if self.num_classes != 3:
            raise ValueError(f"Expected num_classes=3, got {self.num_classes}")

        self.ppp_weight = float(ppp_weight)
        self.depth_weight = float(depth_weight)
        self.velocity_weight = float(velocity_weight)
        self.loss_ppp = MODELS.build(loss_ppp)

        # BaseDecodeHead creates self.conv_seg, but this head does not use the
        # standard segmentation convolution. We replace it to avoid unused
        # trainable parameters.
        self.conv_seg = nn.Identity()

        # Channel up-projection before rearrangement:
        #
        # B x 384 x 16 x 32
        # -> B x (3 * 16 * 16) x 16 x 32
        # -> B x 768 x 16 x 32
        self.patch_proj = nn.Conv2d(
            self.in_channels,
            self.num_classes * self.patch_h * self.patch_w,
            kernel_size=1,
        )

    def forward(self, inputs):
        x = self._transform_inputs(inputs)

        patch_logits = self.patch_proj(x)

        out = rearrange(
            patch_logits,
            "b (c ph pw) h w -> b c (h ph) (w pw)",
            c=self.num_classes,
            ph=self.patch_h,
            pw=self.patch_w,
        )

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

        # Safety fallback. In the intended configuration, this should not be used:
        # the patch rearrangement should already restore the target resolution.
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

'''
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
    

if False: # Jonas
    # (B x 384 x 16 x 32) -> (B x 512, 16, 32) und dann
    einops.reanrange("b (c ph pw) h w -> b c (h ph) (w pw)", ph=16, pw=16)

Note: Since we have three channels (i.e. intensity, depth, velocity), 
we instead perform 

# (B x 384 x 16 x 32) -> (B x 768 x 16 x 32) und dann
einops.rearrange(
    x,
    "b (c ph pw) h w -> b c (h ph) (w pw)",
    c=3,
    ph=16,
    pw=16,
)
'''