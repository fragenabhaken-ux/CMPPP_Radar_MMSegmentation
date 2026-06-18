from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from mmseg.registry import MODELS
from .decode_head import BaseDecodeHead


@MODELS.register_module()
class PPPMarkViTPatchRearrangeHead(BaseDecodeHead):
    """PPP decode head for ViT patch features using two-stage patch-to-pixel rearrangement.

    This head consumes a low-resolution spatial ViT feature map and converts it
    into a dense PPP prediction. The spatial expansion is performed by rearranging
    channel groups into spatial sub-pixels. Between the two rearrangement steps, a
    small convolutional refinement block mixes neighbouring intermediate pixels.

    For example, with patch_size=(16, 16), refine_channels=32, pre_factor=4 and
    num_classes=3:
    For training:
        B x 384 x 16 x 32
        -> B x (32 * 4 * 4) x 16 x 32 = B x 512 x 16 x 32
        -> B x 32 x 64 x 128
        -> B x (32 * 4 * 4) x 64 x 128 = B x 512 x 64 x 128
        -> B x 32 x 256 x 512
        -> B x 3 x 256 x 512
    For validation:
        B x 384 x 34 x 112
        -> B x (32 * 4 * 4) x 34 x 112 = B x 512 x 34 x 112
        -> B x 32 x 136 x 448
        -> B x (32 * 4 * 4) x 136 x 448 = B x 512 x 136 x 448
        -> B x 32 x 544 x 1792
        -> B x 3 x 544 x 1792
    Output channels:

        channel 0: intensity logits
        channel 1: depth mark, transformed with exp in the loss
        channel 2: velocity mark, transformed with cube in the loss
    """

    def __init__(
        self,
        patch_size: tuple[int, int] = (16, 16),
        refine_channels: int = 32,
        num_refine_convs: int = 2,
        pre_factor: int = 4,
        ppp_weight: float = 1.0,
        depth_weight: float = 1.0,
        velocity_weight: float = 1.0,
        velocity_mode: str = "signed",
        loss_ppp: dict = dict(type="PPPLoss"),
        **kwargs,
    ):
        self.patch_size = tuple(int(v) for v in patch_size)

        if len(self.patch_size) != 2:
            raise ValueError(f"patch_size must have length 2, got {self.patch_size}")

        self.patch_h, self.patch_w = self.patch_size

        self.refine_channels = int(refine_channels)
        self.num_refine_convs = int(num_refine_convs)
        self.pre_factor = int(pre_factor)

        if self.patch_h % self.pre_factor != 0 or self.patch_w % self.pre_factor != 0:
            raise ValueError(
                f"patch_size={self.patch_size} must be divisible by "
                f"pre_factor={self.pre_factor}."
            )

        self.post_factor_h = self.patch_h // self.pre_factor
        self.post_factor_w = self.patch_w // self.pre_factor

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

        # BaseDecodeHead creates self.conv_seg, but this head does not use the
        # standard segmentation convolution. We replace it to avoid unused
        # trainable parameters.
        self.conv_seg = nn.Identity()

        # First patch-to-pixel expansion:
        #
        # With patch_size=(16, 16), pre_factor=4, refine_channels=32:
        # B x 384 x 16 x 32
        # -> B x (32 * 4 * 4) x 16 x 32
        # -> B x 32 x 64 x 128
        self.patch_proj = nn.Conv2d(
            self.in_channels,
            self.refine_channels * self.pre_factor * self.pre_factor,
            kernel_size=1,
        )

        pre_refine_layers = []
        for _ in range(self.num_refine_convs):
            pre_refine_layers.extend([
                nn.Conv2d(
                    self.refine_channels,
                    self.refine_channels,
                    kernel_size=3,
                    padding=1,
                ),
                nn.BatchNorm2d(self.refine_channels),
                nn.GELU(),
            ])

        # Prepare channels for the second patch-to-pixel expansion:
        #
        # B x 32 x 64 x 128
        # -> B x (32 * 4 * 4) x 64 x 128
        pre_refine_layers.append(
            nn.Conv2d(
                self.refine_channels,
                self.refine_channels * self.post_factor_h * self.post_factor_w,
                kernel_size=1,
            )
        )

        self.pre_refine = nn.Sequential(*pre_refine_layers)

        # Final local refinement and projection:
        #
        # B x 32 x 256 x 512
        # -> B x 3 x 256 x 512
        self.refine = nn.Sequential(
            nn.Conv2d(
                self.refine_channels,
                self.refine_channels,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm2d(self.refine_channels),
            nn.GELU(),
            nn.Conv2d(
                self.refine_channels,
                self.num_classes,
                kernel_size=1,
            ),
        )


    def forward(self, inputs):
        x = self._transform_inputs(inputs)

        pixel_features = self.patch_proj(x)
        # B x (C_ref * pre_factor * pre_factor) x H x W

        pre_pixel_features = rearrange(
            pixel_features,
            "b (c ph pw) h w -> b c (h ph) (w pw)",
            c=self.refine_channels,
            ph=self.pre_factor,
            pw=self.pre_factor,
        )
        # B x C_ref x (H * pre_factor) x (W * pre_factor)

        mixed_pre_pixel = self.pre_refine(pre_pixel_features)
        # B x (C_ref * post_factor_h * post_factor_w)
        #   x (H * pre_factor) x (W * pre_factor)

        pixel_features = rearrange(
            mixed_pre_pixel,
            "b (c ph pw) h w -> b c (h ph) (w pw)",
            c=self.refine_channels,
            ph=self.post_factor_h,
            pw=self.post_factor_w,
        )
        # B x C_ref x (H * patch_h) x (W * patch_w)

        out = self.refine(pixel_features)
        # B x 3 x (H * patch_h) x (W * patch_w)

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

        # # Safety fallback. In the intended configuration, this should not be used:
        # # the patch rearrangement should already restore the target resolution.
        # if preds.shape[2:] != gt.shape[2:]:
        #     preds = F.interpolate(
        #         preds,
        #         size=gt.shape[2:],
        #         mode="bilinear",
        #         align_corners=self.align_corners,
        #     )
        # Was the above code the cause of the high expected error bug? Here is a hard error:
        if preds.shape[2:] != gt.shape[2:]:
            raise ValueError(
                f"ViT head output shape {preds.shape[2:]} != GT shape {gt.shape[2:]}. "
                "Do not interpolate PPP intensity logits silently."
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