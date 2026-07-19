from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from mmseg.registry import MODELS
from .decode_head import BaseDecodeHead


class LocalConvResidualBlock(nn.Module):
    """Small local 3x3 residual block on the ViT patch feature grid.

    Input/output shape:
        (B, C, H_patch, W_patch)

    This block is intended to add local spatial refinement after the transformer
    neck and before the patch-to-pixel rearrangement.
    """

    def __init__(
        self,
        channels: int,
        norm: str = "gn",
    ):
        super().__init__()

        channels = int(channels)
        norm = str(norm).lower()

        if channels <= 0:
            raise ValueError(f"channels must be positive, got {channels}")

        def make_norm():
            if norm == "gn":
                return nn.GroupNorm(1, channels)
            if norm == "bn":
                return nn.BatchNorm2d(channels)
            if norm in {"none", "identity"}:
                return nn.Identity()
            raise ValueError(
                f"Unsupported local_refine_norm={norm!r}. "
                "Use 'gn', 'bn', or 'none'."
            )

        self.block = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            make_norm(),
            nn.GELU(),
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            make_norm(),
        )

        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.block(x))


@MODELS.register_module()
class PPPfinalsViTPatchRearrangeHead(BaseDecodeHead):
    """PPP decode head for ViT patch features using two-stage patch-to-pixel rearrangement.

    This variant keeps the original disparity-depth behavior by default, but
    adds a selectable depth loss, selectable velocity target, and optional local
    3x3 residual refinement.

    Output channels:
        channel 0: intensity logits
        channel 1: depth mark, interpreted as log-depth
        channel 2: velocity mark
        channel 3: RCS mark, transformed with exp in the loss

    Supported depth losses:
        depth_loss_type="disparity":
            L_depth = |1 / exp(raw_depth) - 1 / depth_gt|

        depth_loss_type="log":
            L_depth = |raw_depth - log(depth_gt)|

        depth_loss_type="metric":
            L_depth = |exp(raw_depth) - depth_gt|

    Supported velocity modes:
        velocity_mode="signed":
            old behavior, signed wrapped LOS velocity
            pred_vel = raw_vel^3
            target_vel = gt_vel

        velocity_mode="absolute":
            absolute wrapped LOS velocity magnitude
            pred_vel = abs(raw_vel)
            target_vel = abs(gt_vel)

    Optional local refinement:
        local_refine_blocks=0:
            old behavior, disabled

        local_refine_blocks=2:
            two local 3x3 residual blocks on the patch feature grid before
            patch-to-pixel rearrangement

    Expects GT in data_samples[i].gt_sem_seg.data with shape (4, H, W):
        channel 0: intensity mask {0, 1}
        channel 1: depth in meters
        channel 2: signed wrapped LOS velocity
        channel 3: RCS

    For velocity_mode="signed", the loss uses the shortest distance on the
    Doppler circle. The frame-specific period is read from
    data_samples[i].metainfo["doppler_period"].
    """

    valid_depth_loss_types = {"disparity", "log", "metric"}

    def __init__(
        self,
        patch_size: tuple[int, int] = (16, 16),
        refine_channels: int = 32,
        num_refine_convs: int = 2,
        pre_factor: int = 4,
        ppp_weight: float = 1.0,
        depth_weight: float = 1.0,
        velocity_weight: float = 1.0,
        rcs_weight: float = 1.0,
        depth_loss_type: str = "disparity",
        velocity_mode: str = "signed",
        depth_disparity_eps: float = 1e-3,
        depth_disparity_max: float = 1.0,
        local_refine_blocks: int = 0,
        local_refine_norm: str = "gn",
        loss_ppp: dict | None = None,
        **kwargs,
    ):
        self.patch_size = tuple(int(v) for v in patch_size)

        if len(self.patch_size) != 2:
            raise ValueError(f"patch_size must have length 2, got {self.patch_size}")

        self.patch_h, self.patch_w = self.patch_size

        self.refine_channels = int(refine_channels)
        self.num_refine_convs = int(num_refine_convs)
        self.pre_factor = int(pre_factor)

        self.local_refine_blocks = int(local_refine_blocks)
        self.local_refine_norm = str(local_refine_norm).lower()

        if self.refine_channels <= 0:
            raise ValueError(
                f"refine_channels must be positive, got {self.refine_channels}"
            )

        if self.num_refine_convs < 0:
            raise ValueError(
                f"num_refine_convs must be non-negative, got {self.num_refine_convs}"
            )

        if self.pre_factor <= 0:
            raise ValueError(f"pre_factor must be positive, got {self.pre_factor}")

        if self.local_refine_blocks < 0:
            raise ValueError(
                f"local_refine_blocks must be non-negative, "
                f"got {self.local_refine_blocks}"
            )

        if self.patch_h % self.pre_factor != 0 or self.patch_w % self.pre_factor != 0:
            raise ValueError(
                f"patch_size={self.patch_size} must be divisible by "
                f"pre_factor={self.pre_factor}."
            )

        self.post_factor_h = self.patch_h // self.pre_factor
        self.post_factor_w = self.patch_w // self.pre_factor

        super().__init__(**kwargs)

        if self.num_classes != 4:
            raise ValueError(f"Expected num_classes=4, got {self.num_classes}")

        if not isinstance(self.in_channels, int):
            raise TypeError(
                "PPPMarkViTPatchRearrangeDisparityDepthHead expects "
                f"self.in_channels to be int, got {type(self.in_channels)}. "
                "Use a single input feature map."
            )

        self.ppp_weight = float(ppp_weight)
        self.depth_weight = float(depth_weight)
        self.velocity_weight = float(velocity_weight)
        self.rcs_weight = float(rcs_weight)

        self.depth_loss_type = str(depth_loss_type).lower()
        if self.depth_loss_type not in self.valid_depth_loss_types:
            raise ValueError(
                f"depth_loss_type must be one of {sorted(self.valid_depth_loss_types)}, "
                f"got {depth_loss_type!r}."
            )

        self.velocity_mode = str(velocity_mode).lower()
        if self.velocity_mode not in {"signed", "absolute"}:
            raise ValueError(
                f"velocity_mode must be 'signed' or 'absolute', got {velocity_mode!r}."
            )

        self.depth_disparity_eps = float(depth_disparity_eps)
        self.depth_disparity_max = float(depth_disparity_max)

        if self.depth_disparity_eps <= 0:
            raise ValueError(
                f"depth_disparity_eps must be > 0, got {self.depth_disparity_eps}"
            )

        if self.depth_disparity_max <= 0:
            raise ValueError(
                f"depth_disparity_max must be > 0, got {self.depth_disparity_max}"
            )

        if loss_ppp is None:
            loss_ppp = dict(type="PPPLoss")

        self.loss_ppp = MODELS.build(loss_ppp)

        # BaseDecodeHead normally creates conv_seg. We do not use it here because
        # this head has its own custom output path.
        self.conv_seg = nn.Identity()

        if self.local_refine_blocks > 0:
            self.local_refine = nn.Sequential(
                *[
                    LocalConvResidualBlock(
                        channels=self.in_channels,
                        norm=self.local_refine_norm,
                    )
                    for _ in range(self.local_refine_blocks)
                ]
            )
        else:
            self.local_refine = nn.Identity()

        self.patch_proj = nn.Conv2d(
            self.in_channels,
            self.refine_channels * self.pre_factor * self.pre_factor,
            kernel_size=1,
        )

        pre_refine_layers = []
        for _ in range(self.num_refine_convs):
            pre_refine_layers.extend(
                [
                    nn.Conv2d(
                        self.refine_channels,
                        self.refine_channels,
                        kernel_size=3,
                        padding=1,
                    ),
                    nn.BatchNorm2d(self.refine_channels),
                    nn.GELU(),
                ]
            )

        pre_refine_layers.append(
            nn.Conv2d(
                self.refine_channels,
                self.refine_channels * self.post_factor_h * self.post_factor_w,
                kernel_size=1,
            )
        )

        self.pre_refine = nn.Sequential(*pre_refine_layers)

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

        # With local_refine_blocks=0, this is nn.Identity() and keeps old behavior.
        x = self.local_refine(x)

        pixel_features = self.patch_proj(x)

        pre_pixel_features = rearrange(
            pixel_features,
            "b (c ph pw) h w -> b c (h ph) (w pw)",
            c=self.refine_channels,
            ph=self.pre_factor,
            pw=self.pre_factor,
        )

        mixed_pre_pixel = self.pre_refine(pre_pixel_features)

        pixel_features = rearrange(
            mixed_pre_pixel,
            "b (c ph pw) h w -> b c (h ph) (w pw)",
            c=self.refine_channels,
            ph=self.post_factor_h,
            pw=self.post_factor_w,
        )

        out = self.refine(pixel_features)

        return out

    def _depth_loss_per_pixel(
        self,
        raw_depth: torch.Tensor,
        gt_depth: torch.Tensor,
    ) -> torch.Tensor:
        """Compute unmasked per-pixel depth loss.

        raw_depth is the direct network output. It is interpreted as log-depth.
        gt_depth is metric depth in meters.
        """
        gt_depth_safe = torch.clamp(gt_depth, min=self.depth_disparity_eps)

        if self.depth_loss_type == "log":
            gt_log_depth = torch.log(gt_depth_safe)
            return F.l1_loss(raw_depth, gt_log_depth, reduction="none")

        pred_depth = torch.exp(raw_depth)

        if self.depth_loss_type == "metric":
            return F.l1_loss(pred_depth, gt_depth, reduction="none")

        if self.depth_loss_type == "disparity":
            pred_depth_safe = torch.clamp(pred_depth, min=self.depth_disparity_eps)

            disp_pred = 1.0 / pred_depth_safe
            disp_gt = 1.0 / gt_depth_safe

            disp_pred = torch.clamp(disp_pred, max=self.depth_disparity_max)
            disp_gt = torch.clamp(disp_gt, max=self.depth_disparity_max)

            return F.l1_loss(disp_pred, disp_gt, reduction="none")

        raise RuntimeError(f"Unsupported depth_loss_type={self.depth_loss_type!r}")

    @staticmethod
    def _doppler_periods(
        batch_data_samples,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Return one validated Doppler period per batch element."""
        periods = []
        for index, sample in enumerate(batch_data_samples):
            metainfo = sample.metainfo
            if "doppler_period" not in metainfo:
                raise KeyError(
                    "Missing data_samples[%d].metainfo['doppler_period']; "
                    "the transform must pass the frame-specific Doppler period."
                    % index
                )

            period = float(metainfo["doppler_period"])
            if not math.isfinite(period) or period <= 0.0:
                raise ValueError(
                    f"Invalid Doppler period for sample {index}: {period}"
                )
            periods.append(period)

        return torch.tensor(
            periods,
            device=device,
            dtype=dtype,
        ).view(-1, 1, 1)

    @staticmethod
    def _circular_velocity_error(
        prediction: torch.Tensor,
        target: torch.Tensor,
        period: torch.Tensor,
    ) -> torch.Tensor:
        """Signed shortest difference on a circle with the given period."""
        return torch.remainder(
            prediction - target + 0.5 * period,
            period,
        ) - 0.5 * period

    def loss(self, inputs, batch_data_samples, train_cfg=None):
        preds = self.forward(inputs)

        gt = torch.stack(
            [sample.gt_sem_seg.data for sample in batch_data_samples]
        ).to(preds.device)

        if gt.ndim != 4:
            raise ValueError(f"GT must be (N,C,H,W), got {tuple(gt.shape)}")
        if gt.shape[1] != 4:
            raise ValueError(f"Expected 4 GT channels, got {gt.shape[1]}")
        if gt.shape[0] != preds.shape[0]:
            raise ValueError("Batch size mismatch between preds and GT")

        if preds.shape[2:] != gt.shape[2:]:
            raise ValueError(
                f"ViT head output shape {preds.shape[2:]} != GT shape {gt.shape[2:]}. "
                "Do not interpolate PPP intensity logits silently."
            )

        gt_int = gt[:, 0:1]
        gt_depth = gt[:, 1]
        gt_vel = gt[:, 2]
        gt_rcs = gt[:, 3]

        unique_vals = torch.unique(gt_int)
        if not torch.all((unique_vals == 0) | (unique_vals == 1)):
            raise ValueError(f"Intensity channel is not binary, values={unique_vals}")

        pred_int = preds[:, 0:1]
        raw_depth = preds[:, 1]
        raw_vel = preds[:, 2]
        raw_rcs = preds[:, 3]

        if self.velocity_mode == "signed":
            pred_vel = raw_vel ** 3
            target_vel = gt_vel
            doppler_period = self._doppler_periods(
                batch_data_samples,
                device=preds.device,
                dtype=preds.dtype,
            )
            velocity_loss_per_pixel = torch.abs(
                self._circular_velocity_error(
                    pred_vel,
                    target_vel,
                    doppler_period,
                )
            )
        elif self.velocity_mode == "absolute":
            pred_vel = torch.abs(raw_vel)
            target_vel = torch.abs(gt_vel)
            velocity_loss_per_pixel = F.l1_loss(
                pred_vel,
                target_vel,
                reduction="none",
            )
        else:
            raise RuntimeError(f"Unsupported velocity_mode={self.velocity_mode!r}")

        pred_rcs = raw_rcs**3

        loss_ppp = self.loss_ppp(pred_int, gt_int)

        mask = gt_int.squeeze(1)
        nbr_points = mask.sum(dim=(1, 2)).clamp_min(1)

        loss_depth_per_pixel = self._depth_loss_per_pixel(raw_depth, gt_depth)
        loss_depth = (
            (loss_depth_per_pixel * mask).sum(dim=(1, 2))
            / nbr_points
        ).mean()

        loss_velocity = (
            (velocity_loss_per_pixel * mask).sum(dim=(1, 2))
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