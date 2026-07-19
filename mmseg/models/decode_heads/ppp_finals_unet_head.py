from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule

from mmseg.registry import MODELS
from .decode_head import BaseDecodeHead


@MODELS.register_module()
class PPPfinalsUNetDepthLossHead(BaseDecodeHead):
    """PPP mark head for MMSeg UNet with selectable depth loss.

    This head follows the original PPPMarkUNetHead structure, but adds a
    configurable depth loss.

    Output channels:
        channel 0: intensity logits
        channel 1: depth mark, interpreted as log-depth
        channel 2: velocity mark, transformed with cube in the loss
        channel 3: RCS mark, transformed with exp in the loss

    Supported depth losses:
        depth_loss_type="log":
            L_depth = |raw_depth - log(depth_gt)|

        depth_loss_type="disparity":
            L_depth = |1 / exp(raw_depth) - 1 / depth_gt|

        depth_loss_type="metric":
            L_depth = |exp(raw_depth) - depth_gt|
            This is the previous behavior and is kept for compatibility.

    Expects GT in data_samples[i].gt_sem_seg.data with shape (4, H, W):
        channel 0: intensity mask {0, 1}
        channel 1: depth in meters
        channel 2: velocity
        channel 3: RCS
    """

    valid_depth_loss_types = {"log", "disparity", "metric"}

    def __init__(
        self,
        num_convs: int = 1,
        kernel_size: int = 3,
        dilation: int = 1,
        ppp_weight: float = 1.0,
        depth_weight: float = 1.0,
        velocity_weight: float = 1.0,
        rcs_weight: float = 1.0,
        velocity_mode: str = "signed",
        depth_loss_type: str = "log",
        depth_eps: float = 1e-3,
        depth_disparity_max: float = 1.0,
        loss_ppp: dict = dict(type="PPPLoss"),
        **kwargs,
    ):
        assert num_convs >= 0 and dilation > 0 and isinstance(dilation, int)

        self.num_convs = int(num_convs)
        self.kernel_size = int(kernel_size)

        super().__init__(**kwargs)

        if self.num_classes != 4:
            raise ValueError(f"Expected num_classes=4, got {self.num_classes}")
        if self.num_convs == 0 and self.in_channels != self.channels:
            raise ValueError(
                "If num_convs=0, in_channels must equal channels. "
                f"Got in_channels={self.in_channels}, channels={self.channels}."
            )

        self.ppp_weight = float(ppp_weight)
        self.depth_weight = float(depth_weight)
        self.velocity_weight = float(velocity_weight)
        self.rcs_weight = float(rcs_weight)

        self.velocity_mode = str(velocity_mode).lower()
        if self.velocity_mode not in {"signed", "absolute"}:
            raise ValueError(
                f"velocity_mode must be 'signed' or 'absolute', got {velocity_mode!r}."
            )

        self.depth_loss_type = str(depth_loss_type).lower()
        if self.depth_loss_type not in self.valid_depth_loss_types:
            raise ValueError(
                f"depth_loss_type must be one of {sorted(self.valid_depth_loss_types)}, "
                f"got {depth_loss_type!r}."
            )

        self.depth_eps = float(depth_eps)
        if self.depth_eps <= 0:
            raise ValueError(f"depth_eps must be > 0, got {self.depth_eps}")

        self.depth_disparity_max = float(depth_disparity_max)
        if self.depth_disparity_max <= 0:
            raise ValueError(
                "depth_disparity_max must be > 0, "
                f"got {self.depth_disparity_max}"
            )

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

    def _depth_loss_per_pixel(
        self,
        raw_depth: torch.Tensor,
        gt_depth: torch.Tensor,
    ) -> torch.Tensor:
        """Compute unmasked per-pixel depth loss.

        raw_depth is the direct network output. It is interpreted as log-depth.
        gt_depth is metric depth in meters.
        """
        gt_depth_safe = torch.clamp(gt_depth, min=self.depth_eps)

        if self.depth_loss_type == "log":
            gt_log_depth = torch.log(gt_depth_safe)
            return F.l1_loss(raw_depth, gt_log_depth, reduction="none")

        pred_depth = torch.exp(raw_depth)

        if self.depth_loss_type == "metric":
            return F.l1_loss(pred_depth, gt_depth, reduction="none")

        if self.depth_loss_type == "disparity":
            pred_depth_safe = torch.clamp(pred_depth, min=self.depth_eps)

            disp_pred = 1.0 / pred_depth_safe
            disp_gt = 1.0 / gt_depth_safe

            # Avoid exploding gradients if predicted or GT depth becomes too
            # close to zero.
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
            if not torch.isfinite(
                torch.tensor(period, dtype=torch.float64)
            ).item() or period <= 0:
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
        """Signed shortest difference on a circle with the given period."""
        return torch.remainder(
            pred_vel - target_vel + 0.5 * doppler_period,
            doppler_period,
        ) - 0.5 * doppler_period

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
            preds = F.interpolate(
                preds,
                size=gt.shape[2:],
                mode="bilinear",
                align_corners=self.align_corners,
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
                device=pred_vel.device,
                dtype=pred_vel.dtype,
            )
            velocity_error = self._circular_velocity_error(
                pred_vel,
                target_vel,
                doppler_period,
            )
            loss_velocity_per_pixel = torch.abs(velocity_error)
        elif self.velocity_mode == "absolute":
            pred_vel = torch.abs(raw_vel)
            target_vel = torch.abs(gt_vel)
            loss_velocity_per_pixel = F.l1_loss(
                pred_vel,
                target_vel,
                reduction="none",
            )
        else:
            raise RuntimeError(f"Unsupported velocity_mode={self.velocity_mode!r}")

        pred_rcs = torch.exp(raw_rcs)

        loss_ppp = self.loss_ppp(pred_int, gt_int)

        mask = gt_int.squeeze(1)
        nbr_points = mask.sum(dim=(1, 2)).clamp_min(1)

        loss_depth_per_pixel = self._depth_loss_per_pixel(raw_depth, gt_depth)
        loss_depth = (
            (loss_depth_per_pixel * mask).sum(dim=(1, 2))
            / nbr_points
        ).mean()

        loss_velocity = (
            (loss_velocity_per_pixel * mask).sum(dim=(1, 2))
            / nbr_points
        ).mean()

        loss_rcs = (
            (F.l1_loss(pred_rcs, gt_rcs, reduction="none") * mask).sum(dim=(1, 2))
            / nbr_points
        ).mean()

        return dict(
            loss_ppp=self.ppp_weight * loss_ppp,
            loss_depth=self.depth_weight * loss_depth,
            loss_velocity=self.velocity_weight * loss_velocity,
            loss_rcs=self.rcs_weight * loss_rcs,
        )