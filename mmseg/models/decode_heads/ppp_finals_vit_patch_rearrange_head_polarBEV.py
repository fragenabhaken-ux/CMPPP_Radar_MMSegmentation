"""Direct dense polar PPP head for Transformer-encoder image features.

The head leaves the ViT token lattice as early as possible:

    camera feature lattice
        -> learned channel projection
        -> lossless pixel rearrangement into a dense feature plane
        -> learned anisotropic CNN compression
        -> adaptive average pooling to the exact polar grid size
        -> dense polar PPP outputs

The dense intermediate plane is not geometrically calibrated BEV. It becomes a
polar-BEV representation through the supervision on the fixed polar target grid.

Output channels:
    0: polar log intensity with respect to dr dtheta
    1: radial velocity mark
    2: RCS mark
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from mmseg.registry import MODELS
from .decode_head import BaseDecodeHead


def _make_norm(norm: str, channels: int) -> nn.Module:
    norm = str(norm).lower()

    if norm == "gn":
        return nn.GroupNorm(1, channels)
    if norm == "bn":
        return nn.BatchNorm2d(channels)
    if norm in {"none", "identity"}:
        return nn.Identity()

    raise ValueError(
        f"Unsupported norm={norm!r}. Use 'gn', 'bn', or 'none'."
    )


class ConvResidualBlock(nn.Module):
    """Local 3x3 residual refinement block."""

    def __init__(self, channels: int, norm: str = "gn") -> None:
        super().__init__()

        channels = int(channels)
        if channels <= 0:
            raise ValueError(f"channels must be positive, got {channels}.")

        self.block = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            _make_norm(norm, channels),
            nn.GELU(),
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            _make_norm(norm, channels),
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.block(x))


class ConvDownsampleResidualBlock(nn.Module):
    """Learned residual downsampling with an anisotropic stride."""

    def __init__(
        self,
        channels: int,
        stride: tuple[int, int],
        norm: str = "gn",
    ) -> None:
        super().__init__()

        channels = int(channels)
        stride = (int(stride[0]), int(stride[1]))

        if channels <= 0:
            raise ValueError(f"channels must be positive, got {channels}.")
        if stride[0] <= 0 or stride[1] <= 0:
            raise ValueError(f"stride must be positive, got {stride}.")
        if stride == (1, 1):
            raise ValueError(
                "ConvDownsampleResidualBlock requires a nontrivial stride."
            )

        self.main = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False,
            ),
            _make_norm(norm, channels),
            nn.GELU(),
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            _make_norm(norm, channels),
        )

        self.skip = nn.Conv2d(
            channels,
            channels,
            kernel_size=1,
            stride=stride,
            bias=False,
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.main(x) + self.skip(x))


@MODELS.register_module()
class PPPPolarViTPatchRearrangeHead(BaseDecodeHead):
    """Predict a marked PPP directly on a fixed dense polar grid.

    The camera-domain Transformer feature lattice is first projected to
    ``dense_channels * rearrange_factor**2`` channels. Pixel rearrangement then
    converts those values losslessly into a much denser two-dimensional feature
    plane. A CNN performs learned anisotropic compression before one final
    adaptive average-pooling step produces the exact polar output resolution.

    Expects ``gt_sem_seg.data`` with shape ``[3, H_r, W_theta]``:
        channel 0: binary occupied-cell indicator
        channel 1: signed wrapped radial velocity
        channel 2: RCS

    For an occupied polar cell, the PPP observation term uses the constant log
    intensity predicted for that cell. The integral uses ``dr * dtheta``.
    """

    def __init__(
        self,
        output_grid_shape: tuple[int, int] = (256, 256),
        rearrange_factor: int = 8,
        dense_channels: int = 64,
        token_refine_blocks: int = 2,
        dense_refine_blocks: int = 2,
        downsample_strides: Sequence[tuple[int, int]] = ((1, 2),),
        post_downsample_refine_blocks: int = 2,
        post_pool_refine_blocks: int = 2,
        norm: str = "gn",
        ppp_weight: float = 1.0,
        velocity_weight: float = 1.0,
        rcs_weight: float = 1.0,
        velocity_mode: str = "signed",
        velocity_link: str = "cube",
        rcs_link: str = "exp",
        r_min: float = 0.0,
        r_max: float = 104.0,
        theta_min: float = -1.5707963267948966,
        theta_max: float = 1.5707963267948966,
        loss_ppp: dict | None = None,
        **kwargs,
    ) -> None:
        self.output_grid_shape = tuple(int(v) for v in output_grid_shape)
        if len(self.output_grid_shape) != 2:
            raise ValueError(
                "output_grid_shape must have length 2, "
                f"got {self.output_grid_shape}."
            )

        self.output_h, self.output_w = self.output_grid_shape
        if self.output_h <= 0 or self.output_w <= 0:
            raise ValueError(
                "output_grid_shape must contain positive dimensions."
            )

        self.rearrange_factor = int(rearrange_factor)
        self.dense_channels = int(dense_channels)
        self.token_refine_blocks = int(token_refine_blocks)
        self.dense_refine_blocks = int(dense_refine_blocks)
        self.post_downsample_refine_blocks = int(
            post_downsample_refine_blocks
        )
        self.post_pool_refine_blocks = int(post_pool_refine_blocks)
        self.norm = str(norm).lower()

        if self.rearrange_factor <= 0:
            raise ValueError(
                "rearrange_factor must be positive, "
                f"got {self.rearrange_factor}."
            )
        if self.dense_channels <= 0:
            raise ValueError(
                f"dense_channels must be positive, got {self.dense_channels}."
            )

        for name, value in (
            ("token_refine_blocks", self.token_refine_blocks),
            ("dense_refine_blocks", self.dense_refine_blocks),
            (
                "post_downsample_refine_blocks",
                self.post_downsample_refine_blocks,
            ),
            ("post_pool_refine_blocks", self.post_pool_refine_blocks),
        ):
            if value < 0:
                raise ValueError(f"{name} must be non-negative, got {value}.")

        parsed_strides: list[tuple[int, int]] = []
        for stride in downsample_strides:
            if len(stride) != 2:
                raise ValueError(
                    "Each downsample stride must contain two integers, "
                    f"got {stride}."
                )
            parsed = (int(stride[0]), int(stride[1]))
            if parsed[0] <= 0 or parsed[1] <= 0:
                raise ValueError(
                    f"Downsample strides must be positive, got {parsed}."
                )
            if parsed == (1, 1):
                raise ValueError(
                    "Use a refinement block instead of stride=(1, 1)."
                )
            parsed_strides.append(parsed)

        self.downsample_strides = tuple(parsed_strides)

        self.r_min = float(r_min)
        self.r_max = float(r_max)
        self.theta_min = float(theta_min)
        self.theta_max = float(theta_max)

        if self.r_max <= self.r_min:
            raise ValueError("r_max must be larger than r_min.")
        if self.theta_max <= self.theta_min:
            raise ValueError("theta_max must be larger than theta_min.")

        super().__init__(**kwargs)

        if self.num_classes != 3:
            raise ValueError(
                "Expected num_classes=3 for [intensity, velocity, RCS], "
                f"got {self.num_classes}."
            )
        if not isinstance(self.in_channels, int):
            raise TypeError(
                "PPPPolarViTPatchRearrangeHead expects one input feature map "
                f"and integer in_channels, got {type(self.in_channels)}."
            )

        self.ppp_weight = float(ppp_weight)
        self.velocity_weight = float(velocity_weight)
        self.rcs_weight = float(rcs_weight)

        self.velocity_mode = str(velocity_mode).lower()
        if self.velocity_mode not in {"signed", "absolute"}:
            raise ValueError(
                "velocity_mode must be 'signed' or 'absolute', "
                f"got {velocity_mode!r}."
            )

        self.velocity_link = str(velocity_link).lower()
        if self.velocity_link not in {"identity", "cube"}:
            raise ValueError(
                "velocity_link must be 'identity' or 'cube', "
                f"got {velocity_link!r}."
            )

        self.rcs_link = str(rcs_link).lower()
        if self.rcs_link not in {"identity", "exp"}:
            raise ValueError(
                "rcs_link must be 'identity' or 'exp', "
                f"got {rcs_link!r}."
            )

        if loss_ppp is None:
            loss_ppp = dict(
                type="PolarPointPPPLoss",
                grid_shape=self.output_grid_shape,
                r_min=self.r_min,
                r_max=self.r_max,
                theta_min=self.theta_min,
                theta_max=self.theta_max,
                normalize_by_gt_count=True,
            )
        self.loss_ppp = MODELS.build(loss_ppp)

        # BaseDecodeHead's semantic-segmentation classifier is not used.
        self.conv_seg = nn.Identity()

        self.token_refine = nn.Sequential(
            *[
                ConvResidualBlock(
                    channels=self.in_channels,
                    norm=self.norm,
                )
                for _ in range(self.token_refine_blocks)
            ]
        )

        # This projection increases channels only so pixel rearrangement can
        # move them losslessly into the spatial dimensions.
        self.dense_projection = nn.Conv2d(
            self.in_channels,
            self.dense_channels
            * self.rearrange_factor
            * self.rearrange_factor,
            kernel_size=1,
        )

        self.dense_refine = nn.Sequential(
            *[
                ConvResidualBlock(
                    channels=self.dense_channels,
                    norm=self.norm,
                )
                for _ in range(self.dense_refine_blocks)
            ]
        )

        self.learned_downsample = nn.Sequential(
            *[
                ConvDownsampleResidualBlock(
                    channels=self.dense_channels,
                    stride=stride,
                    norm=self.norm,
                )
                for stride in self.downsample_strides
            ]
        )

        self.post_downsample_refine = nn.Sequential(
            *[
                ConvResidualBlock(
                    channels=self.dense_channels,
                    norm=self.norm,
                )
                for _ in range(self.post_downsample_refine_blocks)
            ]
        )

        self.output_pool = nn.AdaptiveAvgPool2d(
            self.output_grid_shape
        )

        self.post_pool_refine = nn.Sequential(
            *[
                ConvResidualBlock(
                    channels=self.dense_channels,
                    norm=self.norm,
                )
                for _ in range(self.post_pool_refine_blocks)
            ]
        )

        self.output_projection = nn.Sequential(
            nn.Conv2d(
                self.dense_channels,
                self.dense_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            _make_norm(self.norm, self.dense_channels),
            nn.GELU(),
            nn.Conv2d(
                self.dense_channels,
                self.num_classes,
                kernel_size=1,
            ),
        )

    def forward(self, inputs) -> torch.Tensor:
        x = self._transform_inputs(inputs)
        x = self.token_refine(x)

        # [B, C, H_token, W_token]
        x = self.dense_projection(x)

        # Lossless channel-to-space permutation:
        # [B, dense_channels * f^2, H, W]
        #     -> [B, dense_channels, H*f, W*f]
        x = rearrange(
            x,
            "b (c fh fw) h w -> b c (h fh) (w fw)",
            c=self.dense_channels,
            fh=self.rearrange_factor,
            fw=self.rearrange_factor,
        )

        x = self.dense_refine(x)
        x = self.learned_downsample(x)
        x = self.post_downsample_refine(x)

        # This final fixed operation may only compress. It must never expand
        # the learned dense representation.
        current_h, current_w = x.shape[-2:]
        if current_h < self.output_h or current_w < self.output_w:
            raise RuntimeError(
                "Dense feature plane became smaller than the requested polar "
                "grid before adaptive pooling: "
                f"current={(current_h, current_w)}, "
                f"target={self.output_grid_shape}. "
                "Increase rearrange_factor or reduce downsample strides."
            )

        x = self.output_pool(x)
        x = self.post_pool_refine(x)
        out = self.output_projection(x)

        if tuple(out.shape[-2:]) != self.output_grid_shape:
            raise RuntimeError(
                f"Head produced grid {tuple(out.shape[-2:])}, expected "
                f"{self.output_grid_shape}."
            )

        return out

    @staticmethod
    def _doppler_periods(
        batch_data_samples,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        periods = []

        for index, sample in enumerate(batch_data_samples):
            if "doppler_period" not in sample.metainfo:
                raise KeyError(
                    "Missing data_samples[%d].metainfo['doppler_period']."
                    % index
                )

            period = float(sample.metainfo["doppler_period"])
            if not math.isfinite(period) or period <= 0.0:
                raise ValueError(
                    f"Invalid Doppler period for sample {index}: {period}."
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
        return torch.remainder(
            prediction - target + 0.5 * period,
            period,
        ) - 0.5 * period

    def _velocity_prediction(
        self,
        raw_velocity: torch.Tensor,
    ) -> torch.Tensor:
        if self.velocity_link == "identity":
            return raw_velocity
        if self.velocity_link == "cube":
            return raw_velocity**3

        raise RuntimeError(
            f"Unsupported velocity_link={self.velocity_link!r}."
        )

    def _rcs_prediction(self, raw_rcs: torch.Tensor) -> torch.Tensor:
        if self.rcs_link == "identity":
            return raw_rcs
        if self.rcs_link == "exp":
            return torch.exp(raw_rcs)

        raise RuntimeError(f"Unsupported rcs_link={self.rcs_link!r}.")

    def _validate_sample_geometry(self, batch_data_samples) -> None:
        expected = self.output_grid_shape

        for index, sample in enumerate(batch_data_samples):
            meta = sample.metainfo

            if "polar_grid_shape" in meta:
                got = tuple(int(v) for v in meta["polar_grid_shape"])
                if got != expected:
                    raise ValueError(
                        f"Sample {index} polar_grid_shape={got}, expected "
                        f"{expected}."
                    )

            for key, expected_value in (
                ("polar_r_min", self.r_min),
                ("polar_r_max", self.r_max),
                ("polar_theta_min", self.theta_min),
                ("polar_theta_max", self.theta_max),
            ):
                if key in meta and not math.isclose(
                    float(meta[key]),
                    expected_value,
                    rel_tol=0.0,
                    abs_tol=1e-7,
                ):
                    raise ValueError(
                        f"Sample {index} {key}={meta[key]}, expected "
                        f"{expected_value}."
                    )

    @staticmethod
    def _polar_observation_tensors(
        grid_log_lambda: torch.Tensor,
        occupancy: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extract one PPP observation term per occupied polar cell."""
        mask = occupancy.bool()
        batch_size = grid_log_lambda.shape[0]

        point_log_lambda = grid_log_lambda[:, 0][mask]

        batch_grid = torch.arange(
            batch_size,
            device=grid_log_lambda.device,
            dtype=torch.long,
        ).view(batch_size, 1, 1).expand_as(mask)
        batch_idx = batch_grid[mask]

        num_points = mask.sum(dim=(1, 2)).to(grid_log_lambda.dtype)

        return point_log_lambda, batch_idx, num_points

    def loss(self, inputs, batch_data_samples, train_cfg=None) -> dict:
        del train_cfg

        preds = self.forward(inputs)

        gt = torch.stack(
            [sample.gt_sem_seg.data for sample in batch_data_samples]
        ).to(device=preds.device, dtype=preds.dtype)

        if gt.ndim != 4:
            raise ValueError(
                f"GT must be [B, C, H, W], got {tuple(gt.shape)}."
            )
        if gt.shape[1] != 3:
            raise ValueError(
                "Expected 3 GT channels [occupancy, velocity, RCS], "
                f"got {gt.shape[1]}."
            )
        if gt.shape[0] != preds.shape[0]:
            raise ValueError(
                "Batch size mismatch between predictions and GT."
            )
        if tuple(gt.shape[-2:]) != self.output_grid_shape:
            raise ValueError(
                f"GT polar grid {tuple(gt.shape[-2:])} does not match "
                f"configured output_grid_shape {self.output_grid_shape}."
            )
        if preds.shape != gt.shape:
            raise ValueError(
                f"Prediction shape {tuple(preds.shape)} does not match GT "
                f"shape {tuple(gt.shape)}."
            )

        self._validate_sample_geometry(batch_data_samples)

        gt_intensity = gt[:, 0:1]
        gt_velocity = gt[:, 1]
        gt_rcs = gt[:, 2]

        unique = torch.unique(gt_intensity)
        if not torch.all((unique == 0) | (unique == 1)):
            raise ValueError(
                "Polar occupancy must be binary. "
                f"Found values {unique.tolist()}."
            )

        grid_log_lambda = preds[:, 0:1]
        raw_velocity = preds[:, 1]
        raw_rcs = preds[:, 2]

        occupancy = gt_intensity[:, 0]
        point_log_lambda, batch_idx, num_points = (
            self._polar_observation_tensors(
                grid_log_lambda=grid_log_lambda,
                occupancy=occupancy,
            )
        )

        loss_ppp = self.loss_ppp(
            grid_log_lambda=grid_log_lambda,
            point_log_lambda=point_log_lambda,
            batch_idx=batch_idx,
            num_points=num_points,
        )

        pred_velocity = self._velocity_prediction(raw_velocity)

        if self.velocity_mode == "signed":
            doppler_period = self._doppler_periods(
                batch_data_samples,
                device=preds.device,
                dtype=preds.dtype,
            )
            velocity_loss_per_cell = torch.abs(
                self._circular_velocity_error(
                    pred_velocity,
                    gt_velocity,
                    doppler_period,
                )
            )
        elif self.velocity_mode == "absolute":
            velocity_loss_per_cell = F.l1_loss(
                torch.abs(pred_velocity),
                torch.abs(gt_velocity),
                reduction="none",
            )
        else:
            raise RuntimeError(
                f"Unsupported velocity_mode={self.velocity_mode!r}."
            )

        pred_rcs = self._rcs_prediction(raw_rcs)
        rcs_loss_per_cell = F.l1_loss(
            pred_rcs,
            gt_rcs,
            reduction="none",
        )

        mask = occupancy.to(preds.dtype)
        denominator = num_points.clamp_min(1.0)

        loss_velocity = (
            (velocity_loss_per_cell * mask).sum(dim=(1, 2))
            / denominator
        ).mean()

        loss_rcs = (
            (rcs_loss_per_cell * mask).sum(dim=(1, 2))
            / denominator
        ).mean()

        return dict(
            loss_ppp=self.ppp_weight * loss_ppp,
            loss_velocity=self.velocity_weight * loss_velocity,
            loss_rcs=self.rcs_weight * loss_rcs,
        )

    def predict(
        self,
        inputs,
        batch_img_metas,
        test_cfg=None,
    ) -> torch.Tensor:
        del batch_img_metas, test_cfg
        return self.forward(inputs)