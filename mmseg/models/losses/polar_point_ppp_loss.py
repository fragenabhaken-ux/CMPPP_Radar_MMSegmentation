from __future__ import annotations

import torch
import torch.nn as nn

from mmseg.registry import MODELS


@MODELS.register_module()
class PolarPointPPPLoss(nn.Module):
    """Negative log-likelihood of a polar Poisson point process.

    The intensity is defined with respect to the polar measure ``dr dtheta``.
    For a fixed quadrature grid, the per-sample loss is

        sum_ij exp(log_lambda_ij) * dr * dtheta
        - sum_k log_lambda(r_k, theta_k).

    No target heatmap and no clipping of ``log_lambda`` are used.
    """

    def __init__(
        self,
        grid_shape: tuple[int, int] = (256, 256),
        r_min: float = 0.0,
        r_max: float = 104.0,
        theta_min: float = -1.5707963267948966,
        theta_max: float = 1.5707963267948966,
        normalize_by_gt_count: bool = True,
        loss_weight: float = 1.0,
    ) -> None:
        super().__init__()

        h_r, w_theta = (int(grid_shape[0]), int(grid_shape[1]))
        if h_r <= 0 or w_theta <= 0:
            raise ValueError('grid_shape must contain positive dimensions.')
        if r_max <= r_min:
            raise ValueError('r_max must be larger than r_min.')
        if theta_max <= theta_min:
            raise ValueError('theta_max must be larger than theta_min.')

        self.grid_shape = (h_r, w_theta)
        self.normalize_by_gt_count = bool(normalize_by_gt_count)
        self.loss_weight = float(loss_weight)

        dr = (float(r_max) - float(r_min)) / h_r
        dtheta = (float(theta_max) - float(theta_min)) / w_theta
        self.register_buffer(
            'cell_measure',
            torch.tensor(dr * dtheta, dtype=torch.float32),
            persistent=False,
        )

    def forward(
        self,
        grid_log_lambda: torch.Tensor,
        point_log_lambda: torch.Tensor,
        batch_idx: torch.Tensor,
        num_points: torch.Tensor,
    ) -> torch.Tensor:
        if grid_log_lambda.ndim != 4 or grid_log_lambda.shape[1] != 1:
            raise ValueError(
                'grid_log_lambda must have shape [B, 1, H_r, W_theta], '
                f'got {tuple(grid_log_lambda.shape)}.'
            )
        if tuple(grid_log_lambda.shape[-2:]) != self.grid_shape:
            raise ValueError(
                'grid_log_lambda grid shape does not match configured grid_shape: '
                f'got {tuple(grid_log_lambda.shape[-2:])}, '
                f'expected {self.grid_shape}.'
            )

        batch_size = grid_log_lambda.shape[0]
        dtype = grid_log_lambda.dtype
        device = grid_log_lambda.device

        point_log_lambda = point_log_lambda.to(device=device, dtype=dtype)
        batch_idx = batch_idx.to(device=device, dtype=torch.long)
        num_points = num_points.to(device=device, dtype=dtype)

        if num_points.shape != (batch_size,):
            raise ValueError(
                f'num_points must have shape [{batch_size}], '
                f'got {tuple(num_points.shape)}.'
            )
        if point_log_lambda.numel() != batch_idx.numel():
            raise ValueError('point_log_lambda and batch_idx must have equal length.')

        integral_per_sample = (
            torch.exp(grid_log_lambda).sum(dim=(1, 2, 3))
            * self.cell_measure.to(device=device, dtype=dtype)
        )

        observation_per_sample = torch.zeros(
            batch_size,
            device=device,
            dtype=dtype,
        )
        if point_log_lambda.numel() > 0:
            observation_per_sample.scatter_add_(
                dim=0,
                index=batch_idx,
                src=point_log_lambda,
            )

        loss_per_sample = integral_per_sample - observation_per_sample
        if self.normalize_by_gt_count:
            loss_per_sample = (
                loss_per_sample
                / num_points.clamp_min(1.0)
            )

        return self.loss_weight * loss_per_sample.mean()