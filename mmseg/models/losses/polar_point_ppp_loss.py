from __future__ import annotations

import torch
import torch.nn as nn

from mmseg.registry import MODELS


@MODELS.register_module()
class PolarPointPPPLoss(nn.Module):
    """PPP negative log-likelihood for point-list targets in polar space.

    The model intensity is defined with respect to polar measure dr dtheta:

        E[N] = integral lambda(r, theta) dr dtheta

    Therefore the integral term is approximated by

        sum exp(log_lambda_grid) * dr * dtheta

    There is no target heatmap. GT points only enter through the observation term:

        - sum_i log lambda(r_i, theta_i)
    """

    def __init__(
        self,
        grid_shape: tuple[int, int] = (64, 64),
        r_min: float = 0.0,
        r_max: float = 104.0,
        theta_min: float = -1.5707963267948966,  # -pi / 2
        theta_max: float = 1.5707963267948966,   #  pi / 2
        normalize_by_gt_count: bool = True,
        loss_weight: float = 1.0,
        max_log_lambda: float | None = 8.0,
    ) -> None:
        super().__init__()

        self.grid_shape = grid_shape
        self.normalize_by_gt_count = normalize_by_gt_count
        self.loss_weight = loss_weight
        self.max_log_lambda = max_log_lambda

        h_r, w_theta = grid_shape
        dr = (r_max - r_min) / h_r
        dtheta = (theta_max - theta_min) / w_theta

        area_weights = torch.full(
            (1, 1, h_r, w_theta),
            dr * dtheta,
            dtype=torch.float32,
        )

        self.register_buffer("area_weights", area_weights, persistent=False)

    def forward(
        self,
        grid_log_lambda: torch.Tensor,
        point_log_lambda: torch.Tensor,
        batch_idx: torch.Tensor,
        num_points: torch.Tensor,
    ) -> torch.Tensor:
        """Compute polar PPP loss.

        Args:
            grid_log_lambda: Tensor[B, 1, H_r, W_theta]
            point_log_lambda: Tensor[M]
            batch_idx: Tensor[M]
            num_points: Tensor[B]
        """
        b = grid_log_lambda.shape[0]
        dtype = grid_log_lambda.dtype
        device = grid_log_lambda.device

        point_log_lambda = point_log_lambda.to(device=device, dtype=dtype)
        batch_idx = batch_idx.to(device=device, dtype=torch.long)
        num_points = num_points.to(device=device, dtype=dtype)
        area_weights = self.area_weights.to(device=device, dtype=dtype)

        if self.max_log_lambda is not None:
            grid_log_lambda_exp = grid_log_lambda.clamp(
                max=self.max_log_lambda,
            )
            point_log_lambda = point_log_lambda.clamp(
                max=self.max_log_lambda,
            )
        else:
            grid_log_lambda_exp = grid_log_lambda

        lambda_grid = torch.exp(grid_log_lambda_exp)

        integral_per_sample = (lambda_grid * area_weights).sum(dim=(1, 2, 3))

        observation_per_sample = torch.zeros(
            b,
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
            loss_per_sample = loss_per_sample / num_points.clamp_min(1.0)

        return self.loss_weight * loss_per_sample.mean()