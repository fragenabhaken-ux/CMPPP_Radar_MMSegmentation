from __future__ import annotations

import torch
import torch.nn as nn

from mmseg.registry import MODELS


@MODELS.register_module()
class PolarPointPPPLoss(nn.Module):
    """PPP negative log-likelihood for continuous polar point-list targets.

    Ground truth consists only of variable-length point lists:
        {(r_i, theta_i)} for i = 1, ..., N.

    The model defines one continuous intensity field lambda(r, theta).
    During training, that field is evaluated in two ways:

        1. On a fine regular quadrature grid for the integral:
           sum lambda(r_q, theta_q) * dr * dtheta

        2. At the exact continuous GT point coordinates:
           -sum log lambda(r_i, theta_i)

    The quadrature grid is a predicted intensity heatmap only.
    It is not a rasterized GT target.
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

        self.grid_shape = tuple(int(v) for v in grid_shape)
        self.normalize_by_gt_count = bool(normalize_by_gt_count)
        self.loss_weight = float(loss_weight)

        h_r, w_theta = self.grid_shape

        if h_r <= 0 or w_theta <= 0:
            raise ValueError("grid_shape must contain positive dimensions.")

        dr = (r_max - r_min) / h_r
        dtheta = (theta_max - theta_min) / w_theta

        area_weights = torch.full(
            (1, 1, h_r, w_theta),
            dr * dtheta,
            dtype=torch.float32,
        )
        self.register_buffer(
            "area_weights",
            area_weights,
            persistent=False,
        )

    def forward(
        self,
        grid_log_lambda: torch.Tensor,
        point_log_lambda: torch.Tensor,
        batch_idx: torch.Tensor,
        num_points: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the polar PPP negative log-likelihood.

        Args:
            grid_log_lambda:
                Fine predicted quadrature grid, shape [B, 1, H_r, W_theta].

            point_log_lambda:
                Predictions from the same continuous field at exact GT point
                coordinates, shape [M].

            batch_idx:
                Batch assignment of each GT point, shape [M].

            num_points:
                Number of valid GT points per sample, shape [B].
        """
        b = grid_log_lambda.shape[0]
        dtype = grid_log_lambda.dtype
        device = grid_log_lambda.device

        if tuple(grid_log_lambda.shape[-2:]) != self.grid_shape:
            raise ValueError(
                "grid_log_lambda shape does not match configured grid_shape: "
                f"got {tuple(grid_log_lambda.shape[-2:])}, "
                f"expected {self.grid_shape}."
            )

        point_log_lambda = point_log_lambda.to(
            device=device,
            dtype=dtype,
        )
        batch_idx = batch_idx.to(
            device=device,
            dtype=torch.long,
        )
        num_points = num_points.to(
            device=device,
            dtype=dtype,
        )
        area_weights = self.area_weights.to(
            device=device,
            dtype=dtype,
        )

        lambda_grid = torch.exp(grid_log_lambda)

        integral_per_sample = (
            lambda_grid * area_weights
        ).sum(dim=(1, 2, 3))

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

        loss_per_sample = (
            integral_per_sample
            - observation_per_sample
        )

        if self.normalize_by_gt_count:
            loss_per_sample = (
                loss_per_sample
                / num_points.clamp_min(1.0)
            )

        return self.loss_weight * loss_per_sample.mean()