from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmseg.registry import MODELS
from .decode_head import BaseDecodeHead


@MODELS.register_module()
class PolarLIIFPPPHead(BaseDecodeHead):
    """Polar LIIF PPP head.

    The head predicts a continuous polar intensity lambda(r, theta).

    Training:
      1. Build a 64x64 polar latent grid from image features.
      2. Query LIIF on the 64x64 polar grid for the PPP integral.
      3. Assign each GT radar point to its polar grid cell. Use that same cell intensity for the observation term.
      4. Apply PolarPointPPPLoss.

    The point process is defined with respect to polar measure dr dtheta.
    """

    def __init__(
        self,
        polar_grid_shape: tuple[int, int] = (64, 64),
        r_min: float = 0.0,
        r_max: float = 104.0,
        theta_min: float = -1.5707963267948966,  # -pi / 2
        theta_max: float = 1.5707963267948966,   #  pi / 2
        num_latent_convs: int = 2,
        liif_hidden_channels: int = 128,
        liif_num_layers: int = 3,
        loss_ppp: dict | None = None,
        ppp_weight: float = 1.0,
        velocity_weight: float = 0.0,
        velocity_mode: str = "signed",
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        self.polar_grid_shape = tuple(int(v) for v in polar_grid_shape)
        self.r_min = float(r_min)
        self.r_max = float(r_max)
        self.theta_min = float(theta_min)
        self.theta_max = float(theta_max)

        self.ppp_weight = float(ppp_weight)
        self.velocity_weight = float(velocity_weight)
        self.velocity_mode = str(velocity_mode).lower()

        # BaseDecodeHead creates conv_seg for segmentation.
        # We do not use it here.
        self.conv_seg = nn.Identity()

        latent_layers: list[nn.Module] = [
            nn.Conv2d(self.in_channels, self.channels, kernel_size=1),
            nn.GroupNorm(1, self.channels),
            nn.GELU(),
        ]

        for _ in range(int(num_latent_convs)):
            latent_layers.extend(
                [
                    nn.Conv2d(
                        self.channels,
                        self.channels,
                        kernel_size=3,
                        padding=1,
                    ),
                    nn.GroupNorm(1, self.channels),
                    nn.GELU(),
                ]
            )

        self.latent_proj = nn.Sequential(*latent_layers)

        # LIIF input = local latent feature + normalized query coordinate (r, theta).
        liif_in_channels = self.channels + 2
        liif_layers: list[nn.Module] = []

        last_channels = liif_in_channels
        for _ in range(max(int(liif_num_layers) - 1, 0)):
            liif_layers.extend(
                [
                    nn.Linear(last_channels, liif_hidden_channels),
                    nn.GELU(),
                ]
            )
            last_channels = liif_hidden_channels

        # Output channels:
        #   0: log lambda(r, theta)
        #   1: velocity mark
        liif_layers.append(nn.Linear(last_channels, 2))
        self.liif_mlp = nn.Sequential(*liif_layers)

        if loss_ppp is None:
            loss_ppp = dict(
                type="PolarPointPPPLoss",
                grid_shape=self.polar_grid_shape,
                r_min=self.r_min,
                r_max=self.r_max,
                theta_min=self.theta_min,
                theta_max=self.theta_max,
                normalize_by_gt_count=True,
            )

        self.loss_ppp = MODELS.build(loss_ppp)

        grid_rtheta, grid_norm = self._make_polar_grid()
        self.register_buffer("grid_rtheta", grid_rtheta, persistent=False)
        self.register_buffer("grid_norm", grid_norm, persistent=False)

    def _make_polar_grid(self) -> tuple[torch.Tensor, torch.Tensor]:
        h_r, w_theta = self.polar_grid_shape

        dr = (self.r_max - self.r_min) / h_r
        dtheta = (self.theta_max - self.theta_min) / w_theta

        r = torch.linspace(
            self.r_min + 0.5 * dr,
            self.r_max - 0.5 * dr,
            h_r,
            dtype=torch.float32,
        )
        theta = torch.linspace(
            self.theta_min + 0.5 * dtheta,
            self.theta_max - 0.5 * dtheta,
            w_theta,
            dtype=torch.float32,
        )

        rr, tt = torch.meshgrid(r, theta, indexing="ij")
        grid_rtheta = torch.stack([rr, tt], dim=-1)  # [H_r, W_theta, 2]

        r_norm = 2.0 * (rr - self.r_min) / (self.r_max - self.r_min) - 1.0
        theta_norm = (
            2.0 * (tt - self.theta_min) / (self.theta_max - self.theta_min)
            - 1.0
        )
        grid_norm = torch.stack([r_norm, theta_norm], dim=-1)

        return grid_rtheta, grid_norm

    def _normalize_points(self, points_rtheta: torch.Tensor) -> torch.Tensor:
        r = points_rtheta[:, 0]
        theta = points_rtheta[:, 1]

        r_norm = 2.0 * (r - self.r_min) / (self.r_max - self.r_min) - 1.0
        theta_norm = (
            2.0 * (theta - self.theta_min) / (self.theta_max - self.theta_min)
            - 1.0
        )

        return torch.stack([r_norm, theta_norm], dim=-1)

    def _build_polar_latent(self, inputs) -> torch.Tensor:
        x = self._transform_inputs(inputs)
        x = self.latent_proj(x)

        x = F.interpolate(
            x,
            size=self.polar_grid_shape,
            mode="bilinear",
            align_corners=False,
        )

        return x

    def _query_grid(self, polar_latent: torch.Tensor) -> torch.Tensor:
        b, c, h_r, w_theta = polar_latent.shape

        coords_norm = self.grid_norm.to(
            device=polar_latent.device,
            dtype=polar_latent.dtype,
        )
        coords_norm = coords_norm.view(1, h_r * w_theta, 2).expand(b, -1, -1)

        # grid_sample expects coordinate order (x, y) = (theta, r).
        sample_grid = torch.stack(
            [coords_norm[..., 1], coords_norm[..., 0]],
            dim=-1,
        )
        sample_grid = sample_grid.view(b, h_r * w_theta, 1, 2)

        sampled = F.grid_sample(
            polar_latent,
            sample_grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        )
        sampled = sampled.squeeze(-1).permute(0, 2, 1).contiguous()

        liif_in = torch.cat([sampled, coords_norm], dim=-1)
        liif_out = self.liif_mlp(liif_in)

        liif_out = liif_out.permute(0, 2, 1).contiguous()
        liif_out = liif_out.view(b, 2, h_r, w_theta)

        return liif_out
    
    def _gather_grid_log_lambda(
        self,
        grid_log_lambda: torch.Tensor,
        points_rtheta: torch.Tensor,
        batch_idx: torch.Tensor,
    ) -> torch.Tensor:
        """Return the log intensity of the polar cell containing each GT point.

        The PPP intensity is treated as piecewise constant on the polar grid.
        This makes the observation term consistent with the grid-based integral.
        """
        if points_rtheta.numel() == 0:
            return grid_log_lambda.new_zeros((0,))

        h_r, w_theta = self.polar_grid_shape

        r = points_rtheta[:, 0]
        theta = points_rtheta[:, 1]

        r_idx = torch.floor(
            (r - self.r_min) / (self.r_max - self.r_min) * h_r
        ).long()

        theta_idx = torch.floor(
            (theta - self.theta_min)
            / (self.theta_max - self.theta_min)
            * w_theta
        ).long()

        # Points exactly on r_max / theta_max belong to the final cell.
        r_idx = r_idx.clamp(0, h_r - 1)
        theta_idx = theta_idx.clamp(0, w_theta - 1)

        return grid_log_lambda[
            batch_idx,
            0,
            r_idx,
            theta_idx,
        ]

    def _query_points(
        self,
        polar_latent: torch.Tensor,
        points_rtheta: torch.Tensor,
        batch_idx: torch.Tensor,
    ) -> torch.Tensor:
        device = polar_latent.device
        dtype = polar_latent.dtype

        if points_rtheta.numel() == 0:
            return torch.zeros((0, 2), device=device, dtype=dtype)

        b = polar_latent.shape[0]
        outputs = []

        for batch_id in range(b):
            mask = batch_idx == batch_id
            if not torch.any(mask):
                continue

            pts = points_rtheta[mask]
            coords_norm = self._normalize_points(pts)

            # grid_sample expects coordinate order (x, y) = (theta, r).
            sample_grid = torch.stack(
                [coords_norm[:, 1], coords_norm[:, 0]],
                dim=-1,
            )
            sample_grid = sample_grid.view(1, -1, 1, 2)

            sampled = F.grid_sample(
                polar_latent[batch_id:batch_id + 1],
                sample_grid,
                mode="bilinear",
                padding_mode="border",
                align_corners=False,
            )
            sampled = sampled.squeeze(0).squeeze(-1).transpose(0, 1).contiguous()

            liif_in = torch.cat([sampled, coords_norm], dim=-1)
            outputs.append(self.liif_mlp(liif_in))

        return torch.cat(outputs, dim=0)

    def _collect_gt_points(
        self,
        batch_data_samples,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        points_all = []
        batch_idx_all = []
        num_points = []

        for batch_id, sample in enumerate(batch_data_samples):
            pts = sample.gt_radar_points.polar.to(device=device, dtype=dtype)

            valid = (
                (pts[:, 0] >= self.r_min)
                & (pts[:, 0] <= self.r_max)
                & (pts[:, 1] >= self.theta_min)
                & (pts[:, 1] <= self.theta_max)
            )
            pts = pts[valid]

            num_points.append(pts.shape[0])

            if pts.shape[0] > 0:
                points_all.append(pts)
                batch_idx_all.append(
                    torch.full(
                        (pts.shape[0],),
                        batch_id,
                        device=device,
                        dtype=torch.long,
                    )
                )

        if points_all:
            points_all_t = torch.cat(points_all, dim=0)
            batch_idx_t = torch.cat(batch_idx_all, dim=0)
        else:
            points_all_t = torch.zeros((0, 3), device=device, dtype=dtype)
            batch_idx_t = torch.zeros((0,), device=device, dtype=torch.long)

        num_points_t = torch.tensor(num_points, device=device, dtype=dtype)

        return points_all_t, batch_idx_t, num_points_t

    def forward(self, inputs) -> dict[str, torch.Tensor]:
        polar_latent = self._build_polar_latent(inputs)
        grid_out = self._query_grid(polar_latent)

        return dict(
            polar_latent=polar_latent,
            grid_log_lambda=grid_out[:, 0:1],
            grid_velocity=grid_out[:, 1:2],
        )

    def loss(self, inputs, batch_data_samples, train_cfg=None) -> dict:
        preds = self.forward(inputs)

        grid_log_lambda = preds["grid_log_lambda"]

        device = grid_log_lambda.device
        dtype = grid_log_lambda.dtype

        points_all, batch_idx, num_points = self._collect_gt_points(
            batch_data_samples=batch_data_samples,
            device=device,
            dtype=dtype,
        )

        # PPP observation term uses the same 64x64 field as the integral.
        point_log_lambda = self._gather_grid_log_lambda(
            grid_log_lambda=grid_log_lambda,
            points_rtheta=points_all[:, :2],
            batch_idx=batch_idx,
        )

        loss_ppp = self.loss_ppp(
            grid_log_lambda=grid_log_lambda,
            point_log_lambda=point_log_lambda,
            batch_idx=batch_idx,
            num_points=num_points,
        )

        zero = grid_log_lambda.sum() * 0.0

        if self.velocity_weight > 0.0 and points_all.shape[0] > 0:
            point_out = self._query_points(
                polar_latent=preds["polar_latent"],
                points_rtheta=points_all[:, :2],
                batch_idx=batch_idx,
            )

            pred_vel = point_out[:, 1]
            target_vel = points_all[:, 2]

            if self.velocity_mode == "absolute":
                pred_vel = torch.abs(pred_vel)
                target_vel = torch.abs(target_vel)

            loss_vel_point = F.l1_loss(
                pred_vel,
                target_vel,
                reduction="none",
            )

            loss_vel_per_sample = torch.zeros(
                grid_log_lambda.shape[0],
                device=device,
                dtype=dtype,
            )
            loss_vel_per_sample.scatter_add_(
                dim=0,
                index=batch_idx,
                src=loss_vel_point,
            )

            loss_velocity = (
                loss_vel_per_sample / num_points.clamp_min(1.0)
            ).mean()
        else:
            loss_velocity = zero

        return dict(
            loss_ppp=self.ppp_weight * loss_ppp,
            loss_velocity=self.velocity_weight * loss_velocity,
        )

    def predict(self, inputs, batch_img_metas, test_cfg=None) -> torch.Tensor:
        preds = self.forward(inputs)
        return preds["grid_log_lambda"]