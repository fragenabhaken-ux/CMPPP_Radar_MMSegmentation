from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmseg.registry import MODELS
from .decode_head import BaseDecodeHead


class FFN(nn.Module):
    def __init__(
        self,
        embed_dims: int,
        hidden_dims: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dims, hidden_dims),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims, embed_dims),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PolarSelfCrossAttentionBlock(nn.Module):
    def __init__(
        self,
        embed_dims: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        attn_dropout: float = 0.0,
        ffn_dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.norm1 = nn.LayerNorm(embed_dims)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=embed_dims,
            num_heads=num_heads,
            dropout=attn_dropout,
            batch_first=True,
        )

        self.norm2 = nn.LayerNorm(embed_dims)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dims,
            num_heads=num_heads,
            dropout=attn_dropout,
            batch_first=True,
        )

        self.norm3 = nn.LayerNorm(embed_dims)
        self.ffn = FFN(
            embed_dims=embed_dims,
            hidden_dims=int(embed_dims * mlp_ratio),
            dropout=ffn_dropout,
        )

    def forward(
        self,
        polar_tokens: torch.Tensor,
        image_tokens: torch.Tensor,
    ) -> torch.Tensor:
        q = self.norm1(polar_tokens)
        polar_tokens = polar_tokens + self.self_attn(
            q,
            q,
            q,
            need_weights=False,
        )[0]

        q = self.norm2(polar_tokens)
        polar_tokens = polar_tokens + self.cross_attn(
            q,
            image_tokens,
            image_tokens,
            need_weights=False,
        )[0]

        polar_tokens = polar_tokens + self.ffn(self.norm3(polar_tokens))
        return polar_tokens


@MODELS.register_module()
class PPPfinalsPolarLIIFPPPHead(BaseDecodeHead):
    """Polar PPP head with transformer decoder and continuous LIIF queries.

    The PPP integral uses deterministic midpoint quadrature at the fixed
    centres of ``quadrature_grid_shape``. Ground-truth radar points are
    queried at their exact continuous polar coordinates.

    LIIF output channels:
        channel 0: log intensity
        channel 1: signed wrapped radial velocity
        channel 2: RCS mark

    The RCS target is read from ``gt_radar_points.meta[:, 2]``.
    """

    def __init__(
        self,
        latent_grid_shape: tuple[int, int] = (64, 64),
        quadrature_grid_shape: tuple[int, int] = (256, 256),
        r_min: float = 0.0,
        r_max: float = 104.0,
        theta_min: float = -1.5707963267948966,
        theta_max: float = 1.5707963267948966,
        decoder_embed_dims: int = 384,
        decoder_num_layers: int = 4,
        decoder_num_heads: int = 6,
        decoder_mlp_ratio: float = 4.0,
        decoder_attn_dropout: float = 0.0,
        decoder_ffn_dropout: float = 0.0,
        num_latent_convs: int = 2,
        liif_hidden_channels: int = 128,
        liif_num_layers: int = 3,
        use_feature_unfold: bool = False,
        query_chunk_size: int = 32768,
        loss_ppp: dict | None = None,
        ppp_weight: float = 1.0,
        velocity_weight: float = 0.0,
        rcs_weight: float = 1.0,
        velocity_mode: str = "signed",
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        self.latent_grid_shape = tuple(int(v) for v in latent_grid_shape)
        self.quadrature_grid_shape = tuple(
            int(v) for v in quadrature_grid_shape
        )

        self.r_min = float(r_min)
        self.r_max = float(r_max)
        self.theta_min = float(theta_min)
        self.theta_max = float(theta_max)

        self.decoder_embed_dims = int(decoder_embed_dims)
        self.use_feature_unfold = bool(use_feature_unfold)
        self.query_chunk_size = int(query_chunk_size)

        self.ppp_weight = float(ppp_weight)
        self.velocity_weight = float(velocity_weight)
        self.rcs_weight = float(rcs_weight)
        self.velocity_mode = str(velocity_mode).lower()

        if self.velocity_mode not in {"signed", "absolute"}:
            raise ValueError(
                f"Unsupported velocity_mode: {self.velocity_mode!r}"
            )

        self.conv_seg = nn.Identity()

        self.image_proj = nn.Conv2d(
            self.in_channels,
            self.decoder_embed_dims,
            kernel_size=1,
        )

        self.image_token_norm = nn.LayerNorm(self.decoder_embed_dims)

        self.image_coord_embed = nn.Sequential(
            nn.Linear(2, self.decoder_embed_dims),
            nn.GELU(),
            nn.Linear(self.decoder_embed_dims, self.decoder_embed_dims),
        )

        self.polar_query_token = nn.Parameter(
            torch.zeros(1, 1, self.decoder_embed_dims)
        )

        self.polar_coord_embed = nn.Sequential(
            nn.Linear(2, self.decoder_embed_dims),
            nn.GELU(),
            nn.Linear(self.decoder_embed_dims, self.decoder_embed_dims),
        )

        self.decoder_blocks = nn.ModuleList(
            [
                PolarSelfCrossAttentionBlock(
                    embed_dims=self.decoder_embed_dims,
                    num_heads=decoder_num_heads,
                    mlp_ratio=decoder_mlp_ratio,
                    attn_dropout=decoder_attn_dropout,
                    ffn_dropout=decoder_ffn_dropout,
                )
                for _ in range(int(decoder_num_layers))
            ]
        )

        latent_layers: list[nn.Module] = [
            nn.Conv2d(
                self.decoder_embed_dims,
                self.channels,
                kernel_size=1,
            ),
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

        code_channels = (
            self.channels * 9
            if self.use_feature_unfold
            else self.channels
        )
        liif_in_channels = code_channels + 2

        liif_layers: list[nn.Module] = []
        last_channels = liif_in_channels

        for _ in range(max(int(liif_num_layers) - 1, 0)):
            liif_layers.extend(
                [
                    nn.Linear(
                        last_channels,
                        liif_hidden_channels,
                    ),
                    nn.GELU(),
                ]
            )

            last_channels = liif_hidden_channels

        liif_layers.append(nn.Linear(last_channels, 3))

        self.liif_mlp = nn.Sequential(*liif_layers)

        if loss_ppp is None:
            loss_ppp = dict(
                type="PolarPointPPPLoss",
                grid_shape=self.quadrature_grid_shape,
                r_min=self.r_min,
                r_max=self.r_max,
                theta_min=self.theta_min,
                theta_max=self.theta_max,
                normalize_by_gt_count=True,
            )

        self.loss_ppp = MODELS.build(loss_ppp)

        latent_rtheta = self._make_regular_grid(self.latent_grid_shape)
        quadrature_rtheta = self._make_regular_grid(
            self.quadrature_grid_shape
        )

        self.register_buffer(
            "latent_rtheta",
            latent_rtheta,
            persistent=False,
        )
        self.register_buffer(
            "quadrature_rtheta",
            quadrature_rtheta,
            persistent=False,
        )
        self.register_buffer(
            "latent_rtheta_norm",
            self._normalize_polar_coords(latent_rtheta),
            persistent=False,
        )

    def _make_regular_grid(
        self,
        grid_shape: tuple[int, int],
    ) -> torch.Tensor:
        h_r, w_theta = grid_shape

        dr = (self.r_max - self.r_min) / h_r
        dtheta = (
            self.theta_max - self.theta_min
        ) / w_theta

        r = self.r_min + (
            torch.arange(h_r, dtype=torch.float32) + 0.5
        ) * dr

        theta = self.theta_min + (
            torch.arange(w_theta, dtype=torch.float32) + 0.5
        ) * dtheta

        rr, tt = torch.meshgrid(r, theta, indexing="ij")
        return torch.stack([rr, tt], dim=-1).reshape(-1, 2)

    def _normalize_polar_coords(
        self,
        points_rtheta: torch.Tensor,
    ) -> torch.Tensor:
        r = points_rtheta[:, 0]
        theta = points_rtheta[:, 1]

        r_norm = 2.0 * (r - self.r_min) / (self.r_max - self.r_min) - 1.0
        theta_norm = (
            2.0 * (theta - self.theta_min)
            / (self.theta_max - self.theta_min)
            - 1.0
        )

        return torch.stack([r_norm, theta_norm], dim=-1)

    def _make_image_coord_tokens(
        self,
        h: int,
        w: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        yy = torch.linspace(-1.0, 1.0, h, device=device, dtype=dtype)
        xx = torch.linspace(-1.0, 1.0, w, device=device, dtype=dtype)

        yy, xx = torch.meshgrid(yy, xx, indexing="ij")
        return torch.stack([yy, xx], dim=-1).reshape(-1, 2)

    def _build_polar_latent(self, inputs) -> torch.Tensor:
        x = self._transform_inputs(inputs)
        x = self.image_proj(x)

        b, _, h, w = x.shape

        image_tokens = x.flatten(2).transpose(1, 2).contiguous()

        image_coords = self._make_image_coord_tokens(
            h=h,
            w=w,
            device=x.device,
            dtype=x.dtype,
        )

        image_tokens = image_tokens + self.image_coord_embed(
            image_coords
        ).unsqueeze(0)

        image_tokens = self.image_token_norm(image_tokens)

        num_polar_tokens = (
            self.latent_grid_shape[0] * self.latent_grid_shape[1]
        )

        polar_tokens = self.polar_query_token.expand(
            b,
            num_polar_tokens,
            -1,
        )

        polar_tokens = polar_tokens + self.polar_coord_embed(
            self.latent_rtheta_norm.to(
                device=x.device,
                dtype=x.dtype,
            )
        ).unsqueeze(0)

        for block in self.decoder_blocks:
            polar_tokens = block(
                polar_tokens=polar_tokens,
                image_tokens=image_tokens,
            )

        polar_latent = polar_tokens.transpose(1, 2).reshape(
            b,
            self.decoder_embed_dims,
            self.latent_grid_shape[0],
            self.latent_grid_shape[1],
        ).contiguous()

        return self.latent_proj(polar_latent)

    def _prepare_latent_codes(
        self,
        polar_latent: torch.Tensor,
    ) -> torch.Tensor:
        if not self.use_feature_unfold:
            return polar_latent

        b, c, h_r, w_theta = polar_latent.shape

        codes = F.unfold(
            polar_latent,
            kernel_size=3,
            padding=1,
        )

        return codes.view(b, c * 9, h_r, w_theta)

    def _query_liif(
        self,
        latent_codes: torch.Tensor,
        points_rtheta: torch.Tensor,
        batch_idx: torch.Tensor,
    ) -> torch.Tensor:
        if points_rtheta.numel() == 0:
            return latent_codes.new_zeros((0, 3))

        latent_codes_hw = latent_codes.permute(
            0,
            2,
            3,
            1,
        ).contiguous()

        outputs = []

        for start in range(
            0,
            points_rtheta.shape[0],
            self.query_chunk_size,
        ):
            end = min(
                start + self.query_chunk_size,
                points_rtheta.shape[0],
            )

            outputs.append(
                self._query_liif_chunk(
                    latent_codes_hw=latent_codes_hw,
                    points_rtheta=points_rtheta[start:end],
                    batch_idx=batch_idx[start:end],
                )
            )

        return torch.cat(outputs, dim=0)

    def _query_liif_chunk(
        self,
        latent_codes_hw: torch.Tensor,
        points_rtheta: torch.Tensor,
        batch_idx: torch.Tensor,
    ) -> torch.Tensor:
        _, h_r, w_theta, _ = latent_codes_hw.shape

        device = latent_codes_hw.device
        dtype = latent_codes_hw.dtype

        points_rtheta = points_rtheta.to(
            device=device,
            dtype=dtype,
        )
        batch_idx = batch_idx.to(
            device=device,
            dtype=torch.long,
        )

        r = points_rtheta[:, 0]
        theta = points_rtheta[:, 1]

        dr = (self.r_max - self.r_min) / h_r
        dtheta = (self.theta_max - self.theta_min) / w_theta

        u = (r - self.r_min) / dr - 0.5
        v = (theta - self.theta_min) / dtheta - 0.5

        u = u.clamp(0.0, float(h_r - 1))
        v = v.clamp(0.0, float(w_theta - 1))

        r0 = torch.floor(u).long()
        t0 = torch.floor(v).long()

        r1 = (r0 + 1).clamp(max=h_r - 1)
        t1 = (t0 + 1).clamp(max=w_theta - 1)

        du = u - r0.to(dtype)
        dv = v - t0.to(dtype)

        def decode(
            r_idx: torch.Tensor,
            t_idx: torch.Tensor,
        ) -> torch.Tensor:
            local_code = latent_codes_hw[
                batch_idx,
                r_idx,
                t_idx,
            ]

            center_r = self.r_min + (
                r_idx.to(dtype) + 0.5
            ) * dr

            center_theta = self.theta_min + (
                t_idx.to(dtype) + 0.5
            ) * dtheta

            relative_coord = torch.stack(
                [
                    (r - center_r) / dr,
                    (theta - center_theta) / dtheta,
                ],
                dim=-1,
            )

            return self.liif_mlp(
                torch.cat(
                    [local_code, relative_coord],
                    dim=-1,
                )
            )

        pred_00 = decode(r0, t0)
        pred_01 = decode(r0, t1)
        pred_10 = decode(r1, t0)
        pred_11 = decode(r1, t1)

        w_00 = ((1.0 - du) * (1.0 - dv)).unsqueeze(-1)
        w_01 = ((1.0 - du) * dv).unsqueeze(-1)
        w_10 = (du * (1.0 - dv)).unsqueeze(-1)
        w_11 = (du * dv).unsqueeze(-1)

        return (
            w_00 * pred_00
            + w_01 * pred_01
            + w_10 * pred_10
            + w_11 * pred_11
        )

    def _query_quadrature_grid(
        self,
        latent_codes: torch.Tensor,
    ) -> torch.Tensor:
        b = latent_codes.shape[0]
        h_q, w_q = self.quadrature_grid_shape
        q = h_q * w_q

        points = self.quadrature_rtheta.to(
            device=latent_codes.device,
            dtype=latent_codes.dtype,
        )

        # Fixed polar cell centres. No random quadrature jitter.
        points = points.unsqueeze(0).expand(b, -1, -1).reshape(-1, 2)

        batch_idx = torch.arange(
            b,
            device=latent_codes.device,
            dtype=torch.long,
        ).repeat_interleave(q)

        output = self._query_liif(
            latent_codes=latent_codes,
            points_rtheta=points,
            batch_idx=batch_idx,
        )

        return output.view(
            b,
            h_q,
            w_q,
            3,
        ).permute(0, 3, 1, 2).contiguous()

    def _collect_gt_points(
        self,
        batch_data_samples,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Collect in-domain polar points and their aligned RCS targets."""
        points_all = []
        rcs_all = []
        batch_idx_all = []
        num_points = []

        for batch_id, sample in enumerate(batch_data_samples):
            pts = sample.gt_radar_points.polar.to(
                device=device,
                dtype=dtype,
            )
            meta = sample.gt_radar_points.meta.to(
                device=device,
                dtype=dtype,
            )

            if pts.ndim != 2 or pts.shape[1] < 3:
                raise ValueError(
                    f"Expected polar points [N, >=3], got {tuple(pts.shape)}."
                )
            if meta.ndim != 2 or meta.shape[0] != pts.shape[0]:
                raise ValueError(
                    "gt_radar_points.meta must have the same point count "
                    "as gt_radar_points.polar."
                )
            if meta.shape[1] <= 2:
                raise ValueError(
                    "Expected RCS in gt_radar_points.meta[:, 2], "
                    f"but meta has shape {tuple(meta.shape)}."
                )

            valid = (
                (pts[:, 0] >= self.r_min)
                & (pts[:, 0] <= self.r_max)
                & (pts[:, 1] >= self.theta_min)
                & (pts[:, 1] <= self.theta_max)
            )

            pts = pts[valid]
            rcs = meta[valid, 2]
            num_points.append(pts.shape[0])

            if pts.shape[0] > 0:
                points_all.append(pts)
                rcs_all.append(rcs)
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
            rcs_all_t = torch.cat(rcs_all, dim=0)
            batch_idx_t = torch.cat(batch_idx_all, dim=0)
        else:
            points_all_t = torch.zeros(
                (0, 3),
                device=device,
                dtype=dtype,
            )
            rcs_all_t = torch.zeros(
                (0,),
                device=device,
                dtype=dtype,
            )
            batch_idx_t = torch.zeros(
                (0,),
                device=device,
                dtype=torch.long,
            )

        num_points_t = torch.tensor(
            num_points,
            device=device,
            dtype=dtype,
        )

        return points_all_t, rcs_all_t, batch_idx_t, num_points_t

    @staticmethod
    def _doppler_periods(
        batch_data_samples,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Return one positive Doppler period per sample as a tensor [B]."""
        periods = []

        for sample_idx, sample in enumerate(batch_data_samples):
            metainfo = sample.metainfo
            if "doppler_period" not in metainfo:
                raise KeyError(
                    "Missing 'doppler_period' in data sample "
                    f"{sample_idx}. Ensure the polar transform packs "
                    "dopplerRange."
                )

            period = float(metainfo["doppler_period"])
            period_check = torch.tensor(period, dtype=torch.float64)
            if not torch.isfinite(period_check).item() or period <= 0:
                raise ValueError(
                    f"Invalid Doppler period for sample {sample_idx}: "
                    f"{period}"
                )

            periods.append(period)

        return torch.tensor(
            periods,
            device=device,
            dtype=dtype,
        )

    @staticmethod
    def _circular_velocity_error(
        pred_vel: torch.Tensor,
        target_vel: torch.Tensor,
        doppler_period: torch.Tensor,
    ) -> torch.Tensor:
        """Return the signed shortest velocity difference modulo the period."""
        return torch.remainder(
            pred_vel - target_vel + 0.5 * doppler_period,
            doppler_period,
        ) - 0.5 * doppler_period

    @staticmethod
    def _mean_point_loss_per_sample(
        point_loss: torch.Tensor,
        batch_idx: torch.Tensor,
        num_points: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        """Average pointwise losses per sample, then across the batch."""
        loss_per_sample = point_loss.new_zeros(batch_size)
        loss_per_sample.scatter_add_(
            dim=0,
            index=batch_idx,
            src=point_loss,
        )
        return (
            loss_per_sample / num_points.clamp_min(1.0)
        ).mean()

    def forward(
        self,
        inputs,
    ) -> dict[str, torch.Tensor]:
        polar_latent = self._build_polar_latent(inputs)
        latent_codes = self._prepare_latent_codes(polar_latent)

        quadrature_out = self._query_quadrature_grid(latent_codes)
        grid_raw_rcs = quadrature_out[:, 2:3]

        return dict(
            polar_latent=polar_latent,
            latent_codes=latent_codes,
            grid_log_lambda=quadrature_out[:, 0:1],
            grid_velocity=quadrature_out[:, 1:2],
            grid_raw_rcs=grid_raw_rcs,
            grid_rcs=grid_raw_rcs**3,
        )

    def loss(
        self,
        inputs,
        batch_data_samples,
        train_cfg=None,
    ) -> dict:
        preds = self.forward(inputs)
        grid_log_lambda = preds["grid_log_lambda"]

        device = grid_log_lambda.device
        dtype = grid_log_lambda.dtype
        batch_size = grid_log_lambda.shape[0]

        points_all, rcs_all, batch_idx, num_points = (
            self._collect_gt_points(
                batch_data_samples=batch_data_samples,
                device=device,
                dtype=dtype,
            )
        )

        point_out = self._query_liif(
            latent_codes=preds["latent_codes"],
            points_rtheta=points_all[:, :2],
            batch_idx=batch_idx,
        )

        point_log_lambda = point_out[:, 0]

        loss_ppp = self.loss_ppp(
            grid_log_lambda=grid_log_lambda,
            point_log_lambda=point_log_lambda,
            batch_idx=batch_idx,
            num_points=num_points,
        )

        zero = grid_log_lambda.sum() * 0.0

        if self.velocity_weight > 0.0 and points_all.shape[0] > 0:
            pred_vel = point_out[:, 1]
            target_vel = points_all[:, 2]

            if self.velocity_mode == "signed":
                doppler_periods = self._doppler_periods(
                    batch_data_samples,
                    device=device,
                    dtype=dtype,
                )
                point_period = doppler_periods[batch_idx]
                velocity_error = self._circular_velocity_error(
                    pred_vel,
                    target_vel,
                    point_period,
                )
                loss_vel_point = torch.abs(velocity_error)
            elif self.velocity_mode == "absolute":
                loss_vel_point = F.l1_loss(
                    torch.abs(pred_vel),
                    torch.abs(target_vel),
                    reduction="none",
                )
            else:
                raise RuntimeError(
                    f"Unsupported velocity_mode={self.velocity_mode!r}"
                )

            loss_velocity = self._mean_point_loss_per_sample(
                point_loss=loss_vel_point,
                batch_idx=batch_idx,
                num_points=num_points,
                batch_size=batch_size,
            )
        else:
            loss_velocity = zero

        if self.rcs_weight > 0.0 and points_all.shape[0] > 0:
            pred_rcs = torch.exp(point_out[:, 2])
            loss_rcs_point = F.l1_loss(
                pred_rcs,
                rcs_all,
                reduction="none",
            )
            loss_rcs = self._mean_point_loss_per_sample(
                point_loss=loss_rcs_point,
                batch_idx=batch_idx,
                num_points=num_points,
                batch_size=batch_size,
            )
        else:
            loss_rcs = zero

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
        return self.forward(inputs)["grid_log_lambda"]