from __future__ import annotations

import math

import numpy as np
import torch
from mmengine.evaluator import BaseMetric
from mmengine.logging import MMLogger
from mmengine.registry import METRICS

from .ppp_metrics_loss_mmd_tools import compute_optimal_sigma_metric


@METRICS.register_module()
class PolarPPPMetric(BaseMetric):
    """Gaussianized BEV metric for polar PPP predictions.

    The model predicts log intensity with respect to polar measure dr dtheta.

    For each polar quadrature cell:

        mass_ij = exp(log_lambda_ij) * dr * dtheta

    That cell mass is bilinearly splatted to a Cartesian BEV grid. GT radar
    points are rasterized onto the same BEV grid. The old optimal-sigma
    Gaussianized metric is then evaluated on these two BEV representations.

    sigma is configured and reported in metres.
    """

    def __init__(
        self,
        grid_shape: tuple[int, int],
        r_min: float,
        r_max: float,
        theta_min: float,
        theta_max: float,
        bev_x_min: float = 0.0,
        bev_x_max: float = 104.0,
        bev_y_min: float = -85.0,
        bev_y_max: float = 85.0,
        bev_resolution: float = 0.5,
        score_type: str = 'loglik',
        sigma_min_m: float = 0.0,
        sigma_max_m: float = 20.0,
        max_iter: int = 30,
        tol_sigma_m: float = 0.25,
        normalize_by_gt: bool = True,
        eps: float = 1e-12,
        check_zero: bool = True,
        pred_channel: int = 0,
        verbose_logging: bool = False,
        prefix: str = 'val',
        **kwargs,
    ) -> None:
        super().__init__(prefix=prefix, **kwargs)

        self.grid_shape = tuple(int(v) for v in grid_shape)

        if min(*self.grid_shape) <= 0:
            raise ValueError('grid_shape must contain positive dimensions.')

        self.r_min = float(r_min)
        self.r_max = float(r_max)
        self.theta_min = float(theta_min)
        self.theta_max = float(theta_max)

        if self.r_max <= self.r_min:
            raise ValueError('r_max must be larger than r_min.')

        if self.theta_max <= self.theta_min:
            raise ValueError('theta_max must be larger than theta_min.')

        self.bev_x_min = float(bev_x_min)
        self.bev_x_max = float(bev_x_max)
        self.bev_y_min = float(bev_y_min)
        self.bev_y_max = float(bev_y_max)
        self.bev_resolution = float(bev_resolution)

        if self.bev_x_max <= self.bev_x_min:
            raise ValueError('bev_x_max must be larger than bev_x_min.')

        if self.bev_y_max <= self.bev_y_min:
            raise ValueError('bev_y_max must be larger than bev_y_min.')

        if self.bev_resolution <= 0.0:
            raise ValueError('bev_resolution must be positive.')

        self.bev_grid_shape = self._make_bev_grid_shape()

        score_type = str(score_type).lower()

        if score_type not in {'loglik', 'mmd'}:
            raise ValueError(
                "score_type must be 'loglik' or 'mmd', "
                f'got {score_type!r}.'
            )

        self.score_type = score_type

        self.sigma_min_m = float(sigma_min_m)
        self.sigma_max_m = float(sigma_max_m)
        self.max_iter = int(max_iter)
        self.tol_sigma_m = float(tol_sigma_m)

        self.normalize_by_gt = bool(normalize_by_gt)
        self.eps = float(eps)
        self.check_zero = bool(check_zero)
        self.pred_channel = int(pred_channel)
        self.verbose_logging = bool(verbose_logging)

        if self.sigma_min_m < 0.0:
            raise ValueError('sigma_min_m must be nonnegative.')

        if self.sigma_max_m <= self.sigma_min_m:
            raise ValueError(
                'sigma_max_m must be larger than sigma_min_m.'
            )

        if self.tol_sigma_m <= 0.0:
            raise ValueError('tol_sigma_m must be positive.')

        h_r, w_theta = self.grid_shape

        self.dr = (self.r_max - self.r_min) / h_r
        self.dtheta = (
            self.theta_max - self.theta_min
        ) / w_theta

        self.polar_cell_measure = self.dr * self.dtheta

        self._polar_x, self._polar_y = (
            self._make_polar_cell_centres_xy()
        )

    def _make_bev_grid_shape(self) -> tuple[int, int]:
        x_cells_float = (
            (self.bev_x_max - self.bev_x_min)
            / self.bev_resolution
        )

        y_cells_float = (
            (self.bev_y_max - self.bev_y_min)
            / self.bev_resolution
        )

        h = int(round(x_cells_float))
        w = int(round(y_cells_float))

        if h <= 0 or w <= 0:
            raise ValueError(
                'BEV grid must contain positive dimensions.'
            )

        if not math.isclose(
            x_cells_float,
            h,
            rel_tol=0.0,
            abs_tol=1e-6,
        ):
            raise ValueError(
                'BEV x-range must be divisible by bev_resolution.'
            )

        if not math.isclose(
            y_cells_float,
            w,
            rel_tol=0.0,
            abs_tol=1e-6,
        ):
            raise ValueError(
                'BEV y-range must be divisible by bev_resolution.'
            )

        return h, w

    def _make_polar_cell_centres_xy(
        self,
    ) -> tuple[np.ndarray, np.ndarray]:
        h_r, w_theta = self.grid_shape

        r = self.r_min + (
            np.arange(h_r, dtype=np.float64) + 0.5
        ) * self.dr

        theta = self.theta_min + (
            np.arange(w_theta, dtype=np.float64) + 0.5
        ) * self.dtheta

        rr, tt = np.meshgrid(r, theta, indexing='ij')

        x = rr * np.cos(tt)
        y = rr * np.sin(tt)

        return x.reshape(-1), y.reshape(-1)

    @staticmethod
    def _get_attr(sample, key: str):
        if isinstance(sample, dict):
            return sample.get(key, None)

        return getattr(sample, key, None)

    @staticmethod
    def _unwrap_data(value):
        if value is None:
            return None

        if isinstance(value, dict):
            return value.get('data', value)

        if hasattr(value, 'data'):
            return value.data

        return value

    def _get_log_lambda_grid(
        self,
        sample,
    ) -> torch.Tensor:
        seg_logits = self._unwrap_data(
            self._get_attr(sample, 'seg_logits')
        )

        if seg_logits is None:
            raise AttributeError(
                'Prediction sample has no seg_logits.'
            )

        if not isinstance(seg_logits, torch.Tensor):
            seg_logits = torch.as_tensor(seg_logits)

        if seg_logits.ndim == 3:
            log_lambda = seg_logits[self.pred_channel]
        elif seg_logits.ndim == 2:
            log_lambda = seg_logits
        else:
            raise ValueError(
                'Expected seg_logits with shape [C, H, W] or [H, W], '
                f'got {tuple(seg_logits.shape)}.'
            )

        if tuple(log_lambda.shape) != self.grid_shape:
            raise ValueError(
                'Prediction grid shape does not match grid_shape: '
                f'got {tuple(log_lambda.shape)}, '
                f'expected {self.grid_shape}.'
            )

        return log_lambda.detach().float()

    def _get_gt_points_polar(
        self,
        sample,
    ) -> torch.Tensor:
        gt_radar_points = self._get_attr(
            sample,
            'gt_radar_points',
        )

        if gt_radar_points is None:
            raise AttributeError(
                'Validation sample has no gt_radar_points.'
            )

        polar = self._unwrap_data(
            self._get_attr(gt_radar_points, 'polar')
        )

        if polar is None:
            raise AttributeError(
                'gt_radar_points has no polar attribute.'
            )

        if not isinstance(polar, torch.Tensor):
            polar = torch.as_tensor(polar)

        polar = polar.detach().to(
            dtype=torch.float32,
            device='cpu',
        )

        if polar.ndim != 2 or polar.shape[1] < 2:
            raise ValueError(
                'Expected polar GT points with shape [N, >=2], '
                f'got {tuple(polar.shape)}.'
            )

        valid = (
            (polar[:, 0] >= self.r_min)
            & (polar[:, 0] <= self.r_max)
            & (polar[:, 1] >= self.theta_min)
            & (polar[:, 1] <= self.theta_max)
        )

        return polar[valid, :2]

    def _splat_polar_mass_to_bev(
        self,
        polar_cell_mass: np.ndarray,
    ) -> np.ndarray:
        """Mass-preserving bilinear splat from polar cells to BEV cells."""
        mass = np.asarray(
            polar_cell_mass,
            dtype=np.float64,
        ).reshape(-1)

        if mass.shape[0] != self._polar_x.shape[0]:
            raise ValueError(
                'polar_cell_mass has incompatible number of entries.'
            )

        if not np.isfinite(mass).all():
            raise FloatingPointError(
                'Predicted polar mass contains NaN or Inf.'
            )

        h_bev, w_bev = self.bev_grid_shape

        u = (
            (self._polar_x - self.bev_x_min)
            / self.bev_resolution
            - 0.5
        )

        v = (
            (self._polar_y - self.bev_y_min)
            / self.bev_resolution
            - 0.5
        )

        u = np.clip(u, 0.0, h_bev - 1.0)
        v = np.clip(v, 0.0, w_bev - 1.0)

        x0 = np.floor(u).astype(np.int64)
        y0 = np.floor(v).astype(np.int64)

        x1 = np.minimum(x0 + 1, h_bev - 1)
        y1 = np.minimum(y0 + 1, w_bev - 1)

        du = u - x0
        dv = v - y0

        bev_mass = np.zeros(
            h_bev * w_bev,
            dtype=np.float64,
        )

        np.add.at(
            bev_mass,
            x0 * w_bev + y0,
            mass * (1.0 - du) * (1.0 - dv),
        )

        np.add.at(
            bev_mass,
            x0 * w_bev + y1,
            mass * (1.0 - du) * dv,
        )

        np.add.at(
            bev_mass,
            x1 * w_bev + y0,
            mass * du * (1.0 - dv),
        )

        np.add.at(
            bev_mass,
            x1 * w_bev + y1,
            mass * du * dv,
        )

        return bev_mass.reshape(h_bev, w_bev)

    def _rasterize_gt_support(
        self,
        points_rtheta: torch.Tensor,
    ) -> np.ndarray:
        """Rasterize GT point support onto the Cartesian BEV grid."""
        h_bev, w_bev = self.bev_grid_shape

        gt_support = np.zeros(
            (h_bev, w_bev),
            dtype=np.float64,
        )

        if points_rtheta.numel() == 0:
            return gt_support

        points = points_rtheta.numpy().astype(
            np.float64,
            copy=False,
        )

        x = points[:, 0] * np.cos(points[:, 1])
        y = points[:, 0] * np.sin(points[:, 1])

        x_idx = np.floor(
            (x - self.bev_x_min) / self.bev_resolution
        ).astype(np.int64)

        y_idx = np.floor(
            (y - self.bev_y_min) / self.bev_resolution
        ).astype(np.int64)

        valid = (
            (x_idx >= 0)
            & (x_idx < h_bev)
            & (y_idx >= 0)
            & (y_idx < w_bev)
        )

        gt_support[x_idx[valid], y_idx[valid]] = 1.0

        return gt_support

    def process(self, data_batch, data_samples) -> None:
        for sample in data_samples:
            log_lambda = self._get_log_lambda_grid(sample)

            log_lambda_np = log_lambda.cpu().numpy().astype(
                np.float64,
                copy=False,
            )

            polar_cell_mass = (
                np.exp(log_lambda_np)
                * self.polar_cell_measure
            )

            bev_mass = self._splat_polar_mass_to_bev(
                polar_cell_mass
            )

            points_rtheta = self._get_gt_points_polar(sample)

            gt_support = self._rasterize_gt_support(
                points_rtheta
            )

            result = compute_optimal_sigma_metric(
                pred_grid=bev_mass,
                gt_support=gt_support,
                score_type=self.score_type,
                sigma_min=(
                    self.sigma_min_m / self.bev_resolution
                ),
                sigma_max=(
                    self.sigma_max_m / self.bev_resolution
                ),
                max_iter=self.max_iter,
                tol_sigma=(
                    self.tol_sigma_m / self.bev_resolution
                ),
                eps=self.eps,
                normalize_by_gt=self.normalize_by_gt,
                check_zero=self.check_zero,
            )

            predicted_count = float(polar_cell_mass.sum())
            gt_count = int(points_rtheta.shape[0])

            self.results.append(
                dict(
                    optimal_score=float(result['best_score']),
                    optimal_sigma=float(
                        result['best_sigma']
                        * self.bev_resolution
                    ),
                    predicted_count=predicted_count,
                    gt_count=float(gt_count),
                    count_error=predicted_count - gt_count,
                )
            )

            if self.verbose_logging:
                logger = MMLogger.get_current_instance()

                logger.info(
                    f"optimal_score={result['best_score']:.6f}, "
                    'optimal_sigma_m='
                    f'{result["best_sigma"] * self.bev_resolution:.3f}'
                )

    def compute_metrics(
        self,
        results: list[dict],
    ) -> dict:
        if not results:
            return {}

        def mean(key: str) -> float:
            return float(
                np.mean([result[key] for result in results])
            )

        count_error = np.asarray(
            [result['count_error'] for result in results],
            dtype=np.float64,
        )

        return dict(
            optimal_score=mean('optimal_score'),
            optimal_sigma=mean('optimal_sigma'),
            expected_count_mae=float(
                np.mean(np.abs(count_error))
            ),
            expected_count_bias=float(
                np.mean(count_error)
            ),
            expected_count_rmse=float(
                np.sqrt(np.mean(count_error**2))
            ),
            mean_pred_count=mean('predicted_count'),
            mean_gt_count=mean('gt_count'),
        )