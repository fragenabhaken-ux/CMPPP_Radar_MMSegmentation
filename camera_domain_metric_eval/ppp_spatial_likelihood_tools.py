"""Optimally Gaussianized image-space PPP log-likelihood metric."""

from __future__ import annotations

import math
from typing import Any, Callable

import numpy as np
from scipy.ndimage import gaussian_filter

EPS_LOG = 1e-12


def prepare_mass_grid(grid: np.ndarray, negative_tolerance: float = 1e-8) -> np.ndarray:
    """Validate and return a nonnegative float64 mass grid."""
    x = np.asarray(grid, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"Prediction mass grid must be 2D, got {x.shape}")
    if not np.all(np.isfinite(x)):
        raise ValueError("Prediction mass grid contains NaN or Inf")
    min_value = float(x.min(initial=0.0))
    if min_value < -float(negative_tolerance):
        raise ValueError(
            "Prediction mass grid contains materially negative values: "
            f"minimum={min_value:.6g}"
        )
    return np.clip(x, 0.0, None)


def gaussianize_mass_grid(grid: np.ndarray, sigma_px: float) -> np.ndarray:
    """Apply isotropic Gaussian smoothing in pixel coordinates, preserving mass."""
    x = prepare_mass_grid(grid)
    sigma_px = float(sigma_px)
    if sigma_px < 0.0:
        raise ValueError("sigma_px must be nonnegative")
    if sigma_px == 0.0:
        return x.copy()

    total_mass = float(x.sum())
    if total_mass <= 0.0:
        return np.zeros_like(x)

    smoothed = gaussian_filter(
        x / total_mass,
        sigma=sigma_px,
        mode="constant",
        cval=0.0,
    )
    smoothed_sum = float(smoothed.sum())
    if smoothed_sum <= 0.0:
        return np.zeros_like(x)
    smoothed *= total_mass / smoothed_sum
    return smoothed


def image_ppp_loglik_score(
    pred_mass: np.ndarray,
    gt_support: np.ndarray,
    *,
    eps: float = EPS_LOG,
    normalize_by_gt: bool = True,
) -> float:
    """Compute the discrete image-space PPP log-likelihood score.

    ``pred_mass`` is expected-count mass per image cell. The corresponding
    image-domain intensity density is ``pred_mass * H * W``. Therefore

        score = sum_GT log(pred_mass) + n_GT log(HW) - sum(pred_mass).

    Higher values are better.
    """
    h = prepare_mass_grid(pred_mass)
    gt = np.asarray(gt_support)
    if gt.shape != h.shape:
        raise ValueError(
            f"gt_support shape {gt.shape} does not match prediction {h.shape}"
        )
    if not np.all(np.isfinite(gt)):
        raise ValueError("gt_support contains NaN or Inf")

    gt_mask = gt > 0.0
    n_gt = int(gt_mask.sum())
    height, width = h.shape

    score = -float(h.sum())
    if n_gt > 0:
        score += float(np.log(np.clip(h[gt_mask], eps, None)).sum())
        score += float(n_gt) * math.log(float(height * width))
        if normalize_by_gt:
            score /= float(n_gt)
    return float(score)


def optimize_gaussian_sigma(
    pred_mass: np.ndarray,
    gt_support: np.ndarray,
    *,
    sigma_min_px: float = 0.0,
    sigma_max_px: float = 120.0,
    max_iter: int = 30,
    tol_sigma_px: float = 0.5,
    eps: float = EPS_LOG,
    normalize_by_gt: bool = True,
    check_zero: bool = True,
) -> dict[str, Any]:
    """Maximize the spatial score with the established log-space golden search.

    This follows the earlier camera-domain metric implementation so that final
    offline scores remain comparable with the metric used during training-time
    validation. The unsmoothed case is evaluated explicitly.
    """
    pred_mass = prepare_mass_grid(pred_mass)
    gt_support = np.asarray(gt_support)
    if pred_mass.shape != gt_support.shape:
        raise ValueError("pred_mass and gt_support must have the same shape")

    sigma_min_px = float(sigma_min_px)
    sigma_max_px = float(sigma_max_px)
    if sigma_min_px < 0.0:
        raise ValueError("sigma_min_px must be >= 0")
    if sigma_max_px <= sigma_min_px:
        raise ValueError("sigma_max_px must be greater than sigma_min_px")

    if float(pred_mass.sum()) <= 0.0:
        return {
            "best_sigma_px": sigma_min_px,
            "best_score": float("-inf"),
            "score_sigma0": float("-inf"),
            "num_evaluations": 0,
            "status": "empty_prediction",
        }

    gt_count = int((gt_support > 0.0).sum())
    if gt_count == 0:
        score_zero = image_ppp_loglik_score(
            pred_mass,
            gt_support,
            eps=eps,
            normalize_by_gt=normalize_by_gt,
        )
        return {
            "best_sigma_px": 0.0,
            "best_score": score_zero,
            "score_sigma0": score_zero,
            "num_evaluations": 1,
            "status": "empty_ground_truth",
        }

    cache: dict[float, float] = {}

    def evaluate(sigma: float) -> float:
        sigma = float(max(0.0, sigma))
        key = round(sigma, 12)
        if key not in cache:
            smoothed = gaussianize_mass_grid(pred_mass, sigma)
            cache[key] = image_ppp_loglik_score(
                smoothed,
                gt_support,
                eps=eps,
                normalize_by_gt=normalize_by_gt,
            )
        return cache[key]

    best_sigma: float | None = None
    best_score = float("-inf")
    score_zero = np.nan

    if check_zero and sigma_min_px == 0.0:
        score_zero = evaluate(0.0)
        best_sigma = 0.0
        best_score = float(score_zero)

    positive_left = max(sigma_min_px, np.finfo(np.float64).eps)
    positive_right = sigma_max_px
    if positive_right <= positive_left:
        return {
            "best_sigma_px": float(best_sigma if best_sigma is not None else sigma_min_px),
            "best_score": float(best_score),
            "score_sigma0": float(score_zero),
            "num_evaluations": int(len(cache)),
            "status": "ok",
        }

    phi = (1.0 + math.sqrt(5.0)) / 2.0
    inv_phi = 1.0 / phi

    # The established implementation searches in log sigma and uses 0.1 px as
    # the smallest positive scale. Sigma=0 remains a separate candidate.
    log_left = math.log(max(positive_left, 0.1))
    log_right = math.log(max(positive_right, 0.1))

    c = log_right - (log_right - log_left) * inv_phi
    d = log_left + (log_right - log_left) * inv_phi
    c_sigma = math.exp(c)
    d_sigma = math.exp(d)
    score_c = evaluate(c_sigma)
    score_d = evaluate(d_sigma)

    if score_c > best_score:
        best_score, best_sigma = score_c, c_sigma
    if score_d > best_score:
        best_score, best_sigma = score_d, d_sigma

    for _ in range(int(max_iter)):
        if math.exp(log_right) - math.exp(log_left) < float(tol_sigma_px):
            break
        if score_c < score_d:
            log_left = c
            c = d
            score_c = score_d
            d = log_left + (log_right - log_left) * inv_phi
            d_sigma = math.exp(d)
            score_d = evaluate(d_sigma)
            if score_d > best_score:
                best_score, best_sigma = score_d, d_sigma
        else:
            log_right = d
            d = c
            score_d = score_c
            c = log_right - (log_right - log_left) * inv_phi
            c_sigma = math.exp(c)
            score_c = evaluate(c_sigma)
            if score_c > best_score:
                best_score, best_sigma = score_c, c_sigma

    sigma_mid = math.exp(0.5 * (log_left + log_right))
    score_mid = evaluate(sigma_mid)
    if score_mid > best_score:
        best_score, best_sigma = score_mid, sigma_mid

    # Prefer sigma=0 when it is numerically tied with a positive candidate.
    if check_zero and np.isfinite(score_zero) and float(score_zero) >= best_score - 1e-12:
        best_sigma = 0.0
        best_score = float(score_zero)

    return {
        "best_sigma_px": float(best_sigma),
        "best_score": float(best_score),
        "score_sigma0": float(score_zero),
        "num_evaluations": int(len(cache)),
        "status": "ok",
    }


def compute_likelihood_metrics(
    sample: dict[str, Any],
    config: dict[str, Any] | None = None,
) -> dict[str, float | int | str]:
    """Adapter from one loaded sample to named likelihood metrics."""
    cfg = {
        "sigma_min_px": 0.0,
        "sigma_max_px": 120.0,
        "max_iter": 30,
        "tol_sigma_px": 0.25,
        "eps": EPS_LOG,
        "normalize_by_gt": True,
        "check_zero": True,
    }
    if config is not None:
        cfg.update(config)
    cfg.pop("enabled", None)

    result = optimize_gaussian_sigma(
        pred_mass=sample["pred_lambda"],
        gt_support=sample["gt_support"],
        **cfg,
    )
    return {
        "likelihood_optimal_score": float(result["best_score"]),
        "likelihood_optimal_sigma_px": float(result["best_sigma_px"]),
        "likelihood_score_sigma0": float(result["score_sigma0"]),
        "likelihood_num_evaluations": int(result["num_evaluations"]),
        "likelihood_status": str(result["status"]),
    }
