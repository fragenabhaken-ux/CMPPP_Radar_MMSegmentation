from __future__ import annotations

import math
from typing import Any, Callable

import numpy as np
from scipy.ndimage import gaussian_filter

from .types import BEVGeometry


DEFAULT_CONFIG: dict[str, Any] = {
    "sigma_min_m": 0.0,
    "sigma_max_m": 20.0,
    "positive_sigma_floor_m": 0.01,
    "max_iter": 40,
    "tol_sigma_m": 0.05,
    "eps": 1e-12,
    "normalization": "physical_density",
    "check_zero": True,
}


def gaussianize_mass(
    mass: np.ndarray,
    sigma_m: float,
    geometry: BEVGeometry,
) -> np.ndarray:
    """Gaussian-smooth a Cartesian mass grid with a circular metric kernel.

    Because the raster can have dx != dy, scipy receives anisotropic pixel
    sigmas so that the Gaussian standard deviation is sigma_m in both metric
    axes. The result is rescaled to preserve total expected count.
    """
    x = np.clip(np.asarray(mass, dtype=np.float64), 0.0, None)
    total = float(x.sum())
    sigma_m = float(max(0.0, sigma_m))
    if sigma_m == 0.0 or total <= 0.0:
        return x.copy()

    sigma_pixels = (sigma_m / geometry.dx, sigma_m / geometry.dy)
    blurred = gaussian_filter(
        x / total,
        sigma=sigma_pixels,
        mode="constant",
        cval=0.0,
    )
    blurred_sum = float(blurred.sum())
    if blurred_sum <= 0.0:
        return np.zeros_like(x)
    blurred *= total / blurred_sum
    return blurred


def spatial_log_likelihood_score(
    mass: np.ndarray,
    gt_support: np.ndarray,
    geometry: BEVGeometry,
    *,
    eps: float = 1e-12,
    normalization: str = "physical_density",
) -> float:
    """Normalized PPP log-likelihood score on a Cartesian BEV raster.

    ``gt_support`` is a count map, so a cell containing k GT points contributes
    k copies of the log-density term.
    """
    mass = np.clip(np.asarray(mass, dtype=np.float64), 0.0, None)
    gt = np.clip(np.asarray(gt_support, dtype=np.float64), 0.0, None)
    if mass.shape != gt.shape or mass.shape != geometry.shape:
        raise ValueError(
            f"Shape mismatch: mass={mass.shape}, gt={gt.shape}, geometry={geometry.shape}."
        )

    n_gt = float(gt.sum())
    if n_gt <= 0.0:
        return np.nan

    normalization = str(normalization).lower()
    if normalization == "physical_density":
        intensity = mass / geometry.cell_area
    elif normalization == "normalized_grid_density":
        intensity = mass * float(geometry.height * geometry.width)
    elif normalization == "cell_mass":
        intensity = mass
    else:
        raise ValueError(
            "normalization must be 'physical_density', "
            "'normalized_grid_density', or 'cell_mass'."
        )

    observation = float((gt * np.log(np.clip(intensity, eps, None))).sum())
    integral = float(mass.sum())
    return (observation - integral) / n_gt


def optimize_sigma_golden(
    mass: np.ndarray,
    gt_support: np.ndarray,
    geometry: BEVGeometry,
    config: dict[str, Any] | None = None,
) -> dict[str, float | int | str]:
    cfg = dict(DEFAULT_CONFIG)
    if config:
        cfg.update(config)

    sigma_min = float(cfg["sigma_min_m"])
    sigma_max = float(cfg["sigma_max_m"])
    positive_floor = float(cfg["positive_sigma_floor_m"])
    max_iter = int(cfg["max_iter"])
    tol = float(cfg["tol_sigma_m"])
    eps = float(cfg["eps"])
    normalization = str(cfg["normalization"])
    check_zero = bool(cfg["check_zero"])

    if sigma_min < 0.0:
        raise ValueError("sigma_min_m must be >= 0.")
    if sigma_max <= sigma_min:
        raise ValueError("sigma_max_m must be greater than sigma_min_m.")
    if positive_floor <= 0.0:
        raise ValueError("positive_sigma_floor_m must be > 0.")
    if positive_floor > sigma_max:
        raise ValueError("positive_sigma_floor_m must be <= sigma_max_m.")
    if max_iter <= 0:
        raise ValueError("max_iter must be > 0.")
    if tol <= 0.0:
        raise ValueError("tol_sigma_m must be > 0.")

    gt_count = float(np.asarray(gt_support, dtype=np.float64).sum())
    if gt_count <= 0.0:
        return {
            "likelihood_optimal_score": np.nan,
            "likelihood_optimal_sigma_m": np.nan,
            "likelihood_score_sigma0": np.nan,
            "likelihood_num_evaluations": 0,
            "likelihood_status": "empty_ground_truth",
        }

    cache: dict[float, float] = {}

    def evaluate(sigma: float) -> float:
        sigma = float(max(0.0, sigma))
        key = round(sigma, 12)
        if key not in cache:
            smoothed = gaussianize_mass(mass, sigma, geometry)
            cache[key] = spatial_log_likelihood_score(
                smoothed,
                gt_support,
                geometry,
                eps=eps,
                normalization=normalization,
            )
        return cache[key]

    best_sigma = 0.0
    score_zero = evaluate(0.0)
    best_score = score_zero if check_zero else -np.inf

    left_sigma = max(sigma_min, positive_floor)
    right_sigma = sigma_max

    # Evaluate boundaries explicitly because the optimum can lie there.
    for candidate in (left_sigma, right_sigma):
        score = evaluate(candidate)
        if score > best_score:
            best_score = score
            best_sigma = candidate

    if right_sigma > left_sigma:
        log_left = math.log(left_sigma)
        log_right = math.log(right_sigma)
        inv_phi = 2.0 / (1.0 + math.sqrt(5.0))

        c_log = log_right - inv_phi * (log_right - log_left)
        d_log = log_left + inv_phi * (log_right - log_left)
        c_sigma = math.exp(c_log)
        d_sigma = math.exp(d_log)
        c_score = evaluate(c_sigma)
        d_score = evaluate(d_sigma)

        for _ in range(max_iter):
            if math.exp(log_right) - math.exp(log_left) <= tol:
                break
            if c_score > d_score:
                log_right = d_log
                d_log, d_sigma, d_score = c_log, c_sigma, c_score
                c_log = log_right - inv_phi * (log_right - log_left)
                c_sigma = math.exp(c_log)
                c_score = evaluate(c_sigma)
            else:
                log_left = c_log
                c_log, c_sigma, c_score = d_log, d_sigma, d_score
                d_log = log_left + inv_phi * (log_right - log_left)
                d_sigma = math.exp(d_log)
                d_score = evaluate(d_sigma)

            if c_score > best_score:
                best_sigma, best_score = c_sigma, c_score
            if d_score > best_score:
                best_sigma, best_score = d_sigma, d_score

    status = "ok" if np.isfinite(best_score) else "nonfinite_score"
    if float(np.asarray(mass).sum()) <= 0.0:
        status = "empty_prediction"

    return {
        "likelihood_optimal_score": float(best_score),
        "likelihood_optimal_sigma_m": float(best_sigma),
        "likelihood_score_sigma0": float(score_zero),
        "likelihood_num_evaluations": int(len(cache)),
        "likelihood_status": status,
    }


def compute_likelihood_metrics(
    sample: dict[str, Any],
    config: dict[str, Any] | None = None,
) -> dict[str, float | int | str]:
    return optimize_sigma_golden(
        mass=sample["pred_lambda"],
        gt_support=sample["gt_support"],
        geometry=sample["geometry"],
        config=config,
    )
