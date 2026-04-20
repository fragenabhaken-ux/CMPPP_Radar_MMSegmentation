"""
Gaussianized evaluation metric for prediction grids and ground-truth point supports.

This module implements two score variants for comparing a predicted nonnegative
2D grid against a ground-truth support set: a PPP log-likelihood style score
and an overlap-based score derived from the MMD perspective. In both cases, the
predicted grid is first Gaussianized with an optimally chosen blur scale sigma,
and the optimal sigma is determined by golden-section search.

Overview
--------
The intended use case is the evaluation of spatial predictions where the model
output is already given on a 2D grid, for example as:
- a predicted PPP intensity / heatmap, or
- a binary / count grid obtained from a point-cloud prediction.

The ground truth is represented as a support mask on the same grid. The metric
then asks:

    "How well does the prediction match the ground-truth support, after adding
     the amount of Gaussian smoothing that yields the best possible agreement?"

This removes the need to fix a Gaussian scale by hand and allows evaluation of
different prediction representations within one common framework.

Implemented score variants
--------------------------
Two score variants are supported.

1. score_type="loglik"
   PPP log-likelihood style score on the Gaussianized grid h:

       score(h) = sum_{x in GT} log h(x) + n_GT * log(HW) - sum h

   Interpretation:
   - rewards high predicted intensity at GT locations,
   - penalizes excessive total mass,
   - is sensitive to whether predicted mass is placed in the right regions.

2. score_type="mmd"
   Overlap-style score on the Gaussianized grid h:

       score(h) = sum_{x in GT} h(x)

   Interpretation:
   - rewards predicted mass on GT support,
   - does not include the PPP mass penalty term,
   - is simpler and more directly overlap-oriented.

Although these are mathematically different objectives, they share the same
pipeline:
- prepare the prediction grid,
- Gaussianize it,
- optimize sigma,
- evaluate the resulting score.

For this reason, both variants are implemented in one common module.

Why sigma is optimized
----------------------
The Gaussian scale sigma controls how much spatial tolerance is allowed.

- Small sigma favors almost pixel-exact alignment.
- Larger sigma tolerates small localization errors by spreading predicted mass.

Instead of selecting sigma manually, this module chooses the sigma that
maximizes the selected score for the given prediction/GT pair. This avoids an
arbitrary fixed choice and makes the metric less dependent on a user-selected
blur parameter.

Why sigma = 0 is checked explicitly
-----------------------------------
The case sigma = 0 corresponds to "no Gaussianization": the raw input grid is
evaluated directly.

This case is checked explicitly because sigma = 0 is a meaningful candidate of
the optimization problem and corresponds to the no-smoothing baseline. The
golden-section search itself is then carried out on the positive interval
(sigma > 0). At the end, the best positive-sigma candidate is compared against
sigma = 0.

This is not meant as a separate optimization method, but simply ensures that
the no-blur case remains an admissible candidate.

Optimization method
-------------------
The Gaussian scale is optimized with golden-section search on the positive
sigma interval.

Why golden search:
- it is derivative-free,
- simple and robust,
- requires only repeated score evaluations,
- well suited here because the metric is ultimately defined by evaluating the
  score after Gaussian filtering, not by exploiting analytic derivatives.

The implementation assumes that a local maximum can be found sufficiently well
within the specified search interval. In practice, the method is intended as a
stable and compact optimizer for metric evaluation rather than as a general
global optimization framework.

Mass-preserving Gaussianization
-------------------------------
For sigma > 0, the prediction grid is blurred with a Gaussian filter. After
blurring, the result is rescaled so that the total mass approximately matches
the original total mass.

Concretely, for input grid x with total mass m:

    1. normalize x by m,
    2. apply Gaussian blur,
    3. rescale back to total mass m.

This is important because both score variants depend on the magnitude and
distribution of the predicted mass. Without this rescaling, Gaussian filtering
with zero-padding at the boundary would artificially change the total mass.

Expected inputs
---------------
The module assumes:

- pred_grid:
    A nonnegative 2D array representing the prediction on the grid.
    Negative values are clipped to zero by prepare_grid().

- gt_support:
    A 2D array of the same shape as pred_grid.
    Every entry > 0 is treated as a ground-truth support location.

The code intentionally keeps these assumptions minimal. Dataset-specific tensor
handling, channel selection, exponentiation of logits, or conversion from model
outputs into evaluation grids should be handled outside this module.

Main entry point
----------------
The usual entry point is:

    compute_optimal_sigma_metric(...)

It:
- validates the inputs,
- optimizes sigma with golden search,
- evaluates the selected score at the optimal sigma,
- returns the optimal score and the corresponding sigma.

Design note
-----------
This module is intentionally focused. It supports exactly:
- two score variants ("loglik" and "mmd"),
- one optimization strategy (golden-section search).

The goal is not to expose a large optimization interface, but to provide a
compact and readable implementation of the metric used in evaluation.
"""

import math
from typing import Any

import numpy as np
from scipy.ndimage import gaussian_filter


EPS_LOG = 1e-12


def prepare_grid(grid: np.ndarray) -> np.ndarray:
    """Convert input to a nonnegative float64 grid."""
    x = np.asarray(grid, dtype=np.float64)
    return np.clip(x, 0.0, None)


def gaussianize_grid(grid: np.ndarray, sigma: float) -> np.ndarray:
    """
    Apply Gaussian blur while approximately preserving total mass.
    """
    x = prepare_grid(grid)
    mass = float(x.sum())

    if mass <= 0.0:
        return np.zeros_like(x, dtype=np.float64)

    sigma = float(sigma)
    if sigma == 0.0:
        return x.copy()

    y = gaussian_filter(x / mass, sigma=sigma, mode="constant")
    y_sum = float(y.sum())

    if y_sum > 0.0:
        y *= mass / y_sum
    else:
        y = np.zeros_like(x, dtype=np.float64)

    return y


def _validate_score_type(score_type: str) -> str:
    score_type = str(score_type).lower()
    if score_type not in {"loglik", "mmd"}:
        raise ValueError(f"Unknown score_type: {score_type}")
    return score_type


def loglik_score(
    pred_grid: np.ndarray,
    gt_support: np.ndarray,
    eps: float = EPS_LOG,
    normalize_by_gt: bool = True,
) -> float:
    """
    PPP log-likelihood style score on a discrete grid:

        score(h) = sum_{x in GT} log h(x) + n_GT * log(HW) - sum h
    """
    h = np.asarray(pred_grid, dtype=np.float64)
    gt_mask = np.asarray(gt_support) > 0

    if h.shape != gt_mask.shape:
        raise ValueError("pred_grid and gt_support must have the same shape")

    h_h, h_w = h.shape
    n_gt = int(gt_mask.sum())

    score = float(
        np.log(np.clip(h[gt_mask], eps, None)).sum()
        + n_gt * np.log(h_h * h_w)
        - h.sum()
    )

    if normalize_by_gt and n_gt > 0:
        score /= n_gt

    return float(score)


def mmd_overlap_score(
    pred_grid: np.ndarray,
    gt_support: np.ndarray,
    normalize_by_gt: bool = True,
) -> float:
    """
    Overlap-based score on a discrete grid:

        score(h) = sum_{x in GT} h(x)
    """
    h = np.asarray(pred_grid, dtype=np.float64)
    gt_mask = np.asarray(gt_support) > 0

    if h.shape != gt_mask.shape:
        raise ValueError("pred_grid and gt_support must have the same shape")

    score = float(h[gt_mask].sum())

    if normalize_by_gt:
        n_gt = int(gt_mask.sum())
        if n_gt > 0:
            score /= n_gt

    return float(score)


def score_grid(
    pred_grid: np.ndarray,
    gt_support: np.ndarray,
    score_type: str = "loglik",
    eps: float = EPS_LOG,
    normalize_by_gt: bool = True,
) -> float:
    """Evaluate the selected score on a grid."""
    score_type = _validate_score_type(score_type)

    if score_type == "loglik":
        return loglik_score(
            pred_grid=pred_grid,
            gt_support=gt_support,
            eps=eps,
            normalize_by_gt=normalize_by_gt,
        )

    if score_type == "mmd":
        return mmd_overlap_score(
            pred_grid=pred_grid,
            gt_support=gt_support,
            normalize_by_gt=normalize_by_gt,
        )

    raise RuntimeError("Unreachable")


def _evaluate_sigma(
    pred_grid: np.ndarray,
    gt_support: np.ndarray,
    sigma: float,
    score_type: str = "loglik",
    eps: float = EPS_LOG,
    normalize_by_gt: bool = True,
) -> float:
    """Blur prediction with sigma and evaluate the selected score."""
    sigma = float(max(0.0, sigma))

    if sigma == 0.0:
        blurred = prepare_grid(pred_grid)
    else:
        blurred = gaussianize_grid(pred_grid, sigma)

    return score_grid(
        pred_grid=blurred,
        gt_support=gt_support,
        score_type=score_type,
        eps=eps,
        normalize_by_gt=normalize_by_gt,
    )


def optimize_sigma_golden(
    pred_grid: np.ndarray,
    gt_support: np.ndarray,
    score_type: str = "loglik",
    sigma_min: float = 0.0,
    sigma_max: float = 120.0,
    max_iter: int = 30,
    tol_sigma: float = 0.5,
    eps: float = EPS_LOG,
    normalize_by_gt: bool = True,
    check_zero: bool = True,
) -> dict[str, Any]:
    """
    Optimize the Gaussian scale with golden-section search on the positive
    sigma interval and optionally compare against sigma = 0.
    """
    score_type = _validate_score_type(score_type)
    pred_grid = prepare_grid(pred_grid)
    gt_support = np.asarray(gt_support)

    if pred_grid.shape != gt_support.shape:
        raise ValueError("pred_grid and gt_support must have the same shape")

    if sigma_min < 0.0:
        raise ValueError("sigma_min must be >= 0")
    if sigma_max <= sigma_min:
        raise ValueError("sigma_max must be > sigma_min")

    if pred_grid.sum() <= 0.0:
        return {
            "best_sigma": float(sigma_min),
            "best_score": float("-inf"),
            "status": "empty_prediction",
        }

    eval_cache: dict[float, float] = {}

    def eval_sigma(sigma: float) -> float:
        sigma = float(max(0.0, sigma))
        key = round(sigma, 12)

        if key in eval_cache:
            return eval_cache[key]

        score = _evaluate_sigma(
            pred_grid=pred_grid,
            gt_support=gt_support,
            sigma=sigma,
            score_type=score_type,
            eps=eps,
            normalize_by_gt=normalize_by_gt,
        )
        eval_cache[key] = float(score)
        return float(score)

    best_sigma = None
    best_score = float("-inf")

    if check_zero and sigma_min == 0.0:
        score_zero = eval_sigma(0.0)
        best_sigma = 0.0
        best_score = score_zero

    positive_left = max(sigma_min, np.finfo(np.float64).eps)
    positive_right = sigma_max

    if positive_right <= positive_left:
        return {
            "best_sigma": float(best_sigma if best_sigma is not None else sigma_min),
            "best_score": float(best_score),
            "status": "ok",
        }

    phi = (1.0 + math.sqrt(5.0)) / 2.0
    inv_phi = 1.0 / phi

    # Search in log-space for finer relative resolution at small sigma values.
    log_left = math.log(max(positive_left, 0.1))
    log_right = math.log(max(positive_right, 0.1))

    c = log_right - (log_right - log_left) * inv_phi
    d = log_left + (log_right - log_left) * inv_phi

    c_sigma = math.exp(c)
    d_sigma = math.exp(d)

    score_c = eval_sigma(c_sigma)
    score_d = eval_sigma(d_sigma)

    if score_c > best_score:
        best_score = score_c
        best_sigma = c_sigma

    if score_d > best_score:
        best_score = score_d
        best_sigma = d_sigma

    for _ in range(max_iter):
        if (math.exp(log_right) - math.exp(log_left)) < tol_sigma:
            break

        if score_c < score_d:
            log_left = c
            c = d
            score_c = score_d
            d = log_left + (log_right - log_left) * inv_phi
            d_sigma = math.exp(d)
            score_d = eval_sigma(d_sigma)

            if score_d > best_score:
                best_score = score_d
                best_sigma = d_sigma
        else:
            log_right = d
            d = c
            score_d = score_c
            c = log_right - (log_right - log_left) * inv_phi
            c_sigma = math.exp(c)
            score_c = eval_sigma(c_sigma)

            if score_c > best_score:
                best_score = score_c
                best_sigma = c_sigma

    sigma_mid = math.exp(0.5 * (log_left + log_right))
    score_mid = eval_sigma(sigma_mid)

    if score_mid > best_score:
        best_score = score_mid
        best_sigma = sigma_mid

    return {
        "best_sigma": float(best_sigma),
        "best_score": float(best_score),
        "status": "ok",
    }


def compute_optimal_sigma_metric(
    pred_grid: np.ndarray,
    gt_support: np.ndarray,
    score_type: str = "loglik",
    sigma_min: float = 0.0,
    sigma_max: float = 120.0,
    max_iter: int = 30,
    tol_sigma: float = 0.5,
    eps: float = EPS_LOG,
    normalize_by_gt: bool = True,
    check_zero: bool = True,
) -> dict[str, Any]:
    """
    Compute the optimal-sigma score for a prediction grid and GT support.
    """
    score_type = _validate_score_type(score_type)
    pred_grid = prepare_grid(pred_grid)
    gt_support = np.asarray(gt_support)

    if pred_grid.shape != gt_support.shape:
        raise ValueError("pred_grid and gt_support must have the same shape")

    result = optimize_sigma_golden(
        pred_grid=pred_grid,
        gt_support=gt_support,
        score_type=score_type,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        max_iter=max_iter,
        tol_sigma=tol_sigma,
        eps=eps,
        normalize_by_gt=normalize_by_gt,
        check_zero=check_zero,
    )

    best_sigma = float(result["best_sigma"])
    best_score = float(result["best_score"])

    return {
        "best_sigma": best_sigma,
        "best_score": best_score,
        "score_type": score_type,
        "status": result["status"],
    }


__all__ = [
    "EPS_LOG",
    "prepare_grid",
    "gaussianize_grid",
    "loglik_score",
    "mmd_overlap_score",
    "score_grid",
    "optimize_sigma_golden",
    "compute_optimal_sigma_metric",
]