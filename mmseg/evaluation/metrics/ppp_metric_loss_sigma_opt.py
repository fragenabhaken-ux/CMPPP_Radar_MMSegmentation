"""
Sigma-optimized PPP loss metric helpers.

This module provides the core functionality for evaluating a predicted
nonnegative 2D grid against a ground-truth support mask by first applying
Gaussian smoothing with blur scale sigma and then scoring the smoothed grid
with a PPP-style loss-derived metric.

Big picture
-----------
The goal of this module is to answer the following question:

    "At which Gaussian smoothing scale sigma does a given prediction best
     match the ground-truth support under a chosen score?"

This is useful because sparse point-cloud-like targets on a large pixel grid
lead to very brittle exact-pixel comparisons. A Gaussian blur introduces a
controlled spatial tolerance: small sigma rewards almost exact localization,
while larger sigma allows near misses to still contribute to the score.

What `pred_grid` means
----------------------
The central input is called `pred_grid`. This name is intentionally generic.

`pred_grid` is a nonnegative 2D array on the final output pixel grid. Depending
on the calling context, it may represent:

- a predicted discrete PPP intensity or
- a discretized point cloud / count grid.

In other words, this module does not assume that the prediction necessarily
comes from one specific representation. It only assumes that the prediction has
already been converted into a nonnegative 2D grid on the evaluation domain.

What `gt_support` means
-----------------------
`gt_support` is a 2D array of the same shape as `pred_grid`.

All entries with value `> 0` are interpreted as ground-truth support locations.
Internally, the module converts this to a boolean mask via

    gt_idx = (gt_support > 0).

Thus, the exact numeric values inside `gt_support` do not matter here beyond
their support-indicating role.

Core workflow
-------------
For a given prediction `pred_grid` and support mask `gt_support`, the workflow
is:

1. Start from the raw nonnegative prediction grid.
2. For a candidate sigma, Gaussianize the grid.
3. Score the Gaussianized grid against the GT support.
4. Optimize sigma with a chosen search method.
5. Return the best sigma and corresponding score.

Gaussianization
---------------
For `sigma > 0`, Gaussianization means applying a Gaussian blur to `pred_grid`
while approximately preserving total mass:

    h_sigma = gaussian_filter(pred_grid / mass, sigma)
    h_sigma <- h_sigma * mass / h_sigma.sum()

If `sigma == 0`, the original grid is returned unchanged.

Metric score options
------------------------

1. PPP-style log-likelihood score (`score_kind="loglik"`)

    score(h) = sum_{x in GT} log h(x) + n_GT * log(HW) - sum h

Interpretation:
- rewards high mass exactly on GT support locations
- penalizes total predicted mass
- prefers placing the bulk of the mass near the correct locations

2. Overlap-style score (`score_kind="mmd"`)

    score(h) = sum_{x in GT} h(x)

Interpretation:
- rewards mass on GT support
- does not include the PPP mass penalty
- is more overlap-like than a full likelihood criterion

Normalization
-------------
`normalize_by_gt=True`.

If enabled, the score is divided by the number of GT support points. This makes
scores more comparable across samples with different numbers of GT points.

Supported optimization methods
------------------------------
This production version supports the following methods:

- `"grid_naive"`:
    Exact grid search over a user-provided sigma grid.

- `"ternary"`:
    Derivative-free ternary search in log-space.

- `"golden"`:
    Derivative-free golden-section search in log-space.

- `"hybrid"`:
    Two-stage approach:
      1. coarse exact grid search
      2. local golden refinement in the best coarse interval

return contract
-----------------------
The main production entry point is

    evaluate_ppp_loss_sigma_opt(...)

It returns a stable minimal contract:

    {
        "best_sigma": float,
        "best_score": float,
        "status": str,
        "score_kind": str,
        "method": str,
    }

This contract is intended to be passed directly into `self.results` inside an
MMSegmentation / MMEngine metric.

Practical interpretation
------------------------
This module provides a rich notion of prediction quality:
- count / total mass matters implicitly in the PPP log-likelihood score
- spatial localization matters directly
- the optimal sigma indicates how much tolerance was needed to obtain a good
  match
"""

from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np
from scipy.ndimage import gaussian_filter


LOGGER_NAME = "ppp_metric_loss"
LOG_LEVEL = logging.INFO  # change to logging.INFO, logging.DEBUG

logger = logging.getLogger(LOGGER_NAME)
logger.setLevel(LOG_LEVEL)

EPS_LOG = 1e-12


# ============================================================
# Basic helpers
# ============================================================

def prepare_pred_grid(pred_grid: np.ndarray) -> np.ndarray:
    """Convert input to float64 and clip negative values to zero."""
    x = np.asarray(pred_grid, dtype=np.float64)
    return np.clip(x, 0.0, None)


def prepare_gt_support(gt_support: np.ndarray) -> np.ndarray:
    """Convert GT support to a boolean mask."""
    return np.asarray(gt_support) > 0


def validate_same_shape(pred_grid: np.ndarray, gt_support: np.ndarray) -> None:
    if np.asarray(pred_grid).shape != np.asarray(gt_support).shape:
        raise ValueError("pred_grid and gt_support must have the same shape")


def _validate_score_kind(score_kind: str) -> str:
    score_kind = str(score_kind).lower()
    if score_kind not in {"loglik", "mmd"}:
        raise ValueError(f"Unknown score_kind: {score_kind}")
    return score_kind


def _validate_method(method: str) -> str:
    method = str(method).lower()
    if method not in {"hybrid", "golden", "ternary", "grid_naive"}:
        raise ValueError(f"Unknown method: {method}")
    return method


# ============================================================
# Score functions
# ============================================================

def ppp_loglik_score(
    h: np.ndarray,
    gt_support: np.ndarray,
    eps: float = EPS_LOG,
    normalize_by_gt: bool = True,
) -> float:
    """
    PPP-style score on a discrete grid:

        score(h) = sum_{x in GT} log h(x) + n_GT * log(HW) - sum h

    If normalize_by_gt=True, divide by n_GT.
    """
    h = np.asarray(h, dtype=np.float64)
    gt_idx = prepare_gt_support(gt_support)

    H, W = h.shape
    n_gt = int(gt_idx.sum())

    if n_gt == 0:
        return float(-h.sum())

    score = float(
        np.log(np.clip(h[gt_idx], eps, None)).sum()
        + n_gt * np.log(H * W)
        - h.sum()
    )

    if normalize_by_gt:
        score /= n_gt

    return float(score)


def ppp_default_mmd_score(
    h: np.ndarray,
    gt_support: np.ndarray,
    normalize_by_gt: bool = True,
) -> float:
    """
    Overlap-style score:

        score(h) = sum_{x in GT} h(x)

    If normalize_by_gt=True, divide by n_GT.
    """
    h = np.asarray(h, dtype=np.float64)
    gt_idx = prepare_gt_support(gt_support)

    score = float(h[gt_idx].sum())

    if normalize_by_gt:
        n_gt = int(gt_idx.sum())
        if n_gt > 0:
            score /= n_gt

    return float(score)


def score_grid(
    h: np.ndarray,
    gt_support: np.ndarray,
    score_kind: str = "loglik",
    eps_h: float = EPS_LOG,
    normalize_by_gt: bool = True,
) -> float:
    """Unified score evaluation on a Gaussianized grid h."""
    score_kind = _validate_score_kind(score_kind)

    if score_kind == "loglik":
        return ppp_loglik_score(
            h=h,
            gt_support=gt_support,
            eps=eps_h,
            normalize_by_gt=normalize_by_gt,
        )

    if score_kind == "mmd":
        return ppp_default_mmd_score(
            h=h,
            gt_support=gt_support,
            normalize_by_gt=normalize_by_gt,
        )

    raise RuntimeError("Unreachable")


def score_sigma_zero(
    pred_grid: np.ndarray,
    gt_support: np.ndarray,
    eps_h: float = EPS_LOG,
    normalize_by_gt: bool = True,
    score_kind: str = "loglik",
) -> float:
    """Evaluate the chosen score on the raw grid (sigma = 0)."""
    h = np.asarray(pred_grid, dtype=np.float64)
    return score_grid(
        h=h,
        gt_support=gt_support,
        score_kind=score_kind,
        eps_h=eps_h,
        normalize_by_gt=normalize_by_gt,
    )


# ============================================================
# Gaussianization
# ============================================================

def gaussianize_grid(
    pred_grid: np.ndarray,
    sigma: float,
    truncate: float = 4.0,
) -> np.ndarray:
    """
    Apply Gaussian blur while approximately preserving total mass.

    If sigma == 0, the original grid is returned.
    """
    x = prepare_pred_grid(pred_grid)
    mass = float(x.sum())

    if mass <= 0:
        return np.zeros_like(x, dtype=np.float64)

    sigma = float(sigma)
    if sigma < 0:
        raise ValueError("sigma must be >= 0")

    if sigma == 0.0:
        return x.copy()

    y = gaussian_filter(
        x / mass,
        sigma=sigma,
        mode="constant",
        cval=0.0,
        truncate=truncate,
    )
    s = float(y.sum())

    if s > 0:
        y *= mass / s
    else:
        y = np.zeros_like(x, dtype=np.float64)

    return y


# ============================================================
# Fixed-sigma evaluation
# ============================================================

def evaluate_sigma_value(
    pred_grid: np.ndarray,
    gt_support: np.ndarray,
    sigma: float,
    score_kind: str = "loglik",
    eps_h: float = EPS_LOG,
    normalize_by_gt: bool = True,
    truncate: float = 4.0,
) -> float:
    """Evaluate the metric at one fixed sigma."""
    sigma = float(max(0.0, sigma))

    if sigma == 0.0:
        return score_sigma_zero(
            pred_grid=pred_grid,
            gt_support=gt_support,
            eps_h=eps_h,
            normalize_by_gt=normalize_by_gt,
            score_kind=score_kind,
        )

    h = gaussianize_grid(pred_grid=pred_grid, sigma=sigma, truncate=truncate)
    return score_grid(
        h=h,
        gt_support=gt_support,
        score_kind=score_kind,
        eps_h=eps_h,
        normalize_by_gt=normalize_by_gt,
    )


# ============================================================
# Optimizers
# ============================================================

def optimize_sigma_grid_naive(
    pred_grid: np.ndarray,
    gt_support: np.ndarray,
    sigma_grid: np.ndarray,
    score_kind: str = "loglik",
    eps_h: float = EPS_LOG,
    normalize_by_gt: bool = True,
    truncate: float = 4.0,
    verbose: bool = False,
) -> dict[str, Any]:
    """Exact grid search over a user-provided sigma grid."""
    score_kind = _validate_score_kind(score_kind)
    sigma_grid = np.asarray(sigma_grid, dtype=np.float64)

    if sigma_grid.ndim != 1 or len(sigma_grid) == 0:
        raise ValueError("sigma_grid must be a nonempty 1D array")
    if np.any(sigma_grid < 0):
        raise ValueError("sigma_grid must be nonnegative")
    if np.any(np.diff(sigma_grid) <= 0):
        raise ValueError("sigma_grid must be strictly increasing")

    x = prepare_pred_grid(pred_grid)
    gt_support = np.asarray(gt_support)

    validate_same_shape(x, gt_support)

    if x.sum() <= 0:
        return {
            "best_sigma": float(sigma_grid[0]),
            "best_score": float("-inf"),
            "status": "empty_grid",
            "score_kind": score_kind,
            "method": "grid_naive",
        }

    best_sigma = float(sigma_grid[0])
    best_score = float("-inf")

    for sigma in sigma_grid:
        score = evaluate_sigma_value(
            pred_grid=x,
            gt_support=gt_support,
            sigma=float(sigma),
            score_kind=score_kind,
            eps_h=eps_h,
            normalize_by_gt=normalize_by_gt,
            truncate=truncate,
        )
        if score > best_score:
            best_score = float(score)
            best_sigma = float(sigma)

        if verbose:
            logger.debug(
                "grid_naive | sigma=%.6f | score=%.6f",
                float(sigma),
                float(score),
            )

    return {
        "best_sigma": float(best_sigma),
        "best_score": float(best_score),
        "status": "ok",
        "score_kind": score_kind,
        "method": "grid_naive",
    }


def optimize_sigma_ternary(
    pred_grid: np.ndarray,
    gt_support: np.ndarray,
    sigma_min: float = 0.0,
    sigma_max: float = 100.0,
    max_iter: int = 30,
    tol_sigma: float = 0.5,
    score_kind: str = "loglik",
    eps_h: float = EPS_LOG,
    normalize_by_gt: bool = True,
    truncate: float = 4.0,
    check_zero: bool = True,
    verbose: bool = False,
) -> dict[str, Any]:
    """Derivative-free ternary search in log-space."""
    score_kind = _validate_score_kind(score_kind)

    if sigma_max <= sigma_min:
        raise ValueError("Need sigma_max > sigma_min")
    if sigma_min < 0:
        raise ValueError("sigma_min must be >= 0")

    x = prepare_pred_grid(pred_grid)
    gt_support = np.asarray(gt_support)

    validate_same_shape(x, gt_support)

    if x.sum() <= 0:
        return {
            "best_sigma": float(sigma_min),
            "best_score": float("-inf"),
            "status": "empty_grid",
            "score_kind": score_kind,
            "method": "ternary",
        }

    eval_cache: dict[float, float] = {}

    def eval_sigma(s: float) -> float:
        s = float(max(0.0, s))
        key = round(s, 12)
        if key not in eval_cache:
            eval_cache[key] = float(
                evaluate_sigma_value(
                    pred_grid=x,
                    gt_support=gt_support,
                    sigma=s,
                    score_kind=score_kind,
                    eps_h=eps_h,
                    normalize_by_gt=normalize_by_gt,
                    truncate=truncate,
                )
            )
        return eval_cache[key]

    best_sigma = None
    best_score = float("-inf")

    if check_zero and sigma_min == 0.0:
        score0 = eval_sigma(0.0)
        best_sigma = 0.0
        best_score = score0

    left = math.log(max(sigma_min, 0.1))
    right = math.log(max(sigma_max, 0.1))

    for k in range(max_iter):
        if (math.exp(right) - math.exp(left)) < tol_sigma:
            break

        m1 = left + (right - left) / 3.0
        m2 = right - (right - left) / 3.0

        s1_sigma = math.exp(m1)
        s2_sigma = math.exp(m2)

        s1 = eval_sigma(s1_sigma)
        s2 = eval_sigma(s2_sigma)

        if s1 > best_score:
            best_score = s1
            best_sigma = s1_sigma
        if s2 > best_score:
            best_score = s2
            best_sigma = s2_sigma

        if verbose:
            logger.debug(
                "ternary | iter=%d | left=%.6f | s1=%.6f | s2=%.6f | right=%.6f | score1=%.6f | score2=%.6f",
                k,
                math.exp(left),
                s1_sigma,
                s2_sigma,
                math.exp(right),
                s1,
                s2,
            )

        if s1 < s2:
            left = m1
        else:
            right = m2

    sigma_mid = math.exp(0.5 * (left + right))
    score_mid = eval_sigma(sigma_mid)

    if score_mid > best_score or best_sigma is None:
        best_score = score_mid
        best_sigma = sigma_mid

    return {
        "best_sigma": float(best_sigma),
        "best_score": float(best_score),
        "status": "ok",
        "score_kind": score_kind,
        "method": "ternary",
    }


def optimize_sigma_golden(
    pred_grid: np.ndarray,
    gt_support: np.ndarray,
    sigma_min: float = 0.0,
    sigma_max: float = 100.0,
    max_iter: int = 30,
    tol_sigma: float = 0.5,
    score_kind: str = "loglik",
    eps_h: float = EPS_LOG,
    normalize_by_gt: bool = True,
    truncate: float = 4.0,
    check_zero: bool = True,
    verbose: bool = False,
) -> dict[str, Any]:
    """Derivative-free golden-section search in log-space."""
    score_kind = _validate_score_kind(score_kind)

    if sigma_max <= sigma_min:
        raise ValueError("Need sigma_max > sigma_min")
    if sigma_min < 0:
        raise ValueError("sigma_min must be >= 0")

    x = prepare_pred_grid(pred_grid)
    gt_support = np.asarray(gt_support)

    validate_same_shape(x, gt_support)

    if x.sum() <= 0:
        return {
            "best_sigma": float(sigma_min),
            "best_score": float("-inf"),
            "status": "empty_grid",
            "score_kind": score_kind,
            "method": "golden",
        }

    eval_cache: dict[float, float] = {}

    def eval_sigma(s: float) -> float:
        s = float(max(0.0, s))
        key = round(s, 12)
        if key not in eval_cache:
            eval_cache[key] = float(
                evaluate_sigma_value(
                    pred_grid=x,
                    gt_support=gt_support,
                    sigma=s,
                    score_kind=score_kind,
                    eps_h=eps_h,
                    normalize_by_gt=normalize_by_gt,
                    truncate=truncate,
                )
            )
        return eval_cache[key]

    best_sigma = None
    best_score = float("-inf")

    if check_zero and sigma_min == 0.0:
        score0 = eval_sigma(0.0)
        best_sigma = 0.0
        best_score = score0

    log_left = math.log(max(sigma_min, 0.1))
    log_right = math.log(max(sigma_max, 0.1))

    phi = (1.0 + math.sqrt(5.0)) / 2.0
    inv_phi = 1.0 / phi

    c = log_right - (log_right - log_left) * inv_phi
    d = log_left + (log_right - log_left) * inv_phi

    sc = eval_sigma(math.exp(c))
    sd = eval_sigma(math.exp(d))

    if sc > best_score:
        best_score = sc
        best_sigma = math.exp(c)
    if sd > best_score:
        best_score = sd
        best_sigma = math.exp(d)

    for k in range(max_iter):
        if (math.exp(log_right) - math.exp(log_left)) < tol_sigma:
            break

        if verbose:
            logger.debug(
                "golden | iter=%d | left=%.6f | c=%.6f | d=%.6f | right=%.6f | sc=%.6f | sd=%.6f",
                k,
                math.exp(log_left),
                math.exp(c),
                math.exp(d),
                math.exp(log_right),
                sc,
                sd,
            )

        if sc < sd:
            log_left = c
            c = d
            sc = sd
            d = log_left + (log_right - log_left) * inv_phi
            sd = eval_sigma(math.exp(d))
            if sd > best_score:
                best_score = sd
                best_sigma = math.exp(d)
        else:
            log_right = d
            d = c
            sd = sc
            c = log_right - (log_right - log_left) * inv_phi
            sc = eval_sigma(math.exp(c))
            if sc > best_score:
                best_score = sc
                best_sigma = math.exp(c)

    sigma_mid = math.exp(0.5 * (log_left + log_right))
    score_mid = eval_sigma(sigma_mid)

    if score_mid > best_score or best_sigma is None:
        best_score = score_mid
        best_sigma = sigma_mid

    return {
        "best_sigma": float(best_sigma),
        "best_score": float(best_score),
        "status": "ok",
        "score_kind": score_kind,
        "method": "golden",
    }


def optimize_sigma_hybrid(
    pred_grid: np.ndarray,
    gt_support: np.ndarray,
    sigma_max: float = 120.0,
    n_coarse: int = 21,
    refine_max_iter: int = 30,
    refine_tol_sigma: float = 0.5,
    score_kind: str = "loglik",
    eps_h: float = EPS_LOG,
    normalize_by_gt: bool = True,
    truncate: float = 4.0,
    verbose: bool = False,
) -> dict[str, Any]:
    """
    Two-stage optimizer:
      1) exact coarse grid search
      2) local golden refinement around the best coarse interval
    """
    score_kind = _validate_score_kind(score_kind)

    if sigma_max <= 0:
        raise ValueError("sigma_max must be > 0")
    if n_coarse < 3:
        raise ValueError("n_coarse must be >= 3")

    x = prepare_pred_grid(pred_grid)
    gt_support = np.asarray(gt_support)

    validate_same_shape(x, gt_support)

    if x.sum() <= 0:
        return {
            "best_sigma": 0.0,
            "best_score": float("-inf"),
            "status": "empty_grid",
            "score_kind": score_kind,
            "method": "hybrid",
        }

    sigma_grid = np.linspace(0.0, sigma_max, n_coarse, dtype=np.float64)

    coarse_scores = np.empty_like(sigma_grid, dtype=np.float64)
    for i, sigma in enumerate(sigma_grid):
        coarse_scores[i] = evaluate_sigma_value(
            pred_grid=x,
            gt_support=gt_support,
            sigma=float(sigma),
            score_kind=score_kind,
            eps_h=eps_h,
            normalize_by_gt=normalize_by_gt,
            truncate=truncate,
        )

        if verbose:
            logger.debug(
                "hybrid coarse | idx=%d | sigma=%.6f | score=%.6f",
                i,
                float(sigma),
                float(coarse_scores[i]),
            )

    best_idx = int(np.argmax(coarse_scores))
    best_sigma = float(sigma_grid[best_idx])
    best_score = float(coarse_scores[best_idx])

    if best_idx == 0:
        return {
            "best_sigma": float(best_sigma),
            "best_score": float(best_score),
            "status": "zero_boundary_selected",
            "score_kind": score_kind,
            "method": "hybrid",
        }

    if best_idx == len(sigma_grid) - 1:
        left = float(sigma_grid[-2])
        right = float(sigma_grid[-1])
    else:
        left = float(sigma_grid[best_idx - 1])
        right = float(sigma_grid[best_idx + 1])

    refine_res = optimize_sigma_golden(
        pred_grid=x,
        gt_support=gt_support,
        sigma_min=max(0.0, left),
        sigma_max=right,
        max_iter=refine_max_iter,
        tol_sigma=refine_tol_sigma,
        score_kind=score_kind,
        eps_h=eps_h,
        normalize_by_gt=normalize_by_gt,
        truncate=truncate,
        check_zero=False,
        verbose=verbose,
    )

    if refine_res["best_score"] > best_score:
        return {
            "best_sigma": float(refine_res["best_sigma"]),
            "best_score": float(refine_res["best_score"]),
            "status": "refined",
            "score_kind": score_kind,
            "method": "hybrid",
        }

    return {
        "best_sigma": float(best_sigma),
        "best_score": float(best_score),
        "status": "coarse_best",
        "score_kind": score_kind,
        "method": "hybrid",
    }


# ============================================================
# Dispatcher + stable wrapper
# ============================================================

def optimize_sigma_dispatch(
    pred_grid: np.ndarray,
    gt_support: np.ndarray,
    method: str = "hybrid",
    **kwargs: Any,
) -> dict[str, Any]:
    """Dispatch to the selected optimizer."""
    method = _validate_method(method)

    if method == "grid_naive":
        sigma_grid = kwargs.pop("sigma_grid", np.linspace(0.0, 120.0, 25))
        return optimize_sigma_grid_naive(
            pred_grid=pred_grid,
            gt_support=gt_support,
            sigma_grid=sigma_grid,
            **kwargs,
        )

    if method == "ternary":
        defaults = dict(
            sigma_min=0.0,
            sigma_max=120.0,
            max_iter=30,
            tol_sigma=0.5,
            score_kind="loglik",
            eps_h=EPS_LOG,
            normalize_by_gt=True,
            truncate=4.0,
            check_zero=True,
            verbose=False,
        )
        defaults.update(kwargs)
        return optimize_sigma_ternary(pred_grid=pred_grid, gt_support=gt_support, **defaults)

    if method == "golden":
        defaults = dict(
            sigma_min=0.0,
            sigma_max=120.0,
            max_iter=30,
            tol_sigma=0.5,
            score_kind="loglik",
            eps_h=EPS_LOG,
            normalize_by_gt=True,
            truncate=4.0,
            check_zero=True,
            verbose=False,
        )
        defaults.update(kwargs)
        return optimize_sigma_golden(pred_grid=pred_grid, gt_support=gt_support, **defaults)

    if method == "hybrid":
        defaults = dict(
            sigma_max=120.0,
            n_coarse=21,
            refine_max_iter=30,
            refine_tol_sigma=0.5,
            score_kind="loglik",
            eps_h=EPS_LOG,
            normalize_by_gt=True,
            truncate=4.0,
            verbose=False,
        )
        defaults.update(kwargs)
        return optimize_sigma_hybrid(pred_grid=pred_grid, gt_support=gt_support, **defaults)

    raise RuntimeError("Unreachable")


def evaluate_ppp_loss_sigma_opt(
    pred_grid: np.ndarray,
    gt_support: np.ndarray,
    method: str = "hybrid",
    score_kind: str = "loglik",
    verbose: bool = False,
    **kwargs: Any,
) -> dict[str, Any]:
    """
    Stable production wrapper returning a minimal contract only.

    Returned dict:
        {
            "best_sigma": float,
            "best_score": float,
            "status": str,
            "score_kind": str,
            "method": str,
        }
    """
    method = _validate_method(method)
    score_kind = _validate_score_kind(score_kind)

    res = optimize_sigma_dispatch(
        pred_grid=pred_grid,
        gt_support=gt_support,
        method=method,
        score_kind=score_kind,
        verbose=verbose,
        **kwargs,
    )

    return {
        "best_sigma": float(res["best_sigma"]),
        "best_score": float(res["best_score"]),
        "status": str(res.get("status", "ok")),
        "score_kind": score_kind,
        "method": method,
    }


__all__ = [
    "LOGGER_NAME",
    "LOG_LEVEL",
    "EPS_LOG",
    "prepare_pred_grid",
    "prepare_gt_support",
    "validate_same_shape",
    "ppp_loglik_score",
    "ppp_default_mmd_score",
    "score_grid",
    "score_sigma_zero",
    "gaussianize_grid",
    "evaluate_sigma_value",
    "optimize_sigma_grid_naive",
    "optimize_sigma_ternary",
    "optimize_sigma_golden",
    "optimize_sigma_hybrid",
    "optimize_sigma_dispatch",
    "evaluate_ppp_loss_sigma_opt",
]