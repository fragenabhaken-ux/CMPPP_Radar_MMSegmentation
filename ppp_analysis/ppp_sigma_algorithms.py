import math
import time
from typing import Any

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.signal import fftconvolve


EPS_LOG = 1e-12


# ============================================================
# Basic PPP score helpers
# ============================================================

def prepare_grid(grid):
    """Convert input to a nonnegative float64 grid."""
    x = np.asarray(grid, dtype=np.float64)
    return np.clip(x, 0.0, None)


def ppp_loglik_score(h, gt_support, eps=EPS_LOG, normalize_by_gt=True):
    """
    PPP log-likelihood score on the discrete grid

        score(h) = sum_{x in GT} log h(x) + n_GT * log(HW) - sum h

    If normalize_by_gt=True, divide by the number of GT points.
    """
    h = np.asarray(h, dtype=np.float64)
    gt_idx = np.asarray(gt_support) > 0

    H, W = h.shape
    n_gt = int(gt_idx.sum())

    score = float(
        np.log(np.clip(h[gt_idx], eps, None)).sum()
        + n_gt * np.log(H * W)
        - h.sum()
    )

    if normalize_by_gt and n_gt > 0:
        score /= n_gt

    return float(score)

def ppp_default_MMD(h, gt_support, ps=EPS_LOG, normalize_by_gt=True):
    """
    PPP MMD score on the discrete grid

        score(h) = sum_{x in GT} h(x)

    This is a special case of the PPP score with a Gaussian kernel and zero bandwidth.
    """
    h = np.asarray(h, dtype=np.float64)
    gt_idx = np.asarray(gt_support) > 0

    score = float(h[gt_idx].sum())
    if normalize_by_gt:
        n_gt = int(gt_idx.sum())
        if n_gt > 0:
            score /= n_gt
    return float(score)


def score_sigma_zero(grid, gt_support, eps_h=EPS_LOG, normalize_by_gt=True):
    h = np.asarray(grid, dtype=np.float64)
    return ppp_loglik_score(
        h,
        gt_support,
        eps=eps_h,
        normalize_by_gt=normalize_by_gt,
    )


def gaussianize_grid(grid, sigma):
    """Apply Gaussian blur while approximately preserving total mass."""
    x = prepare_grid(grid)
    mass = x.sum()

    if mass <= 0:
        return np.zeros_like(x, dtype=np.float64)

    if float(sigma) == 0.0:
        return x.copy()

    y = gaussian_filter(x / mass, sigma=float(sigma), mode="constant", cval=0.0)
    s = y.sum()
    if s > 0:
        y *= mass / s

    return y


# ============================================================
# Naive coarse grid search
# ============================================================

def optimize_sigma_for_grid_naive(
    grid,
    gt_support,
    sigma_grid,
    verbose=True,
    eps_h=EPS_LOG,
):
    """
    Naive grid search over sigma.

    For each sigma in sigma_grid:
    - Gaussianize the original grid from scratch
    - evaluate PPP log-likelihood score
    """
    sigma_grid = np.asarray(sigma_grid, dtype=np.float64)

    if sigma_grid.ndim != 1 or len(sigma_grid) == 0:
        raise ValueError("sigma_grid must be a nonempty 1D array")
    if np.any(sigma_grid < 0):
        raise ValueError("sigma_grid must be nonnegative")
    if np.any(np.diff(sigma_grid) <= 0):
        raise ValueError("sigma_grid must be strictly increasing")

    x = prepare_grid(grid)

    if x.sum() <= 0:
        scores = np.full(len(sigma_grid), -np.inf, dtype=np.float64)
        return {
            "sigma_grid": sigma_grid,
            "scores": scores,
            "best_sigma": float(sigma_grid[0]),
            "best_score": float(scores[0]),
            "best_idx": 0,
            "status": "empty_grid",
        }

    scores = []

    for sigma in sigma_grid:
        st = time.perf_counter()

        if float(sigma) == 0.0:
            score = score_sigma_zero(
                x,
                gt_support,
                eps_h=eps_h,
                normalize_by_gt=True,
            )
        else:
            h = gaussianize_grid(x, sigma)
            score = ppp_loglik_score(
                h,
                gt_support,
                eps=eps_h,
                normalize_by_gt=True,
            )

        elapsed = time.perf_counter() - st
        scores.append(float(score))

        if verbose:
            print(
                f"sigma={sigma:8.3f}  "
                f"score={score:12.6f}  "
                f"time={elapsed:.4f}s"
            )

    scores = np.asarray(scores, dtype=np.float64)
    best_idx = int(np.argmax(scores))

    return {
        "sigma_grid": sigma_grid,
        "scores": scores,
        "best_sigma": float(sigma_grid[best_idx]),
        "best_score": float(scores[best_idx]),
        "best_idx": best_idx,
        "status": "ok",
    }


# ============================================================
# Coarse search using Gaussian group property
# ============================================================

def optimize_sigma_for_grid(
    grid,
    gt_support,
    sigma_grid,
    verbose=False,
    truncate=4.0,
    eps_h=EPS_LOG,
):
    """
    Approximate coarse sigma search using Gaussian variance additivity.

    IMPORTANT:
    - Incremental blurring uses the Gaussian semigroup / group property.
    - Before each PPP score evaluation, the blurred grid is rescaled
      back to the original total mass.
    - This keeps the PPP loss definition intact while still exploiting
      the group property approximately.

    This is still an approximation to evaluating scipy's gaussian_filter
    from scratch at each sigma, because of repeated truncation/boundary effects.
    """
    sigma_grid = np.asarray(sigma_grid, dtype=np.float64)

    if sigma_grid.ndim != 1 or len(sigma_grid) == 0:
        raise ValueError("sigma_grid must be a nonempty 1D array")
    if np.any(sigma_grid < 0):
        raise ValueError("sigma_grid must be nonnegative")
    if np.any(np.diff(sigma_grid) <= 0):
        raise ValueError("sigma_grid must be strictly increasing")

    x = prepare_grid(grid)
    gt_support = np.asarray(gt_support)

    if x.shape != gt_support.shape:
        raise ValueError("grid and gt_support must have the same shape")

    mass = x.sum()
    if mass <= 0:
        scores = np.full(len(sigma_grid), -np.inf, dtype=np.float64)
        return {
            "sigma_grid": sigma_grid,
            "scores": scores,
            "best_sigma": float(sigma_grid[0]),
            "best_score": float(scores[0]),
            "best_idx": 0,
            "status": "empty_grid",
        }

    h_work = x / mass
    prev_sigma2 = 0.0
    scores = []

    for sigma in sigma_grid:
        sigma = float(sigma)
        sigma2 = sigma * sigma
        sigma_inc = math.sqrt(max(0.0, sigma2 - prev_sigma2))

        st = time.perf_counter()

        if sigma_inc > 0.0:
            h_work = gaussian_filter(
                h_work,
                sigma=sigma_inc,
                mode="constant",
                cval=0.0,
                truncate=truncate,
            )

        h_score = h_work.copy()
        s = h_score.sum()
        if s > 0:
            h_score *= mass / s
        else:
            h_score = np.zeros_like(h_score, dtype=np.float64)

        score = ppp_loglik_score(
            h_score,
            gt_support,
            eps=eps_h,
            normalize_by_gt=True,
        )
        elapsed = time.perf_counter() - st
        scores.append(float(score))

        if verbose:
            print(
                f"sigma={sigma:8.3f} | sigma_inc={sigma_inc:8.3f} | "
                f"time={elapsed:.4f}s | "
                f"sum_before_rescale={s:.6e} | score={score:12.6f}"
            )

        prev_sigma2 = sigma2

    scores = np.asarray(scores, dtype=np.float64)
    best_idx = int(np.argmax(scores))

    return {
        "sigma_grid": sigma_grid,
        "scores": scores,
        "best_sigma": float(sigma_grid[best_idx]),
        "best_score": float(scores[best_idx]),
        "best_idx": best_idx,
        "status": "ok",
    }


# ============================================================
# Exact first/second derivative wrt sigma via FFT convolution
# ============================================================

def gaussian_kernel_and_first_derivative(sigma, shape):
    sigma = float(sigma)
    if sigma <= 0:
        raise ValueError("sigma must be > 0")

    H, W = shape
    cy, cx = H // 2, W // 2

    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    dy = yy - cy
    dx = xx - cx
    r2 = dx**2 + dy**2

    K = (1.0 / (2.0 * np.pi * sigma**2)) * np.exp(-r2 / (2.0 * sigma**2))
    dK = K * (-2.0 / sigma + r2 / sigma**3)

    return K, dK


def gaussian_kernel_and_derivatives(sigma, shape):
    sigma = float(sigma)
    if sigma <= 0:
        raise ValueError("sigma must be > 0 for Gaussian derivatives")

    H, W = shape
    cy, cx = H // 2, W // 2

    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    dy = yy - cy
    dx = xx - cx
    r2 = dx**2 + dy**2

    K = (1.0 / (2.0 * np.pi * sigma**2)) * np.exp(-r2 / (2.0 * sigma**2))
    dK = K * (-2.0 / sigma + r2 / sigma**3)
    d2K = K * (6.0 / sigma**2 - 6.0 * r2 / sigma**4 + r2**2 / sigma**6)

    return K, dK, d2K


def eval_score_grad_sigma(grid, gt_support, sigma, eps_h=EPS_LOG, normalize_by_gt=True):
    """Exact score and exact first derivative wrt sigma."""
    z = np.asarray(grid, dtype=np.float64)
    gt_idx = np.asarray(gt_support) > 0
    n_gt = int(gt_idx.sum())

    K, dK = gaussian_kernel_and_first_derivative(sigma, z.shape)

    h = fftconvolve(z, K, mode="same")
    dh = fftconvolve(z, dK, mode="same")

    h_safe = np.maximum(h, eps_h)

    H, W = z.shape
    score = np.log(h_safe[gt_idx]).sum() + n_gt * np.log(H * W) - h.sum()
    grad = (dh[gt_idx] / h_safe[gt_idx]).sum() - dh.sum()

    if normalize_by_gt and n_gt > 0:
        score /= n_gt
        grad /= n_gt

    return float(score), float(grad)


def eval_score_grad_hess_sigma(grid, gt_support, sigma, eps_h=EPS_LOG, normalize_by_gt=True):
    """Exact score, first derivative, and second derivative wrt sigma."""
    z = np.asarray(grid, dtype=np.float64)
    gt_idx = np.asarray(gt_support) > 0
    n_gt = int(gt_idx.sum())

    K, dK, d2K = gaussian_kernel_and_derivatives(sigma, z.shape)

    h = fftconvolve(z, K, mode="same")
    dh = fftconvolve(z, dK, mode="same")
    d2h = fftconvolve(z, d2K, mode="same")

    h_safe = np.maximum(h, eps_h)

    H, W = z.shape
    score = np.log(h_safe[gt_idx]).sum() + n_gt * np.log(H * W) - h.sum()
    grad = (dh[gt_idx] / h_safe[gt_idx]).sum() - dh.sum()
    hess = ((d2h[gt_idx] / h_safe[gt_idx]) - (dh[gt_idx] ** 2) / (h_safe[gt_idx] ** 2)).sum() - d2h.sum()

    if normalize_by_gt and n_gt > 0:
        score /= n_gt
        grad /= n_gt
        hess /= n_gt

    return float(score), float(grad), float(hess)


# ============================================================
# Interval helpers
# ============================================================

def get_selected_interval_from_best_idx(sigma_grid, best_idx):
    sigma_grid = np.asarray(sigma_grid, dtype=np.float64)
    n = len(sigma_grid)

    if n < 2:
        raise ValueError("Need at least 2 grid points")

    if best_idx <= 0:
        return None

    if best_idx >= n - 1:
        return (
            float(sigma_grid[n - 2]),
            float(sigma_grid[n - 1]),
            float(sigma_grid[n - 1]),
        )

    return (
        float(sigma_grid[best_idx - 1]),
        float(sigma_grid[best_idx]),
        float(sigma_grid[best_idx + 1]),
    )


def quadratic_peak(s1, s2, s3, f1, f2, f3):
    denom = (s1 - s2) * (s1 - s3) * (s2 - s3)
    if abs(denom) < 1e-12:
        return float(s2)

    A = (s3 * (f2 - f1) + s2 * (f1 - f3) + s1 * (f3 - f2)) / denom
    B = (s3**2 * (f1 - f2) + s2**2 * (f3 - f1) + s1**2 * (f2 - f3)) / denom

    if abs(A) < 1e-12:
        return float(s2)

    return float(-B / (2.0 * A))


def refine_sigma_with_first_derivative(
    grid,
    gt_support,
    sigma_left,
    sigma_right,
    sigma_init=None,
    loss_tol=1e-2,
    max_iter=50,
    eps_h=EPS_LOG,
    verbose=False,
    h_fd=0.25,
):
    if sigma_left < 0:
        raise ValueError("sigma_left must be >= 0")
    if sigma_right <= sigma_left:
        raise ValueError("Need sigma_right > sigma_left")
    if loss_tol < 0:
        raise ValueError("loss_tol must be >= 0")
    if h_fd <= 0:
        raise ValueError("h_fd must be > 0")

    def eval_score_scipy(sigma):
        sigma = float(max(0.0, sigma))
        x = prepare_grid(grid)

        if sigma == 0.0:
            return score_sigma_zero(
                x,
                gt_support,
                eps_h=eps_h,
                normalize_by_gt=True,
            )

        h = gaussianize_grid(x, sigma)
        return ppp_loglik_score(
            h,
            gt_support,
            eps=eps_h,
            normalize_by_gt=True,
        )

    def eval_score_grad_fd_2eval(sigma):
        sigma = float(max(0.0, sigma))

        # Near zero: forward difference
        if sigma <= h_fd:
            f0 = eval_score_scipy(sigma)
            fp = eval_score_scipy(sigma + h_fd)
            grad = (fp - f0) / h_fd
            return float(f0), float(grad)

        # Interior: central difference with 2 evaluations
        sm = sigma - h_fd
        sp = sigma + h_fd

        fm = eval_score_scipy(sm)
        fp = eval_score_scipy(sp)
        grad = (fp - fm) / (2.0 * h_fd)

        # midpoint proxy for bookkeeping / stopping
        score = 0.5 * (fm + fp)
        return float(score), float(grad)

    left = float(sigma_left)
    right = float(sigma_right)

    if sigma_init is None:
        sigma = 0.5 * (left + right)
    else:
        sigma = float(np.clip(sigma_init, left, right))

    sigma_hist = []
    score_hist = []
    grad_hist = []
    left_hist = []
    right_hist = []
    improvement_hist = []

    best_sigma = None
    best_score = -np.inf
    prev_score = None
    status = "max_iter_stop"

    for k in range(max_iter):
        st = time.perf_counter()

        score, grad = eval_score_grad_fd_2eval(sigma)

        elapsed = time.perf_counter() - st

        if prev_score is None:
            improvement = np.inf
        else:
            improvement = score - prev_score

        sigma_hist.append(float(sigma))
        score_hist.append(float(score))
        grad_hist.append(float(grad))
        left_hist.append(float(left))
        right_hist.append(float(right))
        improvement_hist.append(float(improvement))

        if score > best_score:
            best_score = float(score)
            best_sigma = float(sigma)

        if verbose:
            if np.isfinite(improvement):
                print(
                    f"iter={k:02d}  sigma={sigma:9.5f}  "
                    f"score={score:12.6f}  g={grad:12.6f}  "
                    f"impr={improvement:12.6f}  time={elapsed:.4f}s"
                )
            else:
                print(
                    f"iter={k:02d}  sigma={sigma:9.5f}  "
                    f"score={score:12.6f}  g={grad:12.6f}  "
                    f"impr=         inf  time={elapsed:.4f}s"
                )

        if prev_score is not None and improvement <= loss_tol:
            if verbose:
                print(f"STOP: improvement {improvement:.6f} <= loss_tol {loss_tol:.6f}")
            status = "loss_tol_stop"
            break

        if grad > 0:
            left = sigma
        elif grad < 0:
            right = sigma
        else:
            if verbose:
                print("STOP: gradient exactly zero")
            status = "grad_zero_stop"
            break

        sigma_new = 0.5 * (left + right)

        if abs(sigma_new - sigma) < 1e-12:
            if verbose:
                print("STOP: interval collapsed")
            status = "interval_collapsed_stop"
            break

        prev_score = score
        sigma = sigma_new

    if best_sigma is None:
        best_sigma = float(np.clip(sigma, left, right))

    # final exact scipy score at returned sigma
    best_score = float(eval_score_scipy(best_sigma))

    return {
        "best_sigma": float(best_sigma),
        "best_score": float(best_score),
        "sigma_history": np.asarray(sigma_hist, dtype=np.float64),
        "score_history": np.asarray(score_hist, dtype=np.float64),
        "grad_history": np.asarray(grad_hist, dtype=np.float64),
        "left_history": np.asarray(left_hist, dtype=np.float64),
        "right_history": np.asarray(right_hist, dtype=np.float64),
        "improvement_history": np.asarray(improvement_hist, dtype=np.float64),
        "sigma_left_init": float(sigma_left),
        "sigma_right_init": float(sigma_right),
        "sigma_init": float(np.clip(
            sigma_init if sigma_init is not None else 0.5 * (sigma_left + sigma_right),
            sigma_left,
            sigma_right
        )),
        "status": status,
        "loss_tol": float(loss_tol),
    }

# ============================================================
# Robust Newton-like optimizer with history
# ============================================================

def optimize_sigma_newton_robust_with_history(
    grid,
    gt_support,
    initial_checks=(0.0, 10.0, 50.0, 100.0),
    verbose=False,
    sigma_min=0.0,
    sigma_max=100.0,
    max_iter=20,
    tol_g=1e-6,
    tol_sigma=5e-1,
    min_step=5.0,
    max_step=20.0,
    eps_h=EPS_LOG,
):
    """
    Robust Newton optimization for sigma in [sigma_min, sigma_max].

    Features:
    - explicit check at sigma = 0
    - initial score checks at user-specified sigmas
    - Newton optimization in sigma-space
    - step clipping between min_step and max_step
    - backtracking if the candidate score gets worse
    - full history output
    """
    if sigma_max <= sigma_min:
        raise ValueError("Need sigma_max > sigma_min")
    if sigma_min < 0:
        raise ValueError("sigma_min must be >= 0")

    x = prepare_grid(grid)
    if x.sum() <= 0:
        return {
            "best_sigma": float(sigma_min),
            "best_score": -np.inf,
            "sigma_history": np.asarray([], dtype=np.float64),
            "score_history": np.asarray([], dtype=np.float64),
            "grad_history": np.asarray([], dtype=np.float64),
            "hess_history": np.asarray([], dtype=np.float64),
            "status": "empty_grid",
        }

    sigma_hist = []
    score_hist = []
    grad_hist = []
    hess_hist = []

    best_sigma = float(sigma_min)
    best_score = -np.inf

    checked = []
    for s in initial_checks:
        s = float(np.clip(s, sigma_min, sigma_max))
        if any(abs(s - t) < 1e-12 for t in checked):
            continue
        checked.append(s)

        st = time.perf_counter()

        if s == 0.0:
            score = score_sigma_zero(
                x,
                gt_support,
                eps_h=eps_h,
                normalize_by_gt=True,
            )
            grad = np.nan
            hess = np.nan
        else:
            score, grad, hess = eval_score_grad_hess_sigma(
                x,
                gt_support,
                s,
                eps_h=eps_h,
                normalize_by_gt=True,
            )

        elapsed = time.perf_counter() - st

        sigma_hist.append(s)
        score_hist.append(float(score))
        grad_hist.append(float(grad) if np.isfinite(grad) else np.nan)
        hess_hist.append(float(hess) if np.isfinite(hess) else np.nan)

        if verbose:
            print(f"initial sigma={s:8.3f}  score={score:12.6f}  time={elapsed:.4f}s")

        if score > best_score:
            best_score = float(score)
            best_sigma = float(s)

    sigma = max(best_sigma, 1e-6)
    status = "max_iter_stop"

    for k in range(max_iter):
        st = time.perf_counter()

        score, grad, hess = eval_score_grad_hess_sigma(
            x,
            gt_support,
            sigma,
            eps_h=eps_h,
            normalize_by_gt=True,
        )

        elapsed = time.perf_counter() - st

        sigma_hist.append(float(sigma))
        score_hist.append(float(score))
        grad_hist.append(float(grad))
        hess_hist.append(float(hess))

        if score > best_score:
            best_score = float(score)
            best_sigma = float(sigma)

        if verbose:
            print(
                f"iter={k:02d}  sigma={sigma:9.5f}  score={score:12.6f}  "
                f"g={grad:12.6f}  h={hess:12.6f}  time={elapsed:.4f}s"
            )

        if abs(grad) <= tol_g:
            status = "grad_tol_stop"
            break

        if not np.isfinite(hess) or abs(hess) < 1e-12:
            step = min_step * np.sign(grad)
        else:
            step = -grad / hess
            if abs(step) < min_step:
                step = min_step * np.sign(step if step != 0 else -grad)
            if abs(step) > max_step:
                step = max_step * np.sign(step)

        candidate = float(np.clip(sigma + step, max(sigma_min, 1e-6), sigma_max))

        if abs(candidate - sigma) <= tol_sigma:
            status = "sigma_tol_stop"
            break

        cand_score = eval_score_grad_hess_sigma(
            x,
            gt_support,
            candidate,
            eps_h=eps_h,
            normalize_by_gt=True,
        )[0]
        local_step = step

        while cand_score < score and abs(local_step) > tol_sigma:
            local_step *= 0.5
            candidate = float(np.clip(sigma + local_step, max(sigma_min, 1e-6), sigma_max))
            if abs(candidate - sigma) <= tol_sigma:
                break
            cand_score = eval_score_grad_hess_sigma(
                x,
                gt_support,
                candidate,
                eps_h=eps_h,
                normalize_by_gt=True,
            )[0]

        if abs(candidate - sigma) <= tol_sigma:
            status = "sigma_tol_stop"
            break

        sigma = candidate

    return {
        "best_sigma": float(best_sigma),
        "best_score": float(best_score),
        "sigma_history": np.asarray(sigma_hist, dtype=np.float64),
        "score_history": np.asarray(score_hist, dtype=np.float64),
        "grad_history": np.asarray(grad_hist, dtype=np.float64),
        "hess_history": np.asarray(hess_hist, dtype=np.float64),
        "status": status,
    }


# ============================================================
# Hybrid algorithm: coarse search + derivative refinement
# ============================================================

def optimize_sigma_hybrid_group_then_grad(
    grid,
    gt_support,
    sigma_max=120.0,
    n_coarse=21,
    loss_tol=1e-2,
    truncate=4.0,
    refine_max_iter=50,
    verbose=False,
):
    if sigma_max <= 0:
        raise ValueError("sigma_max must be > 0")
    if n_coarse < 3:
        raise ValueError("n_coarse must be >= 3")
    if loss_tol < 0:
        raise ValueError("loss_tol must be >= 0")

    sigma_grid = np.linspace(0.0, sigma_max, n_coarse)

    coarse_res = optimize_sigma_for_grid(
        grid=grid,
        gt_support=gt_support,
        sigma_grid=sigma_grid,
        verbose=verbose,
        truncate=truncate,
        eps_h=EPS_LOG,
    )

    best_idx = coarse_res["best_idx"]
    best_sigma_coarse = float(coarse_res["best_sigma"])
    best_score_coarse = float(coarse_res["best_score"])

    zero_boundary_selected = (best_idx == 0)
    selected_interval = get_selected_interval_from_best_idx(coarse_res["sigma_grid"], best_idx)

    best_sigma = best_sigma_coarse
    best_score = best_score_coarse
    refinement = None
    status = "zero_boundary_selected" if zero_boundary_selected else "coarse_only"

    if (not zero_boundary_selected) and (selected_interval is not None):
        s1, s2, s3 = selected_interval

        score_lookup = {
            float(s): float(sc)
            for s, sc in zip(coarse_res["sigma_grid"], coarse_res["scores"])
        }

        f1 = score_lookup[float(s1)]
        f2 = score_lookup[float(s2)]
        f3 = score_lookup[float(s3)]

        sigma_init = quadratic_peak(s1, s2, s3, f1, f2, f3)
        sigma_init = float(np.clip(sigma_init, max(s1, 1e-6), s3))

        if verbose:
            print("\nSelected intervals:")
            print(f"  [{s1:.6f}, {s3:.6f}] with center {s2:.6f}")
            print(
                f"\nRefinement 1: interval=[{s1:.6f}, {s3:.6f}], "
                f"quad_init={sigma_init:.6f}"
            )

        refinement = refine_sigma_with_first_derivative(
            grid=grid,
            gt_support=gt_support,
            sigma_left=s1,
            sigma_right=s3,
            sigma_init=sigma_init,
            loss_tol=loss_tol,
            max_iter=refine_max_iter,
            eps_h=EPS_LOG,
            verbose=verbose,
            h_fd=0.25,
        )

        if refinement["best_score"] > best_score:
            best_score = float(refinement["best_score"])
            best_sigma = float(refinement["best_sigma"])
            status = "refined_with_first_derivative"
        else:
            status = "coarse_better_than_refinement"

    return {
        "best_sigma": float(best_sigma),
        "best_score": float(best_score),
        "status": status,
        "zero_boundary_selected": bool(zero_boundary_selected),
        "loss_tol": float(loss_tol),
        "coarse": coarse_res,
        "selected_interval": selected_interval,
        "refinement": refinement,
    }


# ============================================================
# Ternary search in log-space
# ============================================================

def optimize_sigma_ternary_for_grid(
    grid,
    gt_support,
    sigma_min=0.0,
    sigma_max=100.0,
    max_iter=30,
    tol_sigma=0.5,
    eps_h=EPS_LOG,
    check_zero=True,
    verbose=False,
):
    if sigma_max <= sigma_min:
        raise ValueError("Need sigma_max > sigma_min")
    if sigma_min < 0:
        raise ValueError("sigma_min must be >= 0")

    x = prepare_grid(grid)
    if x.sum() <= 0:
        return {
            "best_sigma": float(sigma_min),
            "best_score": -np.inf,
            "sigma_history": np.asarray([], dtype=np.float64),
            "score_history": np.asarray([], dtype=np.float64),
            "status": "empty_grid",
        }

    eval_cache = {}

    def eval_sigma(s):
        s = float(max(0.0, s))
        key = round(s, 12)
        if key in eval_cache:
            return eval_cache[key]

        if s == 0.0:
            score = score_sigma_zero(
                x,
                gt_support,
                eps_h=eps_h,
                normalize_by_gt=True,
            )
        else:
            h = gaussianize_grid(x, s)
            score = ppp_loglik_score(
                h,
                gt_support,
                eps=eps_h,
                normalize_by_gt=True,
            )

        eval_cache[key] = float(score)
        return eval_cache[key]

    sigma_hist = []
    score_hist = []

    best_sigma = None
    best_score = -np.inf

    if check_zero and sigma_min == 0.0:
        score0 = eval_sigma(0.0)
        best_sigma = 0.0
        best_score = score0
        sigma_hist.append(0.0)
        score_hist.append(score0)

    left = math.log(max(sigma_min, 0.1))
    right = math.log(max(sigma_max, 0.1))
    for k in range(max_iter):
        if (right - left) < tol_sigma:
            break

        st = time.perf_counter()

        m1 = left + (right - left) / 3.0
        m2 = right - (right - left) / 3.0

        s1_sigma = math.exp(m1)
        s2_sigma = math.exp(m2)
        s1 = eval_sigma(s1_sigma)
        s2 = eval_sigma(s2_sigma)

        elapsed = time.perf_counter() - st

        sigma_hist.extend([s1_sigma, s2_sigma])
        score_hist.extend([s1, s2])

        if s1 > best_score:
            best_score = s1
            best_sigma = s1_sigma
        if s2 > best_score:
            best_score = s2
            best_sigma = s2_sigma

        if verbose:
            print(
                f"iter={k:02d}  left={math.exp(left):8.3f}  m1={s1_sigma:8.3f}  "
                f"m2={s2_sigma:8.3f}  right={math.exp(right):8.3f}  "
                f"s1={s1:12.6f}  s2={s2:12.6f}  time={elapsed:.4f}s"
            )

        if s1 < s2:
            left = m1
        else:
            right = m2

    sigma_mid = math.exp(0.5 * (left + right))
    score_mid = eval_sigma(sigma_mid)
    sigma_hist.append(sigma_mid)
    score_hist.append(score_mid)

    if score_mid > best_score:
        best_score = score_mid
        best_sigma = sigma_mid

    return {
        "best_sigma": float(best_sigma),
        "best_score": float(best_score),
        "sigma_history": np.asarray(sigma_hist, dtype=np.float64),
        "score_history": np.asarray(score_hist, dtype=np.float64),
        "status": "ok",
        "interval_final": (float(left), float(right)),
    }


# ============================================================
# Golden-section search in log-space
# ============================================================

def optimize_sigma_golden_for_grid(
    grid,
    gt_support,
    sigma_min=0.0,
    sigma_max=100.0,
    max_iter=30,
    tol_sigma=0.5,
    eps_h=EPS_LOG,
    check_zero=True,
    verbose=False,
):
    if sigma_max <= sigma_min:
        raise ValueError("Need sigma_max > sigma_min")
    if sigma_min < 0:
        raise ValueError("sigma_min must be >= 0")

    x = prepare_grid(grid)
    if x.sum() <= 0:
        return {
            "best_sigma": float(sigma_min),
            "best_score": -np.inf,
            "sigma_history": np.asarray([], dtype=np.float64),
            "score_history": np.asarray([], dtype=np.float64),
            "status": "empty_grid",
        }

    eval_cache = {}

    def eval_sigma(s):
        s = float(max(0.0, s))
        key = round(s, 12)
        if key in eval_cache:
            return eval_cache[key]

        if s == 0.0:
            score = score_sigma_zero(
                x,
                gt_support,
                eps_h=eps_h,
                normalize_by_gt=True,
            )
        else:
            h = gaussianize_grid(x, s)
            score = ppp_loglik_score(
                h,
                gt_support,
                eps=eps_h,
                normalize_by_gt=True,
            )

        eval_cache[key] = float(score)
        return eval_cache[key]

    sigma_hist = []
    score_hist = []

    best_sigma = None
    best_score = -np.inf

    if check_zero and sigma_min == 0.0:
        score0 = eval_sigma(0.0)
        best_sigma = 0.0
        best_score = score0
        sigma_hist.append(0.0)
        score_hist.append(score0)

    log_left = math.log(max(sigma_min, 0.1))
    log_right = math.log(max(sigma_max, 0.1))

    phi = (1.0 + math.sqrt(5.0)) / 2.0
    inv_phi = 1.0 / phi

    c = log_right - (log_right - log_left) * inv_phi
    d = log_left + (log_right - log_left) * inv_phi

    c_sigma = math.exp(c)
    d_sigma = math.exp(d)
    sc = eval_sigma(c_sigma)
    sd = eval_sigma(d_sigma)

    sigma_hist.extend([c_sigma, d_sigma])
    score_hist.extend([sc, sd])

    if sc > best_score:
        best_score = sc
        best_sigma = c_sigma
    if sd > best_score:
        best_score = sd
        best_sigma = d_sigma

    for k in range(max_iter):
        if (math.exp(log_right) - math.exp(log_left)) < tol_sigma:
            break

        st = time.perf_counter()

        if verbose:
            print(
                f"iter={k:02d}  left={math.exp(log_left):8.3f}  c={math.exp(c):8.3f}  "
                f"d={math.exp(d):8.3f}  right={math.exp(log_right):8.3f}  "
                f"sc={sc:12.6f}  sd={sd:12.6f}"
            )

        if sc < sd:
            log_left = c
            c = d
            sc = sd
            d = log_left + (log_right - log_left) * inv_phi
            d_sigma = math.exp(d)
            sd = eval_sigma(d_sigma)

            sigma_hist.append(d_sigma)
            score_hist.append(sd)

            if sd > best_score:
                best_score = sd
                best_sigma = d_sigma
        else:
            log_right = d
            d = c
            sd = sc
            c = log_right - (log_right - log_left) * inv_phi
            c_sigma = math.exp(c)
            sc = eval_sigma(c_sigma)

            sigma_hist.append(c_sigma)
            score_hist.append(sc)

            if sc > best_score:
                best_score = sc
                best_sigma = c_sigma

        elapsed = time.perf_counter() - st
        if verbose:
            print(f"          step_time={elapsed:.4f}s")

    sigma_mid = math.exp(0.5 * (log_left + log_right))
    score_mid = eval_sigma(sigma_mid)

    sigma_hist.append(sigma_mid)
    score_hist.append(score_mid)

    if score_mid > best_score:
        best_score = score_mid
        best_sigma = sigma_mid

    return {
        "best_sigma": float(best_sigma),
        "best_score": float(best_score),
        "sigma_history": np.asarray(sigma_hist, dtype=np.float64),
        "score_history": np.asarray(score_hist, dtype=np.float64),
        "status": "ok",
        "interval_final": (float(math.exp(log_left)), float(math.exp(log_right))),
    }


# ============================================================
# Unified dispatcher
# ============================================================

def optimize_sigma_dispatch(grid, gt_support, method="hybrid", **kwargs) -> dict[str, Any]:
    method = str(method).lower()

    if method == "grid_naive":
        sigma_grid = kwargs.pop("sigma_grid", np.linspace(0.0, 150.0, 60))
        return optimize_sigma_for_grid_naive(grid, gt_support, sigma_grid=sigma_grid, **kwargs)

    if method == "grid":
        sigma_grid = kwargs.pop("sigma_grid", np.linspace(0.0, 150.0, 60))
        return optimize_sigma_for_grid(grid, gt_support, sigma_grid=sigma_grid, **kwargs)

    if method in {"newton", "minimize_sigma"}:
        defaults = dict(
            initial_checks=(0.0, 1.0, 50.0, 100.0, 150.0),
            sigma_min=0.0,
            sigma_max=150.0,
            max_iter=20,
            tol_g=1e-6,
            tol_sigma=5e-1,
            min_step=2.0,
            max_step=20.0,
            eps_h=EPS_LOG,
        )
        defaults.update(kwargs)
        return optimize_sigma_newton_robust_with_history(grid, gt_support, **defaults)

    if method == "ternary":
        defaults = dict(
            sigma_min=0.0,
            sigma_max=150.0,
            max_iter=30,
            tol_sigma=0.01,
            eps_h=EPS_LOG,
        )
        defaults.update(kwargs)
        return optimize_sigma_ternary_for_grid(grid, gt_support, **defaults)

    if method == "golden":
        defaults = dict(
            sigma_min=0.0,
            sigma_max=150.0,
            max_iter=30,
            tol_sigma=0.5,
            eps_h=EPS_LOG,
        )
        defaults.update(kwargs)
        return optimize_sigma_golden_for_grid(grid, gt_support, **defaults)

    if method == "hybrid":
        defaults = dict(
            sigma_max=120.0,
            n_coarse=21,
            loss_tol=1e-2,
            truncate=4.0,
            refine_max_iter=50,
        )
        defaults.update(kwargs)
        return optimize_sigma_hybrid_group_then_grad(grid, gt_support, **defaults)

    raise ValueError(f"Unknown method: {method}")


__all__ = [
    "EPS_LOG",
    "prepare_grid",
    "ppp_loglik_score",
    "score_sigma_zero",
    "gaussianize_grid",
    "optimize_sigma_for_grid_naive",
    "optimize_sigma_for_grid",
    "gaussian_kernel_and_first_derivative",
    "gaussian_kernel_and_derivatives",
    "eval_score_grad_sigma",
    "eval_score_grad_hess_sigma",
    "get_selected_interval_from_best_idx",
    "quadratic_peak",
    "refine_sigma_with_first_derivative",
    "optimize_sigma_newton_robust_with_history",
    "optimize_sigma_hybrid_group_then_grad",
    "optimize_sigma_ternary_for_grid",
    "optimize_sigma_golden_for_grid",
    "optimize_sigma_dispatch",
]