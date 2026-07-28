from __future__ import annotations

from typing import Any

import numpy as np


DEFAULT_CONFIG: dict[str, Any] = {
    "velocity_mode": "signed",
    "use_circular_velocity": True,
    "require_doppler_period": True,
}


def circular_difference(
    prediction: np.ndarray,
    target: np.ndarray,
    period: float,
) -> np.ndarray:
    period = float(period)
    if not np.isfinite(period) or period <= 0.0:
        raise ValueError(f"Doppler period must be finite and positive, got {period}.")
    return np.remainder(prediction - target + 0.5 * period, period) - 0.5 * period


def _error_statistics(
    error: np.ndarray,
    *,
    prefix: str,
) -> dict[str, float | int]:
    error = np.asarray(error, dtype=np.float64)
    error = error[np.isfinite(error)]
    n = int(error.size)
    if n == 0:
        return {
            f"{prefix}_mae": np.nan,
            f"{prefix}_rmse": np.nan,
            f"{prefix}_bias": np.nan,
            f"{prefix}_n_points": 0,
            f"{prefix}_abs_error_sum": 0.0,
            f"{prefix}_squared_error_sum": 0.0,
            f"{prefix}_signed_error_sum": 0.0,
        }

    abs_error = np.abs(error)
    squared = error * error
    return {
        f"{prefix}_mae": float(abs_error.mean()),
        f"{prefix}_rmse": float(np.sqrt(squared.mean())),
        f"{prefix}_bias": float(error.mean()),
        f"{prefix}_n_points": n,
        f"{prefix}_abs_error_sum": float(abs_error.sum()),
        f"{prefix}_squared_error_sum": float(squared.sum()),
        f"{prefix}_signed_error_sum": float(error.sum()),
    }


def _extract_pointwise_marks(
    sample: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
    """Return predicted/GT velocity and RCS at individual GT points."""
    geometry = sample["geometry"]
    xy = sample.get("gt_points_xy")
    gt_point_velocity = sample.get("gt_point_velocity")
    gt_point_rcs = sample.get("gt_point_rcs")

    if xy is not None and gt_point_velocity is not None and gt_point_rcs is not None:
        xy = np.asarray(xy, dtype=np.float64)
        gt_velocity = np.asarray(gt_point_velocity, dtype=np.float64).reshape(-1)
        gt_rcs = np.asarray(gt_point_rcs, dtype=np.float64).reshape(-1)
        rows, cols, inside = geometry.point_indices(xy)
        finite = inside & np.isfinite(gt_velocity) & np.isfinite(gt_rcs)
        rows = rows[finite]
        cols = cols[finite]
        return (
            np.asarray(sample["pred_velocity"], dtype=np.float64)[rows, cols],
            gt_velocity[finite],
            np.asarray(sample["pred_rcs"], dtype=np.float64)[rows, cols],
            gt_rcs[finite],
            "exact_gt_points",
        )

    # Fallback for older exports: compare at occupied cells and repeat each
    # cell according to its rounded support count.
    support = np.asarray(sample["gt_support"], dtype=np.float64)
    rounded = np.rint(support).astype(np.int64)
    if np.max(np.abs(support - rounded), initial=0.0) > 1e-5:
        raise ValueError(
            "gt_support is not integer-valued and exact GT point arrays are missing."
        )
    rows, cols = np.nonzero(rounded > 0)
    repeats = rounded[rows, cols]
    rows = np.repeat(rows, repeats)
    cols = np.repeat(cols, repeats)
    return (
        np.asarray(sample["pred_velocity"], dtype=np.float64)[rows, cols],
        np.asarray(sample["gt_velocity"], dtype=np.float64)[rows, cols],
        np.asarray(sample["pred_rcs"], dtype=np.float64)[rows, cols],
        np.asarray(sample["gt_rcs"], dtype=np.float64)[rows, cols],
        "gt_count_grid_fallback",
    )


def compute_mark_metrics(
    sample: dict[str, Any],
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    cfg = dict(DEFAULT_CONFIG)
    if config:
        cfg.update(config)

    velocity_mode = str(cfg["velocity_mode"]).lower()
    if velocity_mode not in {"signed", "absolute"}:
        raise ValueError("velocity_mode must be 'signed' or 'absolute'.")

    pred_v, gt_v, pred_rcs, gt_rcs, source = _extract_pointwise_marks(sample)
    finite_v = np.isfinite(pred_v) & np.isfinite(gt_v)
    finite_rcs = np.isfinite(pred_rcs) & np.isfinite(gt_rcs)

    if velocity_mode == "absolute":
        velocity_error = np.abs(pred_v[finite_v]) - np.abs(gt_v[finite_v])
        velocity_error_mode = "ordinary_absolute"
    elif bool(cfg["use_circular_velocity"]):
        period = sample["attrs"].get("doppler_period")
        if period is None:
            if bool(cfg["require_doppler_period"]):
                raise KeyError(
                    "Circular velocity metrics require the sample attribute "
                    "'doppler_period'."
                )
            velocity_error = pred_v[finite_v] - gt_v[finite_v]
            velocity_error_mode = "ordinary_signed_missing_period"
        else:
            velocity_error = circular_difference(pred_v[finite_v], gt_v[finite_v], float(period))
            velocity_error_mode = "circular_signed"
    else:
        velocity_error = pred_v[finite_v] - gt_v[finite_v]
        velocity_error_mode = "ordinary_signed"

    rcs_error = pred_rcs[finite_rcs] - gt_rcs[finite_rcs]

    result: dict[str, Any] = {
        "mark_evaluation_source": source,
        "velocity_error_mode": velocity_error_mode,
    }
    result.update(_error_statistics(velocity_error, prefix="velocity"))
    result.update(_error_statistics(rcs_error, prefix="rcs"))
    return result
