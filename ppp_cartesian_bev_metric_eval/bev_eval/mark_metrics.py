from __future__ import annotations

from typing import Any

import numpy as np


DEFAULT_CONFIG: dict[str, Any] = {
    "velocity_mode": "signed",
    "use_circular_velocity": True,
    "require_doppler_period": True,
    "velocity_gt_abs_min": 0.5,
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

    n_points = int(error.size)

    if n_points == 0:
        return {
            f"{prefix}_n_points": 0,
            f"{prefix}_abs_error_sum": 0.0,
            f"{prefix}_squared_error_sum": 0.0,
            f"{prefix}_signed_error_sum": 0.0,
            f"{prefix}_mae": np.nan,
            f"{prefix}_rmse": np.nan,
            f"{prefix}_bias": np.nan,
        }

    abs_error_sum = float(np.abs(error).sum())
    squared_error_sum = float(np.square(error).sum())
    signed_error_sum = float(error.sum())

    return {
        f"{prefix}_n_points": n_points,
        f"{prefix}_abs_error_sum": abs_error_sum,
        f"{prefix}_squared_error_sum": squared_error_sum,
        f"{prefix}_signed_error_sum": signed_error_sum,
        f"{prefix}_mae": abs_error_sum / n_points,
        f"{prefix}_rmse": np.sqrt(squared_error_sum / n_points),
        f"{prefix}_bias": signed_error_sum / n_points,
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
    """Compute pointwise velocity and RCS metrics.

    Velocity metrics are computed for:

    1. All valid GT points.
    2. The subset with
       abs(gt_velocity) > velocity_gt_abs_min.

    The non-zero subset uses the same velocity-error definition as the
    all-point evaluation, including circular differences when enabled.
    """
    cfg = dict(DEFAULT_CONFIG)
    if config:
        cfg.update(config)

    velocity_mode = str(cfg["velocity_mode"]).lower()
    if velocity_mode not in {"signed", "absolute"}:
        raise ValueError(
            "velocity_mode must be 'signed' or 'absolute'."
        )

    velocity_gt_abs_min = float(
        cfg.get("velocity_gt_abs_min", 0.5)
    )
    if velocity_gt_abs_min < 0.0:
        raise ValueError(
            "velocity_gt_abs_min must be non-negative."
        )

    pred_v, gt_v, pred_rcs, gt_rcs, source = (
        _extract_pointwise_marks(sample)
    )

    pred_v = np.asarray(pred_v, dtype=np.float64)
    gt_v = np.asarray(gt_v, dtype=np.float64)
    pred_rcs = np.asarray(pred_rcs, dtype=np.float64)
    gt_rcs = np.asarray(gt_rcs, dtype=np.float64)

    finite_v = np.isfinite(pred_v) & np.isfinite(gt_v)
    finite_rcs = np.isfinite(pred_rcs) & np.isfinite(gt_rcs)

    pred_v_valid = pred_v[finite_v]
    gt_v_valid = gt_v[finite_v]

    if velocity_mode == "absolute":
        velocity_error = (
            np.abs(pred_v_valid) - np.abs(gt_v_valid)
        )
        velocity_error_mode = "ordinary_absolute"

    elif bool(cfg["use_circular_velocity"]):
        period = sample["attrs"].get("doppler_period")

        if period is None:
            if bool(cfg["require_doppler_period"]):
                raise KeyError(
                    "Circular velocity metrics require the sample "
                    "attribute 'doppler_period'."
                )

            velocity_error = pred_v_valid - gt_v_valid
            velocity_error_mode = (
                "ordinary_signed_missing_period"
            )

        else:
            period = float(period)

            if not np.isfinite(period) or period <= 0.0:
                raise ValueError(
                    "Sample attribute 'doppler_period' must be a "
                    "finite positive number."
                )

            velocity_error = circular_difference(
                pred_v_valid,
                gt_v_valid,
                period,
            )
            velocity_error_mode = "circular_signed"

    else:
        velocity_error = pred_v_valid - gt_v_valid
        velocity_error_mode = "ordinary_signed"

    # Select GT points whose velocity magnitude is clearly non-zero.
    # This mask is applied after finite-value filtering, so it has the
    # same length and ordering as velocity_error.
    velocity_nonzero_mask = (
        np.abs(gt_v_valid) > velocity_gt_abs_min
    )
    velocity_nonzero_error = velocity_error[
        velocity_nonzero_mask
    ]

    pred_rcs_valid = pred_rcs[finite_rcs]
    gt_rcs_valid = gt_rcs[finite_rcs]
    rcs_error = pred_rcs_valid - gt_rcs_valid

    n_velocity_valid = int(gt_v_valid.size)
    n_velocity_nonzero = int(
        np.count_nonzero(velocity_nonzero_mask)
    )

    if n_velocity_valid > 0:
        velocity_nonzero_fraction = (
            n_velocity_nonzero / n_velocity_valid
        )
    else:
        velocity_nonzero_fraction = np.nan

    result: dict[str, Any] = {
        "mark_evaluation_source": source,
        "velocity_error_mode": velocity_error_mode,
        "velocity_gt_abs_min": velocity_gt_abs_min,
        "velocity_nonzero_fraction": (
            velocity_nonzero_fraction
        ),
    }

    result.update(
        _error_statistics(
            velocity_error,
            prefix="velocity",
        )
    )
    result.update(
        _error_statistics(
            velocity_nonzero_error,
            prefix="velocity_nonzero",
        )
    )
    result.update(
        _error_statistics(
            rcs_error,
            prefix="rcs",
        )
    )

    return result