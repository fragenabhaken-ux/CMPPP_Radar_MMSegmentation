"""Depth, radial-velocity, and RCS metrics at image-space GT support cells."""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np


def _empty(prefix: str, status: str, include_abs_rel: bool = False) -> dict[str, Any]:
    result: dict[str, Any] = {
        f"{prefix}_mae": np.nan,
        f"{prefix}_rmse": np.nan,
        f"{prefix}_bias": np.nan,
        f"{prefix}_n_points": 0,
        f"{prefix}_abs_error_sum": 0.0,
        f"{prefix}_squared_error_sum": 0.0,
        f"{prefix}_error_sum": 0.0,
        f"{prefix}_status": status,
    }
    if include_abs_rel:
        result[f"{prefix}_abs_rel_error"] = np.nan
        result[f"{prefix}_abs_rel_error_sum"] = 0.0
    return result


def _validate_same_shape(name: str, array: np.ndarray, shape: tuple[int, ...]) -> None:
    if array.shape != shape:
        raise ValueError(f"{name} has shape {array.shape}, expected {shape}")


def _metric_from_error(
    error: np.ndarray,
    mask: np.ndarray,
    *,
    prefix: str,
    status: str = "ok",
    abs_rel_denominator: np.ndarray | None = None,
    eps: float = 1e-8,
) -> dict[str, Any]:
    valid = mask & np.isfinite(error)
    if abs_rel_denominator is not None:
        valid &= np.isfinite(abs_rel_denominator)

    n_points = int(valid.sum())
    if n_points == 0:
        return _empty(prefix, "no_valid_points", abs_rel_denominator is not None)

    err = error[valid].astype(np.float64, copy=False)
    abs_err = np.abs(err)
    sq_err = err * err

    result: dict[str, Any] = {
        f"{prefix}_mae": float(abs_err.mean()),
        f"{prefix}_rmse": float(np.sqrt(sq_err.mean())),
        f"{prefix}_bias": float(err.mean()),
        f"{prefix}_n_points": n_points,
        f"{prefix}_abs_error_sum": float(abs_err.sum()),
        f"{prefix}_squared_error_sum": float(sq_err.sum()),
        f"{prefix}_error_sum": float(err.sum()),
        f"{prefix}_status": status,
    }

    if abs_rel_denominator is not None:
        denom = np.maximum(np.abs(abs_rel_denominator[valid]), float(eps))
        abs_rel = abs_err / denom
        result[f"{prefix}_abs_rel_error"] = float(abs_rel.mean())
        result[f"{prefix}_abs_rel_error_sum"] = float(abs_rel.sum())

    return result


def compute_depth_metrics(
    sample: dict[str, Any],
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compute depth errors at occupied GT image cells.

    The current export contains one depth prediction. For compatibility with
    earlier experiments, near/far depth hypotheses are also supported and the
    closer hypothesis is selected independently at each GT point.
    """
    cfg = {"min_gt_depth": 0.0, "eps": 1e-8}
    if config is not None:
        cfg.update(config)

    gt_depth_raw = sample.get("gt_depth")
    if gt_depth_raw is None:
        return _empty("depth", "missing_gt_depth", include_abs_rel=True)

    gt_support = np.asarray(sample["gt_support"])
    gt_depth = np.asarray(gt_depth_raw, dtype=np.float64)
    _validate_same_shape("gt_depth", gt_depth, gt_support.shape)

    near_raw = sample.get("pred_depth_near")
    far_raw = sample.get("pred_depth_far")
    if near_raw is not None and far_raw is not None:
        near = np.asarray(near_raw, dtype=np.float64)
        far = np.asarray(far_raw, dtype=np.float64)
        _validate_same_shape("pred_depth_near", near, gt_support.shape)
        _validate_same_shape("pred_depth_far", far, gt_support.shape)
        error_near = near - gt_depth
        error_far = far - gt_depth
        choose_far = np.abs(error_far) < np.abs(error_near)
        error = np.where(choose_far, error_far, error_near)
        finite_prediction = np.isfinite(near) & np.isfinite(far)
        mode = "two_depth"
    else:
        pred_raw = sample.get("pred_depth")
        if pred_raw is None:
            return _empty("depth", "missing_pred_depth", include_abs_rel=True)
        pred = np.asarray(pred_raw, dtype=np.float64)
        _validate_same_shape("pred_depth", pred, gt_support.shape)
        error = pred - gt_depth
        finite_prediction = np.isfinite(pred)
        choose_far = None
        mode = "single"

    mask = (gt_support > 0.0) & finite_prediction & np.isfinite(gt_depth)
    min_gt_depth = cfg.get("min_gt_depth")
    if min_gt_depth is not None:
        mask &= gt_depth > float(min_gt_depth)

    result = _metric_from_error(
        error,
        mask,
        prefix="depth",
        status=mode,
        abs_rel_denominator=gt_depth,
        eps=float(cfg["eps"]),
    )
    result["depth_mode"] = mode
    result["depth_far_choice_ratio"] = (
        float(choose_far[mask].mean())
        if choose_far is not None and int(mask.sum()) > 0
        else np.nan
    )
    return result


def _first_valid_positive_attr(
    attrs: dict[str, Any],
    names: Iterable[str],
) -> tuple[float | None, str | None]:
    for name in names:
        if name not in attrs:
            continue
        try:
            value = float(np.asarray(attrs[name]).reshape(-1)[0])
        except (TypeError, ValueError, IndexError):
            continue
        if np.isfinite(value) and value > 0.0:
            return value, name
    return None, None


def circular_error(pred: np.ndarray, target: np.ndarray, period: float) -> np.ndarray:
    """Signed shortest-path error on a circle with the given full period."""
    period = float(period)
    if not np.isfinite(period) or period <= 0.0:
        raise ValueError(f"period must be positive and finite, got {period}")
    delta = np.asarray(pred, dtype=np.float64) - np.asarray(target, dtype=np.float64)
    return (delta + 0.5 * period) % period - 0.5 * period


def compute_velocity_metrics(
    sample: dict[str, Any],
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compute radial-velocity errors at occupied GT image cells."""
    cfg = {
        "mode": "auto",
        "circular": True,
        "fixed_period": None,
        "period_attr_candidates": (
            "doppler_period",
            "dopplerRange",
            "doppler_range",
        ),
        "require_period": True,
    }
    if config is not None:
        cfg.update(config)

    pred_raw = sample.get("pred_velocity")
    gt_raw = sample.get("gt_velocity")
    if pred_raw is None:
        return _empty("velocity", "missing_pred_velocity")
    if gt_raw is None:
        return _empty("velocity", "missing_gt_velocity")

    gt_support = np.asarray(sample["gt_support"])
    pred = np.asarray(pred_raw, dtype=np.float64)
    gt = np.asarray(gt_raw, dtype=np.float64)
    _validate_same_shape("pred_velocity", pred, gt_support.shape)
    _validate_same_shape("gt_velocity", gt, gt_support.shape)

    attrs = dict(sample.get("attrs", {}))
    mode = str(cfg.get("mode", "auto")).lower()
    if mode == "auto":
        mode = str(attrs.get("velocity_mode", "signed")).lower()
    if mode not in {"signed", "absolute"}:
        raise ValueError(f"Unsupported velocity mode: {mode!r}")

    mask = (gt_support > 0.0) & np.isfinite(pred) & np.isfinite(gt)
    status = mode
    period = np.nan
    period_source = "none"

    if mode == "absolute":
        error = np.abs(pred) - np.abs(gt)
    elif bool(cfg.get("circular", True)):
        fixed_period = cfg.get("fixed_period")
        if fixed_period is not None:
            period = float(fixed_period)
            period_source = "fixed_period"
        else:
            resolved, source = _first_valid_positive_attr(
                attrs,
                cfg["period_attr_candidates"],
            )
            if resolved is not None:
                period = float(resolved)
                period_source = str(source)

        if not np.isfinite(period) or period <= 0.0:
            if bool(cfg.get("require_period", True)):
                result = _empty("velocity", "missing_doppler_period")
                result["velocity_mode"] = mode
                result["velocity_period"] = np.nan
                result["velocity_period_source"] = "missing"
                return result
            error = pred - gt
            status = "signed_unwrapped_missing_period"
        else:
            error = circular_error(pred, gt, period)
            status = "signed_circular"
    else:
        error = pred - gt
        status = "signed_linear"

    result = _metric_from_error(error, mask, prefix="velocity", status=status)
    result["velocity_mode"] = mode
    result["velocity_period"] = float(period) if np.isfinite(period) else np.nan
    result["velocity_period_source"] = period_source
    return result


def compute_rcs_metrics(
    sample: dict[str, Any],
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compute RCS errors at occupied GT image cells."""
    del config
    pred_raw = sample.get("pred_rcs")
    gt_raw = sample.get("gt_rcs")
    if pred_raw is None:
        return _empty("rcs", "missing_pred_rcs")
    if gt_raw is None:
        return _empty("rcs", "missing_gt_rcs")

    gt_support = np.asarray(sample["gt_support"])
    pred = np.asarray(pred_raw, dtype=np.float64)
    gt = np.asarray(gt_raw, dtype=np.float64)
    _validate_same_shape("pred_rcs", pred, gt_support.shape)
    _validate_same_shape("gt_rcs", gt, gt_support.shape)

    mask = (gt_support > 0.0) & np.isfinite(pred) & np.isfinite(gt)
    return _metric_from_error(pred - gt, mask, prefix="rcs", status="ok")
