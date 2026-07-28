"""Expected-count metrics for image-domain PPP predictions."""

from __future__ import annotations

from typing import Any

import numpy as np


def compute_expected_count_metrics(
    sample: dict[str, Any],
    config: dict[str, Any] | None = None,
) -> dict[str, float | int | str]:
    """Compute per-sample expected-count diagnostics.

    ``sample["pred_lambda"]`` must contain expected-count mass per image cell.
    ``sample["gt_support"]`` is interpreted as an occupied-cell mask.
    """
    cfg = {
        "negative_tolerance": 1e-8,
        "strict_binary_gt": True,
        "binary_tolerance": 1e-6,
    }
    if config is not None:
        cfg.update(config)

    pred_mass = np.asarray(sample["pred_lambda"], dtype=np.float64)
    gt_support = np.asarray(sample["gt_support"], dtype=np.float64)

    if pred_mass.ndim != 2:
        raise ValueError(f"pred_lambda must be 2D, got {pred_mass.shape}")
    if gt_support.shape != pred_mass.shape:
        raise ValueError(
            "gt_support and pred_lambda must have the same shape, "
            f"got {gt_support.shape} and {pred_mass.shape}"
        )
    if not np.all(np.isfinite(pred_mass)):
        raise ValueError("pred_lambda contains NaN or Inf")
    if not np.all(np.isfinite(gt_support)):
        raise ValueError("gt_support contains NaN or Inf")

    negative_tolerance = float(cfg["negative_tolerance"])
    min_mass = float(pred_mass.min(initial=0.0))
    if min_mass < -negative_tolerance:
        raise ValueError(
            "pred_lambda contains materially negative mass: "
            f"minimum={min_mass:.6g}"
        )
    pred_mass = np.clip(pred_mass, 0.0, None)

    if bool(cfg["strict_binary_gt"]):
        tol = float(cfg["binary_tolerance"])
        close_to_zero = np.isclose(gt_support, 0.0, atol=tol, rtol=0.0)
        close_to_one = np.isclose(gt_support, 1.0, atol=tol, rtol=0.0)
        if not np.all(close_to_zero | close_to_one):
            values = np.unique(gt_support[~(close_to_zero | close_to_one)])
            raise ValueError(
                "Camera-domain gt_support must be binary. "
                f"Unexpected values include {values[:10].tolist()}"
            )

    gt_mask = gt_support > 0.0
    expected_count = float(pred_mass.sum())
    gt_count = int(gt_mask.sum())
    count_error = expected_count - float(gt_count)

    return {
        "expected_count": expected_count,
        "gt_count": gt_count,
        "count_error": count_error,
        "count_abs_error": abs(count_error),
        "count_squared_error": count_error * count_error,
        "count_ratio": (
            expected_count / float(gt_count) if gt_count > 0 else np.nan
        ),
        "count_status": "ok",
    }
