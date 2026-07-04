"""
Expected-count metric tools for offline PPP metric evaluation.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def compute_expected_count_metrics(
    sample: dict[str, Any],
    config: dict[str, Any] | None = None,
) -> dict[str, float]:
    """
    Compute expected-count diagnostics for one sample.

    Expected sample fields:
        sample["pred_lambda"]
        sample["gt_support"]
    """
    if "pred_lambda" not in sample:
        raise KeyError("sample is missing 'pred_lambda'")

    if "gt_support" not in sample:
        raise KeyError("sample is missing 'gt_support'")

    pred_lambda = np.asarray(sample["pred_lambda"], dtype=np.float64)
    gt_support = np.asarray(sample["gt_support"], dtype=np.float64)

    if pred_lambda.ndim != 2:
        raise ValueError(f"pred_lambda must be 2D, got shape {pred_lambda.shape}")

    if gt_support.shape != pred_lambda.shape:
        raise ValueError(
            "pred_lambda and gt_support must have the same shape, "
            f"got {pred_lambda.shape} and {gt_support.shape}"
        )

    expected_count = float(pred_lambda.sum())
    gt_count = float(gt_support.sum())

    count_error = expected_count - gt_count
    count_abs_error = abs(count_error)
    count_squared_error = count_error ** 2
    count_ratio = expected_count / gt_count if gt_count > 0 else np.nan

    return {
        "expected_count": expected_count,
        "gt_count": gt_count,
        "count_error": count_error,
        "count_abs_error": count_abs_error,
        "count_squared_error": count_squared_error,
        "count_ratio": count_ratio,
    }