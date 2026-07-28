from __future__ import annotations

from typing import Any

import numpy as np


def compute_count_metrics(
    sample: dict[str, Any],
    config: dict[str, Any] | None = None,
) -> dict[str, float]:
    del config
    pred_mass = np.asarray(sample["pred_lambda"], dtype=np.float64)
    gt_support = np.asarray(sample["gt_support"], dtype=np.float64)

    expected_count = float(pred_mass.sum())
    gt_count = float(gt_support.sum())
    error = expected_count - gt_count

    return {
        "expected_count": expected_count,
        "gt_count": gt_count,
        "count_error": error,
        "count_abs_error": abs(error),
        "count_squared_error": error * error,
        "count_ratio": expected_count / gt_count if gt_count > 0.0 else np.nan,
    }
