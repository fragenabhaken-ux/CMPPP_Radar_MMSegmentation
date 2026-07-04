"""
MMD metric adapter for offline PPP metric evaluation.

This file is the bridge between:

    loaded HDF5 sample
        and
    ppp_metrics_loss_mmd_tools.compute_optimal_sigma_metric

It does not implement the mathematical MMD core itself.
The core stays in ppp_metrics_loss_mmd_tools.py.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ppp_metrics_loss_mmd_tools import compute_optimal_sigma_metric


DEFAULT_MMD_CONFIG: dict[str, Any] = {
    "enabled": True,
    "score_type": "mmd",
    "sigma_min": 0.0,
    "sigma_max": 120.0,
    "max_iter": 30,
    "tol_sigma": 0.5,
    "normalize_by_gt": True,
    "eps": 1e-12,
    "check_zero": True,
}


def _clean_mmd_config(config: dict[str, Any] | None) -> dict[str, Any]:
    """
    Prepare config for compute_optimal_sigma_metric.

    Removes keys that belong to the framework but are not accepted by
    compute_optimal_sigma_metric.
    """
    cfg = dict(DEFAULT_MMD_CONFIG)

    if config is not None:
        cfg.update(config)

    cfg.pop("enabled", None)

    return cfg


def compute_mmd_metrics(
    sample: dict[str, Any],
    config: dict[str, Any] | None = None,
    *,
    prefix: str = "mmd",
) -> dict[str, float | str]:
    """
    Compute MMD-style optimal-sigma metric for one loaded H5 sample.

    Expected sample fields:
        sample["pred_lambda"]
        sample["gt_support"]

    Important:
        pred_lambda is assumed to be mass per pixel already.
        Therefore we do not divide by H*W here.
    """
    if "pred_lambda" not in sample:
        raise KeyError("sample is missing 'pred_lambda'")

    if "gt_support" not in sample:
        raise KeyError("sample is missing 'gt_support'")

    pred_lambda = np.asarray(sample["pred_lambda"], dtype=np.float64)
    gt_support = np.asarray(sample["gt_support"])

    if pred_lambda.ndim != 2:
        raise ValueError(f"pred_lambda must be 2D, got shape {pred_lambda.shape}")

    if gt_support.shape != pred_lambda.shape:
        raise ValueError(
            "pred_lambda and gt_support must have the same shape, "
            f"got {pred_lambda.shape} and {gt_support.shape}"
        )

    cfg = _clean_mmd_config(config)

    score_type = cfg.pop("score_type", "mmd")

    result = compute_optimal_sigma_metric(
        pred_grid=pred_lambda,
        gt_support=gt_support,
        score_type=score_type,
        **cfg,
    )

    return {
        f"{prefix}_optimal_score": float(result["best_score"]),
        f"{prefix}_score_type": str(result["score_type"]),
        f"{prefix}_status": str(result["status"]),
    }