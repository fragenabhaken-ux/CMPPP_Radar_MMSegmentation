"""
Depth metric tools for offline PPP metric evaluation.

This metric evaluates predicted depth only at GT support locations.

Supported formats
-----------------
Single-depth export:
    sample["pred_depth"]

Two-depth export:
    sample["pred_depth_near"]
    sample["pred_depth_far"]

For the two-depth case, the metric uses:

    min(|d_near - gt_depth|, |d_far - gt_depth|)

at GT support locations.

The code is backward-compatible with old metric exports.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def _empty_result(depth_mode: str = "missing") -> dict[str, float | str]:
    return {
        "depth_mae": np.nan,
        "depth_rmse": np.nan,
        "depth_bias": np.nan,
        "depth_abs_rel_error": np.nan,
        "depth_n_points": 0,
        "depth_mode": depth_mode,
        "depth_two_depth_used": 0,
        "depth_far_choice_ratio": np.nan,
    }


def compute_depth_metrics(
    sample: dict[str, Any],
    config: dict[str, Any] | None = None,
) -> dict[str, float | str]:
    """
    Compute depth metrics at GT point locations.

    For single-depth exports:
        err = pred_depth - gt_depth

    For two-depth exports:
        err_near = pred_depth_near - gt_depth
        err_far  = pred_depth_far  - gt_depth

        The selected error is the signed error of the closer hypothesis:

            err = err_near if |err_near| <= |err_far|
                  err_far  otherwise

        Therefore:

            abs_err = min(|err_near|, |err_far|)

    Metrics:
        depth_mae
        depth_rmse
        depth_bias
        depth_abs_rel_error
        depth_n_points

    Additional diagnostics:
        depth_mode:
            "single" or "two_depth"
        depth_two_depth_used:
            0 or 1
        depth_far_choice_ratio:
            fraction of GT support points where far-depth was closer
    """
    cfg = {
        "min_gt_depth": 0.0,
        "eps": 1e-8,
    }
    if config is not None:
        cfg.update(config)

    if sample.get("gt_depth", None) is None:
        return _empty_result("missing_gt_depth")

    if "gt_support" not in sample:
        raise KeyError("sample is missing 'gt_support'")

    gt_depth = np.asarray(sample["gt_depth"], dtype=np.float64)
    gt_support = np.asarray(sample["gt_support"])

    if gt_support.shape != gt_depth.shape:
        raise ValueError(
            "gt_support and gt_depth must have the same shape, "
            f"got {gt_support.shape} and {gt_depth.shape}"
        )

    # ------------------------------------------------------------------
    # Detect prediction format
    # ------------------------------------------------------------------
    has_two_depth = (
        sample.get("pred_depth_near", None) is not None
        and sample.get("pred_depth_far", None) is not None
    )

    if has_two_depth:
        pred_depth_near = np.asarray(sample["pred_depth_near"], dtype=np.float64)
        pred_depth_far = np.asarray(sample["pred_depth_far"], dtype=np.float64)

        if pred_depth_near.shape != gt_depth.shape:
            raise ValueError(
                "pred_depth_near and gt_depth must have the same shape, "
                f"got {pred_depth_near.shape} and {gt_depth.shape}"
            )

        if pred_depth_far.shape != gt_depth.shape:
            raise ValueError(
                "pred_depth_far and gt_depth must have the same shape, "
                f"got {pred_depth_far.shape} and {gt_depth.shape}"
            )

        err_near = pred_depth_near - gt_depth
        err_far = pred_depth_far - gt_depth

        abs_err_near = np.abs(err_near)
        abs_err_far = np.abs(err_far)

        choose_far = abs_err_far < abs_err_near

        # Signed error of whichever hypothesis is closer.
        err = np.where(choose_far, err_far, err_near)
        abs_err = np.minimum(abs_err_near, abs_err_far)

        finite_pred_mask = (
            np.isfinite(pred_depth_near)
            & np.isfinite(pred_depth_far)
        )

        depth_mode = "two_depth"
        two_depth_used = 1

    else:
        if sample.get("pred_depth", None) is None:
            return _empty_result("missing_pred_depth")

        pred_depth = np.asarray(sample["pred_depth"], dtype=np.float64)

        if pred_depth.shape != gt_depth.shape:
            raise ValueError(
                "pred_depth and gt_depth must have the same shape, "
                f"got {pred_depth.shape} and {gt_depth.shape}"
            )

        err = pred_depth - gt_depth
        abs_err = np.abs(err)

        choose_far = None
        finite_pred_mask = np.isfinite(pred_depth)

        depth_mode = "single"
        two_depth_used = 0

    # ------------------------------------------------------------------
    # Masking
    # ------------------------------------------------------------------
    mask = gt_support > 0

    min_gt_depth = cfg.get("min_gt_depth", 0.0)
    if min_gt_depth is not None:
        mask = mask & (gt_depth > float(min_gt_depth))

    mask = mask & finite_pred_mask & np.isfinite(gt_depth)

    n_points = int(mask.sum())

    if n_points == 0:
        return _empty_result(depth_mode)

    err_m = err[mask]
    abs_err_m = abs_err[mask]

    eps = float(cfg["eps"])
    abs_rel = abs_err_m / np.maximum(np.abs(gt_depth[mask]), eps)

    result: dict[str, float | str] = {
        "depth_mae": float(abs_err_m.mean()),
        "depth_rmse": float(np.sqrt(np.mean(err_m ** 2))),
        "depth_bias": float(err_m.mean()),
        "depth_abs_rel_error": float(abs_rel.mean()),
        "depth_n_points": n_points,
        "depth_mode": depth_mode,
        "depth_two_depth_used": two_depth_used,
        "depth_far_choice_ratio": np.nan,
    }

    if has_two_depth:
        result["depth_far_choice_ratio"] = float(choose_far[mask].mean())

    return result



# """
# Depth metric tools for offline PPP metric evaluation.

# This metric evaluates predicted depth only at GT support locations.

# Required sample fields:
#     sample["pred_depth"]
#     sample["gt_depth"]
#     sample["gt_support"]
# """

# from __future__ import annotations

# from typing import Any

# import numpy as np


# def compute_depth_metrics(
#     sample: dict[str, Any],
#     config: dict[str, Any] | None = None,
# ) -> dict[str, float]:
#     """
#     Compute simple depth metrics at GT point locations.

#     Metrics:
#         depth_mae
#         depth_rmse
#         depth_bias
#         depth_abs_rel_error

#     The mask is:
#         gt_support > 0

#     Optional config:
#         min_gt_depth: ignore GT depth values <= this threshold
#         eps: numerical stabilization for relative error
#     """
#     cfg = {
#         "min_gt_depth": 0.0,
#         "eps": 1e-8,
#     }
#     if config is not None:
#         cfg.update(config)

#     if "pred_depth" not in sample or sample["pred_depth"] is None:
#         return {
#             "depth_mae": np.nan,
#             "depth_rmse": np.nan,
#             "depth_bias": np.nan,
#             "depth_abs_rel_error": np.nan,
#             "depth_n_points": 0,
#         }

#     if "gt_depth" not in sample or sample["gt_depth"] is None:
#         return {
#             "depth_mae": np.nan,
#             "depth_rmse": np.nan,
#             "depth_bias": np.nan,
#             "depth_abs_rel_error": np.nan,
#             "depth_n_points": 0,
#         }

#     if "gt_support" not in sample:
#         raise KeyError("sample is missing 'gt_support'")

#     pred_depth = np.asarray(sample["pred_depth"], dtype=np.float64)
#     gt_depth = np.asarray(sample["gt_depth"], dtype=np.float64)
#     gt_support = np.asarray(sample["gt_support"])

#     if pred_depth.shape != gt_depth.shape:
#         raise ValueError(
#             "pred_depth and gt_depth must have the same shape, "
#             f"got {pred_depth.shape} and {gt_depth.shape}"
#         )

#     if gt_support.shape != gt_depth.shape:
#         raise ValueError(
#             "gt_support and gt_depth must have the same shape, "
#             f"got {gt_support.shape} and {gt_depth.shape}"
#         )

#     mask = gt_support > 0

#     min_gt_depth = float(cfg["min_gt_depth"])
#     if min_gt_depth is not None:
#         mask = mask & (gt_depth > min_gt_depth)

#     mask = mask & np.isfinite(pred_depth) & np.isfinite(gt_depth)

#     n_points = int(mask.sum())

#     if n_points == 0:
#         return {
#             "depth_mae": np.nan,
#             "depth_rmse": np.nan,
#             "depth_bias": np.nan,
#             "depth_abs_rel_error": np.nan,
#             "depth_n_points": 0,
#         }

#     err = pred_depth[mask] - gt_depth[mask]
#     abs_err = np.abs(err)

#     eps = float(cfg["eps"])
#     abs_rel = abs_err / np.maximum(np.abs(gt_depth[mask]), eps)

#     return {
#         "depth_mae": float(abs_err.mean()),
#         "depth_rmse": float(np.sqrt(np.mean(err ** 2))),
#         "depth_bias": float(err.mean()),
#         "depth_abs_rel_error": float(abs_rel.mean()),
#         "depth_n_points": n_points,
#     }