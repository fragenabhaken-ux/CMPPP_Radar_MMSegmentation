from __future__ import annotations

from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist

from .mark_metrics import circular_difference
from .pointcloud import MarkedPointCloud
from .types import BEVGeometry


DEFAULT_CONFIG: dict[str, Any] = {
    "iou_threshold_m": 1.0,
    "distance_threshold_m": 1.0,
    "rcs_threshold": 8.0,
    "velocity_threshold_mps": 2.5,
    "use_circular_velocity_matching": True,
    "rcs_min": -60.0,
    "rcs_max": 10.0,
    "velocity_min": -11.0,
    "velocity_max": 11.0,
    "mmd_n_kernels": 5,
    "mmd_multiplier": 2.0,
    "mmd_max_points": 2048,
    "mmd_negative_tolerance": 1e-10,
}


def nearest_neighbor_distances(
    source: np.ndarray,
    target: np.ndarray,
) -> np.ndarray:
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if len(source) == 0 or len(target) == 0:
        return np.zeros((0,), dtype=np.float64)
    tree = cKDTree(target)
    distances, _ = tree.query(source, k=1)
    return np.asarray(distances, dtype=np.float64)


def chamfer_distance_location(pred_xy: np.ndarray, gt_xy: np.ndarray) -> float:
    if len(pred_xy) == 0 or len(gt_xy) == 0:
        return np.nan
    pred_to_gt = nearest_neighbor_distances(pred_xy, gt_xy)
    gt_to_pred = nearest_neighbor_distances(gt_xy, pred_xy)
    return 0.5 * float(pred_to_gt.mean() + gt_to_pred.mean())


def pointcloud_iou_f1(
    pred_xy: np.ndarray,
    gt_xy: np.ndarray,
    threshold_m: float,
) -> dict[str, float]:
    if len(pred_xy) == 0 or len(gt_xy) == 0:
        return {"pc_precision": np.nan, "pc_recall": np.nan, "pc_f1": np.nan, "pc_iou": np.nan}

    pred_to_gt = nearest_neighbor_distances(pred_xy, gt_xy)
    gt_to_pred = nearest_neighbor_distances(gt_xy, pred_xy)
    precision = float(np.mean(pred_to_gt < threshold_m))
    recall = float(np.mean(gt_to_pred < threshold_m))
    if precision + recall > 0.0:
        f1 = 2.0 * precision * recall / (precision + recall)
        iou = precision * recall / (precision + recall - precision * recall)
    else:
        f1 = 0.0
        iou = 0.0
    return {
        "pc_precision": precision,
        "pc_recall": recall,
        "pc_f1": f1,
        "pc_iou": iou,
    }


def normalize_full_point_cloud(
    cloud: MarkedPointCloud,
    geometry: BEVGeometry,
    config: dict[str, Any],
) -> np.ndarray:
    pcl = cloud.as_nx4()
    if len(pcl) == 0:
        return pcl
    bounds_min = np.array(
        [geometry.x_min, geometry.y_min, float(config["rcs_min"]), float(config["velocity_min"])],
        dtype=np.float64,
    )
    bounds_max = np.array(
        [geometry.x_max, geometry.y_max, float(config["rcs_max"]), float(config["velocity_max"])],
        dtype=np.float64,
    )
    widths = bounds_max - bounds_min
    if np.any(widths <= 0.0):
        raise ValueError(f"Invalid normalization bounds: min={bounds_min}, max={bounds_max}.")
    return (pcl - bounds_min) / widths


def chamfer_distance_full_normalized(
    pred: MarkedPointCloud,
    gt: MarkedPointCloud,
    geometry: BEVGeometry,
    config: dict[str, Any],
) -> float:
    if pred.n_points == 0 or gt.n_points == 0:
        return np.nan
    pred_norm = normalize_full_point_cloud(pred, geometry, config)
    gt_norm = normalize_full_point_cloud(gt, geometry, config)
    return chamfer_distance_location(pred_norm, gt_norm)


def distance_attribute_matching(
    pred: MarkedPointCloud,
    gt: MarkedPointCloud,
    *,
    distance_threshold_m: float,
    rcs_threshold: float,
    velocity_threshold_mps: float,
    doppler_period: float | None,
    use_circular_velocity: bool,
) -> dict[str, float | int]:
    n_pred = pred.n_points
    n_gt = gt.n_points
    if n_gt == 0 and n_pred == 0:
        return {"da_precision": 1.0, "da_recall": 1.0, "da_f1": 1.0, "da_matches": 0}
    if n_gt == 0:
        return {"da_precision": 0.0, "da_recall": 1.0, "da_f1": 0.0, "da_matches": 0}
    if n_pred == 0:
        return {"da_precision": 1.0, "da_recall": 0.0, "da_f1": 0.0, "da_matches": 0}

    spatial = cdist(gt.xy, pred.xy, metric="euclidean")
    rcs_diff = np.abs(gt.rcs[:, None] - pred.rcs[None, :])
    if use_circular_velocity:
        if doppler_period is None:
            raise KeyError("Circular distance-attribute matching requires doppler_period.")
        velocity_diff = np.abs(
            circular_difference(
                pred.velocity[None, :],
                gt.velocity[:, None],
                float(doppler_period),
            )
        )
    else:
        velocity_diff = np.abs(gt.velocity[:, None] - pred.velocity[None, :])

    valid = (
        (spatial <= distance_threshold_m)
        & (rcs_diff <= rcs_threshold)
        & (velocity_diff <= velocity_threshold_mps)
    )
    invalid_cost = float(distance_threshold_m) + 1e6
    cost = np.where(valid, spatial, invalid_cost)
    gt_indices, pred_indices = linear_sum_assignment(cost)
    matches = int(valid[gt_indices, pred_indices].sum())
    precision = matches / n_pred
    recall = matches / n_gt
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall > 0.0 else 0.0
    return {
        "da_precision": float(precision),
        "da_recall": float(recall),
        "da_f1": float(f1),
        "da_matches": matches,
    }


def _deterministic_subsample(values: np.ndarray, max_points: int | None) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if max_points is None or len(values) <= int(max_points):
        return values
    # Evenly spaced deterministic indices avoid introducing another RNG stream.
    indices = np.linspace(0, len(values) - 1, int(max_points), dtype=np.int64)
    return values[indices]


def biased_rbf_mmd_squared(
    x: np.ndarray,
    y: np.ndarray,
    *,
    n_kernels: int = 5,
    multiplier: float = 2.0,
    max_points: int | None = 2048,
    negative_tolerance: float = 1e-10,
) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    if y.ndim == 1:
        y = y[:, None]
    if len(x) == 0 or len(y) == 0:
        return np.nan
    if x.shape[1] != y.shape[1]:
        raise ValueError(f"MMD dimensions differ: x={x.shape}, y={y.shape}.")
    if n_kernels < 1 or multiplier <= 0.0:
        raise ValueError("Invalid RBF-kernel configuration.")

    x = _deterministic_subsample(x, max_points)
    y = _deterministic_subsample(y, max_points)
    z = np.vstack([x, y])
    d2 = cdist(z, z, metric="sqeuclidean")
    n_total = len(z)
    if n_total < 2:
        bandwidth = 1.0
    else:
        bandwidth = float(d2.sum() / (n_total * n_total - n_total))
    bandwidth += np.finfo(np.float64).eps

    exponents = np.arange(n_kernels, dtype=np.float64) - n_kernels // 2
    multipliers = multiplier ** exponents
    kernel = np.zeros_like(d2)
    for factor in multipliers:
        kernel += np.exp(-d2 / (bandwidth * factor))

    nx = len(x)
    xx = float(kernel[:nx, :nx].mean())
    xy = float(kernel[:nx, nx:].mean())
    yy = float(kernel[nx:, nx:].mean())
    result = xx - 2.0 * xy + yy
    if result < 0.0:
        if result >= -float(negative_tolerance):
            result = 0.0
        else:
            raise ValueError(f"MMD squared is negative beyond tolerance: {result}.")
    return float(result)


def compute_radargen_pointcloud_metrics(
    pred: MarkedPointCloud,
    gt: MarkedPointCloud,
    geometry: BEVGeometry,
    *,
    doppler_period: float | None,
    config: dict[str, Any] | None = None,
) -> dict[str, float | int]:
    cfg = dict(DEFAULT_CONFIG)
    if config:
        cfg.update(config)

    result: dict[str, float | int] = {
        "sampled_pred_count": pred.n_points,
        "pointcloud_gt_count": gt.n_points,
        "pointcloud_pair_nonempty": int(pred.n_points > 0 and gt.n_points > 0),
    }

    result["chamfer_location_m"] = chamfer_distance_location(pred.xy, gt.xy)
    result["chamfer_full_normalized"] = chamfer_distance_full_normalized(
        pred,
        gt,
        geometry,
        cfg,
    )

    coverage = pointcloud_iou_f1(
        pred.xy,
        gt.xy,
        threshold_m=float(cfg["iou_threshold_m"]),
    )
    result.update({
        "iou_1m": coverage["pc_iou"],
        "coverage_precision_1m": coverage["pc_precision"],
        "coverage_recall_1m": coverage["pc_recall"],
        "coverage_f1_1m": coverage["pc_f1"],
    })

    result.update(
        distance_attribute_matching(
            pred,
            gt,
            distance_threshold_m=float(cfg["distance_threshold_m"]),
            rcs_threshold=float(cfg["rcs_threshold"]),
            velocity_threshold_mps=float(cfg["velocity_threshold_mps"]),
            doppler_period=doppler_period,
            use_circular_velocity=bool(cfg["use_circular_velocity_matching"]),
        )
    )

    mmd_kwargs = {
        "n_kernels": int(cfg["mmd_n_kernels"]),
        "multiplier": float(cfg["mmd_multiplier"]),
        "max_points": cfg.get("mmd_max_points"),
        "negative_tolerance": float(cfg["mmd_negative_tolerance"]),
    }
    result["mmd_location"] = biased_rbf_mmd_squared(pred.xy, gt.xy, **mmd_kwargs)
    result["mmd_rcs"] = biased_rbf_mmd_squared(pred.rcs, gt.rcs, **mmd_kwargs)
    # Deliberately ordinary Euclidean Doppler MMD, matching RadarGen and the
    # thesis definition. Circular distance is used only for DA matching.
    result["mmd_velocity"] = biased_rbf_mmd_squared(
        pred.velocity,
        gt.velocity,
        **mmd_kwargs,
    )
    return result
