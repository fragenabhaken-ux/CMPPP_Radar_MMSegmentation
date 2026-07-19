#!/usr/bin/env python3
"""Shared spatial-domain configuration and geometry for PPP EDA tools.

The camera target stores projected image coordinates and a forward-depth mark.
Using the frame-specific ``rad2cam`` matrix, the horizontal camera coordinate,
and the forward-depth mark, the original lateral radar coordinate is recovered.
The resulting Cartesian coordinates are then converted to polar coordinates.

Configuration
-------------
Add ``ppp_eda_spatial`` to the EDA config::

    ppp_eda_spatial = dict(
        principal_x_rel=0.491,
        lateral_sign=1.0,
        fallback_h_fov_deg=110.0,
        camera=dict(
            x_min=None,
            x_max=None,
            y_min=None,
            y_max=None,
        ),
        cartesian_bev=dict(
            lateral_min=-50.0,
            lateral_max=50.0,
            forward_min=0.0,
            forward_max=150.0,
        ),
        polar=dict(
            range_min=0.0,
            range_max=150.0,
            azimuth_min_deg=-52.5,
            azimuth_max_deg=52.5,
        ),
    )

Every individual bound may be ``None``. ``None`` means that no point filter is
applied for that bound. For plotting, an unconstrained Cartesian or polar bound
is determined automatically from the finite points. Unconstrained camera bounds
mean the full image extent.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple

import numpy as np


DEFAULT_SPATIAL_CONFIG: Dict[str, Any] = {
    "principal_x_rel": 0.491,
    "lateral_sign": 1.0,
    "fallback_h_fov_deg": 110.0,
    "camera": {
        "x_min": None,
        "x_max": None,
        "y_min": None,
        "y_max": None,
    },
    "cartesian_bev": {
        "lateral_min": None,
        "lateral_max": None,
        "forward_min": None,
        "forward_max": None,
    },
    "polar": {
        "range_min": None,
        "range_max": None,
        "azimuth_min_deg": None,
        "azimuth_max_deg": None,
    },
}


def _plain_mapping(value: Any) -> Dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return dict(value)
    try:
        return dict(value)
    except Exception as exc:
        raise TypeError(
            "ppp_eda_spatial must be a mapping/dict-like object."
        ) from exc


def _deep_update(base: Dict[str, Any], updates: Mapping[str, Any]) -> Dict[str, Any]:
    for key, value in updates.items():
        if key not in base:
            raise KeyError(
                f"Unknown ppp_eda_spatial key {key!r}. "
                f"Allowed top-level keys: {sorted(base)}"
            )
        if isinstance(base[key], dict):
            nested = _plain_mapping(value)
            unknown = sorted(set(nested) - set(base[key]))
            if unknown:
                raise KeyError(
                    f"Unknown ppp_eda_spatial.{key} keys: {unknown}. "
                    f"Allowed keys: {sorted(base[key])}"
                )
            base[key].update(nested)
        else:
            base[key] = value
    return base


def _optional_float(value: Any, name: str) -> float | None:
    if value is None:
        return None
    try:
        value = float(value)
    except Exception as exc:
        raise TypeError(f"{name} must be a float or None, got {value!r}.") from exc
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite or None, got {value!r}.")
    return value


def _validate_pair(lower: float | None, upper: float | None, name: str) -> None:
    if lower is not None and upper is not None and lower >= upper:
        raise ValueError(
            f"Invalid {name} limits: lower={lower} must be smaller than upper={upper}."
        )


def normalize_spatial_config(value: Any = None) -> Dict[str, Any]:
    """Return a validated plain dictionary with all spatial settings present."""
    out = copy.deepcopy(DEFAULT_SPATIAL_CONFIG)
    if value is not None:
        out = _deep_update(out, _plain_mapping(value))

    out["principal_x_rel"] = float(out["principal_x_rel"])
    out["lateral_sign"] = float(out["lateral_sign"])
    out["fallback_h_fov_deg"] = float(out["fallback_h_fov_deg"])

    if not np.isfinite(out["principal_x_rel"]):
        raise ValueError("principal_x_rel must be finite.")
    if not np.isfinite(out["lateral_sign"]) or out["lateral_sign"] == 0.0:
        raise ValueError("lateral_sign must be finite and non-zero.")
    if not np.isfinite(out["fallback_h_fov_deg"]) or out["fallback_h_fov_deg"] <= 0.0:
        raise ValueError("fallback_h_fov_deg must be finite and positive.")

    for section, keys in {
        "camera": ("x_min", "x_max", "y_min", "y_max"),
        "cartesian_bev": (
            "lateral_min",
            "lateral_max",
            "forward_min",
            "forward_max",
        ),
        "polar": (
            "range_min",
            "range_max",
            "azimuth_min_deg",
            "azimuth_max_deg",
        ),
    }.items():
        for key in keys:
            out[section][key] = _optional_float(
                out[section][key], f"ppp_eda_spatial.{section}.{key}"
            )

    _validate_pair(out["camera"]["x_min"], out["camera"]["x_max"], "camera x")
    _validate_pair(out["camera"]["y_min"], out["camera"]["y_max"], "camera y")
    _validate_pair(
        out["cartesian_bev"]["lateral_min"],
        out["cartesian_bev"]["lateral_max"],
        "Cartesian BEV lateral",
    )
    _validate_pair(
        out["cartesian_bev"]["forward_min"],
        out["cartesian_bev"]["forward_max"],
        "Cartesian BEV forward",
    )
    _validate_pair(
        out["polar"]["range_min"],
        out["polar"]["range_max"],
        "polar range",
    )
    _validate_pair(
        out["polar"]["azimuth_min_deg"],
        out["polar"]["azimuth_max_deg"],
        "polar azimuth",
    )
    return out


def resolve_spatial_config_from_cfg(cfg: Any) -> Dict[str, Any]:
    """Read ``ppp_eda_spatial`` from an mmengine Config or plain mapping."""
    if cfg is None:
        return normalize_spatial_config()

    if isinstance(cfg, Mapping):
        value = cfg.get("ppp_eda_spatial", None)
    else:
        value = getattr(cfg, "ppp_eda_spatial", None)
    return normalize_spatial_config(value)


def load_spatial_config_from_scan_dir(scan_dir: Path | str) -> Dict[str, Any]:
    """Load the resolved configuration stored by ``scan_ppp_eda.py``.

    Older scan directories without the field remain readable and receive the
    all-``None`` default configuration.
    """
    scan_dir = Path(scan_dir)
    summary_path = scan_dir / "scan_summary.json"
    if not summary_path.exists():
        return normalize_spatial_config()

    with summary_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return normalize_spatial_config(payload.get("ppp_eda_spatial", None))


def spatial_configs_equal(a: Any, b: Any) -> bool:
    return normalize_spatial_config(a) == normalize_spatial_config(b)


def to_numpy(value: Any, dtype=None) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    else:
        value = np.asarray(value)
    if dtype is not None:
        value = value.astype(dtype, copy=False)
    return value


def get_sample_rad2cam(sample: Mapping[str, Any]) -> np.ndarray:
    value = sample.get("rad2cam", None)
    if value is None:
        meta = sample.get("metainfo", {})
        if isinstance(meta, Mapping):
            value = meta.get("rad2cam", None)
    if value is None:
        raise KeyError(
            "Sample lacks rad2cam. Keep results['rad2cam'] in the EDA pipeline "
            "or add it to metainfo before rendering."
        )
    value = to_numpy(value, dtype=np.float64)
    # Some SYS100 samples expose rad2cam with an optional leading
    # batch-size-1 dimension: [1, 3, 4]. Normalize it to [3, 4].
    if value.ndim == 3 and value.shape[0] == 1:
        value = value[0]

    if value.shape != (3, 4):
        raise ValueError(
            f"Expected rad2cam shape (3, 4) or (1, 3, 4), got {value.shape}."
        )
    return value


def get_sample_h_fov_deg(sample: Mapping[str, Any], spatial: Mapping[str, Any]) -> float:
    value = sample.get("fovs", None)
    if value is None:
        meta = sample.get("metainfo", {})
        if isinstance(meta, Mapping):
            value = meta.get("fovs", None)
    if value is not None:
        value = to_numpy(value).reshape(-1)
        if len(value) > 0 and np.isfinite(float(value[0])):
            return float(value[0])
    return float(spatial["fallback_h_fov_deg"])


def recover_lateral_from_forward_depth(
    u_px: np.ndarray,
    forward_m: np.ndarray,
    rad2cam: np.ndarray,
    image_width: int,
    h_fov_deg: float,
    principal_x_rel: float,
) -> np.ndarray:
    """Recover Cartesian lateral position from image x and forward depth."""
    u_px = np.asarray(u_px, dtype=np.float64)
    forward_m = np.asarray(forward_m, dtype=np.float64)
    rad2cam = np.asarray(rad2cam, dtype=np.float64)

    if rad2cam.shape != (3, 4):
        raise ValueError(f"Expected rad2cam shape (3, 4), got {rad2cam.shape}.")

    width = float(max(int(image_width), 1))
    cx = float(principal_x_rel) * (width - 1.0)
    fx = (width / 2.0) / np.tan(np.deg2rad(float(h_fov_deg)) / 2.0)

    qx = (u_px - cx) / fx
    qx, forward_m = np.broadcast_arrays(qx, forward_m)

    a00 = rad2cam[0, 0]
    a01 = rad2cam[0, 1]
    a20 = rad2cam[2, 0]
    a21 = rad2cam[2, 1]
    t0 = rad2cam[0, 3]
    t2 = rad2cam[2, 3]

    numerator = qx * (a20 * forward_m + t2) - (a00 * forward_m + t0)
    denominator = a01 - qx * a21

    lateral = np.full_like(numerator, np.nan, dtype=np.float64)
    np.divide(
        numerator,
        denominator,
        out=lateral,
        where=np.abs(denominator) > 1e-12,
    )
    return lateral


def camera_points_to_spatial(
    x_pix: np.ndarray,
    depth: np.ndarray,
    rad2cam: np.ndarray,
    image_width: int,
    h_fov_deg: float,
    spatial: Mapping[str, Any],
    pixel_center_offset: float = 0.5,
) -> Dict[str, np.ndarray]:
    """Back-project camera target points and return full-length spatial arrays.

    Invalid back-projections are represented by ``NaN`` so all exported arrays
    retain exactly the same number and order of points as the camera target.
    """
    x_pix = np.asarray(x_pix, dtype=np.float64)
    depth = np.asarray(depth, dtype=np.float64)
    if x_pix.shape != depth.shape:
        raise ValueError(
            f"x_pix and depth must have the same shape, got {x_pix.shape} and {depth.shape}."
        )

    forward = depth.astype(np.float64, copy=True)
    lateral = np.full_like(forward, np.nan, dtype=np.float64)

    valid = np.isfinite(x_pix) & np.isfinite(forward)
    if np.any(valid):
        lateral_valid = recover_lateral_from_forward_depth(
            u_px=x_pix[valid] + float(pixel_center_offset),
            forward_m=forward[valid],
            rad2cam=rad2cam,
            image_width=image_width,
            h_fov_deg=h_fov_deg,
            principal_x_rel=float(spatial["principal_x_rel"]),
        )
        lateral[valid] = float(spatial["lateral_sign"]) * lateral_valid

    range_m = np.hypot(forward, lateral)
    azimuth_deg = np.rad2deg(np.arctan2(lateral, forward))

    geometry_valid = (
        np.isfinite(lateral)
        & np.isfinite(forward)
        & np.isfinite(range_m)
        & np.isfinite(azimuth_deg)
    )
    return {
        "lateral": lateral,
        "forward": forward,
        "range": range_m,
        "azimuth_deg": azimuth_deg,
        "geometry_valid": geometry_valid,
    }


def _bound_mask(
    values: np.ndarray,
    lower: float | None,
    upper: float | None,
) -> np.ndarray:
    values = np.asarray(values)
    mask = np.isfinite(values)
    if lower is not None:
        mask &= values >= float(lower)
    if upper is not None:
        mask &= values <= float(upper)
    return mask


def camera_domain_mask(
    x_pix: np.ndarray,
    y_pix: np.ndarray,
    spatial: Mapping[str, Any],
) -> np.ndarray:
    cfg = spatial["camera"]
    return _bound_mask(x_pix, cfg["x_min"], cfg["x_max"]) & _bound_mask(
        y_pix, cfg["y_min"], cfg["y_max"]
    )


def cartesian_domain_mask(
    lateral: np.ndarray,
    forward: np.ndarray,
    spatial: Mapping[str, Any],
) -> np.ndarray:
    cfg = spatial["cartesian_bev"]
    return _bound_mask(lateral, cfg["lateral_min"], cfg["lateral_max"]) & _bound_mask(
        forward, cfg["forward_min"], cfg["forward_max"]
    )


def polar_domain_mask(
    range_m: np.ndarray,
    azimuth_deg: np.ndarray,
    spatial: Mapping[str, Any],
) -> np.ndarray:
    cfg = spatial["polar"]
    return _bound_mask(range_m, cfg["range_min"], cfg["range_max"]) & _bound_mask(
        azimuth_deg, cfg["azimuth_min_deg"], cfg["azimuth_max_deg"]
    )


def filter_point_dict(points: Mapping[str, np.ndarray], mask: np.ndarray) -> Dict[str, np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    out: Dict[str, np.ndarray] = {}
    for key, value in points.items():
        arr = np.asarray(value)
        if arr.ndim >= 1 and len(arr) == len(mask):
            out[key] = arr[mask]
        else:
            out[key] = arr
    return out


def finite_auto_limits(
    values: np.ndarray,
    lower: float | None,
    upper: float | None,
    fallback: Tuple[float, float],
    pad_fraction: float = 0.05,
    include_zero: bool = False,
) -> Tuple[float, float]:
    """Resolve plotting limits while preserving independently optional bounds."""
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        data_lo, data_hi = float(fallback[0]), float(fallback[1])
    else:
        data_lo = float(np.min(values))
        data_hi = float(np.max(values))
        if include_zero:
            data_lo = min(data_lo, 0.0)
            data_hi = max(data_hi, 0.0)
        if data_lo == data_hi:
            delta = max(abs(data_lo) * 0.05, 0.5)
            data_lo -= delta
            data_hi += delta
        else:
            pad = (data_hi - data_lo) * float(pad_fraction)
            data_lo -= pad
            data_hi += pad

    lo = float(lower) if lower is not None else data_lo
    hi = float(upper) if upper is not None else data_hi
    if lo >= hi:
        delta = max(abs(lo) * 0.05, 0.5)
        lo -= delta
        hi += delta
    return lo, hi


def camera_plot_limits(
    image_width: int,
    image_height: int,
    spatial: Mapping[str, Any],
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    cfg = spatial["camera"]
    xlim = (
        0.0 if cfg["x_min"] is None else float(cfg["x_min"]),
        float(image_width) if cfg["x_max"] is None else float(cfg["x_max"]),
    )
    ylim = (
        0.0 if cfg["y_min"] is None else float(cfg["y_min"]),
        float(image_height) if cfg["y_max"] is None else float(cfg["y_max"]),
    )
    return xlim, ylim


def cartesian_plot_limits(
    lateral: np.ndarray,
    forward: np.ndarray,
    spatial: Mapping[str, Any],
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    cfg = spatial["cartesian_bev"]
    xlim = finite_auto_limits(
        lateral,
        cfg["lateral_min"],
        cfg["lateral_max"],
        fallback=(-50.0, 50.0),
        include_zero=True,
    )
    ylim = finite_auto_limits(
        forward,
        cfg["forward_min"],
        cfg["forward_max"],
        fallback=(0.0, 150.0),
        include_zero=True,
    )
    return xlim, ylim


def polar_plot_limits(
    azimuth_deg: np.ndarray,
    range_m: np.ndarray,
    spatial: Mapping[str, Any],
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    cfg = spatial["polar"]
    xlim = finite_auto_limits(
        azimuth_deg,
        cfg["azimuth_min_deg"],
        cfg["azimuth_max_deg"],
        fallback=(-52.5, 52.5),
        include_zero=True,
    )
    ylim = finite_auto_limits(
        range_m,
        cfg["range_min"],
        cfg["range_max"],
        fallback=(0.0, 150.0),
        include_zero=True,
    )
    return xlim, ylim


def points_from_gt(
    gt: np.ndarray,
    sample: Mapping[str, Any],
    image_width: int,
    spatial: Mapping[str, Any],
) -> Dict[str, np.ndarray]:
    """Extract the four-channel camera target and append BEV coordinates."""
    gt = np.asarray(gt)
    if gt.ndim != 3 or gt.shape[-1] < 4:
        raise ValueError(f"Expected HWC GT with at least 4 channels, got {gt.shape}.")

    mask = gt[..., 0] > 0
    y_pix, x_pix = np.nonzero(mask)
    points: Dict[str, np.ndarray] = {
        "x": x_pix.astype(np.float64, copy=False),
        "y": y_pix.astype(np.float64, copy=False),
        "depth": gt[..., 1][mask].astype(np.float64, copy=False),
        "velocity": gt[..., 2][mask].astype(np.float64, copy=False),
        "rcs": gt[..., 3][mask].astype(np.float64, copy=False),
    }

    rad2cam = get_sample_rad2cam(sample)
    h_fov_deg = get_sample_h_fov_deg(sample, spatial)
    geometry = camera_points_to_spatial(
        x_pix=points["x"],
        depth=points["depth"],
        rad2cam=rad2cam,
        image_width=image_width,
        h_fov_deg=h_fov_deg,
        spatial=spatial,
    )
    points.update(geometry)
    return points
