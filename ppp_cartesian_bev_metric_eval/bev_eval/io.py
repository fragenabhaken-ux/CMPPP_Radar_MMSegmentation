from __future__ import annotations

import glob
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from .types import BEVGeometry


REQUIRED_DATASETS = (
    "pred_lambda",
    "pred_velocity",
    "pred_rcs",
    "gt_support",
    "gt_velocity",
    "gt_rcs",
)


@dataclass(frozen=True)
class H5Entry:
    model: str
    h5_path: str
    sample_key: str
    sample_uid: str

    @property
    def rank_file(self) -> str:
        return os.path.basename(self.h5_path)


def _decode_attr(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.ndarray) and value.ndim == 0:
        return value.item()
    if isinstance(value, np.generic):
        return value.item()
    return value


def read_attrs(obj: h5py.Group | h5py.File) -> dict[str, Any]:
    return {key: _decode_attr(value) for key, value in obj.attrs.items()}


def collect_h5_files(h5_dir: str | os.PathLike[str]) -> list[str]:
    h5_dir = str(Path(h5_dir).expanduser())
    if not os.path.isdir(h5_dir):
        raise FileNotFoundError(f"HDF5 directory not found: {h5_dir}")
    files = sorted(glob.glob(os.path.join(h5_dir, "rank*.h5")))
    if not files:
        raise FileNotFoundError(f"No rank*.h5 files found in: {h5_dir}")
    return files


def stable_sample_uid(
    group_attrs: dict[str, Any],
    *,
    rank_file: str,
    sample_key: str,
) -> str:
    video = group_attrs.get("video_name")
    frame = group_attrs.get("video_frame_idx")
    if video is not None and frame is not None:
        try:
            frame_text = str(int(frame))
        except (TypeError, ValueError):
            frame_text = str(frame)
        return f"video={video}|frame={frame_text}"

    sample_id = group_attrs.get("sample_id")
    if sample_id is not None:
        return f"sample_id={sample_id}"

    global_idx = group_attrs.get("global_idx", group_attrs.get("frame_idx"))
    if global_idx is not None:
        return f"global_idx={global_idx}"

    return f"{rank_file}/{sample_key}"


def index_model_samples(
    model_name: str,
    h5_dir: str | os.PathLike[str],
) -> dict[str, H5Entry]:
    index: dict[str, H5Entry] = {}
    for h5_path in collect_h5_files(h5_dir):
        rank_file = os.path.basename(h5_path)
        with h5py.File(h5_path, "r") as h5f:
            for sample_key in sorted(h5f.keys()):
                grp = h5f[sample_key]
                attrs = read_attrs(grp)
                uid = stable_sample_uid(
                    attrs,
                    rank_file=rank_file,
                    sample_key=sample_key,
                )
                if uid in index:
                    other = index[uid]
                    raise ValueError(
                        f"Duplicate stable sample UID {uid!r} for model {model_name!r}: "
                        f"{other.h5_path}/{other.sample_key} and {h5_path}/{sample_key}."
                    )
                index[uid] = H5Entry(
                    model=model_name,
                    h5_path=h5_path,
                    sample_key=sample_key,
                    sample_uid=uid,
                )
    return index


def _read_optional(grp: h5py.Group, name: str) -> np.ndarray | None:
    if name not in grp:
        return None
    return grp[name][:]


def _assert_grid(name: str, value: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    arr = np.asarray(value)
    if arr.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {arr.shape}.")
    return arr


def load_bev_sample(entry: H5Entry) -> dict[str, Any]:
    with h5py.File(entry.h5_path, "r") as h5f:
        if entry.sample_key not in h5f:
            raise KeyError(f"Missing group {entry.sample_key!r} in {entry.h5_path}.")
        grp = h5f[entry.sample_key]
        missing = [name for name in REQUIRED_DATASETS if name not in grp]
        if missing:
            raise KeyError(
                f"Missing required datasets {missing} in "
                f"{entry.h5_path}/{entry.sample_key}."
            )

        file_attrs = read_attrs(h5f)
        group_attrs = read_attrs(grp)
        merged_attrs = dict(file_attrs)
        merged_attrs.update(group_attrs)

        pred_lambda = grp["pred_lambda"][:].astype(np.float64, copy=False)
        if pred_lambda.ndim != 2:
            raise ValueError(
                f"pred_lambda must be 2D, got {pred_lambda.shape} in "
                f"{entry.h5_path}/{entry.sample_key}."
            )
        shape = pred_lambda.shape
        geometry = BEVGeometry.from_attrs(merged_attrs, shape)

        sample: dict[str, Any] = {
            "pred_lambda": pred_lambda,
            "pred_velocity": _assert_grid(
                "pred_velocity",
                grp["pred_velocity"][:].astype(np.float64, copy=False),
                shape,
            ),
            "pred_rcs": _assert_grid(
                "pred_rcs",
                grp["pred_rcs"][:].astype(np.float64, copy=False),
                shape,
            ),
            "gt_support": _assert_grid(
                "gt_support",
                grp["gt_support"][:].astype(np.float64, copy=False),
                shape,
            ),
            "gt_velocity": _assert_grid(
                "gt_velocity",
                grp["gt_velocity"][:].astype(np.float64, copy=False),
                shape,
            ),
            "gt_rcs": _assert_grid(
                "gt_rcs",
                grp["gt_rcs"][:].astype(np.float64, copy=False),
                shape,
            ),
            "gt_points_xy": _read_optional(grp, "gt_points_xy"),
            "gt_point_velocity": _read_optional(grp, "gt_point_velocity"),
            "gt_point_rcs": _read_optional(grp, "gt_point_rcs"),
            "attrs": merged_attrs,
            "file_attrs": file_attrs,
            "group_attrs": group_attrs,
            "geometry": geometry,
            "meta": {
                "model": entry.model,
                "sample_uid": entry.sample_uid,
                "rank_file": entry.rank_file,
                "sample_key": entry.sample_key,
                "h5_path": entry.h5_path,
            },
        }

    _validate_sample(sample)
    return sample


def _validate_sample(sample: dict[str, Any]) -> None:
    mass = sample["pred_lambda"]
    support = sample["gt_support"]
    if not np.all(np.isfinite(mass)):
        raise ValueError("pred_lambda contains NaN or Inf.")
    if np.any(mass < -1e-8):
        raise ValueError(f"pred_lambda contains negative mass, minimum={mass.min()}.")
    # Remove harmless floating-point negatives only.
    sample["pred_lambda"] = np.clip(mass, 0.0, None)

    if not np.all(np.isfinite(support)):
        raise ValueError("gt_support contains NaN or Inf.")
    if np.any(support < -1e-8):
        raise ValueError(f"gt_support contains negative values, minimum={support.min()}.")
    sample["gt_support"] = np.clip(support, 0.0, None)

    for name in ("pred_velocity", "pred_rcs", "gt_velocity", "gt_rcs"):
        array = np.asarray(sample[name], dtype=np.float64)
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} contains NaN or Inf.")

    exact = (
        sample.get("gt_points_xy") is not None
        or sample.get("gt_point_velocity") is not None
        or sample.get("gt_point_rcs") is not None
    )
    if exact:
        if any(
            sample.get(name) is None
            for name in ("gt_points_xy", "gt_point_velocity", "gt_point_rcs")
        ):
            raise ValueError(
                "Exact GT datasets must be present together: gt_points_xy, "
                "gt_point_velocity, gt_point_rcs."
            )
        xy = np.asarray(sample["gt_points_xy"], dtype=np.float64)
        vel = np.asarray(sample["gt_point_velocity"], dtype=np.float64).reshape(-1)
        rcs = np.asarray(sample["gt_point_rcs"], dtype=np.float64).reshape(-1)
        if xy.ndim != 2 or xy.shape[1] != 2:
            raise ValueError(f"gt_points_xy must have shape (N, 2), got {xy.shape}.")
        if not (len(xy) == len(vel) == len(rcs)):
            raise ValueError(
                "Exact GT point arrays have inconsistent lengths: "
                f"xy={len(xy)}, velocity={len(vel)}, rcs={len(rcs)}."
            )
        sample["gt_points_xy"] = xy
        sample["gt_point_velocity"] = vel
        sample["gt_point_rcs"] = rcs


def geometry_signature(geometry: BEVGeometry) -> tuple[float, float, float, float, int, int]:
    return (
        geometry.x_min,
        geometry.x_max,
        geometry.y_min,
        geometry.y_max,
        geometry.height,
        geometry.width,
    )
