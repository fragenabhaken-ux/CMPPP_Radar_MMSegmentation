"""HDF5 input/output helpers for camera-domain PPP metric evaluation."""

from __future__ import annotations

import glob
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np


@dataclass(frozen=True)
class H5Entry:
    model: str
    h5_path: str
    sample_key: str

    @property
    def rank_file(self) -> str:
        return os.path.basename(self.h5_path)

    @property
    def fallback_sample_id(self) -> str:
        return f"{self.rank_file}/{self.sample_key}"


def collect_h5_files(h5_dir: str | os.PathLike[str]) -> list[str]:
    h5_dir = str(Path(h5_dir).expanduser())
    if not os.path.isdir(h5_dir):
        raise FileNotFoundError(f"HDF5 directory not found: {h5_dir}")
    paths = sorted(glob.glob(os.path.join(h5_dir, "rank*.h5")))
    if not paths:
        raise FileNotFoundError(f"No rank*.h5 files found in: {h5_dir}")
    return paths


def list_h5_samples_for_model(
    model_name: str,
    h5_dir: str | os.PathLike[str],
) -> list[H5Entry]:
    entries: list[H5Entry] = []
    for h5_path in collect_h5_files(h5_dir):
        with h5py.File(h5_path, "r") as h5f:
            for sample_key in sorted(h5f.keys()):
                if isinstance(h5f[sample_key], h5py.Group):
                    entries.append(H5Entry(model_name, h5_path, sample_key))
    return entries


def _read_optional(group: h5py.Group, name: str) -> np.ndarray | None:
    return group[name][:] if name in group else None


def _decode_attr(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.ndarray) and value.ndim == 0:
        return _decode_attr(value.item())
    return value


def _read_attrs(group: h5py.Group) -> dict[str, Any]:
    return {key: _decode_attr(value) for key, value in group.attrs.items()}


def _basename_without_query(value: Any) -> str:
    text = str(_decode_attr(value))
    return os.path.basename(text.replace("\\", "/"))


def build_sample_uid(
    attrs: dict[str, Any],
    *,
    fallback: str,
) -> tuple[str, bool]:
    """Build a stable physical-frame identifier when metadata permits it."""
    frame = attrs.get("video_frame_idx", attrs.get("frame_idx"))
    video = attrs.get("video_name")
    if video is None:
        video = attrs.get("video_path")

    if video is not None and frame is not None:
        try:
            frame_int = int(np.asarray(frame).reshape(-1)[0])
            video_name = _basename_without_query(video)
            return f"video={video_name}|frame={frame_int}", True
        except (TypeError, ValueError, IndexError):
            pass

    sample_id = attrs.get("sample_id")
    if sample_id is not None and str(sample_id).strip():
        return f"sample_id={str(sample_id)}", True

    global_idx = attrs.get("global_idx")
    if global_idx is not None:
        try:
            return f"global_idx={int(global_idx)}", True
        except (TypeError, ValueError):
            pass

    return fallback, False


def load_h5_sample(entry: H5Entry) -> dict[str, Any]:
    with h5py.File(entry.h5_path, "r") as h5f:
        if entry.sample_key not in h5f:
            raise KeyError(
                f"Missing sample group {entry.sample_key!r} in {entry.h5_path}"
            )
        group = h5f[entry.sample_key]
        required = ["img", "pred_lambda", "gt_support"]
        missing = [name for name in required if name not in group]
        if missing:
            raise KeyError(
                f"Missing datasets {missing} in {entry.h5_path}/{entry.sample_key}"
            )

        sample: dict[str, Any] = {
            "img": group["img"][:],
            "pred_lambda": group["pred_lambda"][:].astype(np.float64, copy=False),
            "gt_support": group["gt_support"][:],
            "pred_depth": _read_optional(group, "pred_depth"),
            "pred_depth_d0": _read_optional(group, "pred_depth_d0"),
            "pred_depth_near": _read_optional(group, "pred_depth_near"),
            "pred_depth_far": _read_optional(group, "pred_depth_far"),
            "pred_depth_delta": _read_optional(group, "pred_depth_delta"),
            "pred_velocity": _read_optional(group, "pred_velocity"),
            "pred_rcs": _read_optional(group, "pred_rcs"),
            "gt_depth": _read_optional(group, "gt_depth"),
            "gt_velocity": _read_optional(group, "gt_velocity"),
            "gt_rcs": _read_optional(group, "gt_rcs"),
            "attrs": _read_attrs(group),
        }

    pred_lambda = np.asarray(sample["pred_lambda"])
    gt_support = np.asarray(sample["gt_support"])
    if pred_lambda.ndim != 2:
        raise ValueError(
            f"pred_lambda must be 2D, got {pred_lambda.shape} for {entry.h5_path}"
        )
    if gt_support.shape != pred_lambda.shape:
        raise ValueError(
            "pred_lambda and gt_support shapes differ: "
            f"{pred_lambda.shape} vs {gt_support.shape}"
        )

    for name in [
        "pred_depth",
        "pred_depth_d0",
        "pred_depth_near",
        "pred_depth_far",
        "pred_depth_delta",
        "pred_velocity",
        "pred_rcs",
        "gt_depth",
        "gt_velocity",
        "gt_rcs",
    ]:
        value = sample.get(name)
        if value is not None and np.asarray(value).shape != pred_lambda.shape:
            raise ValueError(
                f"{name} shape {np.asarray(value).shape} does not match "
                f"pred_lambda shape {pred_lambda.shape}"
            )

    sample_uid, stable = build_sample_uid(
        sample["attrs"],
        fallback=entry.fallback_sample_id,
    )
    height, width = pred_lambda.shape
    sample.update(
        {
            "shape": {"H": int(height), "W": int(width), "area": int(height * width)},
            "meta": {
                "model": entry.model,
                "rank_file": entry.rank_file,
                "sample_key": entry.sample_key,
                "h5_path": entry.h5_path,
                "sample_uid": sample_uid,
                "sample_uid_is_stable": stable,
            },
        }
    )
    return sample
