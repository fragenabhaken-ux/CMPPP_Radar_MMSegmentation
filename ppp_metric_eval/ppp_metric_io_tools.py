"""
HDF5 IO tools for offline PPP metric evaluation.

This file contains no metric logic.

Responsibilities:
    - find rank*.h5 files
    - list sample groups
    - load one sample
    - attach basic PPP diagnostics:
        H, W, expected_count, gt_count

Supported HDF5 formats
----------------------
Old single-depth export:
    pred_depth
    pred_velocity
    gt_depth
    gt_velocity

New two-depth export:
    pred_depth_d0
    pred_depth_near
    pred_depth_far
    pred_depth_delta
    pred_depth_best
    pred_depth_best_abs_error
    pred_depth_best_choice
    pred_velocity
    gt_depth
    gt_velocity

The loader is backward-compatible:
    - old files get the two-depth fields as None
    - new files may have pred_depth as None, but have pred_depth_near/far
"""

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
    """
    One sample entry inside one HDF5 rank file.
    """

    model: str
    h5_path: str
    sample_key: str

    @property
    def rank_file(self) -> str:
        return os.path.basename(self.h5_path)

    @property
    def sample_id(self) -> str:
        return f"{self.rank_file}/{self.sample_key}"


def collect_h5_files(h5_dir: str | os.PathLike[str]) -> list[str]:
    """
    Return sorted rank*.h5 files from a metric export directory.
    """
    h5_dir = str(Path(h5_dir).expanduser())

    if not os.path.isdir(h5_dir):
        raise FileNotFoundError(f"H5 directory not found: {h5_dir}")

    files = sorted(glob.glob(os.path.join(h5_dir, "rank*.h5")))

    if not files:
        raise FileNotFoundError(f"No rank*.h5 files found in: {h5_dir}")

    return files


def list_h5_samples_for_model(
    model_name: str,
    h5_dir: str | os.PathLike[str],
) -> list[H5Entry]:
    """
    Return all sample entries for one model run.

    Each entry identifies:
        model name
        rank file path
        sample group key
    """
    entries: list[H5Entry] = []

    for h5_path in collect_h5_files(h5_dir):
        with h5py.File(h5_path, "r") as f:
            for sample_key in sorted(f.keys()):
                entries.append(
                    H5Entry(
                        model=model_name,
                        h5_path=h5_path,
                        sample_key=sample_key,
                    )
                )

    return entries


def list_h5_samples(
    h5_dirs: dict[str, str | os.PathLike[str]],
) -> list[H5Entry]:
    """
    Return all sample entries across all configured model runs.
    """
    entries: list[H5Entry] = []

    for model_name, h5_dir in h5_dirs.items():
        entries.extend(
            list_h5_samples_for_model(
                model_name=model_name,
                h5_dir=h5_dir,
            )
        )

    return entries


def _read_optional_dataset(
    grp: h5py.Group,
    name: str,
) -> np.ndarray | None:
    """
    Read optional HDF5 dataset if present.
    """
    if name not in grp:
        return None
    return grp[name][:]


def _read_attrs(grp: h5py.Group) -> dict[str, Any]:
    """
    Convert HDF5 group attributes to a normal dict.
    """
    attrs: dict[str, Any] = {}

    for key, value in grp.attrs.items():
        try:
            if isinstance(value, bytes):
                attrs[key] = value.decode("utf-8")
            else:
                attrs[key] = value
        except Exception:
            attrs[key] = str(value)

    return attrs


def load_h5_sample(entry: H5Entry) -> dict[str, Any]:
    """
    Load one sample from an HDF5 metric export.

    Required datasets:
        img
        pred_lambda
        gt_support

    Optional datasets, old single-depth export:
        pred_depth
        pred_velocity
        gt_depth
        gt_velocity

    Optional datasets, new two-depth export:
        pred_depth_d0
        pred_depth_near
        pred_depth_far
        pred_depth_delta
        pred_depth_best
        pred_depth_best_abs_error
        pred_depth_best_choice

    Returns:
        sample dict with:
            img
            pred_lambda
            gt_support

            pred_depth
            pred_depth_d0
            pred_depth_near
            pred_depth_far
            pred_depth_delta
            pred_depth_best
            pred_depth_best_abs_error
            pred_depth_best_choice

            pred_velocity
            gt_depth
            gt_velocity

            attrs
            shape
            meta
    """
    with h5py.File(entry.h5_path, "r") as f:
        if entry.sample_key not in f:
            raise KeyError(
                f"Sample key {entry.sample_key!r} not found in {entry.h5_path}"
            )

        grp = f[entry.sample_key]

        required = ["img", "pred_lambda", "gt_support"]
        missing = [name for name in required if name not in grp]

        if missing:
            raise KeyError(
                f"Missing required dataset(s) {missing} in "
                f"{entry.h5_path}/{entry.sample_key}"
            )

        # Required datasets.
        img = grp["img"][:]
        pred_lambda = grp["pred_lambda"][:].astype(np.float64, copy=False)
        gt_support = grp["gt_support"][:]

        # Old single-depth field.
        pred_depth = _read_optional_dataset(grp, "pred_depth")

        # New two-depth fields.
        pred_depth_d0 = _read_optional_dataset(grp, "pred_depth_d0")
        pred_depth_near = _read_optional_dataset(grp, "pred_depth_near")
        pred_depth_far = _read_optional_dataset(grp, "pred_depth_far")
        pred_depth_delta = _read_optional_dataset(grp, "pred_depth_delta")
        pred_depth_best = _read_optional_dataset(grp, "pred_depth_best")
        pred_depth_best_abs_error = _read_optional_dataset(
            grp,
            "pred_depth_best_abs_error",
        )
        pred_depth_best_choice = _read_optional_dataset(
            grp,
            "pred_depth_best_choice",
        )

        # Mark fields.
        pred_velocity = _read_optional_dataset(grp, "pred_velocity")
        gt_depth = _read_optional_dataset(grp, "gt_depth")
        gt_velocity = _read_optional_dataset(grp, "gt_velocity")

        attrs = _read_attrs(grp)

    if pred_lambda.ndim != 2:
        raise ValueError(
            f"pred_lambda must be 2D, got shape {pred_lambda.shape} "
            f"for {entry.h5_path}/{entry.sample_key}"
        )

    gt_support_arr = np.asarray(gt_support)

    if gt_support_arr.shape != pred_lambda.shape:
        raise ValueError(
            "pred_lambda and gt_support must have the same shape, "
            f"got {pred_lambda.shape} and {gt_support_arr.shape} "
            f"for {entry.h5_path}/{entry.sample_key}"
        )

    H, W = pred_lambda.shape

    sample: dict[str, Any] = {
        "img": img,
        "pred_lambda": pred_lambda,
        "gt_support": gt_support_arr,

        # Old single-depth prediction.
        # This is usually present for old 3-channel exports.
        "pred_depth": pred_depth,

        # New two-depth predictions.
        # These are None for old exports.
        "pred_depth_d0": pred_depth_d0,
        "pred_depth_near": pred_depth_near,
        "pred_depth_far": pred_depth_far,
        "pred_depth_delta": pred_depth_delta,
        "pred_depth_best": pred_depth_best,
        "pred_depth_best_abs_error": pred_depth_best_abs_error,
        "pred_depth_best_choice": pred_depth_best_choice,

        # Velocity and GT fields.
        "pred_velocity": pred_velocity,
        "gt_depth": gt_depth,
        "gt_velocity": gt_velocity,

        "attrs": attrs,
        "shape": {
            "H": int(H),
            "W": int(W),
            "img_area": int(H * W),
        },
        "meta": {
            "model": entry.model,
            "h5_file": entry.rank_file,
            "sample_key": entry.sample_key,
            "sample_id": entry.sample_id,
            "h5_path": entry.h5_path,
        },
    }

    return sample



# """
# HDF5 IO tools for offline PPP metric evaluation.

# This file contains no metric logic.

# Responsibilities:
#     - find rank*.h5 files
#     - list sample groups
#     - load one sample
#     - attach basic PPP diagnostics:
#         H, W, expected_count, gt_count
# """

# from __future__ import annotations

# import glob
# import os
# from dataclasses import dataclass
# from pathlib import Path
# from typing import Any

# import h5py
# import numpy as np


# @dataclass(frozen=True)
# class H5Entry:
#     """
#     One sample entry inside one HDF5 rank file.
#     """

#     model: str
#     h5_path: str
#     sample_key: str

#     @property
#     def rank_file(self) -> str:
#         return os.path.basename(self.h5_path)

#     @property
#     def sample_id(self) -> str:
#         return f"{self.rank_file}/{self.sample_key}"


# def collect_h5_files(h5_dir: str | os.PathLike[str]) -> list[str]:
#     """
#     Return sorted rank*.h5 files from a metric export directory.
#     """
#     h5_dir = str(Path(h5_dir).expanduser())

#     if not os.path.isdir(h5_dir):
#         raise FileNotFoundError(f"H5 directory not found: {h5_dir}")

#     files = sorted(glob.glob(os.path.join(h5_dir, "rank*.h5")))

#     if not files:
#         raise FileNotFoundError(f"No rank*.h5 files found in: {h5_dir}")

#     return files


# def list_h5_samples_for_model(
#     model_name: str,
#     h5_dir: str | os.PathLike[str],
# ) -> list[H5Entry]:
#     """
#     Return all sample entries for one model run.

#     Each entry identifies:
#         model name
#         rank file path
#         sample group key
#     """
#     entries: list[H5Entry] = []

#     for h5_path in collect_h5_files(h5_dir):
#         with h5py.File(h5_path, "r") as f:
#             for sample_key in sorted(f.keys()):
#                 entries.append(
#                     H5Entry(
#                         model=model_name,
#                         h5_path=h5_path,
#                         sample_key=sample_key,
#                     )
#                 )

#     return entries


# def list_h5_samples(
#     h5_dirs: dict[str, str | os.PathLike[str]],
# ) -> list[H5Entry]:
#     """
#     Return all sample entries across all configured model runs.
#     """
#     entries: list[H5Entry] = []

#     for model_name, h5_dir in h5_dirs.items():
#         entries.extend(
#             list_h5_samples_for_model(
#                 model_name=model_name,
#                 h5_dir=h5_dir,
#             )
#         )

#     return entries


# def _read_optional_dataset(
#     grp: h5py.Group,
#     name: str,
# ) -> np.ndarray | None:
#     """
#     Read optional HDF5 dataset if present.
#     """
#     if name not in grp:
#         return None
#     return grp[name][:]


# def _read_attrs(grp: h5py.Group) -> dict[str, Any]:
#     """
#     Convert HDF5 group attributes to a normal dict.
#     """
#     attrs: dict[str, Any] = {}

#     for key, value in grp.attrs.items():
#         try:
#             if isinstance(value, bytes):
#                 attrs[key] = value.decode("utf-8")
#             else:
#                 attrs[key] = value
#         except Exception:
#             attrs[key] = str(value)

#     return attrs


# def load_h5_sample(entry: H5Entry) -> dict[str, Any]:
#     """
#     Load one sample from an HDF5 metric export.

#     Required datasets:
#         img
#         pred_lambda
#         gt_support

#     Optional datasets:
#         pred_depth
#         pred_velocity
#         gt_depth
#         gt_velocity

#     Returns:
#         sample dict with:
#             img
#             pred_lambda
#             gt_support
#             pred_depth
#             pred_velocity
#             gt_depth
#             gt_velocity
#             attrs
#             ppp
#             meta
#     """
#     with h5py.File(entry.h5_path, "r") as f:
#         if entry.sample_key not in f:
#             raise KeyError(
#                 f"Sample key {entry.sample_key!r} not found in {entry.h5_path}"
#             )

#         grp = f[entry.sample_key]

#         required = ["img", "pred_lambda", "gt_support"]
#         missing = [name for name in required if name not in grp]

#         if missing:
#             raise KeyError(
#                 f"Missing required dataset(s) {missing} in "
#                 f"{entry.h5_path}/{entry.sample_key}"
#             )

#         img = grp["img"][:]
#         pred_lambda = grp["pred_lambda"][:].astype(np.float64, copy=False)
#         gt_support = grp["gt_support"][:]

#         pred_depth = _read_optional_dataset(grp, "pred_depth")
#         pred_velocity = _read_optional_dataset(grp, "pred_velocity")
#         gt_depth = _read_optional_dataset(grp, "gt_depth")
#         gt_velocity = _read_optional_dataset(grp, "gt_velocity")

#         attrs = _read_attrs(grp)

#     if pred_lambda.ndim != 2:
#         raise ValueError(
#             f"pred_lambda must be 2D, got shape {pred_lambda.shape} "
#             f"for {entry.h5_path}/{entry.sample_key}"
#         )

#     gt_support_arr = np.asarray(gt_support)

#     if gt_support_arr.shape != pred_lambda.shape:
#         raise ValueError(
#             "pred_lambda and gt_support must have the same shape, "
#             f"got {pred_lambda.shape} and {gt_support_arr.shape} "
#             f"for {entry.h5_path}/{entry.sample_key}"
#         )

#     H, W = pred_lambda.shape

#     sample: dict[str, Any] = {
#         "img": img,
#         "pred_lambda": pred_lambda,
#         "gt_support": gt_support_arr,
#         "pred_depth": pred_depth,
#         "pred_velocity": pred_velocity,
#         "gt_depth": gt_depth,
#         "gt_velocity": gt_velocity,
#         "attrs": attrs,
#         "shape": {
#             "H": int(H),
#             "W": int(W),
#             "img_area": int(H * W),
#         },
#         "meta": {
#             "model": entry.model,
#             "h5_file": entry.rank_file,
#             "sample_key": entry.sample_key,
#             "sample_id": entry.sample_id,
#             "h5_path": entry.h5_path,
#         },
#     }

#     return sample