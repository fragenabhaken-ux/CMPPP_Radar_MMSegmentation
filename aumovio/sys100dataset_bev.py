"""
SYS100 BEV / polar-radar dataset variant.

This file defines a thin modelling-specific wrapper around Sys100Dataset.

Purpose
-------
The original Sys100Dataset remains the general-purpose SYS100 reader. It loads
the RGB image, radar grid, metadata, video frames, calibration, and optional
visualization data.

This subclass is the clean entry point for the new camera-conditioned polar BEV
PPP modelling setup. It enables polar radar point labels by default.

Required base-dataset support
-----------------------------
The base Sys100Dataset must already support:

    return_polar_points: bool = False

and, when enabled, it must add the following fields to the returned sample:

    radar_points_polar: Tensor[N, 3]
        columns: range r, azimuth theta from x/y, dynamic radial velocity

    radar_points_xy: Tensor[N, 2]
        columns: x_forward, y_lateral

    radar_points_meta: Tensor[N, 6]
        debug/meta channels:
        rangeGateLength, azimuth1, RCS, relative radial velocity,
        expected static relative radial velocity, theta_from_xy_minus_azimuth1

Why this class exists
---------------------
Using a separate named dataset avoids ambiguity in configs. Old experiments can
continue to use Sys100Dataset. New polar BEV experiments can use Sys100BEVDataset.

Usage
---------------------
from aumovio.sys100dataset_bev import Sys100BEVDataset

dataset = Sys100BEVDataset(
    source=source_path,
    extract_video=True,
    out_img_size=(544, 1792),
)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .sys100dataset import Sys100Dataset


class Sys100BEVDataset(Sys100Dataset):
    """SYS100 dataset variant for polar BEV PPP modelling.

    The RGB camera image remains the condition. The radar target is returned as
    a variable-length point list in polar BEV coordinates.

    This class does not reimplement file loading, video extraction, image
    processing, radar orientation handling, or metadata handling. All of that
    stays in Sys100Dataset.

    Main returned target fields
    ---------------------------
    radar_points_polar:
        Tensor of shape (N, 3), with columns:
            0: range r
            1: azimuth theta derived from decoded x/y
            2: dynamic radial velocity

    radar_points_xy:
        Tensor of shape (N, 2), with columns:
            0: x_forward
            1: y_lateral

    radar_points_meta:
        Tensor of shape (N, 6), for debugging and analysis.
    """

    def __init__(
        self,
        source: str | Path | list[str] | list[Path] | list[str | Path],
        **kwargs: Any,
    ) -> None:
        # The BEV modelling variant should always return polar point labels.
        # Allow an explicit user value, but default to True.
        kwargs.setdefault("return_polar_points", True)

        super().__init__(
            source=source,
            **kwargs,
        )