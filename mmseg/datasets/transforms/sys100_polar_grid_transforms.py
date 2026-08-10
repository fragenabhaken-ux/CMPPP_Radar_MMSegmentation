"""Dense polar-grid targets for direct camera-to-polar PPP prediction.

This module is intended to be used together with the existing polar point
transforms:

    LoadSys100PolarSample
    PPPPolarRandomFlip
    PPPPolarFilterDomain

The existing polar point pipeline keeps the radar detections at their exact
continuous ``(r, theta)`` locations for the LIIF model.  The transforms below
instead discretize those points onto a fixed polar quadrature grid and pack a
three-channel dense target:

    channel 0: binary radar-point indicator
    channel 1: signed wrapped radial velocity
    channel 2: RCS

At most one target point is stored in each polar grid cell.  If multiple raw
points fall into the same cell, the point closest to the cell centre is kept.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
import torch
from mmengine.structures import PixelData
from mmseg.datasets.transforms.loading import BaseTransform
from mmseg.registry import TRANSFORMS
from mmseg.structures import SegDataSample


def _as_numpy_float32(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value.astype(np.float32, copy=False)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy().astype(np.float32, copy=False)
    return np.asarray(value, dtype=np.float32)


def _as_float(value: Any) -> float:
    if hasattr(value, "item"):
        return float(value.item())
    return float(value)


@TRANSFORMS.register_module()
class BuildPPPPolarGrid(BaseTransform):
    """Discretize exact polar radar points onto a fixed dense target grid.

    Expected input fields:
        radar_points_polar: Tensor/array [N, >=3]
            Columns ``r, theta, dynamic_radial_velocity``.
        radar_points_meta: Tensor/array [N, >=3]
            RCS is read from column 2.

    Produced field:
        gt_seg_map: numpy float32 [H_r, W_theta, 3]
            Channel 0 is binary occupancy, channel 1 velocity, channel 2 RCS.

    Grid convention:
        row index    -> range ``r``
        column index -> azimuth ``theta``

    Points on an upper domain boundary are assigned to the final cell by
    clipping the computed index. This keeps the convention compatible with the
    existing inclusive ``PPPPolarFilterDomain`` transform.
    """

    def __init__(
        self,
        grid_shape: Tuple[int, int] = (256, 256),
        r_min: float = 0.0,
        r_max: float = 104.0,
        theta_min: float = -1.5707963267948966,
        theta_max: float = 1.5707963267948966,
        collision_policy: str = "nearest_cell_center",
        debug_checks: bool = False,
    ) -> None:
        self.grid_shape = tuple(int(v) for v in grid_shape)
        if len(self.grid_shape) != 2:
            raise ValueError(
                f"grid_shape must have length 2, got {self.grid_shape}."
            )

        self.h_r, self.w_theta = self.grid_shape
        if self.h_r <= 0 or self.w_theta <= 0:
            raise ValueError("grid_shape must contain positive dimensions.")

        self.r_min = float(r_min)
        self.r_max = float(r_max)
        self.theta_min = float(theta_min)
        self.theta_max = float(theta_max)

        if self.r_max <= self.r_min:
            raise ValueError("r_max must be larger than r_min.")
        if self.theta_max <= self.theta_min:
            raise ValueError("theta_max must be larger than theta_min.")

        self.dr = (self.r_max - self.r_min) / self.h_r
        self.dtheta = (self.theta_max - self.theta_min) / self.w_theta

        self.collision_policy = str(collision_policy).lower()
        if self.collision_policy != "nearest_cell_center":
            raise ValueError(
                "Only collision_policy='nearest_cell_center' is supported."
            )

        self.debug_checks = bool(debug_checks)

    def _empty_target(self) -> np.ndarray:
        return np.zeros((self.h_r, self.w_theta, 3), dtype=np.float32)

    def transform(self, results: Dict[str, Any]) -> Dict[str, Any]:
        if "radar_points_polar" not in results:
            raise KeyError("Missing results['radar_points_polar'].")
        if "radar_points_meta" not in results:
            raise KeyError("Missing results['radar_points_meta'].")

        polar = _as_numpy_float32(results["radar_points_polar"])
        meta = _as_numpy_float32(results["radar_points_meta"])

        if polar.ndim != 2 or polar.shape[1] < 3:
            raise ValueError(
                f"Expected radar_points_polar [N, >=3], got {polar.shape}."
            )
        if meta.ndim != 2 or meta.shape[0] != polar.shape[0]:
            raise ValueError(
                "radar_points_meta must have the same point count as "
                "radar_points_polar."
            )
        if meta.shape[1] <= 2:
            raise ValueError(
                "Expected RCS in radar_points_meta[:, 2], "
                f"but got shape {meta.shape}."
            )

        gt = self._empty_target()
        num_input_points = int(polar.shape[0])

        # Exact continuous points are retained for camera-domain visualization.
        # polar_gt_points contains every finite point inside the configured
        # domain. polar_target_points contains the subset that survives the
        # one-point-per-cell collision policy used by the dense target.
        polar_gt_points = np.zeros((0, 4), dtype=np.float32)
        polar_target_points = np.zeros((0, 4), dtype=np.float32)

        if num_input_points == 0:
            num_in_domain = 0
            num_target_points = 0
            collision_count = 0
        else:
            r = polar[:, 0]
            theta = polar[:, 1]
            velocity = polar[:, 2]
            rcs = meta[:, 2]

            finite = (
                np.isfinite(r)
                & np.isfinite(theta)
                & np.isfinite(velocity)
                & np.isfinite(rcs)
            )
            in_domain = (
                finite
                & (r >= self.r_min)
                & (r <= self.r_max)
                & (theta >= self.theta_min)
                & (theta <= self.theta_max)
            )

            r = r[in_domain]
            theta = theta[in_domain]
            velocity = velocity[in_domain]
            rcs = rcs[in_domain]
            num_in_domain = int(r.shape[0])

            if num_in_domain > 0:
                polar_gt_points = np.stack(
                    (r, theta, velocity, rcs),
                    axis=1,
                ).astype(np.float32, copy=False)

            if num_in_domain == 0:
                num_target_points = 0
                collision_count = 0
            else:
                r_idx = np.floor((r - self.r_min) / self.dr).astype(np.int64)
                theta_idx = np.floor(
                    (theta - self.theta_min) / self.dtheta
                ).astype(np.int64)

                # Exact upper-bound points produce index H or W. Assign them to
                # the final quadrature cell.
                r_idx = np.clip(r_idx, 0, self.h_r - 1)
                theta_idx = np.clip(theta_idx, 0, self.w_theta - 1)

                centre_r = self.r_min + (r_idx + 0.5) * self.dr
                centre_theta = (
                    self.theta_min + (theta_idx + 0.5) * self.dtheta
                )
                distance_to_centre_sq = (
                    ((r - centre_r) / self.dr) ** 2
                    + ((theta - centre_theta) / self.dtheta) ** 2
                )

                flat_idx = r_idx * self.w_theta + theta_idx
                original_idx = np.arange(num_in_domain, dtype=np.int64)

                # lexsort uses the last key as the primary key. Within each
                # cell, the point nearest to the cell centre comes first. The
                # original index is a deterministic final tie breaker.
                order = np.lexsort(
                    (original_idx, distance_to_centre_sq, flat_idx)
                )
                sorted_flat = flat_idx[order]
                first_in_cell = np.ones(order.shape[0], dtype=bool)
                first_in_cell[1:] = sorted_flat[1:] != sorted_flat[:-1]
                keep = order[first_in_cell]

                kept_r_idx = r_idx[keep]
                kept_theta_idx = theta_idx[keep]

                gt[kept_r_idx, kept_theta_idx, 0] = 1.0
                gt[kept_r_idx, kept_theta_idx, 1] = velocity[keep]
                gt[kept_r_idx, kept_theta_idx, 2] = rcs[keep]

                polar_target_points = np.stack(
                    (r[keep], theta[keep], velocity[keep], rcs[keep]),
                    axis=1,
                ).astype(np.float32, copy=False)

                num_target_points = int(keep.shape[0])
                collision_count = num_in_domain - num_target_points

        results["gt_seg_map"] = gt
        results["seg_fields"] = ["gt_seg_map"]
        results["polar_gt_points"] = polar_gt_points
        results["polar_target_points"] = polar_target_points
        results["polar_point_columns"] = ("r", "theta", "velocity", "rcs")

        results["polar_grid_shape"] = self.grid_shape
        results["polar_r_min"] = self.r_min
        results["polar_r_max"] = self.r_max
        results["polar_theta_min"] = self.theta_min
        results["polar_theta_max"] = self.theta_max
        results["polar_dr"] = self.dr
        results["polar_dtheta"] = self.dtheta
        results["polar_num_input_points"] = num_input_points
        results["polar_num_in_domain_points"] = num_in_domain
        results["polar_num_target_points"] = num_target_points
        results["polar_collision_count"] = collision_count

        metainfo = results.setdefault("metainfo", {})
        metainfo.update(
            dict(
                polar_grid_shape=self.grid_shape,
                polar_r_min=self.r_min,
                polar_r_max=self.r_max,
                polar_theta_min=self.theta_min,
                polar_theta_max=self.theta_max,
                polar_dr=self.dr,
                polar_dtheta=self.dtheta,
                polar_num_input_points=num_input_points,
                polar_num_in_domain_points=num_in_domain,
                polar_num_target_points=num_target_points,
                polar_collision_count=collision_count,
                polar_gt_points=polar_gt_points.copy(),
                polar_target_points=polar_target_points.copy(),
                polar_point_columns=("r", "theta", "velocity", "rcs"),
            )
        )

        if self.debug_checks:
            occupancy = gt[..., 0]
            unique = np.unique(occupancy)
            if not np.all(np.isin(unique, [0.0, 1.0])):
                raise AssertionError(
                    f"Polar occupancy is not binary: {unique.tolist()}"
                )
            occupied = occupancy > 0
            if int(occupied.sum()) != num_target_points:
                raise AssertionError(
                    "Occupied-cell count does not match polar_num_target_points."
                )
            if polar_gt_points.shape != (num_in_domain, 4):
                raise AssertionError(
                    "polar_gt_points does not match the in-domain point count."
                )
            if polar_target_points.shape != (num_target_points, 4):
                raise AssertionError(
                    "polar_target_points does not match the target point count."
                )
            if np.any(gt[..., 1][~occupied] != 0.0):
                raise AssertionError("Velocity is nonzero in an empty cell.")
            if np.any(gt[..., 2][~occupied] != 0.0):
                raise AssertionError("RCS is nonzero in an empty cell.")

        return results


@TRANSFORMS.register_module()
class PPPPackPolarGridInputs(BaseTransform):
    """Pack a camera image and dense polar grid into a ``SegDataSample``.

    The camera input and polar target deliberately have different spatial
    meanings and may have different shapes. No resizing of the target is done.
    """

    def __init__(
        self,
        debug_checks: bool = False,
        rel_x_centre: float = 0.491,
        rel_y_centre: float = 0.4370,
    ) -> None:
        self.debug_checks = bool(debug_checks)
        self.rel_x_centre = float(rel_x_centre)
        self.rel_y_centre = float(rel_y_centre)

    def transform(self, results: Dict[str, Any]) -> Dict[str, Any]:
        image = results["img"]
        if isinstance(image, np.ndarray):
            if image.ndim != 3:
                raise ValueError(
                    f"Expected HWC image array, got shape {image.shape}."
                )
            image = torch.from_numpy(
                np.ascontiguousarray(image)
            ).permute(2, 0, 1)
        elif not isinstance(image, torch.Tensor):
            raise TypeError(f"Unsupported image type: {type(image)}")
        image = image.contiguous().float()

        gt = results["gt_seg_map"]
        if isinstance(gt, np.ndarray):
            if gt.ndim != 3 or gt.shape[2] != 3:
                raise ValueError(
                    "Expected dense polar target [H_r, W_theta, 3], "
                    f"got {gt.shape}."
                )
            gt = torch.from_numpy(
                np.ascontiguousarray(gt)
            ).permute(2, 0, 1)
        elif isinstance(gt, torch.Tensor):
            if gt.ndim != 3:
                raise ValueError(
                    f"Expected three-dimensional target, got {tuple(gt.shape)}."
                )
            if gt.shape[0] != 3 and gt.shape[-1] == 3:
                gt = gt.permute(2, 0, 1)
            if gt.shape[0] != 3:
                raise ValueError(
                    "Expected dense polar target with three channels, "
                    f"got {tuple(gt.shape)}."
                )
        else:
            raise TypeError(f"Unsupported gt_seg_map type: {type(gt)}")
        gt = gt.contiguous().float()

        doppler_period = _as_float(results["dopplerRange"])
        if not np.isfinite(doppler_period) or doppler_period <= 0.0:
            raise ValueError(f"Invalid Doppler period: {doppler_period}")

        metainfo = dict(results.get("metainfo", {}))
        metainfo.update(
            dict(
                img_shape=results.get("img_shape"),
                ori_shape=results.get("ori_shape"),
                doppler_period=doppler_period,
                polar_grid_shape=tuple(results["polar_grid_shape"]),
                polar_r_min=float(results["polar_r_min"]),
                polar_r_max=float(results["polar_r_max"]),
                polar_theta_min=float(results["polar_theta_min"]),
                polar_theta_max=float(results["polar_theta_max"]),
                polar_dr=float(results["polar_dr"]),
                polar_dtheta=float(results["polar_dtheta"]),
                polar_num_input_points=int(
                    results["polar_num_input_points"]
                ),
                polar_num_in_domain_points=int(
                    results["polar_num_in_domain_points"]
                ),
                polar_num_target_points=int(
                    results["polar_num_target_points"]
                ),
                polar_collision_count=int(
                    results["polar_collision_count"]
                ),
                camera_rel_x_centre=self.rel_x_centre,
                camera_rel_y_centre=self.rel_y_centre,
            )
        )

        for key in ("polar_gt_points", "polar_target_points"):
            if key not in results:
                raise KeyError(
                    f"Missing results[{key!r}]. Use BuildPPPPolarGrid before "
                    "PPPPackPolarGridInputs."
                )
            points = _as_numpy_float32(results[key])
            if points.ndim != 2 or points.shape[1] != 4:
                raise ValueError(
                    f"Expected results[{key!r}] with shape [N, 4], "
                    f"got {points.shape}."
                )
            metainfo[key] = points.copy()

        metainfo["polar_point_columns"] = tuple(
            results.get(
                "polar_point_columns",
                ("r", "theta", "velocity", "rcs"),
            )
        )

        if "fovs" in results:
            metainfo["fovs"] = tuple(float(v) for v in results["fovs"])
        if "rad2cam" in results:
            metainfo["rad2cam"] = _as_numpy_float32(
                results["rad2cam"]
            ).copy()
        if "real_hw" in results:
            metainfo["real_hw"] = _as_numpy_float32(
                results["real_hw"]
            ).copy()
        if "idx" in results:
            metainfo.setdefault("global_idx", int(results["idx"]))
            metainfo.setdefault("frame_idx", int(results["idx"]))

        for key in (
            "prefetch_unseen",
            "prefetch_unseen_count",
            "prefetch_idle_time",
        ):
            if key in results:
                value = results[key]
                metainfo[key] = (
                    value.item() if hasattr(value, "item") else value
                )

        data_sample = SegDataSample()
        data_sample.set_metainfo(metainfo)
        data_sample.gt_sem_seg = PixelData(data=gt)

        if self.debug_checks:
            unique = torch.unique(gt[0])
            if not torch.all((unique == 0) | (unique == 1)):
                raise AssertionError(
                    "Polar occupancy channel is not binary. "
                    f"Found {unique.tolist()}."
                )
            expected_shape = tuple(metainfo["polar_grid_shape"])
            if tuple(gt.shape[-2:]) != expected_shape:
                raise AssertionError(
                    f"Target shape {tuple(gt.shape[-2:])} does not match "
                    f"polar_grid_shape {expected_shape}."
                )
            if metainfo["polar_gt_points"].shape[0] != int(
                metainfo["polar_num_in_domain_points"]
            ):
                raise AssertionError(
                    "Packed exact GT-point count does not match metadata."
                )
            if metainfo["polar_target_points"].shape[0] != int(
                metainfo["polar_num_target_points"]
            ):
                raise AssertionError(
                    "Packed target-point count does not match metadata."
                )

        return dict(inputs=image, data_samples=data_sample)