from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import numpy as np

from .types import BEVGeometry


@dataclass(frozen=True)
class MarkedPointCloud:
    xy: np.ndarray
    rcs: np.ndarray
    velocity: np.ndarray

    def __post_init__(self) -> None:
        xy = np.asarray(self.xy)
        rcs = np.asarray(self.rcs).reshape(-1)
        velocity = np.asarray(self.velocity).reshape(-1)
        if xy.ndim != 2 or xy.shape[1] != 2:
            raise ValueError(f"xy must have shape (N, 2), got {xy.shape}.")
        if not (len(xy) == len(rcs) == len(velocity)):
            raise ValueError(
                f"Point-cloud lengths differ: xy={len(xy)}, rcs={len(rcs)}, "
                f"velocity={len(velocity)}."
            )

    @property
    def n_points(self) -> int:
        return int(len(self.xy))

    def as_nx4(self) -> np.ndarray:
        if self.n_points == 0:
            return np.zeros((0, 4), dtype=np.float64)
        return np.column_stack([self.xy, self.rcs, self.velocity]).astype(
            np.float64,
            copy=False,
        )

    def finite(self) -> "MarkedPointCloud":
        mask = (
            np.isfinite(self.xy).all(axis=1)
            & np.isfinite(self.rcs)
            & np.isfinite(self.velocity)
        )
        return MarkedPointCloud(
            xy=np.asarray(self.xy, dtype=np.float64)[mask],
            rcs=np.asarray(self.rcs, dtype=np.float64)[mask],
            velocity=np.asarray(self.velocity, dtype=np.float64)[mask],
        )


def stable_seed(base_seed: int, sample_uid: str, draw_index: int) -> int:
    payload = f"{int(base_seed)}|{sample_uid}|{int(draw_index)}".encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, "little", signed=False)


def _prepare_mass_grid(mass: np.ndarray) -> np.ndarray:
    mass = np.asarray(mass, dtype=np.float64)
    if mass.ndim != 2:
        raise ValueError(f"PPP mass grid must be 2D, got shape {mass.shape}.")
    if not np.all(np.isfinite(mass)):
        raise ValueError("PPP mass grid contains NaN or Inf.")
    if np.any(mass < 0.0):
        raise ValueError("PPP mass grid contains negative values.")
    return mass


def sample_ppp_thinning_grid(
    mass: np.ndarray,
    *,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict[str, int]]:
    """Sample the piecewise-constant PPP by global thinning.

    This is the same algorithm used by ``PPPMarkVisHookLight``. Coordinates are
    returned in continuous grid units as ``(column, row)``. The subsequent BEV
    conversion maps rows to forward x and columns to lateral y.
    """
    mass = _prepare_mass_grid(mass)
    height, width = mass.shape
    mass_max = float(mass.max())

    if mass_max <= 0.0:
        return np.zeros((0, 2), dtype=np.float64), {
            "num_points": 0,
            "num_proposals": 0,
        }

    num_proposals = int(rng.poisson(mass_max * height * width))
    if num_proposals == 0:
        return np.zeros((0, 2), dtype=np.float64), {
            "num_points": 0,
            "num_proposals": 0,
        }

    columns = rng.random(num_proposals) * width
    rows = rng.random(num_proposals) * height
    column_indices = np.clip(columns.astype(np.int64), 0, width - 1)
    row_indices = np.clip(rows.astype(np.int64), 0, height - 1)

    keep = rng.random(num_proposals) < (
        mass[row_indices, column_indices] / mass_max
    )
    points_grid = np.stack([columns[keep], rows[keep]], axis=1).astype(
        np.float64,
        copy=False,
    )
    return points_grid, {
        "num_points": int(len(points_grid)),
        "num_proposals": num_proposals,
    }


def sample_ppp_block_grid(
    mass: np.ndarray,
    *,
    block_size: int = 8,
    placement: str = "weighted",
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict[str, int]]:
    """Sample with the existing block-activation approximation.

    For every block, activation probability is ``1 - exp(-block_mass)``. An
    active block contributes one point. ``weighted`` placement first selects a
    cell proportional to its mass and jitters inside that cell. ``uniform``
    placement samples uniformly over the full active block. Coordinates are
    returned as continuous ``(column, row)`` grid coordinates.
    """
    mass = _prepare_mass_grid(mass)
    block_size = int(block_size)
    placement = str(placement).lower()
    if block_size <= 0:
        raise ValueError("block_size must be > 0.")
    if placement not in {"weighted", "uniform"}:
        raise ValueError("placement must be 'weighted' or 'uniform'.")

    height, width = mass.shape
    n_block_rows = (height + block_size - 1) // block_size
    n_block_cols = (width + block_size - 1) // block_size
    padded_height = n_block_rows * block_size
    padded_width = n_block_cols * block_size

    mass_padded = np.zeros((padded_height, padded_width), dtype=np.float64)
    mass_padded[:height, :width] = mass
    mass_blocks = mass_padded.reshape(
        n_block_rows,
        block_size,
        n_block_cols,
        block_size,
    ).transpose(0, 2, 1, 3)
    block_mass = mass_blocks.sum(axis=(2, 3))
    activation_probability = 1.0 - np.exp(-block_mass)

    active = rng.random((n_block_rows, n_block_cols)) < activation_probability
    block_rows, block_cols = np.nonzero(active)
    num_active = int(len(block_rows))
    if num_active == 0:
        return np.zeros((0, 2), dtype=np.float64), {
            "num_points": 0,
            "num_active_blocks": 0,
        }

    points_grid = np.empty((num_active, 2), dtype=np.float64)
    column_starts = block_cols * block_size
    row_starts = block_rows * block_size
    block_widths = np.minimum(block_size, width - column_starts)
    block_heights = np.minimum(block_size, height - row_starts)

    if placement == "uniform":
        points_grid[:, 0] = (
            column_starts + rng.random(num_active) * block_widths
        )
        points_grid[:, 1] = row_starts + rng.random(num_active) * block_heights
    else:
        for index, (column0, row0, block_width, block_height) in enumerate(
            zip(column_starts, row_starts, block_widths, block_heights)
        ):
            block = mass[
                row0 : row0 + block_height,
                column0 : column0 + block_width,
            ]
            block_sum = float(block.sum())
            if block_sum <= 0.0:
                points_grid[index, 0] = column0 + rng.random() * block_width
                points_grid[index, 1] = row0 + rng.random() * block_height
                continue

            flat_index = int(rng.choice(block.size, p=block.ravel() / block_sum))
            local_row, local_column = divmod(flat_index, int(block_width))
            points_grid[index, 0] = column0 + local_column + rng.random()
            points_grid[index, 1] = row0 + local_row + rng.random()

    return points_grid, {
        "num_points": num_active,
        "num_active_blocks": num_active,
    }


def _subsample_grid_points(
    points_grid: np.ndarray,
    *,
    rng: np.random.Generator,
    max_points: int | None,
) -> np.ndarray:
    if max_points is None or len(points_grid) <= int(max_points):
        return points_grid
    max_points = int(max_points)
    if max_points <= 0:
        raise ValueError("max_points must be positive or None.")
    chosen = rng.choice(len(points_grid), size=max_points, replace=False)
    return points_grid[np.sort(chosen)]


def _grid_points_to_marked_bev(
    sample: dict[str, Any],
    points_grid: np.ndarray,
) -> MarkedPointCloud:
    geometry: BEVGeometry = sample["geometry"]
    points_grid = np.asarray(points_grid, dtype=np.float64)
    if len(points_grid) == 0:
        return MarkedPointCloud(
            xy=np.zeros((0, 2), dtype=np.float64),
            rcs=np.zeros((0,), dtype=np.float64),
            velocity=np.zeros((0,), dtype=np.float64),
        )
    if points_grid.ndim != 2 or points_grid.shape[1] != 2:
        raise ValueError(
            f"points_grid must have shape (N, 2), got {points_grid.shape}."
        )

    columns_float = points_grid[:, 0]
    rows_float = points_grid[:, 1]
    rows = np.clip(rows_float.astype(np.int64), 0, geometry.height - 1)
    columns = np.clip(columns_float.astype(np.int64), 0, geometry.width - 1)

    # The exported raster convention is array[row=x_forward, column=y_lateral].
    x_forward = geometry.x_min + rows_float * geometry.dx
    y_lateral = geometry.y_min + columns_float * geometry.dy
    xy = np.column_stack([x_forward, y_lateral])

    return MarkedPointCloud(
        xy=xy.astype(np.float64, copy=False),
        rcs=np.asarray(sample["pred_rcs"], dtype=np.float64)[rows, columns],
        velocity=np.asarray(sample["pred_velocity"], dtype=np.float64)[rows, columns],
    ).finite()


def sample_predicted_ppp(
    sample: dict[str, Any],
    *,
    rng: np.random.Generator,
    method: str = "thinning",
    block_size: int = 8,
    block_placement: str = "weighted",
    max_points: int | None = None,
) -> MarkedPointCloud:
    """Sample a marked Cartesian-BEV point cloud with an existing PPP sampler."""
    method = str(method).lower()
    mass = np.asarray(sample["pred_lambda"], dtype=np.float64)

    if method == "thinning":
        points_grid, _ = sample_ppp_thinning_grid(mass, rng=rng)
    elif method == "block":
        points_grid, _ = sample_ppp_block_grid(
            mass,
            block_size=block_size,
            placement=block_placement,
            rng=rng,
        )
    else:
        raise ValueError("method must be 'thinning' or 'block'.")

    points_grid = _subsample_grid_points(
        points_grid,
        rng=rng,
        max_points=max_points,
    )
    return _grid_points_to_marked_bev(sample, points_grid)


def ground_truth_point_cloud(sample: dict[str, Any]) -> MarkedPointCloud:
    geometry: BEVGeometry = sample["geometry"]
    if (
        sample.get("gt_points_xy") is not None
        and sample.get("gt_point_velocity") is not None
        and sample.get("gt_point_rcs") is not None
    ):
        xy = np.asarray(sample["gt_points_xy"], dtype=np.float64)
        velocity = np.asarray(sample["gt_point_velocity"], dtype=np.float64).reshape(-1)
        rcs = np.asarray(sample["gt_point_rcs"], dtype=np.float64).reshape(-1)
        _, _, inside = geometry.point_indices(xy)
        cloud = MarkedPointCloud(xy=xy[inside], rcs=rcs[inside], velocity=velocity[inside])
        return cloud.finite()

    support = np.asarray(sample["gt_support"], dtype=np.float64)
    counts = np.rint(support).astype(np.int64)
    if np.max(np.abs(support - counts), initial=0.0) > 1e-5:
        raise ValueError(
            "Cannot reconstruct GT points: gt_support is not integer-valued and "
            "exact point arrays are unavailable."
        )
    rows, cols = np.nonzero(counts > 0)
    repetitions = counts[rows, cols]
    rows = np.repeat(rows, repetitions)
    cols = np.repeat(cols, repetitions)
    return MarkedPointCloud(
        xy=geometry.cell_centres(rows, cols),
        rcs=np.asarray(sample["gt_rcs"], dtype=np.float64)[rows, cols],
        velocity=np.asarray(sample["gt_velocity"], dtype=np.float64)[rows, cols],
    ).finite()
