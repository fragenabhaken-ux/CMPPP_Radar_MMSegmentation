from __future__ import annotations

import numpy as np


def _unwrap_doppler(velocity: np.ndarray, doppler_range: float) -> np.ndarray:
    if not np.isfinite(doppler_range) or doppler_range <= 0.0:
        return velocity
    return velocity - doppler_range * np.floor(velocity / doppler_range + 0.5)


def radar_grid_to_polar_points(
    radar_grid: np.ndarray,
    doppler_range: float,
    thres: float = 0.1,
    xmin: float = 0.0,
    ymin: float = -100.0,
    ymax: float = 100.0,
) -> dict[str, np.ndarray]:
    """Decode one orientation-corrected raw SYS100 radar frame.

    The raw grid is HWC with channels
    ``x_offset, y_offset, range_gate_length, azimuth, rcs, velocity, static_velocity``.
    Only cells whose range-gate-length channel exceeds ``thres`` form points.
    """
    grid = np.asarray(radar_grid)
    if grid.ndim != 3 or grid.shape[-1] < 7:
        raise ValueError(
            'Expected raw radar grid with shape [H, W, >=7], '
            f'got {tuple(grid.shape)}.'
        )

    h, w, _ = grid.shape
    row_idx, col_idx = np.where(grid[..., 2] > float(thres))

    if row_idx.size == 0:
        return dict(
            radar_points_polar=np.zeros((0, 3), dtype=np.float32),
            radar_points_xy=np.zeros((0, 2), dtype=np.float32),
            radar_points_meta=np.zeros((0, 6), dtype=np.float32),
        )

    entries = grid[row_idx, col_idx, :].astype(np.float64, copy=False)

    # This is the existing SYS100 grid decoding used by the image-space PPP
    # pipeline: range_gate_length scales the forward extent of each row.
    x_max = entries[:, 2] * 256.0
    x = (entries[:, 0] + row_idx + 0.5) * (x_max - xmin) / h + xmin
    y = (entries[:, 1] + col_idx + 0.5) * (ymax - ymin) / w + ymin

    r = np.hypot(x, y)
    theta = np.arctan2(y, x)
    velocity = _unwrap_doppler(entries[:, 5] - entries[:, 6], float(doppler_range))

    polar = np.stack([r, theta, velocity], axis=-1).astype(np.float32)
    xy = np.stack([x, y], axis=-1).astype(np.float32)
    meta = np.stack(
        [
            entries[:, 2],
            entries[:, 3],
            entries[:, 4],
            entries[:, 5],
            entries[:, 6],
            np.full(entries.shape[0], float(doppler_range)),
        ],
        axis=-1,
    ).astype(np.float32)

    return dict(
        radar_points_polar=polar,
        radar_points_xy=xy,
        radar_points_meta=meta,
    )
