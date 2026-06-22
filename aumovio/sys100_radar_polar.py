from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.floating]


# Raw SYS100 RadarFrame channel convention, before radar_transform.
#
# This follows the convention used by the current visualization code:
#   radar_clusters[:, 4] -> RCS
#   radar_clusters[:, 5] - radar_clusters[:, 6] -> dynamic radial velocity
#
# Raw grid channels:
#   0: x / depth offset inside grid cell
#   1: y / lateral offset inside grid cell
#   2: rangeGateLength, also used as presence signal
#   3: azimuth1
#   4: Radar Cross Section strength, RCS
#   5: relative radial velocity
#   6: expected static relative radial velocity
RADAR_X_OFFSET = 0
RADAR_Y_OFFSET = 1
RADAR_RANGE_GATE_LENGTH = 2
RADAR_AZIMUTH = 3
RADAR_RCS = 4
RADAR_REL_RADIAL_VELOCITY = 5
RADAR_STATIC_REL_RADIAL_VELOCITY = 6


XY_POINT_COLUMNS = (
    "x_forward",
    "y_lateral",
    "rangeGateLength",
    "azimuth1",
    "rcs",
    "relative_radial_velocity",
    "expected_static_relative_radial_velocity",
)

POLAR_POINT_COLUMNS = (
    "range",
    "azimuth_from_xy",
    "dynamic_radial_velocity",
)

POLAR_META_COLUMNS = (
    "rangeGateLength",
    "azimuth1",
    "rcs",
    "relative_radial_velocity",
    "expected_static_relative_radial_velocity",
    "azimuth_from_xy_minus_azimuth1",
)


def _as_hwc(
    grid: FloatArray,
    *,
    inorder: Literal["H W C", "C H W", "HWC", "CHW"] = "H W C",
) -> FloatArray:
    """Return radar grid in HWC format.

    The raw SYS100 RadarFrame should usually already be HWC:
        (64, 64, 7)

    The old model-facing radar tensor is often CHW:
        (7, 64, 64)

    This helper is intentionally small and explicit.
    """
    if grid.ndim != 3:
        raise ValueError(f"Expected radar grid with 3 dimensions, got shape {grid.shape}.")

    if inorder in ("H W C", "HWC"):
        return grid

    if inorder in ("C H W", "CHW"):
        return np.transpose(grid, (1, 2, 0))

    raise ValueError(f"Invalid inorder={inorder!r}.")


def wrap_angle_to_pi(angle: FloatArray) -> NDArray[np.float32]:
    """Wrap angles to [-pi, pi)."""
    wrapped = (angle + np.pi) % (2.0 * np.pi) - np.pi
    return wrapped.astype(np.float32, copy=False)


def unwrap_doppler_velocity(
    velocity: FloatArray,
    doppler_range: float | np.ndarray | None,
    *,
    enabled: bool = True,
) -> NDArray[np.float32]:
    """Unwrap / fold radial velocity with the same convention as existing vis code.

    Existing code uses:

        vel -= doppler * floor(vel / doppler + 1/2)

    This maps the velocity to the centered interval induced by doppler_range.

    Parameters
    ----------
    velocity:
        Dynamic radial velocity before wrapping correction.
    doppler_range:
        Usually row_data["RadarClusterHead"]["f_AmbFreeDopplerRange"].
        If None, non-positive, or not finite, the velocity is returned unchanged.
    enabled:
        Allows switching this off for debugging.
    """
    velocity = np.asarray(velocity, dtype=np.float32)

    if not enabled or doppler_range is None:
        return velocity.astype(np.float32, copy=False)

    doppler = np.asarray(doppler_range, dtype=np.float32)

    if np.any(~np.isfinite(doppler)) or np.any(doppler <= 0):
        return velocity.astype(np.float32, copy=False)

    unwrapped = velocity - doppler * np.floor(velocity / doppler + 0.5)
    return unwrapped.astype(np.float32, copy=False)


def radar_grid_to_xy_points(
    grid: FloatArray,
    *,
    xmin: float = 0.0,
    ymin: float = -100.0,
    ymax: float = 100.0,
    x_scale: float = 256.0,
    thres: float = 0.1,
    inorder: Literal["H W C", "C H W", "HWC", "CHW"] = "H W C",
) -> NDArray[np.float32]:
    """Decode a discretized SYS100 radar grid into metric x/y point coordinates.

    This is a cleaned-up version of point_from_gridentry_vectorized.

    Input
    -----
    grid:
        RadarFrame-like array with shape (H, W, C), usually (64, 64, 7).
        Pass the raw, orientation-corrected radar_data before radar_transform.

    Output
    ------
    xy_points:
        Array with shape (N, 7).

        Columns:
            0: x_forward
            1: y_lateral
            2: rangeGateLength
            3: azimuth1
            4: RCS
            5: relative radial velocity
            6: expected static relative radial velocity
    """
    grid_hwc = _as_hwc(grid, inorder=inorder).astype(np.float32, copy=False)

    if grid_hwc.shape[-1] < 7:
        raise ValueError(
            "Expected at least 7 radar channels. "
            f"Got shape {grid_hwc.shape}."
        )

    h_steps, w_steps, feature_dim = grid_hwc.shape

    row_idx, col_idx = np.where(grid_hwc[..., RADAR_RANGE_GATE_LENGTH] > thres)

    if row_idx.size == 0:
        return np.zeros((0, feature_dim), dtype=np.float32)

    entries = grid_hwc[row_idx, col_idx, :]

    range_gate_length = entries[:, RADAR_RANGE_GATE_LENGTH]
    xmax = range_gate_length * x_scale

    x_forward = (
        (entries[:, RADAR_X_OFFSET] + (row_idx.astype(np.float32) + 0.5))
        * (xmax - xmin)
        / float(h_steps)
        + xmin
    )

    y_lateral = (
        (entries[:, RADAR_Y_OFFSET] + (col_idx.astype(np.float32) + 0.5))
        * (ymax - ymin)
        / float(w_steps)
        + ymin
    )

    out = np.empty((entries.shape[0], feature_dim), dtype=np.float32)
    out[:, 0] = x_forward
    out[:, 1] = y_lateral
    out[:, 2:] = entries[:, 2:]

    return out


def xy_points_to_polar(
    xy_points: FloatArray,
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Convert decoded x/y radar points to polar coordinates.

    Input
    -----
    xy_points:
        Shape (N, >=2), with columns:
            0: x_forward
            1: y_lateral

    Output
    ------
    r:
        Shape (N,), metric range in the BEV plane.
    theta:
        Shape (N,), azimuth derived from x/y via atan2(y, x).
    """
    xy_points = np.asarray(xy_points, dtype=np.float32)

    if xy_points.ndim != 2 or xy_points.shape[1] < 2:
        raise ValueError(
            "Expected xy_points with shape (N, >=2), "
            f"got {xy_points.shape}."
        )

    x = xy_points[:, 0]
    y = xy_points[:, 1]

    r = np.sqrt(x * x + y * y)
    theta = np.arctan2(y, x)

    return r.astype(np.float32, copy=False), theta.astype(np.float32, copy=False)


def dynamic_radial_velocity_from_xy_points(
    xy_points: FloatArray,
    *,
    doppler_range: float | np.ndarray | None = None,
    unwrap: bool = True,
) -> NDArray[np.float32]:
    """Compute dynamic radial velocity mark.

    The raw relative radial velocity contains ego-motion effects.
    The expected static relative radial velocity is subtracted:

        dynamic_velocity = rel_velocity - static_rel_velocity

    Then the result is Doppler-unwrapped with the existing code convention.
    """
    xy_points = np.asarray(xy_points, dtype=np.float32)

    if xy_points.ndim != 2 or xy_points.shape[1] <= RADAR_STATIC_REL_RADIAL_VELOCITY:
        raise ValueError(
            "Expected xy_points with shape (N, >=7), "
            f"got {xy_points.shape}."
        )

    rel_vel = xy_points[:, RADAR_REL_RADIAL_VELOCITY]
    static_rel_vel = xy_points[:, RADAR_STATIC_REL_RADIAL_VELOCITY]

    dynamic_vel = rel_vel - static_rel_vel

    return unwrap_doppler_velocity(
        dynamic_vel,
        doppler_range=doppler_range,
        enabled=unwrap,
    )


def build_polar_outputs_from_xy_points(
    xy_points: FloatArray,
    *,
    doppler_range: float | np.ndarray | None = None,
    unwrap_velocity: bool = True,
    filter_invalid: bool = True,
    r_min: float | None = None,
    r_max: float | None = None,
    theta_min: float | None = None,
    theta_max: float | None = None,
) -> dict[str, NDArray[np.float32]]:
    """Build polar point tensors from decoded x/y points.

    Returns a dictionary with the fields expected by the new polar PPP pipeline.

    Returned fields
    ---------------
    radar_points_polar:
        Shape (N, 3), columns:
            0: range r
            1: theta derived from x/y
            2: dynamic radial velocity

    radar_points_xy:
        Shape (N, 2), columns:
            0: x_forward
            1: y_lateral

    radar_points_meta:
        Shape (N, 6), columns:
            0: rangeGateLength
            1: azimuth1
            2: RCS
            3: relative radial velocity
            4: expected static relative radial velocity
            5: theta_from_xy - azimuth1, wrapped to [-pi, pi)
    """
    xy_points = np.asarray(xy_points, dtype=np.float32)

    if xy_points.ndim != 2 or xy_points.shape[1] < 7:
        raise ValueError(
            "Expected xy_points with shape (N, >=7), "
            f"got {xy_points.shape}."
        )

    if xy_points.shape[0] == 0:
        return {
            "radar_points_polar": np.zeros((0, 3), dtype=np.float32),
            "radar_points_xy": np.zeros((0, 2), dtype=np.float32),
            "radar_points_meta": np.zeros((0, 6), dtype=np.float32),
        }

    r, theta = xy_points_to_polar(xy_points)
    dynamic_velocity = dynamic_radial_velocity_from_xy_points(
        xy_points,
        doppler_range=doppler_range,
        unwrap=unwrap_velocity,
    )

    azimuth1 = xy_points[:, RADAR_AZIMUTH]
    theta_minus_azimuth1 = wrap_angle_to_pi(theta - azimuth1)

    radar_points_polar = np.stack(
        [r, theta, dynamic_velocity],
        axis=1,
    ).astype(np.float32, copy=False)

    radar_points_xy = xy_points[:, :2].astype(np.float32, copy=False)

    radar_points_meta = np.stack(
        [
            xy_points[:, RADAR_RANGE_GATE_LENGTH],
            xy_points[:, RADAR_AZIMUTH],
            xy_points[:, RADAR_RCS],
            xy_points[:, RADAR_REL_RADIAL_VELOCITY],
            xy_points[:, RADAR_STATIC_REL_RADIAL_VELOCITY],
            theta_minus_azimuth1,
        ],
        axis=1,
    ).astype(np.float32, copy=False)

    mask = np.ones((radar_points_polar.shape[0],), dtype=bool)

    if filter_invalid:
        mask &= np.isfinite(radar_points_polar).all(axis=1)
        mask &= np.isfinite(radar_points_xy).all(axis=1)
        mask &= np.isfinite(radar_points_meta).all(axis=1)

    if r_min is not None:
        mask &= radar_points_polar[:, 0] >= float(r_min)
    if r_max is not None:
        mask &= radar_points_polar[:, 0] <= float(r_max)
    if theta_min is not None:
        mask &= radar_points_polar[:, 1] >= float(theta_min)
    if theta_max is not None:
        mask &= radar_points_polar[:, 1] <= float(theta_max)

    if not mask.all():
        radar_points_polar = radar_points_polar[mask]
        radar_points_xy = radar_points_xy[mask]
        radar_points_meta = radar_points_meta[mask]

    return {
        "radar_points_polar": radar_points_polar,
        "radar_points_xy": radar_points_xy,
        "radar_points_meta": radar_points_meta,
    }


def radar_grid_to_polar_points(
    grid: FloatArray,
    *,
    doppler_range: float | np.ndarray | None = None,
    xmin: float = 0.0,
    ymin: float = -100.0,
    ymax: float = 100.0,
    x_scale: float = 256.0,
    thres: float = 0.1,
    inorder: Literal["H W C", "C H W", "HWC", "CHW"] = "H W C",
    unwrap_velocity: bool = True,
    filter_invalid: bool = True,
    r_min: float | None = None,
    r_max: float | None = None,
    theta_min: float | None = None,
    theta_max: float | None = None,
    return_xy_points_full: bool = False,
) -> dict[str, NDArray[np.float32]]:
    """Full SYS100 radar-grid-to-polar conversion.

    This is the function to call from Sys100Dataset._getitem.

    Important:
        Pass raw, orientation-corrected radar_data here, before radar_transform.
        Do not pass the normalized/flipped model-facing radar_frame unless you
        intentionally undo that transform first.

    Example
    -------
    >>> polar = radar_grid_to_polar_points(
    ...     radar_data,
    ...     doppler_range=dopplerRange,
    ...     thres=self.threshold_cluster_present,
    ... )
    >>> polar["radar_points_polar"].shape
    (N, 3)
    """
    xy_points = radar_grid_to_xy_points(
        grid,
        xmin=xmin,
        ymin=ymin,
        ymax=ymax,
        x_scale=x_scale,
        thres=thres,
        inorder=inorder,
    )

    out = build_polar_outputs_from_xy_points(
        xy_points,
        doppler_range=doppler_range,
        unwrap_velocity=unwrap_velocity,
        filter_invalid=filter_invalid,
        r_min=r_min,
        r_max=r_max,
        theta_min=theta_min,
        theta_max=theta_max,
    )

    if return_xy_points_full:
        # Useful for debugging exact channel values.
        out["radar_points_xy_full"] = xy_points.astype(np.float32, copy=False)

    return out


def polar_to_xy(
    polar_points: FloatArray,
) -> NDArray[np.float32]:
    """Convert polar points back to x/y.

    Input can be:
        (N, 2): r, theta
        (N, >=2): first two columns are r, theta

    Output:
        (N, 2): x_forward, y_lateral
    """
    polar_points = np.asarray(polar_points, dtype=np.float32)

    if polar_points.ndim != 2 or polar_points.shape[1] < 2:
        raise ValueError(
            "Expected polar_points with shape (N, >=2), "
            f"got {polar_points.shape}."
        )

    r = polar_points[:, 0]
    theta = polar_points[:, 1]

    x = r * np.cos(theta)
    y = r * np.sin(theta)

    return np.stack([x, y], axis=1).astype(np.float32, copy=False)


def make_polar_grid_centers(
    *,
    num_range_bins: int = 64,
    num_azimuth_bins: int = 64,
    r_min: float = 0.0,
    r_max: float = 104.0,
    theta_min: float = -np.pi / 2,
    theta_max: float = np.pi / 2,
) -> NDArray[np.float32]:
    """Create polar grid-cell centers for later PPP integration/debugging.

    Output:
        grid_centers: (num_range_bins, num_azimuth_bins, 2)
        last dimension: r, theta
    """
    if num_range_bins <= 0 or num_azimuth_bins <= 0:
        raise ValueError(
            "num_range_bins and num_azimuth_bins must be positive. "
            f"Got {num_range_bins=} and {num_azimuth_bins=}."
        )

    r_edges = np.linspace(r_min, r_max, num_range_bins + 1, dtype=np.float32)
    theta_edges = np.linspace(theta_min, theta_max, num_azimuth_bins + 1, dtype=np.float32)

    r_centers = 0.5 * (r_edges[:-1] + r_edges[1:])
    theta_centers = 0.5 * (theta_edges[:-1] + theta_edges[1:])

    rr, tt = np.meshgrid(r_centers, theta_centers, indexing="ij")

    return np.stack([rr, tt], axis=-1).astype(np.float32, copy=False)


def polar_bin_area_dr_dtheta(
    *,
    num_range_bins: int = 64,
    num_azimuth_bins: int = 64,
    r_min: float = 0.0,
    r_max: float = 104.0,
    theta_min: float = -np.pi / 2,
    theta_max: float = np.pi / 2,
) -> float:
    """Area element in polar measure dr dtheta.

    Use this if the PPP intensity is defined w.r.t. polar measure.

    For Cartesian-area density, do not use this alone. You then need the
    Jacobian factor r during integration.
    """
    dr = (r_max - r_min) / float(num_range_bins)
    dtheta = (theta_max - theta_min) / float(num_azimuth_bins)
    return float(dr * dtheta)


def cartesian_area_weights_for_polar_grid(
    *,
    num_range_bins: int = 64,
    num_azimuth_bins: int = 64,
    r_min: float = 0.0,
    r_max: float = 104.0,
    theta_min: float = -np.pi / 2,
    theta_max: float = np.pi / 2,
) -> NDArray[np.float32]:
    """Approximate Cartesian area weights for polar grid centers.

    Use this if the network output is interpreted as Cartesian BEV density
    sampled on a polar grid:

        integral lambda_xy(x, y) dx dy
        approx sum lambda_xy(r, theta) * r * dr * dtheta
    """
    centers = make_polar_grid_centers(
        num_range_bins=num_range_bins,
        num_azimuth_bins=num_azimuth_bins,
        r_min=r_min,
        r_max=r_max,
        theta_min=theta_min,
        theta_max=theta_max,
    )

    r = centers[..., 0]
    dr_dtheta = polar_bin_area_dr_dtheta(
        num_range_bins=num_range_bins,
        num_azimuth_bins=num_azimuth_bins,
        r_min=r_min,
        r_max=r_max,
        theta_min=theta_min,
        theta_max=theta_max,
    )

    return (r * dr_dtheta).astype(np.float32, copy=False)