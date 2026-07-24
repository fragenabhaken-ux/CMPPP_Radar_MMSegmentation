#!/usr/bin/env python3
"""
Create quantitative PPP EDA summary plots from a scan directory.

Input:
    scan_dir/
        frames.csv
        point_values.npz

Output:
    scan_dir/plots/
        point_count_hist.png
        point_count_bucket_bar.png
        brightness_hist.png
        contrast_hist.png
        laplacian_var_hist.png
        depth_hist.png
        velocity_signed_hist.png
        velocity_abs_hist.png
        velocity_bucket_bar.png
        rcs_hist.png
        rcs_hist_log_y.png
        image_space_density.png
        depth_vs_y_pix.png
        velocity_vs_depth.png
        rcs_vs_depth.png
        rgb_at_radar_points_hist.png
        summary_frames.csv
        summary_points.csv
        plot_summary.json

Example:

python tools/ppp_eda/plot_ppp_eda_summary.py \
    --scan-dir work_dirs/ppp_eda/train_seed42_n500 \
    --overwrite
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ppp_eda_attribute_limits import (
    DEPTH_DISPLAY_RANGE,
    DOPPLER_RANGE_DISPLAY_RANGE,
    RANGE_GATE_LENGTH_DISPLAY_RANGE,
    RCS_DISPLAY_RANGE,
    VELOCITY_DISPLAY_RANGE,
    VELOCITY_MEASURED_RELATIVE_DISPLAY_RANGE,
    VELOCITY_EXPECTED_STATIC_DISPLAY_RANGE,
    VELOCITY_COMPENSATED_PRE_WRAP_DISPLAY_RANGE,
    EGO_LONGITUDINAL_VELOCITY_DISPLAY_RANGE,
    EGO_YAW_RATE_DISPLAY_RANGE,
)
from ppp_eda_geometry import (
    camera_domain_mask,
    camera_plot_limits,
    cartesian_domain_mask,
    cartesian_plot_limits,
    load_spatial_config_from_scan_dir,
    polar_domain_mask,
    polar_plot_limits,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create PPP EDA summary plots.")

    parser.add_argument(
        "--scan-dir",
        type=str,
        required=True,
        help="Directory containing frames.csv and point_values.npz.",
    )
    parser.add_argument(
        "--frames-csv",
        type=str,
        default=None,
        help="Optional explicit path to frames.csv.",
    )
    parser.add_argument(
        "--point-values-npz",
        type=str,
        default=None,
        help="Optional explicit path to point_values.npz.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Output directory. Default: scan_dir/plots.",
    )
    parser.add_argument(
        "--image-height",
        type=int,
        default=544,
        help="Image height for image-space density plot.",
    )
    parser.add_argument(
        "--image-width",
        type=int,
        default=1792,
        help="Image width for image-space density plot.",
    )
    parser.add_argument(
        "--density-y-bins",
        type=int,
        default=136,
        help="Number of vertical bins for image-space density plot.",
    )
    parser.add_argument(
        "--density-x-bins",
        type=int,
        default=448,
        help="Number of horizontal bins for image-space density plot.",
    )
    parser.add_argument(
        "--max-scatter-points",
        type=int,
        default=200_000,
        help="Maximum points used in scatter plots.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Seed for scatter point subsampling.",
    )
    parser.add_argument(
        "--velocity-static-eps",
        type=float,
        default=0.5,
        help="Velocity threshold for near-static bucket.",
    )
    parser.add_argument("--depth-min", type=float, default=DEPTH_DISPLAY_RANGE[0])
    parser.add_argument("--depth-max", type=float, default=DEPTH_DISPLAY_RANGE[1])
    parser.add_argument("--velocity-min", type=float, default=VELOCITY_DISPLAY_RANGE[0])
    parser.add_argument("--velocity-max", type=float, default=VELOCITY_DISPLAY_RANGE[1])
    parser.add_argument("--rcs-min", type=float, default=RCS_DISPLAY_RANGE[0])
    parser.add_argument("--rcs-max", type=float, default=RCS_DISPLAY_RANGE[1])
    parser.add_argument(
        "--range-gate-length-min",
        type=float,
        default=RANGE_GATE_LENGTH_DISPLAY_RANGE[0],
    )
    parser.add_argument(
        "--range-gate-length-max",
        type=float,
        default=RANGE_GATE_LENGTH_DISPLAY_RANGE[1],
    )
    parser.add_argument(
        "--doppler-range-min",
        type=float,
        default=DOPPLER_RANGE_DISPLAY_RANGE[0],
    )
    parser.add_argument(
        "--doppler-range-max",
        type=float,
        default=DOPPLER_RANGE_DISPLAY_RANGE[1],
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite plot directory if it exists.",
    )

    return parser.parse_args()


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")


def prepare_out_dir(out_dir: Path, overwrite: bool) -> None:
    if out_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output directory already exists: {out_dir}\n"
                f"Use --overwrite to replace it."
            )
        shutil.rmtree(out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)


def save_fig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()


def finite_array(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    return x[np.isfinite(x)]


def quantile_dict(x: np.ndarray, prefix: str) -> Dict[str, float]:
    x = finite_array(x)

    if len(x) == 0:
        return {
            f"{prefix}_count": 0,
            f"{prefix}_min": np.nan,
            f"{prefix}_q01": np.nan,
            f"{prefix}_q05": np.nan,
            f"{prefix}_q25": np.nan,
            f"{prefix}_q50": np.nan,
            f"{prefix}_mean": np.nan,
            f"{prefix}_std": np.nan,
            f"{prefix}_var": np.nan,
            f"{prefix}_q75": np.nan,
            f"{prefix}_q95": np.nan,
            f"{prefix}_q99": np.nan,
            f"{prefix}_max": np.nan,
        }

    q01, q05, q25, q50, q75, q95, q99 = np.quantile(
        x,
        [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99],
    )

    return {
        f"{prefix}_count": int(len(x)),
        f"{prefix}_min": float(np.min(x)),
        f"{prefix}_q01": float(q01),
        f"{prefix}_q05": float(q05),
        f"{prefix}_q25": float(q25),
        f"{prefix}_q50": float(q50),
        f"{prefix}_mean": float(np.mean(x)),
        f"{prefix}_std": float(np.std(x)),
        f"{prefix}_var": float(np.var(x)),
        f"{prefix}_q75": float(q75),
        f"{prefix}_q95": float(q95),
        f"{prefix}_q99": float(q99),
        f"{prefix}_max": float(np.max(x)),
    }


def make_summary_frame_stats(frames: pd.DataFrame) -> pd.DataFrame:
    rows = []

    columns = [
        "point_count",
        "ego_longitudinal_velocity",
        "ego_yaw_rate",
        "doppler_range",
        "velocity_wrap_count",
        "velocity_wrap_fraction",
        "velocity_mean_abs_wrap_index",
        "velocity_max_abs_wrap_index",
        "brightness_mean",
        "brightness_std",
        "brightness_q01",
        "brightness_q05",
        "brightness_q50",
        "brightness_q95",
        "brightness_q99",
        "contrast_q95_minus_q05",
        "contrast_q99_minus_q01",
        "laplacian_var",
    ]

    for col in columns:
        if col not in frames.columns:
            continue

        stats = quantile_dict(frames[col].to_numpy(dtype=float), prefix=col)
        row = {"variable": col}
        row.update({k.replace(f"{col}_", ""): v for k, v in stats.items()})
        rows.append(row)

    return pd.DataFrame(rows)


def make_summary_point_stats(points: Dict[str, np.ndarray]) -> pd.DataFrame:
    rows = []

    for name in [
        "depth",
        "velocity",
        "velocity_compensated_wrapped",
        "velocity_measured_relative",
        "velocity_expected_static",
        "velocity_compensated_pre_wrap",
        "doppler_wrap_index",
        "rcs",
        "range_gate_length",
    ]:
        if name in points:
            stats = quantile_dict(points[name].astype(float), prefix=name)
            row = {"variable": name}
            row.update({k.replace(f"{name}_", ""): v for k, v in stats.items()})
            rows.append(row)

    if "velocity" in points:
        abs_velocity = np.abs(points["velocity"].astype(float))
        stats = quantile_dict(abs_velocity, prefix="abs_velocity")
        row = {"variable": "abs_velocity"}
        row.update({k.replace("abs_velocity_", ""): v for k, v in stats.items()})
        rows.append(row)

    if "rgb" in points:
        rgb = points["rgb"].astype(np.float32) / 255.0
        radar_brightness = rgb.mean(axis=1)

        stats = quantile_dict(radar_brightness, prefix="radar_point_brightness")
        row = {"variable": "radar_point_brightness"}
        row.update({k.replace("radar_point_brightness_", ""): v for k, v in stats.items()})
        rows.append(row)

        for channel_idx, channel_name in enumerate(["r", "g", "b"]):
            stats = quantile_dict(rgb[:, channel_idx], prefix=f"radar_point_{channel_name}")
            row = {"variable": f"radar_point_{channel_name}"}
            row.update({k.replace(f"radar_point_{channel_name}_", ""): v for k, v in stats.items()})
            rows.append(row)

    return pd.DataFrame(rows)


def _resolved_hist_range(
    values: np.ndarray,
    value_range: Tuple[float | None, float | None] | None,
    lower_quantile: float = 0.01,
    upper_quantile: float = 0.99,
) -> Tuple[float, float] | None:
    """Resolve fixed plot limits from explicit bounds or empirical tail quantiles."""
    if value_range is None:
        return None

    lo, hi = value_range
    finite = finite_array(values).astype(float, copy=False)
    if len(finite) == 0 and lo is None and hi is None:
        return None

    if lo is None:
        lo = float(np.quantile(finite, lower_quantile)) if len(finite) else 0.0
    if hi is None:
        hi = float(np.quantile(finite, upper_quantile)) if len(finite) else float(lo) + 1.0
    if not np.isfinite(lo) or not np.isfinite(hi):
        return None
    if lo == hi:
        delta = max(abs(float(lo)) * 0.05, 0.5)
        lo -= delta
        hi += delta
    if lo > hi:
        lo, hi = hi, lo
    return float(lo), float(hi)


def empirical_tail_range(
    values: np.ndarray,
    configured: Tuple[float | None, float | None],
) -> Tuple[float, float]:
    resolved = _resolved_hist_range(values, configured)
    if resolved is None:
        return (0.0, 1.0)
    return resolved

def plot_hist(
    values: np.ndarray,
    path: Path,
    title: str,
    xlabel: str,
    bins: int | Iterable[float] = 80,
    log_y: bool = False,
    value_range: Tuple[float | None, float | None] | None = None,
) -> None:
    values = finite_array(values).astype(float, copy=False)
    resolved_range = _resolved_hist_range(values, value_range)

    plt.figure(figsize=(10, 6))

    if len(values) == 0:
        plt.text(0.5, 0.5, "No finite values", ha="center", va="center")
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel("Count")
        save_fig(path)
        return

    plotted = values
    outside_count = 0
    hist_bins = bins
    if resolved_range is not None:
        lo, hi = resolved_range
        inside = (values >= lo) & (values <= hi)
        outside_count = int(np.sum(~inside))
        plotted = values[inside]
        if isinstance(bins, int):
            hist_bins = np.linspace(lo, hi, bins + 1)
        plt.xlim(lo, hi)

    if len(plotted) > 0:
        plt.hist(plotted, bins=hist_bins)
    else:
        plt.text(0.5, 0.5, "No values inside display range", ha="center", va="center", transform=plt.gca().transAxes)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Count")

    if outside_count > 0:
        fraction = outside_count / len(values)
        plt.text(
            0.99,
            0.98,
            f"outside display range: {outside_count:,} ({fraction:.2%})",
            transform=plt.gca().transAxes,
            ha="right",
            va="top",
            fontsize=9,
        )

    if log_y:
        plt.yscale("log")

    save_fig(path)


def plot_bar(labels: List[str], counts: List[int], path: Path, title: str, ylabel: str) -> None:
    plt.figure(figsize=(10, 6))
    x = np.arange(len(labels))
    plt.bar(x, counts)
    plt.xticks(x, labels, rotation=30, ha="right")
    plt.title(title)
    plt.ylabel(ylabel)

    for i, c in enumerate(counts):
        plt.text(i, c, str(c), ha="center", va="bottom", fontsize=9)

    save_fig(path)


def point_count_buckets(point_count: np.ndarray) -> Tuple[List[str], List[int]]:
    bins = [
        -1,
        0,
        1,
        10,
        25,
        50,
        75,
        100,
        125,
        150,
        175,
        200,
        225,
        250,
        275,
        300,
        325,
        350,
        np.inf,
    ]

    labels = [
        "0",
        "1",
        "2-10",
        "11-25",
        "26-50",
        "51-75",
        "76-100",
        "101-125",
        "126-150",
        "151-175",
        "176-200",
        "201-225",
        "226-250",
        "251-275",
        "276-300",
        "301-325",
        "326-350",
        ">350",
    ]

    cats = pd.cut(point_count, bins=bins, labels=labels)
    counts_series = pd.Series(cats).value_counts(sort=False)

    return labels, [int(counts_series.get(label, 0)) for label in labels]

def velocity_buckets(velocity: np.ndarray, eps: float) -> Tuple[List[str], List[int]]:
    velocity = finite_array(velocity)

    labels = [
        f"v < -{eps:g}",
        f"|v| <= {eps:g}",
        f"v > {eps:g}",
    ]

    counts = [
        int((velocity < -eps).sum()),
        int((np.abs(velocity) <= eps).sum()),
        int((velocity > eps).sum()),
    ]

    return labels, counts


def sample_for_scatter(
    arrays: List[np.ndarray],
    max_points: int,
    seed: int,
) -> List[np.ndarray]:
    n = len(arrays[0])

    if n <= max_points:
        return arrays

    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=max_points, replace=False)
    idx.sort()

    return [a[idx] for a in arrays]


def plot_hexbin_xy(
    x_values: np.ndarray,
    y_values: np.ndarray,
    path: Path,
    title: str,
    xlabel: str,
    ylabel: str,
    x_range: Tuple[float, float] | None = None,
    y_range: Tuple[float, float] | None = None,
    identity_line: bool = False,
) -> None:
    x_values = np.asarray(x_values, dtype=float)
    y_values = np.asarray(y_values, dtype=float)
    mask = np.isfinite(x_values) & np.isfinite(y_values)
    x_values = x_values[mask]
    y_values = y_values[mask]

    plt.figure(figsize=(9, 7))
    if len(x_values) == 0:
        plt.text(0.5, 0.5, "No finite values", ha="center", va="center")
    else:
        hb = plt.hexbin(x_values, y_values, gridsize=100, bins="log", mincnt=1)
        plt.colorbar(hb, label="log10(count)")
        if identity_line:
            lo = max(
                np.min(x_values) if x_range is None else x_range[0],
                np.min(y_values) if y_range is None else y_range[0],
            )
            hi = min(
                np.max(x_values) if x_range is None else x_range[1],
                np.max(y_values) if y_range is None else y_range[1],
            )
            if lo < hi:
                plt.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1.0)
    if x_range is not None:
        plt.xlim(*x_range)
    if y_range is not None:
        plt.ylim(*y_range)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    save_fig(path)


def plot_pre_wrap_vs_wrapped(
    pre_wrap: np.ndarray,
    wrapped: np.ndarray,
    wrap_index: np.ndarray,
    path: Path,
    max_points: int,
    seed: int,
    x_range: Tuple[float, float],
    y_range: Tuple[float, float],
) -> None:
    pre_wrap = np.asarray(pre_wrap, dtype=float)
    wrapped = np.asarray(wrapped, dtype=float)
    wrap_index = np.asarray(wrap_index, dtype=float)
    mask = np.isfinite(pre_wrap) & np.isfinite(wrapped) & np.isfinite(wrap_index)
    pre_wrap, wrapped, wrap_index = [
        arr[mask] for arr in (pre_wrap, wrapped, wrap_index)
    ]
    pre_wrap, wrapped, wrap_index = sample_for_scatter(
        [pre_wrap, wrapped, wrap_index], max_points=max_points, seed=seed
    )

    plt.figure(figsize=(10, 7))
    if len(pre_wrap) == 0:
        plt.text(0.5, 0.5, "No finite values", ha="center", va="center")
    else:
        sc = plt.scatter(
            pre_wrap, wrapped, c=wrap_index, s=3, alpha=0.25, cmap="coolwarm"
        )
        plt.colorbar(sc, label="Doppler wrap index")
        lo = max(x_range[0], y_range[0])
        hi = min(x_range[1], y_range[1])
        if lo < hi:
            plt.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1.0)
    plt.xlim(*x_range)
    plt.ylim(*y_range)
    plt.title("Compensated velocity before and after Doppler folding")
    plt.xlabel("compensated velocity before wrapping [m s$^{-1}$]")
    plt.ylabel("wrapped compensated target [m s$^{-1}$]")
    save_fig(path)


def plot_wrap_index_distribution(wrap_index: np.ndarray, path: Path) -> None:
    values = np.asarray(wrap_index)
    values = values[np.isfinite(values)].astype(int, copy=False)
    if len(values) == 0:
        labels, counts = [], []
    else:
        unique, raw_counts = np.unique(values, return_counts=True)
        labels = [str(int(value)) for value in unique]
        counts = [int(value) for value in raw_counts]
    plot_bar(
        labels=labels,
        counts=counts,
        path=path,
        title="Doppler wrap-index distribution",
        ylabel="Number of projected radar points",
    )


def plot_frame_xy(
    x_values: np.ndarray,
    y_values: np.ndarray,
    path: Path,
    title: str,
    xlabel: str,
    ylabel: str,
) -> None:
    x_values = np.asarray(x_values, dtype=float)
    y_values = np.asarray(y_values, dtype=float)
    mask = np.isfinite(x_values) & np.isfinite(y_values)
    plt.figure(figsize=(9, 7))
    if not np.any(mask):
        plt.text(0.5, 0.5, "No finite values", ha="center", va="center")
    else:
        plt.scatter(x_values[mask], y_values[mask], s=8, alpha=0.25)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    save_fig(path)


def plot_spatial_density(
    x_values: np.ndarray,
    y_values: np.ndarray,
    mask: np.ndarray,
    path: Path,
    title: str,
    xlabel: str,
    ylabel: str,
    xlim: Tuple[float, float],
    ylim: Tuple[float, float],
    x_bins: int,
    y_bins: int,
    origin: str = "lower",
    reverse_x: bool = False,
) -> None:
    x_values = np.asarray(x_values, dtype=float)
    y_values = np.asarray(y_values, dtype=float)
    mask = np.asarray(mask, dtype=bool)
    mask &= np.isfinite(x_values) & np.isfinite(y_values)

    plt.figure(figsize=(12, 7))

    if not np.any(mask):
        plt.text(0.5, 0.5, "No points inside configured domain", ha="center", va="center")
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        if reverse_x:
            plt.xlim(xlim[1], xlim[0])
        else:
            plt.xlim(*xlim)
        plt.ylim(*ylim)
        save_fig(path)
        return

    density, _, _ = np.histogram2d(
        x_values[mask],
        y_values[mask],
        bins=[x_bins, y_bins],
        range=[list(xlim), list(ylim)],
    )

    extent = [xlim[0], xlim[1], ylim[0], ylim[1]]
    if origin == "upper":
        extent = [xlim[0], xlim[1], ylim[1], ylim[0]]

    plt.imshow(
        np.log1p(density.T),
        origin=origin,
        extent=extent,
        aspect="auto",
    )
    if reverse_x:
        plt.xlim(xlim[1], xlim[0])
    plt.colorbar(label="log(1 + count)")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    save_fig(path)


def plot_image_space_density(
    x_pix: np.ndarray,
    y_pix: np.ndarray,
    path: Path,
    image_width: int,
    image_height: int,
    x_bins: int,
    y_bins: int,
    spatial: Dict[str, object],
) -> None:
    xlim, ylim = camera_plot_limits(image_width, image_height, spatial)
    mask = camera_domain_mask(x_pix, y_pix, spatial)
    plot_spatial_density(
        x_values=x_pix,
        y_values=y_pix,
        mask=mask,
        path=path,
        title="Image-space radar GT density",
        xlabel="x pixel",
        ylabel="y pixel",
        xlim=xlim,
        ylim=ylim,
        x_bins=x_bins,
        y_bins=y_bins,
        origin="upper",
    )


def plot_cartesian_bev_density(
    lateral: np.ndarray,
    forward: np.ndarray,
    path: Path,
    x_bins: int,
    y_bins: int,
    spatial: Dict[str, object],
) -> None:
    xlim, ylim = cartesian_plot_limits(lateral, forward, spatial)
    mask = cartesian_domain_mask(lateral, forward, spatial)
    plot_spatial_density(
        x_values=lateral,
        y_values=forward,
        mask=mask,
        path=path,
        title="Cartesian BEV radar GT density",
        xlabel="lateral position [m]",
        ylabel="forward position [m]",
        xlim=xlim,
        ylim=ylim,
        x_bins=x_bins,
        y_bins=y_bins,
        reverse_x=True,
    )


def plot_polar_density(
    azimuth_deg: np.ndarray,
    range_m: np.ndarray,
    path: Path,
    x_bins: int,
    y_bins: int,
    spatial: Dict[str, object],
) -> None:
    xlim, ylim = polar_plot_limits(azimuth_deg, range_m, spatial)
    mask = polar_domain_mask(range_m, azimuth_deg, spatial)
    plot_spatial_density(
        x_values=azimuth_deg,
        y_values=range_m,
        mask=mask,
        path=path,
        title="Polar radar GT density",
        xlabel="azimuth [deg]",
        ylabel="range [m]",
        xlim=xlim,
        ylim=ylim,
        x_bins=x_bins,
        y_bins=y_bins,
        reverse_x=True,
    )


def plot_depth_vs_y(
    y_pix: np.ndarray,
    depth: np.ndarray,
    path: Path,
    max_scatter_points: int,
    seed: int,
    depth_range: Tuple[float, float],
) -> None:
    y_pix = y_pix.astype(float)
    depth = depth.astype(float)

    mask = np.isfinite(y_pix) & np.isfinite(depth)
    y_pix = y_pix[mask]
    depth = depth[mask]

    y_plot, depth_plot = sample_for_scatter(
        [y_pix, depth],
        max_points=max_scatter_points,
        seed=seed,
    )

    plt.figure(figsize=(10, 7))

    if len(y_plot) == 0:
        plt.text(0.5, 0.5, "No finite values", ha="center", va="center")
    else:
        plt.scatter(y_plot, depth_plot, s=2, alpha=0.15)

    plt.title("GT depth vs vertical image position")
    plt.xlabel("y pixel")
    plt.ylabel("GT depth [m]")
    plt.ylim(*depth_range)

    save_fig(path)


def plot_velocity_vs_depth(
    depth: np.ndarray,
    velocity: np.ndarray,
    path: Path,
    max_scatter_points: int,
    seed: int,
    depth_range: Tuple[float, float],
    velocity_range: Tuple[float, float],
) -> None:
    depth = depth.astype(float)
    velocity = velocity.astype(float)

    mask = np.isfinite(depth) & np.isfinite(velocity)
    depth = depth[mask]
    velocity = velocity[mask]

    depth_plot, velocity_plot = sample_for_scatter(
        [depth, velocity],
        max_points=max_scatter_points,
        seed=seed,
    )

    plt.figure(figsize=(10, 7))

    if len(depth_plot) == 0:
        plt.text(0.5, 0.5, "No finite values", ha="center", va="center")
    else:
        plt.scatter(depth_plot, velocity_plot, s=2, alpha=0.15)

    plt.title("GT velocity vs GT depth")
    plt.xlabel("GT depth [m]")
    plt.ylabel("GT radial velocity [m s$^{-1}$]")
    plt.xlim(*depth_range)
    plt.ylim(*velocity_range)

    save_fig(path)


def plot_rcs_vs_depth(
    depth: np.ndarray,
    rcs: np.ndarray,
    path: Path,
    max_scatter_points: int,
    seed: int,
    depth_range: Tuple[float, float],
    rcs_range: Tuple[float, float],
) -> None:
    depth = depth.astype(float)
    rcs = rcs.astype(float)

    mask = np.isfinite(depth) & np.isfinite(rcs)
    depth = depth[mask]
    rcs = rcs[mask]

    depth_plot, rcs_plot = sample_for_scatter(
        [depth, rcs],
        max_points=max_scatter_points,
        seed=seed,
    )

    plt.figure(figsize=(10, 7))

    if len(depth_plot) == 0:
        plt.text(0.5, 0.5, "No finite values", ha="center", va="center")
    else:
        plt.scatter(depth_plot, rcs_plot, s=2, alpha=0.15)

    plt.title("GT RCS vs GT depth")
    plt.xlabel("GT depth [m]")
    plt.ylabel("GT RCS [dBsm]")
    plt.xlim(*depth_range)
    plt.ylim(*rcs_range)

    save_fig(path)


def plot_rgb_at_radar_points(rgb: np.ndarray, path: Path) -> None:
    plt.figure(figsize=(10, 6))

    if len(rgb) == 0:
        plt.text(0.5, 0.5, "No RGB values", ha="center", va="center")
        plt.title("RGB at radar GT pixels")
        save_fig(path)
        return

    rgb01 = rgb.astype(np.float32) / 255.0
    brightness = rgb01.mean(axis=1)

    plt.hist(brightness, bins=80, alpha=0.65, label="brightness")
    plt.hist(rgb01[:, 0], bins=80, alpha=0.35, label="R")
    plt.hist(rgb01[:, 1], bins=80, alpha=0.35, label="G")
    plt.hist(rgb01[:, 2], bins=80, alpha=0.35, label="B")

    plt.title("RGB and brightness at radar GT pixels")
    plt.xlabel("Value in [0, 1]")
    plt.ylabel("Count")
    plt.legend()

    save_fig(path)


def load_points(npz_path: Path) -> Dict[str, np.ndarray]:
    pts = np.load(npz_path)

    required = [
        "eda_idx", "x_pix", "y_pix", "depth", "velocity",
        "velocity_compensated_wrapped", "velocity_measured_relative",
        "velocity_expected_static", "velocity_compensated_pre_wrap",
        "doppler_wrap_index", "rcs", "range_gate_length", "lateral",
        "forward", "range", "azimuth_deg", "rgb",
    ]
    missing = [k for k in required if k not in pts.files]
    if missing:
        raise KeyError(
            f"Missing arrays in {npz_path}: {missing}. "
            f"Available arrays: {pts.files}"
        )

    return {k: pts[k] for k in required}


def main() -> None:
    args = parse_args()

    scan_dir = Path(args.scan_dir)
    frames_csv = Path(args.frames_csv) if args.frames_csv else scan_dir / "frames.csv"
    point_values_npz = (
        Path(args.point_values_npz)
        if args.point_values_npz
        else scan_dir / "point_values.npz"
    )
    out_dir = Path(args.out_dir) if args.out_dir else scan_dir / "plots"

    require_file(frames_csv)
    require_file(point_values_npz)
    prepare_out_dir(out_dir, overwrite=args.overwrite)

    frames = pd.read_csv(frames_csv)
    points = load_points(point_values_npz)
    spatial = load_spatial_config_from_scan_dir(scan_dir)

    if "point_count" not in frames.columns:
        raise KeyError("frames.csv must contain column 'point_count'.")
    required_frame_columns = [
        "doppler_range",
        "ego_longitudinal_velocity",
        "ego_yaw_rate",
        "velocity_wrap_fraction",
    ]
    missing_frame_columns = [
        name for name in required_frame_columns if name not in frames.columns
    ]
    if missing_frame_columns:
        raise KeyError(
            f"frames.csv is missing {missing_frame_columns}. Rerun scan_ppp_eda.py "
            "with the updated velocity-stage EDA pipeline."
        )

    point_count = frames["point_count"].to_numpy(dtype=float)

    summary_frames = make_summary_frame_stats(frames)
    summary_points = make_summary_point_stats(points)

    summary_frames_path = out_dir / "summary_frames.csv"
    summary_points_path = out_dir / "summary_points.csv"

    summary_frames.to_csv(summary_frames_path, index=False)
    summary_points.to_csv(summary_points_path, index=False)

    # -------------------------------------------------------------------------
    # Frame-level plots
    # -------------------------------------------------------------------------

    plot_hist(
        values=point_count,
        path=out_dir / "point_count_hist.png",
        title="Projected radar GT point count per frame",
        xlabel="point_count",
        bins=80,
        log_y=False,
    )

    labels, counts = point_count_buckets(point_count)
    plot_bar(
        labels=labels,
        counts=counts,
        path=out_dir / "point_count_bucket_bar.png",
        title="Projected radar GT point count buckets",
        ylabel="Number of frames",
    )

    if "brightness_mean" in frames.columns:
        plot_hist(
            values=frames["brightness_mean"].to_numpy(dtype=float),
            path=out_dir / "brightness_hist.png",
            title="Whole-image brightness distribution",
            xlabel="brightness_mean",
            bins=80,
        )

    if "contrast_q95_minus_q05" in frames.columns:
        plot_hist(
            values=frames["contrast_q95_minus_q05"].to_numpy(dtype=float),
            path=out_dir / "contrast_hist.png",
            title="Whole-image contrast distribution",
            xlabel="contrast_q95_minus_q05",
            bins=80,
        )

    if "laplacian_var" in frames.columns:
        lap = frames["laplacian_var"].to_numpy(dtype=float)
        plot_hist(
            values=np.log1p(lap),
            path=out_dir / "laplacian_var_hist.png",
            title="Whole-image sharpness proxy distribution",
            xlabel="log(1 + laplacian_var)",
            bins=80,
        )

    ego_velocity = frames["ego_longitudinal_velocity"].to_numpy(dtype=float)
    ego_yaw_rate = frames["ego_yaw_rate"].to_numpy(dtype=float)
    velocity_wrap_fraction = frames["velocity_wrap_fraction"].to_numpy(dtype=float)
    ego_velocity_range = empirical_tail_range(
        ego_velocity, EGO_LONGITUDINAL_VELOCITY_DISPLAY_RANGE
    )
    ego_yaw_rate_range = empirical_tail_range(
        ego_yaw_rate, EGO_YAW_RATE_DISPLAY_RANGE
    )

    plot_hist(
        ego_velocity, out_dir / "ego_longitudinal_velocity_hist.png",
        "Frame-level ego longitudinal velocity distribution",
        "ego longitudinal velocity [m s$^{-1}$]", bins=80,
        value_range=ego_velocity_range,
    )
    plot_hist(
        ego_yaw_rate, out_dir / "ego_yaw_rate_hist.png",
        "Frame-level ego yaw-rate distribution",
        "ego yaw rate", bins=80, value_range=ego_yaw_rate_range,
    )
    plot_hist(
        velocity_wrap_fraction, out_dir / "velocity_wrap_fraction_hist.png",
        "Fraction of projected points changed by Doppler folding per frame",
        "wrap fraction", bins=50, value_range=(0.0, 1.0),
    )
    plot_frame_xy(
        ego_velocity, velocity_wrap_fraction,
        out_dir / "wrap_fraction_vs_ego_velocity.png",
        "Doppler-wrap fraction versus ego speed",
        "ego longitudinal velocity [m s$^{-1}$]", "wrap fraction",
    )

    # -------------------------------------------------------------------------
    # Point-level plots
    # -------------------------------------------------------------------------

    depth = points["depth"].astype(float)
    velocity = points["velocity"].astype(float)
    velocity_measured_relative = points["velocity_measured_relative"].astype(float)
    velocity_expected_static = points["velocity_expected_static"].astype(float)
    velocity_compensated_pre_wrap = points[
        "velocity_compensated_pre_wrap"
    ].astype(float)
    doppler_wrap_index = points["doppler_wrap_index"].astype(float)
    abs_velocity = np.abs(velocity)
    rcs = points["rcs"].astype(float)
    range_gate_length = points["range_gate_length"].astype(float)
    doppler_range = frames["doppler_range"].to_numpy(dtype=float)

    measured_range = empirical_tail_range(
        velocity_measured_relative, VELOCITY_MEASURED_RELATIVE_DISPLAY_RANGE
    )
    expected_static_range = empirical_tail_range(
        velocity_expected_static, VELOCITY_EXPECTED_STATIC_DISPLAY_RANGE
    )
    pre_wrap_range = empirical_tail_range(
        velocity_compensated_pre_wrap,
        VELOCITY_COMPENSATED_PRE_WRAP_DISPLAY_RANGE,
    )
    range_gate_display_range = empirical_tail_range(
        range_gate_length,
        (args.range_gate_length_min, args.range_gate_length_max),
    )
    doppler_display_range = empirical_tail_range(
        doppler_range, (args.doppler_range_min, args.doppler_range_max)
    )

    plot_hist(
        values=depth,
        path=out_dir / "depth_hist.png",
        title="GT depth distribution at projected radar points",
        xlabel="GT depth [m]",
        bins=100,
        value_range=(args.depth_min, args.depth_max),
    )

    plot_hist(
        values=velocity,
        path=out_dir / "velocity_signed_hist.png",
        title="Signed GT radial velocity distribution",
        xlabel="GT radial velocity [m s$^{-1}$]",
        bins=100,
        value_range=(args.velocity_min, args.velocity_max),
    )

    plot_hist(
        values=abs_velocity,
        path=out_dir / "velocity_abs_hist.png",
        title="Absolute GT radial velocity distribution",
        xlabel="abs(GT radial velocity) [m s$^{-1}$]",
        bins=100,
        value_range=(0.0, max(abs(args.velocity_min), abs(args.velocity_max))),
    )

    plot_hist(
        values=velocity,
        path=out_dir / "velocity_signed_hist_log_y.png",
        title="Signed GT radial velocity distribution, log-y",
        xlabel="GT radial velocity [m s$^{-1}$]",
        bins=100,
        log_y=True,
        value_range=(args.velocity_min, args.velocity_max),
    )

    plot_hist(
        values=abs_velocity,
        path=out_dir / "velocity_abs_hist_log_y.png",
        title="Absolute GT radial velocity distribution, log-y",
        xlabel="abs(GT radial velocity) [m s$^{-1}$]",
        bins=100,
        log_y=True,
        value_range=(0.0, max(abs(args.velocity_min), abs(args.velocity_max))),
    )

    for values, stem, title, xlabel, display_range in [
        (
            velocity_measured_relative,
            "velocity_measured_relative",
            "Measured relative radial velocity distribution",
            "measured relative radial velocity [m s$^{-1}$]",
            measured_range,
        ),
        (
            velocity_expected_static,
            "velocity_expected_static",
            "Expected static radial velocity distribution",
            "expected static radial velocity [m s$^{-1}$]",
            expected_static_range,
        ),
        (
            velocity_compensated_pre_wrap,
            "velocity_compensated_pre_wrap",
            "Ego-motion-compensated velocity before Doppler folding",
            "compensated velocity before wrapping [m s$^{-1}$]",
            pre_wrap_range,
        ),
    ]:
        plot_hist(
            values, out_dir / f"{stem}_hist.png", title, xlabel, bins=100,
            value_range=display_range,
        )
        plot_hist(
            values, out_dir / f"{stem}_hist_log_y.png",
            f"{title}, log-y", xlabel, bins=100, log_y=True,
            value_range=display_range,
        )

    plot_wrap_index_distribution(
        doppler_wrap_index, out_dir / "doppler_wrap_index_bar.png"
    )
    plot_hexbin_xy(
        velocity_measured_relative, velocity_expected_static,
        out_dir / "velocity_measured_vs_expected_static.png",
        "Measured versus expected-static radial velocity",
        "measured relative radial velocity [m s$^{-1}$]",
        "expected static radial velocity [m s$^{-1}$]",
        x_range=measured_range, y_range=expected_static_range,
        identity_line=True,
    )
    plot_pre_wrap_vs_wrapped(
        velocity_compensated_pre_wrap, velocity, doppler_wrap_index,
        out_dir / "velocity_pre_wrap_vs_wrapped.png",
        max_points=args.max_scatter_points, seed=args.random_seed,
        x_range=pre_wrap_range,
        y_range=(args.velocity_min, args.velocity_max),
    )
    plot_hexbin_xy(
        points["azimuth_deg"], velocity_expected_static,
        out_dir / "velocity_expected_static_vs_azimuth.png",
        "Expected static radial velocity versus azimuth",
        "azimuth [deg]", "expected static radial velocity [m s$^{-1}$]",
        y_range=expected_static_range,
    )

    frame_doppler = pd.Series(
        frames["doppler_range"].to_numpy(dtype=float),
        index=frames["eda_idx"].to_numpy(dtype=int),
    )
    point_doppler = frame_doppler.reindex(
        points["eda_idx"].astype(int)
    ).to_numpy(dtype=float)
    valid_doppler = np.isfinite(point_doppler) & (point_doppler > 0.0)
    normalized_pre_wrap = np.full_like(velocity_compensated_pre_wrap, np.nan)
    normalized_wrapped = np.full_like(velocity, np.nan)
    normalized_pre_wrap[valid_doppler] = (
        velocity_compensated_pre_wrap[valid_doppler]
        / point_doppler[valid_doppler]
    )
    normalized_wrapped[valid_doppler] = (
        velocity[valid_doppler] / point_doppler[valid_doppler]
    )
    plot_hist(
        normalized_pre_wrap, out_dir / "velocity_pre_wrap_over_doppler_range.png",
        "Compensated pre-wrap velocity normalized by Doppler range",
        "pre-wrap velocity / dopplerRange", bins=100,
        value_range=empirical_tail_range(normalized_pre_wrap, (None, None)),
    )
    plot_hist(
        normalized_wrapped, out_dir / "velocity_wrapped_over_doppler_range.png",
        "Wrapped target normalized by Doppler range",
        "wrapped velocity / dopplerRange", bins=100, value_range=(-0.5, 0.5),
    )

    plot_hist(
        values=rcs,
        path=out_dir / "rcs_hist.png",
        title="GT RCS distribution at projected radar points",
        xlabel="GT RCS [dBsm]",
        bins=100,
        value_range=(args.rcs_min, args.rcs_max),
    )

    plot_hist(
        values=rcs,
        path=out_dir / "rcs_hist_log_y.png",
        title="GT RCS distribution at projected radar points, log-y",
        xlabel="GT RCS [dBsm]",
        bins=100,
        log_y=True,
        value_range=(args.rcs_min, args.rcs_max),
    )

    plot_hist(
        values=range_gate_length,
        path=out_dir / "range_gate_length_hist.png",
        title="Range-gate length distribution at projected radar points",
        xlabel="rangeGateLength [m]",
        bins=100,
        value_range=range_gate_display_range,
    )

    plot_hist(
        values=doppler_range,
        path=out_dir / "doppler_range_hist.png",
        title="Frame-level Doppler ambiguity-range distribution",
        xlabel="dopplerRange [m s$^{-1}$]",
        bins=80,
        value_range=doppler_display_range,
    )

    labels, counts = velocity_buckets(velocity, eps=args.velocity_static_eps)
    plot_bar(
        labels=labels,
        counts=counts,
        path=out_dir / "velocity_bucket_bar.png",
        title=f"GT radial velocity buckets, eps={args.velocity_static_eps:g}",
        ylabel="Number of radar GT points",
    )

    plot_image_space_density(
        x_pix=points["x_pix"].astype(float),
        y_pix=points["y_pix"].astype(float),
        path=out_dir / "image_space_density.png",
        image_width=args.image_width,
        image_height=args.image_height,
        x_bins=args.density_x_bins,
        y_bins=args.density_y_bins,
        spatial=spatial,
    )

    plot_cartesian_bev_density(
        lateral=points["lateral"].astype(float),
        forward=points["forward"].astype(float),
        path=out_dir / "cartesian_bev_density.png",
        x_bins=args.density_x_bins,
        y_bins=args.density_y_bins,
        spatial=spatial,
    )

    plot_polar_density(
        azimuth_deg=points["azimuth_deg"].astype(float),
        range_m=points["range"].astype(float),
        path=out_dir / "polar_density.png",
        x_bins=args.density_x_bins,
        y_bins=args.density_y_bins,
        spatial=spatial,
    )

    plot_depth_vs_y(
        y_pix=points["y_pix"],
        depth=points["depth"],
        path=out_dir / "depth_vs_y_pix.png",
        max_scatter_points=args.max_scatter_points,
        seed=args.random_seed,
        depth_range=(args.depth_min, args.depth_max),
    )

    plot_velocity_vs_depth(
        depth=points["depth"],
        velocity=points["velocity"],
        path=out_dir / "velocity_vs_depth.png",
        max_scatter_points=args.max_scatter_points,
        seed=args.random_seed,
        depth_range=(args.depth_min, args.depth_max),
        velocity_range=(args.velocity_min, args.velocity_max),
    )

    plot_rcs_vs_depth(
        depth=points["depth"],
        rcs=points["rcs"],
        path=out_dir / "rcs_vs_depth.png",
        max_scatter_points=args.max_scatter_points,
        seed=args.random_seed,
        depth_range=(args.depth_min, args.depth_max),
        rcs_range=(args.rcs_min, args.rcs_max),
    )

    plot_rgb_at_radar_points(
        rgb=points["rgb"],
        path=out_dir / "rgb_at_radar_points_hist.png",
    )

    # -------------------------------------------------------------------------
    # JSON summary
    # -------------------------------------------------------------------------

    labels_pc, counts_pc = point_count_buckets(point_count)
    labels_v, counts_v = velocity_buckets(velocity, eps=args.velocity_static_eps)

    plot_summary = {
        "scan_dir": str(scan_dir),
        "frames_csv": str(frames_csv),
        "point_values_npz": str(point_values_npz),
        "out_dir": str(out_dir),
        "num_frames": int(len(frames)),
        "num_point_rows": int(len(points["eda_idx"])),
        "image_height": int(args.image_height),
        "image_width": int(args.image_width),
        "ppp_eda_spatial": spatial,
        "attribute_display_ranges": {
            "depth": [args.depth_min, args.depth_max],
            "velocity": [args.velocity_min, args.velocity_max],
            "velocity_measured_relative": list(measured_range),
            "velocity_expected_static": list(expected_static_range),
            "velocity_compensated_pre_wrap": list(pre_wrap_range),
            "ego_longitudinal_velocity": list(ego_velocity_range),
            "ego_yaw_rate": list(ego_yaw_rate_range),
            "rcs": [args.rcs_min, args.rcs_max],
            "range_gate_length": list(range_gate_display_range),
            "doppler_range": list(doppler_display_range),
        },
        "camera_domain_points": int(np.sum(camera_domain_mask(points["x_pix"], points["y_pix"], spatial))),
        "cartesian_bev_domain_points": int(np.sum(cartesian_domain_mask(points["lateral"], points["forward"], spatial))),
        "polar_domain_points": int(np.sum(polar_domain_mask(points["range"], points["azimuth_deg"], spatial))),
        "point_count_buckets": {
            label: int(count)
            for label, count in zip(labels_pc, counts_pc)
        },
        "velocity_buckets": {
            label: int(count)
            for label, count in zip(labels_v, counts_v)
        },
        "doppler_wrap_fraction_points": float(
            np.mean(doppler_wrap_index != 0) if len(doppler_wrap_index) else 0.0
        ),
        "doppler_wrap_index_counts": {
            str(int(value)): int(count)
            for value, count in zip(
                *np.unique(doppler_wrap_index.astype(int), return_counts=True)
            )
        },
        "wrapped_normalized_interval_check": {
            "min": float(np.nanmin(normalized_wrapped)) if np.any(np.isfinite(normalized_wrapped)) else None,
            "max": float(np.nanmax(normalized_wrapped)) if np.any(np.isfinite(normalized_wrapped)) else None,
            "outside_half_period_count": int(
                np.sum(
                    np.isfinite(normalized_wrapped)
                    & ((normalized_wrapped < -0.5) | (normalized_wrapped >= 0.5))
                )
            ),
        },
        "summary_frames_csv": str(summary_frames_path),
        "summary_points_csv": str(summary_points_path),
    }

    with open(out_dir / "plot_summary.json", "w", encoding="utf-8") as f:
        json.dump(plot_summary, f, indent=2)

    print("Done.")
    print(f"Plot directory: {out_dir}")
    print(f"Frames:         {len(frames)}")
    print(f"Point rows:     {len(points['eda_idx'])}")
    print(f"Summary frames: {summary_frames_path}")
    print(f"Summary points: {summary_points_path}")


if __name__ == "__main__":
    main()