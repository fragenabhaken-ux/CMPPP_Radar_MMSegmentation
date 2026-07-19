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

    for name in ["depth", "velocity", "rcs"]:
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


def plot_hist(
    values: np.ndarray,
    path: Path,
    title: str,
    xlabel: str,
    bins: int | Iterable[float] = 80,
    log_y: bool = False,
) -> None:
    values = finite_array(values)

    plt.figure(figsize=(10, 6))

    if len(values) == 0:
        plt.text(0.5, 0.5, "No finite values", ha="center", va="center")
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel("Count")
        save_fig(path)
        return

    plt.hist(values, bins=bins)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Count")

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

    save_fig(path)


def plot_velocity_vs_depth(
    depth: np.ndarray,
    velocity: np.ndarray,
    path: Path,
    max_scatter_points: int,
    seed: int,
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
    plt.ylabel("GT radial velocity")

    save_fig(path)


def plot_rcs_vs_depth(
    depth: np.ndarray,
    rcs: np.ndarray,
    path: Path,
    max_scatter_points: int,
    seed: int,
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
    plt.ylabel("GT RCS")

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
        "eda_idx", "x_pix", "y_pix", "depth", "velocity", "rcs",
        "lateral", "forward", "range", "azimuth_deg", "rgb",
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

    # -------------------------------------------------------------------------
    # Point-level plots
    # -------------------------------------------------------------------------

    depth = points["depth"].astype(float)
    velocity = points["velocity"].astype(float)
    abs_velocity = np.abs(velocity)
    rcs = points["rcs"].astype(float)

    plot_hist(
        values=depth,
        path=out_dir / "depth_hist.png",
        title="GT depth distribution at projected radar points",
        xlabel="GT depth [m]",
        bins=100,
    )

    plot_hist(
        values=velocity,
        path=out_dir / "velocity_signed_hist.png",
        title="Signed GT radial velocity distribution",
        xlabel="GT radial velocity",
        bins=100,
    )

    plot_hist(
        values=abs_velocity,
        path=out_dir / "velocity_abs_hist.png",
        title="Absolute GT radial velocity distribution",
        xlabel="abs(GT radial velocity)",
        bins=100,
    )

    plot_hist(
    values=velocity,
    path=out_dir / "velocity_signed_hist_log_y.png",
    title="Signed GT radial velocity distribution, log-y",
    xlabel="GT radial velocity",
    bins=100,
    log_y=True,
    )

    plot_hist(
        values=abs_velocity,
        path=out_dir / "velocity_abs_hist_log_y.png",
        title="Absolute GT radial velocity distribution, log-y",
        xlabel="abs(GT radial velocity)",
        bins=100,
        log_y=True,
    )

    plot_hist(
        values=rcs,
        path=out_dir / "rcs_hist.png",
        title="GT RCS distribution at projected radar points",
        xlabel="GT RCS",
        bins=100,
    )

    plot_hist(
        values=rcs,
        path=out_dir / "rcs_hist_log_y.png",
        title="GT RCS distribution at projected radar points, log-y",
        xlabel="GT RCS",
        bins=100,
        log_y=True,
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
    )

    plot_velocity_vs_depth(
        depth=points["depth"],
        velocity=points["velocity"],
        path=out_dir / "velocity_vs_depth.png",
        max_scatter_points=args.max_scatter_points,
        seed=args.random_seed,
    )

    plot_rcs_vs_depth(
        depth=points["depth"],
        rcs=points["rcs"],
        path=out_dir / "rcs_vs_depth.png",
        max_scatter_points=args.max_scatter_points,
        seed=args.random_seed,
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