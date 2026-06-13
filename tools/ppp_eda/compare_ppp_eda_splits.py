#!/usr/bin/env python3
"""
Compare two PPP EDA scan directories.

Typical use:
    compare train vs validation scans.

Input:
    train_scan_dir/
        frames.csv
        point_values.npz

    val_scan_dir/
        frames.csv
        point_values.npz

Output:
    out_dir/
        frame_summary_train_val.csv
        point_summary_train_val.csv
        point_count_buckets_train_val.csv
        velocity_buckets_train_val.csv
        compare_summary.json

        point_count_train_vs_val.png
        point_count_bucket_train_vs_val.png
        brightness_train_vs_val.png
        contrast_train_vs_val.png
        laplacian_var_train_vs_val.png
        depth_train_vs_val.png
        velocity_signed_train_vs_val.png
        velocity_abs_train_vs_val.png
        velocity_bucket_train_vs_val.png
        radar_point_brightness_train_vs_val.png
        image_space_density_train.png
        image_space_density_val.png
        image_space_density_difference_val_minus_train.png
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare PPP EDA scan directories.")

    parser.add_argument(
        "--train-dir",
        type=str,
        required=True,
        help="Train scan directory containing frames.csv and point_values.npz.",
    )
    parser.add_argument(
        "--val-dir",
        type=str,
        required=True,
        help="Validation scan directory containing frames.csv and point_values.npz.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Output directory. Default: work_dirs/ppp_eda/compare_train_val_n<N>.",
    )
    parser.add_argument(
        "--image-height",
        type=int,
        default=544,
        help="Image height for image-space density plots.",
    )
    parser.add_argument(
        "--image-width",
        type=int,
        default=1792,
        help="Image width for image-space density plots.",
    )
    parser.add_argument(
        "--density-y-bins",
        type=int,
        default=136,
        help="Number of vertical bins for image-space density plots.",
    )
    parser.add_argument(
        "--density-x-bins",
        type=int,
        default=448,
        help="Number of horizontal bins for image-space density plots.",
    )
    parser.add_argument(
        "--velocity-static-eps",
        type=float,
        default=0.5,
        help="Velocity threshold for near-static bucket.",
    )
    parser.add_argument(
        "--hist-bins",
        type=int,
        default=80,
        help="Number of histogram bins.",
    )
    parser.add_argument(
        "--clip-quantile",
        type=float,
        default=0.995,
        help=(
            "Upper quantile used to clip histogram range for nonnegative variables. "
            "Use 1.0 to disable."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output directory if it exists.",
    )

    return parser.parse_args()


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")


def prepare_out_dir(out_dir: Path, overwrite: bool) -> None:
    if out_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output directory exists: {out_dir}\n"
                f"Use --overwrite to replace it."
            )
        shutil.rmtree(out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)


def finite_array(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    return x[np.isfinite(x)]


def save_fig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()


def load_scan(scan_dir: Path) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    frames_csv = scan_dir / "frames.csv"
    points_npz = scan_dir / "point_values.npz"

    require_file(frames_csv)
    require_file(points_npz)

    frames = pd.read_csv(frames_csv)

    pts_raw = np.load(points_npz)
    required = ["eda_idx", "x_pix", "y_pix", "depth", "velocity", "rgb"]
    missing = [k for k in required if k not in pts_raw.files]
    if missing:
        raise KeyError(
            f"Missing arrays in {points_npz}: {missing}. "
            f"Available arrays: {pts_raw.files}"
        )

    points = {k: pts_raw[k] for k in required}

    return frames, points


def quantile_dict(x: np.ndarray) -> Dict[str, float]:
    x = finite_array(x)

    if len(x) == 0:
        return {
            "count": 0,
            "min": np.nan,
            "q01": np.nan,
            "q05": np.nan,
            "q25": np.nan,
            "q50": np.nan,
            "mean": np.nan,
            "std": np.nan,
            "q75": np.nan,
            "q95": np.nan,
            "q99": np.nan,
            "max": np.nan,
        }

    q01, q05, q25, q50, q75, q95, q99 = np.quantile(
        x,
        [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99],
    )

    return {
        "count": int(len(x)),
        "min": float(np.min(x)),
        "q01": float(q01),
        "q05": float(q05),
        "q25": float(q25),
        "q50": float(q50),
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "q75": float(q75),
        "q95": float(q95),
        "q99": float(q99),
        "max": float(np.max(x)),
    }


def make_frame_summary(split_name: str, frames: pd.DataFrame) -> pd.DataFrame:
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

        row = {
            "split": split_name,
            "variable": col,
        }
        row.update(quantile_dict(frames[col].to_numpy(dtype=float)))
        rows.append(row)

    return pd.DataFrame(rows)


def make_point_summary(split_name: str, points: Dict[str, np.ndarray]) -> pd.DataFrame:
    rows = []

    depth = points["depth"].astype(float)
    velocity = points["velocity"].astype(float)
    abs_velocity = np.abs(velocity)

    for name, arr in [
        ("depth", depth),
        ("velocity", velocity),
        ("abs_velocity", abs_velocity),
    ]:
        row = {
            "split": split_name,
            "variable": name,
        }
        row.update(quantile_dict(arr))
        rows.append(row)

    rgb = points["rgb"].astype(np.float32) / 255.0
    radar_brightness = rgb.mean(axis=1)

    for name, arr in [
        ("radar_point_brightness", radar_brightness),
        ("radar_point_r", rgb[:, 0]),
        ("radar_point_g", rgb[:, 1]),
        ("radar_point_b", rgb[:, 2]),
    ]:
        row = {
            "split": split_name,
            "variable": name,
        }
        row.update(quantile_dict(arr))
        rows.append(row)

    return pd.DataFrame(rows)


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
        ">300",
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
        int(np.sum(velocity < -eps)),
        int(np.sum(np.abs(velocity) <= eps)),
        int(np.sum(velocity > eps)),
    ]

    return labels, counts


def make_point_count_bucket_df(
    train_frames: pd.DataFrame,
    val_frames: pd.DataFrame,
) -> pd.DataFrame:
    labels, train_counts = point_count_buckets(
        train_frames["point_count"].to_numpy(dtype=float)
    )
    _, val_counts = point_count_buckets(
        val_frames["point_count"].to_numpy(dtype=float)
    )

    train_total = max(sum(train_counts), 1)
    val_total = max(sum(val_counts), 1)

    rows = []
    for label, train_count, val_count in zip(labels, train_counts, val_counts):
        rows.append(
            {
                "bucket": label,
                "train_count": int(train_count),
                "val_count": int(val_count),
                "train_fraction": float(train_count / train_total),
                "val_fraction": float(val_count / val_total),
                "val_minus_train_fraction": float(
                    val_count / val_total - train_count / train_total
                ),
            }
        )

    return pd.DataFrame(rows)


def make_velocity_bucket_df(
    train_points: Dict[str, np.ndarray],
    val_points: Dict[str, np.ndarray],
    eps: float,
) -> pd.DataFrame:
    labels, train_counts = velocity_buckets(train_points["velocity"].astype(float), eps)
    _, val_counts = velocity_buckets(val_points["velocity"].astype(float), eps)

    train_total = max(sum(train_counts), 1)
    val_total = max(sum(val_counts), 1)

    rows = []
    for label, train_count, val_count in zip(labels, train_counts, val_counts):
        rows.append(
            {
                "bucket": label,
                "train_count": int(train_count),
                "val_count": int(val_count),
                "train_fraction": float(train_count / train_total),
                "val_fraction": float(val_count / val_total),
                "val_minus_train_fraction": float(
                    val_count / val_total - train_count / train_total
                ),
            }
        )

    return pd.DataFrame(rows)


def common_hist_edges(
    a: np.ndarray,
    b: np.ndarray,
    bins: int,
    clip_quantile: float = 1.0,
    symmetric: bool = False,
) -> np.ndarray:
    a = finite_array(a)
    b = finite_array(b)
    both = np.concatenate([a, b]) if len(a) + len(b) > 0 else np.array([])

    if len(both) == 0:
        return np.linspace(0.0, 1.0, bins + 1)

    if symmetric:
        max_abs = float(np.max(np.abs(both)))
        if max_abs <= 0:
            max_abs = 1.0
        return np.linspace(-max_abs, max_abs, bins + 1)

    lo = float(np.min(both))

    if clip_quantile < 1.0:
        hi = float(np.quantile(both, clip_quantile))
    else:
        hi = float(np.max(both))

    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        lo = lo - 0.5
        hi = hi + 0.5

    return np.linspace(lo, hi, bins + 1)


def plot_hist_compare(
    train_values: np.ndarray,
    val_values: np.ndarray,
    path: Path,
    title: str,
    xlabel: str,
    bins: int,
    clip_quantile: float = 1.0,
    symmetric: bool = False,
    density: bool = True,
    log_y: bool = False,
) -> None:
    train_values = finite_array(train_values)
    val_values = finite_array(val_values)

    edges = common_hist_edges(
        train_values,
        val_values,
        bins=bins,
        clip_quantile=clip_quantile,
        symmetric=symmetric,
    )

    plt.figure(figsize=(10, 6))

    if len(train_values) > 0:
        plt.hist(
            train_values,
            bins=edges,
            alpha=0.55,
            density=density,
            label="train",
        )

    if len(val_values) > 0:
        plt.hist(
            val_values,
            bins=edges,
            alpha=0.55,
            density=density,
            label="val",
        )

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Density" if density else "Count")
    if log_y:
        plt.yscale("log")
    plt.legend()

    save_fig(path)


def plot_grouped_bucket_bar(
    df: pd.DataFrame,
    path: Path,
    title: str,
    ylabel: str = "Fraction",
) -> None:
    labels = df["bucket"].tolist()
    x = np.arange(len(labels))
    width = 0.42

    plt.figure(figsize=(13, 6))
    plt.bar(x - width / 2, df["train_fraction"].to_numpy(), width, label="train")
    plt.bar(x + width / 2, df["val_fraction"].to_numpy(), width, label="val")

    plt.xticks(x, labels, rotation=35, ha="right")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()

    save_fig(path)


def image_space_density(
    x_pix: np.ndarray,
    y_pix: np.ndarray,
    image_width: int,
    image_height: int,
    x_bins: int,
    y_bins: int,
    normalize: bool = True,
) -> np.ndarray:
    density, _, _ = np.histogram2d(
        x_pix.astype(float),
        y_pix.astype(float),
        bins=[x_bins, y_bins],
        range=[[0, image_width], [0, image_height]],
    )

    density = density.T

    if normalize:
        total = density.sum()
        if total > 0:
            density = density / total

    return density


def plot_density_image(
    density: np.ndarray,
    path: Path,
    title: str,
    image_width: int,
    image_height: int,
    log_scale: bool = True,
) -> None:
    plt.figure(figsize=(16, 6))

    img = np.log1p(density) if log_scale else density

    plt.imshow(
        img,
        origin="upper",
        extent=[0, image_width, image_height, 0],
        aspect="auto",
    )
    plt.colorbar(label="log(1 + normalized count)" if log_scale else "normalized count")
    plt.title(title)
    plt.xlabel("x pixel")
    plt.ylabel("y pixel")

    save_fig(path)


def plot_density_difference(
    train_density: np.ndarray,
    val_density: np.ndarray,
    path: Path,
    image_width: int,
    image_height: int,
) -> None:
    diff = val_density - train_density
    max_abs = float(np.max(np.abs(diff)))
    if max_abs <= 0:
        max_abs = 1.0

    plt.figure(figsize=(16, 6))
    plt.imshow(
        diff,
        origin="upper",
        extent=[0, image_width, image_height, 0],
        aspect="auto",
        vmin=-max_abs,
        vmax=max_abs,
    )
    plt.colorbar(label="val normalized density minus train normalized density")
    plt.title("Image-space radar GT density difference, val minus train")
    plt.xlabel("x pixel")
    plt.ylabel("y pixel")

    save_fig(path)


def get_radar_point_brightness(points: Dict[str, np.ndarray]) -> np.ndarray:
    rgb = points["rgb"].astype(np.float32) / 255.0
    return rgb.mean(axis=1)


def default_out_dir(train_dir: Path, val_dir: Path) -> Path:
    return Path("work_dirs/ppp_eda") / f"compare_{train_dir.name}_vs_{val_dir.name}"


def main() -> None:
    args = parse_args()

    train_dir = Path(args.train_dir)
    val_dir = Path(args.val_dir)
    out_dir = Path(args.out_dir) if args.out_dir else default_out_dir(train_dir, val_dir)

    prepare_out_dir(out_dir, overwrite=args.overwrite)

    train_frames, train_points = load_scan(train_dir)
    val_frames, val_points = load_scan(val_dir)

    if "point_count" not in train_frames.columns:
        raise KeyError(f"Missing point_count in {train_dir / 'frames.csv'}")
    if "point_count" not in val_frames.columns:
        raise KeyError(f"Missing point_count in {val_dir / 'frames.csv'}")

    # -------------------------------------------------------------------------
    # Summary tables
    # -------------------------------------------------------------------------

    frame_summary = pd.concat(
        [
            make_frame_summary("train", train_frames),
            make_frame_summary("val", val_frames),
        ],
        ignore_index=True,
    )

    point_summary = pd.concat(
        [
            make_point_summary("train", train_points),
            make_point_summary("val", val_points),
        ],
        ignore_index=True,
    )

    point_count_bucket_df = make_point_count_bucket_df(train_frames, val_frames)
    velocity_bucket_df = make_velocity_bucket_df(
        train_points,
        val_points,
        eps=args.velocity_static_eps,
    )

    frame_summary.to_csv(out_dir / "frame_summary_train_val.csv", index=False)
    point_summary.to_csv(out_dir / "point_summary_train_val.csv", index=False)
    point_count_bucket_df.to_csv(out_dir / "point_count_buckets_train_val.csv", index=False)
    velocity_bucket_df.to_csv(out_dir / "velocity_buckets_train_val.csv", index=False)

    # -------------------------------------------------------------------------
    # Frame-level comparisons
    # -------------------------------------------------------------------------

    plot_hist_compare(
        train_values=train_frames["point_count"].to_numpy(dtype=float),
        val_values=val_frames["point_count"].to_numpy(dtype=float),
        path=out_dir / "point_count_train_vs_val.png",
        title="Projected radar GT point count, train vs val",
        xlabel="point_count",
        bins=args.hist_bins,
        clip_quantile=args.clip_quantile,
    )

    plot_grouped_bucket_bar(
        df=point_count_bucket_df,
        path=out_dir / "point_count_bucket_train_vs_val.png",
        title="Projected radar GT point count buckets, train vs val",
    )

    if "brightness_mean" in train_frames.columns and "brightness_mean" in val_frames.columns:
        plot_hist_compare(
            train_values=train_frames["brightness_mean"].to_numpy(dtype=float),
            val_values=val_frames["brightness_mean"].to_numpy(dtype=float),
            path=out_dir / "brightness_train_vs_val.png",
            title="Whole-image brightness, train vs val",
            xlabel="brightness_mean",
            bins=args.hist_bins,
            clip_quantile=1.0,
        )

    if (
        "contrast_q95_minus_q05" in train_frames.columns
        and "contrast_q95_minus_q05" in val_frames.columns
    ):
        plot_hist_compare(
            train_values=train_frames["contrast_q95_minus_q05"].to_numpy(dtype=float),
            val_values=val_frames["contrast_q95_minus_q05"].to_numpy(dtype=float),
            path=out_dir / "contrast_train_vs_val.png",
            title="Whole-image contrast, train vs val",
            xlabel="contrast_q95_minus_q05",
            bins=args.hist_bins,
            clip_quantile=1.0,
        )

    if "laplacian_var" in train_frames.columns and "laplacian_var" in val_frames.columns:
        plot_hist_compare(
            train_values=np.log1p(train_frames["laplacian_var"].to_numpy(dtype=float)),
            val_values=np.log1p(val_frames["laplacian_var"].to_numpy(dtype=float)),
            path=out_dir / "laplacian_var_train_vs_val.png",
            title="Sharpness proxy, train vs val",
            xlabel="log(1 + laplacian_var)",
            bins=args.hist_bins,
            clip_quantile=1.0,
        )

    # -------------------------------------------------------------------------
    # Point-level comparisons
    # -------------------------------------------------------------------------

    train_depth = train_points["depth"].astype(float)
    val_depth = val_points["depth"].astype(float)

    train_velocity = train_points["velocity"].astype(float)
    val_velocity = val_points["velocity"].astype(float)

    plot_hist_compare(
        train_values=train_depth,
        val_values=val_depth,
        path=out_dir / "depth_train_vs_val.png",
        title="GT depth at radar points, train vs val",
        xlabel="GT depth [m]",
        bins=args.hist_bins,
        clip_quantile=args.clip_quantile,
    )

    plot_hist_compare(
        train_values=train_velocity,
        val_values=val_velocity,
        path=out_dir / "velocity_signed_train_vs_val.png",
        title="Signed GT radial velocity, train vs val",
        xlabel="GT radial velocity",
        bins=args.hist_bins,
        symmetric=True,
    )

    plot_hist_compare(
        train_values=train_velocity,
        val_values=val_velocity,
        path=out_dir / "velocity_signed_train_vs_val_log_y.png",
        title="Signed GT radial velocity, train vs val, log-y",
        xlabel="GT radial velocity",
        bins=args.hist_bins,
        symmetric=True,
        log_y=True,
    )

    plot_hist_compare(
        train_values=np.abs(train_velocity),
        val_values=np.abs(val_velocity),
        path=out_dir / "velocity_abs_train_vs_val.png",
        title="Absolute GT radial velocity, train vs val",
        xlabel="abs(GT radial velocity)",
        bins=args.hist_bins,
        clip_quantile=args.clip_quantile,
    )

    plot_hist_compare(
        train_values=np.abs(train_velocity),
        val_values=np.abs(val_velocity),
        path=out_dir / "velocity_abs_train_vs_val_log_y.png",
        title="Absolute GT radial velocity, train vs val, log-y",
        xlabel="abs(GT radial velocity)",
        bins=args.hist_bins,
        clip_quantile=args.clip_quantile,
        log_y=True,
    )

    plot_grouped_bucket_bar(
        df=velocity_bucket_df,
        path=out_dir / "velocity_bucket_train_vs_val.png",
        title=f"GT radial velocity buckets, train vs val, eps={args.velocity_static_eps:g}",
    )

    plot_hist_compare(
        train_values=get_radar_point_brightness(train_points),
        val_values=get_radar_point_brightness(val_points),
        path=out_dir / "radar_point_brightness_train_vs_val.png",
        title="Brightness at projected radar GT points, train vs val",
        xlabel="radar point brightness",
        bins=args.hist_bins,
        clip_quantile=1.0,
    )

    # -------------------------------------------------------------------------
    # Image-space density comparisons
    # -------------------------------------------------------------------------

    train_density = image_space_density(
        x_pix=train_points["x_pix"],
        y_pix=train_points["y_pix"],
        image_width=args.image_width,
        image_height=args.image_height,
        x_bins=args.density_x_bins,
        y_bins=args.density_y_bins,
        normalize=True,
    )

    val_density = image_space_density(
        x_pix=val_points["x_pix"],
        y_pix=val_points["y_pix"],
        image_width=args.image_width,
        image_height=args.image_height,
        x_bins=args.density_x_bins,
        y_bins=args.density_y_bins,
        normalize=True,
    )

    plot_density_image(
        density=train_density,
        path=out_dir / "image_space_density_train.png",
        title="Image-space radar GT density, train",
        image_width=args.image_width,
        image_height=args.image_height,
        log_scale=True,
    )

    plot_density_image(
        density=val_density,
        path=out_dir / "image_space_density_val.png",
        title="Image-space radar GT density, val",
        image_width=args.image_width,
        image_height=args.image_height,
        log_scale=True,
    )

    plot_density_difference(
        train_density=train_density,
        val_density=val_density,
        path=out_dir / "image_space_density_difference_val_minus_train.png",
        image_width=args.image_width,
        image_height=args.image_height,
    )

    # -------------------------------------------------------------------------
    # JSON summary
    # -------------------------------------------------------------------------

    summary = {
        "train_dir": str(train_dir),
        "val_dir": str(val_dir),
        "out_dir": str(out_dir),
        "train_num_frames": int(len(train_frames)),
        "val_num_frames": int(len(val_frames)),
        "train_num_points": int(len(train_points["eda_idx"])),
        "val_num_points": int(len(val_points["eda_idx"])),
        "velocity_static_eps": float(args.velocity_static_eps),
        "image_height": int(args.image_height),
        "image_width": int(args.image_width),
        "outputs": {
            "frame_summary": str(out_dir / "frame_summary_train_val.csv"),
            "point_summary": str(out_dir / "point_summary_train_val.csv"),
            "point_count_buckets": str(out_dir / "point_count_buckets_train_val.csv"),
            "velocity_buckets": str(out_dir / "velocity_buckets_train_val.csv"),
        },
    }

    with open(out_dir / "compare_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Done.")
    print(f"Train dir: {train_dir}")
    print(f"Val dir:   {val_dir}")
    print(f"Out dir:   {out_dir}")
    print(f"Train frames: {len(train_frames)}")
    print(f"Val frames:   {len(val_frames)}")
    print(f"Train points: {len(train_points['eda_idx'])}")
    print(f"Val points:   {len(val_points['eda_idx'])}")


if __name__ == "__main__":
    main()