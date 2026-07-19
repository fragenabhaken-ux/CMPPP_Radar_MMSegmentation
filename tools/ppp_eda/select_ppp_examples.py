#!/usr/bin/env python3
"""
Select representative PPP EDA examples from a scan directory.

Input:
    scan_dir/
        frames.csv
        point_values.npz

Output:
    scan_dir/
        selected_examples.json
        selected_examples.csv

Categories:
    empty
    single_point
    sparse
    dark
    bright
    high_positive_velocity
    high_negative_velocity
    high_mean_abs_velocity
    mostly_static
    high_rcs
    low_rcs

Depth is still computed for selected_examples.csv, but it is no longer used
as a selection category.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select PPP EDA example frames.")

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
        "--out-json",
        type=str,
        default=None,
        help="Output JSON path. Default: scan_dir/selected_examples.json.",
    )
    parser.add_argument(
        "--out-csv",
        type=str,
        default=None,
        help="Output CSV path. Default: scan_dir/selected_examples.csv.",
    )
    parser.add_argument(
        "--num-per-category",
        type=int,
        default=20,
        help="Number of frames to select per category.",
    )
    parser.add_argument(
        "--min-points-for-velocity-stats",
        type=int,
        default=5,
        help="Minimum radar GT points required for velocity categories.",
    )
    parser.add_argument(
        "--min-points-for-rcs-stats",
        type=int,
        default=5,
        help="Minimum radar GT points required for RCS categories.",
    )
    parser.add_argument(
        "--sparse-max-points",
        type=int,
        default=10,
        help="Maximum point_count for sparse category.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing selected_examples outputs.",
    )

    return parser.parse_args()


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")


def maybe_prepare_output(path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"Output already exists: {path}\n"
            f"Use --overwrite to replace it."
        )
    path.parent.mkdir(parents=True, exist_ok=True)


def top_n(
    df: pd.DataFrame,
    column: str,
    n: int,
    ascending: bool,
    min_points: Optional[int] = None,
    require_finite: bool = True,
) -> List[int]:
    if column not in df.columns:
        return []

    sel = df.copy()

    if min_points is not None:
        sel = sel[sel["point_count"] >= min_points]

    if require_finite:
        sel = sel[np.isfinite(sel[column].to_numpy(dtype=float))]

    if len(sel) == 0:
        return []

    sel = sel.sort_values(column, ascending=ascending)
    return sel["eda_idx"].head(n).astype(int).tolist()


def filter_and_rank(
    df: pd.DataFrame,
    condition,
    sort_column: str,
    n: int,
    ascending: bool = True,
) -> List[int]:
    sel = df[condition(df)].copy()

    if len(sel) == 0:
        return []

    if sort_column in sel.columns:
        sel = sel.sort_values(sort_column, ascending=ascending)

    return sel["eda_idx"].head(n).astype(int).tolist()


def load_point_stats(point_values_npz: Path) -> pd.DataFrame:
    pts = np.load(point_values_npz)

    required = ["eda_idx", "depth", "velocity", "rcs"]
    for key in required:
        if key not in pts.files:
            raise KeyError(
                f"{point_values_npz} does not contain array {key!r}. "
                f"Available arrays: {pts.files}"
            )

    eda_idx = pts["eda_idx"].astype(np.int32)
    depth = pts["depth"].astype(np.float32)
    velocity = pts["velocity"].astype(np.float32)
    rcs = pts["rcs"].astype(np.float32)

    if len(eda_idx) == 0:
        return pd.DataFrame(
            columns=[
                "eda_idx",
                "depth_count",
                "depth_min",
                "depth_q05",
                "depth_q50",
                "depth_q95",
                "depth_max",
                "depth_mean",
                "velocity_mean",
                "velocity_abs_mean",
                "velocity_abs_max",
                "velocity_q05",
                "velocity_q50",
                "velocity_q95",
                "velocity_pos_fraction_eps_0_5",
                "velocity_neg_fraction_eps_0_5",
                "velocity_near_zero_fraction_eps_0_5",
                "rcs_min",
                "rcs_q05",
                "rcs_q50",
                "rcs_q95",
                "rcs_max",
                "rcs_mean",
            ]
        )

    point_df = pd.DataFrame(
        {
            "eda_idx": eda_idx,
            "depth": depth,
            "velocity": velocity,
            "rcs": rcs,
        }
    )
    point_df["abs_velocity"] = point_df["velocity"].abs()

    grouped = point_df.groupby("eda_idx", sort=False)

    stats = grouped.agg(
        depth_count=("depth", "size"),
        depth_min=("depth", "min"),
        depth_max=("depth", "max"),
        depth_mean=("depth", "mean"),
        velocity_mean=("velocity", "mean"),
        velocity_abs_mean=("abs_velocity", "mean"),
        velocity_abs_max=("abs_velocity", "max"),
        rcs_min=("rcs", "min"),
        rcs_max=("rcs", "max"),
        rcs_mean=("rcs", "mean"),
    )

    depth_q = grouped["depth"].quantile([0.05, 0.50, 0.95]).unstack()
    depth_q.columns = ["depth_q05", "depth_q50", "depth_q95"]

    velocity_q = grouped["velocity"].quantile([0.05, 0.50, 0.95]).unstack()
    velocity_q.columns = ["velocity_q05", "velocity_q50", "velocity_q95"]

    rcs_q = grouped["rcs"].quantile([0.05, 0.50, 0.95]).unstack()
    rcs_q.columns = ["rcs_q05", "rcs_q50", "rcs_q95"]

    eps = 0.5
    velocity_flags = point_df.assign(
        vel_pos=(point_df["velocity"] > eps).astype(np.float32),
        vel_neg=(point_df["velocity"] < -eps).astype(np.float32),
        vel_zero=(point_df["velocity"].abs() <= eps).astype(np.float32),
    ).groupby("eda_idx", sort=False).agg(
        velocity_pos_fraction_eps_0_5=("vel_pos", "mean"),
        velocity_neg_fraction_eps_0_5=("vel_neg", "mean"),
        velocity_near_zero_fraction_eps_0_5=("vel_zero", "mean"),
    )

    out = stats.join(depth_q).join(velocity_q).join(rcs_q).join(velocity_flags)
    out = out.reset_index()

    return out


def build_selected_examples(
    df: pd.DataFrame,
    n: int,
    min_points_for_velocity_stats: int,
    min_points_for_rcs_stats: int,
    sparse_max_points: int,
) -> Dict[str, List[int]]:
    selected: Dict[str, List[int]] = {}

    selected["empty"] = filter_and_rank(
        df,
        condition=lambda x: x["point_count"] == 0,
        sort_column="eda_idx",
        n=n,
        ascending=True,
    )

    selected["single_point"] = filter_and_rank(
        df,
        condition=lambda x: x["point_count"] == 1,
        sort_column="eda_idx",
        n=n,
        ascending=True,
    )

    selected["sparse"] = filter_and_rank(
        df,
        condition=lambda x: (x["point_count"] > 1)
        & (x["point_count"] <= sparse_max_points),
        sort_column="point_count",
        n=n,
        ascending=True,
    )

    selected["dark"] = top_n(
        df,
        column="brightness_mean",
        n=n,
        ascending=True,
    )

    selected["bright"] = top_n(
        df,
        column="brightness_mean",
        n=n,
        ascending=False,
    )

    selected["high_positive_velocity"] = top_n(
        df,
        column="velocity_mean",
        n=n,
        ascending=False,
        min_points=min_points_for_velocity_stats,
    )

    selected["high_negative_velocity"] = top_n(
        df,
        column="velocity_mean",
        n=n,
        ascending=True,
        min_points=min_points_for_velocity_stats,
    )

    selected["high_mean_abs_velocity"] = top_n(
        df,
        column="velocity_abs_mean",
        n=n,
        ascending=False,
        min_points=min_points_for_velocity_stats,
    )

    selected["mostly_static"] = top_n(
        df,
        column="velocity_abs_mean",
        n=n,
        ascending=True,
        min_points=min_points_for_velocity_stats,
    )

    selected["high_rcs"] = top_n(
        df,
        column="rcs_mean",
        n=n,
        ascending=False,
        min_points=min_points_for_rcs_stats,
    )

    selected["low_rcs"] = top_n(
        df,
        column="rcs_mean",
        n=n,
        ascending=True,
        min_points=min_points_for_rcs_stats,
    )

    return selected


def build_selected_csv(df: pd.DataFrame, selected: Dict[str, List[int]]) -> pd.DataFrame:
    rows = []

    for category, eda_indices in selected.items():
        for rank, eda_idx in enumerate(eda_indices, start=1):
            match = df[df["eda_idx"] == eda_idx]

            if len(match) == 0:
                rows.append(
                    {
                        "category": category,
                        "rank": rank,
                        "eda_idx": eda_idx,
                    }
                )
                continue

            row = match.iloc[0].to_dict()
            row["category"] = category
            row["rank"] = rank
            rows.append(row)

    if len(rows) == 0:
        return pd.DataFrame(columns=["category", "rank", "eda_idx"])

    out = pd.DataFrame(rows)

    first_cols = [
        "category",
        "rank",
        "eda_idx",
        "idx",
        "demo_idx",
        "global_idx",
        "sample_id",
        "video_name",
        "video_path",
        "video_frame_idx",
        "point_count",
        "brightness_mean",
        "brightness_std",
        "contrast_q95_minus_q05",
        "laplacian_var",
        "depth_q05",
        "depth_q50",
        "depth_q95",
        "velocity_mean",
        "velocity_abs_mean",
        "velocity_abs_max",
        "velocity_q05",
        "velocity_q50",
        "velocity_q95",
        "velocity_pos_fraction_eps_0_5",
        "velocity_neg_fraction_eps_0_5",
        "velocity_near_zero_fraction_eps_0_5",
        "rcs_mean",
        "rcs_min",
        "rcs_max",
        "rcs_q05",
        "rcs_q50",
        "rcs_q95",
    ]

    cols = [c for c in first_cols if c in out.columns]
    cols += [c for c in out.columns if c not in cols]

    return out[cols]


def main() -> None:
    args = parse_args()

    scan_dir = Path(args.scan_dir)

    frames_csv = Path(args.frames_csv) if args.frames_csv else scan_dir / "frames.csv"
    point_values_npz = (
        Path(args.point_values_npz)
        if args.point_values_npz
        else scan_dir / "point_values.npz"
    )

    out_json = Path(args.out_json) if args.out_json else scan_dir / "selected_examples.json"
    out_csv = Path(args.out_csv) if args.out_csv else scan_dir / "selected_examples.csv"

    require_file(frames_csv)
    require_file(point_values_npz)

    maybe_prepare_output(out_json, overwrite=args.overwrite)
    maybe_prepare_output(out_csv, overwrite=args.overwrite)

    frames = pd.read_csv(frames_csv)
    point_stats = load_point_stats(point_values_npz)

    if "eda_idx" not in frames.columns:
        raise KeyError("frames.csv must contain column 'eda_idx'.")

    if "point_count" not in frames.columns:
        raise KeyError("frames.csv must contain column 'point_count'.")

    df = frames.merge(point_stats, on="eda_idx", how="left")

    selected = build_selected_examples(
        df=df,
        n=args.num_per_category,
        min_points_for_velocity_stats=args.min_points_for_velocity_stats,
        min_points_for_rcs_stats=args.min_points_for_rcs_stats,
        sparse_max_points=args.sparse_max_points,
    )

    selected_csv = build_selected_csv(df, selected)

    pts = np.load(point_values_npz)

    payload = {
        "meta": {
            "scan_dir": str(scan_dir),
            "frames_csv": str(frames_csv),
            "point_values_npz": str(point_values_npz),
            "num_frames": int(len(frames)),
            "num_point_rows": int(pts["eda_idx"].shape[0]),
            "num_per_category": int(args.num_per_category),
            "min_points_for_velocity_stats": int(args.min_points_for_velocity_stats),
            "min_points_for_rcs_stats": int(args.min_points_for_rcs_stats),
            "sparse_max_points": int(args.sparse_max_points),
        },
        "categories": selected,
    }

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    selected_csv.to_csv(out_csv, index=False)

    print("Done.")
    print(f"Saved JSON: {out_json}")
    print(f"Saved CSV:  {out_csv}")
    print()
    print("Selected counts:")
    for category, values in selected.items():
        print(f"  {category:28s} {len(values):4d}")


if __name__ == "__main__":
    main()