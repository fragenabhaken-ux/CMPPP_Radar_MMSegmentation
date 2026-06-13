#!/usr/bin/env python3
"""
Render selected PPP EDA examples.

This script rebuilds the same EDA dataloader used during scanning and renders
selected eda_idx frames.

Important safety checks:
    1. point_count check:
       recomputed GT point count must match frames.csv point_count.

    2. metadata check:
       available metadata such as idx, global_idx, sample_id, video_path,
       video_frame_idx must match frames.csv.

If these checks fail, scan and render do not point to the same samples.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import re
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from mmengine.config import Config
from mmengine.registry import init_default_scope
from mmengine.runner import Runner
from mmengine.utils import import_modules_from_strings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render selected PPP EDA examples.")

    parser.add_argument(
        "config",
        type=str,
        help="Path to EDA config, e.g. configs/ppp_eda/ppp_eda.py.",
    )
    parser.add_argument(
        "--dataloader",
        type=str,
        default="eda_val_dataloader",
        help="Dataloader variable in config.",
    )
    parser.add_argument(
        "--scan-dir",
        type=str,
        required=True,
        help="Scan directory containing selected_examples.json and frames.csv.",
    )
    parser.add_argument(
        "--selected-json",
        type=str,
        default=None,
        help="Optional explicit path to selected_examples.json.",
    )
    parser.add_argument(
        "--frames-csv",
        type=str,
        default=None,
        help="Optional explicit path to frames.csv.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Output directory. Default: scan_dir/examples.",
    )
    parser.add_argument(
        "--categories",
        type=str,
        nargs="*",
        default=None,
        help="Optional subset of categories to render.",
    )
    parser.add_argument(
        "--max-per-category",
        type=int,
        default=None,
        help="Optional maximum number of examples to render per category.",
    )
    parser.add_argument(
        "--point-size",
        type=float,
        default=7.0,
        help="Scatter point size. Matplotlib uses marker area.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.95,
        help="Scatter point alpha.",
    )
    parser.add_argument(
        "--fig-width",
        type=float,
        default=16.0,
        help="Figure width in inches.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=140,
        help="Saved image DPI.",
    )
    parser.add_argument(
        "--depth-vmin",
        type=float,
        default=0.0,
        help="Depth color minimum.",
    )
    parser.add_argument(
        "--depth-vmax",
        type=float,
        default=150.0,
        help="Depth color maximum.",
    )
    parser.add_argument(
        "--velocity-vmax",
        type=float,
        default=10.0,
        help="Velocity color range is [-velocity_vmax, +velocity_vmax].",
    )
    parser.add_argument(
        "--skip-verification",
        action="store_true",
        help="Disable point_count and metadata checks. Not recommended.",
    )
    parser.add_argument(
        "--metadata-keys",
        type=str,
        nargs="*",
        default=[
            "idx",
            "demo_idx",
            "global_idx",
            "sample_id",
            "video_name",
            "video_path",
            "video_frame_idx",
        ],
        help="Metadata keys to compare between frames.csv and rendered sample.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output directory if it exists.",
    )

    return parser.parse_args()


def to_numpy(x: Any) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def squeeze_optional_batch_dim(arr: np.ndarray, name: str) -> np.ndarray:
    if arr.ndim == 4:
        if arr.shape[0] != 1:
            raise ValueError(
                f"{name} has batch_size={arr.shape[0]}. "
                f"Use batch_size=1 for this renderer."
            )
        arr = arr[0]
    return arr


def normalize_img_to_hwc_float01(img: Any) -> np.ndarray:
    img = to_numpy(img)
    img = squeeze_optional_batch_dim(img, name="img")

    if img.ndim != 3:
        raise ValueError(f"Expected image ndim=3 after squeezing, got {img.shape}")

    if img.shape[0] == 3 and img.shape[-1] != 3:
        img = np.transpose(img, (1, 2, 0))

    if img.shape[-1] != 3:
        raise ValueError(f"Expected RGB image with last dim 3, got {img.shape}")

    img = img.astype(np.float32, copy=False)

    # Your EDA pipeline returns image in [-1, 1].
    img01 = np.clip((img + 1.0) / 2.0, 0.0, 1.0)

    return img01


def normalize_gt_to_hwc(gt: Any) -> np.ndarray:
    gt = to_numpy(gt)
    gt = squeeze_optional_batch_dim(gt, name="gt_seg_map")

    if gt.ndim != 3:
        raise ValueError(f"Expected gt_seg_map ndim=3 after squeezing, got {gt.shape}")

    if gt.shape[0] == 3 and gt.shape[-1] != 3:
        gt = np.transpose(gt, (1, 2, 0))

    if gt.shape[-1] != 3:
        raise ValueError(f"Expected gt_seg_map with last dim 3, got {gt.shape}")

    return gt.astype(np.float32, copy=False)


def unpack_batch(batch: Any) -> List[Dict[str, Any]]:
    if isinstance(batch, list):
        return batch
    if isinstance(batch, tuple):
        return list(batch)
    if isinstance(batch, dict):
        return [batch]
    raise TypeError(f"Unsupported batch type: {type(batch)}")


def scalarize(x: Any) -> Any:
    if x is None:
        return None

    if torch.is_tensor(x):
        x = x.detach().cpu().numpy()

    if isinstance(x, np.ndarray):
        if x.size == 1:
            return x.reshape(-1)[0].item()
        return x.tolist()

    if isinstance(x, np.generic):
        return x.item()

    if isinstance(x, list) and len(x) == 1:
        return scalarize(x[0])

    if isinstance(x, tuple) and len(x) == 1:
        return scalarize(x[0])

    return x


def is_missing(x: Any) -> bool:
    if x is None:
        return True

    if isinstance(x, float) and math.isnan(x):
        return True

    try:
        if pd.isna(x):
            return True
    except Exception:
        pass

    return False


def normalize_for_compare(x: Any) -> Any:
    x = scalarize(x)

    if is_missing(x):
        return None

    if isinstance(x, bytes):
        return x.decode("utf-8")

    if isinstance(x, str):
        s = x.strip()
        if s.lower() in {"nan", "none", ""}:
            return None

        # Convert integer-looking strings safely.
        try:
            if re.fullmatch(r"[-+]?\d+", s):
                return int(s)
        except Exception:
            pass

        # Convert float-looking strings safely.
        try:
            if re.fullmatch(r"[-+]?\d+(\.\d+)?", s):
                f = float(s)
                if f.is_integer():
                    return int(f)
                return f
        except Exception:
            pass

        return s

    if isinstance(x, float):
        if x.is_integer():
            return int(x)
        return x

    return x


def values_equal(a: Any, b: Any) -> bool:
    a = normalize_for_compare(a)
    b = normalize_for_compare(b)

    if a is None and b is None:
        return True

    if a is None or b is None:
        return False

    return a == b


def get_meta_value(sample: Dict[str, Any], key: str, default: Any = None) -> Any:
    meta = sample.get("metainfo", {})
    if isinstance(meta, dict) and key in meta:
        return scalarize(meta[key])
    return scalarize(sample.get(key, default))


def sample_identity(sample: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "idx": scalarize(sample.get("idx", None)),
        "demo_idx": scalarize(sample.get("demo_idx", None)),
        "global_idx": get_meta_value(sample, "global_idx", sample.get("idx", None)),
        "sample_id": get_meta_value(sample, "sample_id", None),
        "video_name": get_meta_value(sample, "video_name", None),
        "video_path": get_meta_value(sample, "video_path", None),
        "video_frame_idx": get_meta_value(sample, "video_frame_idx", None),
    }


def sanitize_filename_part(x: str) -> str:
    x = str(x)
    x = re.sub(r"[^a-zA-Z0-9_+-]+", "_", x)
    x = re.sub(r"_+", "_", x)
    return x.strip("_")


def load_config(config_path: str) -> Config:
    cfg = Config.fromfile(config_path)

    if hasattr(cfg, "custom_imports"):
        import_modules_from_strings(**cfg.custom_imports)

    init_default_scope("mmseg")
    return cfg


def build_dataloader_from_cfg(cfg: Config, dataloader_name: str):
    if not hasattr(cfg, dataloader_name):
        available = [
            k for k in cfg.keys()
            if k.endswith("_dataloader") or k == "dataloader"
        ]
        raise KeyError(
            f"Config has no dataloader named {dataloader_name!r}. "
            f"Available dataloader-like keys: {available}"
        )

    dataloader_cfg = copy.deepcopy(getattr(cfg, dataloader_name))
    return Runner.build_dataloader(dataloader_cfg)


def prepare_out_dir(out_dir: Path, overwrite: bool) -> None:
    if out_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output directory already exists: {out_dir}\n"
                f"Use --overwrite if you want to replace it."
            )
        shutil.rmtree(out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)


def load_selected_examples(
    selected_json: Path,
    categories: List[str] | None,
    max_per_category: int | None,
) -> Dict[str, List[int]]:
    with open(selected_json, "r", encoding="utf-8") as f:
        payload = json.load(f)

    if "categories" in payload:
        selected = payload["categories"]
    else:
        selected = payload

    if categories is not None:
        keep = set(categories)
        selected = {k: v for k, v in selected.items() if k in keep}

    out: Dict[str, List[int]] = {}
    for category, values in selected.items():
        values = [int(v) for v in values]
        if max_per_category is not None:
            values = values[:max_per_category]
        out[str(category)] = values

    return out


def invert_selected(selected: Dict[str, List[int]]) -> Dict[int, List[Tuple[str, int]]]:
    out: Dict[int, List[Tuple[str, int]]] = defaultdict(list)

    for category, eda_indices in selected.items():
        for rank, eda_idx in enumerate(eda_indices, start=1):
            out[int(eda_idx)].append((category, rank))

    return dict(out)


def frame_row_to_dict(row: pd.Series | Dict[str, Any] | None) -> Dict[str, Any] | None:
    if row is None:
        return None
    if isinstance(row, dict):
        return row
    return row.to_dict()


def verify_sample_matches_scan(
    *,
    eda_idx: int,
    sample: Dict[str, Any],
    gt: np.ndarray,
    frame_row: Dict[str, Any] | None,
    metadata_keys: List[str],
) -> None:
    if frame_row is None:
        raise RuntimeError(
            f"EDA verification failed: eda_idx={eda_idx} is missing in frames.csv."
        )

    # -------------------------------------------------------------------------
    # Check 1: point_count
    # -------------------------------------------------------------------------
    render_point_count = int((gt[..., 0] > 0).sum())

    if "point_count" not in frame_row:
        raise RuntimeError("frames.csv has no column 'point_count'.")

    scan_point_count = int(normalize_for_compare(frame_row["point_count"]))

    if render_point_count != scan_point_count:
        identity = sample_identity(sample)
        raise RuntimeError(
            "\nEDA SAMPLE MISMATCH: point_count differs.\n"
            f"  eda_idx:              {eda_idx}\n"
            f"  scan point_count:     {scan_point_count}\n"
            f"  render point_count:   {render_point_count}\n"
            "\nThis means the scan dataloader and render dataloader do not "
            "point to the same sample.\n"
            "\nRendered sample identity:\n"
            + "\n".join(f"  {k}: {v}" for k, v in identity.items())
            + "\n\nframes.csv row:\n"
            + "\n".join(f"  {k}: {frame_row.get(k, None)}" for k in frame_row.keys())
        )

    # -------------------------------------------------------------------------
    # Check 2: metadata
    # -------------------------------------------------------------------------
    identity = sample_identity(sample)
    mismatches = []

    for key in metadata_keys:
        if key not in frame_row:
            continue

        scan_value = frame_row.get(key, None)
        render_value = identity.get(key, None)

        # Only compare if at least one side is present.
        if is_missing(scan_value) and is_missing(render_value):
            continue

        if not values_equal(scan_value, render_value):
            mismatches.append((key, scan_value, render_value))

    if mismatches:
        msg = [
            "",
            "EDA SAMPLE MISMATCH: metadata differs.",
            f"  eda_idx: {eda_idx}",
            "",
            "Mismatching keys:",
        ]

        for key, scan_value, render_value in mismatches:
            msg.append(f"  {key}:")
            msg.append(f"    frames.csv: {scan_value}")
            msg.append(f"    rendered:   {render_value}")

        msg.extend(
            [
                "",
                "This means the scan dataloader and render dataloader do not "
                "point to the same sample.",
            ]
        )

        raise RuntimeError("\n".join(msg))


def build_title(
    category: str,
    rank: int,
    eda_idx: int,
    frame_row: Dict[str, Any] | None,
    suffix: str,
) -> str:
    parts = [
        f"{category}",
        f"rank {rank}",
        f"eda_idx {eda_idx}",
        suffix,
    ]

    if frame_row is not None:
        if "point_count" in frame_row:
            parts.append(f"points {int(frame_row['point_count'])}")
        if "video_frame_idx" in frame_row and pd.notna(frame_row["video_frame_idx"]):
            parts.append(f"video_frame {frame_row['video_frame_idx']}")

    return " | ".join(parts)


def extract_points(gt: np.ndarray) -> Dict[str, np.ndarray]:
    mask = gt[..., 0] > 0
    y_pix, x_pix = np.nonzero(mask)

    depth = gt[..., 1][mask]
    velocity = gt[..., 2][mask]

    return dict(
        x=x_pix.astype(np.float32, copy=False),
        y=y_pix.astype(np.float32, copy=False),
        depth=depth.astype(np.float32, copy=False),
        velocity=velocity.astype(np.float32, copy=False),
    )


def make_figure(img01: np.ndarray, title: str, fig_width: float):
    h, w = img01.shape[:2]
    fig_height = fig_width * h / w

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.imshow(img01)
    ax.set_title(title, fontsize=10)
    ax.axis("off")

    return fig, ax


def save_points_overlay(
    img01: np.ndarray,
    points: Dict[str, np.ndarray],
    out_path: Path,
    title: str,
    point_size: float,
    alpha: float,
    fig_width: float,
    dpi: int,
) -> None:
    fig, ax = make_figure(img01, title=title, fig_width=fig_width)

    if len(points["x"]) > 0:
        ax.scatter(
            points["x"],
            points["y"],
            s=point_size,
            c="white",
            alpha=alpha,
            edgecolors="black",
            linewidths=0.20,
        )
    else:
        ax.text(
            0.5,
            0.5,
            "No projected GT radar points",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=14,
            color="white",
            bbox=dict(facecolor="black", alpha=0.65, pad=8),
        )

    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def save_depth_overlay(
    img01: np.ndarray,
    points: Dict[str, np.ndarray],
    out_path: Path,
    title: str,
    point_size: float,
    alpha: float,
    fig_width: float,
    dpi: int,
    depth_vmin: float,
    depth_vmax: float,
) -> None:
    fig, ax = make_figure(img01, title=title, fig_width=fig_width)

    if len(points["x"]) > 0:
        sc = ax.scatter(
            points["x"],
            points["y"],
            s=point_size,
            c=points["depth"],
            cmap="turbo",
            vmin=depth_vmin,
            vmax=depth_vmax,
            alpha=alpha,
            edgecolors="black",
            linewidths=0.20,
        )

        cbar = fig.colorbar(sc, ax=ax, fraction=0.025, pad=0.01)
        cbar.set_label("GT depth [m]")
    else:
        ax.text(
            0.5,
            0.5,
            "No projected GT radar points",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=14,
            color="white",
            bbox=dict(facecolor="black", alpha=0.65, pad=8),
        )

    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def save_velocity_overlay(
    img01: np.ndarray,
    points: Dict[str, np.ndarray],
    out_path: Path,
    title: str,
    point_size: float,
    alpha: float,
    fig_width: float,
    dpi: int,
    velocity_vmax: float,
) -> None:
    fig, ax = make_figure(img01, title=title, fig_width=fig_width)

    if len(points["x"]) > 0:
        sc = ax.scatter(
            points["x"],
            points["y"],
            s=point_size,
            c=points["velocity"],
            cmap="coolwarm",
            vmin=-velocity_vmax,
            vmax=velocity_vmax,
            alpha=alpha,
            edgecolors="black",
            linewidths=0.20,
        )

        cbar = fig.colorbar(sc, ax=ax, fraction=0.025, pad=0.01)
        cbar.set_label("GT radial velocity")
    else:
        ax.text(
            0.5,
            0.5,
            "No projected GT radar points",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=14,
            color="white",
            bbox=dict(facecolor="black", alpha=0.65, pad=8),
        )

    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def render_one_selection(
    img01: np.ndarray,
    gt: np.ndarray,
    out_dir: Path,
    category: str,
    rank: int,
    eda_idx: int,
    frame_row: Dict[str, Any] | None,
    args: argparse.Namespace,
) -> None:
    safe_category = sanitize_filename_part(category)
    category_dir = out_dir / safe_category
    category_dir.mkdir(parents=True, exist_ok=True)

    base = f"{safe_category}_rank{rank:03d}_eda{eda_idx:06d}"

    points = extract_points(gt)

    save_points_overlay(
        img01=img01,
        points=points,
        out_path=category_dir / f"{base}_points.png",
        title=build_title(category, rank, eda_idx, frame_row, "points"),
        point_size=args.point_size,
        alpha=args.alpha,
        fig_width=args.fig_width,
        dpi=args.dpi,
    )

    save_depth_overlay(
        img01=img01,
        points=points,
        out_path=category_dir / f"{base}_depth.png",
        title=build_title(category, rank, eda_idx, frame_row, "depth"),
        point_size=args.point_size,
        alpha=args.alpha,
        fig_width=args.fig_width,
        dpi=args.dpi,
        depth_vmin=args.depth_vmin,
        depth_vmax=args.depth_vmax,
    )

    save_velocity_overlay(
        img01=img01,
        points=points,
        out_path=category_dir / f"{base}_velocity.png",
        title=build_title(category, rank, eda_idx, frame_row, "velocity"),
        point_size=args.point_size,
        alpha=args.alpha,
        fig_width=args.fig_width,
        dpi=args.dpi,
        velocity_vmax=args.velocity_vmax,
    )


def main() -> None:
    args = parse_args()

    scan_dir = Path(args.scan_dir)
    selected_json = (
        Path(args.selected_json)
        if args.selected_json is not None
        else scan_dir / "selected_examples.json"
    )
    frames_csv = (
        Path(args.frames_csv)
        if args.frames_csv is not None
        else scan_dir / "frames.csv"
    )
    out_dir = Path(args.out_dir) if args.out_dir is not None else scan_dir / "examples"

    if not selected_json.exists():
        raise FileNotFoundError(f"Missing selected_examples.json: {selected_json}")

    if not frames_csv.exists():
        raise FileNotFoundError(f"Missing frames.csv: {frames_csv}")

    prepare_out_dir(out_dir, overwrite=args.overwrite)

    selected = load_selected_examples(
        selected_json=selected_json,
        categories=args.categories,
        max_per_category=args.max_per_category,
    )

    selected_by_eda_idx = invert_selected(selected)
    wanted_eda_indices = set(selected_by_eda_idx.keys())

    if len(wanted_eda_indices) == 0:
        print("No selected examples to render.")
        return

    max_needed_eda_idx = max(wanted_eda_indices)

    frames_df = pd.read_csv(frames_csv)
    if "eda_idx" not in frames_df.columns:
        raise KeyError("frames.csv must contain column 'eda_idx'.")

    frames_by_eda_idx = {
        int(row["eda_idx"]): row.to_dict()
        for _, row in frames_df.iterrows()
    }

    cfg = load_config(args.config)
    loader = build_dataloader_from_cfg(cfg, args.dataloader)

    rendered = 0
    visited = 0
    verified = 0
    rendered_eda_indices = set()

    try:
        total = min(len(loader.dataset), max_needed_eda_idx + 1)
    except Exception:
        total = max_needed_eda_idx + 1

    pbar = tqdm(loader, total=total, desc="Rendering selected PPP examples")

    stop = False

    for batch in pbar:
        samples = unpack_batch(batch)

        for sample in samples:
            eda_idx = visited
            visited += 1

            if eda_idx in wanted_eda_indices:
                img01 = normalize_img_to_hwc_float01(sample["img"])
                gt = normalize_gt_to_hwc(sample["gt_seg_map"])

                if img01.shape[:2] != gt.shape[:2]:
                    raise ValueError(
                        f"Image and GT shape mismatch for eda_idx={eda_idx}: "
                        f"img={img01.shape}, gt={gt.shape}"
                    )

                frame_row = frames_by_eda_idx.get(eda_idx)

                if not args.skip_verification:
                    verify_sample_matches_scan(
                        eda_idx=eda_idx,
                        sample=sample,
                        gt=gt,
                        frame_row=frame_row,
                        metadata_keys=args.metadata_keys,
                    )
                    verified += 1

                for category, rank in selected_by_eda_idx[eda_idx]:
                    render_one_selection(
                        img01=img01,
                        gt=gt,
                        out_dir=out_dir,
                        category=category,
                        rank=rank,
                        eda_idx=eda_idx,
                        frame_row=frame_row,
                        args=args,
                    )
                    rendered += 1

                rendered_eda_indices.add(eda_idx)
                pbar.set_postfix(rendered=rendered, verified=verified, eda_idx=eda_idx)

            if visited > max_needed_eda_idx:
                stop = True
                break

        if stop:
            break

    missing_not_visited = sorted(wanted_eda_indices - rendered_eda_indices)
    if missing_not_visited:
        raise RuntimeError(
            "Some selected eda_idx values were not rendered because they were not reached:\n"
            + ", ".join(map(str, missing_not_visited))
        )

    summary = {
        "config": str(args.config),
        "dataloader": args.dataloader,
        "scan_dir": str(scan_dir),
        "selected_json": str(selected_json),
        "frames_csv": str(frames_csv),
        "out_dir": str(out_dir),
        "verification_enabled": not args.skip_verification,
        "metadata_keys": args.metadata_keys,
        "num_categories": int(len(selected)),
        "num_unique_eda_indices": int(len(wanted_eda_indices)),
        "num_verified_eda_indices": int(verified),
        "num_rendered_category_examples": int(rendered),
        "max_needed_eda_idx": int(max_needed_eda_idx),
        "visited_samples": int(visited),
        "categories": {k: [int(x) for x in v] for k, v in selected.items()},
    }

    summary_path = out_dir / "render_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Done.")
    print(f"Saved examples to: {out_dir}")
    print(f"Saved summary to:  {summary_path}")
    print(f"Verified eda_idx:  {verified}")
    print(f"Rendered category examples: {rendered}")


if __name__ == "__main__":
    main()