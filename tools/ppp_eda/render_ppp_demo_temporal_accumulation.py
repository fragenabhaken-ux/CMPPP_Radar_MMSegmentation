#!/usr/bin/env python3
"""
Render temporal accumulation of projected PPP radar GT points for one demo frame.

Neighboring frames are loaded with the demo dataloader. Their projected camera
points are drawn onto the center image and are also converted into Cartesian BEV
and polar coordinates. None of the accumulated views is ego-motion compensated.

Example:

python tools/ppp_eda/render_ppp_demo_temporal_accumulation.py \
  configs/ppp_eda/ppp_eda.py \
  --dataloader eda_demo_dataloader \
  --center-eda-idx 160 \
  --radius 20 \
  --stride 1 \
  --out-dir work_dirs/ppp_eda/demo_temporal_accumulation \
  --overwrite
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Dict, List

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ppp_eda_attribute_limits import (
    DEPTH_DISPLAY_RANGE,
    RCS_DISPLAY_RANGE,
    VELOCITY_DISPLAY_RANGE,
)
from ppp_eda_geometry import (
    camera_domain_mask,
    camera_plot_limits,
    cartesian_domain_mask,
    cartesian_plot_limits,
    points_from_gt,
    polar_domain_mask,
    polar_plot_limits,
    resolve_spatial_config_from_cfg,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render temporal accumulation of projected demo radar GT points."
    )

    parser.add_argument(
        "config",
        type=str,
        help="EDA config path, e.g. configs/ppp_eda/ppp_eda.py.",
    )
    parser.add_argument(
        "--dataloader",
        type=str,
        default="eda_demo_dataloader",
        help="Config key of the demo dataloader.",
    )
    parser.add_argument(
        "--center-eda-idx",
        type=int,
        required=True,
        help="Center local demo dataloader index.",
    )
    parser.add_argument(
        "--radius",
        type=int,
        default=20,
        help="Number of neighboring frames before and after the center frame.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Use every k-th frame inside the temporal window.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        required=True,
        help="Output directory.",
    )
    parser.add_argument(
        "--point-size",
        type=float,
        default=7.0,
        help="Scatter point size. Matplotlib uses area, not diameter.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.85,
        help="Base point alpha.",
    )
    parser.add_argument(
        "--alpha-min",
        type=float,
        default=0.12,
        help="Minimum alpha for alpha-fade plot.",
    )
    parser.add_argument(
        "--depth-vmin",
        type=float,
        default=DEPTH_DISPLAY_RANGE[0],
        help="Depth colorbar minimum.",
    )
    parser.add_argument(
        "--depth-vmax",
        type=float,
        default=DEPTH_DISPLAY_RANGE[1],
        help="Depth colorbar maximum.",
    )
    parser.add_argument(
        "--velocity-vmin",
        type=float,
        default=VELOCITY_DISPLAY_RANGE[0],
        help="Velocity colorbar minimum.",
    )
    parser.add_argument(
        "--velocity-vmax",
        type=float,
        default=VELOCITY_DISPLAY_RANGE[1],
        help="Velocity colorbar maximum.",
    )
    parser.add_argument(
        "--rcs-vmin",
        type=float,
        default=RCS_DISPLAY_RANGE[0],
        help="Fixed RCS colorbar minimum.",
    )
    parser.add_argument(
        "--rcs-vmax",
        type=float,
        default=RCS_DISPLAY_RANGE[1],
        help="Fixed RCS colorbar maximum.",
    )
    parser.add_argument(
        "--save-csv",
        action="store_true",
        help="Also save accumulated_points.csv.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output directory if it exists.",
    )

    return parser.parse_args()


def prepare_out_dir(out_dir: Path, overwrite: bool) -> None:
    if out_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output directory already exists: {out_dir}\n"
                f"Use --overwrite to replace it."
            )
        shutil.rmtree(out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)


def to_numpy(x: Any) -> np.ndarray:
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    elif hasattr(x, "cpu") and hasattr(x, "numpy"):
        x = x.cpu().numpy()
    else:
        x = np.asarray(x)
    return x


def squeeze_optional_batch_dim(arr: np.ndarray, name: str) -> np.ndarray:
    arr = to_numpy(arr)

    if arr.ndim == 4:
        if arr.shape[0] != 1:
            raise ValueError(
                f"{name} has batch dimension {arr.shape[0]}. "
                f"This script expects batch_size=1."
            )
        arr = arr[0]

    return arr


def normalize_img_to_hwc_float01(img: Any) -> np.ndarray:
    img = squeeze_optional_batch_dim(to_numpy(img), "img")

    if img.ndim != 3:
        raise ValueError(f"Expected img with 3 dims, got shape {img.shape}")

    # Support HWC and CHW just in case.
    if img.shape[0] in (1, 3) and img.shape[-1] not in (1, 3):
        img = np.transpose(img, (1, 2, 0))

    if img.shape[-1] != 3:
        raise ValueError(f"Expected RGB image with 3 channels, got shape {img.shape}")

    img = img.astype(np.float32)

    # SYS100 loader usually gives [-1, 1].
    if img.min() < 0.0:
        img = (img + 1.0) / 2.0

    img = np.clip(img, 0.0, 1.0)

    return img


def normalize_gt_to_hwc(gt: Any) -> np.ndarray:
    gt = squeeze_optional_batch_dim(to_numpy(gt), "gt_seg_map")

    if gt.ndim != 3:
        raise ValueError(f"Expected gt_seg_map with 3 dims, got shape {gt.shape}")

    # Support CHW just in case.
    if gt.shape[0] == 4 and gt.shape[-1] != 4:
        gt = np.transpose(gt, (1, 2, 0))

    if gt.shape[-1] < 4:
        raise ValueError(
            f"Expected gt_seg_map with at least 4 channels, got shape {gt.shape}"
        )

    return gt.astype(np.float32)


def unwrap_batch(batch: Any) -> Dict[str, Any]:
    if isinstance(batch, (list, tuple)):
        if len(batch) != 1:
            raise ValueError(
                f"This script expects batch_size=1. Got list/tuple length {len(batch)}."
            )
        batch = batch[0]

    if not isinstance(batch, dict):
        raise TypeError(f"Expected batch dict, got {type(batch)}")

    return batch


def get_field(sample: Dict[str, Any], key: str) -> Any:
    if key not in sample:
        raise KeyError(f"Sample has no key '{key}'. Available keys: {list(sample.keys())}")

    value = sample[key]

    if isinstance(value, (list, tuple)) and len(value) == 1:
        value = value[0]

    return value


def scalarize(x: Any) -> Any:
    if isinstance(x, np.ndarray):
        if x.size == 1:
            return scalarize(x.reshape(-1)[0])
        return x.tolist()

    if hasattr(x, "item"):
        try:
            return x.item()
        except Exception:
            pass

    if isinstance(x, (list, tuple)) and len(x) == 1:
        return scalarize(x[0])

    if isinstance(x, (np.integer,)):
        return int(x)

    if isinstance(x, (np.floating,)):
        return float(x)

    if isinstance(x, (np.bool_,)):
        return bool(x)

    return x


def normalize_metainfo(metainfo: Any) -> Dict[str, Any]:
    if metainfo is None:
        return {}

    if isinstance(metainfo, (list, tuple)) and len(metainfo) == 1:
        metainfo = metainfo[0]

    if not isinstance(metainfo, dict):
        return {"metainfo": str(metainfo)}

    return {str(k): scalarize(v) for k, v in metainfo.items()}


def build_dataloader(config_path: str, dataloader_key: str):
    from mmengine.config import Config
    from mmengine.registry import init_default_scope
    from mmengine.runner import Runner
    from mmengine.utils import import_modules_from_strings

    cfg = Config.fromfile(config_path)

    custom_imports = cfg.get("custom_imports", None)
    if custom_imports is not None:
        import_modules_from_strings(**custom_imports)

    default_scope = cfg.get("default_scope", "mmseg")
    init_default_scope(default_scope)

    if dataloader_key not in cfg:
        raise KeyError(
            f"Dataloader key '{dataloader_key}' not found in config. "
            f"Available keys include: {list(cfg.keys())}"
        )

    return cfg, Runner.build_dataloader(cfg[dataloader_key])


def get_needed_indices(center_idx: int, radius: int, stride: int) -> List[int]:
    start = max(0, center_idx - radius)
    end = center_idx + radius

    needed = list(range(start, end + 1, stride))

    if center_idx not in needed:
        needed.append(center_idx)
        needed = sorted(set(needed))

    return needed


def load_window_samples(
    dataloader,
    needed_indices: List[int],
    spatial: Dict[str, Any],
) -> Dict[int, Dict[str, Any]]:
    needed_set = set(needed_indices)
    max_needed = max(needed_indices)

    loaded: Dict[int, Dict[str, Any]] = {}

    for eda_idx, batch in enumerate(dataloader):
        if eda_idx > max_needed:
            break

        if eda_idx not in needed_set:
            continue

        sample = unwrap_batch(batch)

        img = normalize_img_to_hwc_float01(get_field(sample, "img"))
        gt = normalize_gt_to_hwc(get_field(sample, "gt_seg_map"))

        metainfo = {}
        if "metainfo" in sample:
            metainfo = normalize_metainfo(sample["metainfo"])

        points = points_from_gt(
            gt=gt,
            sample=sample,
            image_width=img.shape[1],
            spatial=spatial,
        )
        points["x_pix"] = points.pop("x")
        points["y_pix"] = points.pop("y")

        loaded[eda_idx] = {
            "eda_idx": eda_idx,
            "img": img,
            "gt": gt,
            "points": points,
            "metainfo": metainfo,
        }

    missing = sorted(set(needed_indices) - set(loaded.keys()))
    if missing:
        raise RuntimeError(
            f"Could not load all requested demo indices. Missing: {missing}"
        )

    return loaded


def collect_accumulated_points(
    samples: Dict[int, Dict[str, Any]],
    center_idx: int,
    radius: int,
) -> pd.DataFrame:
    rows = []

    for eda_idx, sample in sorted(samples.items()):
        rel_idx = eda_idx - center_idx
        points = sample["points"]

        n = len(points["x_pix"])

        for i in range(n):
            rows.append(
                {
                    "eda_idx": int(eda_idx),
                    "rel_idx": int(rel_idx),
                    "rel_abs_idx": int(abs(rel_idx)),
                    "time_weight": float(
                        1.0 - min(abs(rel_idx), radius) / max(radius, 1)
                    ),
                    "x_pix": float(points["x_pix"][i]),
                    "y_pix": float(points["y_pix"][i]),
                    "depth": float(points["depth"][i]),
                    "velocity": float(points["velocity"][i]),
                    "rcs": float(points["rcs"][i]),
                    "lateral": float(points["lateral"][i]),
                    "forward": float(points["forward"][i]),
                    "range": float(points["range"][i]),
                    "azimuth_deg": float(points["azimuth_deg"][i]),
                }
            )

    return pd.DataFrame(rows)


def draw_base_image(
    img: np.ndarray,
    title: str,
    spatial: Dict[str, Any],
) -> None:
    h, w = img.shape[:2]
    xlim, ylim = camera_plot_limits(w, h, spatial)
    plt.imshow(img)
    plt.xlim(*xlim)
    plt.ylim(ylim[1], ylim[0])
    plt.title(title)
    plt.axis("off")


def save_center_points(
    center_img: np.ndarray,
    center_points: Dict[str, np.ndarray],
    out_path: Path,
    point_size: float,
    alpha: float,
    spatial: Dict[str, Any],
) -> None:
    plt.figure(figsize=(16, 6))
    draw_base_image(center_img, "Center frame with GT radar points", spatial)

    if len(center_points["x_pix"]) > 0:
        plt.scatter(
            center_points["x_pix"],
            center_points["y_pix"],
            s=point_size,
            c="white",
            alpha=alpha,
            edgecolors="black",
            linewidths=0.20,
        )

    plt.tight_layout()
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close()


def save_time_colored(
    center_img: np.ndarray,
    df: pd.DataFrame,
    out_path: Path,
    center_idx: int,
    radius: int,
    point_size: float,
    alpha: float,
    spatial: Dict[str, Any],
) -> None:
    plt.figure(figsize=(16, 6))
    draw_base_image(
        center_img,
        f"Temporal accumulation, colored by relative frame index, center={center_idx}",
        spatial,
    )

    if len(df) > 0:
        sc = plt.scatter(
            df["x_pix"],
            df["y_pix"],
            s=point_size,
            c=df["rel_idx"],
            cmap="coolwarm",
            vmin=-radius,
            vmax=radius,
            alpha=alpha,
            edgecolors="black",
            linewidths=0.10,
        )
        cbar = plt.colorbar(sc, fraction=0.025, pad=0.02)
        cbar.set_label("relative frame index")

    plt.tight_layout()
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close()


def save_alpha_fade(
    center_img: np.ndarray,
    df: pd.DataFrame,
    out_path: Path,
    center_idx: int,
    radius: int,
    point_size: float,
    alpha: float,
    alpha_min: float,
    spatial: Dict[str, Any],
) -> None:
    plt.figure(figsize=(16, 6))
    draw_base_image(
        center_img,
        f"Temporal accumulation, alpha fades with distance from center, center={center_idx}",
        spatial,
    )

    if len(df) > 0:
        # Plot far frames first, center frame last.
        df_plot = df.sort_values("rel_abs_idx", ascending=False)

        for _, row in df_plot.iterrows():
            weight = float(row["time_weight"])
            point_alpha = alpha_min + (alpha - alpha_min) * weight

            plt.scatter(
                row["x_pix"],
                row["y_pix"],
                s=point_size,
                c="white",
                alpha=point_alpha,
                edgecolors="black",
                linewidths=0.10,
            )

    plt.tight_layout()
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close()


def save_depth(
    center_img: np.ndarray,
    df: pd.DataFrame,
    out_path: Path,
    point_size: float,
    alpha: float,
    vmin: float,
    vmax: float,
    spatial: Dict[str, Any],
) -> None:
    plt.figure(figsize=(16, 6))
    draw_base_image(center_img, "Temporal accumulation, colored by GT depth", spatial)

    if len(df) > 0:
        sc = plt.scatter(
            df["x_pix"],
            df["y_pix"],
            s=point_size,
            c=df["depth"],
            cmap="turbo",
            vmin=vmin,
            vmax=vmax,
            alpha=alpha,
            edgecolors="black",
            linewidths=0.10,
        )
        cbar = plt.colorbar(sc, fraction=0.025, pad=0.02)
        cbar.set_label("GT depth [m]")

    plt.tight_layout()
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close()


def save_velocity(
    center_img: np.ndarray,
    df: pd.DataFrame,
    out_path: Path,
    point_size: float,
    alpha: float,
    vmin: float,
    vmax: float,
    spatial: Dict[str, Any],
) -> None:
    plt.figure(figsize=(16, 6))
    draw_base_image(center_img, "Temporal accumulation, colored by GT radial velocity", spatial)

    if len(df) > 0:
        sc = plt.scatter(
            df["x_pix"],
            df["y_pix"],
            s=point_size,
            c=df["velocity"],
            cmap="coolwarm",
            vmin=vmin,
            vmax=vmax,
            alpha=alpha,
            edgecolors="black",
            linewidths=0.10,
        )
        cbar = plt.colorbar(sc, fraction=0.025, pad=0.02)
        cbar.set_label("GT radial velocity [m s$^{-1}$]")

    plt.tight_layout()
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close()


def save_rcs(
    center_img: np.ndarray,
    df: pd.DataFrame,
    out_path: Path,
    point_size: float,
    alpha: float,
    vmin: float | None,
    vmax: float | None,
    spatial: Dict[str, Any],
) -> None:
    plt.figure(figsize=(16, 6))
    draw_base_image(center_img, "Temporal accumulation, colored by GT RCS", spatial)

    if len(df) > 0:
        sc = plt.scatter(
            df["x_pix"],
            df["y_pix"],
            s=point_size,
            c=df["rcs"],
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
            alpha=alpha,
            edgecolors="black",
            linewidths=0.10,
        )
        cbar = plt.colorbar(sc, fraction=0.025, pad=0.02)
        cbar.set_label("GT RCS [dBsm]")

    plt.tight_layout()
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close()


def save_cartesian_bev_accumulation(
    df: pd.DataFrame,
    out_path: Path,
    point_size: float,
    alpha: float,
    depth_vmin: float | None,
    depth_vmax: float | None,
    spatial: Dict[str, Any],
) -> None:
    lateral = df["lateral"].to_numpy(dtype=float) if len(df) else np.empty((0,))
    forward = df["forward"].to_numpy(dtype=float) if len(df) else np.empty((0,))
    mask = cartesian_domain_mask(lateral, forward, spatial)
    xlim, ylim = cartesian_plot_limits(lateral, forward, spatial)

    fig, ax = plt.subplots(figsize=(9, 9))
    ax.set_facecolor("black")
    ax.set_xlim(xlim[1], xlim[0])
    ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.18)
    ax.set_xlabel("lateral position [m]")
    ax.set_ylabel("forward position [m]")
    ax.set_title("Temporal accumulation in Cartesian BEV, no ego-motion compensation")
    ax.scatter([0.0], [0.0], marker="^", s=90, c="white", edgecolors="black")

    if np.any(mask):
        rows = df.loc[mask]
        sc = ax.scatter(
            rows["lateral"], rows["forward"], s=point_size,
            c=rows["range"], cmap="turbo", vmin=depth_vmin, vmax=depth_vmax,
            alpha=alpha, edgecolors="white", linewidths=0.20,
        )
        cbar = fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.02)
        cbar.set_label("GT range [m]")
    else:
        ax.text(0.5, 0.5, "No points inside configured Cartesian domain",
                transform=ax.transAxes, ha="center", va="center", color="white")

    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def save_polar_accumulation(
    df: pd.DataFrame,
    out_path: Path,
    point_size: float,
    alpha: float,
    depth_vmin: float | None,
    depth_vmax: float | None,
    spatial: Dict[str, Any],
) -> None:
    range_m = df["range"].to_numpy(dtype=float) if len(df) else np.empty((0,))
    azimuth = df["azimuth_deg"].to_numpy(dtype=float) if len(df) else np.empty((0,))
    mask = polar_domain_mask(range_m, azimuth, spatial)
    xlim, ylim = polar_plot_limits(azimuth, range_m, spatial)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_xlim(xlim[1], xlim[0])
    ax.set_ylim(*ylim)
    ax.grid(True, alpha=0.30)
    ax.set_xlabel("azimuth [deg]")
    ax.set_ylabel("range [m]")
    ax.set_title("Temporal accumulation in polar domain, no ego-motion compensation")

    if np.any(mask):
        rows = df.loc[mask]
        sc = ax.scatter(
            rows["azimuth_deg"], rows["range"], s=point_size,
            c=rows["range"], cmap="turbo", vmin=depth_vmin, vmax=depth_vmax,
            alpha=alpha, edgecolors="black", linewidths=0.20,
        )
        cbar = fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.02)
        cbar.set_label("GT range [m]")
    else:
        ax.text(0.5, 0.5, "No points inside configured polar domain",
                transform=ax.transAxes, ha="center", va="center")

    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def make_summary(
    args: argparse.Namespace,
    needed_indices: List[int],
    samples: Dict[int, Dict[str, Any]],
    accumulated_df: pd.DataFrame,
    spatial: Dict[str, Any],
) -> Dict[str, Any]:
    per_frame = []

    for eda_idx in needed_indices:
        sample = samples[eda_idx]
        points = sample["points"]
        metainfo = sample["metainfo"]

        item = {
            "eda_idx": int(eda_idx),
            "rel_idx": int(eda_idx - args.center_eda_idx),
            "point_count": int(len(points["x_pix"])),
            "metainfo": metainfo,
        }
        per_frame.append(item)

    if len(accumulated_df) > 0:
        depth_stats = {
            "min": float(accumulated_df["depth"].min()),
            "mean": float(accumulated_df["depth"].mean()),
            "max": float(accumulated_df["depth"].max()),
        }
        velocity_stats = {
            "min": float(accumulated_df["velocity"].min()),
            "mean": float(accumulated_df["velocity"].mean()),
            "max": float(accumulated_df["velocity"].max()),
        }
        rcs_stats = {
            "min": float(accumulated_df["rcs"].min()),
            "mean": float(accumulated_df["rcs"].mean()),
            "max": float(accumulated_df["rcs"].max()),
        }
    else:
        depth_stats = {"min": None, "mean": None, "max": None}
        velocity_stats = {"min": None, "mean": None, "max": None}
        rcs_stats = {"min": None, "mean": None, "max": None}

    return {
        "config": str(args.config),
        "dataloader": str(args.dataloader),
        "center_eda_idx": int(args.center_eda_idx),
        "radius": int(args.radius),
        "stride": int(args.stride),
        "needed_indices": [int(x) for x in needed_indices],
        "num_loaded_frames": int(len(samples)),
        "num_accumulated_points": int(len(accumulated_df)),
        "point_size": float(args.point_size),
        "ppp_eda_spatial": spatial,
        "camera_domain_points": int(np.sum(camera_domain_mask(
            accumulated_df["x_pix"].to_numpy(dtype=float),
            accumulated_df["y_pix"].to_numpy(dtype=float),
            spatial,
        ))) if len(accumulated_df) else 0,
        "cartesian_bev_domain_points": int(np.sum(cartesian_domain_mask(
            accumulated_df["lateral"].to_numpy(dtype=float),
            accumulated_df["forward"].to_numpy(dtype=float),
            spatial,
        ))) if len(accumulated_df) else 0,
        "polar_domain_points": int(np.sum(polar_domain_mask(
            accumulated_df["range"].to_numpy(dtype=float),
            accumulated_df["azimuth_deg"].to_numpy(dtype=float),
            spatial,
        ))) if len(accumulated_df) else 0,
        "image_space_accumulation": True,
        "ego_motion_compensated": False,
        "note": (
            "Projected radar GT points from neighboring frames are drawn onto the "
            "center image without ego-motion compensation."
        ),
        "depth_stats": depth_stats,
        "velocity_stats": velocity_stats,
        "rcs_stats": rcs_stats,
        "per_frame": per_frame,
    }


def main() -> None:
    args = parse_args()

    if args.radius < 0:
        raise ValueError("--radius must be >= 0")

    if args.stride <= 0:
        raise ValueError("--stride must be >= 1")

    out_dir = Path(args.out_dir)
    prepare_out_dir(out_dir, overwrite=args.overwrite)

    needed_indices = get_needed_indices(
        center_idx=args.center_eda_idx,
        radius=args.radius,
        stride=args.stride,
    )

    print(f"Config:       {args.config}")
    print(f"Dataloader:   {args.dataloader}")
    print(f"Center idx:   {args.center_eda_idx}")
    print(f"Radius:       {args.radius}")
    print(f"Stride:       {args.stride}")
    print(f"Needed idx:   {needed_indices}")
    print(f"Output dir:   {out_dir}")

    cfg, dataloader = build_dataloader(args.config, args.dataloader)
    spatial = resolve_spatial_config_from_cfg(cfg)
    samples = load_window_samples(dataloader, needed_indices, spatial)

    center_sample = samples[args.center_eda_idx]
    center_img = center_sample["img"]
    center_points = center_sample["points"]

    accumulated_df = collect_accumulated_points(
        samples=samples,
        center_idx=args.center_eda_idx,
        radius=args.radius,
    )

    center_camera_mask = camera_domain_mask(
        center_points["x_pix"], center_points["y_pix"], spatial
    )
    center_points_camera = {
        key: value[center_camera_mask] if isinstance(value, np.ndarray) and len(value) == len(center_camera_mask) else value
        for key, value in center_points.items()
    }
    if len(accumulated_df):
        accumulated_camera_mask = camera_domain_mask(
            accumulated_df["x_pix"].to_numpy(dtype=float),
            accumulated_df["y_pix"].to_numpy(dtype=float),
            spatial,
        )
        accumulated_camera_df = accumulated_df.loc[accumulated_camera_mask].copy()
    else:
        accumulated_camera_df = accumulated_df.copy()

    save_center_points(
        center_img=center_img,
        center_points=center_points_camera,
        out_path=out_dir / "center_frame_points.png",
        point_size=args.point_size,
        alpha=args.alpha,
        spatial=spatial,
    )

    save_time_colored(
        center_img=center_img,
        df=accumulated_camera_df,
        out_path=out_dir / "accumulated_points_time_colored.png",
        center_idx=args.center_eda_idx,
        radius=args.radius,
        point_size=args.point_size,
        alpha=args.alpha,
        spatial=spatial,
    )

    save_alpha_fade(
        center_img=center_img,
        df=accumulated_camera_df,
        out_path=out_dir / "accumulated_points_alpha_fade.png",
        center_idx=args.center_eda_idx,
        radius=args.radius,
        point_size=args.point_size,
        alpha=args.alpha,
        alpha_min=args.alpha_min,
        spatial=spatial,
    )

    save_depth(
        center_img=center_img,
        df=accumulated_camera_df,
        out_path=out_dir / "accumulated_depth.png",
        point_size=args.point_size,
        alpha=args.alpha,
        vmin=args.depth_vmin,
        vmax=args.depth_vmax,
        spatial=spatial,
    )

    save_velocity(
        center_img=center_img,
        df=accumulated_camera_df,
        out_path=out_dir / "accumulated_velocity.png",
        point_size=args.point_size,
        alpha=args.alpha,
        vmin=args.velocity_vmin,
        vmax=args.velocity_vmax,
        spatial=spatial,
    )

    save_rcs(
        center_img=center_img,
        df=accumulated_camera_df,
        out_path=out_dir / "accumulated_rcs.png",
        point_size=args.point_size,
        alpha=args.alpha,
        vmin=args.rcs_vmin,
        vmax=args.rcs_vmax,
        spatial=spatial,
    )

    save_cartesian_bev_accumulation(
        df=accumulated_df,
        out_path=out_dir / "accumulated_cartesian_bev.png",
        point_size=args.point_size,
        alpha=args.alpha,
        depth_vmin=args.depth_vmin,
        depth_vmax=args.depth_vmax,
        spatial=spatial,
    )

    save_polar_accumulation(
        df=accumulated_df,
        out_path=out_dir / "accumulated_polar.png",
        point_size=args.point_size,
        alpha=args.alpha,
        depth_vmin=args.depth_vmin,
        depth_vmax=args.depth_vmax,
        spatial=spatial,
    )

    if args.save_csv:
        accumulated_df.to_csv(out_dir / "accumulated_points.csv", index=False)

    summary = make_summary(
        args=args,
        needed_indices=needed_indices,
        samples=samples,
        accumulated_df=accumulated_df,
        spatial=spatial,
    )

    with open(out_dir / "temporal_accumulation_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Done.")
    print(f"Output dir: {out_dir}")
    print(f"Accumulated points: {len(accumulated_df)}")


if __name__ == "__main__":
    main()