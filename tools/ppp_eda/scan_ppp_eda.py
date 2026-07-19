#!/usr/bin/env python3
"""
Scan a PPP EDA dataloader and save:

1. frames.csv
   One row per sampled frame:
       metadata
       projected radar GT count
       whole-image brightness / contrast / blur statistics

2. point_values.npz
   One row per projected radar GT point:
       eda_idx
       x_pix
       y_pix
       depth
       velocity
       rcs
       lateral
       forward
       range
       azimuth_deg
       rgb

Example:

python tools/ppp_eda/scan_ppp_eda.py \
    configs/ppp_eda/ppp_eda.py \
    --dataloader eda_val_dataloader \
    --out-dir work_dirs/ppp_eda/val_seed43_n5000
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List

import cv2
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from mmengine.config import Config
from mmengine.registry import init_default_scope
from mmengine.runner import Runner
from mmengine.utils import import_modules_from_strings

from ppp_eda_geometry import (
    camera_domain_mask,
    camera_points_to_spatial,
    cartesian_domain_mask,
    get_sample_h_fov_deg,
    get_sample_rad2cam,
    polar_domain_mask,
    resolve_spatial_config_from_cfg,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scan PPP EDA dataloader.")
    parser.add_argument(
        "config",
        type=str,
        help="Path to EDA config, e.g. configs/ppp_eda/ppp_eda.py",
    )
    parser.add_argument(
        "--dataloader",
        type=str,
        default="eda_val_dataloader",
        help=(
            "Name of dataloader variable in config, e.g. "
            "eda_train_dataloader, eda_val_dataloader, eda_demo_dataloader."
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        required=True,
        help="Output directory for frames.csv, point_values.npz, scan_summary.json.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional maximum number of samples to scan.",
    )
    parser.add_argument(
        "--print-interval",
        type=int,
        default=100,
        help="Progress print interval in samples.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting an existing output directory.",
    )
    return parser.parse_args()


def to_numpy(x: Any) -> np.ndarray:
    """Convert torch tensor / numpy / array-like to numpy."""
    if isinstance(x, np.ndarray):
        return x
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def squeeze_optional_batch_dim(arr: np.ndarray, name: str) -> np.ndarray:
    """
    Accept both unbatched HWC/CHW arrays and batch_size=1 arrays.

    Examples:
        H, W, 3        -> unchanged
        1, H, W, 3     -> H, W, 3
        1, 3, H, W     -> 3, H, W
    """
    if arr.ndim == 4:
        if arr.shape[0] != 1:
            raise ValueError(
                f"{name} has a batch dimension with batch_size={arr.shape[0]}. "
                f"For this EDA script, use PPP_EDA_BATCH_SIZE=1."
            )
        arr = arr[0]

    return arr


def scalarize(x: Any) -> Any:
    """Convert tensors / numpy scalars / one-element arrays to plain Python values."""
    if x is None:
        return None

    if torch.is_tensor(x):
        x = x.detach().cpu().numpy()

    if isinstance(x, np.ndarray):
        if x.size == 1:
            return x.reshape(-1)[0].item()
        return x.tolist()

    if isinstance(x, (np.generic,)):
        return x.item()

    if isinstance(x, list) and len(x) == 1:
        return scalarize(x[0])

    if isinstance(x, tuple) and len(x) == 1:
        return scalarize(x[0])

    return x


def normalize_img_to_hwc_float01(img: Any) -> np.ndarray:
    """
    Convert image to HWC float32 in [0, 1].

    Accepts:
        H, W, 3
        1, H, W, 3
        3, H, W
        1, 3, H, W
    """
    img = to_numpy(img)
    img = squeeze_optional_batch_dim(img, name="img")

    if img.ndim != 3:
        raise ValueError(f"Expected image with ndim=3 after squeezing, got shape {img.shape}")

    # Defensive CHW -> HWC conversion.
    if img.shape[0] == 3 and img.shape[-1] != 3:
        img = np.transpose(img, (1, 2, 0))

    if img.shape[-1] != 3:
        raise ValueError(f"Expected RGB image with last dim 3, got shape {img.shape}")

    img = img.astype(np.float32, copy=False)

    # Current SYS100/MMseg pipeline stores image in [-1, 1].
    img01 = (img + 1.0) / 2.0
    img01 = np.clip(img01, 0.0, 1.0)

    return img01


def normalize_gt_to_hwc(gt: Any) -> np.ndarray:
    """
    Convert GT map to HWC numpy.

    Accepts:
        H, W, 4
        1, H, W, 4
        4, H, W
        1, 4, H, W
    """
    gt = to_numpy(gt)
    gt = squeeze_optional_batch_dim(gt, name="gt_seg_map")

    if gt.ndim != 3:
        raise ValueError(f"Expected gt_seg_map with ndim=3 after squeezing, got shape {gt.shape}")

    # Defensive CHW -> HWC conversion.
    if gt.shape[0] == 4 and gt.shape[-1] != 4:
        gt = np.transpose(gt, (1, 2, 0))

    if gt.shape[-1] != 4:
        raise ValueError(f"Expected gt_seg_map last dim 4, got shape {gt.shape}")

    return gt.astype(np.float32, copy=False)


def unpack_batch(batch: Any) -> List[Dict[str, Any]]:
    """
    Handle pseudo_collate output.

    With pseudo_collate and batch_size=1, batch is usually:
        [sample_dict]

    With larger batch_size:
        [sample_dict_0, sample_dict_1, ...]
    """
    if isinstance(batch, list):
        return batch

    if isinstance(batch, tuple):
        return list(batch)

    if isinstance(batch, dict):
        return [batch]

    raise TypeError(f"Unsupported batch type: {type(batch)}")


def get_meta_value(sample: Dict[str, Any], key: str, default: Any = None) -> Any:
    meta = sample.get("metainfo", {})
    if isinstance(meta, dict) and key in meta:
        return scalarize(meta[key])
    return scalarize(sample.get(key, default))


def compute_image_stats(img01: np.ndarray) -> Dict[str, float]:
    """
    Whole-image camera statistics.

    img01:
        H, W, 3 float32 in [0, 1]
    """
    gray = img01.mean(axis=2)

    q01, q05, q50, q95, q99 = np.quantile(
        gray,
        [0.01, 0.05, 0.50, 0.95, 0.99],
    )

    gray_u8 = np.clip(gray * 255.0, 0.0, 255.0).astype(np.uint8)
    laplacian_var = float(cv2.Laplacian(gray_u8, cv2.CV_64F).var())

    stats = dict(
        brightness_mean=float(gray.mean()),
        brightness_std=float(gray.std()),
        brightness_q01=float(q01),
        brightness_q05=float(q05),
        brightness_q50=float(q50),
        brightness_q95=float(q95),
        brightness_q99=float(q99),
        dark_fraction_0_05=float((gray < 0.05).mean()),
        dark_fraction_0_10=float((gray < 0.10).mean()),
        dark_fraction_0_15=float((gray < 0.15).mean()),
        dark_fraction_0_20=float((gray < 0.20).mean()),
        bright_fraction_0_80=float((gray > 0.80).mean()),
        bright_fraction_0_85=float((gray > 0.85).mean()),
        bright_fraction_0_90=float((gray > 0.90).mean()),
        bright_fraction_0_95=float((gray > 0.95).mean()),
        contrast_q95_minus_q05=float(q95 - q05),
        contrast_q99_minus_q01=float(q99 - q01),
        laplacian_var=laplacian_var,
    )

    return stats


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


def load_config(config_path: str) -> Config:
    cfg = Config.fromfile(config_path)

    if hasattr(cfg, "custom_imports"):
        import_modules_from_strings(**cfg.custom_imports)

    init_default_scope("mmseg")
    return cfg


def prepare_out_dir(out_dir: Path, overwrite: bool) -> None:
    if out_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output directory already exists: {out_dir}\n"
                f"Use --overwrite if you want to replace it."
            )
        shutil.rmtree(out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)


def save_point_values(
    out_path: Path,
    point_eda_idx: List[np.ndarray],
    point_x: List[np.ndarray],
    point_y: List[np.ndarray],
    point_depth: List[np.ndarray],
    point_velocity: List[np.ndarray],
    point_rcs: List[np.ndarray],
    point_lateral: List[np.ndarray],
    point_forward: List[np.ndarray],
    point_range: List[np.ndarray],
    point_azimuth_deg: List[np.ndarray],
    point_rgb: List[np.ndarray],
) -> int:
    if len(point_x) == 0:
        np.savez_compressed(
            out_path,
            eda_idx=np.empty((0,), dtype=np.int32),
            x_pix=np.empty((0,), dtype=np.int16),
            y_pix=np.empty((0,), dtype=np.int16),
            depth=np.empty((0,), dtype=np.float32),
            velocity=np.empty((0,), dtype=np.float32),
            rcs=np.empty((0,), dtype=np.float32),
            lateral=np.empty((0,), dtype=np.float32),
            forward=np.empty((0,), dtype=np.float32),
            range=np.empty((0,), dtype=np.float32),
            azimuth_deg=np.empty((0,), dtype=np.float32),
            rgb=np.empty((0, 3), dtype=np.uint8),
        )
        return 0

    arrays = dict(
        eda_idx=np.concatenate(point_eda_idx).astype(np.int32, copy=False),
        x_pix=np.concatenate(point_x).astype(np.int16, copy=False),
        y_pix=np.concatenate(point_y).astype(np.int16, copy=False),
        depth=np.concatenate(point_depth).astype(np.float32, copy=False),
        velocity=np.concatenate(point_velocity).astype(np.float32, copy=False),
        rcs=np.concatenate(point_rcs).astype(np.float32, copy=False),
        lateral=np.concatenate(point_lateral).astype(np.float32, copy=False),
        forward=np.concatenate(point_forward).astype(np.float32, copy=False),
        range=np.concatenate(point_range).astype(np.float32, copy=False),
        azimuth_deg=np.concatenate(point_azimuth_deg).astype(np.float32, copy=False),
        rgb=np.concatenate(point_rgb).astype(np.uint8, copy=False),
    )
    np.savez_compressed(out_path, **arrays)
    return int(len(arrays["x_pix"]))


def scan(
    loader,
    max_samples: int | None,
    print_interval: int,
    spatial: Dict[str, Any],
) -> tuple[pd.DataFrame, Dict[str, List[np.ndarray]], Dict[str, Any]]:
    frame_rows: List[Dict[str, Any]] = []

    point_eda_idx: List[np.ndarray] = []
    point_x: List[np.ndarray] = []
    point_y: List[np.ndarray] = []
    point_depth: List[np.ndarray] = []
    point_velocity: List[np.ndarray] = []
    point_rcs: List[np.ndarray] = []
    point_lateral: List[np.ndarray] = []
    point_forward: List[np.ndarray] = []
    point_range: List[np.ndarray] = []
    point_azimuth_deg: List[np.ndarray] = []
    point_rgb: List[np.ndarray] = []

    started_at = time.time()
    num_frames = 0
    num_points = 0

    try:
        total = len(loader.dataset)
    except Exception:
        total = None

    if max_samples is not None and total is not None:
        total = min(total, max_samples)

    pbar = tqdm(loader, total=total, desc="Scanning PPP EDA samples")

    stop = False

    for batch in pbar:
        samples = unpack_batch(batch)

        for sample in samples:
            if max_samples is not None and num_frames >= max_samples:
                stop = True
                break

            eda_idx = num_frames

            img01 = normalize_img_to_hwc_float01(sample["img"])
            gt = normalize_gt_to_hwc(sample["gt_seg_map"])

            if img01.shape[:2] != gt.shape[:2]:
                raise ValueError(
                    f"Image and GT spatial shape mismatch: "
                    f"img={img01.shape}, gt={gt.shape}"
                )

            mask = gt[..., 0] > 0
            y_pix, x_pix = np.nonzero(mask)
            point_count = int(len(x_pix))

            depth = gt[..., 1][mask]
            velocity = gt[..., 2][mask]
            rcs = gt[..., 3][mask]

            if point_count > 0:
                geometry = camera_points_to_spatial(
                    x_pix=x_pix,
                    depth=depth,
                    rad2cam=get_sample_rad2cam(sample),
                    image_width=img01.shape[1],
                    h_fov_deg=get_sample_h_fov_deg(sample, spatial),
                    spatial=spatial,
                )
                lateral = geometry["lateral"]
                forward = geometry["forward"]
                range_m = geometry["range"]
                azimuth_deg = geometry["azimuth_deg"]

                rgb = np.rint(img01[y_pix, x_pix, :] * 255.0)
                rgb = np.clip(rgb, 0, 255).astype(np.uint8)

                point_eda_idx.append(np.full(point_count, eda_idx, dtype=np.int32))
                point_x.append(x_pix.astype(np.int16, copy=False))
                point_y.append(y_pix.astype(np.int16, copy=False))
                point_depth.append(depth.astype(np.float32, copy=False))
                point_velocity.append(velocity.astype(np.float32, copy=False))
                point_rcs.append(rcs.astype(np.float32, copy=False))
                point_lateral.append(lateral.astype(np.float32, copy=False))
                point_forward.append(forward.astype(np.float32, copy=False))
                point_range.append(range_m.astype(np.float32, copy=False))
                point_azimuth_deg.append(azimuth_deg.astype(np.float32, copy=False))
                point_rgb.append(rgb)

                geometry_valid_count = int(np.sum(geometry["geometry_valid"]))
                camera_domain_count = int(np.sum(camera_domain_mask(x_pix, y_pix, spatial)))
                cartesian_domain_count = int(
                    np.sum(cartesian_domain_mask(lateral, forward, spatial))
                )
                polar_domain_count = int(
                    np.sum(polar_domain_mask(range_m, azimuth_deg, spatial))
                )
            else:
                geometry_valid_count = 0
                camera_domain_count = 0
                cartesian_domain_count = 0
                polar_domain_count = 0

            img_stats = compute_image_stats(img01)

            row = dict(
                eda_idx=int(eda_idx),

                # The actual SYS100/MMseg index used by the transform.
                idx=scalarize(sample.get("idx", None)),

                # For demo datasets this may exist.
                demo_idx=scalarize(sample.get("demo_idx", None)),

                global_idx=get_meta_value(sample, "global_idx", sample.get("idx", None)),
                sample_id=get_meta_value(sample, "sample_id", None),
                video_name=get_meta_value(sample, "video_name", None),
                video_path=get_meta_value(sample, "video_path", None),
                video_frame_idx=get_meta_value(sample, "video_frame_idx", None),

                point_count=point_count,
                geometry_valid_count=geometry_valid_count,
                camera_domain_point_count=camera_domain_count,
                cartesian_bev_domain_point_count=cartesian_domain_count,
                polar_domain_point_count=polar_domain_count,
            )

            row.update(img_stats)
            frame_rows.append(row)

            num_frames += 1
            num_points += point_count

            if print_interval > 0 and num_frames % print_interval == 0:
                elapsed = time.time() - started_at
                pbar.set_postfix(
                    frames=num_frames,
                    points=num_points,
                    fps=f"{num_frames / max(elapsed, 1e-9):.2f}",
                )

        if stop:
            break

    elapsed = time.time() - started_at

    frames_df = pd.DataFrame(frame_rows)

    point_data = dict(
        eda_idx=point_eda_idx,
        x_pix=point_x,
        y_pix=point_y,
        depth=point_depth,
        velocity=point_velocity,
        rcs=point_rcs,
        lateral=point_lateral,
        forward=point_forward,
        range=point_range,
        azimuth_deg=point_azimuth_deg,
        rgb=point_rgb,
    )

    summary = dict(
        num_frames=int(num_frames),
        num_points=int(num_points),
        elapsed_seconds=float(elapsed),
        frames_per_second=float(num_frames / max(elapsed, 1e-9)),
        mean_points_per_frame=float(num_points / max(num_frames, 1)),
    )

    return frames_df, point_data, summary


def main() -> None:
    args = parse_args()

    config_path = Path(args.config)
    out_dir = Path(args.out_dir)

    prepare_out_dir(out_dir, overwrite=args.overwrite)

    cfg = load_config(str(config_path))
    loader = build_dataloader_from_cfg(cfg, args.dataloader)
    spatial = resolve_spatial_config_from_cfg(cfg)

    # Save resolved config for reproducibility.
    cfg.dump(str(out_dir / "used_config.py"))

    frames_df, point_data, summary = scan(
        loader=loader,
        max_samples=args.max_samples,
        print_interval=args.print_interval,
        spatial=spatial,
    )

    frames_csv_path = out_dir / "frames.csv"
    points_npz_path = out_dir / "point_values.npz"
    summary_json_path = out_dir / "scan_summary.json"

    frames_df.to_csv(frames_csv_path, index=False)

    num_points_saved = save_point_values(
        out_path=points_npz_path,
        point_eda_idx=point_data["eda_idx"],
        point_x=point_data["x_pix"],
        point_y=point_data["y_pix"],
        point_depth=point_data["depth"],
        point_velocity=point_data["velocity"],
        point_rcs=point_data["rcs"],
        point_lateral=point_data["lateral"],
        point_forward=point_data["forward"],
        point_range=point_data["range"],
        point_azimuth_deg=point_data["azimuth_deg"],
        point_rgb=point_data["rgb"],
    )

    summary.update(
        dict(
            config=str(config_path),
            dataloader=args.dataloader,
            out_dir=str(out_dir),
            frames_csv=str(frames_csv_path),
            point_values_npz=str(points_npz_path),
            num_points_saved=int(num_points_saved),
            ppp_eda_spatial=spatial,
        )
    )

    if hasattr(cfg, "ppp_eda_meta"):
        try:
            summary["ppp_eda_meta"] = dict(cfg.ppp_eda_meta)
        except Exception:
            summary["ppp_eda_meta"] = str(cfg.ppp_eda_meta)

    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nDone.")
    print(f"Saved frames:       {frames_csv_path}")
    print(f"Saved point values: {points_npz_path}")
    print(f"Saved summary:      {summary_json_path}")
    print(f"Frames scanned:     {summary['num_frames']}")
    print(f"Points saved:       {summary['num_points_saved']}")
    print(f"Elapsed seconds:    {summary['elapsed_seconds']:.2f}")


if __name__ == "__main__":
    main()