#!/usr/bin/env python3
"""
Render the fixed PPP EDA demo sequence.

This script uses the EDA demo dataloader from configs/ppp_eda/ppp_eda.py.
It does not use selected_examples.json and does not create categories.

Output:
    out_dir/
        points/
            frame_000000_eda000000_points.png
            ...
        depth/
            frame_000000_eda000000_depth.png
            ...
        velocity/
            frame_000000_eda000000_velocity.png
            ...
        render_summary.json

Example:

python tools/ppp_eda/render_ppp_demo_sequence.py \
    configs/ppp_eda/ppp_eda.py \
    --dataloader eda_demo_dataloader \
    --out-dir work_dirs/ppp_eda/demo_sequence \
    --point-size 7 \
    --overwrite
"""

from __future__ import annotations

import argparse
import copy
import json
import re
import shutil
from pathlib import Path
from typing import Any, Dict, List

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from mmengine.config import Config
from mmengine.registry import init_default_scope
from mmengine.runner import Runner
from mmengine.utils import import_modules_from_strings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render PPP EDA demo sequence.")

    parser.add_argument(
        "config",
        type=str,
        help="Path to EDA config, e.g. configs/ppp_eda/ppp_eda.py.",
    )
    parser.add_argument(
        "--dataloader",
        type=str,
        default="eda_demo_dataloader",
        help="Dataloader variable in config.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="work_dirs/ppp_eda/demo_sequence",
        help="Output directory.",
    )
    parser.add_argument(
        "--render-every",
        type=int,
        default=1,
        help="Render every nth demo sample.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Maximum number of dataloader samples to visit.",
    )
    parser.add_argument(
        "--max-rendered",
        type=int,
        default=None,
        help="Maximum number of rendered frames.",
    )
    parser.add_argument(
        "--outputs",
        type=str,
        nargs="*",
        default=["points", "depth", "velocity"],
        choices=["points", "depth", "velocity"],
        help="Which overlays to save.",
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
                f"Use batch_size=1 for demo rendering."
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


def safe_text(x: Any) -> str:
    if x is None:
        return "None"
    return str(x)


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


def prepare_out_dir(out_dir: Path, overwrite: bool, outputs: List[str]) -> None:
    if out_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output directory already exists: {out_dir}\n"
                f"Use --overwrite if you want to replace it."
            )
        shutil.rmtree(out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)

    for name in outputs:
        (out_dir / name).mkdir(parents=True, exist_ok=True)


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


def build_title(
    *,
    overlay_name: str,
    eda_idx: int,
    rendered_idx: int,
    identity: Dict[str, Any],
    point_count: int,
) -> str:
    video_frame_idx = identity.get("video_frame_idx", None)
    demo_idx = identity.get("demo_idx", None)
    raw_idx = identity.get("idx", None)

    return (
        f"demo {overlay_name} | "
        f"rendered {rendered_idx} | "
        f"eda_idx {eda_idx} | "
        f"idx {safe_text(raw_idx)} | "
        f"demo_idx {safe_text(demo_idx)} | "
        f"video_frame {safe_text(video_frame_idx)} | "
        f"points {point_count}"
    )


def render_frame(
    *,
    img01: np.ndarray,
    gt: np.ndarray,
    out_dir: Path,
    eda_idx: int,
    rendered_idx: int,
    args: argparse.Namespace,
    identity: Dict[str, Any],
) -> Dict[str, Any]:
    points = extract_points(gt)
    point_count = int(len(points["x"]))

    base = f"frame_{rendered_idx:06d}_eda{eda_idx:06d}"

    if "points" in args.outputs:
        save_points_overlay(
            img01=img01,
            points=points,
            out_path=out_dir / "points" / f"{base}_points.png",
            title=build_title(
                overlay_name="points",
                eda_idx=eda_idx,
                rendered_idx=rendered_idx,
                identity=identity,
                point_count=point_count,
            ),
            point_size=args.point_size,
            alpha=args.alpha,
            fig_width=args.fig_width,
            dpi=args.dpi,
        )

    if "depth" in args.outputs:
        save_depth_overlay(
            img01=img01,
            points=points,
            out_path=out_dir / "depth" / f"{base}_depth.png",
            title=build_title(
                overlay_name="depth",
                eda_idx=eda_idx,
                rendered_idx=rendered_idx,
                identity=identity,
                point_count=point_count,
            ),
            point_size=args.point_size,
            alpha=args.alpha,
            fig_width=args.fig_width,
            dpi=args.dpi,
            depth_vmin=args.depth_vmin,
            depth_vmax=args.depth_vmax,
        )

    if "velocity" in args.outputs:
        save_velocity_overlay(
            img01=img01,
            points=points,
            out_path=out_dir / "velocity" / f"{base}_velocity.png",
            title=build_title(
                overlay_name="velocity",
                eda_idx=eda_idx,
                rendered_idx=rendered_idx,
                identity=identity,
                point_count=point_count,
            ),
            point_size=args.point_size,
            alpha=args.alpha,
            fig_width=args.fig_width,
            dpi=args.dpi,
            velocity_vmax=args.velocity_vmax,
        )

    return {
        "eda_idx": int(eda_idx),
        "rendered_idx": int(rendered_idx),
        "point_count": int(point_count),
        **identity,
    }


def main() -> None:
    args = parse_args()

    if args.render_every <= 0:
        raise ValueError("--render-every must be >= 1")

    out_dir = Path(args.out_dir)
    prepare_out_dir(out_dir, overwrite=args.overwrite, outputs=args.outputs)

    cfg = load_config(args.config)
    loader = build_dataloader_from_cfg(cfg, args.dataloader)

    try:
        total = len(loader.dataset)
    except Exception:
        total = None

    if args.max_frames is not None and total is not None:
        total = min(total, args.max_frames)

    rows: List[Dict[str, Any]] = []

    visited = 0
    rendered = 0

    pbar = tqdm(loader, total=total, desc="Rendering PPP demo sequence")

    stop = False

    for batch in pbar:
        samples = unpack_batch(batch)

        for sample in samples:
            eda_idx = visited
            visited += 1

            if args.max_frames is not None and eda_idx >= args.max_frames:
                stop = True
                break

            if eda_idx % args.render_every != 0:
                continue

            if args.max_rendered is not None and rendered >= args.max_rendered:
                stop = True
                break

            img01 = normalize_img_to_hwc_float01(sample["img"])
            gt = normalize_gt_to_hwc(sample["gt_seg_map"])

            if img01.shape[:2] != gt.shape[:2]:
                raise ValueError(
                    f"Image and GT shape mismatch for eda_idx={eda_idx}: "
                    f"img={img01.shape}, gt={gt.shape}"
                )

            identity = sample_identity(sample)

            row = render_frame(
                img01=img01,
                gt=gt,
                out_dir=out_dir,
                eda_idx=eda_idx,
                rendered_idx=rendered,
                args=args,
                identity=identity,
            )
            rows.append(row)

            rendered += 1
            pbar.set_postfix(visited=visited, rendered=rendered)

        if stop:
            break

    summary = {
        "config": str(args.config),
        "dataloader": args.dataloader,
        "out_dir": str(out_dir),
        "outputs": args.outputs,
        "render_every": int(args.render_every),
        "max_frames": args.max_frames,
        "max_rendered": args.max_rendered,
        "visited_samples": int(visited),
        "rendered_frames": int(rendered),
        "point_size": float(args.point_size),
        "depth_vmin": float(args.depth_vmin),
        "depth_vmax": float(args.depth_vmax),
        "velocity_vmax": float(args.velocity_vmax),
    }

    with open(out_dir / "render_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    with open(out_dir / "rendered_frames.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    print("Done.")
    print(f"Output directory: {out_dir}")
    print(f"Visited samples:  {visited}")
    print(f"Rendered frames:  {rendered}")
    print(f"Summary:          {out_dir / 'render_summary.json'}")
    print(f"Frame manifest:   {out_dir / 'rendered_frames.json'}")


if __name__ == "__main__":
    main()