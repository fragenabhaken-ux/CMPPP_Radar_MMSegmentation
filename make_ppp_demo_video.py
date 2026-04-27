"""Create a demo video from PNG frames produced by PPPDemoVisHook.

Example:
    python make_ppp_demo_video.py \
        --frames /p/scratch/nxtaim-1/huber7/work_dirs/MY_RUN/demo_frames/test \
        --out /p/scratch/nxtaim-1/huber7/work_dirs/MY_RUN/demo_video/demo \
        --fps 10 \
        --quality 20 \
        --preset fast

The output path is given without suffix. FFMPEGVideoWriter appends e.g.
_qt20_fast.mp4 depending on the quality/preset arguments.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
from PIL import Image

from aumovio.jonas_FFMPEGVideoWriter import FFMPEGVideoWriter


def collect_frames(frames_root: Path, pattern: str) -> list[Path]:
    frames = sorted(frames_root.rglob(pattern))
    if not frames:
        raise FileNotFoundError(f"No frames found below {frames_root} with pattern {pattern!r}")
    return frames


def read_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"), dtype=np.uint8)


def make_video(
    frames_root: str | os.PathLike,
    out: str | os.PathLike,
    *,
    pattern: str = "*.png",
    fps: float = 10.0,
    quality: int | None = 20,
    preset: str | None = "fast",
    width: int | None = None,
    height: int | None = None,
    title_from_filename: bool = False,
    max_frames: int | None = None,
) -> Path:
    frames_root = Path(frames_root)
    frames = collect_frames(frames_root, pattern)
    if max_frames is not None:
        frames = frames[:max_frames]

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)

    with FFMPEGVideoWriter(
        video_path=str(out),
        file_extension="mp4",
        fps=fps,
        quality=quality,
        preset=preset,
        width=width,
        height=height,
    ) as vw:
        for i, frame_path in enumerate(frames):
            img = read_rgb(frame_path)
            title = frame_path.stem if title_from_filename else None
            vw.write(img, title=title)
            if (i + 1) % 25 == 0:
                print(f"written {i + 1}/{len(frames)} frames")

        video_path = Path(vw.video_path)

    print(f"Done. Wrote {len(frames)} frames to: {video_path}")
    return video_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", required=True, help="Root folder containing PNG frames.")
    parser.add_argument("--out", required=True, help="Output video path without file suffix.")
    parser.add_argument("--pattern", default="*.png")
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--quality", type=int, default=20)
    parser.add_argument("--preset", default="fast")
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--title-from-filename", action="store_true")
    parser.add_argument("--max-frames", type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    make_video(
        frames_root=args.frames,
        out=args.out,
        pattern=args.pattern,
        fps=args.fps,
        quality=args.quality,
        preset=args.preset,
        width=args.width,
        height=args.height,
        title_from_filename=args.title_from_filename,
        max_frames=args.max_frames,
    )
