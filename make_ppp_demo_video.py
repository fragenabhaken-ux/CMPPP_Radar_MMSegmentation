"""
Create a demo video from PNG frames produced by PPPDemoVisHook / PPPDemoBEVVisHook.

This version preserves aspect ratio.

Default behavior:
    - If --width and --height are not given, the video uses the first frame size.
    - If --width and --height are given, frames are resized without distortion
      and padded with black background.
    - Use --fit crop if you want to fill the target canvas by cropping.
    - Use --fit stretch only if you explicitly want distortion.

Example:
    python make_ppp_demo_video.py \
        --frames /p/scratch/nxtaim-1/huber7/work_dirs/MY_RUN/demo_frames/test \
        --out /p/scratch/nxtaim-1/huber7/work_dirs/MY_RUN/demo_video/demo \
        --fps 10 \
        --quality 20 \
        --preset fast \
        --fit pad

For a 1080p video without stretching:
    python make_ppp_demo_video.py \
        --frames /p/scratch/nxtaim-1/huber7/work_dirs/MY_RUN/demo_frames/test \
        --out /p/scratch/nxtaim-1/huber7/work_dirs/MY_RUN/demo_video/demo \
        --fps 10 \
        --quality 20 \
        --preset fast \
        --width 1920 \
        --height 1080 \
        --fit pad
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from aumovio.jonas_FFMPEGVideoWriter import FFMPEGVideoWriter


def collect_frames(frames_root: Path, pattern: str) -> list[Path]:
    frames = sorted(frames_root.rglob(pattern))
    if not frames:
        raise FileNotFoundError(
            f"No frames found below {frames_root} with pattern {pattern!r}"
        )
    return frames


def read_rgb_pil(path: Path) -> Image.Image:
    with Image.open(path) as im:
        return im.convert("RGB")


def make_even(x: int) -> int:
    """
    Many mp4 encoders prefer even dimensions.
    """
    x = int(x)
    return x if x % 2 == 0 else x - 1


def infer_target_size(
    first_frame: Image.Image,
    width: int | None,
    height: int | None,
) -> tuple[int, int]:
    """
    Infer video canvas size.

    If neither width nor height is given, use the first frame size.
    If only one is given, preserve first-frame aspect ratio.
    If both are given, use exactly that canvas.
    """
    src_w, src_h = first_frame.size

    if width is None and height is None:
        return make_even(src_w), make_even(src_h)

    if width is not None and height is not None:
        return make_even(width), make_even(height)

    aspect = src_w / src_h

    if width is not None:
        target_w = make_even(width)
        target_h = make_even(round(target_w / aspect))
        return target_w, target_h

    assert height is not None
    target_h = make_even(height)
    target_w = make_even(round(target_h * aspect))
    return target_w, target_h


def prepare_frame(
    img: Image.Image,
    target_size: tuple[int, int],
    fit: str,
    background: tuple[int, int, int] = (0, 0, 0),
) -> np.ndarray:
    """
    Convert one PIL image to an RGB uint8 array with target_size.

    fit='pad':
        Preserve aspect ratio and pad with background.

    fit='crop':
        Preserve aspect ratio and crop to fill the canvas.

    fit='stretch':
        Resize directly to target_size, distorting aspect ratio.
    """
    target_w, target_h = target_size

    if fit == "stretch":
        out = img.resize((target_w, target_h), Image.Resampling.LANCZOS)

    elif fit == "crop":
        out = ImageOps.fit(
            img,
            (target_w, target_h),
            method=Image.Resampling.LANCZOS,
            centering=(0.5, 0.5),
        )

    elif fit == "pad":
        resized = ImageOps.contain(
            img,
            (target_w, target_h),
            method=Image.Resampling.LANCZOS,
        )
        out = Image.new("RGB", (target_w, target_h), background)
        x = (target_w - resized.size[0]) // 2
        y = (target_h - resized.size[1]) // 2
        out.paste(resized, (x, y))

    else:
        raise ValueError(f"Invalid fit={fit!r}. Use 'pad', 'crop', or 'stretch'.")

    return np.asarray(out, dtype=np.uint8)


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
    fit: str = "pad",
    title_from_filename: bool = False,
    max_frames: int | None = None,
) -> Path:
    frames_root = Path(frames_root)
    frames = collect_frames(frames_root, pattern)

    if max_frames is not None:
        frames = frames[:max_frames]

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)

    first = read_rgb_pil(frames[0])
    target_w, target_h = infer_target_size(first, width=width, height=height)

    print(f"Found {len(frames)} frames.")
    print(f"First frame size: {first.size[0]} x {first.size[1]}")
    print(f"Video canvas:     {target_w} x {target_h}")
    print(f"Fit mode:         {fit}")

    with FFMPEGVideoWriter(
        video_path=str(out),
        file_extension="mp4",
        fps=fps,
        quality=quality,
        preset=preset,
        width=target_w,
        height=target_h,
    ) as vw:
        for i, frame_path in enumerate(frames):
            img_pil = read_rgb_pil(frame_path)
            img = prepare_frame(
                img=img_pil,
                target_size=(target_w, target_h),
                fit=fit,
                background=(0, 0, 0),
            )

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
    parser.add_argument(
        "--fit",
        default="pad",
        choices=["pad", "crop", "stretch"],
        help=(
            "pad preserves aspect ratio and adds black bars, "
            "crop preserves aspect ratio and cuts borders, "
            "stretch distorts the image."
        ),
    )
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
        fit=args.fit,
        title_from_filename=args.title_from_filename,
        max_frames=args.max_frames,
    )


# """Create a demo video from PNG frames produced by PPPDemoVisHook.

# Example:
#     python make_ppp_demo_video.py \
#         --frames /p/scratch/nxtaim-1/huber7/work_dirs/MY_RUN/demo_frames/test \
#         --out /p/scratch/nxtaim-1/huber7/work_dirs/MY_RUN/demo_video/demo \
#         --fps 10 \
#         --quality 20 \
#         --preset fast

# The output path is given without suffix. FFMPEGVideoWriter appends e.g.
# _qt20_fast.mp4 depending on the quality/preset arguments.
# """

# from __future__ import annotations

# import argparse
# import os
# from pathlib import Path

# import numpy as np
# from PIL import Image

# from aumovio.jonas_FFMPEGVideoWriter import FFMPEGVideoWriter


# def collect_frames(frames_root: Path, pattern: str) -> list[Path]:
#     frames = sorted(frames_root.rglob(pattern))
#     if not frames:
#         raise FileNotFoundError(f"No frames found below {frames_root} with pattern {pattern!r}")
#     return frames


# def read_rgb(path: Path) -> np.ndarray:
#     with Image.open(path) as im:
#         return np.asarray(im.convert("RGB"), dtype=np.uint8)


# def make_video(
#     frames_root: str | os.PathLike,
#     out: str | os.PathLike,
#     *,
#     pattern: str = "*.png",
#     fps: float = 10.0,
#     quality: int | None = 20,
#     preset: str | None = "fast",
#     width: int | None = None,
#     height: int | None = None,
#     title_from_filename: bool = False,
#     max_frames: int | None = None,
# ) -> Path:
#     frames_root = Path(frames_root)
#     frames = collect_frames(frames_root, pattern)
#     if max_frames is not None:
#         frames = frames[:max_frames]

#     out = Path(out)
#     out.parent.mkdir(parents=True, exist_ok=True)

#     with FFMPEGVideoWriter(
#         video_path=str(out),
#         file_extension="mp4",
#         fps=fps,
#         quality=quality,
#         preset=preset,
#         width=width,
#         height=height,
#     ) as vw:
#         for i, frame_path in enumerate(frames):
#             img = read_rgb(frame_path)
#             title = frame_path.stem if title_from_filename else None
#             vw.write(img, title=title)
#             if (i + 1) % 25 == 0:
#                 print(f"written {i + 1}/{len(frames)} frames")

#         video_path = Path(vw.video_path)

#     print(f"Done. Wrote {len(frames)} frames to: {video_path}")
#     return video_path


# def parse_args() -> argparse.Namespace:
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--frames", required=True, help="Root folder containing PNG frames.")
#     parser.add_argument("--out", required=True, help="Output video path without file suffix.")
#     parser.add_argument("--pattern", default="*.png")
#     parser.add_argument("--fps", type=float, default=10.0)
#     parser.add_argument("--quality", type=int, default=20)
#     parser.add_argument("--preset", default="fast")
#     parser.add_argument("--width", type=int, default=None)
#     parser.add_argument("--height", type=int, default=None)
#     parser.add_argument("--title-from-filename", action="store_true")
#     parser.add_argument("--max-frames", type=int, default=None)
#     return parser.parse_args()


# if __name__ == "__main__":
#     args = parse_args()
#     make_video(
#         frames_root=args.frames,
#         out=args.out,
#         pattern=args.pattern,
#         fps=args.fps,
#         quality=args.quality,
#         preset=args.preset,
#         width=args.width,
#         height=args.height,
#         title_from_filename=args.title_from_filename,
#         max_frames=args.max_frames,
#     )
