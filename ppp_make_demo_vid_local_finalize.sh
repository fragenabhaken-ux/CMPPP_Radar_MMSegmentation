#!/bin/bash
set -euo pipefail

cd ~/repos/mmsegmentation_ppp

FRAME_WORK_DIR="${1:?Usage: bash ppp_make_demo_vid_local_finalize.sh /mnt/c/.../unet}"
FRAME_DIR=$(find "$FRAME_WORK_DIR/test" -mindepth 1 -maxdepth 1 -type d | grep -i EVEN | head -n 1)

echo "FRAME_DIR=$FRAME_DIR"
find "$FRAME_DIR" -name "*.png" | wc -l

mkdir -p "$FRAME_WORK_DIR/demo_video"

micromamba run -n mmseg python make_ppp_demo_video.py \
  --frames "$FRAME_DIR" \
  --out "$FRAME_WORK_DIR/demo_video/ppp_demo_even" \
  --fps 10 \
  --quality 20 \
  --preset fast \
  --width 1260 \
  --height 1796