#!/bin/bash
set -euo pipefail

cd /p/project1/nxtaim-1/huber7/repos/mmsegmentation_ppp

source jsc/modules.sh
source jsc/activate.sh

FRAME_WORK_DIR="${1:?Usage: bash make_video.sh /p/scratch/.../unet_160k_demo_vid_JOBID}"
FRAME_DIR=$(find "$FRAME_WORK_DIR/demo_frames/test" -mindepth 1 -maxdepth 1 -type d | grep -i EVEN | head -n 1)

mkdir -p "$FRAME_WORK_DIR/demo_video"

python make_ppp_demo_video.py \
  --frames "$FRAME_DIR" \
  --out "$FRAME_WORK_DIR/demo_video/ppp_demo_even" \
  --fps 10 \
  --quality 20 \
  --preset fast