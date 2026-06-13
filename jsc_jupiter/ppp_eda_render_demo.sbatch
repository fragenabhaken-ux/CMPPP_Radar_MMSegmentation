#!/usr/bin/env bash
#SBATCH --job-name=ppp_eda_demo
#SBATCH --account=nxtaim-1
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=12
#SBATCH --time=00:30:00
#SBATCH --output=logs/%j_%x.out
#SBATCH --error=logs/%j_%x.err
#SBATCH --export=NONE

set -euo pipefail

# -----------------------------------------------------------------------------
# Usage from inside jsc_jupiter/
# -----------------------------------------------------------------------------
#
#   cd /e/project1/nxtaim-1/huber7/repos/mmsegmentation_ppp/jsc_jupiter
#   mkdir -p logs
#
#   sbatch ppp_eda_render_demo.sbatch
#   sbatch ppp_eda_render_demo.sbatch demo_sequence_debug_5 1 5
#   sbatch ppp_eda_render_demo.sbatch demo_sequence 1 none
#   sbatch ppp_eda_render_demo.sbatch demo_sequence_every4 4 none
#
# Arguments:
#   $1  OUT_NAME       output folder name below work_dirs/ppp_eda/   default: demo_sequence
#   $2  RENDER_EVERY   render every nth demo sample                  default: 1
#   $3  MAX_RENDERED   max rendered frames, or "none"                default: none
#   $4  POINT_SIZE     matplotlib area                               default: 7
#   $5  OVERWRITE      1 or 0                                        default: 1
#
# -----------------------------------------------------------------------------

OUT_NAME="${1:-demo_sequence}"
RENDER_EVERY="${2:-1}"
MAX_RENDERED="${3:-none}"
POINT_SIZE="${4:-7}"
OVERWRITE="${5:-1}"

JSC_DIR="${SLURM_SUBMIT_DIR}"
REPO_DIR="$(realpath "${JSC_DIR}/..")"
CONFIG="configs/ppp_eda/ppp_eda.py"
DATALOADER="eda_demo_dataloader"
OUT_DIR="work_dirs/ppp_eda/${OUT_NAME}"

cd "$REPO_DIR"

source "${JSC_DIR}/activate_jupiter.sh"

PYTHON_BIN="$(command -v python)"

export PPP_EDA_NUM_WORKERS=4
export PPP_EDA_BATCH_SIZE=1
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=""

echo "============================================================"
echo "PPP EDA demo render"
echo "============================================================"
echo "Date:           $(date)"
echo "Host:           $(hostname)"
echo "Submit dir:     $JSC_DIR"
echo "Repo:           $REPO_DIR"
echo "Config:         $CONFIG"
echo "Dataloader:     $DATALOADER"
echo "Out dir:        $OUT_DIR"
echo "Render every:   $RENDER_EVERY"
echo "Max rendered:   $MAX_RENDERED"
echo "Point size:     $POINT_SIZE"
echo "Overwrite:      $OVERWRITE"
echo "Python:         $PYTHON_BIN"
echo "============================================================"

CMD=(
  "$PYTHON_BIN" -u tools/ppp_eda/render_ppp_demo_sequence.py
  "$CONFIG"
  --dataloader "$DATALOADER"
  --out-dir "$OUT_DIR"
  --render-every "$RENDER_EVERY"
  --point-size "$POINT_SIZE"
)

if [[ "$MAX_RENDERED" != "none" ]]; then
  CMD+=(--max-rendered "$MAX_RENDERED")
fi

if [[ "$OVERWRITE" == "1" ]]; then
  CMD+=(--overwrite)
fi

echo "Command:"
printf ' %q' "${CMD[@]}"
echo
echo "============================================================"

srun --export=ALL "${CMD[@]}"

echo "============================================================"
echo "Finished at: $(date)"
echo "Output files:"
find "$OUT_DIR" -maxdepth 2 -type f | sort | head -80
echo "============================================================"