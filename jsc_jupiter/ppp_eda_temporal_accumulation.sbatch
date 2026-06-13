#!/usr/bin/env bash
#SBATCH --job-name=ppp_eda_temporal_accumulation
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

CENTER_EDA_IDX="${1:-160}"
RADIUS="${2:-20}"
STRIDE="${3:-1}"
OUT_NAME="${4:-demo_temporal_accumulation}"
POINT_SIZE="${5:-7}"
OVERWRITE="${6:-1}"

JSC_DIR="${SLURM_SUBMIT_DIR}"
REPO_DIR="$(realpath "${JSC_DIR}/..")"
PPP_EDA_ROOT="${PPP_EDA_ROOT:-/e/scratch/nxtaim-1/huber7/ppp_eda}"

cd "$REPO_DIR"

source "${JSC_DIR}/activate_jupiter.sh"

PYTHON_BIN="$(command -v python)"

CONFIG="configs/ppp_eda/ppp_eda.py"
DATALOADER="eda_demo_dataloader"
OUT_DIR="${PPP_EDA_ROOT}/${OUT_NAME}"

export PPP_EDA_NUM_WORKERS=4
export PPP_EDA_BATCH_SIZE=1
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=""

CMD=(
  "$PYTHON_BIN" -u tools/ppp_eda/render_ppp_demo_temporal_accumulation.py
  "$CONFIG"
  --dataloader "$DATALOADER"
  --center-eda-idx "$CENTER_EDA_IDX"
  --radius "$RADIUS"
  --stride "$STRIDE"
  --out-dir "$OUT_DIR"
  --point-size "$POINT_SIZE"
  --save-csv
)

if [[ "$OVERWRITE" == "1" ]]; then
  CMD+=(--overwrite)
fi

echo "REPO_DIR       = $REPO_DIR"
echo "CONFIG         = $CONFIG"
echo "DATALOADER     = $DATALOADER"
echo "CENTER_EDA_IDX = $CENTER_EDA_IDX"
echo "RADIUS         = $RADIUS"
echo "STRIDE         = $STRIDE"
echo "OUT_DIR        = $OUT_DIR"
echo "POINT_SIZE     = $POINT_SIZE"
echo "PYTHON_BIN     = $PYTHON_BIN"
echo "CMD            = ${CMD[*]}"

srun --export=ALL "${CMD[@]}"