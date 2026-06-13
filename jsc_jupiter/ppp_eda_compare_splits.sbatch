#!/usr/bin/env bash
#SBATCH --job-name=ppp_eda_compare_splits
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

NUM_SAMPLES="${1:-5000}"
OVERWRITE="${2:-1}"

JSC_DIR="${SLURM_SUBMIT_DIR}"
REPO_DIR="$(realpath "${JSC_DIR}/..")"
PPP_EDA_ROOT="${PPP_EDA_ROOT:-/e/scratch/nxtaim-1/huber7/ppp_eda}"

cd "$REPO_DIR"

source "${JSC_DIR}/activate_jupiter.sh"

PYTHON_BIN="$(command -v python)"

TRAIN_DIR="${PPP_EDA_ROOT}/train_seed42_n${NUM_SAMPLES}"
VAL_DIR="${PPP_EDA_ROOT}/val_seed43_n${NUM_SAMPLES}"
OUT_DIR="${PPP_EDA_ROOT}/compare_train_seed42_n${NUM_SAMPLES}_vs_val_seed43_n${NUM_SAMPLES}"

for DIR in "$TRAIN_DIR" "$VAL_DIR"; do
  if [[ ! -d "$DIR" ]]; then
    echo "Missing scan directory:"
    echo "$DIR"
    exit 1
  fi

  if [[ ! -f "${DIR}/frames.csv" ]]; then
    echo "Missing frames.csv:"
    echo "${DIR}/frames.csv"
    exit 1
  fi

  if [[ ! -f "${DIR}/point_values.npz" ]]; then
    echo "Missing point_values.npz:"
    echo "${DIR}/point_values.npz"
    exit 1
  fi
done

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=""

CMD=(
  "$PYTHON_BIN" -u tools/ppp_eda/compare_ppp_eda_splits.py
  --train-dir "$TRAIN_DIR"
  --val-dir "$VAL_DIR"
  --out-dir "$OUT_DIR"
)

if [[ "$OVERWRITE" == "1" ]]; then
  CMD+=(--overwrite)
fi

echo "REPO_DIR    = $REPO_DIR"
echo "NUM_SAMPLES = $NUM_SAMPLES"
echo "TRAIN_DIR   = $TRAIN_DIR"
echo "VAL_DIR     = $VAL_DIR"
echo "OUT_DIR     = $OUT_DIR"
echo "PYTHON_BIN  = $PYTHON_BIN"
echo "CMD         = ${CMD[*]}"

srun --export=ALL "${CMD[@]}"