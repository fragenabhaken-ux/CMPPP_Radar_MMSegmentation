#!/usr/bin/env bash
#SBATCH --job-name=ppp_eda_plot_summary
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

SPLIT="${1:-train}"
NUM_SAMPLES="${2:-5000}"
OVERWRITE="${3:-1}"

JSC_DIR="${SLURM_SUBMIT_DIR}"
REPO_DIR="$(realpath "${JSC_DIR}/..")"
PPP_EDA_ROOT="${PPP_EDA_ROOT:-/e/scratch/nxtaim-1/huber7/ppp_eda}"

cd "$REPO_DIR"

source "${JSC_DIR}/activate_jupiter.sh"

PYTHON_BIN="$(command -v python)"

case "$SPLIT" in
  train)
    SCAN_DIR="${PPP_EDA_ROOT}/train_seed42_n${NUM_SAMPLES}"
    ;;
  val)
    SCAN_DIR="${PPP_EDA_ROOT}/val_seed43_n${NUM_SAMPLES}"
    ;;
  *)
    echo "Unknown split: ${SPLIT}"
    echo "Use: train or val"
    exit 1
    ;;
esac

if [[ ! -d "$SCAN_DIR" ]]; then
  echo "Missing scan directory:"
  echo "$SCAN_DIR"
  exit 1
fi

if [[ ! -f "${SCAN_DIR}/frames.csv" ]]; then
  echo "Missing frames.csv:"
  echo "${SCAN_DIR}/frames.csv"
  exit 1
fi

if [[ ! -f "${SCAN_DIR}/point_values.npz" ]]; then
  echo "Missing point_values.npz:"
  echo "${SCAN_DIR}/point_values.npz"
  exit 1
fi

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=""

CMD=(
  "$PYTHON_BIN" -u tools/ppp_eda/plot_ppp_eda_summary.py
  --scan-dir "$SCAN_DIR"
)

if [[ "$OVERWRITE" == "1" ]]; then
  CMD+=(--overwrite)
fi

echo "REPO_DIR   = $REPO_DIR"
echo "SPLIT      = $SPLIT"
echo "SCAN_DIR   = $SCAN_DIR"
echo "PYTHON_BIN = $PYTHON_BIN"
echo "CMD        = ${CMD[*]}"

srun --export=ALL "${CMD[@]}"