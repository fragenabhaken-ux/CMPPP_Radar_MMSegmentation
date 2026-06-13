#!/usr/bin/env bash
#SBATCH --job-name=ppp_eda_scan
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
#   sbatch ppp_eda_scan.sbatch val 5000
#   sbatch ppp_eda_scan.sbatch train 5000
#   sbatch ppp_eda_scan.sbatch val 500
#   sbatch ppp_eda_scan.sbatch train 500
#   sbatch ppp_eda_scan.sbatch demo
#
# Arguments:
#   $1  SPLIT        val | train | demo       default: val
#   $2  NUM_SAMPLES  only for val/train       default: 5000
#   $3  NUM_WORKERS                         default: 4
#   $4  BATCH_SIZE                          default: 1
#   $5  OVERWRITE    1 or 0                  default: 1
#
# -----------------------------------------------------------------------------

SPLIT="${1:-val}"
NUM_SAMPLES="${2:-5000}"
NUM_WORKERS="${3:-4}"
BATCH_SIZE="${4:-1}"
OVERWRITE="${5:-1}"

# This assumes the job is submitted from:
#   mmsegmentation_ppp/jsc_jupiter/
JSC_DIR="${SLURM_SUBMIT_DIR}"
REPO_DIR="$(realpath "${JSC_DIR}/..")"
PPP_EDA_ROOT="${PPP_EDA_ROOT:-/e/scratch/nxtaim-1/huber7/ppp_eda}"
CONFIG="configs/ppp_eda/ppp_eda.py"

cd "$REPO_DIR"

mkdir -p "$PPP_EDA_ROOT"

# -----------------------------------------------------------------------------
# Activate environment
# -----------------------------------------------------------------------------

source "${JSC_DIR}/activate_jupiter.sh"

# Important:
# With #SBATCH --export=NONE, the environment is very clean.
# Therefore we resolve the absolute python path after activation
# and pass the environment explicitly to srun.
PYTHON_BIN="$(command -v python)"

# -----------------------------------------------------------------------------
# Resolve dataloader and output directory
# -----------------------------------------------------------------------------

case "$SPLIT" in
  val)
    DATALOADER="eda_val_dataloader"
    OUT_DIR="${PPP_EDA_ROOT}/val_seed43_n${NUM_SAMPLES}"
    export PPP_EDA_VAL_SAMPLES="$NUM_SAMPLES"
    ;;

  train)
    DATALOADER="eda_train_dataloader"
    OUT_DIR="${PPP_EDA_ROOT}/train_seed42_n${NUM_SAMPLES}"
    export PPP_EDA_TRAIN_SAMPLES="$NUM_SAMPLES"
    ;;

  demo)
    DATALOADER="eda_demo_dataloader"
    OUT_DIR="${PPP_EDA_ROOT}/demo_scene"
    ;;

  *)
    echo "ERROR: unknown SPLIT='$SPLIT'. Use val, train, or demo."
    exit 1
    ;;
esac

export PPP_EDA_NUM_WORKERS="$NUM_WORKERS"
export PPP_EDA_BATCH_SIZE="$BATCH_SIZE"
export PYTHONUNBUFFERED=1

# We allocate a Booster node, but this EDA script itself does not need CUDA.
export CUDA_VISIBLE_DEVICES=""

echo "============================================================"
echo "PPP EDA scan"
echo "============================================================"
echo "Date:        $(date)"
echo "Host:        $(hostname)"
echo "Submit dir:  $JSC_DIR"
echo "Repo:        $REPO_DIR"
echo "Config:      $CONFIG"
echo "Split:       $SPLIT"
echo "Dataloader:  $DATALOADER"
echo "EDA root:    $PPP_EDA_ROOT"
echo "Out dir:     $OUT_DIR"
echo "Num samples: $NUM_SAMPLES"
echo "Workers:     $NUM_WORKERS"
echo "Batch size:  $BATCH_SIZE"
echo "Overwrite:   $OVERWRITE"
echo "Python:      $PYTHON_BIN"
echo "============================================================"

CMD=(
  "$PYTHON_BIN" -u tools/ppp_eda/scan_ppp_eda.py
  "$CONFIG"
  --dataloader "$DATALOADER"
  --out-dir "$OUT_DIR"
)

if [[ "$SPLIT" != "demo" ]]; then
  CMD+=(--max-samples "$NUM_SAMPLES")
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
ls -lh "$OUT_DIR"
echo "============================================================"