#!/usr/bin/env bash
#SBATCH --job-name=ppp_eda_selrend
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
#   sbatch ppp_eda_select_render.sbatch train 500
#   sbatch ppp_eda_select_render.sbatch train 5000
#   sbatch ppp_eda_select_render.sbatch val 5000
#
# Arguments:
#   $1  SPLIT              val | train          default: val
#   $2  NUM_SAMPLES        scan size            default: 5000
#   $3  NUM_PER_CATEGORY   examples/category    default: 20
#   $4  POINT_SIZE         matplotlib area      default: 7
#   $5  OVERWRITE          1 or 0               default: 1
#
# -----------------------------------------------------------------------------

SPLIT="${1:-val}"
NUM_SAMPLES="${2:-5000}"
NUM_PER_CATEGORY="${3:-20}"
POINT_SIZE="${4:-7}"
OVERWRITE="${5:-1}"

PPP_EDA_ROOT="${PPP_EDA_ROOT:-/e/scratch/nxtaim-1/huber7/ppp_eda}"
JSC_DIR="${SLURM_SUBMIT_DIR}"
REPO_DIR="$(realpath "${JSC_DIR}/..")"
CONFIG="configs/ppp_eda/ppp_eda.py"

cd "$REPO_DIR"

source "${JSC_DIR}/activate_jupiter.sh"

PYTHON_BIN="$(command -v python)"

case "$SPLIT" in
  val)
    DATALOADER="eda_val_dataloader"
    SCAN_DIR="${PPP_EDA_ROOT}/val_seed43_n${NUM_SAMPLES}"

    # Critical: must match the scan job.
    export PPP_EDA_VAL_SAMPLES="$NUM_SAMPLES"
    ;;

  train)
    DATALOADER="eda_train_dataloader"
    SCAN_DIR="${PPP_EDA_ROOT}/train_seed42_n${NUM_SAMPLES}"

    # Critical: must match the scan job.
    export PPP_EDA_TRAIN_SAMPLES="$NUM_SAMPLES"
    ;;

  *)
    echo "ERROR: unknown SPLIT='$SPLIT'. Use val or train."
    exit 1
    ;;
esac

if [[ ! -d "$SCAN_DIR" ]]; then
  echo "ERROR: scan directory does not exist:"
  echo "$SCAN_DIR"
  echo
  echo "Run the scan first, for example:"
  echo "  sbatch ppp_eda_scan.sbatch $SPLIT $NUM_SAMPLES"
  exit 1
fi

if [[ ! -f "$SCAN_DIR/frames.csv" ]]; then
  echo "ERROR: missing $SCAN_DIR/frames.csv"
  exit 1
fi

if [[ ! -f "$SCAN_DIR/point_values.npz" ]]; then
  echo "ERROR: missing $SCAN_DIR/point_values.npz"
  exit 1
fi

export PPP_EDA_NUM_WORKERS=4
export PPP_EDA_BATCH_SIZE=1
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=""

echo "============================================================"
echo "PPP EDA select + render"
echo "============================================================"
echo "Date:             $(date)"
echo "Host:             $(hostname)"
echo "Submit dir:       $JSC_DIR"
echo "Repo:             $REPO_DIR"
echo "Config:           $CONFIG"
echo "Split:            $SPLIT"
echo "Dataloader:       $DATALOADER"
echo "Scan dir:         $SCAN_DIR"
echo "Num samples:      $NUM_SAMPLES"
echo "Num/category:     $NUM_PER_CATEGORY"
echo "Point size:       $POINT_SIZE"
echo "Overwrite:        $OVERWRITE"
echo "Python:           $PYTHON_BIN"
echo "PPP_EDA_TRAIN_SAMPLES: ${PPP_EDA_TRAIN_SAMPLES:-not set}"
echo "PPP_EDA_VAL_SAMPLES:   ${PPP_EDA_VAL_SAMPLES:-not set}"
echo "============================================================"

SELECT_CMD=(
  "$PYTHON_BIN" -u tools/ppp_eda/select_ppp_examples.py
  --scan-dir "$SCAN_DIR"
  --num-per-category "$NUM_PER_CATEGORY"
)

RENDER_CMD=(
  "$PYTHON_BIN" -u tools/ppp_eda/render_ppp_examples.py
  "$CONFIG"
  --dataloader "$DATALOADER"
  --scan-dir "$SCAN_DIR"
  --point-size "$POINT_SIZE"
)

if [[ "$OVERWRITE" == "1" ]]; then
  SELECT_CMD+=(--overwrite)
  RENDER_CMD+=(--overwrite)
fi

echo "Select command:"
printf ' %q' "${SELECT_CMD[@]}"
echo
echo "============================================================"

srun --export=ALL "${SELECT_CMD[@]}"

echo "============================================================"
echo "Render command:"
printf ' %q' "${RENDER_CMD[@]}"
echo
echo "============================================================"

srun --export=ALL "${RENDER_CMD[@]}"

echo "============================================================"
echo "Finished at: $(date)"
echo "Output files:"
ls -lh "$SCAN_DIR"
echo
echo "Example folders:"
find "$SCAN_DIR/examples" -maxdepth 2 -type f | sort | head -80
echo "============================================================"