#!/bin/bash
#SBATCH --job-name=extract_prefetch_unseen
#SBATCH --account=nxtaim-1
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --output=prefetch_unseen_%j.out
#SBATCH --error=prefetch_unseen_%j.err
#SBATCH --export=NONE

set -euo pipefail

# Save this file in jsc_jupiter/ and submit it from that directory:
#   sbatch extract_prefetch_unseen.sbatch
REPO_ROOT="$(cd "$SLURM_SUBMIT_DIR/.." && pwd)"
SCRIPT="$REPO_ROOT/ppp_analysis/extract_prefetch_unseen.py"
OUTPUT_DIR="$REPO_ROOT/ppp_analysis/prefetch_unseen_results"

module --force purge
module load Stages/2025 GCCcore/.13.3.0
module load tensorboard

PYTHON_BIN="$(command -v python3 || command -v python)"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

mkdir -p "$OUTPUT_DIR"

echo "SLURM_JOB_ID=${SLURM_JOB_ID:-unset}"
echo "HOST=$(hostname)"
echo "REPO_ROOT=$REPO_ROOT"
echo "PYTHON_BIN=$PYTHON_BIN"
echo "SCRIPT=$SCRIPT"
echo "OUTPUT_DIR=$OUTPUT_DIR"
echo

if [[ ! -f "$SCRIPT" ]]; then
    echo "ERROR: Python script not found: $SCRIPT" >&2
    exit 1
fi

"$PYTHON_BIN" "$SCRIPT" \
    --workers "$SLURM_CPUS_PER_TASK" \
    --output-dir "$OUTPUT_DIR"
