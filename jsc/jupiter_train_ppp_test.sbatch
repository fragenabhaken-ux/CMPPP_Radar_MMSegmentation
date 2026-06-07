#!/bin/bash
#SBATCH --job-name=ppp_dinov3_rope_big_neck_40k_dbwa1_jupiter_test
#SBATCH --account=nxtaim-1
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=12
#SBATCH --time=02:00:00
#SBATCH --output=logs/%j_%x.out
#SBATCH --error=logs/%j_%x.err
#SBATCH --export=ALL

set -euo pipefail

# Submit this script from the jsc folder.
# Then SLURM_SUBMIT_DIR is .../mmsegmentation_ppp/jsc,
# and this moves to the repo root.
cd "$SLURM_SUBMIT_DIR/.." || exit 1

mkdir -p logs

source jsc/activate_jupiter.sh

export SRUN_CPUS_PER_TASK="${SLURM_CPUS_PER_TASK}"

MASTER_HOST="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
export MASTER_ADDR="$(getent ahostsv4 "$MASTER_HOST" | awk 'NR==1{print $1}')"
export MASTER_PORT=7010

echo "MASTER_HOST=$MASTER_HOST"
echo "MASTER_ADDR=$MASTER_ADDR"
echo "MASTER_PORT=$MASTER_PORT"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-unset}"
echo "SLURM_JOB_NAME=${SLURM_JOB_NAME:-unset}"
echo "SLURM_NODELIST=${SLURM_JOB_NODELIST:-unset}"
echo "SLURM_NTASKS=${SLURM_NTASKS:-unset}"
echo "SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-unset}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

echo
echo "=== Environment check ==="
which python
python -V

python - <<'PY'
mods = ["torch", "torchvision", "timm", "mmengine", "mmcv", "mmseg"]
for m in mods:
    try:
        mod = __import__(m)
        print(f"{m:12s} OK   {getattr(mod, '__version__', 'no __version__')}")
    except Exception as e:
        print(f"{m:12s} FAIL {type(e).__name__}: {e}")

import torch
print("torch file:", torch.__file__)
print("torch cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("device 0:", torch.cuda.get_device_name(0))
PY

CONFIG="configs/ppp/ppp_dinov3_rope_big_neck_40k_dbwa1.py"

WORK_DIR="/e/scratch/nxtaim-1/huber7/work_dirs/${SLURM_JOB_ID}_${SLURM_JOB_NAME}"

mkdir -p "$WORK_DIR"

echo
echo "CONFIG=$CONFIG"
echo "WORK_DIR=$WORK_DIR"

echo
echo "=== Starting MMSeg training ==="

srun --ntasks="$SLURM_NTASKS" \
     --gpus-per-task=1 \
     --cpus-per-task="$SLURM_CPUS_PER_TASK" \
     --gpu-bind=none \
  python tools/train.py "$CONFIG" \
    --launcher slurm \
    --work-dir "$WORK_DIR"