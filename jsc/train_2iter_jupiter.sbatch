#!/bin/bash
#SBATCH --job-name=ppp_2iter_jupiter_test
#SBATCH --account=nxtaim-1
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=12
#SBATCH --time=00:20:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --export=NONE

set -euo pipefail

# Submit this script from the jsc folder.
cd "$SLURM_SUBMIT_DIR/.." || exit 1

mkdir -p jsc/logs

source jsc/activate_jupiter.sh

export SRUN_CPUS_PER_TASK="${SLURM_CPUS_PER_TASK}"

MASTER_HOST="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
export MASTER_ADDR="$(getent ahostsv4 "$MASTER_HOST" | awk 'NR==1{print $1}')"
export MASTER_PORT=7010

export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

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
import torch
print("torch:", torch.__version__, torch.version.cuda, torch.__file__)
print("cuda available:", torch.cuda.is_available())
print("device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("device 0:", torch.cuda.get_device_name(0))

import torchvision
print("torchvision:", torchvision.__version__, torchvision.__file__)
from torchvision.ops import nms
print("torchvision nms OK")

import timm
print("timm:", timm.__version__, timm.__file__)

import mmengine
print("mmengine:", mmengine.__version__, mmengine.__file__)

import mmcv
print("mmcv:", mmcv.__version__, mmcv.__file__)
from mmcv.ops import sigmoid_focal_loss, point_sample
print("mmcv ops OK")

import mmseg
print("mmseg:", mmseg.__version__, mmseg.__file__)

import prettytable
print("prettytable:", prettytable.__version__, prettytable.__file__)
PY

CONFIG="configs/ppp/ppp_dinov3_rope_big_neck_40k_dbwa1.py"
WORK_DIR="/e/scratch/nxtaim-1/huber7/work_dirs/${SLURM_JOB_ID}_${SLURM_JOB_NAME}"

mkdir -p "$WORK_DIR"

echo
echo "CONFIG=$CONFIG"
echo "WORK_DIR=$WORK_DIR"

echo
echo "=== Config parse check ==="
python - <<PY
from mmengine.config import Config
cfg = Config.fromfile("$CONFIG")
print("config loaded")
print("model type:", cfg.model.get("type", None))
PY

echo
echo "=== Starting 2-iteration MMSeg training test ==="

PYTHON_BIN="$(command -v python)"
echo "PYTHON_BIN=$PYTHON_BIN"

export SLURM_EXPORT_ENV=ALL

srun --export=ALL \
     --ntasks="$SLURM_NTASKS" \
     --gpus-per-task=1 \
     --cpus-per-task="$SLURM_CPUS_PER_TASK" \
     --gpu-bind=none \
  "$PYTHON_BIN" tools/train.py "$CONFIG" \
    --launcher slurm \
    --work-dir "$WORK_DIR" \
    --cfg-options \
    train_cfg.max_iters=2 \
    val_cfg=None \
    val_dataloader=None \
    val_evaluator=None \
    model.backbone.checkpoint_path="/e/project1/nxtaim-1/huber7/pretrained/vit_small_patch16_dinov3.lvd1689m/model.safetensors"