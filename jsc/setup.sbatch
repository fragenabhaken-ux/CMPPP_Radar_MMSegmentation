#!/bin/bash
#SBATCH --job-name=ppp_setup
#SBATCH --account=nxtaim-1
#SBATCH --partition=develbooster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=48
#SBATCH --time=00:10:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

#set -euo pipefail

cd $SLURM_SUBMIT_DIR
#source activate.sh

MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
MASTER_ADDR="${MASTER_ADDR}i"
export MASTER_ADDR="$(nslookup "$MASTER_ADDR" | grep -oP '(?<=Address: ).*' )"
export MASTER_PORT=7010

echo "MASTER_ADDR:MASTER_PORT=$MASTER_ADDR:$MASTER_PORT"

export CUDA_VISIBLE_DEVICES=0,1,2,3

source setup.sh

python - <<'EOF'
import torch
print(torch.__version__)
print("torch.version.cuda:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
import mmcv
from pprint import pprint
print("mmcv dir:")
pprint(dir(mmcv))
import mmcv._ext
print("MMCV file:", mmcv._ext.__file__)
print("MMCV CUDA available:", mmcv.ops.get_compiling_cuda_version())
print("MMCV CUDA archs:", mmcv.ops.get_compiling_cuda_arch())
EOF