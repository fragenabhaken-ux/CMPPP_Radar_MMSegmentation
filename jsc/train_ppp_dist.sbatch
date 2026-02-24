#!/bin/bash
#SBATCH --job-name=ppp_train
#SBATCH --account=nxtaim-1
#SBATCH --partition=develbooster
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=12
#SBATCH --time=00:10:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

cd "$SLURM_SUBMIT_DIR/.." || exit 1
# Note: Run source jsc/setup.sh before!
source jsc/activate.sh

# wichtig: NICHT CUDA_VISIBLE_DEVICES setzen!
unset CUDA_VISIBLE_DEVICES

export SRUN_CPUS_PER_TASK="$SLURM_CPUS_PER_TASK"

# MASTER_ADDR sauber (ohne "i") + IPv4 erzwingen
MASTER_HOST=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$(getent ahostsv4 "$MASTER_HOST" | awk 'NR==1{print $1}')
export MASTER_PORT=7010

export CUDA_VISIBLE_DEVICES=0,1,2,3

srun --ntasks=4 bash -lc 'echo rank=$SLURM_PROCID localid=$SLURM_LOCALID cuda=$CUDA_VISIBLE_DEVICES; python -c "import torch; print(torch.cuda.device_count())"'

srun --cpu-bind=none --gpu-bind=none \
  python tools/train.py configs/mytest/fast_scnn_mppp_256x512_juwels.py --launcher slurm