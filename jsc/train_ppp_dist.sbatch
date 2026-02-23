#!/bin/bash
#SBATCH --job-name=ppp_setup
#SBATCH --account=nxtaim-1
#SBATCH --partition=develbooster
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=12
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err


# Without this, srun does not inherit cpus-per-task from sbatch. #128
echo "START SETTING UP SBATCH"
echo "----------------------------------"

# # activating the virtual environment. TODO! needs your local base path
# BASE_PATH="/p/project1/nxtaim-1/neuhoefer1/CleanStableDiffusion"
# source $BASE_PATH/JSC_training/activate.sh

export SRUN_CPUS_PER_TASK="$SLURM_CPUS_PER_TASK"
# so processes know who to talk to
echo "SLURM_JOB_NODELIST: $SLURM_JOB_NODELIST"
MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
# Allow communication over InfiniBand cells.
MASTER_ADDR="${MASTER_ADDR}i"
# Get IP for hostname.
export MASTER_ADDR="$(nslookup "$MASTER_ADDR" | grep -oP '(?<=Address: ).*')"
export MASTER_PORT=7010

echo "MASTER_ADDR:MASTER_PORT=""$MASTER_ADDR":"$MASTER_PORT"
echo "----------------------------------"


cd "$SLURM_SUBMIT_DIR/.." || exit 1

source jsc/setup.sh

srun python tools/train.py configs/mytest/fast_scnn_mppp_256x512_juwels.py --launcher slurm