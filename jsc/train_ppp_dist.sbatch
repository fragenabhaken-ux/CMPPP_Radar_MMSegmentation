#!/bin/bash
#SBATCH --job-name=ppp_train
#SBATCH --account=nxtaim-1
#SBATCH --partition=develbooster
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=12
#SBATCH --time=00:15:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --export=ALL

set -euo pipefail

cd "$SLURM_SUBMIT_DIR/.." || exit 1

# Note: run `source jsc/setup.sh` once beforehand
source jsc/modules.sh
source jsc/activate.sh

# IMPORTANT:
# Do NOT unset or hard-set CUDA_VISIBLE_DEVICES.
# Let Slurm manage it (or override via --gpu-bind below as needed).
# unset CUDA_VISIBLE_DEVICES   # <-- remove this

export SRUN_CPUS_PER_TASK="${SLURM_CPUS_PER_TASK}"

# MASTER_ADDR (IPv4) + stable port (helps avoid IPv6 "address family" issues)
MASTER_HOST="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
export MASTER_ADDR="$(getent ahostsv4 "$MASTER_HOST" | awk 'NR==1{print $1}')"
export MASTER_PORT=7010

echo "MASTER_HOST=$MASTER_HOST"
echo "MASTER_ADDR=$MASTER_ADDR"
echo "MASTER_PORT=$MASTER_PORT"
echo "SLURM_NTASKS=${SLURM_NTASKS:-unset} SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-unset}"

###############################################################################
# SANITY CHECK
# We want to detect the exact condition that caused your crash:
# - If Slurm binds 1 GPU per task (CUDA_VISIBLE_DEVICES is a single id),
#   then torch.cuda.device_count() == 1.
# - MMEngine then *must* not call set_device(local_rank>0) with only 1 visible GPU.
#   Our fix below uses --gpu-bind=none so each task sees 4 GPUs and local_rank works.
###############################################################################
srun --ntasks=4 bash -lc '
python - << "PY"
import os, torch
rank=int(os.environ["SLURM_PROCID"])
local=int(os.environ["SLURM_LOCALID"])
cvis=os.environ.get("CUDA_VISIBLE_DEVICES","<unset>")

torch.cuda.init()
n = torch.cuda.device_count()

# If Slurm bound us to 1 GPU, n==1 and the only valid index is 0.
# If not bound, we choose local rank as device index.
dev = 0 if n==1 else local
torch.cuda.set_device(dev)

print(f"rank={rank} localid={local} cvis={cvis} "
      f"device_count={n} set_device={dev} current={torch.cuda.current_device()} "
      f"name={torch.cuda.get_device_name(torch.cuda.current_device())}")
PY'

###############################################################################
# TRAINING
# Fix for: RuntimeError: CUDA error: invalid device ordinal
#
# Reason: MMEngine's Slurm init uses local_rank (=SLURM_LOCALID) and calls
# torch.cuda.set_device(local_rank). This only works if each process sees all GPUs.
# Therefore: --gpu-bind=none (so device indices 0..3 are valid).
###############################################################################
echo "=== TRAINING ==="
srun --ntasks=4 --gpus-per-task=1 --cpus-per-task="$SLURM_CPUS_PER_TASK" \
     --gpu-bind=none \
  python tools/train.py configs/mytest/fast_scnn_mppp_256x512_juwels.py --launcher slurm