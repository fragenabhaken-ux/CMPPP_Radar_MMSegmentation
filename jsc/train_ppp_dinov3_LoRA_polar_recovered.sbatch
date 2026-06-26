#!/bin/bash
#SBATCH --job-name=ppp_dinov3_LoRA_V3_polar_liif_12k
#SBATCH --account=nxtaim-1
#SBATCH --partition=develbooster
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=12
#SBATCH --time=02:00:00
#SBATCH --output=logs/%j_%x.out
#SBATCH --error=logs/%j_%x.err
#SBATCH --export=ALL

set -euo pipefail

cd "$SLURM_SUBMIT_DIR/.." || exit 1

source jsc/modules.sh
source jsc/activate.sh

mkdir -p logs

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

# CONFIG="configs/ppp_dinov3_LoRA_recovered/ppp_dinov3_LoRA_V3_polar_liif_smoke_recovered.py"
# CONFIG="configs/ppp_dinov3_LoRA_recovered/ppp_dinov3_LoRA_V3_polar_liif_LRrangetest.py"
CONFIG="configs/ppp_dinov3_LoRA_recovered/ppp_dinov3_LoRA_V3_polar_liif_12k.py"

WORK_DIR="/p/scratch/nxtaim-1/huber7/work_dirs/${SLURM_JOB_ID}_${SLURM_JOB_NAME}"
mkdir -p "$WORK_DIR"

echo "CONFIG=$CONFIG"
echo "WORK_DIR=$WORK_DIR"

srun --ntasks="$SLURM_NTASKS" \
     --gpus-per-task=1 \
     --cpus-per-task="$SLURM_CPUS_PER_TASK" \
     --gpu-bind=none \
  python tools/train.py "$CONFIG" \
    --launcher slurm \
    --work-dir "$WORK_DIR"