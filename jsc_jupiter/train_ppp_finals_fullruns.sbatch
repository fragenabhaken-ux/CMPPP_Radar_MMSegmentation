#!/bin/bash
#SBATCH --job-name=ppp_finals_TrafoEncDec_run2_20k_woVel
#SBATCH --account=nxtaim-1
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=12
#SBATCH --time=12:00:00
#SBATCH --output=logs/%j_%x.out
#SBATCH --error=logs/%j_%x.err
#SBATCH --export=NONE

set -euo pipefail

# Submit this script from the jsc_jupiter folder.
cd "$SLURM_SUBMIT_DIR/.." || exit 1

source jsc_jupiter/activate_jupiter.sh

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

# CONFIG="configs/ppp_finals_UNet/ppp_finals_unet_54k.py"
# CONFIG="configs/ppp_finals_UNet/ppp_finals_unet_54k_woVel.py"
# CONFIG="configs/ppp_finals_UNet/ppp_finals_unet_54k_woVel_woRCS.py"
# CONFIG="configs/ppp_finals_UperNet/ppp_finals_upernet_47k.py"
# CONFIG="configs/ppp_finals_TrafoEnc_CamDom/ppp_finals_TrafoEnc_CamDom_24k.py"
# CONFIG="configs/ppp_finals_TrafoEnc_BEVpolDom/ppp_finals_TrafoEnc_BEVpolDom_20k.py"
# CONFIG="configs/ppp_finals_TrafoEncDec/ppp_finals_TrafoEncDec_22k.py"
# CONFIG="configs/ppp_finals_TrafoEncDec/ppp_finals_TrafoEncDec_22k_woVel.py"
# CONFIG="configs/ppp_finals_TrafoEncDec/ppp_finals_TrafoEncDec_22k_woVel_woRCS.py"
# CONFIG="configs/ppp_finals_TrafoEncDec/ppp_finals_TrafoEncDec_run2_20k.py"
# CONFIG="configs/ppp_finals_UperNet/ppp_finals_upernet_run2_21k.py"
# CONFIG="configs/ppp_finals_UperNet/ppp_finals_upernet_run3_50k.py"
CONFIG="configs/ppp_finals_TrafoEncDec/ppp_finals_TrafoEncDec_run2_20k_woVel.py"

WORK_DIR="/e/scratch/nxtaim-1/huber7/work_dirs/${SLURM_JOB_ID}_${SLURM_JOB_NAME}"

mkdir -p "$WORK_DIR"

echo
echo "CONFIG=$CONFIG"
echo "WORK_DIR=$WORK_DIR"

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
    --work-dir "$WORK_DIR"