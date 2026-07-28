#!/bin/bash
#SBATCH --job-name=ppp_finals_TrafoEncDec_run2_20k_last_demo_vid
#SBATCH --account=nxtaim-1
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=12
#SBATCH --time=00:10:00
#SBATCH --output=logs/%j_%x.out
#SBATCH --error=logs/%j_%x.err
#SBATCH --export=NONE

set -euo pipefail

# Submit this script from the jsc_jupiter folder.
cd "$SLURM_SUBMIT_DIR/.." || exit 1

source jsc_jupiter/activate_jupiter.sh

WORK_DIR="/e/scratch/nxtaim-1/huber7/work_dirs/${SLURM_JOB_ID}_${SLURM_JOB_NAME}"

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

# CONFIG=configs/ppp_jupiter_unet/ppp_unet_50k_demo_vid.py
# CONFIG=configs/ppp_jupiter_unet/ppp_unet_50k_demo_vid_BEV.py
# CHECKPOINT=/e/scratch/nxtaim-1/huber7/data/260615_unet_50k/858912_ppp_unet_50k/iter_50000.pth

# CONFIG=configs/ppp_jupiter_dinov3_LoRA/ppp_dinov3_LoRA_V3_25k_demo_vid.py
# CONFIG=configs/ppp_jupiter_dinov3_LoRA/ppp_dinov3_LoRA_V3_25k_demo_vid_BEV.py
# CHECKPOINT=/e/scratch/nxtaim-1/huber7/work_dirs/889927_ppp_dinov3_LoRA_V3_25k/best_ppp_optimal_score_iter_24000.pth

# CONFIG=configs/ppp_jupiter_dinov3_LoRA/ppp_dinov3_LoRA_V3_disparity_25k_demo_vid.py
# CONFIG=configs/ppp_jupiter_dinov3_LoRA/ppp_dinov3_LoRA_V3_disparity_25k_demo_vid_BEV.py
# CHECKPOINT=/e/scratch/nxtaim-1/huber7/data/260617_dinov3_LoRA_V3/906347_ppp_dinov3_LoRA_V3_disparity_25k/best_ppp_optimal_score_iter_25000.pth

# CONFIG=configs/ppp_jupiter_dinov3_LoRA/ppp_dinov3_LoRA_V3_fullMarkAbsVel_25k_demo_vid.py
# CHECKPOINT=/e/scratch/nxtaim-1/huber7/work_dirs/946592_ppp_dinov3_LoRA_V3_fullMarkAbsVel_25k/iter_23000.pth

# CONFIG=configs/ppp_finals_UNet/ppp_finals_unet_54k_demo_vid_CamDom.py
# CONFIG=configs/ppp_finals_UNet/ppp_finals_unet_54k_demo_vid_BEVDom.py
# CHECKPOINT=/e/scratch/nxtaim-1/huber7/data/260725_finals/run_1/1041277_ppp_finals_unet_54k/best_ppp_optimal_score_iter_52000.pth

# CONFIG=configs/ppp_finals_TrafoEncDec/ppp_finals_TrafoEncDec_demovid.py
# CHECKPOINT=/e/scratch/nxtaim-1/huber7/data/260725_finals/run_1/1041329_ppp_finals_TrafoEncDec_22k/best_ppp_optimal_score_iter_8000.pth

# CONFIG=configs/ppp_finals_TrafoEnc_BEVpolDom/ppp_finals_TrafoEnc_BEVpolDom_20k_demo_vid.py
# CHECKPOINT=/e/scratch/nxtaim-1/huber7/data/260725_finals/run_1/1041092_ppp_finals_TrafoEnc_BEVpolDom_20k/best_ppp_optimal_score_iter_19000.pth

# CONFIG=configs/ppp_finals_TrafoEnc_CamDom/ppp_finals_TrafoEnc_CamDom_24k_demo_vid_BEVDom.py
# CONFIG=configs/ppp_finals_TrafoEnc_CamDom/ppp_finals_TrafoEnc_CamDom_24k_demo_vid_CamDom.py
# CHECKPOINT=/e/scratch/nxtaim-1/huber7/data/260725_finals/run_1/1041091_ppp_finals_TrafoEnc_CamDom_24k/iter_24000.pth

CONFIG=configs/ppp_finals_TrafoEncDec/ppp_finals_TrafoEncDec_run2_20k_demo_vid.py
# CHECKPOINT=/e/scratch/nxtaim-1/huber7/work_dirs/1065402_ppp_finals_TrafoEncDec_run2_20k/best_ppp_optimal_score_iter_19000.pth
CHECKPOINT=/e/scratch/nxtaim-1/huber7/work_dirs/1065402_ppp_finals_TrafoEncDec_run2_20k/iter_20000.pth

# CONFIG=configs/ppp_finals_UperNet/ppp_finals_upernet_run2_21k_demo_vid_CamDom.py
# CONFIG=configs/ppp_finals_UperNet/ppp_finals_upernet_run2_21k_demo_vid_BEVDom.py
# CHECKPOINT=/e/scratch/nxtaim-1/huber7/work_dirs/1069825_ppp_finals_upernet_run2_21k/iter_21000.pth

mkdir -p "$WORK_DIR"

echo
echo "CONFIG=$CONFIG"
echo "WORK_DIR=$WORK_DIR"

PYTHON_BIN="$(command -v python)"
echo "PYTHON_BIN=$PYTHON_BIN"

export SLURM_EXPORT_ENV=ALL

srun --ntasks=4 --gpus-per-task=1 --cpus-per-task="$SLURM_CPUS_PER_TASK" \
     --gpu-bind=none \
  python tools/test.py $CONFIG $CHECKPOINT --launcher slurm --work-dir "$WORK_DIR"