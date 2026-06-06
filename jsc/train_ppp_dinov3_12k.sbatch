#!/bin/bash
#SBATCH --job-name=ppp_dinov3_LoRA_rope_big_neck_noCrop_V2_12k
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

# CONFIG=configs/ppp/ppp_dinov3_40k_dbwa1.py
# CONFIG=configs/ppp/ppp_dinov3_40k_dbwa2.py
# CONFIG=configs/ppp/ppp_dinov3_40k_dbwa3.py
# CONFIG=configs/ppp/ppp_dinov3_40k_dbwa4.py
# CONFIG=configs/ppp/ppp_dinov3_40k_dbwa5.py
# CONFIG=configs/ppp/ppp_dinov3_rope_big_neck_14k_depthLossExp_Vwdepth.py
# CONFIG=configs/ppp/ppp_dinov3_rope_big_neck_14k_depthLossExp_Vwodepth.py
# CONFIG=configs/ppp/ppp_dinov3_rope_big_neck_7k_depthLoss_multitask_head.py
# CONFIG=configs/ppp/ppp_dinov3_rope_big_neck_7k_depthLoss_multitask_two_depth_head.py
# CONFIG=configs/ppp/ppp_dinov3_LoRA_x2cap_rope_big_neck_7k_testrun_S1.py
# CONFIG=configs/ppp/ppp_dinov3_mid_LoRA_rope_big_neck_7k_testrun_S1.py

WORK_DIR="/p/scratch/nxtaim-1/huber7/work_dirs/${SLURM_JOB_ID}_${SLURM_JOB_NAME}"
# CHECKPOINT="/p/scratch/nxtaim-1/huber7/work_dirs/13882410_dinov3_40k_dbwa1/iter_7000.pth"
# CHECKPOINT="/p/scratch/nxtaim-1/huber7/work_dirs/13882688_dinov3_40k_dbwa2/iter_7000.pth"
# CHECKPOINT="/p/scratch/nxtaim-1/huber7/work_dirs/13882814_dinov3_40k_dbwa3/iter_7000.pth"
# CHECKPOINT="/p/scratch/nxtaim-1/huber7/work_dirs/13883022_dinov3_40k_dbwa4/iter_6000.pth"

# CONFIG=configs/ppp/ppp_dinov3_LoRA_x2cap_rope_big_neck_7k_testrun_S2.py
# CHECKPOINT="/p/scratch/nxtaim-1/huber7/data1/260529_dinov3_LoRA/13904625_ppp_dinov3_LoRA_x2cap_rope_big_neck_7k_testrun_S1/iter_6000.pth"

# CONFIG=configs/ppp/ppp_dinov3_mid_LoRA_rope_big_neck_7k_testrun_S2.py
# CHECKPOINT="/p/scratch/nxtaim-1/huber7/data1/260529_dinov3_LoRA/13904628_ppp_dinov3_mid_LoRA_rope_big_neck_7k_testrun_S1/iter_6000.pth"

##################################

# CONFIG=configs/ppp/ppp_dinov3_rope_big_neck_noCrop_12k_P1.py
# CONFIG=configs/ppp/ppp_dinov3_rope_big_neck_noCrop_12k_P2.py
# CHECKPOINT="/p/scratch/nxtaim-1/huber7/data1/260601_dinov3_noCrop/13909632_ppp_dinov3_rope_big_neck_noCrop_12k_P1/iter_6000.pth"

# CONFIG=configs/ppp/ppp_dinov3_rope_big_neck_depthLoss_multitask_head_noCrop_12k_P1.py
# CONFIG=configs/ppp/ppp_dinov3_rope_big_neck_depthLoss_multitask_head_noCrop_12k_P2.py
# CHECKPOINT="/p/scratch/nxtaim-1/huber7/data1/260601_dinov3_noCrop/13914010_ppp_dinov3_rope_big_neck_depthLoss_multitask_head_noCrop_12k_P1/iter_6000.pth"

# CONFIG=configs/ppp/ppp_dinov3_rope_big_neck_depthLoss_multitask_two_depth_head_noCrop_12k_P1.py
# CONFIG=configs/ppp/ppp_dinov3_rope_big_neck_depthLoss_multitask_two_depth_head_noCrop_12k_P2.py
# CHECKPOINT="/p/scratch/nxtaim-1/huber7/data1/260601_dinov3_noCrop/13917929_ppp_dinov3_rope_big_neck_depthLoss_multitask_two_depth_head_noCrop_12k_P1/iter_6000.pth"

# CONFIG=configs/ppp/ppp_dinov3_LoRA_rope_big_neck_noCrop_12k_P1.py
# CONFIG=configs/ppp/ppp_dinov3_LoRA_rope_big_neck_noCrop_12k_P2.py
# CHECKPOINT="/p/scratch/nxtaim-1/huber7/data1/260601_dinov3_noCrop/13917932_ppp_dinov3_LoRA_rope_big_neck_noCrop_12k_P1/iter_5000.pth"

##################################

# CONFIG=configs/ppp/ppp_dinov3_mid_LoRA_rope_big_neck_noCrop_12k_P1.py
# CONFIG=configs/ppp/ppp_dinov3_mid_LoRA_rope_big_neck_noCrop_12k_P2.py
# CHECKPOINT="/p/scratch/nxtaim-1/huber7/data1/260601_dinov3_noCrop/13919690_ppp_dinov3_mid_LoRA_rope_big_neck_noCrop_12k_P1/iter_5000.pth"

# CONFIG=configs/ppp/ppp_dinov3_big_LoRA_rope_big_neck_noCrop_12k_P1.py
# CONFIG=configs/ppp/ppp_dinov3_big_LoRA_rope_big_neck_noCrop_12k_P2.py
# CHECKPOINT="/p/scratch/nxtaim-1/huber7/data1/260601_dinov3_noCrop/13919691_ppp_dinov3_big_LoRA_rope_big_neck_noCrop_12k_P1/iter_5000.pth"

#################################

CONFIG=configs/ppp/ppp_dinov3_LoRA_rope_big_neck_noCrop_V2_12k.py
CHECKPOINT="/p/scratch/nxtaim-1/huber7/work_dirs/13942624_ppp_dinov3_LoRA_rope_big_neck_noCrop_V2_12k/iter_9000.pth"

mkdir -p "$WORK_DIR"

echo "CONFIG=$CONFIG"
echo "WORK_DIR=$WORK_DIR"

srun --ntasks="$SLURM_NTASKS" \
     --gpus-per-task=1 \
     --cpus-per-task="$SLURM_CPUS_PER_TASK" \
     --gpu-bind=none \
  python tools/train.py "$CONFIG" \
    --launcher slurm \
    --work-dir "$WORK_DIR" \
    --resume \
    --cfg-options \
    load_from="$CHECKPOINT"