#!/bin/bash
#SBATCH --job-name=ppp_dinov3_LoRA_rope_big_neck_noCrop_V2_demo_vid_wDepth
#SBATCH --account=nxtaim-1
#SBATCH --partition=develbooster
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=12
#SBATCH --time=00:20:00
#SBATCH --output=logs/%j_%x.out
#SBATCH --error=logs/%j_%x.err
#SBATCH --export=ALL

set -euo pipefail

cd "$SLURM_SUBMIT_DIR/.." || exit 1

# Note: run `source jsc/setup.sh` once beforehand
source jsc/modules.sh
source jsc/activate.sh

# We do not hard-set CUDA_VISIBLE_DEVICES.
# Let Slurm manage it (or override via --gpu-bind below as needed).
export SRUN_CPUS_PER_TASK="${SLURM_CPUS_PER_TASK}"

# MASTER_ADDR (IPv4) + stable port (helps avoid IPv6 "address family" issues)
MASTER_HOST="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
export MASTER_ADDR="$(getent ahostsv4 "$MASTER_HOST" | awk 'NR==1{print $1}')"
export MASTER_PORT=7010

echo "MASTER_HOST=$MASTER_HOST"
echo "MASTER_ADDR=$MASTER_ADDR"
echo "MASTER_PORT=$MASTER_PORT"
echo "SLURM_NTASKS=${SLURM_NTASKS:-unset} SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-unset}"

# CONFIG=configs/ppp/ppp_demo_vid.py
# demo_vid_dinov3__rope_big_neck_40k_last_wdepth / demo_vid_dinov3_40k_last_wdepth

# CONFIG=configs/ppp/ppp_dinov3_40k_demo_vid.py
# CONFIG=configs/ppp/ppp_dinov3_rope_big_neck_40k_demo_vid.py
# CONFIG=configs/ppp/ppp_dinov3_40k_demo_vid_wdepth.py
# CONFIG=configs/ppp/ppp_dinov3_rope_big_neck_40k_demo_vid_wdepth.py
# CONFIG=configs/ppp/ppp_dinov3_rope_big_neck_7k_depthLoss_multitask_head_demovid.py
# CONFIG=configs/ppp/ppp_dinov3_rope_big_neck_7k_depthLoss_multitask_two_depth_head_demovid.py

# CHECKPOINT=/p/scratch/nxtaim-1/huber7/work_dirs/13763261_unet_160k/best_ppp_optimal_score_iter_160000.pth
# CHECKPOINT=/p/scratch/nxtaim-1/huber7/work_dirs/13767623_unet_160k_8nodes/best_ppp_optimal_score_iter_85000.pth
# CHECKPOINT=/p/scratch/nxtaim-1/huber7/work_dirs/13767623_unet_160k_8nodes/iter_135000.pth
# CHECKPOINT=/p/scratch/nxtaim-1/huber7/work_dirs/13767582_resnet34_fcn_160k_8nodes/iter_54000.pth
# CHECKPOINT=/p/scratch/nxtaim-1/huber7/work_dirs/13767582_resnet34_fcn_160k_8nodes/best_ppp_optimal_score_iter_38000.pth

# CHECKPOINT=/p/scratch/nxtaim-1/huber7/work_dirs/13883122_dinov3_40k_dbwa5/best_ppp_optimal_score_iter_4000.pth
# CHECKPOINT=/p/scratch/nxtaim-1/huber7/work_dirs/13883122_dinov3_40k_dbwa5/iter_7000.pth
# CHECKPOINT=/p/scratch/nxtaim-1/huber7/work_dirs/13883121_dinov3__rope_big_neck_40k_dbwa5/best_ppp_optimal_score_iter_5000.pth
# CHECKPOINT=/p/scratch/nxtaim-1/huber7/work_dirs/13883121_dinov3__rope_big_neck_40k_dbwa5/iter_7000.pth

# CHECKPOINT=/p/scratch/nxtaim-1/huber7/work_dirs/13887322_dinov3_rope_big_neck_14k_depthLossExp_Vwdepth/best_ppp_optimal_score_iter_3000.pth
# CHECKPOINT=/p/scratch/nxtaim-1/huber7/work_dirs/13887322_dinov3_rope_big_neck_14k_depthLossExp_Vwdepth/iter_7000.pth
# CHECKPOINT=/p/scratch/nxtaim-1/huber7/work_dirs/13887325_dinov3_rope_big_neck_14k_depthLossExp_Vwodepth/best_ppp_optimal_score_iter_2000.pth
# CHECKPOINT=/p/scratch/nxtaim-1/huber7/work_dirs/13887325_dinov3_rope_big_neck_14k_depthLossExp_Vwodepth/iter_7000.pth

# CHECKPOINT=/p/scratch/nxtaim-1/huber7/work_dirs/13887756_ppp_dinov3_rope_big_neck_7k_depthLoss_multitask_head/best_ppp_optimal_score_iter_7000.pth
# CHECKPOINT=/p/scratch/nxtaim-1/huber7/work_dirs/13887760_ppp_dinov3_rope_big_neck_7k_depthLoss_multitask_two_depth_head/best_ppp_optimal_score_iter_2000.pth
# CHECKPOINT=/p/scratch/nxtaim-1/huber7/work_dirs/13887760_ppp_dinov3_rope_big_neck_7k_depthLoss_multitask_two_depth_head/iter_7000.pth

# CONFIG=configs/ppp/ppp_dinov3_rope_big_neck_noCrop_12k_demo_vid.py
# CHECKPOINT=/p/scratch/nxtaim-1/huber7/data1/260601_dinov3_noCrop/13914026_ppp_dinov3_rope_big_neck_noCrop_12k_P2/best_ppp_optimal_score_iter_5000.pth

# CONFIG=configs/ppp/ppp_dinov3_rope_big_neck_depthLoss_multitask_head_noCrop_demo_vid.py
# CHECKPOINT=/p/scratch/nxtaim-1/huber7/data1/260601_dinov3_noCrop/13914735_ppp_dinov3_rope_big_neck_depthLoss_multitask_head_noCrop_12k_P2/best_ppp_optimal_score_iter_3000.pth

# CONFIG=configs/ppp/ppp_dinov3_rope_big_neck_depthLoss_multitask_two_depth_head_noCrop_demo_vid.py
# CHECKPOINT=/p/scratch/nxtaim-1/huber7/data1/260601_dinov3_noCrop/13918698_ppp_dinov3_rope_big_neck_depthLoss_multitask_two_depth_head_noCrop_12k_P2/best_ppp_optimal_score_iter_4000.pth

# CONFIG=configs/ppp/ppp_dinov3_LoRA_rope_big_neck_noCrop_demo_vid.py
# CHECKPOINT=/p/scratch/nxtaim-1/huber7/data1/260601_dinov3_noCrop/13918853_ppp_dinov3_LoRA_rope_big_neck_noCrop_12k_P2/best_ppp_optimal_score_iter_6000.pth

# CONFIG=configs/ppp/ppp_dinov3_mid_LoRA_rope_big_neck_noCrop_demo_vid.py
# CHECKPOINT=/p/scratch/nxtaim-1/huber7/data1/260601_dinov3_noCrop/13920972_ppp_dinov3_mid_LoRA_rope_big_neck_noCrop_12k_P2/best_ppp_optimal_score_iter_2000.pth

# CONFIG=configs/ppp/ppp_dinov3_big_LoRA_rope_big_neck_noCrop_demo_vid.py
# CHECKPOINT=/p/scratch/nxtaim-1/huber7/data1/260601_dinov3_noCrop/13920974_ppp_dinov3_big_LoRA_rope_big_neck_noCrop_12k_P2/best_ppp_optimal_score_iter_2000.pth

# CONFIG=configs/ppp/ppp_dinov3_rope_big_neck_depthLoss_multitask_head_noCrop_demo_vid_wDepth.py
# CHECKPOINT=/p/scratch/nxtaim-1/huber7/data1/260601_dinov3_noCrop/13914735_ppp_dinov3_rope_big_neck_depthLoss_multitask_head_noCrop_12k_P2/best_ppp_optimal_score_iter_3000.pth

# CONFIG=configs/ppp/ppp_dinov3_LoRA_rope_big_neck_noCrop_demo_vid_wDepth.py
# CHECKPOINT=/p/scratch/nxtaim-1/huber7/data1/260601_dinov3_noCrop/13918853_ppp_dinov3_LoRA_rope_big_neck_noCrop_12k_P2/best_ppp_optimal_score_iter_6000.pth

# CONFIG=configs/ppp/ppp_dinov3_mid_LoRA_rope_big_neck_noCrop_demo_vid_wDepth.py
# CHECKPOINT=/p/scratch/nxtaim-1/huber7/data1/260601_dinov3_noCrop/13920972_ppp_dinov3_mid_LoRA_rope_big_neck_noCrop_12k_P2/best_ppp_optimal_score_iter_2000.pth

# CONFIG=configs/ppp/ppp_dinov3_big_LoRA_rope_big_neck_noCrop_demo_vid_wDepth.py
# CHECKPOINT=/p/scratch/nxtaim-1/huber7/data1/260601_dinov3_noCrop/13920974_ppp_dinov3_big_LoRA_rope_big_neck_noCrop_12k_P2/best_ppp_optimal_score_iter_2000.pth

# CONFIG=configs/ppp/ppp_dinov3_rope_big_neck_noCrop_demo_vid_wDepth.py
# CHECKPOINT=/p/scratch/nxtaim-1/huber7/data1/260601_dinov3_noCrop/13925792_ppp_dinov3_rope_big_neck_noCrop_12k_P2/best_ppp_optimal_score_iter_6000.pth

CONFIG=configs/ppp/ppp_dinov3_LoRA_rope_big_neck_noCrop_V2_demo_vid_wDepth.py
CHECKPOINT=/p/scratch/nxtaim-1/huber7/work_dirs/13943679_ppp_dinov3_LoRA_rope_big_neck_noCrop_V2_12k/iter_12000.pth

WORK_DIR="/p/scratch/nxtaim-1/huber7/work_dirs/${SLURM_JOB_ID}_${SLURM_JOB_NAME}"
mkdir -p "$WORK_DIR"

srun --ntasks=4 --gpus-per-task=1 --cpus-per-task="$SLURM_CPUS_PER_TASK" \
     --gpu-bind=none \
  python tools/test.py $CONFIG $CHECKPOINT --launcher slurm --work-dir "$WORK_DIR"