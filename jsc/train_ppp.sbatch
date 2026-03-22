#!/bin/bash
#SBATCH --job-name=LRRangeTest
#SBATCH --account=nxtaim-1
#SBATCH --partition=develbooster
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=12
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
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

# CONFIG=configs/ppp/ppp_resnet34_256x512_baseline.py
# CONFIG=configs/ppp/ppp_resnet34_256x512_test1500Iter_GrClOff.py
# CONFIG=configs/ppp/ppp_resnet34_256x512_test1500Iter_GrCl1.py
# CONFIG=configs/ppp/ppp_resnet34_256x512_test1500Iter_GrCl10.py
# CONFIG=configs/ppp/ppp_resnet34_256x512_test1500Iter_GrCl1_LR0001.py
# CONFIG=configs/ppp/ppp_resnet34_256x512_test1500Iter_GrCl1_LR0005.py
# CONFIG=configs/ppp/ppp_resnet34_256x512_test1500Iter_GrCl1_LR0010.py
# CONFIG=configs/ppp/ppp_resnet34_256x512_test1500Iter_GrCl1_LR0050.py
# CONFIG=configs/ppp/ppp_resnet34_256x512_test1500Iter_GrCl1_LR0100.py
# CONFIG=configs/ppp/ppp_resnet34_256x512_test1500Iter_SGD_GrClOff_LR0010.py
# CONFIG=configs/ppp/ppp_resnet34_256x512_test1500Iter_GrCl1_LRRangeTest.py
# CONFIG=configs/ppp/ppp_resnet34_256x512_test1500Iter_SGD_GrClOff_LRRangeTest.py
CONFIG=configs/ppp/ppp_resnet34_256x512_test5000Iter_GrCl1_LRRangeTest.py

WORK_DIR="/p/scratch/nxtaim-1/huber7/work_dirs/${SLURM_JOB_ID}_${SLURM_JOB_NAME}"
mkdir -p "$WORK_DIR"

srun --ntasks=4 --gpus-per-task=1 --cpus-per-task="$SLURM_CPUS_PER_TASK" \
     --gpu-bind=none \
  python tools/train.py $CONFIG --launcher slurm --work-dir "$WORK_DIR" \
  --cfg-options load_from=/p/scratch/nxtaim-1/huber7/work_dirs/mmsegmentation_ppp/iter_40000.pth