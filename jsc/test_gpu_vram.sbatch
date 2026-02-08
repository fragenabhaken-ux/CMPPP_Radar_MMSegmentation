#!/bin/bash
#SBATCH --job-name=ppp_setup
#SBATCH --account=nxtaim-1
#SBATCH --partition=develbooster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=48
#SBATCH --time=00:05:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

cd "$SLURM_SUBMIT_DIR/.." || exit 1

source jsc/setup.sh

echo " -- before --"
python test_gpu_vram.py
echo " -- after --"