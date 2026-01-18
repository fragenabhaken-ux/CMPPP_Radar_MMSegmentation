#!/bin/bash
#SBATCH --job-name=ppp_train
#SBATCH --account=nxtaim-1
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

cd $SLURM_SUBMIT_DIR
source activate.sh
python tools/train.py configs/mytest/fast_scnn_mppp_256x512_juwels.py
