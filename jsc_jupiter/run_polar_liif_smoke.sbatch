#!/bin/bash
#SBATCH --job-name=polar_liif_smoke
#SBATCH --account=nxtaim-1
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=12
#SBATCH --time=00:10:00
#SBATCH --output=/e/project1/nxtaim-1/huber7/repos/mmsegmentation_ppp/jsc_jupiter/logs/%j_%x.out
#SBATCH --error=/e/project1/nxtaim-1/huber7/repos/mmsegmentation_ppp/jsc_jupiter/logs/%j_%x.err
#SBATCH --export=ALL

set -euo pipefail

REPO_DIR="/e/project1/nxtaim-1/huber7/repos/mmsegmentation_ppp"
cd "$REPO_DIR"

source jsc_jupiter/activate_jupiter.sh

export PYTHONUNBUFFERED=1

CONFIG="configs/ppp_jupiter_dinov3_LoRA_polar_liif/ppp_dinov3_LoRA_V3_polar_loader_test.py"

python -u tools/train.py "$CONFIG"