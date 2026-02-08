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

#set -euo pipefail

cd $SLURM_SUBMIT_DIR
#source activate.sh

MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
MASTER_ADDR="${MASTER_ADDR}i"
export MASTER_ADDR="$(nslookup "$MASTER_ADDR" | grep -oP '(?<=Address: ).*' )"
export MASTER_PORT=7010

echo "MASTER_ADDR:MASTER_PORT=$MASTER_ADDR:$MASTER_PORT"

export CUDA_VISIBLE_DEVICES=0,1,2,3

source setup.sh

python - <<'EOF'
import torch
from pprint import pprint

print("=== Torch ===")
try:
    print("torch version:", torch.__version__)
    print("torch.version.cuda:", torch.version.cuda)
    print("CUDA available:", torch.cuda.is_available())
except Exception as e:
    print("Torch check FAILED:", repr(e))

print("\n=== MMENGINE ===")
try:
    import mmengine
    print("mmengine.__version__:", mmengine.__version__)
except Exception as e:
    print("MMENGINE import FAILED:", repr(e))
    mmengine = None

print("\n=== MMCV ===")
try:
    import mmcv
    print("mmcv.__version__:", mmcv.__version__)
except Exception as e:
    print("MMCV import FAILED:", repr(e))
    mmcv = None

print("\n=== MMCV dir ===")
if mmcv is not None:
    try:
        pprint(list(dir(mmcv)))
    except Exception as e:
        print("MMCV dir FAILED:", repr(e))

print("\n=== MMCV _ext ===")
if mmcv is not None:
    try:
        import mmcv._ext
        print("MMCV _ext file:", mmcv._ext.__file__)
    except Exception as e:
        print("MMCV _ext import FAILED:", repr(e))

print("\n=== MMCV CUDA info ===")
if mmcv is not None:
    try:
        cuda_ver = mmcv.ops.get_compiling_cuda_version()
        print("MMCV CUDA compiling version:", cuda_ver)
    except Exception as e:
        print("MMCV CUDA version query FAILED:", repr(e))

    try:
        archs = mmcv.ops.get_compiling_cuda_arch()
        print("MMCV CUDA archs:", archs)
    except Exception as e:
        print("MMCV CUDA arch query FAILED:", repr(e))

print("\n=== MMSEG ===")
try:
    import mmseg
    print("mmseg.__version__:", mmseg.__version__)
except Exception as e:
    print("MMSEG import FAILED:", repr(e))
    mmseg = None

try:
    from mmseg.apis import init_model
    print("init_model import OK")
except Exception as e:
    import traceback
    print("init_model import FAILED")
    traceback.print_exc()
EOF