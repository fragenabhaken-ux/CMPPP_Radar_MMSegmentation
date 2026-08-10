#!/bin/bash
#SBATCH --job-name=build_torchvision
#SBATCH --account=nxtaim-1
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=12
#SBATCH --time=00:05:00
#SBATCH --output=logs/%j_%x.out
#SBATCH --error=logs/%j_%x.err
#SBATCH --export=NONE

set -euo pipefail

cd "$SLURM_SUBMIT_DIR/.." || exit 1

mkdir -p jsc_jupiter/logs
mkdir -p jsc_jupiter/tmp

source jsc_jupiter/activate_jupiter.sh

export CUDA_HOME="/e/software/default/stages/2025/software/CUDA/12"
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"

export FORCE_CUDA=1
export MAX_JOBS="${SLURM_CPUS_PER_TASK}"
export TORCH_CUDA_ARCH_LIST="9.0"
export BUILD_VERSION="0.20.1"
export TMPDIR="$PWD/jsc_jupiter/tmp"

TV_SRC="$PWD/jsc_jupiter/packages/torchvision-0.20.1.tar.gz"

echo "=== Job info ==="
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-unset}"
echo "SLURM_NODELIST=${SLURM_JOB_NODELIST:-unset}"
echo "PWD=$PWD"
echo "TV_SRC=$TV_SRC"

echo
echo "=== Environment ==="
module list
which python
python -V
which nvcc
nvcc --version

echo
echo "=== Torch check ==="
python - <<'PY'
import torch
from torch.utils.cpp_extension import CUDA_HOME

print("torch:", torch.__version__, torch.version.cuda, torch.__file__)
print("CUDA_HOME from torch:", CUDA_HOME)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
    print("capability:", torch.cuda.get_device_capability(0))
PY

echo
echo "=== Source tarball check ==="
if [[ ! -f "$TV_SRC" ]]; then
    echo "ERROR: torchvision source tarball not found:"
    echo "$TV_SRC"
    exit 1
fi
ls -lh "$TV_SRC"

echo
echo "=== Remove old torchvision ==="
python -m pip uninstall -y torchvision || true

echo
echo "=== Build torchvision from source, offline ==="
python -m pip install \
    -v \
    --no-index \
    --no-build-isolation \
    --no-deps \
    "$TV_SRC"

echo
echo "=== Verify torchvision ==="
python -X faulthandler - <<'PY'
import torch
print("torch:", torch.__version__, torch.version.cuda)

import torchvision
print("torchvision:", torchvision.__version__, torchvision.__file__)

from torchvision.ops import nms
print("torchvision.ops.nms OK:", nms)
PY

echo
echo "torchvision build finished."