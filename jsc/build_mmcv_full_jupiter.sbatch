#!/bin/bash
#SBATCH --job-name=build_mmcv_full
#SBATCH --account=nxtaim-1
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=12
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --export=NONE

set -euo pipefail

cd "$SLURM_SUBMIT_DIR/.." || exit 1

mkdir -p jsc/logs
mkdir -p jsc/tmp

source jsc/activate_jupiter.sh

export CUDA_HOME="/e/software/default/stages/2025/software/CUDA/12"
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"

export MMCV_WITH_OPS=1
export FORCE_CUDA=1
export MAX_JOBS="${SLURM_CPUS_PER_TASK}"
export TORCH_CUDA_ARCH_LIST="9.0"

export TMPDIR="$PWD/jsc/tmp"

MMCV_VERSION="2.1.0"
MMCV_SRC="$PWD/jsc/packages/mmcv-${MMCV_VERSION}.tar.gz"

echo "=== Job info ==="
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-unset}"
echo "SLURM_JOB_NAME=${SLURM_JOB_NAME:-unset}"
echo "SLURM_NODELIST=${SLURM_JOB_NODELIST:-unset}"
echo "SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-unset}"
echo "PWD=$PWD"
echo "TMPDIR=$TMPDIR"
echo "MMCV_SRC=$MMCV_SRC"

echo
echo "=== Environment ==="
module list
which python
python -V
python -m pip --version

echo
echo "=== CUDA compiler ==="
which nvcc
nvcc --version
echo "CUDA_HOME=$CUDA_HOME"

echo
echo "=== Torch check ==="
python - <<'PY'
import torch
import platform
from torch.utils.cpp_extension import CUDA_HOME

print("machine:", platform.machine())
print("torch:", torch.__version__)
print("torch file:", torch.__file__)
print("torch cuda:", torch.version.cuda)
print("CUDA_HOME from torch:", CUDA_HOME)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
    print("capability:", torch.cuda.get_device_capability(0))
PY

echo
echo "=== Required build helpers check, offline ==="
python - <<'PY'
mods = ["setuptools", "wheel", "ninja", "packaging", "psutil", "Cython"]
for m in mods:
    try:
        mod = __import__(m)
        print(f"{m:12s} OK   {getattr(mod, '__version__', 'no __version__')}   {getattr(mod, '__file__', '')}")
    except Exception as e:
        print(f"{m:12s} FAIL {type(e).__name__}: {e}")
        raise SystemExit(1)
PY

echo
echo "=== Existing package check before MMCV ==="
python - <<'PY'
mods = ["numpy", "cv2", "torch", "mmengine"]
for m in mods:
    try:
        mod = __import__(m)
        print(f"{m:12s} OK   {getattr(mod, '__version__', 'no __version__')}   {getattr(mod, '__file__', '')}")
    except Exception as e:
        print(f"{m:12s} FAIL {type(e).__name__}: {e}")
PY

echo
echo "=== Source tarball check ==="
if [[ ! -f "$MMCV_SRC" ]]; then
    echo "ERROR: MMCV source tarball not found:"
    echo "$MMCV_SRC"
    echo "Download it on the login node first."
    exit 1
fi

ls -lh "$MMCV_SRC"

echo
echo "=== Remove old MMCV packages ==="
python -m pip uninstall -y mmcv-lite mmcv || true

echo
echo "=== Build full MMCV from local source tarball, offline ==="
echo "MMCV_WITH_OPS=$MMCV_WITH_OPS"
echo "FORCE_CUDA=$FORCE_CUDA"
echo "MAX_JOBS=$MAX_JOBS"
echo "TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST"

python -m pip install \
    -v \
    --no-index \
    --no-build-isolation \
    --no-deps \
    "$MMCV_SRC"

echo
echo "=== Verify MMCV ops ==="
python - <<'PY'
import mmcv
print("mmcv:", mmcv.__version__, mmcv.__file__)

import mmcv.ops
print("mmcv.ops imported:", mmcv.ops)

from mmcv.ops import sigmoid_focal_loss, point_sample
print("sigmoid_focal_loss OK:", sigmoid_focal_loss)
print("point_sample OK:", point_sample)
PY

echo
echo "Full MMCV build finished."