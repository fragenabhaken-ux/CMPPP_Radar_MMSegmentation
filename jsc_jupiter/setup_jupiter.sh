#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

source "$SCRIPT_DIR/config_jupiter.sh"
source "$SCRIPT_DIR/modules_jupiter.sh"

echo "Removing old env if it exists:"
echo "$ENV_DIR"
rm -rf "$ENV_DIR"

python -m venv --prompt "$ENV_NAME" --system-site-packages "$ENV_DIR"
source "$ENV_DIR/bin/activate"

export PYTHONNOUSERSITE=1
export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"

export CUDA_HOME="/e/software/default/stages/2025/software/CUDA/12"
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"

# Upgrade pip first. setuptools/wheel are pinned again after all base
# dependencies, because packages such as openxlab can otherwise downgrade
# setuptools to a Python-3.12-incompatible release.
python -m pip install --upgrade pip
python -m pip install "setuptools==69.5.1" "wheel==0.43.0"

# OpenMMLab base tooling and Python dependencies.
# requirements_jupiter_base.txt should NOT contain:
#   torch
#   torchvision
#   timm
#   mmcv
#   mmcv-lite
#   opencv-python
#   opencv-python-headless
#   numpy
#   openmim  (not needed here; it pulls openxlab, which pins old setuptools)
python -m pip install -r "$SCRIPT_DIR/requirements_jupiter_base.txt"

# Extra dependency needed by MMSeg metrics/depth_metric.py.
python -m pip install prettytable

# Important:
# Install OpenCV headless without dependencies, so pip does not pull
# opencv-python or NumPy 2.x. NumPy 1.26.4 comes from the JSC SciPy-bundle.
python -m pip uninstall -y opencv-python opencv-python-headless || true
python -m pip install --no-deps "opencv-python-headless==4.10.0.84"

# Re-assert Python 3.12-compatible build tooling after all base dependencies.
# This must remain after the requirements installation so no transitive
# dependency can leave setuptools at the incompatible 60.2.0 release.
python -m pip install \
    --upgrade \
    --force-reinstall \
    "setuptools==69.5.1" \
    "wheel==0.43.0" \
    "Cython<3"

# Verify the exact import path used by PyTorch C++/CUDA extensions.
python - <<'PY'
import setuptools
import pkg_resources
from torch.utils.cpp_extension import CUDA_HOME

print("setuptools:", setuptools.__version__)
print("pkg_resources: OK")
print("torch cpp_extension: OK")
print("CUDA_HOME:", CUDA_HOME)
PY

# Do NOT install torchvision here.
# The pip torchvision wheel caused:
#   RuntimeError: operator torchvision::nms does not exist
# Therefore torchvision must be built from source in:
#   build_torchvision_jupiter.sbatch

# Do NOT install timm here yet.
# Install timm after torchvision is built successfully:
#   python -m pip install --no-deps "timm>=1.0.20"

# Do NOT install mmcv or mmcv-lite here.
# Full MMCV with ops must be built from source in:
#   build_mmcv_full_jupiter.sbatch

# Prepare local package directory.
mkdir -p "$SCRIPT_DIR/packages"

# Download MMCV source on the login node if not already present.
# Compute nodes do not have internet access.
if [[ ! -f "$SCRIPT_DIR/packages/mmcv-2.1.0.tar.gz" ]]; then
  echo "Downloading mmcv-2.1.0 source tarball..."
  curl -L --retry 10 --connect-timeout 20 \
    -o "$SCRIPT_DIR/packages/mmcv-2.1.0.tar.gz" \
    "https://files.pythonhosted.org/packages/source/m/mmcv/mmcv-2.1.0.tar.gz"
else
  echo "MMCV source tarball already exists:"
  ls -lh "$SCRIPT_DIR/packages/mmcv-2.1.0.tar.gz"
fi

# Download torchvision source on the login node if not already present.
# The build itself happens later in Slurm.
if [[ ! -f "$SCRIPT_DIR/packages/torchvision-0.20.1.tar.gz" ]]; then
  echo "Downloading torchvision-0.20.1 source tarball..."
  curl -L --retry 10 --connect-timeout 20 \
    -o "$SCRIPT_DIR/packages/torchvision-0.20.1.tar.gz" \
    "https://github.com/pytorch/vision/archive/refs/tags/v0.20.1.tar.gz"
else
  echo "torchvision source tarball already exists:"
  ls -lh "$SCRIPT_DIR/packages/torchvision-0.20.1.tar.gz"
fi

echo
echo "=== Base environment check ==="
python - <<'PY'
import sys
import platform
import numpy
import cv2
import torch
import mmengine
import prettytable

print("python:", sys.version)
print("machine:", platform.machine())
print("numpy:", numpy.__version__, numpy.__file__)
print("cv2:", cv2.__version__, cv2.__file__)
print("torch:", torch.__version__, torch.version.cuda, torch.__file__)
print("mmengine:", mmengine.__version__, mmengine.__file__)
print("prettytable:", prettytable.__version__, prettytable.__file__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
    print("capability:", torch.cuda.get_device_capability(0))
PY

echo
echo "=== Package sanity check ==="
python - <<'PY'
import importlib.metadata as md

for name in [
    "opencv-python",
    "opencv-python-headless",
    "numpy",
    "torchvision",
    "timm",
    "mmcv",
    "mmcv-lite",
    "prettytable",
    "setuptools",
    "wheel",
    "Cython",
    "openmim",
    "openxlab",
]:
    try:
        d = md.distribution(name)
        print(f"{name}: {d.version} at {d.locate_file('')}")
    except md.PackageNotFoundError:
        print(f"{name}: NOT INSTALLED")
PY

echo
echo "Base env ready."
echo "Next steps:"
echo "  1. sbatch build_torchvision_jupiter.sbatch"
echo "  2. source activate_jupiter.sh"
echo "  3. python -m pip install --no-deps 'timm>=1.0.20'"
echo "  4. sbatch build_mmcv_full_jupiter.sbatch"