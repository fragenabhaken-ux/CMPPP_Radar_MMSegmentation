#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

source "$SCRIPT_DIR/config_jupiter.sh"
source "$SCRIPT_DIR/modules_jupiter.sh"

source "$ENV_DIR/bin/activate"

export PYTHONNOUSERSITE=1

if [[ -n "${PYTHONPATH:-}" ]]; then
  export PYTHONPATH="${REPO_DIR}:${PYTHONPATH}"
else
  export PYTHONPATH="${REPO_DIR}"
fi

export CUDA_HOME="/e/software/default/stages/2025/software/CUDA/12"
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"

echo "Activated $ENV_NAME at $ENV_DIR"
which python
python -V