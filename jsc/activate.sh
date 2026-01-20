# #### ALT ###

# #!/bin/bash

# # See https://stackoverflow.com/a/28336473
# SOURCE_PATH="${BASH_SOURCE[0]:-${(%):-%x}}"

# RELATIVE_PATH="$(dirname "$SOURCE_PATH")"
# ABSOLUTE_PATH="$(realpath "${RELATIVE_PATH}")"

# [[ "$0" != "${SOURCE_PATH}" ]] && echo "The activation script must be sourced, otherwise the virtual environment will not work." || ( echo "Vars script must be sourced." && exit 1) ;

# source "${ABSOLUTE_PATH}"/config.sh
# source "${ABSOLUTE_PATH}"/modules.sh

# export PYTHONPATH="$(echo "${ENV_DIR}"/lib/python*/site-packages):${PYTHONPATH}"

# source "${ENV_DIR}"/bin/activate

### NEU ###

#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

source "$SCRIPT_DIR/config.sh"
source "$SCRIPT_DIR/modules.sh"

echo $ENV_DIR
source "$ENV_DIR/bin/activate"
export PYTHONNOUSERSITE=1
export PYTHONPATH="/p/project1/nxtaim-1/huber7/repos/mmsegmentation_ppp:${PYTHONPATH}"

echo "Activated $ENV_NAME at $ENV_DIR" # Frage: Ist wirklich der venv-Python aktiv
python -V
