# ### ALT ###

# SOURCE_PATH="${BASH_SOURCE[0]:-${(%):-%x}}"

# ## Check if this script is sourced
# [[ "$0" != "${SOURCE_PATH}" ]] && echo "Setting vars" || ( echo "Vars script must be sourced." && exit 1) ;
# ## Determine location of this file
# RELATIVE_PATH="$(dirname "$SOURCE_PATH")"
# ABSOLUTE_PATH="$(realpath "${RELATIVE_PATH}")"
# ####################################

# ### User Configuration
# export ENV_NAME="$(basename "$ABSOLUTE_PATH")"             # Default Name of the venv is the directory that contains this file
# export ENV_DIR="${ABSOLUTE_PATH}"/venv         # Default location of this VENV is "./venv"

### NEU ###


SOURCE_PATH="${BASH_SOURCE[0]:-${(%):-%x}}"

# Check if sourced
[[ "$0" != "${SOURCE_PATH}" ]] && echo "Setting vars" || ( echo "Vars script must be sourced." && exit 1)

RELATIVE_PATH="$(dirname "$SOURCE_PATH")"
ABSOLUTE_PATH="$(realpath "${RELATIVE_PATH}")"

# # --- User Configuration (JUWELS) ---
# export ENV_NAME="mmseg"
# export ENV_DIR="/p/project1/nxtaim-1/huber7/env_mmseg"

# ENV_NAME bleibt konstant, aber per Repo überschreibbar
export ENV_NAME="${ENV_NAME:-mmseg}"
export ENV_DIR="${ENV_DIR:-/p/project1/nxtaim-1/huber7/env_mmseg}"

export SYS100_DATA_PATH="/p/data1/nxtaim/proprietary/continental/"

export MMSEG_WORK_DIR="/p/scratch/nxtaim-1/huber7/work_dirs/$(basename $(pwd))"