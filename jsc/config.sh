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

# ### JONAS ###

# SOURCE_PATH="${BASH_SOURCE[0]:-${(%):-%x}}"

# ## Check if this script is sourced

# [[ "$0" != "${SOURCE_PATH}" ]] && echo "Setting vars" || ( echo "Vars script must be sourced." && exit 1) ;

# ## Determine location of this file and from this the location of the parent CleanStableDiffusion directory

# RELATIVE_PATH="$(dirname "$SOURCE_PATH")"

# ABSOLUTE_PARENT_PATH="$(realpath "${RELATIVE_PATH}/..")"

# ####################################

# ### User Configuration

# export ENV_NAME="$(basename "$ABSOLUTE_PARENT_PATH")" # Default Name of the venv is the directory that contains this file

# export ENV_DIR="${ABSOLUTE_PARENT_PATH}"/venv         # Default location of this VENV is "./venv"
 

### NEU ###


SOURCE_PATH="${BASH_SOURCE[0]:-${(%):-%x}}"

# Check if sourced
[[ "$0" != "${SOURCE_PATH}" ]] && echo "Setting vars" || ( echo "Vars script must be sourced." && exit 1)

RELATIVE_PATH="$(dirname "$SOURCE_PATH")"
ABSOLUTE_PATH="$(realpath "${RELATIVE_PATH}")"

ABSOLUTE_PARENT_PATH="$(realpath "${RELATIVE_PATH}/../../..")"

# # --- User Configuration (JUWELS) ---
# export ENV_NAME="mmseg"
# export ENV_DIR="/p/project1/nxtaim-1/huber7/env_mmseg"

# ENV_NAME = mmseg, oder bleibt anders, falls anders gesetzt
export ENV_NAME="${ENV_NAME:-mmseg}" # env name == mmseg
export ENV_DIR="$ABSOLUTE_PARENT_PATH/env_mmseg"

# export SYS100_DATA_PATH="/p/data1/nxtaim/proprietary/continental/"
# export SYS100_DATA_PATH="/p/data1/nxtaim/proprietary_samples/continental_sys100/"

# i.e. MMSEG_WORK_DIR=/p/scratch/nxtaim-1/huber7/work_dirs/mmsegmentation_ppp
export MMSEG_WORK_DIR="/p/scratch/nxtaim-1/huber7/work_dirs/$(basename $(pwd))"

echo "SOURCE_PATH           = $SOURCE_PATH"
echo "ABSOLUTE_PATH         = $ABSOLUTE_PATH"
echo "ABSOLUTE_PARENT_PATH  = $ABSOLUTE_PARENT_PATH"
echo "ENV_DIR               = $ENV_DIR"