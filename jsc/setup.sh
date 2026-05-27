# ### ALT ###

# #!/bin/bash

# SOURCE_PATH="${BASH_SOURCE[0]:-${(%):-%x}}"

# RELATIVE_PATH="$(dirname "$SOURCE_PATH")"
# ABSOLUTE_PATH="$(realpath "${RELATIVE_PATH}")"

# source "${ABSOLUTE_PATH}"/config.sh
# source "${ABSOLUTE_PATH}"/modules.sh

# python3 -m venv --prompt "$ENV_NAME" --system-site-packages "${ENV_DIR}"

# source "${ABSOLUTE_PATH}"/activate.sh

# python3 -m pip install --upgrade -r "${ABSOLUTE_PATH}"/requirements.txt

### NEU ###

#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo $SCRIPT_DIR

source "$SCRIPT_DIR/config.sh"
source "$SCRIPT_DIR/modules.sh"

python -m venv --prompt "$ENV_NAME" --system-site-packages "$ENV_DIR"
source "$SCRIPT_DIR/activate.sh"

pip install "typing-extensions>=4.8.0"
pip install --upgrade pip setuptools wheel
pip install -r "$SCRIPT_DIR/requirements.txt"
#pip install -e . # Hier wurde die remote connection abgebrochen
