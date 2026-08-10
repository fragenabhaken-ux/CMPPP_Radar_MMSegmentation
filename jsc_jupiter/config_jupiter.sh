#!/usr/bin/env bash

SOURCE_PATH="${BASH_SOURCE[0]:-${(%):-%x}}"

[[ "$0" != "${SOURCE_PATH}" ]] && echo "Setting vars" || ( echo "Config script must be sourced." && exit 1 )

SCRIPT_DIR="$(cd "$(dirname "$SOURCE_PATH")" && pwd)"
REPO_DIR="$(realpath "${SCRIPT_DIR}/..")"
USER_PROJECT_DIR="$(realpath "${SCRIPT_DIR}/../../..")"

export ENV_NAME="${ENV_NAME:-env_mmseg_jupiter}"
export ENV_DIR="${USER_PROJECT_DIR}/env_mmseg_jupiter"
export REPO_DIR="${REPO_DIR}"

echo "SCRIPT_DIR       = $SCRIPT_DIR"
echo "REPO_DIR         = $REPO_DIR"
echo "USER_PROJECT_DIR = $USER_PROJECT_DIR"
echo "ENV_NAME         = $ENV_NAME"
echo "ENV_DIR          = $ENV_DIR"