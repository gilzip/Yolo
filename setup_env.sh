#!/usr/bin/env bash
# Creates a virtual environment and installs project dependencies.
# Usage: ./setup_env.sh
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-.venv}"

echo "Creating virtual environment in ${VENV_DIR} using ${PYTHON_BIN}..."
"${PYTHON_BIN}" -m venv "${VENV_DIR}"

# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

echo "Upgrading pip..."
pip install --upgrade pip

echo "Installing dependencies from requirements.txt..."
pip install -r requirements.txt

echo ""
echo "Setup complete."
echo "Activate the environment with: source ${VENV_DIR}/bin/activate"
