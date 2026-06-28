#!/usr/bin/env bash
set -euo pipefail

# Ensure working directories
mkdir -p /code/output

# Run full pipeline end-to-end
python3 /code/predict.py

