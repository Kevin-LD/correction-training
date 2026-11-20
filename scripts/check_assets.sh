#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
source "$ROOT/scripts/activate_env.sh" cpu
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
mkdir -p "$ROOT/logs/setup"
python -u "$ROOT/scripts/check_assets.py" 2>&1 | tee "$ROOT/logs/setup/assets-check-$(date -u +%Y%m%dT%H%M%SZ)-$$.log"
