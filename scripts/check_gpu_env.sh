#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
if [[ "${1:-}" == --no-log ]]; then
  shift
else
  case "${1:-}" in
    '') LABEL=gpu-check ;;
    --distributed) LABEL=gpu-distributed ;;
    *) echo 'Usage: bash scripts/check_gpu_env.sh [--distributed]' >&2; exit 1 ;;
  esac
  mkdir -p "$ROOT/logs/setup"
  LOG_PATH="$ROOT/logs/setup/$LABEL-$(date -u +%Y%m%dT%H%M%SZ)-$$.log"
  {
    echo "Log: $LOG_PATH"
    PYTHONUNBUFFERED=1 bash "$ROOT/scripts/check_gpu_env.sh" --no-log "$@"
  } 2>&1 | tee "$LOG_PATH"
  exit
fi
source "$ROOT/scripts/activate_env.sh" gpu
cd "$ROOT"
python -m pip check
if [[ "${1:-}" == --distributed ]]; then
  OMP_NUM_THREADS=1 torchrun --standalone --nproc_per_node=8 scripts/check_gpu_env.py --distributed
elif [[ $# == 0 ]]; then
  python scripts/check_gpu_env.py
else
  echo 'Usage: bash scripts/check_gpu_env.sh [--distributed]' >&2
  exit 1
fi
