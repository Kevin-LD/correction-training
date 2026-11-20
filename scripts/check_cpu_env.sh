#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
if [[ "${1:-}" == --no-log ]]; then
  shift
else
  mkdir -p "$ROOT/logs/setup"
  LOG_PATH="$ROOT/logs/setup/cpu-check-$(date -u +%Y%m%dT%H%M%SZ)-$$.log"
  {
    echo "Log: $LOG_PATH"
    PYTHONUNBUFFERED=1 bash "$ROOT/scripts/check_cpu_env.sh" --no-log "$@"
  } 2>&1 | tee "$LOG_PATH"
  exit
fi
source "$ROOT/scripts/activate_env.sh" cpu
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
cd "$ROOT"
python -m pip check
python - <<'PY'
import torch, diffusers, accelerate, huggingface_hub
print('PyTorch:', torch.__version__, 'CUDA build:', torch.version.cuda)
assert torch.__version__.split('+')[0] == '2.5.1'
assert torch.version.cuda == '12.4', 'Expected the CUDA 12.4 build'
x = torch.randn(32, 32, requires_grad=True)
(x @ x.T).square().mean().backward()
assert x.grad.isfinite().all()
print('CPU forward/backward passed; GPU available:', torch.cuda.is_available())
PY
python -m unittest discover -s src/tests -p 'test_*.py'
