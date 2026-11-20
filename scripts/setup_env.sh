#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
CONDA_BIN=/inspire/hdd/global_user/ky26466/.tools/miniconda3/bin/conda
export CONDA_PKGS_DIRS="$ROOT/.cache/conda"
export CONDA_ENVS_PATH="$ROOT/envs"
export PIP_CACHE_DIR="$ROOT/.cache/pip"
export PIP_CONFIG_FILE=/dev/null
export PIP_INDEX_URL="${CORRECTION_PIP_INDEX_URL:-http://nexus.sii.shaipower.online/repository/pypi/simple/}"
export PIP_EXTRA_INDEX_URL=
mkdir -p "$CONDA_PKGS_DIRS" "$PIP_CACHE_DIR" "$ROOT/logs/setup" \
  "$ROOT/.cache/huggingface" "$ROOT/.cache/torch" "$ROOT/checkpoints/pretrained" "$ROOT/data"
if [[ ! -x "$ROOT/envs/correction/bin/python" ]]; then
  "$CONDA_BIN" create --yes --prefix "$ROOT/envs/correction" --override-channels \
    --channel http://nexus.sii.shaipower.online/repository/anaconda/pkgs/main \
    python=3.11 pip
fi
"$ROOT/envs/correction/bin/python" -m pip install \
  --trusted-host nexus.sii.shaipower.online --timeout 30 --retries 2 \
  -r "$ROOT/requirements-latent.txt" 'numpy<2.1'
"$ROOT/envs/correction/bin/python" -m pip check
"$ROOT/envs/correction/bin/python" -m pip freeze > "$ROOT/environment/requirements.lock.txt"
"$CONDA_BIN" list --prefix "$ROOT/envs/correction" --explicit > "$ROOT/environment/conda-linux-64.explicit.txt"
