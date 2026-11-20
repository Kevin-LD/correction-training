#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
CONDA_BIN=/inspire/hdd/global_user/ky26466/.tools/miniconda3/bin/conda
export CONDA_PKGS_DIRS="$ROOT/.cache/conda"
export CONDA_ENVS_PATH="$ROOT/envs"
export PIP_CACHE_DIR="$ROOT/.cache/pip"
export PIP_CONFIG_FILE=/dev/null
export PIP_INDEX_URL=https://repo.huaweicloud.com/repository/pypi/simple/
export PIP_EXTRA_INDEX_URL=
if [[ ! -x "$ROOT/envs/correction-adm/bin/python" ]]; then
  "$CONDA_BIN" create --yes --prefix "$ROOT/envs/correction-adm" --override-channels \
    --channel http://nexus.sii.shaipower.online/repository/anaconda/pkgs/main python=3.11 pip
fi
"$ROOT/envs/correction-adm/bin/python" -m pip install --timeout 30 --retries 2 -r "$ROOT/requirements-adm.txt"
"$ROOT/envs/correction-adm/bin/python" -m pip check
"$ROOT/envs/correction-adm/bin/python" -m pip freeze > "$ROOT/environment/requirements-adm.lock.txt"
