#!/usr/bin/env bash
# Source this file: source scripts/activate_env.sh [cpu|gpu]
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo 'Use: source scripts/activate_env.sh [cpu|gpu]' >&2
  exit 1
fi
_correction_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
_correction_mode=${1:-cpu}
if [[ "$_correction_mode" != cpu && "$_correction_mode" != gpu ]]; then
  echo 'Expected cpu or gpu' >&2
  return 1
fi
source /inspire/hdd/global_user/ky26466/.tools/miniconda3/etc/profile.d/conda.sh
conda activate "$_correction_root/envs/correction"
export HF_HOME="$_correction_root/.cache/huggingface"
export TORCH_HOME="$_correction_root/.cache/torch"
export CORRECTION_CHECKPOINT_DIR="$_correction_root/checkpoints"
export PIP_CACHE_DIR="$_correction_root/.cache/pip"
if [[ "$_correction_mode" == gpu ]]; then
  export HF_HUB_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
else
  export HF_ENDPOINT=https://hf-mirror.com
  export HF_HUB_OFFLINE=0
  export TRANSFORMERS_OFFLINE=0
fi
unset _correction_root _correction_mode
