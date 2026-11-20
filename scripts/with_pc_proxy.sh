#!/usr/bin/env bash
# Execute a command with a temporary, process-local PC proxy.
set -euo pipefail
if [[ "${BASH_SOURCE[0]}" != "$0" || $# == 0 ]]; then
  echo 'Usage: bash scripts/with_pc_proxy.sh COMMAND [ARG ...]' >&2
  exit 2
fi
pc_proxy="${PC_DOWNLOAD_PROXY:-http://127.0.0.1:17897}"
export HTTP_PROXY="$pc_proxy" HTTPS_PROXY="$pc_proxy" ALL_PROXY="$pc_proxy"
export http_proxy="$pc_proxy" https_proxy="$pc_proxy" all_proxy="$pc_proxy"
export NO_PROXY=localhost,127.0.0.1,::1,nexus.sii.shaipower.online
export no_proxy="$NO_PROXY"
exec "$@"
