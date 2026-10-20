#!/usr/bin/env bash
# Download the requested Qwen base/instruct models to /root/autodl-tmp/models.
# Use: bash scripts/download_models.sh --dry-run
#      nohup bash scripts/download_models.sh > /root/autodl-tmp/models/download.log 2>&1 &
set -uo pipefail

if [[ -f /etc/network_turbo ]]; then
    # shellcheck disable=SC1091
    source /etc/network_turbo
else
    echo "WARN: /etc/network_turbo not found; continuing without proxy" >&2
fi

export HF_HOME=/root/autodl-tmp/hf_home
export HF_HUB_DISABLE_XET=1
export TOKENIZERS_PARALLELISM=false
PY=/root/autodl-tmp/miniconda3/envs/www27/bin/python
exec "$PY" /root/www27/scripts/download_models.py "$@"
