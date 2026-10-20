#!/usr/bin/env bash
# One-off recovery for the stalled Qwen3.8-27B final shard.
set -euo pipefail

source /etc/network_turbo

MODEL_DIR=/root/autodl-tmp/models/Qwen3.8-27B
TARGET="$MODEL_DIR/model-00017-of-00018.safetensors"
URL="https://huggingface.co/Qwen/Qwen3.8-27B/resolve/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0/model-00017-of-00018.safetensors"

pkill -f '/root/www27/scripts/download_models.py' 2>/dev/null || true
pkill -f '/root/www27/scripts/monitor_models.sh' 2>/dev/null || true
sleep 1

if [[ ! -s "$TARGET" ]]; then
    incomplete=$(find "$MODEL_DIR/.cache/huggingface/download" -name '*.incomplete' -type f | head -1 || true)
    if [[ -n "$incomplete" ]]; then
        cp "$incomplete" "$TARGET"
        echo "resumed from $incomplete"
    fi
fi

curl -L --retry 100 --retry-delay 5 --retry-all-errors \
    --connect-timeout 30 --speed-time 120 --speed-limit 1024 \
    -C - -o "$TARGET" "$URL"

echo "shards=$(find "$MODEL_DIR" -maxdepth 1 -name 'model-*.safetensors' | wc -l)"
ls -lh "$TARGET"
