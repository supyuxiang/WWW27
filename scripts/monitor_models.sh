#!/usr/bin/env bash
# Lightweight monitor for the background model download.
# Writes a durable progress log; exits when download_models.py exits.
set -uo pipefail
LOG=/root/autodl-tmp/models/download_progress.log
DOWNLOAD_LOG=/root/autodl-tmp/models/download.log

while pgrep -f 'scripts/download_models.py' >/dev/null 2>&1; do
    {
        date '+%Y-%m-%d %H:%M:%S'
        df -h /root/autodl-tmp | tail -1
        du -sh /root/autodl-tmp/models 2>/dev/null
        find /root/autodl-tmp/models -maxdepth 1 -mindepth 1 -type d -printf '%f\n' 2>/dev/null | sort | tr '\n' ' '
        echo
        tail -1 "$DOWNLOAD_LOG" 2>/dev/null
        echo '---'
    } >> "$LOG" 2>&1
    sleep 300
done

echo "$(date '+%Y-%m-%d %H:%M:%S') DOWNLOAD_PROCESS_EXITED" >> "$LOG"
