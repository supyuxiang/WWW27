#!/usr/bin/env bash
# Run the non-retrieval llm_gen_only baseline for every local LLM.
# Dataset protocol: AVeriTeC=all (500), SciFact=all (300), FEVER=1000 sample.
set -uo pipefail

PY=/root/autodl-tmp/miniconda3/envs/verl1/bin/python
CODE=/root/www27
MODEL_ROOT=/root/autodl-tmp/models
OUT_ROOT=/root/autodl-tmp/www27/results/llm_gen_all
LOG_ROOT="$OUT_ROOT/_logs"
mkdir -p "$OUT_ROOT" "$LOG_ROOT"
cd "$CODE"

MODELS=(
  Qwen2.5-0.5B
  Qwen2.5-0.5B-Instruct
  Qwen2.5-1.5B
  Qwen2.5-1.5B-Instruct
  Qwen2.5-3B
  Qwen2.5-3B-Instruct
  Qwen2.5-7B
  Qwen2.5-7B-Instruct
  Qwen2.5-14B
  Qwen2.5-14B-Instruct
  Qwen2.5-32B
  Qwen2.5-32B-Instruct
  Qwen3-0.6B
  Qwen3-1.7B
  Qwen3-4B
  Qwen3-4B-Instruct-2507
  Qwen3-4B-Thinking-2507
  Qwen3-8B
  Qwen3-14B
  Qwen3-14B-Base
  Llama-3.2-3B-Instruct
  Qwen3.8-27B
)

for name in "${MODELS[@]}"; do
  model="$MODEL_ROOT/$name"
  out="$OUT_ROOT/$name"
  log="$LOG_ROOT/${name}.log"
  if [[ ! -d "$model" ]]; then
    echo "[SKIP] missing model: $model" | tee -a "$LOG_ROOT/runner.log"
    continue
  fi
  if [[ -s "$out/llm_gen_only_summary.json" ]]; then
    echo "[SKIP] existing summary: $name" | tee -a "$LOG_ROOT/runner.log"
    continue
  fi
  mkdir -p "$out"
  echo "[$(date '+%F %T')] START $name" | tee -a "$LOG_ROOT/runner.log"
  "$PY" baselines/llm_gen_only.py \
    --model "$model" \
    --benchmarks averitec fever scifact \
    --limit 1000 \
    --seed 42 \
    --output-dir "$out" \
    --batch-size 256 \
    --max-new-tokens 1024 \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.85 \
    --max-model-len 8192 \
    >"$log" 2>&1
  status=$?
  if [[ $status -eq 0 ]]; then
    echo "[$(date '+%F %T')] OK $name" | tee -a "$LOG_ROOT/runner.log"
  else
    echo "[$(date '+%F %T')] FAIL $name status=$status" | tee -a "$LOG_ROOT/runner.log"
  fi
done

echo "=== ALL DONE $(date '+%F %T') ===" | tee -a "$LOG_ROOT/runner.log"

