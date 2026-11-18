#!/usr/bin/env bash
# Dense Doffy / retrieve-once on the four small datasets. FEVER is not run.
# Start this inside the ADA allocation, for example:
#   srun --jobid=148958 --overlap --gres=gpu:4 bash scripts/run_paper_small.sh
set -uo pipefail

PY=/home/yxfeng/miniconda3/envs/www27/bin/python
PREP=/home/yxfeng/miniconda3/envs/verl1/bin/python
ROOT=/home/yxfeng/www27
LOG=/data3/yxfeng/www27/logs
RES=/data3/yxfeng/www27/results
mkdir -p "$LOG" "$RES"

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
cd "$ROOT"

echo "[prepare] $(date -Is)"
"$PREP" data/prepare_small.py > "$LOG/prepare.log" 2>&1
prep_status=$?
if [[ $prep_status -ne 0 ]]; then
  echo "prepare failed, see $LOG/prepare.log"
  exit "$prep_status"
fi

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "[index] $(date -Is)"
CUDA_VISIBLE_DEVICES=0 "$PY" data/build_index.py > "$LOG/build_index.log" 2>&1
index_status=$?
if [[ $index_status -ne 0 ]]; then
  echo "index failed, see $LOG/build_index.log"
  exit "$index_status"
fi

BENCH="scifact averitec healthver climate_fever"

echo "[smoke] $(date -Is)"
CUDA_VISIBLE_DEVICES=0 "$PY" eval/eval_main.py \
  --benchmarks scifact \
  --limit 2 \
  --binary \
  --fs-limit 0 \
  --bm25-k 128 \
  --rerank-k 2 \
  --rerank-batch-size 32 \
  --rerank-max-length 512 \
  --output-dir "$RES/smoke" \
  > "$LOG/smoke.log" 2>&1
smoke_status=$?
if [[ $smoke_status -ne 0 ]]; then
  echo "smoke failed, see $LOG/smoke.log"
  exit "$smoke_status"
fi
echo "[smoke ok] $(date -Is)"

run_shards() {
  local script="$1"
  local out="$2"
  local tag="$3"
  shift 3
  local pids=()
  local shard
  for shard in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=$shard "$PY" "$script" \
      --benchmarks $BENCH \
      --shard "$shard" \
      --num-shards 4 \
      --output-dir "$out" \
      "$@" \
      > "$LOG/${tag}_shard${shard}.log" 2>&1 &
    pids+=($!)
  done
  local fail=0
  local pid
  for pid in "${pids[@]}"; do
    wait "$pid" || fail=1
  done
  "$PY" eval/merge_results.py "$out" > "$LOG/${tag}_merged.log" 2>&1 || fail=1
  return "$fail"
}

echo "[doffy] $(date -Is)"
run_shards eval/eval_main.py "$RES/doffy/Qwen3-8B" doffy \
  --fs-limit 0 \
  --bm25-k 128 \
  --rerank-k 2 \
  --search-cap 3 \
  --alpha 3 \
  --tau-aspect 0.5 \
  --rerank-batch-size 32 \
  --rerank-max-length 512
doffy_status=$?
echo "[doffy exit $doffy_status] $(date -Is)"

echo "[retrieve-once] $(date -Is)"
run_shards eval/jev_retrieval_once.py "$RES/retrieve_once/Qwen3-8B" retrieve_once \
  --fs-limit 0 \
  --bm25-k 128 \
  --rerank-k 2 \
  --rerank-batch-size 32 \
  --rerank-max-length 512
once_status=$?
echo "[retrieve-once exit $once_status] $(date -Is)"

echo "[jev-only] $(date -Is)"
run_shards eval/jev_only.py "$RES/jev_only/Qwen3-8B" jev_only \
  --fs_limit 0 \
  --closed-book
jev_status=$?
echo "[jev-only exit $jev_status] $(date -Is)"

echo "[llm-gen] $(date -Is)"
run_shards eval/llm_gen.py "$RES/llm_gen/Qwen3-8B" llm_gen \
  --fs_limit 0 \
  --closed-book \
  --tensor_parallel_size 1 \
  --gpu_memory_utilization 0.90 \
  --max_model_len 4096 \
  --max_tokens 512
gen_status=$?
echo "[llm-gen exit $gen_status] $(date -Is)"

echo "done doffy=$doffy_status retrieve_once=$once_status jev=$jev_status gen=$gen_status"
exit $((doffy_status || once_status))
