#!/bin/bash
# select_joint on node7: aspect-conditioned top-2, drop Insufficient, one joint Judge-2.
set -uo pipefail
PY=/home/yxfeng/miniconda3/envs/www27/bin/python
cd /home/yxfeng/www27
LOG=/data3/yxfeng/www27/logs
OUT=/data3/yxfeng/www27/results/select_joint
mkdir -p "$LOG" "$OUT" "$OUT/smoke"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "[select_joint start] $(date -Is)"
CUDA_VISIBLE_DEVICES=0 "$PY" eval/eval_main.py \
  --variant select_joint --benchmarks scifact --limit 1 \
  --binary --fs-limit 0 --bm25-k 128 --rerank-k 2 --tau-aspect 0.5 \
  --rerank-batch-size 16 --rerank-max-length 512 \
  --output-dir "$OUT/smoke" \
  > "$LOG/select_joint_smoke.log" 2>&1
smoke_code=$?
if [[ "$smoke_code" -ne 0 ]] || grep -q 'Traceback\|CUDA out of memory' "$LOG/select_joint_smoke.log"; then
  echo "[SMOKE_FAIL exit $smoke_code] $(date -Is)"
  exit 1
fi
echo "[SMOKE_OK] $(date -Is)"

launch() {
  local shard="$1"
  CUDA_VISIBLE_DEVICES="$shard" "$PY" eval/eval_main.py \
    --variant select_joint --shard "$shard" --num-shards 8 \
    --benchmarks scifact averitec healthver climate_fever \
    --binary --fs-limit 0 --bm25-k 128 --rerank-k 2 --tau-aspect 0.5 \
    --rerank-batch-size 16 --rerank-max-length 512 \
    --output-dir "$OUT" \
    > "$LOG/select_joint_s${shard}.log" 2>&1 &
}
for shard in 0 1 2 3 4 5 6 7; do
  launch "$shard"
done
fail=0
wait || fail=1
"$PY" eval/merge_results.py "$OUT" > "$LOG/select_joint_merged.log" 2>&1 || fail=1
"$PY" - << 'PY' || fail=1
import json
from pathlib import Path
order = ["averitec", "scifact", "healthver", "climate_fever"]
once = {"averitec": 88.06, "scifact": 84.57, "healthver": 78.32, "climate_fever": 74.86}
report = json.loads(Path("/data3/yxfeng/www27/results/select_joint/merged.json").read_text())
print("benchmark\tn\tacc\tcalls\tretrieve_once\tdelta")
for name in order:
    row = report[name]
    acc = 100.0 * row["accuracy"]
    delta = acc - once[name]
    print(f"{name}\t{row['n']}\t{acc:.2f}\t{row['mean_calls']:.2f}\t{once[name]:.2f}\t{delta:+.2f}")
PY
echo "[select_joint exit $fail] $(date -Is)"
exit "$fail"
