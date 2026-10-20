#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
PY="${PY:-/root/autodl-tmp/miniconda3/envs/verl1/bin/python}"
"$PY" -m unittest discover -s tests -v
"$PY" -m src.run \
  --config configs/default.yaml \
  --limit 1 \
  --max-aspects 1 \
  --search-cap 2 \
  --output output/smoke.jsonl
"$PY" - << 'PY'
import json
from pathlib import Path
row = json.loads(Path("output/smoke.jsonl").read_text(encoding="utf-8").splitlines()[0])
assert row["prediction"] in {"real", "fake"}, row
assert row["error"] is None, row["error"]
assert row["retrieval_rounds"] >= 1, row
assert row["stop_reason"] in {"gate", "budget", "no_uncovered_aspect", "no_query"}, row
print("smoke ok", row["prediction"], "calls", row["search_calls"], "rounds", row["retrieval_rounds"], "stop", row["stop_reason"])
PY
