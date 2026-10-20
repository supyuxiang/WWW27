"""Measure token counts of the evidence blocks returned by current retrieval."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from transformers import AutoTokenizer

from baselines.jev_only import load_benchmark
from baselines.jev_retrieval import RetrieverBundle

DEFAULT_MODEL = "/root/autodl-tmp/models/Qwen2.5-3B-Instruct"


def pct(values: list[int], q: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    index = min(len(values) - 1, max(0, int(round(q * (len(values) - 1)))))
    return float(values[index])


def summarize(values: list[int]) -> dict:
    return {
        "n": len(values),
        "mean": statistics.mean(values) if values else 0.0,
        "p50": pct(values, 0.50),
        "p90": pct(values, 0.90),
        "p99": pct(values, 0.99),
        "max": max(values) if values else 0,
        "min": min(values) if values else 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--output", default="/root/autodl-tmp/www27/results/evidence_token_stats.json")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    bundle = RetrieverBundle()
    result = {}
    for benchmark in ("scifact", "fever", "averitec"):
        examples = load_benchmark(benchmark, args.samples, 0)
        passage_tokens: list[int] = []
        passage_chars: list[int] = []
        block_tokens: list[int] = []
        per_claim: list[int] = []
        for example in examples:
            passages = bundle.search(benchmark, example["id"], example["claim"], args.top_k)
            total = 0
            for index, passage in enumerate(passages, start=1):
                text = passage.text or ""
                title = passage.title or ""
                block = f"[{index}] {title}\n{text}" if title else f"[{index}] {text}"
                text_ids = tokenizer.encode(text, add_special_tokens=False)
                block_ids = tokenizer.encode(block, add_special_tokens=False)
                passage_tokens.append(len(text_ids))
                passage_chars.append(len(text))
                block_tokens.append(len(block_ids))
                total += len(block_ids)
            per_claim.append(total)
        result[benchmark] = {
            "claims": len(examples),
            "passages": len(passage_tokens),
            "text_tokens": summarize(passage_tokens),
            "block_tokens": summarize(block_tokens),
            "text_chars": summarize(passage_chars),
            "claim_total_tokens": summarize(per_claim),
        }
    bundle.close()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
