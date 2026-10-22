from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.worker import LLM_as_Jev
from src.prompts import prompt4what_choice
from src.utils import (
    Benchmark,
    compute_accuracy,
    compute_f1,
    load_benchmark
)
DEFAULT_MODEL = "/root/autodl-tmp/models/Qwen3-8B"


def run_benchmark(
    name: str,
    examples: list[dict],
    bench: Benchmark,
    model_path: str,
    output_dir: Path,
) -> dict:
    llm = LLM_as_Jev(model_path)
    labels = list(bench.choice_dict)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"jev_only_{name}.jsonl"
    started = time.perf_counter()
    gold: list[str] = []
    pred: list[str] = []
    n_errors = 0
    latencies: list[float] = []
    with output_path.open("w", encoding="utf-8") as handle:
        for index, example in enumerate(examples, start=1):
            item_started = time.perf_counter()
            prediction = None
            probs: dict[str, float] = {}
            error = None
            system_prompt, user_prompt, options = prompt4what_choice(bench, example["claim"])
            scored = llm.score_options(system_prompt, user_prompt, options)
            best_idx = int(scored["best_idx"])
            prediction = labels[best_idx]
            probs = {label: float(scored["probs"][i]) for i, label in enumerate(labels)}
            gold.append(example["label"])
            pred.append(prediction)
            
            latency = time.perf_counter() - item_started
            latencies.append(latency)
            if error:
                n_errors += 1
            row = {
                "benchmark": name,
                "id": example.get("id"),
                "claim": example["claim"],
                "gold": example.get("label"),
                "prediction": prediction,
                "judge_probs": probs,
                "latency_s": latency,
                "retrieved": False,
                "search_calls": 0,
                "error": error,
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            if index % 25 == 0 or index == len(examples):
                print(f"{name}: {index}/{len(examples)}", flush=True)
    elapsed = time.perf_counter() - started
    summary = {
        "benchmark": name,
        "model": model_path,
        "retrieval": "none",
        "n": len(gold),
        "n_errors": n_errors,
        "n_input": len(examples),
        "accuracy": compute_accuracy(gold, pred),
        "f1": compute_f1(gold, pred, bench),
        "gold_counts": dict(Counter(gold)),
        "prediction_counts": dict(Counter(pred)),
        "elapsed_s": elapsed,
        "avg_latency_s": sum(latencies) / len(latencies) if latencies else 0.0,
        "output": str(output_path),
    }
    (output_dir / f"jev_only_{name}.summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="No-retrieval JEV baseline")
    parser.add_argument("--benchmarks", nargs="+", default=["averitec", "fever", "scifact"])
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--limit", type=int, default=500, help="<=0 means the full split")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", default="/root/www27/output")
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    all_summaries = {}
    for name in args.benchmarks:
        examples, bench = load_benchmark(name, args.limit)
        print(f"{name}: {len(examples)} claims", flush=True)
        all_summaries[name] = run_benchmark(name, examples, bench, args.model, output_dir)
    (output_dir / "jev_only_summary.json").write_text(
        json.dumps(all_summaries, indent=2), encoding="utf-8"
    )
    lines = [
        "# No-retrieval JEV baseline",
        "",
        "| Benchmark | n | Accuracy | F1 | Errors |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, summary in all_summaries.items():
        lines.append(
            f"| {name} | {summary['n']} | {summary['accuracy']:.4f} | "
            f"{summary['f1']:.4f} | {summary['n_errors']} |"
        )
    (output_dir / "jev_only_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("wrote", output_dir / "jev_only_summary.json", flush=True)


if __name__ == "__main__":
    main()
