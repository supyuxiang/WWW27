"""No-retrieval LLM-as-Jev baseline on AVeriTeC, FEVER, and SciFact.

This baseline asks the frozen model for a binary Real/Fake decision using
the same candidate-suffix scoring interface as JARV.  It never calls a
retriever: the judge always receives ``subclaims = none`` and
``evidence = none``.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.prompts import SYSTEM_PROMPT, judge_prompt  # noqa: E402
from src.utils.metrics import accuracy, macro_f1  # noqa: E402

BENCHMARK_ROOT = Path("/root/autodl-tmp/www27/benchmark")
DEFAULT_MODEL = "/root/autodl-tmp/models/Qwen3-8B"


def _normalize_label(value: object) -> str | None:
    text = str(value or "").strip().lower()
    if text in {"supported", "supports", "support", "true", "real", "real news"}:
        return "real"
    if text in {"refuted", "refutes", "refute", "false", "fake", "fake news"}:
        return "fake"
    return None


def _record_id(record: dict, index: int) -> str:
    for key in ("id", "claim_id", "claimId"):
        if record.get(key) is not None:
            return str(record[key])
    # AVeriTeC dev rows have no explicit id; the claim-scoped SQLite index
    # uses the row/file index as claim_id.
    return str(index)


def _load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _load_json_array(path: Path) -> list[dict]:
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(loaded, list):
        return loaded
    if isinstance(loaded, dict):
        for key in ("data", "claims", "rows", "examples"):
            if isinstance(loaded.get(key), list):
                return loaded[key]
    raise SystemExit(f"Unsupported AVeriTeC JSON shape: {path}")


def _scifact_label(record: dict) -> str | None:
    evidence = record.get("evidence") or {}
    labels: set[str] = set()

    def visit(value: object) -> None:
        if isinstance(value, dict):
            label = value.get("label")
            if isinstance(label, str):
                labels.add(label.strip().upper())
            for item in value.values():
                visit(item)
        elif isinstance(value, list):
            for item in value:
                visit(item)

    visit(evidence)
    if labels == {"SUPPORT"}:
        return "real"
    if labels and labels <= {"CONTRADICT", "REFUTE"}:
        return "fake"
    return None


def load_benchmark(name: str, limit: int, seed: int) -> list[dict]:
    if name == "fever":
        rows = _load_jsonl(BENCHMARK_ROOT / "fever" / "shared_task_dev.jsonl")
        examples = [
            {
                "id": _record_id(row, index),
                "claim": str(row.get("claim") or "").strip(),
                "gold": _normalize_label(row.get("label")),
                "raw_label": row.get("label"),
            }
            for index, row in enumerate(rows)
        ]
    elif name == "averitec":
        rows = _load_json_array(BENCHMARK_ROOT / "averitec" / "data" / "dev.json")
        examples = [
            {
                "id": _record_id(row, index),
                "claim": str(row.get("claim") or "").strip(),
                "gold": _normalize_label(row.get("label")),
                "raw_label": row.get("label"),
            }
            for index, row in enumerate(rows)
        ]
    elif name == "scifact":
        rows = _load_jsonl(BENCHMARK_ROOT / "scifact" / "data" / "claims_dev.jsonl")
        examples = [
            {
                "id": _record_id(row, index),
                "claim": str(row.get("claim") or "").strip(),
                "gold": _scifact_label(row),
                "raw_label": row.get("evidence"),
            }
            for index, row in enumerate(rows)
        ]
    else:
        raise SystemExit(f"Unknown benchmark {name}")
    examples = [example for example in examples if example["claim"] and example["gold"]]
    if limit > 0 and len(examples) > limit:
        examples = random.Random(seed).sample(examples, limit)
        examples.sort(key=lambda item: item["id"])
    return examples


def run_benchmark(name: str, examples: list[dict], model_path: str, output_dir: Path) -> dict:
    from src.llm import LocalLLM

    llm = LocalLLM(model_path)
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
            try:
                user, options = judge_prompt(example["claim"], "none", "none")
                choice = llm.score_options(SYSTEM_PROMPT, user, options)
                prediction = choice.key
                probs = choice.probs
                if prediction not in {"real", "fake"}:
                    error = f"invalid prediction {prediction!r}"
                else:
                    gold.append(example["gold"])
                    pred.append(prediction)
            except Exception as exc:  # keep a partially usable run
                error = f"{type(exc).__name__}: {exc}"
            latency = time.perf_counter() - item_started
            latencies.append(latency)
            if error:
                n_errors += 1
            row = {
                "benchmark": name,
                "id": example["id"],
                "claim": example["claim"],
                "gold": example["gold"],
                "raw_label": example["raw_label"],
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
        "accuracy": accuracy(gold, pred),
        "macro_f1": macro_f1(gold, pred),
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
    parser = argparse.ArgumentParser(description="No-retrieval LLM-as-Jev baseline")
    parser.add_argument("--benchmarks", nargs="+", default=["averitec", "fever", "scifact"])
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--limit", type=int, default=200, help="0 means all binary claims")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", default="/root/www27/output")
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    all_summaries = {}
    for name in args.benchmarks:
        examples = load_benchmark(name, args.limit, args.seed)
        print(f"{name}: {len(examples)} binary claims", flush=True)
        all_summaries[name] = run_benchmark(name, examples, args.model, output_dir)
    (output_dir / "jev_only_summary.json").write_text(
        json.dumps(all_summaries, indent=2), encoding="utf-8"
    )
    lines = [
        "# No-retrieval LLM-as-Jev baseline",
        "",
        "| Benchmark | n | Accuracy | Macro-F1 | Errors |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, summary in all_summaries.items():
        lines.append(
            f"| {name} | {summary['n']} | {summary['accuracy']:.4f} | "
            f"{summary['macro_f1']:.4f} | {summary['n_errors']} |"
        )
    (output_dir / "jev_only_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("wrote", output_dir / "jev_only_summary.json", flush=True)


if __name__ == "__main__":
    main()
