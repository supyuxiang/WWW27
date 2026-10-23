from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.worker import (
    LLM_as_Jev,
    API_Jev,
)
from src.prompts import (
    prompt4jev_judge,
    SYSTEM_PROMPT,
    build_few_shot,
)
from src.utils import (
    Benchmark,
    compute_accuracy,
    compute_f1,
    load_benchmark,
)

DEFAULT_OUTPUT_DIR = "/root/autodl-tmp/www27/jev_only"


def _as_floats(value: Any) -> list[float]:
    if hasattr(value, "detach"):
        value = value.detach().float().cpu().tolist()
    return [float(item) for item in value]


def run_one(
    examples: list[dict],
    bench: Benchmark,
    jev: LLM_as_Jev | API_Jev,
    few_shot: str,
    output_dir: Path,
    batch_size: int = 4,
    max_sequences: int = 8,
    model_path: str = "",
) -> dict[str, Any]:
    name = bench.name
    output_dir.mkdir(parents=True, exist_ok=True)
    labels = list(bench.choice_dict)
    prompts = [
        prompt4jev_judge(
            benchmark_cls=bench,
            claim=item["claim"],
            few_shot=few_shot,
            evidence=item["evidence"],
        )
        for item in examples
    ]  # list[tuple[str, str, list[str]]]
    if not prompts:
        raise ValueError(f"{name}: no examples")

    user_prompts = [prompt[1] for prompt in prompts]
    options = prompts[0][-1]
    if any(prompt[-1] != options for prompt in prompts):
        raise ValueError(f"{name}: options differ across prompts")
    if len(options) != len(labels):
        raise ValueError(
            f"{name}: {len(options)} options but {len(labels)} labels"
        )

    gold: list[str] = []
    pred: list[str] = []
    trajectory_path = output_dir / f"jev_only_{name}.jsonl"
    started = time.perf_counter()
    with trajectory_path.open("w", encoding="utf-8") as handle:
        for idx in tqdm(range(0, len(user_prompts), batch_size), desc=name):
            batch_examples = examples[idx: idx + batch_size]
            batch = user_prompts[idx: idx + batch_size]
            batch_out = jev.score_batch(
                SYSTEM_PROMPT,
                batch,
                options,
                max_sequences=max_sequences,
            )
            if len(batch_out) != len(batch):
                raise RuntimeError(
                    f"{name}: score_batch returned {len(batch_out)} rows "
                    f"for {len(batch)} prompts"
                )
            for item, scored in zip(batch_examples, batch_out):
                best_idx = int(scored["best_idx"])
                prediction = labels[best_idx]
                probs = _as_floats(scored["probs"])
                scores = _as_floats(scored["scores"])
                gold.append(item["label"])
                pred.append(prediction)
                handle.write(json.dumps({
                    "benchmark": name,
                    "id": item.get("id"),
                    "claim": item["claim"],
                    "gold": item["label"],
                    "evidence": item["evidence"],
                    "prediction": prediction,
                    "judge_probs": {
                        label: probs[i] for i, label in enumerate(labels)
                    },
                    "scores": scores,
                }, ensure_ascii=False) + "\n")
            handle.flush()

    summary = {
        "benchmark": name,
        "model": model_path,
        "n": len(gold),
        "accuracy": compute_accuracy(gold, pred),
        "f1": compute_f1(gold, pred, bench),
        "gold_counts": dict(Counter(gold)),
        "prediction_counts": dict(Counter(pred)),
        "few_shot": few_shot,
        "elapsed_s": time.perf_counter() - started,
        "output": str(trajectory_path),
    }
    (output_dir / f"jev_only_{name}.summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM-as-Jev with gold evidence")
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        default=["averitec", "fever", "scifact", "healthver"],
    )
    parser.add_argument("--model_path", default="/root/autodl-tmp/models/Qwen3-8B")
    parser.add_argument("--limit", type=int, default=-1, help="<=0 means the full split")
    parser.add_argument("--output_dir", default="")
    parser.add_argument("--fs_limit", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_sequences", type=int, default=8)
    args = parser.parse_args()

    if args.batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {args.batch_size}")
    if args.max_sequences <= 0:
        raise ValueError(f"max_sequences must be positive, got {args.max_sequences}")

    model_name = Path(args.model_path).name
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else Path(DEFAULT_OUTPUT_DIR) / model_name
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    jev = LLM_as_Jev(model_path=args.model_path)
    summaries: dict[str, dict] = {}
    for name in args.benchmarks:
        examples, bench = load_benchmark(name, limit=args.limit)
        print(f"{name}: {len(examples)} claims", flush=True)
        few_shot = build_few_shot(
            bench_cls=bench,
            limit=args.fs_limit,
            split="train",
        )
        summaries[name] = run_one(
            examples=examples,
            bench=bench,
            jev=jev,
            few_shot=few_shot,
            output_dir=output_dir,
            batch_size=args.batch_size,
            max_sequences=args.max_sequences,
            model_path=args.model_path,
        )

    (output_dir / "jev_only_summary.json").write_text(
        json.dumps(summaries, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    lines = [
        "# LLM-as-Jev with gold evidence",
        "",
        f"Model: `{args.model_path}`",
        "",
        "| Benchmark | n | Accuracy | F1 |",
        "|---|---:|---:|---:|",
    ]
    for name, summary in summaries.items():
        lines.append(
            f"| {name} | {summary['n']} | {summary['accuracy']:.4f} | "
            f"{summary['f1']:.4f} |"
        )
    (output_dir / "jev_only_summary.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    print("wrote", output_dir, flush=True)


if __name__ == "__main__":
    main()
