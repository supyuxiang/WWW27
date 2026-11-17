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

from src.workers.jev import API_Jev, LLM_as_Jev
from src.prompts import (
    prompt4jev_judge,
    build_few_shot,
)
from src.paths import RESULTS
from src.utils import (
    Benchmark,
    compute_accuracy,
    compute_f1,
    iter_split,
    per_class_scores,
)

DEFAULT_OUTPUT_DIR = str(RESULTS / "jev_only")


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
    model_path: str = "",
    closed_book: bool = True,
    shard: int = 0,
    num_shards: int = 1,
) -> dict[str, Any]:
    name = bench.name
    output_dir.mkdir(parents=True, exist_ok=True)
    if not examples:
        return {
            "benchmark": name,
            "n": 0,
            "accuracy": 0.0,
            "macro_f1": 0.0,
            "per_class": {},
            "gold_counts": {},
            "prediction_counts": {},
            "few_shot": few_shot,
            "elapsed_s": 0.0,
            "output": "",
        }
    labels = list(bench.choice_dict)
    prompts = [
        prompt4jev_judge(
            benchmark_cls=bench,
            claim=item["claim"],
            few_shot=few_shot or None,
            evidence=None if closed_book else item.get("evidence"),
        )
        for item in examples
    ]
    if not prompts:
        raise ValueError(f"{name}: no examples")

    options = prompts[0][-1]
    if any(prompt[-1] != options for prompt in prompts):
        raise ValueError(f"{name}: options differ across prompts")
    if len(options) != len(labels):
        raise ValueError(
            f"{name}: {len(options)} options but {len(labels)} labels"
        )

    gold: list[str] = []
    pred: list[str] = []
    suffix = f".shard{shard}" if num_shards > 1 else ""
    trajectory_path = output_dir / f"jev_only_{name}{suffix}.jsonl"
    started = time.perf_counter()
    with trajectory_path.open("w", encoding="utf-8") as handle:
        for item, (system_prompt, user_prompt, prompt_options) in tqdm(
            zip(examples, prompts),
            total=len(examples),
            desc=name,
        ):
            scored = jev.score_options(system_prompt, user_prompt, prompt_options)
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
        "macro_f1": compute_f1(gold, pred, bench),
        "per_class": per_class_scores(gold, pred, labels),
        "gold_counts": dict(Counter(gold)),
        "prediction_counts": dict(Counter(pred)),
        "few_shot": few_shot,
        "elapsed_s": time.perf_counter() - started,
        "output": str(trajectory_path),
    }
    (output_dir / f"jev_only_{name}{suffix}.summary.json").write_text(
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
        default=["scifact", "averitec", "healthver", "climate_fever"],
    )
    parser.add_argument("--jev_type", choices=["llm", "api"], default="llm")
    parser.add_argument("--model_path", default="/home/yxfeng/models/Qwen3/Qwen3-8B")
    parser.add_argument("--limit", type=int, default=-1, help="<=0 means the full split")
    parser.add_argument("--output_dir", default="")
    parser.add_argument("--fs_limit", type=int, default=0)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--binary", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--closed-book", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    model_name = Path(args.model_path).name
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else Path(DEFAULT_OUTPUT_DIR) / model_name
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.jev_type == 'llm':
        jev = LLM_as_Jev(model_path=args.model_path)
    else:
        jev = API_Jev()
        
    summaries: dict[str, dict] = {}
    for name in args.benchmarks:
        examples, bench, stats = iter_split(
            name,
            split="dev",
            limit=args.limit,
            binary=args.binary,
            shard=args.shard,
            num_shards=args.num_shards,
        )
        print(f"{name}: {len(examples)} claims {stats}", flush=True)
        few_shot = ""
        if args.fs_limit > 0:
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
            model_path=args.model_path,
            closed_book=args.closed_book,
            shard=args.shard,
            num_shards=args.num_shards,
        )

    suffix = f".shard{args.shard}" if args.num_shards > 1 else ""
    (output_dir / f"jev_only_summary{suffix}.json").write_text(
        json.dumps(summaries, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    lines = [
        "# LLM-as-Jev with gold evidence",
        "",
        f"Model: `{args.model_path}`",
        "",
        "| Benchmark | n | Accuracy | Macro-F1 |",
        "|---|---:|---:|---:|",
    ]
    for name, summary in summaries.items():
        lines.append(
            f"| {name} | {summary['n']} | {summary['accuracy']:.4f} | "
            f"{summary['macro_f1']:.4f} |"
        )
    for name, summary in summaries.items():
        lines.extend([
            "",
            f"## {name}",
            "",
            "| Label | Precision | Recall | F1 |",
            "|---|---:|---:|---:|",
        ])
        for label, metrics in summary["per_class"].items():
            lines.append(
                f"| {label} | {metrics['precision']:.4f} | "
                f"{metrics['recall']:.4f} | {metrics['f1']:.4f} |"
            )
    (output_dir / f"jev_only_summary{suffix}.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    print("wrote", output_dir, flush=True)


if __name__ == "__main__":
    main()
