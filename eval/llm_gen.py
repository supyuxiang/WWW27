from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.prompts import (
    prompt4llm_judge,
    build_few_shot,
)
from src.paths import RESULTS
from src.utils import (
    Benchmark,
    compute_accuracy,
    compute_f1,
    per_class_scores,
    iter_split,
    parse_generation
)
from src.workers.llm_generator import APIGenerator, VanillaGenerator, VLLMGenerator


def run_one(
    examples: list[dict],
    bench: Benchmark,
    generator: VLLMGenerator | APIGenerator | VanillaGenerator,
    few_shot: str,
    output_dir: Path,
    is_cot: bool = False,
    closed_book: bool = True,
    shard: int = 0,
    num_shards: int = 1,
) -> dict[str,Any]:
    name = bench.name
    output_dir.mkdir(parents=True, exist_ok=True)
    if not examples:
        return {
            "benchmark": name,
            "n": 0,
            "invalid": 0,
            "accuracy": 0.0,
            "macro_f1": 0.0,
            "per_class": {},
            "gold_counts": {},
            "prediction_counts": {},
            "few_shot": few_shot,
            "output": "",
        }
    labels = list(bench.choice_dict)

    prompts: list[tuple[str, str, list[str]]] = [
        prompt4llm_judge(
            benchmark_cls=bench,
            claim=item["claim"],
            few_shot=few_shot,
            evidence=None if closed_book else item.get("evidence"),
            is_cot=is_cot
        )
        for item in examples
    ]
    user_prompts = [prompt[1] for prompt in prompts]
    generations = generator.generate_batch(user_prompts)

    gold: list[str] = []
    pred: list[str] = []
    invalid = 0
    suffix = f".shard{shard}" if num_shards > 1 else ""
    trajectory_path = output_dir / f"llm_gen_only_{name}{suffix}.jsonl"
    with trajectory_path.open("w", encoding="utf-8") as handle:
        for item, generation in zip(examples, generations):
            prediction = parse_generation(generation, labels)
            if prediction is None:
                prediction = "invalid"
                invalid += 1
            gold.append(item["label"])
            pred.append(prediction)
            handle.write(json.dumps({
                "benchmark": name,
                "id": item.get("id"),
                "claim": item["claim"],
                "gold": item["label"],
                "evidence": item["evidence"],
                "generation": generation,
                "prediction": prediction,
            }, ensure_ascii=False) + "\n")

    class_scores = per_class_scores(gold, pred, labels)
    summary = {
        "benchmark": name,
        "n": len(gold),
        "invalid": invalid,
        "accuracy": compute_accuracy(gold, pred),
        "macro_f1": compute_f1(gold, pred, bench),
        "per_class": class_scores,
        "gold_counts": dict(Counter(gold)),
        "prediction_counts": dict(Counter(pred)),
        "few_shot": few_shot,
        "output": str(trajectory_path),
    }
    (output_dir / f"llm_gen_only_{name}{suffix}.summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM generation with gold evidence")
    parser.add_argument("--benchmarks", nargs="+", default=["scifact", "averitec", "healthver", "climate_fever"])
    parser.add_argument("--model", default="/home/yxfeng/models/Qwen3/Qwen3-8B")
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--output_dir", "--output-dir", default=str(RESULTS / "llm_gen"))
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.92)
    parser.add_argument("--max_model_len", type=int, default=8192)
    parser.add_argument("--fs_limit", type=int, default=0)
    parser.add_argument("--generator", choices=["vllm", "api", "hf"], default="vllm")
    parser.add_argument(
        "--enable_thinking",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument('--is_cot', action='store_true')
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--binary", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--closed-book", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.generator == 'vllm':
        generator = VLLMGenerator(
            model_path=args.model,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
            max_tokens=args.max_tokens,
            enable_thinking=args.enable_thinking,
        )
    elif args.generator == "hf":
        generator = VanillaGenerator(
            model_path=args.model,
            max_new_tokens=args.max_tokens,
            enable_thinking=args.enable_thinking,
        )
    elif args.generator == 'api':
        generator = APIGenerator(
            base_url="https://api.deepseek.com",
            api_key="",
            model="deepseek-flash",
            max_tokens=args.max_tokens,
            enable_thinking=True,
            reasoning_effort="high",
            max_retries=3,
        )
    else:
        raise ValueError(f"unknown generator {args.generator!r}")

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
            generator=generator,
            few_shot=few_shot,
            output_dir=output_dir,
            is_cot=args.is_cot,
            closed_book=args.closed_book,
            shard=args.shard,
            num_shards=args.num_shards,
        )

    suffix = f".shard{args.shard}" if args.num_shards > 1 else ""
    (output_dir / f"llm_gen_only_summary{suffix}.json").write_text(
        json.dumps(summaries, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    lines = [
        "# LLM generation with gold evidence",
        "",
        "| Benchmark | n | Accuracy | Macro-F1 | Invalid |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, summary in summaries.items():
        lines.append(
            f"| {name} | {summary['n']} | {summary['accuracy']:.4f} | "
            f"{summary['macro_f1']:.4f} | {summary['invalid']} |"
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
    (output_dir / f"llm_gen_only_summary{suffix}.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    print("wrote", output_dir, flush=True)

    import gc
    import torch
    
    del generator
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
