from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import Passage
from src.prompts import build_few_shot, prompt4llm_judge
from src.utils import (
    Benchmark,
    compute_accuracy,
    compute_f1,
    load_benchmark,
    parse_generation,
    per_class_scores,
)
from src.workers import (
    APIGenerator,
    ClaimRetriever,
    Qwen3Reranker,
    VLLMGenerator,
)

DEFAULT_OUTPUT_DIR = "/root/autodl-tmp/www27/results/llm_gen_retrieval_once"


def retrieve_once(
    retriever: ClaimRetriever,
    reranker: Qwen3Reranker,
    benchmark: str,
    claim: str,
    bm25_k: int,
    rerank_k: int,
) -> tuple[list[Passage], list[Passage]]:
    candidates = retriever.search(benchmark, claim, bm25_k)
    ranked = reranker.rerank(claim, candidates, top_k=rerank_k)
    selected = [
        replace(passage, score=float(score), query=claim)
        for passage, score in ranked
    ]
    return candidates, selected


def format_passages(passages: list[Passage]) -> str:
    if not passages:
        return "none"
    parts = []
    for index, passage in enumerate(passages, start=1):
        title = f"{passage.title}\n" if passage.title else ""
        parts.append(f"[{index}] {title}{passage.text}")
    return "\n\n".join(parts)


def _passage_record(passage: Passage) -> dict[str, Any]:
    return {
        "doc_id": passage.doc_id,
        "title": passage.title,
        "text": passage.text,
        "rerank_score": passage.score,
    }


def retrieve_examples(
    examples: list[dict],
    bench: Benchmark,
    reranker: Qwen3Reranker,
    retriever: ClaimRetriever,
    few_shot: str,
    bm25_k: int,
    rerank_k: int,
    is_cot: bool,
) -> list[dict[str, Any]]:
    """Retrieve once per claim and build the generation prompt."""
    name = bench.name
    rows: list[dict[str, Any]] = []
    for example in tqdm(examples, desc=f"{name} retrieve"):
        started = time.perf_counter()
        candidates: list[Passage] = []
        selected: list[Passage] = []
        user_prompt = None
        error = None
        try:
            candidates, selected = retrieve_once(
                retriever,
                reranker,
                name,
                example["claim"],
                bm25_k,
                rerank_k,
            )
            _, user_prompt, _ = prompt4llm_judge(
                benchmark_cls=bench,
                claim=example["claim"],
                few_shot=few_shot,
                evidence=format_passages(selected),
                is_cot=is_cot,
            )
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        rows.append({
            "id": example.get("id"),
            "claim": example["claim"],
            "gold": example.get("label"),
            "user_prompt": user_prompt,
            "n_bm25": len(candidates),
            "evidence": [_passage_record(passage) for passage in selected],
            "retrieval_s": time.perf_counter() - started,
            "error": error,
        })
    return rows


def score_generations(
    rows: list[dict[str, Any]],
    generations: list[str],
    bench: Benchmark,
    output_dir: Path,
    model_path: str,
    reranker_path: str,
    few_shot: str,
    bm25_k: int,
    rerank_k: int,
    elapsed_s: float,
) -> dict[str, Any]:
    name = bench.name
    labels = list(bench.choice_dict)
    output_dir.mkdir(parents=True, exist_ok=True)
    trajectory_path = output_dir / f"llm_gen_retrieval_once_{name}.jsonl"
    expected = sum(row["error"] is None for row in rows)
    if len(generations) != expected:
        raise RuntimeError(
            f"{name}: expected {expected} generations, got {len(generations)}"
        )
    gold: list[str] = []
    pred: list[str] = []
    invalid = 0
    n_errors = 0
    cursor = 0
    with trajectory_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            generation = None
            prediction = None
            if row["error"] is None:
                generation = generations[cursor]
                cursor += 1
                prediction = parse_generation(generation, labels)
                if prediction is None:
                    prediction = "invalid"
                    invalid += 1
                gold.append(row["gold"])
                pred.append(prediction)
            else:
                n_errors += 1
            record = {
                "benchmark": name,
                "id": row["id"],
                "claim": row["claim"],
                "gold": row["gold"],
                "prediction": prediction,
                "generation": generation,
                "n_bm25": row["n_bm25"],
                "evidence": row["evidence"],
                "retrieval_s": row["retrieval_s"],
                "retrieval": f"bm25_top{bm25_k}_rerank_top{rerank_k}",
                "error": row["error"],
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    summary = {
        "benchmark": name,
        "model": model_path,
        "reranker": reranker_path,
        "bm25_k": bm25_k,
        "rerank_k": rerank_k,
        "n": len(gold),
        "invalid": invalid,
        "n_errors": n_errors,
        "n_input": len(rows),
        "accuracy": compute_accuracy(gold, pred),
        "macro_f1": compute_f1(gold, pred, bench),
        "per_class": per_class_scores(gold, pred, labels),
        "gold_counts": dict(Counter(gold)),
        "prediction_counts": dict(Counter(pred)),
        "few_shot": few_shot,
        "elapsed_s": elapsed_s,
        "output": str(trajectory_path),
    }
    (output_dir / f"llm_gen_retrieval_once_{name}.summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    return summary


def _build_generator(args: argparse.Namespace) -> VLLMGenerator | APIGenerator:
    if args.generator == "vllm":
        return VLLMGenerator(
            model_path=args.model,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
            max_tokens=args.max_tokens,
            enable_thinking=args.enable_thinking,
        )
    if args.generator == "api":
        return APIGenerator(
            base_url="https://api.deepseek.com",
            model="deepseek-flash",
            max_tokens=args.max_tokens,
            enable_thinking=True,
            reasoning_effort="high",
            max_retries=3,
        )
    raise ValueError(f"unknown generator {args.generator!r}")


def _write_report(
    output_dir: Path,
    summaries: dict[str, dict],
    model: str,
    reranker: str,
    bm25_k: int,
    rerank_k: int,
) -> None:
    (output_dir / "llm_gen_retrieval_once_summary.json").write_text(
        json.dumps(summaries, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    lines = [
        "# LLM generation with one BM25 retrieval and reranking",
        "",
        f"Model: `{model}`",
        "",
        f"Reranker: `{reranker}`",
        "",
        f"Retrieval: BM25 top {bm25_k}, rerank top {rerank_k}",
        "",
        "| Benchmark | n | Accuracy | Macro-F1 | Invalid | Errors |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, summary in summaries.items():
        lines.append(
            f"| {name} | {summary['n']} | {summary['accuracy']:.4f} | "
            f"{summary['macro_f1']:.4f} | {summary['invalid']} | "
            f"{summary['n_errors']} |"
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
    (output_dir / "llm_gen_retrieval_once_summary.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="LLM generation with one BM25 retrieval and reranking",
    )
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        default=["averitec", "fever", "scifact", "healthver"],
    )
    parser.add_argument("--model", default="/root/autodl-tmp/models/Qwen3-8B")
    parser.add_argument("--reranker", default="/root/autodl-tmp/models/Qwen3-Reranker-4B")
    parser.add_argument("--limit", type=int, default=-1, help="<=0 means the full split")
    parser.add_argument("--output-dir", "--output_dir", dest="output_dir", default="")
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.92)
    parser.add_argument("--max_model_len", type=int, default=8192)
    parser.add_argument("--fs_limit", type=int, default=0)
    parser.add_argument("--generator", choices=["vllm", "api"], default="vllm")
    parser.add_argument(
        "--enable_thinking",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--is_cot", action="store_true")
    parser.add_argument("--bm25-k", type=int, default=500)
    parser.add_argument("--rerank-k", type=int, default=2)
    parser.add_argument("--rerank-batch-size", type=int, default=8)
    args = parser.parse_args()
    if args.bm25_k <= 0:
        raise ValueError(f"bm25-k must be positive, got {args.bm25_k}")
    if args.rerank_k <= 0:
        raise ValueError(f"rerank-k must be positive, got {args.rerank_k}")

    model_name = (
        Path(args.model).name if args.generator == "vllm" else "deepseek-flash"
    )
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else Path(DEFAULT_OUTPUT_DIR) / model_name
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    # Retrieve while the reranker owns the GPU, then drop it before vLLM starts.
    reranker = Qwen3Reranker(
        model_path=args.reranker,
        batch_size=args.rerank_batch_size,
    )
    retriever = ClaimRetriever()
    prepared: dict[str, tuple[Benchmark, str, list[dict[str, Any]], float]] = {}
    try:
        for name in args.benchmarks:
            examples, bench = load_benchmark(name, limit=args.limit)
            print(f"{name}: {len(examples)} claims", flush=True)
            few_shot = build_few_shot(
                bench_cls=bench,
                limit=args.fs_limit,
                split="train",
            )
            bench_started = time.perf_counter()
            rows = retrieve_examples(
                examples=examples,
                bench=bench,
                reranker=reranker,
                retriever=retriever,
                few_shot=few_shot,
                bm25_k=args.bm25_k,
                rerank_k=args.rerank_k,
                is_cot=args.is_cot,
            )
            prepared[name] = (
                bench,
                few_shot,
                rows,
                time.perf_counter() - bench_started,
            )
    finally:
        retriever.close()
        del reranker
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    generator = _build_generator(args)
    summaries: dict[str, dict] = {}
    try:
        for name, (bench, few_shot, rows, retrieve_elapsed) in prepared.items():
            prompts = [
                row["user_prompt"] for row in rows if row["error"] is None
            ]
            print(f"{name}: generating {len(prompts)} prompts", flush=True)
            generate_started = time.perf_counter()
            generations = generator.generate_batch(prompts) if prompts else []
            summaries[name] = score_generations(
                rows=rows,
                generations=generations,
                bench=bench,
                output_dir=output_dir,
                model_path=args.model,
                reranker_path=args.reranker,
                few_shot=few_shot,
                bm25_k=args.bm25_k,
                rerank_k=args.rerank_k,
                elapsed_s=retrieve_elapsed + time.perf_counter() - generate_started,
            )
    finally:
        del generator
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    _write_report(
        output_dir,
        summaries,
        model=args.model,
        reranker=args.reranker,
        bm25_k=args.bm25_k,
        rerank_k=args.rerank_k,
    )
    print("wrote", output_dir, flush=True)


if __name__ == "__main__":
    main()
