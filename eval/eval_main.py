from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any

from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.aspects import ASPECTS, Aspect
from src.config import Passage
from src.prompts import (
    build_few_shot,
    prompt4jev_judge,
    prompt4jev_what_aspect,
    prompt4jev_whether_retrieval,
)
from src.utils import (
    Benchmark,
    compute_accuracy,
    compute_f1,
    load_benchmark,
    per_class_scores,
)
from src.workers import (
    LLM_as_Jev,
    Qwen3Reranker,
    ClaimRetriever
)

DEFAULT_OUTPUT_DIR = "/root/autodl-tmp/www27/results/eval_main"

# prompt4jev_whether_retrieval lists WHETHER_RETRIEVAL in insertion order.
RETRIEVAL_OPTION_INDEX = 0


def _as_floats(value: Any) -> list[float]:
    if hasattr(value, "detach"):
        value = value.detach().float().cpu().tolist()
    return [float(item) for item in value]


def format_passages(passages: list[Passage]) -> str | None:
    if not passages:
        return None
    parts = []
    for index, passage in enumerate(passages, start=1):
        title = f"{passage.title}\n" if passage.title else ""
        parts.append(f"[{index}] {title}{passage.text}")
    return "\n\n".join(parts)


def _choose_from_scores(
    scored: dict,
    inspected: set[str],
) -> tuple[Aspect | None, dict[str, float]]:
    probs = _as_floats(scored["probs"])
    named = {
        aspect.name: probs[index]
        for index, aspect in enumerate(ASPECTS)
    }
    ranked = sorted(ASPECTS, key=lambda aspect: named[aspect.name], reverse=True)
    for aspect in ranked:
        if aspect.name not in inspected:
            return aspect, named
    return None, named


def select_aspect(
    jev: LLM_as_Jev,
    claim: str,
    evidence: str | None,
    inspected: set[str],
) -> tuple[Aspect | None, dict[str, float]]:
    system_prompt, user_prompt, options = prompt4jev_what_aspect(claim, evidence)
    scored = jev.score_options(system_prompt, user_prompt, options)
    return _choose_from_scores(scored, inspected)


def needs_retrieval(
    jev: LLM_as_Jev,
    claim: str,
    evidence: str | None,
) -> tuple[bool, dict[str, float]]:
    system_prompt, user_prompt, options = prompt4jev_whether_retrieval(claim, evidence)
    scored = jev.score_options(system_prompt, user_prompt, options)
    probs = _as_floats(scored["probs"])
    names = ("retrieval", "skip")
    named = {name: probs[index] for index, name in enumerate(names)}
    return int(scored["best_idx"]) == RETRIEVAL_OPTION_INDEX, named


def retrieve_for_aspect(
    retriever: ClaimRetriever,
    reranker: Qwen3Reranker,
    benchmark: str,
    claim: str,
    aspect: Aspect,
    seen_ids: set[str],
    bm25_k: int,
    rerank_k: int,
) -> list[Passage]:
    candidates = retriever.search(
        benchmark,
        claim,
        bm25_k,
        exclude_ids=seen_ids,
    )
    ranked = reranker.rerank(
        claim,
        candidates,
        top_k=rerank_k,
        aspect=aspect,
    )
    selected = []
    for passage, score in ranked:
        if passage.doc_id in seen_ids:
            continue
        selected.append(replace(
            passage,
            score=float(score),
            query=claim,
            aspect=aspect.name,
        ))
        seen_ids.add(passage.doc_id)
    return selected


def run_claim(
    example: dict,
    bench: Benchmark,
    jev: LLM_as_Jev,
    reranker: Qwen3Reranker,
    retriever: ClaimRetriever,
    few_shot: str,
    bm25_k: int,
    rerank_k: int,
    search_cap: int,
) -> dict[str, Any]:
    claim = example["claim"]
    labels = list(bench.choice_dict)
    evidence_passages: list[Passage] = []
    inspected: set[str] = set()
    seen_ids: set[str] = set()
    rounds: list[dict[str, Any]] = []
    stop_reason = "budget"

    for _ in range(search_cap):
        evidence_text = format_passages(evidence_passages)
        retrieve, gate_probs = needs_retrieval(jev, claim, evidence_text)
        if not retrieve:
            rounds.append({"gate": gate_probs, "action": "skip"})
            stop_reason = "skip"
            break

        aspect, aspect_probs = select_aspect(jev, claim, evidence_text, inspected)
        if aspect is None:
            rounds.append({
                "gate": gate_probs,
                "aspect_probs": aspect_probs,
                "action": "no_aspect",
            })
            stop_reason = "no_aspect"
            break

        picked = retrieve_for_aspect(
            retriever,
            reranker,
            bench.name,
            claim,
            aspect,
            seen_ids,
            bm25_k,
            rerank_k,
        )
        inspected.add(aspect.name)
        evidence_passages.extend(picked)
        rounds.append({
            "gate": gate_probs,
            "aspect": aspect.name,
            "aspect_probs": aspect_probs,
            "n_kept": len(picked),
            "doc_ids": [passage.doc_id for passage in picked],
            "action": "retrieve",
        })
        if not picked:
            continue

    evidence_text = format_passages(evidence_passages)
    system_prompt, user_prompt, options = prompt4jev_judge(
        benchmark_cls=bench,
        claim=claim,
        few_shot=few_shot,
        evidence=evidence_text,
    )
    scored = jev.score_options(system_prompt, user_prompt, options)
    best_idx = int(scored["best_idx"])
    probs = _as_floats(scored["probs"])
    return {
        "prediction": labels[best_idx],
        "judge_probs": {label: probs[index] for index, label in enumerate(labels)},
        "scores": _as_floats(scored["scores"]),
        "stop_reason": stop_reason,
        "rounds": rounds,
        "evidence": [
            {
                "doc_id": passage.doc_id,
                "title": passage.title,
                "text": passage.text,
                "rerank_score": passage.score,
                "aspect": passage.aspect,
            }
            for passage in evidence_passages
        ],
    }


def run_one(
    examples: list[dict],
    bench: Benchmark,
    jev: LLM_as_Jev,
    reranker: Qwen3Reranker,
    retriever: ClaimRetriever,
    few_shot: str,
    output_dir: Path,
    bm25_k: int,
    rerank_k: int,
    search_cap: int,
    model_path: str,
    reranker_path: str,
) -> dict[str, Any]:
    name = bench.name
    output_dir.mkdir(parents=True, exist_ok=True)
    labels = list(bench.choice_dict)
    gold: list[str] = []
    pred: list[str] = []
    n_errors = 0
    trajectory_path = output_dir / f"eval_main_{name}.jsonl"
    started = time.perf_counter()
    with trajectory_path.open("w", encoding="utf-8") as handle:
        for example in tqdm(examples, desc=name):
            item_started = time.perf_counter()
            error = None
            result: dict[str, Any] = {}
            try:
                result = run_claim(
                    example,
                    bench,
                    jev,
                    reranker,
                    retriever,
                    few_shot,
                    bm25_k,
                    rerank_k,
                    search_cap,
                )
                gold.append(example["label"])
                pred.append(result["prediction"])
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                n_errors += 1
            row = {
                "benchmark": name,
                "id": example.get("id"),
                "claim": example["claim"],
                "gold": example.get("label"),
                "prediction": result.get("prediction"),
                "judge_probs": result.get("judge_probs", {}),
                "scores": result.get("scores", []),
                "stop_reason": result.get("stop_reason"),
                "rounds": result.get("rounds", []),
                "evidence": result.get("evidence", []),
                "latency_s": time.perf_counter() - item_started,
                "error": error,
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()

    summary = {
        "benchmark": name,
        "model": model_path,
        "reranker": reranker_path,
        "bm25_k": bm25_k,
        "rerank_k": rerank_k,
        "search_cap": search_cap,
        "n": len(gold),
        "n_errors": n_errors,
        "n_input": len(examples),
        "accuracy": compute_accuracy(gold, pred),
        "macro_f1": compute_f1(gold, pred, bench),
        "per_class": per_class_scores(gold, pred, labels),
        "gold_counts": dict(Counter(gold)),
        "prediction_counts": dict(Counter(pred)),
        "few_shot": few_shot,
        "elapsed_s": time.perf_counter() - started,
        "output": str(trajectory_path),
    }
    (output_dir / f"eval_main_{name}.summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="LLM-as-Jev with a retrieve-or-judge loop",
    )
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        default=["averitec", "fever", "scifact", "healthver"],
    )
    parser.add_argument("--model", default="/root/autodl-tmp/models/Qwen3-8B")
    parser.add_argument("--reranker", default="/root/autodl-tmp/models/Qwen3-Reranker-4B")
    parser.add_argument("--limit", type=int, default=-1, help="<=0 means the full split")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--fs_limit", type=int, default=0)
    parser.add_argument("--bm25-k", type=int, default=500)
    parser.add_argument("--rerank-k", type=int, default=2)
    parser.add_argument("--search-cap", type=int, default=3)
    parser.add_argument("--rerank-batch-size", type=int, default=8)
    args = parser.parse_args()
    if args.bm25_k <= 0 or args.rerank_k <= 0 or args.search_cap <= 0:
        raise ValueError("bm25-k, rerank-k, and search-cap must be positive")

    model_name = Path(args.model).name
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else Path(DEFAULT_OUTPUT_DIR) / model_name
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    jev = LLM_as_Jev(model_path=args.model)
    reranker = Qwen3Reranker(
        model_path=args.reranker,
        batch_size=args.rerank_batch_size,
    )
    retriever = ClaimRetriever()
    summaries: dict[str, dict] = {}
    try:
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
                reranker=reranker,
                retriever=retriever,
                few_shot=few_shot,
                output_dir=output_dir,
                bm25_k=args.bm25_k,
                rerank_k=args.rerank_k,
                search_cap=args.search_cap,
                model_path=args.model,
                reranker_path=args.reranker,
            )
    finally:
        retriever.close()

    (output_dir / "eval_main_summary.json").write_text(
        json.dumps(summaries, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    lines = [
        "# LLM-as-Jev retrieve-or-judge loop",
        "",
        f"Model: `{args.model}`",
        "",
        f"Reranker: `{args.reranker}`",
        "",
        "| Benchmark | n | Accuracy | Macro-F1 | Errors |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, summary in summaries.items():
        lines.append(
            f"| {name} | {summary['n']} | {summary['accuracy']:.4f} | "
            f"{summary['macro_f1']:.4f} | {summary['n_errors']} |"
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
    (output_dir / "eval_main_summary.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    print("wrote", output_dir, flush=True)


if __name__ == "__main__":
    main()
