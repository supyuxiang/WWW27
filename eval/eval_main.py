"""Doffy: dense claim pool, aspect-conditioned query, then rerank.

Gold evidence is never shown to the judge. Retrieval is the precomputed
Qwen3-Embedding-4B pool, not BM25. Judge-1 is re-scored after every
retrieval. One retrieval call keeps the highest unchecked aspect whose
p(Check) is at least tau, or the single best aspect once if none pass.
"""
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
from src.paths import RESULTS
from src.prompts import (
    build_few_shot,
    prompt4aspect_check,
    prompt4aspect_query,
    prompt4jev_judge,
    prompt4jev_whether_retrieval,
)
from src.utils import (
    Benchmark,
    compute_accuracy,
    compute_f1,
    iter_split,
    load_done_ids,
    per_class_scores,
)
from src.workers.jev import LLM_as_Jev
from src.workers.reranker import Qwen3Reranker
from src.workers.retriever import ClaimRetriever

DEFAULT_OUTPUT_DIR = str(RESULTS / "doffy")
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


def needs_retrieval(
    jev: LLM_as_Jev,
    claim: str,
    evidence: str | None,
) -> tuple[bool, dict[str, float]]:
    system_prompt, user_prompt, options = prompt4jev_whether_retrieval(claim, evidence)
    scored = jev.score_options(system_prompt, user_prompt, options)
    probs = _as_floats(scored["probs"])
    named = {"retrieval": probs[0], "skip": probs[1]}
    return int(scored["best_idx"]) == RETRIEVAL_OPTION_INDEX, named


def select_aspect(
    jev: LLM_as_Jev,
    claim: str,
    evidence: str | None,
    inspected: set[str],
    tau_aspect: float,
    allow_fallback: bool,
) -> tuple[Aspect | None, dict[str, float], bool]:
    scored: list[tuple[Aspect, float]] = []
    for aspect in ASPECTS:
        if aspect.name in inspected:
            continue
        system_prompt, user_prompt, options = prompt4aspect_check(claim, aspect, evidence)
        result = jev.score_options(system_prompt, user_prompt, options)
        scored.append((aspect, _as_floats(result["probs"])[0]))
    scored.sort(key=lambda item: item[1], reverse=True)
    named = {aspect.name: prob for aspect, prob in scored}
    passed = [aspect for aspect, prob in scored if prob >= tau_aspect]
    if passed:
        return passed[0], named, allow_fallback
    if allow_fallback and scored:
        return scored[0][0], named, False
    return None, named, allow_fallback


def make_query(
    jev: LLM_as_Jev,
    claim: str,
    aspect: Aspect,
    evidence: str | None,
) -> str:
    system_prompt, user_prompt = prompt4aspect_query(claim, aspect, evidence)
    query = jev.generate(system_prompt, user_prompt, max_new_tokens=32)
    return query or claim


def retrieve_from_pool(
    reranker: Qwen3Reranker,
    pool: list[Passage],
    query: str,
    aspect: Aspect,
    seen_ids: set[str],
    rerank_k: int,
) -> list[Passage]:
    candidates = [passage for passage in pool if passage.doc_id not in seen_ids]
    ranked = reranker.rerank(query, candidates, top_k=rerank_k, aspect=aspect)
    selected = []
    for passage, score in ranked:
        selected.append(replace(
            passage,
            score=float(score),
            query=query,
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
    pool_k: int,
    rerank_k: int,
    search_cap: int,
    alpha: int,
    tau_aspect: float,
) -> dict[str, Any]:
    claim = example["claim"]
    labels = list(bench.choice_dict)
    pool = retriever.search(
        bench.name,
        claim,
        pool_k,
        claim_id=example.get("id"),
    )
    evidence_passages: list[Passage] = []
    inspected: set[str] = set()
    seen_ids: set[str] = set()
    rounds: list[dict[str, Any]] = []
    calls = 0
    stop_reason = "budget"
    allow_fallback = True

    while calls < search_cap and len(inspected) < alpha:
        evidence_text = format_passages(evidence_passages)
        retrieve, gate_probs = needs_retrieval(jev, claim, evidence_text)
        if not retrieve:
            rounds.append({"gate": gate_probs, "action": "skip"})
            stop_reason = "skip"
            break
        aspect, aspect_probs, allow_fallback = select_aspect(
            jev,
            claim,
            evidence_text,
            inspected,
            tau_aspect,
            allow_fallback,
        )
        if aspect is None:
            rounds.append({
                "gate": gate_probs,
                "aspect_probs": aspect_probs,
                "action": "no_aspect",
            })
            stop_reason = "no_aspect"
            break
        query = make_query(jev, claim, aspect, evidence_text)
        picked = retrieve_from_pool(
            reranker,
            pool,
            query,
            aspect,
            seen_ids,
            rerank_k,
        )
        inspected.add(aspect.name)
        evidence_passages.extend(picked)
        calls += 1
        rounds.append({
            "gate": gate_probs,
            "aspect": aspect.name,
            "aspect_probs": aspect_probs,
            "query": query,
            "n_kept": len(picked),
            "doc_ids": [passage.doc_id for passage in picked],
            "action": "retrieve",
        })

    evidence_text = format_passages(evidence_passages)
    system_prompt, user_prompt, _options = prompt4jev_judge(
        benchmark_cls=bench,
        claim=claim,
        few_shot=few_shot or None,
        evidence=evidence_text,
    )
    scored = jev.score_options(system_prompt, user_prompt, _options)
    best_idx = int(scored["best_idx"])
    probs = _as_floats(scored["probs"])
    return {
        "prediction": labels[best_idx],
        "judge_probs": {label: probs[index] for index, label in enumerate(labels)},
        "scores": _as_floats(scored["scores"]),
        "stop_reason": stop_reason,
        "n_calls": calls,
        "rounds": rounds,
        "evidence": [
            {
                "doc_id": passage.doc_id,
                "title": passage.title,
                "text": passage.text,
                "rerank_score": passage.score,
                "aspect": passage.aspect,
                "query": passage.query,
            }
            for passage in evidence_passages
        ],
    }


def summarize_file(path: Path, bench: Benchmark, meta: dict) -> dict[str, Any]:
    gold: list[str] = []
    pred: list[str] = []
    calls: list[int] = []
    latencies: list[float] = []
    n_errors = 0
    n_rows = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            n_rows += 1
            if row.get("error"):
                n_errors += 1
            if row.get("prediction") is None or row.get("gold") is None:
                continue
            gold.append(row["gold"])
            pred.append(row["prediction"])
            calls.append(int(row.get("n_calls") or 0))
            if row.get("latency_s") is not None:
                latencies.append(float(row["latency_s"]))
    labels = list(bench.choice_dict)
    summary = {
        **meta,
        "n_rows": n_rows,
        "n": len(gold),
        "n_errors": n_errors,
        "accuracy": compute_accuracy(gold, pred),
        "macro_f1": compute_f1(gold, pred, bench),
        "per_class": per_class_scores(gold, pred, labels),
        "gold_counts": dict(Counter(gold)),
        "prediction_counts": dict(Counter(pred)),
        "mean_calls": (sum(calls) / len(calls)) if calls else 0.0,
        "mean_latency_s": (sum(latencies) / len(latencies)) if latencies else 0.0,
        "output": str(path),
    }
    return summary


def run_one(
    examples: list[dict],
    bench: Benchmark,
    jev: LLM_as_Jev,
    reranker: Qwen3Reranker,
    retriever: ClaimRetriever,
    few_shot: str,
    output_dir: Path,
    pool_k: int,
    rerank_k: int,
    search_cap: int,
    alpha: int,
    tau_aspect: float,
    model_path: str,
    reranker_path: str,
    stats: dict,
) -> dict[str, Any]:
    name = bench.name
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = f".shard{stats['shard']}" if stats["num_shards"] > 1 else ""
    trajectory_path = output_dir / f"doffy_{name}{suffix}.jsonl"
    done = load_done_ids(trajectory_path)
    pending = [example for example in examples if str(example.get("id")) not in done]
    print(f"{name} shard {stats['shard']}: {len(pending)} pending, {len(done)} done", flush=True)
    with trajectory_path.open("a", encoding="utf-8") as handle:
        for example in tqdm(pending, desc=f"{name}:{stats['shard']}"):
            started = time.perf_counter()
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
                    pool_k,
                    rerank_k,
                    search_cap,
                    alpha,
                    tau_aspect,
                )
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
            row = {
                "benchmark": name,
                "id": example.get("id"),
                "claim": example["claim"],
                "gold": example.get("label"),
                "gold_raw": example.get("label_raw"),
                "prediction": result.get("prediction"),
                "judge_probs": result.get("judge_probs", {}),
                "scores": result.get("scores", []),
                "stop_reason": result.get("stop_reason"),
                "n_calls": result.get("n_calls", 0),
                "rounds": result.get("rounds", []),
                "evidence": result.get("evidence", []),
                "latency_s": time.perf_counter() - started,
                "error": error,
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
    meta = {
        "benchmark": name,
        "model": model_path,
        "reranker": reranker_path,
        "retriever": "qwen3-embedding-4b",
        "pool_k": pool_k,
        "rerank_k": rerank_k,
        "search_cap": search_cap,
        "alpha": alpha,
        "tau_aspect": tau_aspect,
        "few_shot": bool(few_shot),
        **stats,
    }
    summary = summarize_file(trajectory_path, bench, meta)
    summary_path = output_dir / f"doffy_{name}{suffix}.summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({
        "benchmark": name,
        "shard": stats["shard"],
        "n": summary["n"],
        "accuracy": summary["accuracy"],
        "mean_calls": summary["mean_calls"],
        "mean_latency_s": summary["mean_latency_s"],
        "n_errors": summary["n_errors"],
    }, indent=2), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Doffy dense retrieve-or-judge loop")
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        default=["scifact", "averitec", "healthver", "climate_fever"],
    )
    parser.add_argument("--model", default="/home/yxfeng/models/Qwen3/Qwen3-8B")
    parser.add_argument("--reranker", default="/home/yxfeng/models/Qwen3/Qwen3-Reranker-4B")
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--fs-limit", type=int, default=0)
    parser.add_argument("--bm25-k", type=int, default=500, help="dense pool size")
    parser.add_argument("--rerank-k", type=int, default=2)
    parser.add_argument("--search-cap", type=int, default=3)
    parser.add_argument("--alpha", type=int, default=3)
    parser.add_argument("--tau-aspect", type=float, default=0.5)
    parser.add_argument("--rerank-batch-size", type=int, default=16)
    parser.add_argument("--rerank-max-length", type=int, default=512)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--binary", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else Path(DEFAULT_OUTPUT_DIR) / Path(args.model).name
    output_dir.mkdir(parents=True, exist_ok=True)
    jev = LLM_as_Jev(model_path=args.model)
    reranker = Qwen3Reranker(
        model_path=args.reranker,
        batch_size=args.rerank_batch_size,
        max_length=args.rerank_max_length,
    )
    retriever = ClaimRetriever()
    summaries: dict[str, dict] = {}
    try:
        for name in args.benchmarks:
            examples, bench, stats = iter_split(
                name,
                split="dev",
                limit=args.limit,
                binary=args.binary,
                shard=args.shard,
                num_shards=args.num_shards,
            )
            few_shot = ""
            if args.fs_limit > 0:
                few_shot = build_few_shot(bench, limit=args.fs_limit, split="train")
            summaries[name] = run_one(
                examples=examples,
                bench=bench,
                jev=jev,
                reranker=reranker,
                retriever=retriever,
                few_shot=few_shot,
                output_dir=output_dir,
                pool_k=args.bm25_k,
                rerank_k=args.rerank_k,
                search_cap=args.search_cap,
                alpha=args.alpha,
                tau_aspect=args.tau_aspect,
                model_path=args.model,
                reranker_path=args.reranker,
                stats=stats,
            )
    finally:
        retriever.close()
    suffix = f".shard{args.shard}" if args.num_shards > 1 else ""
    (output_dir / f"doffy_summary{suffix}.json").write_text(
        json.dumps(summaries, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print("wrote", output_dir, flush=True)


if __name__ == "__main__":
    main()
