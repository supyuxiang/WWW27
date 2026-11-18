"""Doffy: dense claim pool, aspect-conditioned query, then rerank.

Gold evidence is never shown to the judge. Retrieval is the precomputed
Qwen3-Embedding-4B pool, not BM25. Judge-1 is re-scored after every
retrieval. One retrieval call keeps the highest unchecked aspect whose
p(Check) is at least tau, or the single best aspect once if none pass.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time

import torch
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
    prompt4evidence_vote,
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


def _claim_rng(variant: str, claim_id: str | None) -> random.Random:
    digest = hashlib.md5(f"{variant}:{claim_id}".encode()).hexdigest()
    return random.Random(int(digest[:8], 16))


def _match_label(text: str, labels: list[str]) -> str | None:
    folded = text.casefold()
    hits = []
    for label in labels:
        index = folded.rfind(label.casefold())
        if index >= 0:
            hits.append((index, label))
    if not hits:
        return None
    hits.sort()
    return hits[-1][1]


def _gen_tokens(jev: LLM_as_Jev, text: str) -> int:
    return len(jev.tokenizer.encode(text, add_special_tokens=False))


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


def _rerank_keep(
    reranker: Qwen3Reranker,
    pool: list[Passage],
    query: str,
    aspect: Aspect | None,
    seen_ids: set[str],
    rerank_k: int,
    usage: dict[str, int],
) -> list[Passage]:
    candidates = [passage for passage in pool if passage.doc_id not in seen_ids]
    usage["rerank"] += len(candidates)
    if aspect is None:
        ranked = reranker.rerank(query, candidates, top_k=rerank_k)
        aspect_name = ""
    else:
        ranked = reranker.rerank(query, candidates, top_k=rerank_k, aspect=aspect)
        aspect_name = aspect.name
    selected = []
    for passage, score in ranked:
        selected.append(replace(
            passage,
            score=float(score),
            query=query,
            aspect=aspect_name,
        ))
        seen_ids.add(passage.doc_id)
    return selected


EVIDENCE_KEEP = 4
INSUFFICIENT_TAU = 0.5


def _move_worker(worker: Any, device: str) -> None:
    target = torch.device(device)
    worker.model.to(target)
    worker.device = target
    if target.type == "cpu":
        torch.cuda.empty_cache()


def run_select_joint(
    example: dict,
    bench: Benchmark,
    jev: LLM_as_Jev,
    reranker: Qwen3Reranker,
    retriever: ClaimRetriever,
    few_shot: str,
    pool_k: int,
    rerank_k: int,
    tau_aspect: float,
) -> dict[str, Any]:
    """Retrieve every aspect above tau, drop non-decisive passages, one joint verdict."""
    claim = example["claim"]
    labels = list(bench.choice_dict)
    pool = retriever.search(
        bench.name,
        claim,
        pool_k,
        claim_id=example.get("id"),
    )
    usage = {"llm": 0, "gen_tokens": 0, "rerank": 0}
    swap = reranker.device.type == "cpu" and torch.cuda.is_available()
    scored: list[tuple[Aspect, float]] = []
    for aspect in ASPECTS:
        system_prompt, user_prompt, options = prompt4aspect_check(claim, aspect, None)
        result = jev.score_options(system_prompt, user_prompt, options)
        usage["llm"] += 1
        scored.append((aspect, _as_floats(result["probs"])[0]))
    scored.sort(key=lambda item: item[1], reverse=True)
    aspect_probs = {aspect.name: prob for aspect, prob in scored}
    selected = [aspect for aspect, prob in scored if prob >= tau_aspect]
    if not selected and scored:
        selected = [scored[0][0]]
    selected_names = {aspect.name for aspect in selected}

    queries: dict[str, str] = {}
    for aspect, _prob in scored:
        if aspect.name not in selected_names:
            continue
        query = make_query(jev, claim, aspect, None)
        usage["gen_tokens"] += _gen_tokens(jev, query)
        queries[aspect.name] = query

    best: dict[str, dict[str, Any]] = {}
    rounds: list[dict[str, Any]] = []
    try:
        if swap:
            _move_worker(jev, "cpu")
            _move_worker(reranker, "cuda")
        for aspect, prob in scored:
            if aspect.name not in selected_names:
                continue
            query = queries[aspect.name]
            usage["rerank"] += len(pool)
            ranked = reranker.rerank(query, pool, top_k=rerank_k, aspect=aspect)
            kept_ids = []
            for passage, score in ranked:
                weight = float(prob) * float(score)
                kept_ids.append(passage.doc_id)
                current = best.get(passage.doc_id)
                if current is None or weight > current["weight"]:
                    best[passage.doc_id] = {
                        "passage": replace(
                            passage,
                            score=float(score),
                            query=query,
                            aspect=aspect.name,
                        ),
                        "weight": weight,
                        "rerank": float(score),
                        "aspect": aspect.name,
                        "aspect_prob": float(prob),
                        "query": query,
                    }
            rounds.append({
                "aspect": aspect.name,
                "aspect_prob": float(prob),
                "query": query,
                "doc_ids": kept_ids,
                "action": "retrieve",
            })
    finally:
        if swap:
            if reranker.device.type == "cuda":
                _move_worker(reranker, "cpu")
            if jev.device.type != "cuda":
                _move_worker(jev, "cuda")

    ranked_docs = sorted(best.values(), key=lambda item: item["weight"], reverse=True)
    ranked_docs = ranked_docs[:EVIDENCE_KEEP]
    decisive: list[dict[str, Any]] = []
    for item in ranked_docs:
        evidence_text = format_passages([item["passage"]]) or ""
        system_prompt, user_prompt, options = prompt4evidence_vote(claim, evidence_text, labels)
        result = jev.score_options(system_prompt, user_prompt, options)
        usage["llm"] += 1
        probs = _as_floats(result["probs"])
        item["vote_probs"] = {
            **{label: probs[index] for index, label in enumerate(labels)},
            "Insufficient": probs[len(labels)],
        }
        if probs[len(labels)] < INSUFFICIENT_TAU:
            decisive.append(item)
    if decisive:
        chosen = decisive
        stop_reason = "joint"
    elif ranked_docs:
        chosen = [ranked_docs[0]]
        stop_reason = "fallback_top"
    else:
        chosen = []
        stop_reason = "closed_book"

    evidence_passages = [item["passage"] for item in chosen]
    evidence_text = format_passages(evidence_passages)
    system_prompt, user_prompt, options = prompt4jev_judge(
        benchmark_cls=bench,
        claim=claim,
        few_shot=few_shot or None,
        evidence=evidence_text,
    )
    judged = jev.score_options(system_prompt, user_prompt, options)
    usage["llm"] += 1
    judge_probs_list = _as_floats(judged["probs"])
    prediction = labels[int(judged["best_idx"])]
    return {
        "prediction": prediction,
        "judge_probs": {label: judge_probs_list[index] for index, label in enumerate(labels)},
        "scores": _as_floats(judged["scores"]),
        "stop_reason": stop_reason,
        "n_calls": len(rounds),
        "n_llm_fwd": usage["llm"],
        "n_gen_tokens": usage["gen_tokens"],
        "n_rerank": usage["rerank"],
        "rounds": [{
            "aspect_probs": aspect_probs,
            "n_selected": len(selected),
            "n_candidates": len(ranked_docs),
            "n_decisive": len(decisive),
            "votes": [
                {
                    "doc_id": item["passage"].doc_id,
                    "aspect": item["aspect"],
                    "weight": item["weight"],
                    "vote_probs": item.get("vote_probs"),
                    "kept": item in chosen,
                }
                for item in ranked_docs
            ],
        }, *rounds],
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
    variant: str = "doffy",
) -> dict[str, Any]:
    if variant == "select_joint":
        return run_select_joint(
            example,
            bench,
            jev,
            reranker,
            retriever,
            few_shot,
            pool_k,
            rerank_k,
            tau_aspect,
        )
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
    usage = {"llm": 0, "gen_tokens": 0, "rerank": 0}
    rng = _claim_rng(variant, example.get("id"))
    cap = len(ASPECTS) if variant == "always_all" else search_cap

    def remember_query(text: str) -> str:
        usage["gen_tokens"] += _gen_tokens(jev, text)
        return text

    while calls < cap and (variant == "always_all" or len(inspected) < alpha or variant == "holistic"):
        if variant != "always_all" and variant != "holistic" and len(inspected) >= alpha:
            stop_reason = "aspect_cap"
            break
        evidence_text = format_passages(evidence_passages)
        gate_probs: dict[str, float] = {}
        if variant in {"no_gate", "always_all"}:
            retrieve = True
        elif variant == "random_gate":
            retrieve = rng.random() < 0.5
            gate_probs = {"random_retrieve": 1.0 if retrieve else 0.0}
        elif variant == "doffy_gen":
            system = "Choose whether more external evidence is required. Output only Retrieve or Skip."
            user = f"Claim: {claim}"
            if evidence_text:
                user += f"\nCurrent evidence:\n{evidence_text}"
            text = remember_query(jev.generate(system, user, max_new_tokens=8))
            decision = _match_label(text, ["Retrieve", "Skip"]) or "Skip"
            retrieve = decision == "Retrieve"
            gate_probs = {"generated": decision}
        else:
            retrieve, gate_probs = needs_retrieval(jev, claim, evidence_text)
            usage["llm"] += 1
        if not retrieve:
            rounds.append({"gate": gate_probs, "action": "skip"})
            stop_reason = "skip"
            break

        aspect: Aspect | None
        aspect_probs: dict[str, float] = {}
        if variant == "holistic":
            aspect = None
        elif variant == "random_aspect":
            remaining = [item for item in ASPECTS if item.name not in inspected]
            aspect = rng.choice(remaining) if remaining else None
        elif variant == "doffy_gen":
            remaining = [item for item in ASPECTS if item.name not in inspected]
            names = [item.name for item in remaining]
            system = "Choose one aspect that still needs evidence. Output only the aspect name."
            user = f"Claim: {claim}\nAspects: {', '.join(names)}"
            text = remember_query(jev.generate(system, user, max_new_tokens=12))
            chosen = _match_label(text, names) or (names[0] if names else None)
            aspect = next((item for item in remaining if item.name == chosen), None)
        elif variant == "always_all":
            remaining = [item for item in ASPECTS if item.name not in inspected]
            aspect = remaining[0] if remaining else None
        else:
            aspect, aspect_probs, allow_fallback = select_aspect(
                jev,
                claim,
                evidence_text,
                inspected,
                tau_aspect,
                allow_fallback,
            )
            usage["llm"] += len(aspect_probs)
        if variant != "holistic" and aspect is None:
            rounds.append({
                "gate": gate_probs,
                "aspect_probs": aspect_probs,
                "action": "no_aspect",
            })
            stop_reason = "no_aspect"
            break
        if aspect is None:
            query = remember_query(jev.generate(
                "Write one short search query for the whole claim. Output only the query.",
                f"Claim: {claim}",
                max_new_tokens=32,
            ) or claim)
        else:
            query = remember_query(make_query(jev, claim, aspect, evidence_text))
        picked = _rerank_keep(
            reranker,
            pool,
            query,
            aspect,
            seen_ids,
            rerank_k,
            usage,
        )
        if aspect is not None:
            inspected.add(aspect.name)
        evidence_passages.extend(picked)
        calls += 1
        rounds.append({
            "gate": gate_probs,
            "aspect": aspect.name if aspect is not None else "holistic",
            "aspect_probs": aspect_probs,
            "query": query,
            "n_kept": len(picked),
            "doc_ids": [passage.doc_id for passage in picked],
            "action": "retrieve",
        })
        if variant == "holistic" and calls >= search_cap:
            break

    evidence_text = format_passages(evidence_passages)
    if variant == "doffy_gen":
        system = "Decide whether the claim is Real or Fake. Output only Real or Fake."
        user = f"Claim: {claim}"
        if evidence_text:
            user += f"\nCurrent evidence:\n{evidence_text}"
        text = remember_query(jev.generate(system, user, max_new_tokens=8))
        prediction = _match_label(text, labels) or "invalid"
        judge_probs: dict[str, float] = {}
        scores: list[float] = []
    else:
        system_prompt, user_prompt, options = prompt4jev_judge(
            benchmark_cls=bench,
            claim=claim,
            few_shot=few_shot or None,
            evidence=evidence_text,
        )
        scored = jev.score_options(system_prompt, user_prompt, options)
        usage["llm"] += 1
        best_idx = int(scored["best_idx"])
        probs = _as_floats(scored["probs"])
        prediction = labels[best_idx]
        judge_probs = {label: probs[index] for index, label in enumerate(labels)}
        scores = _as_floats(scored["scores"])
    return {
        "prediction": prediction,
        "judge_probs": judge_probs,
        "scores": scores,
        "stop_reason": stop_reason,
        "n_calls": calls,
        "n_llm_fwd": usage["llm"],
        "n_gen_tokens": usage["gen_tokens"],
        "n_rerank": usage["rerank"],
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
    variant: str = "doffy",
) -> dict[str, Any]:
    name = bench.name
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = f".shard{stats['shard']}" if stats["num_shards"] > 1 else ""
    trajectory_path = output_dir / f"{variant}_{name}{suffix}.jsonl"
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
                    variant,
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
                "n_llm_fwd": result.get("n_llm_fwd", 0),
                "n_gen_tokens": result.get("n_gen_tokens", 0),
                "n_rerank": result.get("n_rerank", 0),
                "variant": variant,
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
        "variant": variant,
        "few_shot": bool(few_shot),
        **stats,
    }
    summary = summarize_file(trajectory_path, bench, meta)
    summary_path = output_dir / f"{variant}_{name}{suffix}.summary.json"
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
    parser.add_argument(
        "--variant",
        default="doffy",
        choices=["doffy", "no_gate", "holistic", "random_gate", "random_aspect", "always_all", "doffy_gen", "select_joint"],
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else Path(DEFAULT_OUTPUT_DIR) / Path(args.model).name
    output_dir.mkdir(parents=True, exist_ok=True)
    jev = LLM_as_Jev(model_path=args.model)
    reranker_device = None
    if (
        args.variant == "select_joint"
        and torch.cuda.is_available()
        and torch.cuda.get_device_properties(0).total_memory < 40 * 1024 ** 3
    ):
        # 8B and 4B do not both stay resident on a 24GB card. Park the reranker
        # on CPU until the aspect queries are ready, then swap for the rerank.
        reranker_device = "cpu"
    reranker = Qwen3Reranker(
        model_path=args.reranker,
        batch_size=args.rerank_batch_size,
        max_length=args.rerank_max_length,
        device=reranker_device,
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
                variant=args.variant,
            )
    finally:
        retriever.close()
    suffix = f".shard{args.shard}" if args.num_shards > 1 else ""
    (output_dir / f"{args.variant}_summary{suffix}.json").write_text(
        json.dumps(summaries, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print("wrote", output_dir, flush=True)


if __name__ == "__main__":
    main()
