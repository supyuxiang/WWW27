"""One dense retrieval, rerank, then the judge. No gold evidence and no BM25."""
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

from src.config import Passage
from src.paths import RESULTS
from src.prompts import build_few_shot, prompt4jev_judge
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

DEFAULT_OUTPUT_DIR = str(RESULTS / "retrieve_once")


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


def retrieve_once(
    retriever: ClaimRetriever,
    reranker: Qwen3Reranker,
    benchmark: str,
    claim: str,
    claim_id: str | None,
    pool_k: int,
    rerank_k: int,
) -> tuple[list[Passage], list[Passage]]:
    candidates = retriever.search(benchmark, claim, pool_k, claim_id=claim_id)
    ranked = reranker.rerank(claim, candidates, top_k=rerank_k)
    selected = [
        replace(passage, score=float(score), query=claim)
        for passage, score in ranked
    ]
    return candidates, selected


def summarize_file(path: Path, bench: Benchmark, meta: dict) -> dict[str, Any]:
    gold: list[str] = []
    pred: list[str] = []
    latencies: list[float] = []
    n_errors = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("error"):
                n_errors += 1
            if row.get("prediction") is None or row.get("gold") is None:
                continue
            gold.append(row["gold"])
            pred.append(row["prediction"])
            if row.get("latency_s") is not None:
                latencies.append(float(row["latency_s"]))
    labels = list(bench.choice_dict)
    return {
        **meta,
        "n": len(gold),
        "n_errors": n_errors,
        "accuracy": compute_accuracy(gold, pred),
        "macro_f1": compute_f1(gold, pred, bench),
        "per_class": per_class_scores(gold, pred, labels),
        "gold_counts": dict(Counter(gold)),
        "prediction_counts": dict(Counter(pred)),
        "mean_calls": 1.0,
        "mean_latency_s": (sum(latencies) / len(latencies)) if latencies else 0.0,
        "output": str(path),
    }


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
    model_path: str,
    reranker_path: str,
    stats: dict,
) -> dict[str, Any]:
    name = bench.name
    output_dir.mkdir(parents=True, exist_ok=True)
    labels = list(bench.choice_dict)
    suffix = f".shard{stats['shard']}" if stats["num_shards"] > 1 else ""
    trajectory_path = output_dir / f"retrieve_once_{name}{suffix}.jsonl"
    done = load_done_ids(trajectory_path)
    pending = [example for example in examples if str(example.get("id")) not in done]
    print(f"{name} shard {stats['shard']}: {len(pending)} pending, {len(done)} done", flush=True)
    with trajectory_path.open("a", encoding="utf-8") as handle:
        for example in tqdm(pending, desc=f"{name}:{stats['shard']}"):
            started = time.perf_counter()
            prediction = None
            probs: dict[str, float] = {}
            scores: list[float] = []
            candidates: list[Passage] = []
            selected: list[Passage] = []
            error = None
            try:
                candidates, selected = retrieve_once(
                    retriever,
                    reranker,
                    name,
                    example["claim"],
                    example.get("id"),
                    pool_k,
                    rerank_k,
                )
                system_prompt, user_prompt, options = prompt4jev_judge(
                    benchmark_cls=bench,
                    claim=example["claim"],
                    few_shot=few_shot or None,
                    evidence=format_passages(selected),
                )
                scored = jev.score_options(system_prompt, user_prompt, options)
                prediction = labels[int(scored["best_idx"])]
                probs = {
                    label: _as_floats(scored["probs"])[index]
                    for index, label in enumerate(labels)
                }
                scores = _as_floats(scored["scores"])
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
            handle.write(json.dumps({
                "benchmark": name,
                "id": example.get("id"),
                "claim": example["claim"],
                "gold": example.get("label"),
                "gold_raw": example.get("label_raw"),
                "prediction": prediction,
                "judge_probs": probs,
                "scores": scores,
                "n_pool": len(candidates),
                "n_calls": 1,
                "evidence": [
                    {
                        "doc_id": passage.doc_id,
                        "title": passage.title,
                        "text": passage.text,
                        "rerank_score": passage.score,
                    }
                    for passage in selected
                ],
                "latency_s": time.perf_counter() - started,
                "retrieval": f"dense_top{pool_k}_rerank_top{rerank_k}",
                "error": error,
            }, ensure_ascii=False) + "\n")
            handle.flush()
    meta = {
        "benchmark": name,
        "model": model_path,
        "reranker": reranker_path,
        "retriever": "qwen3-embedding-4b",
        "pool_k": pool_k,
        "rerank_k": rerank_k,
        "few_shot": bool(few_shot),
        **stats,
    }
    summary = summarize_file(trajectory_path, bench, meta)
    (output_dir / f"retrieve_once_{name}{suffix}.summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps({
        "benchmark": name,
        "shard": stats["shard"],
        "n": summary["n"],
        "accuracy": summary["accuracy"],
        "mean_latency_s": summary["mean_latency_s"],
        "n_errors": summary["n_errors"],
    }, indent=2), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Judge after one dense retrieval")
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
    parser.add_argument("--rerank-batch-size", type=int, default=16)
    parser.add_argument("--rerank-max-length", type=int, default=512)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--binary", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else Path(DEFAULT_OUTPUT_DIR) / Path(args.model).name
    )
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
                model_path=args.model,
                reranker_path=args.reranker,
                stats=stats,
            )
    finally:
        retriever.close()
    suffix = f".shard{args.shard}" if args.num_shards > 1 else ""
    (output_dir / f"retrieve_once_summary{suffix}.json").write_text(
        json.dumps(summaries, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print("wrote", output_dir, flush=True)


if __name__ == "__main__":
    main()
