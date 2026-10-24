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
from src.prompts import build_few_shot, prompt4jev_judge
from src.utils import (
    Benchmark,
    compute_accuracy,
    compute_f1,
    load_benchmark,
    load_corpus,
    per_class_scores,
)
from src.workers import (
    BM25Retriever,
    LLM_as_Jev,
    Qwen3Reranker,
    SQLiteBM25Retriever,
)

INDEX_ROOT = Path("/root/autodl-tmp/www27/index")
DEFAULT_OUTPUT_DIR = "/root/autodl-tmp/www27/results/jev_retrieval_once"
SQLITE_INDEXES = {
    "fever": INDEX_ROOT / "fever_fts.sqlite",
    "averitec": INDEX_ROOT / "averitec_dev_fts.sqlite",
}


def _as_floats(value: Any) -> list[float]:
    if hasattr(value, "detach"):
        value = value.detach().float().cpu().tolist()
    return [float(item) for item in value]


class ClaimRetriever:
    """BM25 over a SQLite FTS index, or an in-memory index for small corpora."""

    def __init__(self, max_passage_chars: int = 2000):
        self.max_passage_chars = max_passage_chars
        self._sqlite: dict[str, SQLiteBM25Retriever] = {}
        self._memory: dict[str, BM25Retriever] = {}

    def search(self, benchmark: str, claim: str, k: int) -> list[Passage]:
        retriever = self._get(benchmark)
        return retriever.search(
            claim,
            k=k,
            max_passage_chars=self.max_passage_chars,
        )

    def _get(self, benchmark: str):
        index = SQLITE_INDEXES.get(benchmark)
        if index is not None:
            if benchmark not in self._sqlite:
                if not index.is_file():
                    raise FileNotFoundError(index)
                self._sqlite[benchmark] = SQLiteBM25Retriever(
                    index,
                    claim_scoped=False,
                )
            return self._sqlite[benchmark]
        if benchmark not in self._memory:
            documents = load_corpus(benchmark)
            self._memory[benchmark] = BM25Retriever(documents)
        return self._memory[benchmark]

    def close(self) -> None:
        for retriever in self._sqlite.values():
            retriever.close()
        self._sqlite.clear()


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
    model_path: str,
    reranker_path: str,
) -> dict[str, Any]:
    name = bench.name
    output_dir.mkdir(parents=True, exist_ok=True)
    labels = list(bench.choice_dict)
    gold: list[str] = []
    pred: list[str] = []
    n_errors = 0
    trajectory_path = output_dir / f"jev_retrieval_once_{name}.jsonl"
    started = time.perf_counter()
    with trajectory_path.open("w", encoding="utf-8") as handle:
        for example in tqdm(examples, desc=name):
            item_started = time.perf_counter()
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
                    bm25_k,
                    rerank_k,
                )
                evidence = format_passages(selected)
                system_prompt, user_prompt, options = prompt4jev_judge(
                    benchmark_cls=bench,
                    claim=example["claim"],
                    few_shot=few_shot,
                    evidence=evidence,
                )
                scored = jev.score_options(system_prompt, user_prompt, options)
                best_idx = int(scored["best_idx"])
                prediction = labels[best_idx]
                probs = {
                    label: _as_floats(scored["probs"])[i]
                    for i, label in enumerate(labels)
                }
                scores = _as_floats(scored["scores"])
                gold.append(example["label"])
                pred.append(prediction)
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                n_errors += 1
                
            handle.write(json.dumps({
                "benchmark": name,
                "id": example.get("id"),
                "claim": example["claim"],
                "gold": example.get("label"),
                "prediction": prediction,
                "judge_probs": probs,
                "scores": scores,
                "n_bm25": len(candidates),
                "evidence": [
                    {
                        "doc_id": passage.doc_id,
                        "title": passage.title,
                        "text": passage.text,
                        "rerank_score": passage.score,
                    }
                    for passage in selected
                ],
                "latency_s": time.perf_counter() - item_started,
                "retrieval": f"bm25_top{bm25_k}_rerank_top{rerank_k}",
                "error": error,
            }, ensure_ascii=False) + "\n")
            handle.flush()

    summary = {
        "benchmark": name,
        "model": model_path,
        "reranker": reranker_path,
        "bm25_k": bm25_k,
        "rerank_k": rerank_k,
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
    (output_dir / f"jev_retrieval_once_{name}.summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="LLM-as-Jev with one BM25 retrieval and reranking",
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
    parser.add_argument("--rerank-batch-size", type=int, default=8)
    args = parser.parse_args()
    if args.bm25_k <= 0:
        raise ValueError(f"bm25-k must be positive, got {args.bm25_k}")
    if args.rerank_k <= 0:
        raise ValueError(f"rerank-k must be positive, got {args.rerank_k}")

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
            summary = run_one(
                examples=examples,
                bench=bench,
                jev=jev,
                reranker=reranker,
                retriever=retriever,
                few_shot=few_shot,
                output_dir=output_dir,
                bm25_k=args.bm25_k,
                rerank_k=args.rerank_k,
                model_path=args.model,
                reranker_path=args.reranker,
            )
            summaries[name] = summary
    finally:
        retriever.close()

    (output_dir / "jev_retrieval_once_summary.json").write_text(
        json.dumps(summaries, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    lines = [
        "# LLM-as-Jev with BM25 top-500 and reranker top-2",
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
    (output_dir / "jev_retrieval_once_summary.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    print("wrote", output_dir, flush=True)


if __name__ == "__main__":
    main()
