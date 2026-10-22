from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.aspects import ASPECTS, ASPECT_BY_NAME
from src.config import Document, Passage, WWW27Config
from src.llm import LocalLLM
from src.prompts import (
    construct_text4rerankng,
    prompt4aspect_check,
    prompt4what_choice,
    prompt4whether_retrieval,
)
from src.rerank import Qwen3Reranker
from src.retrieval import BM25Retriever
from src.retrieval_sqlite import SQLiteBM25Retriever
from src.utils import (
    Benchmark,
    clip,
    compute_accuracy,
    compute_f1,
    format_evidence,
    is_leakage,
    load_benchmark,
)

BENCH_ROOT = Path("/root/autodl-tmp/www27/benchmark")
INDEX_ROOT = Path("/root/autodl-tmp/www27/index")
DEFAULT_MODEL = "/root/autodl-tmp/models/Qwen3-8B"
DEFAULT_RERANKER = "/root/autodl-tmp/models/Qwen3-Reranker-0.6B"

_INDEX_PATHS = {
    "fever": INDEX_ROOT / "fever_fts.sqlite",
    "averitec": INDEX_ROOT / "averitec_dev_fts.sqlite",
}


def _probs(scored: dict) -> list[float]:
    return [float(item) for item in scored["probs"]]


def select_aspects(scores: dict[str, float], tau: float, limit: int) -> list[str]:
    ranked = sorted(scores, key=lambda name: (-scores[name], name))
    chosen = [name for name in ranked if scores[name] >= tau]
    if limit > 0:
        chosen = chosen[:limit]
    if not chosen and ranked:
        chosen = ranked[:1]
    return chosen


def load_scifact_documents(bench_root: Path) -> list[Document]:
    path = bench_root / "scifact" / "data" / "corpus.jsonl"
    documents: list[Document] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            abstract = row.get("abstract") or ""
            if isinstance(abstract, list):
                text = " ".join(str(part) for part in abstract)
            else:
                text = str(abstract)
            doc_id = str(row.get("doc_id") or "")
            if not doc_id or not text.strip():
                continue
            documents.append(
                Document(doc_id=doc_id, title=str(row.get("title") or ""), text=text.strip())
            )
    return documents


class OfflineRetriever:
    """SciFact stays in memory. FEVER and AVeriTeC read a prebuilt FTS index."""

    def __init__(self, bench_root: Path, index_root: Path, max_passage_chars: int):
        self.bench_root = bench_root
        self.index_root = index_root
        self.max_passage_chars = max_passage_chars
        self._scifact: BM25Retriever | None = None
        self._sqlite: dict[str, SQLiteBM25Retriever] = {}

    def search(
        self,
        benchmark: str,
        query: str,
        k: int,
        exclude_ids: set[str] | None = None,
    ) -> list[Passage]:
        if benchmark == "scifact":
            if self._scifact is None:
                self._scifact = BM25Retriever(load_scifact_documents(self.bench_root))
            return self._scifact.search(
                query,
                k=k,
                exclude_ids=exclude_ids,
                max_passage_chars=self.max_passage_chars,
            )
        path = self.index_root / _INDEX_PATHS[benchmark].name
        if benchmark not in self._sqlite:
            if not path.is_file():
                script = "build_fever_fts.py" if benchmark == "fever" else "build_averitec_fts.py"
                raise FileNotFoundError(
                    f"Missing offline index {path}. Build it with scripts/{script} "
                    f"from the corpus under {self.bench_root / benchmark}."
                )
            self._sqlite[benchmark] = SQLiteBM25Retriever(path, claim_scoped=False)
        return self._sqlite[benchmark].search(
            query,
            k=k,
            exclude_ids=exclude_ids,
            max_passage_chars=self.max_passage_chars,
        )

    def close(self) -> None:
        for retriever in self._sqlite.values():
            retriever.close()
        self._sqlite.clear()


def run_example(
    llm: LocalLLM,
    retriever: OfflineRetriever,
    reranker: Qwen3Reranker,
    bench: Benchmark,
    example: dict,
    config: WWW27Config,
) -> dict:
    claim = clip(example.get("claim"), config.max_claim_chars)
    evidence: list[Passage] = []
    seen_docs: set[str] = set()
    inspected: set[str] = set()
    query_log: list[dict] = []
    aspect_probs: dict[str, float] = {}
    gate_probs: list[float] = []
    coarse: list[Passage] | None = None
    calls = 0
    rounds = 0
    stop_reason = ""
    blocked: set[str] = set()
    if example.get("id") is not None:
        blocked.add(str(example["id"]))

    def coarse_pool() -> list[Passage] | None:
        nonlocal coarse, calls
        if coarse is not None:
            return coarse
        if calls >= config.search_cap:
            return None
        hits = retriever.search(bench.name, claim, config.bm25_candidates, blocked)
        calls += 1
        kept: list[Passage] = []
        for hit in hits:
            if hit.doc_id in seen_docs or hit.doc_id in blocked:
                continue
            if is_leakage(claim, blocked, hit.doc_id, hit.text):
                continue
            kept.append(hit)
        coarse = kept
        return coarse

    while calls < config.search_cap or coarse is not None:
        rounds += 1
        evidence_text = format_evidence(evidence, config.max_evidence_chars) if evidence else None
        system, user, options = prompt4whether_retrieval(claim, evidence_text)
        scored = llm.score_options(system, user, options)
        p_retrieve = _probs(scored)[0]
        gate_probs.append(p_retrieve)
        if p_retrieve < config.tau_g:
            stop_reason = "gate"
            break

        uncovered = [aspect for aspect in ASPECTS if aspect.name not in inspected]
        if not uncovered:
            stop_reason = "no_uncovered_aspect"
            break
        scores: dict[str, float] = {}
        for aspect in uncovered:
            system, user, options = prompt4aspect_check(claim, aspect, evidence_text)
            scored = llm.score_options(system, user, options)
            scores[aspect.name] = _probs(scored)[0]
        aspect_probs.update(scores)
        selected = select_aspects(scores, config.tau_a, config.max_aspects)
        if not selected:
            stop_reason = "no_uncovered_aspect"
            break

        pool = coarse_pool()
        if pool is None:
            stop_reason = "budget"
            break
        for name in selected:
            inspected.add(name)
            aspect = ASPECT_BY_NAME[name]
            query = construct_text4rerankng(claim, aspect)
            candidates = [item for item in pool if item.doc_id not in seen_docs]
            ranked = reranker.rerank(query, candidates, config.passages_per_query)
            kept_rows = []
            for hit, rerank_score in ranked:
                seen_docs.add(hit.doc_id)
                picked = replace(
                    hit,
                    score=rerank_score,
                    query=query,
                    aspect=name,
                    sub_claim=aspect.definition,
                )
                evidence.append(picked)
                kept_rows.append({"doc_id": picked.doc_id, "score": rerank_score})
            query_log.append({"aspect": name, "query": query, "kept": kept_rows})
    if not stop_reason:
        stop_reason = "budget"

    evidence_text = format_evidence(evidence, config.max_evidence_chars) if evidence else None
    labels = list(bench.choice_dict)
    system, user, options = prompt4what_choice(bench, claim, evidence_text)
    scored = llm.score_options(system, user, options)
    probs = _probs(scored)
    prediction = labels[int(scored["best_idx"])]
    return {
        "id": example.get("id"),
        "claim": claim,
        "gold": example.get("label"),
        "prediction": prediction,
        "judge_probs": {label: probs[i] for i, label in enumerate(labels)},
        "gate_probs": gate_probs,
        "aspect_probs": aspect_probs,
        "aspects": [item["aspect"] for item in query_log],
        "queries": query_log,
        "stop_reason": stop_reason,
        "retrieval_rounds": rounds,
        "search_calls": calls,
        "n_evidence": len(evidence),
        "evidence": [
            {
                "doc_id": item.doc_id,
                "title": item.title,
                "score": item.score,
                "aspect": item.aspect,
                "query": item.query,
                "text": item.text,
            }
            for item in evidence
        ],
        "error": None,
    }


def run_benchmark(
    name: str,
    examples: list[dict],
    bench: Benchmark,
    llm: LocalLLM,
    retriever: OfflineRetriever,
    reranker: Qwen3Reranker,
    config: WWW27Config,
    output_dir: Path,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"eval_main_{name}.jsonl"
    started = time.perf_counter()
    gold: list[str] = []
    pred: list[str] = []
    n_errors = 0
    calls: list[int] = []
    rounds: list[int] = []
    with output_path.open("w", encoding="utf-8") as handle:
        for index, example in enumerate(examples, start=1):
            item_started = time.perf_counter()
            try:
                row = run_example(llm, retriever, reranker, bench, example, config)
            except Exception as exc:
                row = {
                    "id": example.get("id"),
                    "claim": example.get("claim"),
                    "gold": example.get("label"),
                    "prediction": None,
                    "error": f"{type(exc).__name__}: {exc}",
                    "search_calls": 0,
                    "retrieval_rounds": 0,
                }
            row["latency_s"] = time.perf_counter() - item_started
            row["benchmark"] = name
            if row.get("error"):
                n_errors += 1
            elif row.get("prediction") is not None and row.get("gold") is not None:
                gold.append(row["gold"])
                pred.append(row["prediction"])
                calls.append(int(row.get("search_calls") or 0))
                rounds.append(int(row.get("retrieval_rounds") or 0))
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            if index % 25 == 0 or index == len(examples):
                print(f"{name}: {index}/{len(examples)}", flush=True)
    elapsed = time.perf_counter() - started
    summary = {
        "benchmark": name,
        "model": config.model_path,
        "retrieval": "bm25_top500_qwen3_reranker_top2",
        "n": len(gold),
        "n_errors": n_errors,
        "n_input": len(examples),
        "accuracy": compute_accuracy(gold, pred),
        "f1": compute_f1(gold, pred, bench),
        "avg_search_calls": sum(calls) / len(calls) if calls else 0.0,
        "avg_rounds": sum(rounds) / len(rounds) if rounds else 0.0,
        "gold_counts": dict(Counter(gold)),
        "prediction_counts": dict(Counter(pred)),
        "elapsed_s": elapsed,
        "output": str(output_path),
    }
    (output_dir / f"eval_main_{name}.summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def require_corpus(name: str, bench_root: Path, index_root: Path) -> None:
    if name == "scifact":
        path = bench_root / "scifact" / "data" / "corpus.jsonl"
        if not path.is_file():
            raise FileNotFoundError(path)
        return
    path = index_root / _INDEX_PATHS[name].name
    if path.is_file():
        return
    script = "build_fever_fts.py" if name == "fever" else "build_averitec_fts.py"
    raise FileNotFoundError(
        f"Missing offline index {path}. Build it with scripts/{script} "
        f"from the corpus under {bench_root / name}."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline JARV eval with predefined queries")
    parser.add_argument("--benchmarks", nargs="+", default=["averitec", "fever", "scifact"])
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--limit", type=int, default=200, help="<=0 means the full split")
    parser.add_argument("--output-dir", default="/root/yxfeng/www27/outputs/eval_main")
    parser.add_argument("--bench-root", default=str(BENCH_ROOT))
    parser.add_argument("--index-root", default=str(INDEX_ROOT))
    parser.add_argument("--tau-g", type=float, default=0.5)
    parser.add_argument("--tau-a", type=float, default=0.5)
    parser.add_argument("--search-cap", type=int, default=3)
    parser.add_argument("--max-aspects", type=int, default=3)
    parser.add_argument("--bm25-k", type=int, default=500)
    parser.add_argument("--rerank-k", type=int, default=2)
    parser.add_argument("--rerank-model", default=DEFAULT_RERANKER)
    args = parser.parse_args()

    config = WWW27Config(
        model_path=args.model,
        limit=args.limit,
        tau_g=args.tau_g,
        tau_a=args.tau_a,
        max_aspects=args.max_aspects,
        passages_per_query=args.rerank_k,
        bm25_candidates=args.bm25_k,
        rerank_model_path=args.rerank_model,
        search_cap=args.search_cap,
        queries_per_aspect=1,
    )
    output_dir = Path(args.output_dir)
    bench_root = Path(args.bench_root)
    index_root = Path(args.index_root)
    for name in args.benchmarks:
        require_corpus(name, bench_root, index_root)
    if not Path(config.rerank_model_path).is_dir():
        raise FileNotFoundError(
            f"Missing reranker {config.rerank_model_path}. "
            "Expected the Qwen3-Reranker-0.6B checkpoint."
        )
    llm = LocalLLM(config.model_path)
    reranker = Qwen3Reranker(config.rerank_model_path)
    retriever = OfflineRetriever(bench_root, index_root, config.max_passage_chars)
    summaries = {}
    try:
        for name in args.benchmarks:
            examples, bench = load_benchmark(name, args.limit)
            print(f"{name}: {len(examples)} claims", flush=True)
            summaries[name] = run_benchmark(
                name, examples, bench, llm, retriever, reranker, config, output_dir
            )
    finally:
        retriever.close()
    (output_dir / "eval_main_summary.json").write_text(
        json.dumps(summaries, indent=2), encoding="utf-8"
    )
    lines = [
        "# Offline JARV",
        "",
        "| Benchmark | n | Accuracy | F1 | Calls | Rounds | Errors |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, summary in summaries.items():
        lines.append(
            f"| {name} | {summary['n']} | {summary['accuracy']:.4f} | "
            f"{summary['f1']:.4f} | {summary['avg_search_calls']:.2f} | "
            f"{summary['avg_rounds']:.2f} | {summary['n_errors']} |"
        )
    (output_dir / "eval_main_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("wrote", output_dir / "eval_main_summary.json", flush=True)


if __name__ == "__main__":
    main()
