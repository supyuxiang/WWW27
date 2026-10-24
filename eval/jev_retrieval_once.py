from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.prompts import prompt4what_choice
from src.workers import (
    BM25Retriever,
    LLM_as_Jev,
    SQLiteBM25Retriever,
)
from src.config import Document
from src.utils import (
    Benchmark,
    compute_accuracy,
    compute_f1,
    load_benchmark
)
BENCHMARK_ROOT = Path("/root/autodl-tmp/www27/benchmark")
INDEX_ROOT = Path("/root/autodl-tmp/www27/index")
DEFAULT_MODEL = "/root/autodl-tmp/models/Qwen2.5-3B-Instruct"


def _load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_scifact_corpus() -> list[Document]:
    rows = _load_jsonl(BENCHMARK_ROOT / "scifact" / "data" / "corpus.jsonl")
    documents = []
    for row in rows:
        doc_id = str(row.get("doc_id") or row.get("id") or "")
        title = str(row.get("title") or "")
        text = str(row.get("text") or row.get("abstract") or "").strip()
        if doc_id and text:
            documents.append(Document(doc_id=doc_id, title=title, text=text))
    return documents


class RetrieverBundle:
    def __init__(self) -> None:
        self.fever = SQLiteBM25Retriever(INDEX_ROOT / "fever_fts.sqlite", claim_scoped=False)
        # AVeriTeC's dev file does not expose the claim ids used by the
        # knowledge-store zip.  Use the full dev knowledge store as a global
        # evidence database instead of guessing the per-claim file id.
        self.averitec = SQLiteBM25Retriever(
            INDEX_ROOT / "averitec_dev_fts.sqlite", claim_scoped=False
        )
        self.scifact = BM25Retriever(load_scifact_corpus())

    def search(self, benchmark: str, claim_id: str, claim: str, k: int):
        if benchmark == "fever":
            return self.fever.search(claim, k=k, max_passage_chars=700)
        if benchmark == "averitec":
            return self.averitec.search(claim, k=k, max_passage_chars=700)
        if benchmark == "scifact":
            return self.scifact.search(claim, k=k, max_passage_chars=700)
        raise ValueError(benchmark)

    def close(self) -> None:
        self.fever.close()
        self.averitec.close()


def format_passages(passages) -> str:
    if not passages:
        return "none"
    parts = []
    for index, passage in enumerate(passages, start=1):
        title = f"{passage.title}\n" if passage.title else ""
        parts.append(f"[{index}] {title}{passage.text}")
    return "\n\n".join(parts)


def run_benchmark(
    name: str,
    examples: list[dict],
    bench: Benchmark,
    model_path: str,
    output_dir: Path,
    top_k: int,
) -> dict:
    llm = LLM_as_Jev(model_path)
    labels = list(bench.choice_dict)
    retrievers = RetrieverBundle()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"jev_retrieval_{name}.jsonl"
    started = time.perf_counter()
    gold: list[str] = []
    pred: list[str] = []
    n_errors = 0
    search_calls = 0
    latencies: list[float] = []
    retrieval_counts: Counter = Counter()
    with output_path.open("w", encoding="utf-8") as handle:
        for index, example in enumerate(examples, start=1):
            item_started = time.perf_counter()
            prediction = None
            probs: dict[str, float] = {}
            error = None
            passages = []
            try:
                passages = retrievers.search(name, str(example.get("id") or ""), example["claim"], top_k)
                search_calls += 1
                retrieval_counts[len(passages)] += 1
                evidence = format_passages(passages)
                system_prompt, user_prompt, options = prompt4what_choice(
                    bench, example["claim"], evidence
                )
                scored = llm.score_options(system_prompt, user_prompt, options)
                best_idx = int(scored["best_idx"])
                prediction = labels[best_idx]
                probs = {label: float(scored["probs"][i]) for i, label in enumerate(labels)}
                gold.append(example["label"])
                pred.append(prediction)
            except Exception as exc:  # keep a partially usable run
                error = f"{type(exc).__name__}: {exc}"
            latency = time.perf_counter() - item_started
            latencies.append(latency)
            if error:
                n_errors += 1
            row = {
                "benchmark": name,
                "id": example.get("id"),
                "claim": example["claim"],
                "gold": example.get("label"),
                "prediction": prediction,
                "judge_probs": probs,
                "retrieved": [
                    {"doc_id": p.doc_id, "title": p.title, "text": p.text, "score": p.score}
                    for p in passages
                ],
                "n_retrieved": len(passages),
                "latency_s": latency,
                "retrieval_method": "claim_bm25_top3",
                "search_calls": 1,
                "error": error,
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            if index % 25 == 0 or index == len(examples):
                print(f"{name}: {index}/{len(examples)}", flush=True)
    elapsed = time.perf_counter() - started
    summary = {
        "benchmark": name,
        "model": model_path,
        "retrieval": "claim_bm25_top3",
        "top_k": top_k,
        "n": len(gold),
        "n_errors": n_errors,
        "n_input": len(examples),
        "accuracy": compute_accuracy(gold, pred),
        "f1": compute_f1(gold, pred, bench),
        "gold_counts": dict(Counter(gold)),
        "prediction_counts": dict(Counter(pred)),
        "retrieval_counts": dict(retrieval_counts),
        "search_calls": search_calls,
        "elapsed_s": elapsed,
        "avg_latency_s": sum(latencies) / len(latencies) if latencies else 0.0,
        "output": str(output_path),
    }
    (output_dir / f"jev_retrieval_{name}.summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    retrievers.close()
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrieval-augmented LLM-as-Jev baseline")
    parser.add_argument("--benchmarks", nargs="+", default=["averitec", "fever", "scifact"])
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--limit", type=int, default=0, help="<=0 means the full split")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--output-dir", default="/root/autodl-tmp/www27/results/jev_retrieval_qwen25_3b")
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    summaries = {}
    for name in args.benchmarks:
        examples, bench = load_benchmark(name, args.limit)
        print(f"{name}: {len(examples)} claims", flush=True)
        summaries[name] = run_benchmark(
            name, examples, bench, args.model, output_dir, args.top_k
        )
    (output_dir / "jev_retrieval_summary.json").write_text(
        json.dumps(summaries, indent=2), encoding="utf-8"
    )
    lines = [
        "# Retrieval-augmented LLM-as-Jev baseline",
        "",
        "| Benchmark | n | Accuracy | F1 | Errors |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, summary in summaries.items():
        lines.append(
            f"| {name} | {summary['n']} | {summary['accuracy']:.4f} | "
            f"{summary['f1']:.4f} | {summary['n_errors']} |"
        )
    (output_dir / "jev_retrieval_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print((output_dir / "jev_retrieval_summary.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
