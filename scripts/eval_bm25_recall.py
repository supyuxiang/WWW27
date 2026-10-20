"""CPU-only BM25 evidence recall on SciFact and FEVER dev claims.

This measures retrieval only: for each claim with gold evidence document ids,
how much of the gold evidence is recovered in the top-k BM25 results.  It does
not invoke any language model.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.retrieval import BM25Retriever
from src.retrieval_sqlite import SQLiteBM25Retriever
from src.config import Document

BENCHMARK_ROOT = Path("/root/autodl-tmp/www27/benchmark")
INDEX_ROOT = Path("/root/autodl-tmp/www27/index")


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def scifact_load() -> tuple[list[tuple[str, set[str]]], list[Document]]:
    corpus = load_jsonl(BENCHMARK_ROOT / "scifact" / "data" / "corpus.jsonl")
    documents = []
    for row in corpus:
        text = str(row.get("text") or row.get("abstract") or "").strip()
        if row.get("doc_id") is not None and text:
            documents.append(
                Document(
                    doc_id=str(row["doc_id"]),
                    title=str(row.get("title") or ""),
                    text=text,
                )
            )
    claims = []
    for row in load_jsonl(BENCHMARK_ROOT / "scifact" / "data" / "claims_dev.jsonl"):
        gold: set[str] = set()

        def visit(value: object) -> None:
            if isinstance(value, dict):
                doc_id = value.get("doc_id")
                if doc_id is not None:
                    gold.add(str(doc_id))
                for item in value.values():
                    visit(item)
            elif isinstance(value, list):
                for item in value:
                    visit(item)

        evidence = row.get("evidence") or {}
        if isinstance(evidence, dict):
            gold.update(str(doc_id) for doc_id in evidence.keys())
        visit(evidence)
        claim = str(row.get("claim") or "").strip()
        if claim and gold:
            claims.append((claim, gold))
    return claims, documents


def fever_load(limit: int, seed: int) -> list[tuple[str, set[str]]]:
    rows = load_jsonl(BENCHMARK_ROOT / "fever" / "shared_task_dev.jsonl")
    items = []
    for row in rows:
        claim = str(row.get("claim") or "").strip()
        gold = set()
        for evidence in row.get("evidence") or []:
            if isinstance(evidence, list):
                for chunk in evidence:
                    if isinstance(chunk, list) and len(chunk) >= 3:
                        gold.add(str(chunk[2]))
            elif isinstance(evidence, dict) and evidence.get("page_id") is not None:
                gold.add(str(evidence["page_id"]))
        if claim and gold:
            items.append((claim, gold))
    if limit > 0 and len(items) > limit:
        items = random.Random(seed).sample(items, limit)
        items.sort(key=lambda item: item[0])
    return items


def metrics_for(retrieve, items, ks):
    totals = {k: {"recall": 0.0, "hit": 0, "mrr": 0.0} for k in ks}
    for claim, gold in items:
        ranked = [p.doc_id for p in retrieve(claim, max(ks))]
        for k in ks:
            top = set(ranked[:k])
            found = len(top & gold)
            totals[k]["recall"] += found / len(gold)
            totals[k]["hit"] += int(found > 0)
            for rank, doc_id in enumerate(ranked[:k], start=1):
                if doc_id in gold:
                    totals[k]["mrr"] += 1.0 / rank
                    break
    n = len(items)
    if n == 0:
        raise SystemExit("no claims with qrels")
    return {
        k: {
            "recall": totals[k]["recall"] / n,
            "hit": totals[k]["hit"] / n,
            "mrr": totals[k]["mrr"] / n,
        }
        for k in ks
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fever-limit", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="/root/autodl-tmp/www27/results/bm25_recall.json")
    parser.add_argument("--ks", nargs="+", type=int, default=[1, 3, 5, 10, 20, 50])
    args = parser.parse_args()
    ks = sorted(set(args.ks))

    scifact_claims, scifact_docs = scifact_load()
    print(f"scifact: {len(scifact_claims)} claims with qrels, {len(scifact_docs)} docs", flush=True)
    scifact_retriever = BM25Retriever(scifact_docs)
    scifact_metrics = metrics_for(
        lambda q, k: scifact_retriever.search(q, k=k, max_passage_chars=1000),
        scifact_claims,
        ks,
    )

    fever_claims = fever_load(args.fever_limit, args.seed)
    print(f"fever: {len(fever_claims)} claims with qrels", flush=True)
    fever_retriever = SQLiteBM25Retriever(INDEX_ROOT / "fever_fts.sqlite", claim_scoped=False)
    fever_metrics = metrics_for(
        lambda q, k: fever_retriever.search(q, k=k, max_passage_chars=1000),
        fever_claims,
        ks,
    )
    fever_retriever.close()

    output = {
        "scifact": {"n": len(scifact_claims), "metrics": scifact_metrics},
        "fever": {
            "n": len(fever_claims),
            "requested_limit": args.fever_limit,
            "metrics": fever_metrics,
        },
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")

    lines = [
        "# CPU BM25 evidence recall",
        "",
        f"- SciFact: {len(scifact_claims)} claims with qrels",
        f"- FEVER: {len(fever_claims)} claims with qrels"
        + (f" (random {args.fever_limit})" if args.fever_limit else " (all)"),
        "",
        "| Benchmark | k | Recall | Hit@k | MRR@k |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, data in output.items():
        for k in ks:
            m = data["metrics"][k]
            lines.append(
                f"| {name} | {k} | {m['recall']:.4f} | {m['hit']:.4f} | {m['mrr']:.4f} |"
            )
    text = "\n".join(lines) + "\n"
    out_path.with_suffix(".md").write_text(text, encoding="utf-8")
    print(text, flush=True)


if __name__ == "__main__":
    main()
