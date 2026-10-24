#!/usr/bin/env python3
"""CFEVER three-way adapter for Retrieve-once+JEV and JARV.

This keeps JARV's gate, aspect selection, query generation, budgets, and
teacher-forced suffix scoring.  Only the benchmark I/O, Chinese sparse
retriever, and native three-way final verdict are adapted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sqlite3
import statistics
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baselines.run_main_experiment import BatchedInstrumentedFactChecker, TrackedLocalLLM
from src.config import JarvConfig
from src.prompts import SYSTEM_PROMPT, _options
from src.types import AtomicQuery, Cost, Example, Option, Passage, add_usage
from src.utils.text import clip, format_evidence, format_subclaims

LABELS = ("supports", "refutes", "not_enough_info")
LABEL_TO_OFFICIAL = {
    "supports": "SUPPORTS",
    "refutes": "REFUTES",
    "not_enough_info": "NOT ENOUGH INFO",
}
OFFICIAL_TO_LABEL = {value.lower(): key for key, value in LABEL_TO_OFFICIAL.items()}
OFFICIAL_TO_LABEL.update({"not enough info": "not_enough_info"})
UNIT_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]+|[a-z0-9]+", re.I)
EVIDENCE_ID_RE = re.compile(r"^(.*)#L(-?\d+)$", re.S)


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sparse_tokens(text: str) -> list[str]:
    output: list[str] = []
    for unit in UNIT_RE.findall((text or "").lower()):
        if re.fullmatch(r"[a-z0-9]+", unit):
            output.append(unit)
            continue
        chars = list(unit)
        output.extend("".join(chars[index : index + 2]) for index in range(len(chars) - 1))
        if len(chars) == 1:
            output.extend(chars)
    return list(dict.fromkeys(output))


def normalize_label(value: str) -> str:
    key = str(value).strip().lower().replace("_", " ")
    if key in OFFICIAL_TO_LABEL:
        return OFFICIAL_TO_LABEL[key]
    if key in {"supports", "refutes"}:
        return key
    raise ValueError(f"Unknown CFEVER label: {value!r}")


def parse_lines(lines: str) -> list[tuple[int, str]]:
    parsed = []
    for raw in (lines or "").splitlines():
        number, separator, text = raw.partition("\t")
        if not separator or not text.strip():
            continue
        try:
            line_number = int(number)
        except ValueError:
            continue
        parsed.append((line_number, text.strip()))
    return parsed


class CFEVERRetriever:
    """Read-only Chinese bigram BM25 with sentence-level evidence traces."""

    def __init__(self, database: str | Path):
        self.database = str(database)
        self.connection = sqlite3.connect(
            f"file:{self.database}?mode=ro", uri=True, check_same_thread=False
        )

    @staticmethod
    def _query(text: str) -> tuple[str, list[str]]:
        terms = sparse_tokens(text)
        escaped = [f'"{term.replace(chr(34), "")}"' for term in terms if term]
        return " OR ".join(escaped), terms

    @staticmethod
    def _best_sentence(lines: str, query: str, terms: list[str]) -> tuple[int, str]:
        candidates = parse_lines(lines)
        if not candidates:
            return -1, clip(lines, 800)
        term_set = set(terms)
        query_compact = re.sub(r"\s+", "", query).lower()
        best = None
        for line_number, sentence in candidates:
            sentence_terms = set(sparse_tokens(sentence))
            overlap = len(term_set & sentence_terms)
            compact = re.sub(r"\s+", "", sentence).lower()
            exact = int(bool(query_compact) and query_compact in compact)
            score = (exact, overlap, -line_number)
            if best is None or score > best[0]:
                best = (score, line_number, sentence)
        assert best is not None
        return best[1], best[2]

    def _search(self, connection, query: str, k: int, max_passage_chars: int) -> list[Passage]:
        expression, terms = self._query(query)
        if not expression or k <= 0:
            return []
        candidates: dict[str, tuple[str, float]] = {}
        candidate_limit = max(50, k * 10)
        for column in ("title_tokens", "body_tokens"):
            rows = connection.execute(
                "SELECT d.page_id, d.lines, bm25(docs_fts) AS rank "
                "FROM docs_fts JOIN docs d ON d.rowid = docs_fts.rowid "
                "WHERE docs_fts MATCH ? ORDER BY rank LIMIT ?",
                (f"{column} : ({expression})", candidate_limit),
            ).fetchall()
            for page_id, lines, rank in rows:
                previous = candidates.get(str(page_id))
                if previous is None or float(rank) < previous[1]:
                    candidates[str(page_id)] = (str(lines), float(rank))
        query_terms = set(terms)
        query_compact = re.sub(r"\s+", "", query).lower()
        reranked = []
        for page_id, (lines, rank) in candidates.items():
            title_overlap = len(query_terms & set(sparse_tokens(page_id)))
            exact_title = int(
                len(re.sub(r"\s+", "", page_id)) >= 2
                and re.sub(r"\s+", "", page_id).lower() in query_compact
            )
            line_number, sentence = self._best_sentence(lines, query, terms)
            sentence_overlap = len(query_terms & set(sparse_tokens(sentence)))
            reranked.append(
                (
                    (exact_title, title_overlap, sentence_overlap, -rank, page_id),
                    line_number,
                    sentence,
                    rank,
                    page_id,
                )
            )
        reranked.sort(reverse=True)
        passages = []
        for _, line_number, sentence, rank, page_id in reranked:
            passages.append(
                Passage(
                    doc_id=f"{page_id}#L{line_number}",
                    title=str(page_id),
                    text=clip(sentence, max_passage_chars),
                    score=float(-rank),
                )
            )
            if len(passages) >= k:
                break
        return passages

    def search(self, query: str, k: int, exclude_ids=None, max_passage_chars: int = 800):
        del exclude_ids
        return self._search(self.connection, query, k, max_passage_chars)

    def search_many(self, queries, k: int, exclude_ids=None, max_passage_chars: int = 800):
        del exclude_ids
        if len(queries) <= 1:
            return [self.search(query, k, None, max_passage_chars) for query in queries]

        def worker(query: str):
            connection = sqlite3.connect(f"file:{self.database}?mode=ro", uri=True)
            try:
                return self._search(connection, query, k, max_passage_chars)
            finally:
                connection.close()

        with ThreadPoolExecutor(max_workers=min(3, len(queries))) as executor:
            return list(executor.map(worker, queries))

    def close(self) -> None:
        self.connection.close()


def cfever_judge_prompt(claim: str, subclaims: str, evidence: str):
    pairs = (
        ("supports", "Supports: the evidence entails the claim."),
        ("refutes", "Refutes: the evidence contradicts the claim."),
        ("not_enough_info", "Not Enough Info: the evidence is insufficient to support or refute the claim."),
    )
    listing, options = _options(pairs, False)
    user = (
        f"Original claim: {claim}\n"
        f"Inspected sub-claims: {subclaims}\n"
        f"Retrieved Chinese Wikipedia evidence: {evidence}\n"
        "Use only the available evidence to assign the native CFEVER three-way label. "
        "Do not treat missing evidence as refutation.\n"
        f"{listing}\n"
        "Select one option. Answer only with its bracketed numeric identifier."
    )
    return user, options


class CFEVERFactChecker(BatchedInstrumentedFactChecker):
    def _judge(self, claim, subclaims, evidence, cost):
        self._decision_stage = "final_judge"
        started = time.perf_counter()
        try:
            user, options = cfever_judge_prompt(
                claim,
                format_subclaims(subclaims),
                format_evidence(evidence, self.config.max_evidence_chars),
            )
            choice = self.choose(user, options, cost)
            return choice.key, choice.probs
        finally:
            self.stage_times["final_judge"] += time.perf_counter() - started

    def retrieve(self, example: Example, queries: list[AtomicQuery], cost: Cost):
        added = []
        for query in queries:
            if cost.search_calls >= self.config.search_cap:
                cost.search_truncated = True
                break
            if not query.query.strip():
                continue
            started = time.perf_counter()
            try:
                hits = self.retriever.search(
                    query.query,
                    self.config.passages_per_query,
                    None,
                    self.config.max_passage_chars,
                )
            finally:
                self.stage_times["retrieval"] += time.perf_counter() - started
            cost.search_calls += 1
            for hit in hits:
                if hit.doc_id in self.seen_docs:
                    continue
                self.seen_docs.add(hit.doc_id)
                hit.query, hit.sub_claim, hit.aspect = query.query, query.sub_claim, query.aspect
                self.collected.append(hit)
                added.append(hit)
        return added

    def _retrieve_batched(self, example, queries, cost):
        if not queries:
            return
        started = time.perf_counter()
        try:
            batches = self.retriever.search_many(
                [query.query for query in queries],
                self.config.passages_per_query,
                None,
                self.config.max_passage_chars,
            )
        finally:
            self.stage_times["retrieval"] += time.perf_counter() - started
        cost.search_calls += len(queries)
        for query, hits in zip(queries, batches):
            for hit in hits:
                if hit.doc_id in self.seen_docs:
                    continue
                self.seen_docs.add(hit.doc_id)
                hit.query, hit.sub_claim, hit.aspect = query.query, query.sub_claim, query.aspect
                self.collected.append(hit)


def sample_rows(rows: list[dict], limit: int, seed: int) -> list[dict]:
    if limit <= 0 or limit >= len(rows):
        return sorted(rows, key=lambda row: int(row["id"]))
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[normalize_label(row["label"])].append(row)
    base, remainder = divmod(limit, len(LABELS))
    selected = []
    for index, label in enumerate(LABELS):
        quota = base + int(index < remainder)
        selected.extend(random.Random(f"{seed}:cfever:{label}").sample(groups[label], quota))
    return sorted(selected, key=lambda row: int(row["id"]))


def to_example(row: dict) -> Example:
    return Example(
        id=str(row["id"]),
        claim=str(row["claim"]),
        label=normalize_label(row["label"]),
        domain=str(row.get("domain") or "cfever"),
        source_id="",
    )


def evidence_rows(passages) -> list[dict]:
    return [
        {
            "doc_id": passage.doc_id,
            "page_id": passage.title,
            "line_number": int(EVIDENCE_ID_RE.match(passage.doc_id).group(2)) if EVIDENCE_ID_RE.match(passage.doc_id) else -1,
            "score": passage.score,
            "query": passage.query,
            "sub_claim": passage.sub_claim,
            "aspect": passage.aspect,
            "text": passage.text,
        }
        for passage in passages
    ]


def run_retrieve_once(row, llm, retriever, config):
    started = time.perf_counter()
    llm.reset_counters()
    passages = retriever.search(row["claim"], config.passages_per_query, None, config.max_passage_chars)
    user, options = cfever_judge_prompt(row["claim"], "none", format_evidence(passages, config.max_evidence_chars))
    choice = llm.score_options(SYSTEM_PROMPT, user, options)
    return {
        "id": str(row["id"]),
        "claim": row["claim"],
        "gold": normalize_label(row["label"]),
        "prediction": choice.key,
        "method": "retrieve_once_jev",
        "evidence": evidence_rows(passages),
        "judge_probs": choice.probs,
        "search_calls": 1,
        "retrieval_rounds": 1,
        "input_tokens": choice.usage.input_tokens,
        "output_tokens": choice.usage.output_tokens,
        "query_generation_tokens": 0,
        "total_generated_tokens": 0,
        "latency_s": time.perf_counter() - started,
        "stop_reason": "retrieve_once",
        "error": None,
    }


def macro_f1(rows: list[dict]) -> float:
    values = []
    for label in LABELS:
        tp = sum(row["gold"] == label and row["prediction"] == label for row in rows)
        fp = sum(row["gold"] != label and row["prediction"] == label for row in rows)
        fn = sum(row["gold"] == label and row["prediction"] != label for row in rows)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        values.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return sum(values) / len(values)


def official_evidence(row: dict):
    if row["prediction"] == "not_enough_info":
        # The dataset README permits null, but the pinned official evaluator
        # slices this field unconditionally.  An empty list is semantically
        # equivalent and is the evaluator-compatible representation.
        return []
    output = []
    for evidence in row.get("evidence") or []:
        if evidence.get("line_number", -1) >= 0:
            output.append([evidence["page_id"], evidence["line_number"]])
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev-file", required=True)
    parser.add_argument("--index", required=True)
    parser.add_argument("--model", default="/root/autodl-tmp/models/Qwen2.5-3B-Instruct")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--methods", nargs="+", default=["retrieve_once_jev", "jarv"])
    args = parser.parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(line) for line in Path(args.dev_file).read_text(encoding="utf-8").splitlines() if line.strip()]
    selected = sample_rows(rows, args.limit, args.seed)
    config_payload = {
        "track": "cfever_native_three_way_adapted",
        "methods": args.methods,
        "model": args.model,
        "model_frozen": True,
        "chat_template": "tokenizer.apply_chat_template",
        "enable_thinking": False,
        "seed": args.seed,
        "limit": args.limit,
        "sample_ids": [str(row["id"]) for row in selected],
        "labels": list(LABELS),
        "top_k": 3,
        "search_cap": 3,
        "tau_g": 0.5,
        "tau_a": 0.5,
        "max_aspects": 3,
        "index": args.index,
        "index_sha256": sha256_file(Path(args.index)),
        "dev_file": args.dev_file,
        "dev_sha256": sha256_file(Path(args.dev_file)),
        "retriever": "Chinese character bigram BM25; title weight 5; sentence selected by query overlap",
        "gold_evidence_used_for_retrieval": False,
    }
    fingerprint = hashlib.sha256(json.dumps(config_payload, sort_keys=True).encode()).hexdigest()
    config_payload["config_fingerprint"] = fingerprint
    atomic_json(output_root / "config.json", config_payload)
    atomic_json(output_root / "sample_manifest.json", {"rows": selected})
    config = JarvConfig(
        model_path=args.model,
        method="jarv",
        seed=args.seed,
        tau_g=0.5,
        tau_a=0.5,
        max_aspects=3,
        queries_per_aspect=1,
        passages_per_query=3,
        search_cap=3,
        query_max_new_tokens=192,
        json_retries=2,
    )
    retriever = CFEVERRetriever(args.index)
    llm = TrackedLocalLLM(args.model)
    checker = CFEVERFactChecker(llm, retriever, config)
    all_results = []
    failures = []
    for method in args.methods:
        method_dir = output_root / "predictions" / method
        method_dir.mkdir(parents=True, exist_ok=True)
        for position, row in enumerate(selected, start=1):
            destination = method_dir / f"{row['id']}.json"
            if destination.exists():
                try:
                    existing = json.loads(destination.read_text(encoding="utf-8"))
                    if existing.get("config_fingerprint") == fingerprint and not existing.get("error"):
                        all_results.append(existing)
                        continue
                except Exception:
                    pass
            try:
                if method == "retrieve_once_jev":
                    result = run_retrieve_once(row, llm, retriever, config)
                elif method == "jarv":
                    result = checker.run(to_example(row))
                    result["claim"] = row["claim"]
                    result["evidence"] = evidence_rows(checker.collected)
                else:
                    raise ValueError(f"Unknown method: {method}")
            except Exception as exc:
                result = {
                    "id": str(row["id"]),
                    "claim": row["claim"],
                    "gold": normalize_label(row["label"]),
                    "prediction": "",
                    "method": method,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            result["config_fingerprint"] = fingerprint
            atomic_json(destination, result)
            all_results.append(result)
            if result.get("error"):
                failures.append(result)
            print(json.dumps({"method": method, "processed": position, "total": len(selected), "failed": len(failures), "id": row["id"]}), flush=True)
    retriever.close()
    summaries = []
    for method in args.methods:
        valid = [row for row in all_results if row.get("method") == method and not row.get("error")]
        summaries.append(
            {
                "method": method,
                "n": len(valid),
                "accuracy": sum(row["gold"] == row["prediction"] for row in valid) / len(valid) if valid else 0.0,
                "macro_f1": macro_f1(valid) if valid else 0.0,
                "mean_search_calls": statistics.fmean(row.get("search_calls", 0) for row in valid) if valid else 0.0,
                "mean_latency_s": statistics.fmean(row.get("latency_s", 0.0) for row in valid) if valid else 0.0,
                "prediction_counts": dict(Counter(row["prediction"] for row in valid)),
                "failures": len([row for row in all_results if row.get("method") == method and row.get("error")]),
            }
        )
        gold_subset = output_root / f"gold_{method}.jsonl"
        submission = output_root / f"submission_{method}.jsonl"
        with gold_subset.open("w", encoding="utf-8") as gold_handle, submission.open("w", encoding="utf-8") as pred_handle:
            by_id = {str(row["id"]): row for row in valid}
            for source in selected:
                result = by_id.get(str(source["id"]))
                if result is None:
                    continue
                gold_handle.write(json.dumps(source, ensure_ascii=False) + "\n")
                pred_handle.write(json.dumps({"id": source["id"], "predicted_label": LABEL_TO_OFFICIAL[result["prediction"]], "predicted_evidence": official_evidence(result)}, ensure_ascii=False) + "\n")
    atomic_json(output_root / "summary.json", summaries)
    with (output_root / "failures.jsonl").open("w", encoding="utf-8") as handle:
        for failure in failures:
            handle.write(json.dumps(failure, ensure_ascii=False) + "\n")
    print(json.dumps({"event": "complete", "summary": summaries}, ensure_ascii=False), flush=True)
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
