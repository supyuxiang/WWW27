"""QuanTemp++ native three-way adapter for Retrieve-once+JEV and JARV.

The adapter preserves JARV's gate, aspect selection, query generation, search
budget, and teacher-forced JEV scoring. It changes only benchmark I/O, the
claim-independent offline retriever, temporal filtering, and the native
QuanTemp++ True/False/Conflicting final decision.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import re
import sqlite3
import statistics
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dateutil.parser import parse as parse_date

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))

from baselines.run_main_experiment import BatchedInstrumentedFactChecker, TrackedLocalLLM
from src.config import JarvConfig
from src.prompts import SYSTEM_PROMPT, _options
from src.types import AtomicQuery, Cost, Example, Passage
from src.utils.text import clip, format_evidence, format_subclaims


LABELS = ('false', 'true', 'conflicting')
LABEL_TO_OFFICIAL = {'false': 'False', 'true': 'True', 'conflicting': 'Half True/False'}
OFFICIAL_TO_LABEL = {
    'false': 'false',
    'true': 'true',
    'half true/false': 'conflicting',
    'conflicting': 'conflicting',
}
TOKEN_RE = re.compile(r'[^\\W_]+', re.UNICODE)
STOPWORDS = {
    'a', 'an', 'and', 'are', 'as', 'at', 'be', 'been', 'being', 'by', 'did', 'do', 'does',
    'for', 'from', 'had', 'has', 'have', 'he', 'her', 'hers', 'him', 'his', 'how', 'i', 'in',
    'into', 'is', 'it', 'its', 'of', 'on', 'or', 'our', 'she', 'that', 'the', 'their', 'them',
    'there', 'these', 'they', 'this', 'those', 'to', 'was', 'we', 'were', 'what', 'when', 'where',
    'which', 'who', 'why', 'will', 'with', 'would', 'you', 'your',
}
LATEST_EPOCH = 253402300799.0


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    with temporary.open('w', encoding='utf-8') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write('\n')
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    with temporary.open('w', encoding='utf-8', newline='') as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_label(value: object) -> str:
    key = str(value).strip().lower().replace('_', ' ')
    if key not in OFFICIAL_TO_LABEL:
        raise ValueError(f'Unknown QuanTemp++ label: {value!r}')
    return OFFICIAL_TO_LABEL[key]


def query_terms(text: str) -> list[str]:
    terms = []
    for token in TOKEN_RE.findall((text or '').lower()):
        if token in STOPWORDS or len(token) == 1 and token.isascii():
            continue
        terms.append(token)
    return list(dict.fromkeys(terms))


def cutoff_epoch(value: object) -> float:
    try:
        return parse_date(str(value), fuzzy=True).timestamp()
    except Exception:
        return LATEST_EPOCH


class QuanTempRetriever:
    '''Read-only all-term FTS5 BM25 over the official corpus with temporal filtering.'''

    def __init__(self, database: str | Path):
        self.database = str(database)
        self.connection = sqlite3.connect(
            f'file:{self.database}?mode=ro', uri=True, check_same_thread=False
        )
        self.cutoff = LATEST_EPOCH

    def set_cutoff(self, published: object) -> None:
        self.cutoff = cutoff_epoch(published)

    @staticmethod
    def _expression(text: str) -> str:
        escaped = [f'"{term.replace(chr(34), '')}"' for term in query_terms(text)]
        return ' OR '.join(escaped)

    def _search(
        self,
        connection: sqlite3.Connection,
        query: str,
        k: int,
        max_passage_chars: int,
        cutoff: float,
    ) -> list[Passage]:
        expression = self._expression(query)
        if not expression or k <= 0:
            return []
        rows = connection.execute(
            'SELECT d.doc_id,d.title,d.snippet,d.date_text,bm25(docs_fts,5.0,1.0) AS rank '
            'FROM docs_fts JOIN docs d ON d.rowid=docs_fts.rowid '
            'WHERE docs_fts MATCH ? AND d.date_epoch < ? ORDER BY rank LIMIT ?',
            (expression, cutoff, max(64, k * 16)),
        ).fetchall()
        passages = []
        for doc_id, title, snippet, date_text, rank in rows[:k]:
            dated_text = f'Publication date: {date_text}\n{snippet}' if date_text else str(snippet)
            passages.append(
                Passage(
                    doc_id=str(doc_id),
                    title=str(title),
                    text=clip(dated_text, max_passage_chars),
                    score=float(-rank),
                )
            )
        return passages

    def search(self, query: str, k: int, exclude_ids=None, max_passage_chars: int = 800):
        excluded = set(exclude_ids or [])
        hits = self._search(self.connection, query, k + len(excluded), max_passage_chars, self.cutoff)
        return [hit for hit in hits if hit.doc_id not in excluded][:k]

    def search_many(self, queries, k: int, exclude_ids=None, max_passage_chars: int = 800):
        excluded = set(exclude_ids or [])
        cutoff = self.cutoff
        if len(queries) <= 1:
            return [self.search(query, k, excluded, max_passage_chars) for query in queries]

        def worker(query: str):
            connection = sqlite3.connect(f'file:{self.database}?mode=ro', uri=True)
            try:
                hits = self._search(connection, query, k + len(excluded), max_passage_chars, cutoff)
                return [hit for hit in hits if hit.doc_id not in excluded][:k]
            finally:
                connection.close()

        with ThreadPoolExecutor(max_workers=min(3, len(queries))) as executor:
            return list(executor.map(worker, queries))

    def close(self) -> None:
        self.connection.close()


def quantemp_judge_prompt(claim: str, subclaims: str, evidence: str):
    pairs = (
        ('false', 'False: the available evidence contradicts the claim.'),
        ('true', 'True: the available evidence supports the claim.'),
        (
            'conflicting',
            'Conflicting: the evidence supports some material parts of the claim and contradicts others.',
        ),
    )
    listing, options = _options(pairs, False)
    user = (
        f'Original numerical claim: {claim}\n'
        f'Inspected sub-claims: {subclaims}\n'
        f'Retrieved temporally eligible web evidence: {evidence}\n'
        'Assign the native QuanTemp++ three-way veracity label using only the available evidence. '
        'Use Conflicting only when support and contradiction are both evidenced; missing evidence alone '
        'does not imply Conflicting.\n'
        f'{listing}\n'
        'Select one option. Answer only with its bracketed numeric identifier.'
    )
    return user, options


class QuanTempFactChecker(BatchedInstrumentedFactChecker):
    def _judge(self, claim, subclaims, evidence, cost):
        self._decision_stage = 'final_judge'
        started = time.perf_counter()
        try:
            user, options = quantemp_judge_prompt(
                claim,
                format_subclaims(subclaims),
                format_evidence(evidence, self.config.max_evidence_chars),
            )
            choice = self.choose(user, options, cost)
            return choice.key, choice.probs
        finally:
            self.stage_times['final_judge'] += time.perf_counter() - started

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
                    self.seen_docs,
                    self.config.max_passage_chars,
                )
            finally:
                self.stage_times['retrieval'] += time.perf_counter() - started
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
                self.seen_docs,
                self.config.max_passage_chars,
            )
        finally:
            self.stage_times['retrieval'] += time.perf_counter() - started
        cost.search_calls += len(queries)
        for query, hits in zip(queries, batches):
            for hit in hits:
                if hit.doc_id in self.seen_docs:
                    continue
                self.seen_docs.add(hit.doc_id)
                hit.query, hit.sub_claim, hit.aspect = query.query, query.sub_claim, query.aspect
                self.collected.append(hit)


def proportional_sample(rows: list[dict], limit: int, seed: int) -> list[dict]:
    if limit <= 0 or limit >= len(rows):
        return list(rows)
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[normalize_label(row['label'])].append(row)
    exact = {label: limit * len(groups[label]) / len(rows) for label in LABELS}
    quota = {label: int(math.floor(exact[label])) for label in LABELS}
    if limit >= len(LABELS):
        for label in LABELS:
            quota[label] = max(1, quota[label])
    while sum(quota.values()) > limit:
        label = max(LABELS, key=lambda item: (quota[item] - exact[item], quota[item]))
        quota[label] -= 1
    while sum(quota.values()) < limit:
        label = max(LABELS, key=lambda item: (exact[item] - quota[item], len(groups[item])))
        quota[label] += 1
    selected = []
    for label in LABELS:
        selected.extend(random.Random(f'{seed}:quantemp_plus:{label}').sample(groups[label], quota[label]))
    return sorted(selected, key=lambda row: row['__split_index'])


def to_example(row: dict) -> Example:
    return Example(
        id=str(row['id']),
        claim=str(row['claim']),
        label=normalize_label(row['label']),
        domain='quantemp_plus',
        source_id=str(row['__split_index']),
    )


def evidence_rows(passages) -> list[dict]:
    return [
        {
            'doc_id': passage.doc_id,
            'title': passage.title,
            'score': passage.score,
            'query': passage.query,
            'sub_claim': passage.sub_claim,
            'aspect': passage.aspect,
            'text': passage.text,
        }
        for passage in passages
    ]


def run_retrieve_once(row, llm, retriever, config):
    started = time.perf_counter()
    llm.reset_counters()
    passages = retriever.search(row['claim'], config.passages_per_query, None, config.max_passage_chars)
    user, options = quantemp_judge_prompt(
        row['claim'], 'none', format_evidence(passages, config.max_evidence_chars)
    )
    choice = llm.score_options(SYSTEM_PROMPT, user, options)
    return {
        'id': str(row['id']),
        'split_index': row['__split_index'],
        'claim': row['claim'],
        'gold': normalize_label(row['label']),
        'prediction': choice.key,
        'method': 'retrieve_once_jev',
        'evidence': evidence_rows(passages),
        'judge_probs': choice.probs,
        'search_calls': 1,
        'retrieval_rounds': 1,
        'input_tokens': choice.usage.input_tokens,
        'output_tokens': choice.usage.output_tokens,
        'query_generation_tokens': 0,
        'total_generated_tokens': 0,
        'latency_s': time.perf_counter() - started,
        'stop_reason': 'retrieve_once',
        'error': None,
    }


def class_metrics(rows: list[dict]) -> tuple[float, dict[str, dict[str, float | int]]]:
    output = {}
    f1_values = []
    for label in LABELS:
        tp = sum(row['gold'] == label and row['prediction'] == label for row in rows)
        fp = sum(row['gold'] != label and row['prediction'] == label for row in rows)
        fn = sum(row['gold'] == label and row['prediction'] != label for row in rows)
        support = sum(row['gold'] == label for row in rows)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        f1_values.append(f1)
        output[label] = {'precision': precision, 'recall': recall, 'f1': f1, 'support': support}
    return sum(f1_values) / len(f1_values), output


def retrieval_metrics(rows: list[dict], qrels: dict[str, dict[str, int]]) -> dict[str, float]:
    cutoffs = (1, 3, 5, 10)
    totals = defaultdict(float)
    evaluated = 0
    for row in rows:
        relevant = {str(doc_id) for doc_id, rel in qrels.get(str(row['split_index']), {}).items() if rel}
        if not relevant:
            continue
        ranked = list(dict.fromkeys(str(item['doc_id']) for item in row.get('evidence') or []))
        evaluated += 1
        first = next((rank + 1 for rank, doc_id in enumerate(ranked) if doc_id in relevant), None)
        totals['mrr'] += 1.0 / first if first else 0.0
        for k in cutoffs:
            top = ranked[:k]
            hits = sum(doc_id in relevant for doc_id in top)
            totals[f'recall@{k}'] += hits / len(relevant)
            totals[f'success@{k}'] += float(hits > 0)
            precision_sum = 0.0
            seen_hits = 0
            dcg = 0.0
            for rank, doc_id in enumerate(top, start=1):
                if doc_id in relevant:
                    seen_hits += 1
                    precision_sum += seen_hits / rank
                    dcg += 1.0 / math.log2(rank + 1)
            totals[f'map@{k}'] += precision_sum / min(len(relevant), k)
            ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, min(len(relevant), k) + 1))
            totals[f'ndcg@{k}'] += dcg / ideal if ideal else 0.0
    result = {key: value / evaluated if evaluated else 0.0 for key, value in totals.items()}
    result['evaluated_claims'] = evaluated
    return dict(sorted(result.items()))


def summarize(method: str, rows: list[dict], qrels: dict[str, dict[str, int]]) -> dict[str, Any]:
    macro_f1, per_class = class_metrics(rows)
    return {
        'method': method,
        'n': len(rows),
        'accuracy': sum(row['gold'] == row['prediction'] for row in rows) / len(rows) if rows else 0.0,
        'macro_f1': macro_f1 if rows else 0.0,
        'per_class': per_class,
        'retrieval': retrieval_metrics(rows, qrels),
        'mean_search_calls': statistics.fmean(row.get('search_calls', 0) for row in rows) if rows else 0.0,
        'mean_latency_s': statistics.fmean(row.get('latency_s', 0.0) for row in rows) if rows else 0.0,
        'total_input_tokens': sum(row.get('input_tokens', 0) for row in rows),
        'total_output_tokens': sum(row.get('output_tokens', 0) for row in rows),
        'total_query_generation_tokens': sum(row.get('query_generation_tokens', 0) for row in rows),
        'prediction_counts': dict(Counter(row['prediction'] for row in rows)),
    }


def write_summary_csv(path: Path, summaries: list[dict]) -> None:
    fields = [
        'method', 'n', 'accuracy', 'macro_f1', 'mean_search_calls', 'mean_latency_s',
        'total_input_tokens', 'total_output_tokens', 'total_query_generation_tokens', 'failures',
    ]
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    with temporary.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for summary in summaries:
            writer.writerow({field: summary.get(field) for field in fields})
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--split-file', required=True)
    parser.add_argument('--qrels', required=True)
    parser.add_argument('--index', required=True)
    parser.add_argument('--model', default='/root/autodl-tmp/models/Qwen2.5-3B-Instruct')
    parser.add_argument('--output-root', required=True)
    parser.add_argument('--limit', type=int, default=20)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--methods', nargs='+', default=['retrieve_once_jev', 'jarv'])
    parser.add_argument('--source-revision', default='7ece14e3dcf3e6e8195892c42cd88fe2d4de6e56')
    parser.add_argument('--dataset-zip-sha256', default='45571b9e137eb94da14537a8d031f460fa1d1943bad3c8337d4634d4fa302584')
    args = parser.parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    with Path(args.split_file).open(encoding='utf-8') as handle:
        source_rows = json.load(handle)
    rows = [dict(row, __split_index=index) for index, row in enumerate(source_rows)]
    selected = proportional_sample(rows, args.limit, args.seed)
    with Path(args.qrels).open(encoding='utf-8') as handle:
        qrels = json.load(handle)
    config_payload = {
        'track': 'quantemp_plus_native_three_way_adapted',
        'adaptation': 'SQLite FTS5 BM25 replaces incomplete official Elasticsearch/Contriever runtime; native corpus, temporal cutoff, labels, qrels, and metrics retained',
        'methods': args.methods,
        'model': args.model,
        'model_frozen': True,
        'chat_template': 'tokenizer.apply_chat_template',
        'enable_thinking': False,
        'seed': args.seed,
        'limit': args.limit,
        'split_size': len(rows),
        'sample_ids': [str(row['id']) for row in selected],
        'sample_split_indices': [row['__split_index'] for row in selected],
        'sample_label_counts': dict(Counter(normalize_label(row['label']) for row in selected)),
        'labels': list(LABELS),
        'top_k_per_query': 3,
        'search_cap': 3,
        'tau_g': 0.5,
        'tau_a': 0.5,
        'max_aspects': 3,
        'index': args.index,
        'index_sha256': sha256_file(Path(args.index)),
        'split_file': args.split_file,
        'split_sha256': sha256_file(Path(args.split_file)),
        'qrels': args.qrels,
        'qrels_sha256': sha256_file(Path(args.qrels)),
        'adapter_sha256': sha256_file(Path(__file__)),
        'official_source_revision': args.source_revision,
        'dataset_zip_sha256': args.dataset_zip_sha256,
        'retriever': 'all non-stopword terms; SQLite FTS5 BM25 title weight 5; corpus date strictly before claim published date',
        'gold_evidence_used_for_retrieval': False,
        'precomputed_queries_used': False,
        'precomputed_search_results_used': False,
        'retrieval_metric_implementation': 'standard binary qrels formulas reconstructed because official repository imports an unshipped evaluator module',
    }
    fingerprint = hashlib.sha256(json.dumps(config_payload, sort_keys=True).encode()).hexdigest()
    config_payload['config_fingerprint'] = fingerprint
    started_at = datetime.now(timezone.utc).isoformat()
    existing_config = output_root / 'config.json'
    if existing_config.exists():
        try:
            prior = json.loads(existing_config.read_text(encoding='utf-8'))
            if prior.get('config_fingerprint') == fingerprint:
                started_at = prior.get('started_at_utc', started_at)
        except Exception:
            pass
    config_payload['started_at_utc'] = started_at
    atomic_json(output_root / 'config.json', config_payload)
    atomic_json(
        output_root / 'sample_manifest.json',
        {
            'source_split': args.split_file,
            'source_size': len(rows),
            'selected': [
                {'id': row['id'], 'split_index': row['__split_index'], 'label': row['label']}
                for row in selected
            ],
        },
    )
    config = JarvConfig(
        model_path=args.model,
        method='jarv',
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
    retriever = QuanTempRetriever(args.index)
    llm = TrackedLocalLLM(args.model)
    checker = QuanTempFactChecker(llm, retriever, config)
    all_results = []
    failures = []
    for method in args.methods:
        method_dir = output_root / 'predictions' / method
        method_dir.mkdir(parents=True, exist_ok=True)
        for position, row in enumerate(selected, start=1):
            destination = method_dir / f'{row['id']}.json'
            if destination.exists():
                try:
                    existing = json.loads(destination.read_text(encoding='utf-8'))
                    if existing.get('config_fingerprint') == fingerprint and not existing.get('error'):
                        all_results.append(existing)
                        continue
                except Exception:
                    pass
            retriever.set_cutoff(row.get('published'))
            try:
                if method == 'retrieve_once_jev':
                    result = run_retrieve_once(row, llm, retriever, config)
                elif method == 'jarv':
                    result = checker.run(to_example(row))
                    result['claim'] = row['claim']
                    result['split_index'] = row['__split_index']
                    result['evidence'] = evidence_rows(checker.collected)
                else:
                    raise ValueError(f'Unknown method: {method}')
                result['correct'] = result['gold'] == result['prediction']
            except Exception as exc:
                result = {
                    'id': str(row['id']),
                    'split_index': row['__split_index'],
                    'claim': row['claim'],
                    'gold': normalize_label(row['label']),
                    'prediction': '',
                    'method': method,
                    'error': f'{type(exc).__name__}: {exc}',
                }
            result['config_fingerprint'] = fingerprint
            atomic_json(destination, result)
            all_results.append(result)
            if result.get('error'):
                failures.append(result)
            print(
                json.dumps(
                    {'method': method, 'processed': position, 'total': len(selected), 'failed': len(failures), 'id': row['id']}
                ),
                flush=True,
            )
    retriever.close()
    summaries = []
    for method in args.methods:
        valid = [row for row in all_results if row.get('method') == method and not row.get('error')]
        summary = summarize(method, valid, qrels)
        summary['failures'] = sum(row.get('method') == method and bool(row.get('error')) for row in all_results)
        summaries.append(summary)
        submission_lines = []
        for row in valid:
            submission_lines.append(
                json.dumps(
                    {
                        'id': row['id'],
                        'split_index': row['split_index'],
                        'prediction': LABEL_TO_OFFICIAL[row['prediction']],
                        'retrieved_doc_ids': [item['doc_id'] for item in row.get('evidence') or []],
                    },
                    ensure_ascii=False,
                )
            )
        atomic_text(output_root / f'submission_{method}.jsonl', '\n'.join(submission_lines) + '\n')
    atomic_json(output_root / 'summary.json', summaries)
    write_summary_csv(output_root / 'summary.csv', summaries)
    atomic_text(
        output_root / 'failures.jsonl',
        ''.join(json.dumps(failure, ensure_ascii=False) + '\n' for failure in failures),
    )
    config_payload['finished_at_utc'] = datetime.now(timezone.utc).isoformat()
    config_payload['status'] = 'complete' if not failures else 'complete_with_failures'
    atomic_json(output_root / 'config.json', config_payload)
    report = [
        '# QuanTemp++ adapted evaluation',
        '',
        f'Official test split rows selected: {len(selected)} of {len(rows)}.',
        'The official corpus, native three-way labels, temporal cutoff, and qrels are retained.',
        "Retrieval is adapted to all-term SQLite FTS5 BM25 because the official repository's runnable dependencies/evaluator are incomplete.",
        'Gold qrels, precomputed oracle queries, and stored search results were not used to drive retrieval.',
        '',
        '| Method | n | Accuracy | Macro-F1 | Mean searches | Mean latency (s) | Failures |',
        '|---|---:|---:|---:|---:|---:|---:|',
    ]
    for summary in summaries:
        report.append(
            f'| {summary['method']} | {summary['n']} | {summary['accuracy']:.4f} | {summary['macro_f1']:.4f} | '
            f'{summary['mean_search_calls']:.3f} | {summary['mean_latency_s']:.3f} | {summary['failures']} |'
        )
    atomic_text(output_root / 'report.md', '\n'.join(report) + '\n')
    print(json.dumps({'event': 'complete', 'summary': summaries}, ensure_ascii=False), flush=True)
    return int(bool(failures))


if __name__ == '__main__':
    raise SystemExit(main())
