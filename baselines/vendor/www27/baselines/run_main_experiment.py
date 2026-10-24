'''Formal WWW27/JARV offline-retrieval experiment with resumable atomic outputs.'''

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
import platform
import random
import re
import shutil
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL_ROOT = Path('/root/www27')
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))
if str(ROOT / 'baselines') not in sys.path:
    sys.path.insert(0,str(ROOT / 'baselines'))

from baselines.jev_only import load_benchmark
from src.config import JarvConfig
from src.data import read_jsonl
from src.llm import LocalLLM
from src.pipeline import FactChecker
from src.prompts import SYSTEM_PROMPT, judge_prompt
from src.retrieval import BM25Retriever
from src.retrieval_sqlite import SQLiteBM25Retriever
from src.types import (
    AtomicQuery,
    Choice,
    Document,
    Example,
    Generation,
    Option,
    Usage,
    add_usage,
)
from src.utils.text import clip, format_evidence, is_leakage

LOGGER = logging.getLogger('main_experiment')

BENCHMARKS = ('averitec', 'fever', 'scifact')
METHODS = ('llm_jev', 'retrieve_once_jev', 'jarv', 'jarv_gen')
BENCHMARK_ROOT = Path('/root/autodl-tmp/www27/benchmark')
INDEX_ROOT = Path('/root/autodl-tmp/www27/index')
DEFAULT_MODEL = '/root/autodl-tmp/models/Qwen2.5-3B-Instruct'
REUSED_JEV_ROOT = Path(
    '/root/autodl-tmp/exp_results/model_sweep_20261019_020225/'
    'Qwen2.5-3B-instruct/jev'
)


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    with temporary.open('w', encoding='utf-8') as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_text(path, json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def dataset_digest(rows: list[dict]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        payload = [str(row['id']), str(row['gold']), str(row['claim'])]
        digest.update(json.dumps(payload, ensure_ascii=False).encode('utf-8'))
        digest.update(b'\n')
    return digest.hexdigest()


def proportional_stratified_sample(
    rows: list[dict], limit: int, seed: int, benchmark: str
) -> list[dict]:
    """Deterministically preserve each binary split's observed label ratio."""
    if limit <= 0 or limit >= len(rows):
        return sorted(rows, key=lambda row: str(row['id']))
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[str(row['gold'])].append(row)
    exact = {key: limit * len(value) / len(rows) for key, value in groups.items()}
    quotas = {key: int(value) for key, value in exact.items()}
    remainder = limit - sum(quotas.values())
    order = sorted(groups, key=lambda key: (-(exact[key] - quotas[key]), key))
    for key in order[:remainder]:
        quotas[key] += 1
    sampled = []
    for key in sorted(groups):
        rng = random.Random(f'{seed}:{benchmark}:{key}')
        sampled.extend(rng.sample(groups[key], quotas[key]))
    return sorted(sampled, key=lambda row: str(row['id']))


def config_fingerprint(config: dict[str, Any]) -> str:
    stable = {key: value for key, value in config.items() if key not in {'started_at', 'output_root'}}
    encoded = json.dumps(stable, sort_keys=True, ensure_ascii=False).encode('utf-8')
    return hashlib.sha256(encoded).hexdigest()


def sample_path(root: Path, method: str, benchmark: str, item_id: str) -> Path:
    safe = re.sub(r'[^A-Za-z0-9_.-]+', '_', str(item_id))[:100]
    suffix = hashlib.sha1(str(item_id).encode()).hexdigest()[:10]
    return root / 'predictions' / method / benchmark / f'{safe}__{suffix}.json'


def is_complete(path: Path, fingerprint: str) -> bool:
    if not path.exists():
        return False
    try:
        row = json.loads(path.read_text(encoding='utf-8'))
    except Exception:
        return False
    return (
        row.get('config_fingerprint') == fingerprint
        and row.get('error') in {None, ''}
        and row.get('prediction') in {'real', 'fake'}
    )


REUSE_CONFIG_KEYS = (
    'model',
    'model_frozen',
    'seed',
    'chat_template',
    'enable_thinking',
    'deterministic_generation',
    'temperature',
    'tau_g',
    'tau_a',
    'search_cap',
    'max_aspects_per_round',
    'queries_per_aspect',
    'top_k',
    'max_claim_chars',
    'max_passage_chars',
    'max_evidence_chars',
    'query_max_new_tokens',
    'decision_max_new_tokens',
    'json_retries',
    'fever_query_terms',
    'retrieval_indexes',
)


def import_compatible_predictions(
    source_root: Path,
    output_root: Path,
    datasets: dict[str, list[dict]],
    methods: list[str],
    config: dict,
    fingerprint: str,
) -> dict:
    '''Import only successful rows whose sample-level inference config matches.'''
    source_config_path = source_root / 'config.json'
    if not source_config_path.exists():
        raise RuntimeError(f'Reuse source has no config.json: {source_root}')
    source_config = json.loads(source_config_path.read_text(encoding='utf-8'))
    mismatched = [
        key
        for key in REUSE_CONFIG_KEYS
        if source_config.get(key) != config.get(key)
    ]
    if mismatched:
        raise RuntimeError(
            f'Incompatible reuse source {source_root}; mismatched: {', '.join(mismatched)}'
        )
    imported: dict[str, dict[str, int]] = {}
    source_methods = set(source_config.get('methods') or [])
    for method in methods:
        if method not in source_methods:
            continue
        imported[method] = {}
        for benchmark, expected_rows in datasets.items():
            expected_ids = {str(row['id']) for row in expected_rows}
            count = 0
            source_dir = source_root / 'predictions' / method / benchmark
            if not source_dir.exists():
                imported[method][benchmark] = 0
                continue
            for path in source_dir.glob('*.json'):
                try:
                    row = json.loads(path.read_text(encoding='utf-8'))
                except Exception:
                    continue
                item_id = str(row.get('id', ''))
                if (
                    item_id not in expected_ids
                    or row.get('error') not in {None, ''}
                    or row.get('prediction') not in {'real', 'fake'}
                ):
                    continue
                target = sample_path(output_root, method, benchmark, item_id)
                if is_complete(target, fingerprint):
                    continue
                row['config_fingerprint'] = fingerprint
                row['checkpoint_reused_from'] = str(source_root)
                atomic_write_json(target, row)
                count += 1
            imported[method][benchmark] = count
    return {
        'source': str(source_root),
        'source_config_fingerprint': source_config.get('config_fingerprint'),
        'imported': imported,
    }


def load_scifact_corpus() -> list[Document]:
    rows = read_jsonl(BENCHMARK_ROOT / 'scifact' / 'data' / 'corpus.jsonl')
    documents = []
    for row in rows:
        doc_id = str(row.get('doc_id') or row.get('id') or '')
        title = str(row.get('title') or '')
        abstract = row.get('abstract') or row.get('text') or ''
        if isinstance(abstract, list):
            abstract = ' '.join(str(part) for part in abstract)
        body = str(abstract).strip()
        if doc_id and body:
            documents.append(Document(doc_id=doc_id, title=title, text=body))
    return documents


class RetrieverBundle:
    '''One fixed retrieval backend per benchmark; no gold evidence is loaded.'''

    def __init__(self, args=None):
        self.fever = SQLiteBM25Retriever(INDEX_ROOT / 'fever_fts.sqlite')
        self.averitec = SQLiteBM25Retriever(
            INDEX_ROOT / 'averitec_dev_fts.sqlite', claim_scoped=True
        )
        self.scifact = BM25Retriever(load_scifact_corpus())

    @staticmethod
    def averitec_scope(item_id: str) -> str:
        value = str(item_id)
        if value.isdigit():
            return value
        match = re.fullmatch(r'row-(\\d+)', value)
        if not match:
            raise ValueError(f'Cannot map AVeriTeC id to knowledge-store scope: {item_id}')
        return match.group(1)

    def for_example(self, benchmark: str, item_id: str):
        if benchmark == 'averitec':
            self.averitec.set_claim(self.averitec_scope(item_id))
            return self.averitec
        if benchmark == 'fever':
            return self.fever
        if benchmark == 'scifact':
            return self.scifact
        raise ValueError(benchmark)

    def close(self) -> None:
        self.fever.close()
        self.averitec.close()


class TrackedLocalLLM(LocalLLM):
    def reset_counters(self) -> None:
        self.query_generation_tokens = 0
        self.total_generated_tokens = 0

    def generate(self, system: str, user: str, max_new_tokens: int) -> Generation:
        generation = super().generate(system, user, max_new_tokens)
        self.query_generation_tokens += generation.usage.output_tokens
        self.total_generated_tokens += generation.usage.output_tokens
        return generation

    def generate_many(
        self, system: str, users: list[str], max_new_tokens: int
    ) -> list[Generation]:
        generations = super().generate_many(system, users, max_new_tokens)
        generated = sum(item.usage.output_tokens for item in generations)
        self.query_generation_tokens += generated
        self.total_generated_tokens += generated
        return generations


def parse_generated_choice(text: str, options: list[Option]) -> str | None:
    '''Parse a generated decision without reading labels or external state.'''
    matches = re.findall(r'\\[\\s*(\\d+)\\s*\\]', text or '')
    if matches:
        wanted = matches[-1]
        for option in options:
            if re.sub(r'\\D', '', option.suffix) == wanted:
                return option.key
    lowered = (text or '').lower()
    positions = []
    for option in options:
        hits = list(re.finditer(rf'\b{re.escape(option.key.lower())}\b', lowered))
        if hits:
            positions.append((hits[-1].start(), option.key))
    return max(positions)[1] if positions else None


class GenerativeAdapter:
    '''Expose the FactChecker LLM interface using deterministic vLLM generation.'''

    def __init__(self, generator, decision_max_new_tokens: int = 24):
        self.generator = generator
        self.tokenizer = generator.tokenizer
        self.decision_max_new_tokens = decision_max_new_tokens
        self.reset_counters()

    def reset_counters(self) -> None:
        self.query_generation_tokens = 0
        self.decision_generation_tokens = 0
        self.total_generated_tokens = 0
        self.last_parse_failures = 0
        self.last_fallback = False
        self.last_attempts = 1

    def _tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text or '', add_special_tokens=False))

    def generate(self, system: str, user: str, max_new_tokens: int) -> Generation:
        return self.generate_many(system, [user], max_new_tokens)[0]

    def generate_many(
        self, system: str, users: list[str], max_new_tokens: int
    ) -> list[Generation]:
        texts = self.generator.generate_batch(users, max_new_tokens)
        generations = []
        for user, text in zip(users, texts):
            output_tokens = self._tokens(text)
            generations.append(
                Generation(
                    text=text,
                    usage=Usage(
                        forward_passes=1,
                        input_tokens=self._tokens(self.generator._render(user)),
                        output_tokens=output_tokens,
                    ),
                )
            )
        generated = sum(item.usage.output_tokens for item in generations)
        self.query_generation_tokens += generated
        self.total_generated_tokens += generated
        return generations

    def score_options(self, system: str, user: str, options: list[Option]) -> Choice:
        prompts = [user, user + '\nReply now with exactly one bracketed option number, such as [1].']
        raw_parts = []
        selected = None
        input_tokens = 0
        output_tokens = 0
        attempts = 0
        for prompt in prompts:
            attempts += 1
            text = self.generator.generate_batch([prompt], self.decision_max_new_tokens)[0]
            raw_parts.append(text)
            input_tokens += self._tokens(self.generator._render(prompt))
            output_tokens += self._tokens(text)
            selected = parse_generated_choice(text, options)
            if selected is not None:
                break
        self.last_attempts = attempts
        self.last_parse_failures = attempts - int(selected is not None)
        self.last_fallback = selected is None
        if selected is None:
            keys = {option.key for option in options}
            for conservative in ('retrieve', 'check', 'real'):
                if conservative in keys:
                    selected = conservative
                    break
            selected = selected or options[0].key
        self.decision_generation_tokens += output_tokens
        self.total_generated_tokens += output_tokens
        probs = {option.key: float(option.key == selected) for option in options}
        return Choice(
            key=selected,
            probs=probs,
            scores=dict(probs),
            parsed=not self.last_fallback,
            raw_text='\n--- retry ---\n'.join(raw_parts),
            usage=Usage(
                forward_passes=attempts,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            ),
        )


class SeededVLLMGenerator:
    '''Small deterministic generator with an explicit engine seed.'''

    def __init__(
        self,
        model_path: str,
        seed: int,
        gpu_memory_utilization: float,
        max_model_len: int,
    ):
        os.environ.setdefault('VLLM_USE_FLASHINFER_SAMPLER', '0')
        from vllm import LLM, SamplingParams

        self.SamplingParams = SamplingParams
        self.model_path = model_path
        self.enable_thinking = False
        self.use_chat_template = True
        self.is_qwen3 = 'qwen3' in Path(model_path.rstrip('/')).name.lower()
        self.llm = LLM(
            model=model_path,
            tensor_parallel_size=1,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            dtype='bfloat16',
            seed=seed,
        )
        self.tokenizer = self.llm.get_tokenizer()
        if not getattr(self.tokenizer, 'chat_template', None):
            raise ValueError(f'Tokenizer has no chat template: {model_path}')

    def _render(self, user: str) -> str:
        messages = [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': user},
        ]
        if self.is_qwen3:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    def generate_batch(self, prompts: list[str], max_new_tokens: int) -> list[str]:
        rendered = [self._render(prompt) for prompt in prompts]
        params = self.SamplingParams(
            temperature=0.0,
            max_tokens=max_new_tokens,
            skip_special_tokens=True,
        )
        outputs = self.llm.generate(rendered, params, use_tqdm=False)
        return [output.outputs[0].text.strip() for output in outputs]


class InstrumentedFactChecker(FactChecker):
    '''Add stage timings and generated-choice traces without changing JARV logic.'''

    def run(self, example: Example) -> dict:
        self.stage_times = defaultdict(float)
        self.decision_traces = []
        self._decision_stage = ''
        if hasattr(self.llm, 'reset_counters'):
            self.llm.reset_counters()
        result = super().run(example)
        result['stage_times_s'] = dict(self.stage_times)
        result['decision_traces'] = self.decision_traces
        result['query_generation_tokens'] = int(
            getattr(self.llm, 'query_generation_tokens', 0)
        )
        result['total_generated_tokens'] = int(
            getattr(self.llm, 'total_generated_tokens', 0)
        )
        return result

    def choose(self, user: str, options, cost) -> Choice:
        choice = super().choose(user, options, cost)
        if hasattr(self.llm, 'last_attempts'):
            cost.retries += max(0, int(self.llm.last_attempts) - 1)
            cost.parse_failures += int(self.llm.last_parse_failures)
            cost.fallbacks += int(self.llm.last_fallback)
        self.decision_traces.append(
            {
                'stage': self._decision_stage,
                'selected': choice.key,
                'probabilities': choice.probs,
                'parsed': choice.parsed,
                'raw': choice.raw_text,
            }
        )
        return choice

    def _timed(self, key: str, function, *args):
        started = time.perf_counter()
        try:
            return function(*args)
        finally:
            self.stage_times[key] += time.perf_counter() - started

    def _decide_retrieve(self, *args):
        self._decision_stage = 'gate'
        return self._timed('gate', super()._decide_retrieve, *args)

    def _select_aspects(self, *args):
        self._decision_stage = 'aspect_selection'
        return self._timed('aspect_selection', super()._select_aspects, *args)

    def _aspect_queries(self, *args):
        return self._timed('query_generation', super()._aspect_queries, *args)

    def retrieve(self, *args):
        return self._timed('retrieval', super().retrieve, *args)

    def _judge(self, *args):
        self._decision_stage = 'final_judge'
        return self._timed('final_judge', super()._judge, *args)


class BatchedInstrumentedFactChecker(InstrumentedFactChecker):
    '''Keep original model calls; parallelize only exact read-only lookups.'''

    def _retrieve_batched(self, example: Example, queries: list[AtomicQuery], cost) -> None:
        if not queries:
            return
        if not hasattr(self.retriever, 'search_many'):
            for query in queries:
                self.retrieve(example, [query], cost)
            return
        blocked = {example.id, example.source_id}
        started = time.perf_counter()
        try:
            batches = self.retriever.search_many(
                [query.query for query in queries],
                k=self.config.passages_per_query,
                exclude_ids=blocked,
                max_passage_chars=self.config.max_passage_chars,
            )
        finally:
            self.stage_times['retrieval'] += time.perf_counter() - started
        cost.search_calls += len(queries)
        for query, hits in zip(queries, batches):
            for hit in hits:
                if hit.doc_id in self.seen_docs:
                    continue
                if is_leakage(example.claim, blocked, hit.doc_id, hit.text):
                    continue
                self.seen_docs.add(hit.doc_id)
                hit.query = query.query
                hit.sub_claim = query.sub_claim
                hit.aspect = query.aspect
                self.collected.append(hit)
                if sum(len(item.text) for item in self.collected) >= self.config.max_evidence_chars:
                    break

    def _search_selected(
        self,
        example: Example,
        claim: str,
        aspects: list[str],
        evidence_text: str,
        cost,
    ) -> list[AtomicQuery]:
        remaining = self.config.search_cap - cost.search_calls
        selected = aspects[:remaining]
        if len(selected) < len(aspects):
            cost.search_truncated = True
        if not selected:
            return []
        queries = []
        for key in selected:
            queries.extend(self._aspect_queries(claim, key, evidence_text, cost))
        queries = queries[:remaining]
        if not queries and remaining > 0:
            cost.fallbacks += 1
            queries = [AtomicQuery('holistic', clip(claim, 240), clip(claim, 180), True)]
        self._retrieve_batched(example, queries, cost)
        return queries


def as_example(benchmark: str, row: dict) -> Example:
    return Example(
        id=str(row['id']),
        claim=str(row['claim']),
        label=str(row['gold']),
        domain=benchmark,
        source_id=str(row['id']),
    )


def evidence_rows(passages) -> list[dict[str, Any]]:
    return [
        {
            'doc_id': passage.doc_id,
            'source': passage.title,
            'title': passage.title,
            'score': passage.score,
            'query': passage.query,
            'sub_claim': passage.sub_claim,
            'aspect': passage.aspect,
            'text': passage.text,
            'text_summary': clip(passage.text.replace('\n', ' '), 240),
        }
        for passage in passages
    ]


def run_retrieve_once(
    row: dict,
    benchmark: str,
    llm: TrackedLocalLLM,
    retriever,
    args,
) -> dict:
    started = time.perf_counter()
    stage_times: dict[str, float] = {}
    error = None
    prediction = ''
    probs = {}
    passages = []
    search_calls = 0
    input_tokens = output_tokens = forward_passes = 0
    try:
        search_started = time.perf_counter()
        passages = retriever.search(
            row['claim'],
            k=args.top_k,
            exclude_ids={str(row['id'])},
            max_passage_chars=args.max_passage_chars,
        )
        search_calls = 1
        stage_times['retrieval'] = time.perf_counter() - search_started
        for passage in passages:
            passage.query = row['claim']
            passage.aspect = 'original_claim'
        judge_started = time.perf_counter()
        user, options = judge_prompt(
            row['claim'],
            'none',
            format_evidence(passages, args.max_evidence_chars),
        )
        choice = llm.score_options(SYSTEM_PROMPT, user, options)
        stage_times['final_judge'] = time.perf_counter() - judge_started
        prediction = choice.key
        probs = choice.probs
        input_tokens = choice.usage.input_tokens
        output_tokens = choice.usage.output_tokens
        forward_passes = choice.usage.forward_passes
    except Exception as exc:
        LOGGER.exception('retrieve_once_jev failed for %s/%s', benchmark, row['id'])
        error = f'{type(exc).__name__}: {exc}'
    return {
        'benchmark': benchmark,
        'id': str(row['id']),
        'claim': row['claim'],
        'gold': row['gold'],
        'raw_label': row.get('raw_label'),
        'prediction': prediction,
        'method': 'retrieve_once_jev',
        'judge_probs': probs,
        'gate_probs': [],
        'gate_prob_retrieve': None,
        'stop_reason': 'single_retrieval',
        'aspects': [],
        'aspect_probs': {},
        'queries': [
            {
                'aspect': 'original_claim',
                'sub_claim': '',
                'query': row['claim'],
                'fallback': False,
            }
        ],
        'evidence': evidence_rows(passages),
        'search_calls': search_calls,
        'retrieval_rounds': int(search_calls > 0),
        'forward_passes': forward_passes,
        'input_tokens': input_tokens,
        'output_tokens': output_tokens,
        'query_generation_tokens': 0,
        'total_generated_tokens': 0,
        'latency_s': time.perf_counter() - started,
        'stage_times_s': stage_times,
        'parse_failures': 0,
        'fallbacks': 0,
        'retries': 0,
        'retrieval_failures': int(bool(error) and search_calls == 0),
        'error': error,
    }


def enrich_pipeline_row(result: dict, raw: dict, benchmark: str, method: str) -> dict:
    result.update(
        {
            'benchmark': benchmark,
            'claim': raw['claim'],
            'raw_label': raw.get('raw_label'),
            'method': method,
            'retrieval_failures': int(
                bool(result.get('error')) and not result.get('search_calls')
            ),
        }
    )
    for evidence in result.get('evidence', []):
        evidence['source'] = evidence.get('title') or evidence.get('url') or ''
        evidence['text_summary'] = clip(
            str(evidence.get('text') or '').replace('\n', ' '), 240
        )
    return result


def import_reused_jev(
    output_root: Path,
    benchmark: str,
    expected: list[dict],
    fingerprint: str,
) -> dict[str, Any]:
    source = REUSED_JEV_ROOT / f'jev_only_{benchmark}.jsonl'
    source_rows = read_jsonl(source)
    by_id = {str(row['id']): row for row in source_rows}
    if len(by_id) != len(source_rows):
        raise RuntimeError(f'Duplicate ids in reused JEV output: {source}')
    imported = 0
    for item in expected:
        old = by_id.get(str(item['id']))
        if old is None:
            raise RuntimeError(f'Missing reused JEV id {benchmark}/{item['id']}')
        if old.get('gold') != item['gold'] or old.get('claim') != item['claim']:
            raise RuntimeError(f'Reused JEV dataset mismatch at {benchmark}/{item['id']}')
        target = sample_path(output_root, 'llm_jev', benchmark, str(item['id']))
        if is_complete(target, fingerprint):
            continue
        row = dict(old)
        row.update(
            {
                'method': 'llm_jev',
                'reused': True,
                'reused_from': str(source),
                'stop_reason': 'no_retrieval',
                'gate_probs': [],
                'aspect_probs': {},
                'queries': [],
                'evidence': [],
                'retrieval_rounds': 0,
                'query_generation_tokens': 0,
                'total_generated_tokens': 0,
                'stage_times_s': {'final_judge': old.get('latency_s', 0.0)},
                'parse_failures': 0,
                'fallbacks': 0,
                'retries': 0,
                'retrieval_failures': 0,
                'config_fingerprint': fingerprint,
            }
        )
        atomic_write_json(target, row)
        imported += 1
    extra = set(by_id) - {str(item['id']) for item in expected}
    if extra and len(expected) == len(source_rows):
        raise RuntimeError(f'Unexpected ids in reused JEV output: {benchmark}')
    return {'source': str(source), 'sha256': sha256_file(source), 'imported': imported}


def run_jev_methods(output_root: Path, datasets: dict[str, list[dict]], args, fingerprint: str) -> None:
    methods = [name for name in args.methods if name in {'retrieve_once_jev', 'jarv'}]
    if not methods:
        return
    llm = TrackedLocalLLM(args.model, use_chat_template=True)
    retrievers = RetrieverBundle(args)
    try:
        for benchmark in args.benchmarks:
            rows = datasets[benchmark]
            for method in methods:
                failures = 0
                for index, raw in enumerate(rows, start=1):
                    target = sample_path(output_root, method, benchmark, str(raw['id']))
                    if is_complete(target, fingerprint):
                        continue
                    retriever = retrievers.for_example(benchmark, str(raw['id']))
                    if method == 'retrieve_once_jev':
                        result = run_retrieve_once(raw, benchmark, llm, retriever, args)
                    else:
                        config = JarvConfig(
                            model_path=args.model,
                            method='jarv',
                            tau_g=args.tau_g,
                            tau_a=args.tau_a,
                            max_aspects=args.max_aspects,
                            queries_per_aspect=args.queries_per_aspect,
                            passages_per_query=args.top_k,
                            search_cap=args.search_cap,
                            max_claim_chars=args.max_claim_chars,
                            max_passage_chars=args.max_passage_chars,
                            max_evidence_chars=args.max_evidence_chars,
                            query_max_new_tokens=args.query_max_new_tokens,
                            json_retries=args.json_retries,
                            seed=args.seed,
                        )
                        checker = BatchedInstrumentedFactChecker(llm, retriever, config)
                        result = enrich_pipeline_row(
                            checker.run(as_example(benchmark, raw)), raw, benchmark, method
                        )
                    result['config_fingerprint'] = fingerprint
                    atomic_write_json(target, result)
                    failures += int(bool(result.get('error')))
                    if index % args.log_every == 0 or index == len(rows):
                        LOGGER.info(
                            '%s/%s processed=%d total=%d failures=%d remaining=%d',
                            method,
                            benchmark,
                            index,
                            len(rows),
                            failures,
                            len(rows) - index,
                        )
    finally:
        retrievers.close()
        del llm
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass


def run_gen_method(output_root: Path, datasets: dict[str, list[dict]], args, fingerprint: str) -> None:
    if 'jarv_gen' not in args.methods:
        return
    generator = SeededVLLMGenerator(
        model_path=args.model,
        seed=args.seed,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
    )
    llm = GenerativeAdapter(generator, decision_max_new_tokens=args.decision_max_new_tokens)
    retrievers = RetrieverBundle(args)
    try:
        for benchmark in args.benchmarks:
            rows = datasets[benchmark]
            failures = 0
            for index, raw in enumerate(rows, start=1):
                target = sample_path(output_root, 'jarv_gen', benchmark, str(raw['id']))
                if is_complete(target, fingerprint):
                    continue
                retriever = retrievers.for_example(benchmark, str(raw['id']))
                config = JarvConfig(
                    model_path=args.model,
                    method='jarv',
                    tau_g=args.tau_g,
                    tau_a=args.tau_a,
                    max_aspects=args.max_aspects,
                    queries_per_aspect=args.queries_per_aspect,
                    passages_per_query=args.top_k,
                    search_cap=args.search_cap,
                    max_claim_chars=args.max_claim_chars,
                    max_passage_chars=args.max_passage_chars,
                    max_evidence_chars=args.max_evidence_chars,
                    query_max_new_tokens=args.query_max_new_tokens,
                    json_retries=args.json_retries,
                    seed=args.seed,
                )
                checker = BatchedInstrumentedFactChecker(llm, retriever, config)
                result = enrich_pipeline_row(
                    checker.run(as_example(benchmark, raw)), raw, benchmark, 'jarv_gen'
                )
                result['decision_mode'] = 'deterministic_generation'
                result['config_fingerprint'] = fingerprint
                atomic_write_json(target, result)
                failures += int(bool(result.get('error')))
                if index % args.log_every == 0 or index == len(rows):
                    LOGGER.info(
                        'jarv_gen/%s processed=%d total=%d failures=%d remaining=%d',
                        benchmark,
                        index,
                        len(rows),
                        failures,
                        len(rows) - index,
                    )
    finally:
        retrievers.close()
        del llm, generator


def macro_f1(rows: list[dict]) -> float:
    values = []
    for label in ('real', 'fake'):
        tp = sum(row.get('gold') == label and row.get('prediction') == label for row in rows)
        fp = sum(row.get('gold') != label and row.get('prediction') == label for row in rows)
        fn = sum(row.get('gold') == label and row.get('prediction') != label for row in rows)
        denominator = 2 * tp + fp + fn
        values.append((2 * tp / denominator) if denominator else 0.0)
    return sum(values) / len(values)


def load_prediction_rows(output_root: Path, method: str, benchmark: str) -> list[dict]:
    directory = output_root / 'predictions' / method / benchmark
    rows = []
    if directory.exists():
        for path in sorted(directory.glob('*.json')):
            try:
                rows.append(json.loads(path.read_text(encoding='utf-8')))
            except Exception as exc:
                rows.append({'id': path.name, 'error': f'{type(exc).__name__}: {exc}'})
    return rows


def summarize_rows(method: str, benchmark: str, rows: list[dict], expected_n: int) -> dict:
    correct = sum(row.get('prediction') == row.get('gold') for row in rows)
    failures = [row for row in rows if row.get('error') or row.get('prediction') not in {'real', 'fake'}]
    def average(key: str) -> float:
        return sum(float(row.get(key) or 0) for row in rows) / expected_n if expected_n else 0.0
    stage_totals: dict[str, float] = defaultdict(float)
    for row in rows:
        for key, value in (row.get('stage_times_s') or {}).items():
            stage_totals[key] += float(value or 0)
    return {
        'method': method,
        'benchmark': benchmark,
        'n_expected': expected_n,
        'n_outputs': len(rows),
        'n_failed': len(failures) + max(0, expected_n - len(rows)),
        'complete': len(rows) == expected_n and not failures,
        'correct': correct,
        'accuracy': correct / expected_n if expected_n else 0.0,
        'macro_f1': macro_f1(rows),
        'avg_search_calls': average('search_calls'),
        'avg_retrieval_rounds': average('retrieval_rounds'),
        'avg_latency_s': average('latency_s'),
        'avg_query_generation_tokens': average('query_generation_tokens'),
        'avg_total_generated_tokens': average('total_generated_tokens'),
        'parse_failures': sum(int(row.get('parse_failures') or 0) for row in rows),
        'fallbacks': sum(int(row.get('fallbacks') or 0) for row in rows),
        'retries': sum(int(row.get('retries') or 0) for row in rows),
        'retrieval_failures': sum(int(row.get('retrieval_failures') or 0) for row in rows),
        'avg_gate_s': stage_totals.get('gate', 0.0) / expected_n if expected_n else 0.0,
        'avg_aspect_s': stage_totals.get('aspect_selection', 0.0) / expected_n if expected_n else 0.0,
        'avg_query_s': stage_totals.get('query_generation', 0.0) / expected_n if expected_n else 0.0,
        'avg_retrieval_s': stage_totals.get('retrieval', 0.0) / expected_n if expected_n else 0.0,
        'avg_final_judge_s': stage_totals.get('final_judge', 0.0) / expected_n if expected_n else 0.0,
    }


def build_outputs(output_root: Path, datasets: dict[str, list[dict]], methods: list[str], config: dict) -> None:
    summaries = []
    failure_rows = []
    for method in methods:
        for benchmark, expected in datasets.items():
            rows = load_prediction_rows(output_root, method, benchmark)
            summary = summarize_rows(method, benchmark, rows, len(expected))
            summaries.append(summary)
            for row in rows:
                if row.get('error') or row.get('prediction') not in {'real', 'fake'}:
                    failure_rows.append(
                        {
                            'method': method,
                            'benchmark': benchmark,
                            'id': row.get('id'),
                            'error': row.get('error') or 'invalid prediction',
                        }
                    )
    columns = list(summaries[0]) if summaries else []
    temporary = output_root / '.summary.csv.tmp'
    with temporary.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(summaries)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, output_root / 'summary.csv')
    atomic_write_text(
        output_root / 'failures.jsonl',
        ''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in failure_rows),
    )
    indexed = {(row['method'], row['benchmark']): row for row in summaries}
    limits = config.get('benchmark_limits') or {
        benchmark: config.get('limit', 0) for benchmark in datasets
    }
    dataset_scope = ', '.join(
        f'{benchmark}={len(rows)} '
        + ('(full binary split)' if not limits.get(benchmark) else '(seeded sample)')
        for benchmark, rows in datasets.items()
    )
    lines = [
        '# WWW27 / JARV offline main experiment',
        '',
        f'- Model: `{config['model']}` (frozen)',
        f'- Seed: `{config['seed']}`',
        '- Chat template: tokenizer.apply_chat_template (enabled)',
        '- Thinking: disabled',
        '- Retrieval: offline BM25/FTS5, top-k=3, 800 characters per passage, 3,600-character cumulative evidence budget',
        '- Gold labels are used only during aggregation.',
        f'- Dataset scope: {dataset_scope}.',
        '',
        '## Results',
        '',
        '| Method | Benchmark | N | Complete | Accuracy | Macro-F1 | Calls | Latency (s) | Failures |',
        '|---|---|---:|---|---:|---:|---:|---:|---:|',
    ]
    for row in summaries:
        lines.append(
            f'| {row['method']} | {row['benchmark']} | {row['n_expected']} | '
            f'{row['complete']} | {row['accuracy']:.4f} | {row['macro_f1']:.4f} | '
            f'{row['avg_search_calls']:.3f} | {row['avg_latency_s']:.3f} | {row['n_failed']} |'
        )
    lines.extend(
        [
            '',
            '## JARV relative to Retrieve-once + JEV',
            '',
            '| Benchmark | Accuracy delta | Search-call delta | Latency delta (s) |',
            '|---|---:|---:|---:|',
        ]
    )
    for benchmark in datasets:
        jarv = indexed.get(('jarv', benchmark))
        base = indexed.get(('retrieve_once_jev', benchmark))
        if jarv and base:
            lines.append(
                f'| {benchmark} | {jarv['accuracy'] - base['accuracy']:+.4f} | '
                f'{jarv['avg_search_calls'] - base['avg_search_calls']:+.3f} | '
                f'{jarv['avg_latency_s'] - base['avg_latency_s']:+.3f} |'
            )
    lines.extend(
        [
            '',
            '## Completion and provenance',
            '',
            f'- Complete method/benchmark cells: {sum(row['complete'] for row in summaries)}/{len(summaries)}.',
            f'- Failed or invalid samples: {len(failure_rows)} (see `failures.jsonl`).',
            '- `llm_jev` rows marked `reused=true` were imported only after exact ID, claim, and gold-label validation against the current evaluation subset.',
            '- Per-sample files include gate scores, stop reason, aspect scores/selections, generated subclaims/queries, evidence IDs/sources/snippets, actual calls/rounds, token counts, stage timings, retries, fallbacks, and errors.',
        ]
    )
    atomic_write_text(output_root / 'report.md', '\n'.join(lines) + '\n')


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--output-root', type=Path, required=True)
    result.add_argument('--model', default=DEFAULT_MODEL)
    result.add_argument('--benchmarks', nargs='+', choices=BENCHMARKS, default=list(BENCHMARKS))
    result.add_argument('--methods', nargs='+', choices=METHODS, default=list(METHODS))
    result.add_argument('--limit', type=int, default=0)
    result.add_argument(
        '--benchmark-limit',
        action='append',
        default=[],
        metavar='BENCHMARK=N',
        help=(
            'Override --limit for one benchmark; may be repeated, for example '
            '--benchmark-limit fever=1000. Zero means the full binary split.'
        ),
    )
    result.add_argument('--seed', type=int, default=42)
    result.add_argument('--tau-g', type=float, default=0.5)
    result.add_argument('--tau-a', type=float, default=0.5)
    result.add_argument('--search-cap', type=int, default=3)
    result.add_argument('--max-aspects', type=int, default=3)
    result.add_argument('--queries-per-aspect', type=int, default=1)
    result.add_argument('--top-k', type=int, default=3)
    result.add_argument('--max-claim-chars', type=int, default=1500)
    result.add_argument('--max-passage-chars', type=int, default=800)
    result.add_argument('--max-evidence-chars', type=int, default=3600)
    result.add_argument('--query-max-new-tokens', type=int, default=192)
    result.add_argument('--decision-max-new-tokens', type=int, default=24)
    result.add_argument('--json-retries', type=int, default=2)
    result.add_argument('--gpu-memory-utilization', type=float, default=0.80)
    result.add_argument('--max-model-len', type=int, default=8192)
    result.add_argument('--log-every', type=int, default=25)
    result.add_argument('--log-level', default='INFO')
    result.add_argument(
        '--reuse-predictions-root',
        action='append',
        default=[],
        type=Path,
        help='Import compatible successful per-sample checkpoints before running.',
    )
    return result


def parse_benchmark_limits(values: list[str]) -> dict[str, int]:
    limits: dict[str, int] = {}
    for raw in values:
        name, separator, value = raw.partition('=')
        if not separator or name not in BENCHMARKS:
            raise SystemExit(
                f'Invalid --benchmark-limit {raw!r}; expected one of '
                f'{', '.join(BENCHMARKS)} followed by =N'
            )
        if name in limits:
            raise SystemExit(f'Duplicate --benchmark-limit for {name}')
        try:
            parsed = int(value)
        except ValueError as exc:
            raise SystemExit(f'Invalid limit in --benchmark-limit {raw!r}') from exc
        if parsed < 0:
            raise SystemExit(f'Benchmark limit must be non-negative: {raw!r}')
        limits[name] = parsed
    return limits


def main() -> int:
    args = parser().parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    log_path = args.output_root / 'logs' / 'run.log'
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format='%(asctime)s %(levelname)s %(message)s',
        handlers=[logging.FileHandler(log_path, encoding='utf-8'), logging.StreamHandler()],
    )
    benchmark_limit_overrides = parse_benchmark_limits(args.benchmark_limit)
    effective_limits = {
        benchmark: benchmark_limit_overrides.get(benchmark, args.limit)
        for benchmark in args.benchmarks
    }
    full_datasets = {
        benchmark: load_benchmark(benchmark, 0, args.seed)
        for benchmark in args.benchmarks
    }
    datasets = {
        benchmark: proportional_stratified_sample(
            full_datasets[benchmark], effective_limits[benchmark], args.seed, benchmark
        )
        for benchmark in args.benchmarks
    }
    config = {
        'objective': 'Original-semantics WWW27/JARV fast250 comparison',
        'source_reference': str(ORIGINAL_ROOT),
        'model': args.model,
        'model_frozen': True,
        'seed': args.seed,
        'chat_template': True,
        'enable_thinking': False,
        'deterministic_generation': True,
        'temperature': 0.0,
        'benchmarks': args.benchmarks,
        'methods': args.methods,
        'limit': args.limit,
        'benchmark_limits': effective_limits,
        'sampling': 'deterministic proportional stratified sample by gold label, then sort by id',
        'tau_g': args.tau_g,
        'tau_a': args.tau_a,
        'search_cap': args.search_cap,
        'max_aspects_per_round': args.max_aspects,
        'queries_per_aspect': args.queries_per_aspect,
        'top_k': args.top_k,
        'max_claim_chars': args.max_claim_chars,
        'max_passage_chars': args.max_passage_chars,
        'max_evidence_chars': args.max_evidence_chars,
        'query_max_new_tokens': args.query_max_new_tokens,
        'fever_query_terms': 'all non-stopword terms, untruncated original OR query',
        'semantic_acceleration': 'concurrent read-only same-round SQLite lookup only',
        'rejected_optimizations': [
            'cross-aspect JEV batching (changed real-model probabilities)',
            'query-generation batching (changed real-model generated queries)',
            'query-term truncation including rarest4 (changed retrieval semantics)',
        ],
        'decision_max_new_tokens': args.decision_max_new_tokens,
        'json_retries': args.json_retries,
        'retrieval_indexes': {
            'fever': str(INDEX_ROOT / 'fever_fts.sqlite'),
            'averitec': str(INDEX_ROOT / 'averitec_dev_fts.sqlite'),
            'scifact': str(BENCHMARK_ROOT / 'scifact' / 'data' / 'corpus.jsonl'),
        },
        'dataset_counts': {key: len(value) for key, value in datasets.items()},
        'dataset_sha256': {key: dataset_digest(value) for key, value in datasets.items()},
        'code_sha256': {
            str(path.relative_to(ROOT)): sha256_file(path)
            for path in [
                ROOT / 'src' / 'llm.py',
                ROOT / 'src' / 'pipeline.py',
                ROOT / 'src' / 'prompts.py',
                ROOT / 'src' / 'retrieval_sqlite.py',
                Path(__file__),
            ]
        },
        'source_reference_sha256': {
            str(path.relative_to(ORIGINAL_ROOT)): sha256_file(path)
            for path in [
                ORIGINAL_ROOT / 'src' / 'llm.py',
                ORIGINAL_ROOT / 'src' / 'pipeline.py',
                ORIGINAL_ROOT / 'src' / 'prompts.py',
                ORIGINAL_ROOT / 'src' / 'retrieval_sqlite.py',
            ]
        },
        'environment': {
            'python': sys.version,
            'platform': platform.platform(),
        },
        'output_root': str(args.output_root),
        'started_at': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
    }
    fingerprint = config_fingerprint(config)
    config['config_fingerprint'] = fingerprint
    atomic_write_json(
        args.output_root / 'sample_manifest.json',
        {
            benchmark: {
                'ids': [str(row['id']) for row in rows],
                'label_counts': dict(Counter(str(row['gold']) for row in rows)),
                'sha256': dataset_digest(rows),
            }
            for benchmark, rows in datasets.items()
        },
    )
    config_path = args.output_root / 'config.json'
    if config_path.exists():
        existing = json.loads(config_path.read_text(encoding='utf-8'))
        if existing.get('config_fingerprint') != fingerprint:
            raise SystemExit('Output root contains a different experiment configuration')
        config = existing
    else:
        atomic_write_json(config_path, config)

    reuse_manifest = []
    for source_root in args.reuse_predictions_root:
        reuse_manifest.append(
            import_compatible_predictions(
                source_root,
                args.output_root,
                datasets,
                args.methods,
                config,
                fingerprint,
            )
        )
    if reuse_manifest:
        atomic_write_json(
            args.output_root / 'checkpoint_reuse_manifest.json', reuse_manifest
        )
    audit_source = ROOT / 'audit.md'
    if audit_source.exists() and not (args.output_root / 'audit.md').exists():
        atomic_write_text(args.output_root / 'audit.md', audit_source.read_text(encoding='utf-8'))
    protocol_source = ROOT / 'PROTOCOL.md'
    if protocol_source.exists() and not (args.output_root / 'PROTOCOL.md').exists():
        atomic_write_text(
            args.output_root / 'PROTOCOL.md',
            protocol_source.read_text(encoding='utf-8'),
        )

    reused = {}
    if 'llm_jev' in args.methods:
        for benchmark, rows in datasets.items():
            reused[benchmark] = import_reused_jev(
                args.output_root, benchmark, rows, fingerprint
            )
        config['reused_llm_jev'] = reused
        atomic_write_json(config_path, config)

    run_jev_methods(args.output_root, datasets, args, fingerprint)
    run_gen_method(args.output_root, datasets, args, fingerprint)
    build_outputs(args.output_root, datasets, args.methods, config)
    LOGGER.info('Experiment aggregation complete: %s', args.output_root)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
