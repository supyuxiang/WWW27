import json
import math
import re
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
import sys
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams
from dataclasses import dataclass

ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0,str(ROOT))


@dataclass
class Benchmark:
    name: str
    path:str | Path
    choice_dict: dict
    doc: str | None = None


BENCHMARKS = [
    Benchmark(
        name='averitec',
        path='/root/yxfeng/www27/data/averitec/dev.json',
        choice_dict={
            'Supported': (
                'The evidence is sufficient to conclude that the central factual content of the claim is true.'
            ),
            'Refuted': (
                'The evidence is sufficient to conclude that the central factual content of the claim is false.'
            ),
            'Not Enough Evidence': (
                'The evidence is insufficient to support or refute the claim, '
                'including missing, off-topic, or unreliable sources.'
            ),
            'Conflicting Evidence/Cherrypicking': (
                'The claim is misleading: reliable sources conflict, or the claim '
                'cherry-picks true facts while omitting context needed for a fair reading.'
            ),
        },
        doc=(
            'AVeriTeC is a real-world fact-checking benchmark. Claims come from fact-checking '
            'articles and are verified against open-web evidence. Each claim receives one of four '
            'verdicts: Supported, Refuted, Not Enough Evidence, or Conflicting Evidence/Cherrypicking. '
            'This file is the development split.'
        ),
    ),

    Benchmark(
        name='fever',
        path='/root/yxfeng/www27/data/fever/shared_task_dev.json',
        choice_dict={
            'SUPPORTS': (
                'The evidence supports the claim.'
            ),
            'REFUTES': (
                'The evidence refutes the claim.'
            ),
            'NOT ENOUGH INFO': (
                'The evidence does not contain enough information to support or refute the claim.'
            ),
        },
        doc=(
            'FEVER (Fact Extraction and VERification) checks claims against Wikipedia. '
            'A claim is SUPPORTS when the evidence supports it, REFUTES when the evidence '
            'refutes it, and NOT ENOUGH INFO when Wikipedia does not contain enough information '
            'to decide. This file is the shared-task development split.'
        ),
    ),

    Benchmark(
        name='scifact',
        path='/root/yxfeng/www27/data/scifact/claims_dev.json',
        choice_dict={
            'SUPPORT': (
                'The scientific evidence supports the claim.'
            ),
            'CONTRADICT': (
                'The scientific evidence contradicts the claim.'
            ),
            'NOT ENOUGH INFO': (
                'The evidence does not contain enough information to support or contradict the claim.'
            ),
        },
        doc=(
            'SciFact is a scientific claim-verification benchmark. Claims are checked against '
            'research-paper abstracts and labeled SUPPORT, CONTRADICT, or NOT ENOUGH INFO. '
            'SUPPORT and CONTRADICT come from the evidence rationales; claims with no rationale '
            'are NOT ENOUGH INFO. This file is the development claims set.'
        ),
    ),
]


def load_tokenizer(model_path: str | Path, is_training: bool=False) -> AutoTokenizer:
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    if is_training:
        tok.padding_side='right'
    else:
        tok.padding_side='left'
    return tok



def load_benchmark(
    benchmark_name: str,
    limit: int=-1,
) -> tuple[list[dict], Benchmark]:

    def _record_id(item: dict):
        for key in ('id', 'claim_id', 'claimId'):
            if item.get(key) is not None:
                return item[key]
        return None

    def _record_label(item: dict):
        if item.get('label') is not None:
            return item['label']
        # SciFact stores SUPPORT / CONTRADICT on evidence rationales.
        # An empty evidence dict is the not-enough-info class.
        labels = []
        evidence = item.get('evidence')
        if isinstance(evidence, dict):
            for rationales in evidence.values():
                if not isinstance(rationales, list):
                    continue
                for rationale in rationales:
                    if isinstance(rationale, dict) and rationale.get('label') is not None:
                        labels.append(rationale['label'])
        unique = list(dict.fromkeys(labels))
        if len(unique) == 1:
            return unique[0]
        if not unique:
            return 'NOT ENOUGH INFO'
        return unique[0]

    for bench in BENCHMARKS:
        if benchmark_name != bench.name:
            continue
        benchmark_data: list[dict] = []
        with open(bench.path, 'r', encoding='utf-8') as f:
            obj = json.load(f)
        assert isinstance(obj, list), type(obj)
        
        for item in obj:
            row = {
                'claim': item['claim'],
                'label': _record_label(item),
            }
            record_id = _record_id(item)
            if record_id is not None:
                row['id'] = record_id
            benchmark_data.append(row)

        if limit > 0 and len(benchmark_data) > limit:
            benchmark_data = benchmark_data[:limit]
        return benchmark_data, bench
    raise ValueError(f'{benchmark_name} not found!')



# metrics

def compute_accuracy(gold_ls: list[str], pred_ls: list[str]) -> float:
    assert len(gold_ls) == len(pred_ls)
    correct = sum(truth == guess for truth, guess in zip(gold_ls, pred_ls))
    return correct / max(len(gold_ls),1)

def compute_f1(
    gold_ls: list[str],
    pred_ls: list[str],
    benchmark: Benchmark,
) -> float:
    assert len(gold_ls) == len(pred_ls)
    labels = list(benchmark.choice_dict)
    if not labels:
        return 0.0
    f1_sum = 0.0
    for label in labels:
        tp = fp = fn = 0
        for truth, guess in zip(gold_ls, pred_ls):
            if guess == label and truth == label:
                tp += 1
            elif guess == label and truth != label:
                fp += 1
            elif truth == label and guess != label:
                fn += 1
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1_sum += (2 * precision * recall / (precision + recall)) if precision + recall else 0.0
    return f1_sum / len(labels)




TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?", re.IGNORECASE)
STOPWORDS = {
    "a",
    "an",
    "the",
    "of",
    "to",
    "and",
    "in",
    "on",
    "for",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "it",
    "this",
    "that",
    "with",
    "as",
    "at",
    "by",
    "from",
    "or",
}
_BAD_MARKERS = (
    "<|im_start|>",
    "<|im_end|>",
    "<|endoftext|>",
    "<|object_ref_start|>",
    "<|object_ref_end|>",
)


def sanitize(text: str | None) -> str:
    value = text or ""
    for marker in _BAD_MARKERS:
        value = value.replace(marker, " ")
    return value.replace("\x00", " ")


def tokenize(text: str | None) -> list[str]:
    return TOKEN_RE.findall((text or "").lower())


def clip(text: str | None, max_chars: int) -> str:
    value = sanitize(text).strip()
    if max_chars <= 0 or len(value) <= max_chars:
        return value
    suffix = " ..."
    if max_chars <= len(suffix):
        return value[:max_chars]
    return value[: max_chars - len(suffix)].rstrip() + suffix


def _inline(value: str | None, default: str = "") -> str:
    """Collapse arbitrary text to one safe header line."""
    text = " ".join(sanitize(value).split())
    return text or default


def best_passage(text: str, query: str, max_chars: int) -> str:
    body = sanitize(text).strip()
    if max_chars <= 0 or len(body) <= max_chars:
        return body
    terms = [token for token in tokenize(query) if token not in STOPWORDS] or tokenize(query)
    lowered = body.lower()
    positions = []
    for term in terms:
        found = lowered.find(term)
        if term and found >= 0:
            positions.append(found)
    if not positions:
        return body[:max_chars].rstrip()
    center = positions[len(positions) // 2]
    start = max(0, center - max_chars // 3)
    if start > 0:
        space = body.rfind(" ", 0, start)
        if space >= 0 and start - space < 40:
            start = space + 1
    return body[start : start + max_chars].rstrip()


def near_duplicate(left: str, right: str, threshold: float = 0.9) -> bool:
    left_terms = set(tokenize(left)[:300])
    right_terms = set(tokenize(right)[:300])
    if not left_terms or not right_terms:
        return False
    overlap = len(left_terms & right_terms)
    return overlap / min(len(left_terms), len(right_terms)) >= threshold


def is_leakage(claim: str, source_ids: set[str], doc_id: str, doc_text: str) -> bool:
    if doc_id in source_ids:
        return True
    claim_text = (claim or "").strip().lower()
    if len(claim_text) >= 40 and claim_text in (doc_text or "").lower():
        return True
    return near_duplicate(claim, doc_text)


def format_evidence(passages, max_chars: int = 3600) -> str:
    if not passages:
        return "none"
    blocks = []
    used = 0
    for index, passage in enumerate(passages, start=1):
        source = _inline(passage.doc_id, "unknown")
        date = _inline(passage.date, "unavailable")
        url = _inline(passage.url, "unavailable")
        aspect = _inline(passage.aspect, "n/a")
        query = _inline(passage.query)
        header = (
            f"[{index}] source={source}; date={date}; url={url}; "
            f"aspect={aspect}; query={query}"
        )
        body = sanitize(passage.text).strip()
        remaining = max_chars - used
        if remaining <= 0:
            break
        piece = header + "\n" + body
        if len(piece) > remaining:
            piece = piece[:remaining].rstrip()
        blocks.append(piece)
        used += len(piece) + 2
    return "\n\n".join(blocks) if blocks else "none"


def main():
    pass

if __name__ == '__main__':
    main()
    










