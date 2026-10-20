import json
import math
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
                'label': item['label'],
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

def main():
    pass

if __name__ == '__main__':
    main()
    










