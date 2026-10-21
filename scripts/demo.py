import json
import math
import yaml
import sys
from pathlib import Path


def load_scifact_dev(scifact_path:str | Path, limit:int=-1):
    claim_ls = []
    label_ls = []
    evidence_ls = []
    with open(scifact_path,'r',encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            claim_ls.append(data['claim'])
            evidence = data['evidence']
            evidence_ls.append(evidence)
            if evidence == {}:
                label_ls.append('NEI')  # 注意是 NEI
            else:
                first_entry = next(iter(evidence.values()))[0]   # 取第一个 value（一个 list）
                label_ls.append(first_entry['label'])         # 'SUPPORT' 或 'CONTRADICT'

    if limit > 0 and len(claim_ls) > limit:
        claim_ls = claim_ls[:limit]
        label_ls = label_ls[:limit]
        evidence_ls = evidence_ls[:limit]
    return claim_ls, label_ls, evidence_ls


def load_scifact_corpus(corpus_path: str | Path):
    id_ls = []
    title_ls = []
    abstract_ls = []
    with open(corpus_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            id_ls.append(data['doc_id'])
            title_ls.append(data['title'])
            abstract_ls.append(data['abstract'])
    return id_ls, title_ls, abstract_ls


def test_scifact():
    claim_ls, label_ls, evidence_ls = load_scifact_dev('/root/yxfeng/www27/data/scifact/claims_dev.jsonl', limit=-1)
    id_ls, title_ls, abstract_ls = load_scifact_corpus('/root/yxfeng/www27/data/scifact/corpus.jsonl')
    breakpoint()
    # encoding


def load_fever_dev(path: str | Path, limit:int=-1):
    pass

def load_fever_corpus(path: str | Path):
    pass

def test_fever():
    pass

def load_averitec_dev(path:str | Path, limit:int=-1):
    pass

def load_averitec_corpus(path: str | Path):
    pass

def test_averitec():
    pass


if __name__ == '__main__':
    test_scifact()
    

