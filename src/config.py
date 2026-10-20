from __future__ import annotations

from dataclasses import asdict, dataclass, fields

import yaml


@dataclass
class Document:
    doc_id: str
    title: str = ""
    text: str = ""
    url: str = ""
    date: str = ""


@dataclass
class Passage:
    doc_id: str
    text: str
    score: float = 0.0
    title: str = ""
    url: str = ""
    date: str = ""
    query: str = ""
    sub_claim: str = ""
    aspect: str = ""


@dataclass
class WWW27Config:
    model_path: str = "/root/autodl-tmp/models/Qwen3-8B"
    data_path: str = "data/sample_claims.jsonl"
    corpus_path: str = "data/sample_corpus.jsonl"
    output_path: str = "outputs/predictions.jsonl"
    method: str = "jarv"
    claim_mode: str = "stored"
    domain: str = ""
    limit: int = 0
    offset: int = 0
    seed: int = 0
    tau_g: float = 0.5
    tau_a: float = 0.5
    max_aspects: int = 3
    queries_per_aspect: int = 1
    passages_per_query: int = 2
    bm25_candidates: int = 500
    rerank_model_path: str = "/root/autodl-tmp/models/Qwen3-Reranker-0.6B"
    search_cap: int = 3
    max_claim_chars: int = 1500
    max_passage_chars: int = 800
    max_evidence_chars: int = 3600
    query_max_new_tokens: int = 192
    json_retries: int = 2
    reverse_options: bool = False
    resume: bool = False
    bm25_k1: float = 1.5
    bm25_b: float = 0.75

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _coerce_value(key: str, value: object, default: object) -> object:
    """Validate one YAML value against the type of its dataclass default."""
    if isinstance(default, bool):
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"true", "yes", "1"}:
                return True
            if lowered in {"false", "no", "0"}:
                return False
        raise SystemExit(f"Config key {key!r} must be a boolean, got {value!r}")
    if isinstance(default, int):
        if isinstance(value, bool):
            raise SystemExit(f"Config key {key!r} must be an integer, got {value!r}")
        if isinstance(value, int):
            return value
        if isinstance(value, float) and value.is_integer():
            return int(value)
        if isinstance(value, str):
            try:
                return int(value.strip())
            except ValueError:
                pass
        raise SystemExit(f"Config key {key!r} must be an integer, got {value!r}")
    if isinstance(default, float):
        if isinstance(value, bool):
            raise SystemExit(f"Config key {key!r} must be a number, got {value!r}")
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value.strip())
            except ValueError:
                pass
        raise SystemExit(f"Config key {key!r} must be a number, got {value!r}")
    if isinstance(default, str):
        if not isinstance(value, str):
            raise SystemExit(f"Config key {key!r} must be a string, got {value!r}")
        return value
    return value


def load_config(path: str | None) -> WWW27Config:
    config = WWW27Config()
    if not path:
        return config
    with open(path, "r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
        assert isinstance(raw, dict), type(raw)
        
    defaults = {item.name: item.default for item in fields(WWW27Config)}
    unknown = sorted(set(raw) - set(defaults))
    if unknown:
        raise SystemExit(f"Unknown config keys: {', '.join(unknown)}")
    for key, value in raw.items():
        setattr(config, key, _coerce_value(key, value, defaults[key]))
    return config


METHODS = ("www27",)
