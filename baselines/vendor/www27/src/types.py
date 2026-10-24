'''Shared records for claims, retrieval, and model calls.'''

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Usage:
    forward_passes: int = 0
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass(frozen=True)
class Option:
    key: str
    suffix: str
    text: str


@dataclass
class Choice:
    key: str
    probs: dict[str, float]
    scores: dict[str, float]
    parsed: bool
    raw_text: str
    usage: Usage = field(default_factory=Usage)


@dataclass
class Generation:
    text: str
    usage: Usage = field(default_factory=Usage)


@dataclass
class Cost:
    forward_passes: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    search_calls: int = 0
    parse_failures: int = 0
    fallbacks: int = 0
    retries: int = 0
    search_truncated: bool = False


@dataclass(frozen=True)
class Document:
    doc_id: str
    text: str
    title: str = ''
    url: str = ''
    date: str = ''
    domain: str = ''


@dataclass
class Passage:
    doc_id: str
    text: str
    score: float
    title: str = ''
    url: str = ''
    date: str = ''
    query: str = ''
    sub_claim: str = ''
    aspect: str = ''


@dataclass
class AtomicQuery:
    aspect: str
    sub_claim: str
    query: str
    fallback: bool = False


@dataclass
class Example:
    id: str
    claim: str
    label: str
    domain: str
    title: str = ''
    text: str = ''
    source_id: str = ''
    needs_retrieval: bool | None = None
    gold_aspects: list[str] | None = None


def add_usage(cost: Cost, usage: Usage) -> None:
    cost.forward_passes += usage.forward_passes
    cost.input_tokens += usage.input_tokens
    cost.output_tokens += usage.output_tokens
