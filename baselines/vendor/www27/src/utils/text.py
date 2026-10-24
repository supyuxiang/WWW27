'''Text normalization, clipping, formatting, and lexical helpers.'''

from __future__ import annotations

import math
import re

TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?", re.IGNORECASE)
STOPWORDS = {
    'a',
    'an',
    'the',
    'of',
    'to',
    'and',
    'in',
    'on',
    'for',
    'is',
    'are',
    'was',
    'were',
    'be',
    'been',
    'it',
    'this',
    'that',
    'with',
    'as',
    'at',
    'by',
    'from',
    'or',
}
_BAD_MARKERS = (
    '<|im_start|>',
    '<|im_end|>',
    '<|endoftext|>',
    '<|object_ref_start|>',
    '<|object_ref_end|>',
)


def sanitize(text: str | None) -> str:
    value = text or ''
    for marker in _BAD_MARKERS:
        value = value.replace(marker, ' ')
    return value.replace('\x00', ' ')


def tokenize(text: str | None) -> list[str]:
    return TOKEN_RE.findall((text or '').lower())


def clip(text: str | None, max_chars: int) -> str:
    value = sanitize(text).strip()
    if max_chars <= 0 or len(value) <= max_chars:
        return value
    suffix = ' ...'
    if max_chars <= len(suffix):
        return value[:max_chars]
    return value[: max_chars - len(suffix)].rstrip() + suffix


def _inline(value: str | None, default: str = '') -> str:
    '''Collapse arbitrary text to one safe header line.'''
    text = ' '.join(sanitize(value).split())
    return text or default


def softmax(scores: list[float]) -> list[float]:
    if not scores:
        return []
    peak = max(scores)
    weights = [math.exp(score - peak) for score in scores]
    total = sum(weights)
    if total <= 0:
        return [1.0 / len(scores)] * len(scores)
    return [weight / total for weight in weights]


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
        space = body.rfind(' ', 0, start)
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
    claim_text = (claim or '').strip().lower()
    if len(claim_text) >= 40 and claim_text in (doc_text or '').lower():
        return True
    return near_duplicate(claim, doc_text)


def format_subclaims(items: list[str] | None, max_chars: int = 1500) -> str:
    lines = []
    for item in items or []:
        text = _inline(item)
        if text:
            lines.append(f'- {text}')
    if not lines:
        return 'none'
    return clip('\n'.join(lines), max_chars)


def format_evidence(passages, max_chars: int = 3600) -> str:
    if not passages:
        return 'none'
    blocks = []
    used = 0
    for index, passage in enumerate(passages, start=1):
        source = _inline(passage.doc_id, 'unknown')
        date = _inline(passage.date, 'unavailable')
        url = _inline(passage.url, 'unavailable')
        aspect = _inline(passage.aspect, 'n/a')
        query = _inline(passage.query)
        header = (
            f'[{index}] source={source}; date={date}; url={url}; '
            f'aspect={aspect}; query={query}'
        )
        body = sanitize(passage.text).strip()
        remaining = max_chars - used
        if remaining <= 0:
            break
        piece = header + '\n' + body
        if len(piece) > remaining:
            piece = piece[:remaining].rstrip()
        blocks.append(piece)
        used += len(piece) + 2
    return '\n\n'.join(blocks) if blocks else 'none'
