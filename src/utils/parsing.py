"""Parsing helpers for generated query JSON."""

from __future__ import annotations

import json
import re


def _strip_fence(text: str) -> str:
    cleaned = (text or "").strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip()


def _load_json_blob(text: str):
    cleaned = _strip_fence(text)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    for opener, closer in (("[", "]"), ("{", "}")):
        start = cleaned.find(opener)
        end = cleaned.rfind(closer)
        if start >= 0 and end > start:
            try:
                return json.loads(cleaned[start : end + 1])
            except json.JSONDecodeError:
                continue
    return None


def parse_query_list(text: str, limit: int) -> list[tuple[str, str]] | None:
    """Return ``(sub_claim, query)`` pairs from a generated JSON list.

    ``None`` means the reply is not valid JSON of the expected shape.
    An empty list is a valid reply meaning no proposition applies.
    """
    loaded = _load_json_blob(text)
    if loaded is None:
        return None
    if isinstance(loaded, dict):
        if "query" in loaded or "sub_claim" in loaded or "subclaim" in loaded:
            loaded = [loaded]
        else:
            for key in ("queries", "items", "results"):
                if isinstance(loaded.get(key), list):
                    loaded = loaded[key]
                    break
            else:
                return None
    if not isinstance(loaded, list):
        return None
    pairs: list[tuple[str, str]] = []
    for item in loaded:
        if isinstance(item, str):
            query = item.strip()
            if query:
                pairs.append((query, query))
            continue
        if not isinstance(item, dict):
            return None
        sub = item.get("sub_claim", item.get("subclaim", item.get("proposition", "")))
        query = item.get("query", item.get("search_query", ""))
        if not isinstance(sub, str):
            sub = "" if sub is None else str(sub)
        if not isinstance(query, str):
            query = "" if query is None else str(query)
        sub = sub.strip()
        query = query.strip()
        if not query:
            continue
        pairs.append((sub or query, query))
    if limit >= 0:
        pairs = pairs[:limit]
    return pairs
