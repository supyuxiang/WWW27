"""Disk-backed BM25 retrieval over SQLite FTS5 indexes.

This adapter lets JARV retrieve from a full offline corpus without loading all
documents into Python memory.  It implements the same ``search`` contract as
``src.retrieval.BM25Retriever`` and can optionally restrict search to the
candidate documents of the current claim (used by AVeriTeC's knowledge store).
"""

from __future__ import annotations

import sqlite3

import sys
from pathlib import Path

ROOT=Path(__file__).parents[1].resolve()
sys.path.insert(0,str(ROOT))

from src.utils import STOPWORDS, best_passage, tokenize
from src.config import Passage


class SQLiteBM25Retriever:
    def __init__(self, db_path: str | Path, claim_scoped: bool = False):
        self.db_path = str(db_path)
        uri = f"file:{self.db_path}?mode=ro"
        self.conn = sqlite3.connect(uri, uri=True, check_same_thread=False)
        self.claim_scoped = claim_scoped
        self.current_claim: str | None = None

    def set_claim(self, claim_id: str) -> None:
        if self.claim_scoped:
            self.current_claim = str(claim_id)

    def _fts_query(self, query: str) -> str:
        terms = [term for term in tokenize(query) if term not in STOPWORDS]
        if not terms:
            terms = tokenize(query)
        # Quote every term so punctuation and FTS operators cannot change the
        # query semantics.  OR is used for recall; bm25 still ranks matches.
        safe = []
        for term in terms:
            term = term.replace('"', "")
            if term:
                safe.append(f'"{term}"')
        return " OR ".join(safe)

    def search(
        self,
        query: str,
        k: int,
        exclude_ids: set[str] | None = None,
        max_passage_chars: int = 800,
    ) -> list[Passage]:
        if k <= 0:
            return []
        fts_query = self._fts_query(query)
        if not fts_query:
            return []
        params: list[object] = [fts_query]
        where = "docs MATCH ?"
        if self.claim_scoped and self.current_claim is not None:
            where += " AND claim_id = ?"
            params.append(self.current_claim)
        params.append(max(k * 3, k))
        sql = (
            "SELECT doc_id, title, text, bm25(docs) AS score "
            f"FROM docs WHERE {where} ORDER BY bm25(docs) LIMIT ?"
        )
        blocked = exclude_ids or set()
        rows = self.conn.execute(sql, params).fetchall()
        passages: list[Passage] = []
        for doc_id, title, text, score in rows:
            doc_id = str(doc_id)
            if doc_id in blocked:
                continue
            snippet = best_passage(str(text), query, max_passage_chars)
            if not snippet:
                continue
            passages.append(
                Passage(
                    doc_id=doc_id,
                    text=snippet,
                    score=float(-score),
                    title=str(title or ""),
                )
            )
            if len(passages) >= k:
                break
        return passages

    def close(self) -> None:
        self.conn.close()

    def __del__(self) -> None:  # best-effort cleanup
        try:
            self.conn.close()
        except Exception:
            pass
