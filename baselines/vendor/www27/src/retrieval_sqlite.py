"""Disk-backed BM25 retrieval over SQLite FTS5 indexes."""

from __future__ import annotations

import sqlite3
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import re

from src.types import Passage
from src.utils.text import STOPWORDS, best_passage, tokenize


class SQLiteBM25Retriever:
    """Use the same search contract as the in-memory BM25 retriever."""

    def __init__(
        self,
        db_path: str | Path,
        claim_scoped: bool = False,
    ):
        self.db_path = str(db_path)
        self.conn = sqlite3.connect(
            f"file:{self.db_path}?mode=ro", uri=True, check_same_thread=False
        )
        self.claim_scoped = claim_scoped
        self.current_claim: str | None = None

    def set_claim(self, claim_id: str) -> None:
        if self.claim_scoped:
            value = str(claim_id)
            match = re.fullmatch(r"row-(\d+)", value)
            self.current_claim = match.group(1) if match else value

    @staticmethod
    def _fts_query(query: str) -> str:
        terms = [term for term in tokenize(query) if term not in STOPWORDS]
        if not terms:
            terms = tokenize(query)
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
        return self._search_with_connection(
            self.conn, query, k, exclude_ids, max_passage_chars, self.current_claim
        )

    def _search_with_connection(
        self,
        connection: sqlite3.Connection,
        query: str,
        k: int,
        exclude_ids: set[str] | None,
        max_passage_chars: int,
        claim_id: str | None,
    ) -> list[Passage]:
        if k <= 0:
            return []
        # Preserve the original retriever exactly: every non-stopword query
        # term participates in the OR expression.  Concurrency below changes
        # scheduling only, never query semantics or ranking.
        fts_query = self._fts_query(query)
        if not fts_query:
            return []
        params: list[object] = [fts_query]
        where = "docs MATCH ?"
        if self.claim_scoped:
            if claim_id is None:
                raise RuntimeError("claim-scoped retriever used before set_claim()")
            where += " AND claim_id = ?"
            params.append(claim_id)
        params.append(max(k * 3, k))
        rows = connection.execute(
            "SELECT doc_id, title, text, bm25(docs) AS score "
            f"FROM docs WHERE {where} ORDER BY bm25(docs) LIMIT ?",
            params,
        ).fetchall()
        blocked = exclude_ids or set()
        passages: list[Passage] = []
        for doc_id, title, body, score in rows:
            doc_id = str(doc_id)
            if doc_id in blocked:
                continue
            snippet = best_passage(str(body), query, max_passage_chars)
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

    def search_many(
        self,
        queries: list[str],
        k: int,
        exclude_ids: set[str] | None = None,
        max_passage_chars: int = 800,
    ) -> list[list[Passage]]:
        """Search independent queries concurrently using read-only connections."""
        if len(queries) <= 1:
            return [
                self.search(query, k, exclude_ids, max_passage_chars)
                for query in queries
            ]
        claim_id = self.current_claim

        def worker(query: str) -> list[Passage]:
            connection = sqlite3.connect(
                f"file:{self.db_path}?mode=ro", uri=True, check_same_thread=False
            )
            try:
                return self._search_with_connection(
                    connection,
                    query,
                    k,
                    exclude_ids,
                    max_passage_chars,
                    claim_id,
                )
            finally:
                connection.close()

        with ThreadPoolExecutor(max_workers=min(len(queries), 3)) as executor:
            return list(executor.map(worker, queries))

    def close(self) -> None:
        self.conn.close()

    def __del__(self) -> None:
        try:
            self.conn.close()
        except Exception:
            pass
