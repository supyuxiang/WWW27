##########    retriever    #############
from __future__ import annotations

import os
import sqlite3
import sys
from pathlib import Path
import logging
from abc import ABC, abstractmethod
from collections import defaultdict
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

from fastembed import SparseTextEmbedding

ROOT = Path(__file__).parents[2].resolve()
sys.path.insert(0,str(ROOT))

from src.utils import (
    STOPWORDS,
    best_passage,
    tokenize,
)
from src.config import (
    Document,
    Passage
)

RETRIEVER_MODEL_NAME = os.environ.get("FASTEMBED_BM25_MODEL", "Qdrant/bm25")
RETRIEVER_MODEL_CACHE = os.environ.get(
    "FASTEMBED_BM25_CACHE",
    "/root/autodl-tmp/models/fastembed",
)
RETRIEVER_ENCODE_BATCH_SIZE = int(os.environ.get("FASTEMBED_BM25_BATCH_SIZE", "256"))

logger = logging.getLogger(__file__)


#### Base
class RetrieverBase(ABC):

    @abstractmethod
    def search(self):
        pass
        


class BM25Retriever(RetrieverBase):
    """Sparse BM25 retriever backed by Qdrant/bm25 + fastembed."""

    def __init__(
        self,
        documents: list[Document],
        k1: float = 1.5,
        b: float = 0.75,
    ):
        # 实际加载的pretrained模型默认 k=1.2、b=0.75
        self.documents = documents
        self.k1 = k1
        self.b = b
        self.postings: dict[int, list[tuple[int, float]]] = defaultdict(list)
        self._model = SparseTextEmbedding(
            model_name=RETRIEVER_MODEL_NAME,
            cache_dir=RETRIEVER_MODEL_CACHE,
            local_files_only=True,
        )
        self._build_index()

    def _build_index(self) -> None:
        texts = [
            f"{document.title}\n{document.text}" if document.title else document.text
            for document in self.documents
        ]
        total = len(texts); assert total
        
        for start in range(0, total, RETRIEVER_ENCODE_BATCH_SIZE):
            batch = texts[start : start + RETRIEVER_ENCODE_BATCH_SIZE]
            vectors = list(self._model.embed(batch))
            for offset, vector in enumerate(vectors):
                doc_index = start + offset
                for term, weight in zip(vector.indices, vector.values):
                    self.postings[int(term)].append((doc_index, float(weight)))
            done = min(start + len(batch), total)
            print(f"[fastembed-bm25] indexed {done}/{total}", flush=True)

    def _query_vector(self, query: str) -> dict[int, float]:
        vector = list(self._model.query_embed(query))[0]
        result: dict[int, float] = {}
        for term, weight in zip(vector.indices, vector.values):
            key = int(term)
            result[key] = result.get(key, 0.0) + float(weight)
        return result

    def search(
        self,
        query: str,
        k: int,
        exclude_ids: set[str] | None = None,
        max_passage_chars: int = 800,
    ) -> list[Passage]:
        if k <= 0 or not self.documents:
            return []
        query_vector = self._query_vector(query)
        if not query_vector:
            return []

        scores: dict[int, float] = defaultdict(float)
        for term, query_weight in query_vector.items():
            for doc_index, document_weight in self.postings.get(term, ()):
                scores[doc_index] += query_weight * document_weight

        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        blocked = exclude_ids or set()
        passages: list[Passage] = []
        for doc_index, score in ranked:
            document = self.documents[doc_index]
            if str(document.doc_id) in blocked:
                continue
            text = best_passage(document.text, query, max_passage_chars)
            if not text:
                continue
            passages.append(
                Passage(
                    doc_id=document.doc_id,
                    text=text,
                    score=float(score),
                    title=document.title,
                    url=document.url,
                    date=document.date,
                )
            )
            if len(passages) >= k:
                break
        return passages


# Compatibility alias for code that imported the old index class directly.
BM25Index = BM25Retriever


class SQLiteBM25Retriever(RetrieverBase):
    """Disk-backed BM25 retrieval over a SQLite FTS5 index.

    Same ``search`` contract as ``BM25Retriever``. When ``claim_scoped`` is
    set, ``set_claim`` limits matches to that claim's documents.
    """

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
        # query semantics. OR is used for recall; bm25 still ranks matches.
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

    def __del__(self) -> None:
        try:
            self.conn.close()
        except Exception:
            pass


def main():
    pass

if __name__ == '__main__':
    main()
    