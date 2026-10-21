"""Qdrant/bm25 sparse retrieval.

This module keeps the public ``BM25Retriever.search`` contract used by JARV,
but delegates BM25 tokenization/weighting to fastembed's Qdrant/bm25 model.

Differences from the previous pure-Python BM25 implementation:
  - documents are encoded once at startup into sparse BM25 vectors;
  - an in-memory term -> (doc_index, weight) postings index is built;
  - queries are encoded with the same model and scored by sparse dot product;
  - the model's BM25 weighting replaces the old k1/b formula, so scores are
    not numerically comparable with the previous implementation.

The model is expected at ``Qdrant/bm25`` in the fastembed cache. Set
FASTEMBED_BM25_CACHE to override the cache directory.
"""

from __future__ import annotations

import os
from collections import defaultdict

os.environ.setdefault("HF_HUB_OFFLINE", "1")

from fastembed import SparseTextEmbedding

from src.config import Document, Passage
from src.utils import best_passage

MODEL_NAME = os.environ.get("FASTEMBED_BM25_MODEL", "Qdrant/bm25")
MODEL_CACHE = os.environ.get(
    "FASTEMBED_BM25_CACHE",
    "/root/autodl-tmp/models/fastembed",
)
ENCODE_BATCH_SIZE = int(os.environ.get("FASTEMBED_BM25_BATCH_SIZE", "256"))


class BM25Retriever:
    """Sparse BM25 retriever backed by Qdrant/bm25 + fastembed."""

    def __init__(
        self,
        documents: list[Document],
        k1: float = 1.5,
        b: float = 0.75,
    ):
        # k1/b are kept in the signature for compatibility, but the
        # pretrained Qdrant/bm25 model supplies its own BM25 weighting.
        self.documents = documents
        self.k1 = k1
        self.b = b
        self.postings: dict[int, list[tuple[int, float]]] = defaultdict(list)
        self._model = SparseTextEmbedding(
            model_name=MODEL_NAME,
            cache_dir=MODEL_CACHE,
            local_files_only=True,
        )
        self._build_index()

    def _build_index(self) -> None:
        texts = [
            f"{document.title}\n{document.text}" if document.title else document.text
            for document in self.documents
        ]
        total = len(texts)
        if not total:
            return
        for start in range(0, total, ENCODE_BATCH_SIZE):
            batch = texts[start : start + ENCODE_BATCH_SIZE]
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
