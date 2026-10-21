"""Local Okapi BM25 retrieval. No remote search calls."""

from __future__ import annotations

import math
from collections import Counter, defaultdict

import sys
from pathlib import Path

ROOT=Path(__file__).parents[1].resolve()
sys.path.insert(0,str(ROOT))

from src.utils import STOPWORDS, best_passage, tokenize
from src.config import Document, Passage

class BM25Index:
    def __init__(
        self,
        documents: list[Document],
        k1: float=1.5,
        b: float=0.75
    ):
        self.documents = documents
        self.k1 = k1
        self.b = b
        self.tf: list[Counter] = []
        self.lengths: list[int] = []
        self.df: Counter = Counter()
        self.postings: dict[str, list[int]] = defaultdict(list)
        for index, document in enumerate(documents):
            tokens = tokenize(f"{document.title}\n{document.text}")
            counts = Counter(tokens)
            self.tf.append(counts)
            self.lengths.append(len(tokens) or 1)
            for term in counts:
                self.df[term] += 1
                self.postings[term].append(index)
        self.n_docs = len(documents)
        self.avgdl = (sum(self.lengths) / self.n_docs) if self.n_docs else 1.0
        self.idf = {
            term: math.log(1.0 + (self.n_docs - freq + 0.5) / (freq + 0.5))
            for term, freq in self.df.items()
        }

    def search(
        self,
        query: str,
        k: int,
        exclude_ids: set[str] | None = None
    ) -> list[Passage]:
        if k <= 0 or not self.documents:
            return []
        blocked = exclude_ids or set()
        terms = [token for token in tokenize(query) if token not in STOPWORDS]
        if not terms:
            terms = tokenize(query)
        if not terms:
            return []
        scores: dict[int, float] = defaultdict(float)
        for term, query_tf in Counter(terms).items():
            idf = self.idf.get(term)
            if idf is None:
                continue
            for doc_index in self.postings.get(term, ()):
                tf = self.tf[doc_index][term]
                length = self.lengths[doc_index]
                denom = tf + self.k1 * (1.0 - self.b + self.b * length / self.avgdl)
                scores[doc_index] += idf * (tf * (self.k1 + 1.0) / denom) * query_tf
        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        passages: list[Passage] = []
        for doc_index, score in ranked:
            document = self.documents[doc_index]
            if document.doc_id in blocked:
                continue
            passages.append(
                Passage(
                    doc_id=document.doc_id,
                    text=document.text,
                    score=float(score),
                    title=document.title,
                    url=document.url,
                    date=document.date,
                )
            )
            if len(passages) >= k:
                break
        return passages


class BM25Retriever:
    def __init__(
        self,
        documents: list[Document],
        k1: float=1.5,
        b: float=0.75,
    ):
        self.documents = documents
        self.index = BM25Index(documents, k1=k1, b=b)

    def search(
        self,
        query: str,
        k: int,
        exclude_ids: set[str] | None = None,
        max_passage_chars: int = 800,
    ) -> list[Passage]:
        hits = self.index.search(query, k=k, exclude_ids=exclude_ids)
        trimmed = []
        for hit in hits:
            hit.text = best_passage(hit.text, query, max_passage_chars)
            if hit.text:
                trimmed.append(hit)
        return trimmed
