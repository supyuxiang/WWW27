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
    load_corpus,
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
        total = len(texts)
        
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


INDEX_ROOT = Path("/root/autodl-tmp/www27/index")
SQLITE_INDEXES = {
    "fever": INDEX_ROOT / "fever_fts.sqlite",
    "averitec": INDEX_ROOT / "averitec_dev_fts.sqlite",
}

class ClaimRetriever:
    """BM25 over a SQLite FTS index, or an in-memory index for small corpora."""

    def __init__(self, max_passage_chars: int = 2000):
        self.max_passage_chars = max_passage_chars
        self._sqlite: dict[str, SQLiteBM25Retriever] = {}
        self._memory: dict[str, BM25Retriever] = {}

    def search(self, benchmark: str, claim: str, k: int) -> list[Passage]:
        retriever = self._get(benchmark)
        return retriever.search(
            claim,
            k=k,
            max_passage_chars=self.max_passage_chars,
        )

    def _get(self, benchmark: str):
        index = SQLITE_INDEXES.get(benchmark)
        if index is not None:
            if benchmark not in self._sqlite:
                if not index.is_file():
                    raise FileNotFoundError(index)
                self._sqlite[benchmark] = SQLiteBM25Retriever(
                    index,
                    claim_scoped=False,
                )
            return self._sqlite[benchmark]
        if benchmark not in self._memory:
            documents = load_corpus(benchmark)
            self._memory[benchmark] = BM25Retriever(documents)
        return self._memory[benchmark]

    def close(self) -> None:
        for retriever in self._sqlite.values():
            retriever.close()
        self._sqlite.clear()


######################   demo   #######################

def _show(title: str, passages: list[Passage]) -> None:
    print(f"\n== {title} ==")
    if not passages:
        print("(no hits)")
        return
    for rank, passage in enumerate(passages, start=1):
        snippet = " ".join(passage.text.split())
        if len(snippet) > 180:
            snippet = snippet[:180].rstrip() + "..."
        print(f"{rank}. score={passage.score:.4f}  doc_id={passage.doc_id}")
        if passage.title:
            print(f"   title: {passage.title}")
        print(f"   {snippet}")


def main() -> None:
    """Small walkthrough of the three search entry points.

    BM25Retriever: in-memory index over a list of Document.
    SQLiteBM25Retriever: the same search() on an FTS5 file.
    ClaimRetriever: picks the sqlite index or builds BM25Retriever from the corpus.
    """
    documents = [
        Document(
            doc_id="apple",
            title="Apples",
            text="An apple is a red fruit that grows on a tree.",
        ),
        Document(
            doc_id="banana",
            title="Bananas",
            text="A banana is a long yellow fruit.",
        ),
        Document(
            doc_id="cherry",
            title="Cherries",
            text="A cherry is a small red fruit with a stone.",
        ),
        Document(
            doc_id="embed",
            title="FastEmbed",
            text="FastEmbed supports sparse BM25 embeddings.",
        ),
    ]
    query = "Which fruit is red?"
    memory = BM25Retriever(documents)
    top = memory.search(query, k=2, max_passage_chars=200)
    _show("BM25Retriever.search(query, k=2)", top)
    blocked = {top[0].doc_id} if top else set()
    _show(
        f"BM25Retriever.search(..., exclude_ids={blocked})",
        memory.search(query, k=2, exclude_ids=blocked, max_passage_chars=200),
    )

    claim = "Nikolaj Coster-Waldau worked with the Fox Broadcasting Company."
    fever_path = SQLITE_INDEXES["fever"]
    if fever_path.is_file():
        fever = SQLiteBM25Retriever(fever_path)
        try:
            _show(
                "SQLiteBM25Retriever.search(claim, k=3)",
                fever.search(claim, k=3, max_passage_chars=240),
            )
        finally:
            fever.close()

        routed = ClaimRetriever(max_passage_chars=240)
        try:
            _show(
                'ClaimRetriever.search("fever", claim, k=3)',
                routed.search("fever", claim, k=3),
            )
        finally:
            routed.close()
    else:
        print(f"\nskip sqlite demo, missing {fever_path}")

    averitec_path = SQLITE_INDEXES["averitec"]
    if averitec_path.is_file():
        averitec = SQLiteBM25Retriever(averitec_path, claim_scoped=True)
        try:
            row = averitec.conn.execute(
                "SELECT claim_id, title FROM docs LIMIT 1"
            ).fetchone()
            if row and row[0]:
                averitec.set_claim(str(row[0]))
                _show(
                    f'SQLiteBM25Retriever.set_claim({row[0]!r}).search(...)',
                    averitec.search(str(row[1] or claim), k=2, max_passage_chars=240),
                )
        finally:
            averitec.close()

if __name__ == '__main__':
    main()
    