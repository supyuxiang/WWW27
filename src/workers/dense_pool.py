"""Precomputed dense candidate pools. No BM25 and no embedding model."""
from __future__ import annotations

import json

from src.config import Passage
from src.paths import INDEX
from src.utils import load_corpus


class DensePool:
    def __init__(self, benchmark: str):
        documents = load_corpus(benchmark)
        self.docs = {document.doc_id: document for document in documents}
        pool_path = INDEX / benchmark / "pools.jsonl"
        if not pool_path.is_file():
            raise FileNotFoundError(
                f"missing dense pool {pool_path}. "
                "Build it with data/build_index.py on a compute node."
            )
        self.by_id: dict[str, list[str]] = {}
        self.by_claim: dict[str, list[str]] = {}
        with pool_path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                doc_ids = [str(doc_id) for doc_id in row["doc_ids"]]
                self.by_id[str(row["id"])] = doc_ids
                claim = row.get("claim") or ""
                if claim and claim not in self.by_claim:
                    self.by_claim[claim] = doc_ids

    def search(
        self,
        claim: str,
        k: int,
        exclude_ids: set[str] | None = None,
        claim_id: str | None = None,
        max_passage_chars: int = 2000,
    ) -> list[Passage]:
        doc_ids = None
        if claim_id is not None:
            doc_ids = self.by_id.get(str(claim_id))
        if doc_ids is None:
            doc_ids = self.by_claim.get(claim)
        if doc_ids is None:
            raise KeyError(f"no dense pool for claim id={claim_id!r}")
        blocked = exclude_ids or set()
        passages: list[Passage] = []
        for doc_id in doc_ids:
            if doc_id in blocked:
                continue
            document = self.docs.get(doc_id)
            if document is None:
                continue
            text = document.text
            if max_passage_chars > 0 and len(text) > max_passage_chars:
                text = text[:max_passage_chars]
            passages.append(Passage(
                doc_id=document.doc_id,
                text=text,
                title=document.title,
                url=document.url,
                date=document.date,
                query=claim,
            ))
            if len(passages) >= k:
                break
        return passages
