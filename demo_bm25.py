from __future__ import annotations

import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")

from fastembed import SparseTextEmbedding


MODEL_NAME = "Qdrant/bm25"
MODEL_CACHE = "/root/autodl-tmp/models/fastembed"


def sparse_dot(query, document) -> float:
    q = dict(zip(query.indices.tolist(), query.values.tolist()))
    return sum(
        q.get(int(index), 0.0) * float(value)
        for index, value in zip(document.indices, document.values)
    )


def main() -> None:
    model = SparseTextEmbedding(
        model_name=MODEL_NAME,
        cache_dir=MODEL_CACHE,
        local_files_only=True,
    )

    documents = [
        "An apple is a red fruit that grows on a tree.",
        "A banana is a long yellow fruit.",
        "A cherry is a small red fruit with a stone.",
        "A blueberry is a small blue fruit.",
        "FastEmbed supports sparse BM25 embeddings.",
    ]
    query = "Which fruit is red?"

    doc_vectors = list(model.embed(documents))
    query_vector = list(model.query_embed(query))[0]

    breakpoint()
    
    ranked = sorted(
        zip(documents, doc_vectors),
        key=lambda item: sparse_dot(query_vector, item[1]),
        reverse=True,
    )

    print("query:", query)
    print("query nnz:", len(query_vector.indices))
    print()
    for rank, (document, vector) in enumerate(ranked, start=1):
        print(f"{rank}. score={sparse_dot(query_vector, vector):.4f}")
        print(f"   nnz={len(vector.indices)}")
        print(f"   {document}")


if __name__ == "__main__":
    main()
