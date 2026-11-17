"""Embed small corpora with Qwen3-Embedding-4B and store claim-level pools.

Run this on a compute GPU. The login node has no CUDA.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.paths import CLAIMS, CORPUS, EMBED_MODEL, INDEX

TASK = (
    "Given a claim, retrieve passages that contain evidence useful for verifying the claim."
)
POOL_K = 500
MAX_LENGTH = 512
BATCH_SIZE = 8

CLAIM_FILES = {
    "scifact": CLAIMS / "scifact" / "dev.jsonl",
    "averitec": CLAIMS / "averitec" / "dev.jsonl",
    "healthver": CLAIMS / "healthver" / "test.jsonl",
    "climate_fever": CLAIMS / "climate_fever" / "test.jsonl",
}


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def last_token_pool(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    left_padding = bool((mask[:, -1] == 1).all())
    if left_padding:
        return hidden[:, -1]
    lengths = mask.sum(dim=1) - 1
    return hidden[torch.arange(hidden.shape[0], device=hidden.device), lengths]


def load_encoder():
    tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL, padding_side="left", local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    try:
        model = AutoModel.from_pretrained(
            EMBED_MODEL,
            dtype=torch.bfloat16,
            local_files_only=True,
            attn_implementation="sdpa",
        )
    except Exception:
        model = AutoModel.from_pretrained(
            EMBED_MODEL,
            dtype=torch.bfloat16,
            local_files_only=True,
        )
    model.to("cuda")
    model.eval()
    return tokenizer, model


def embed_texts(tokenizer, model, texts: list[str], batch_size: int = BATCH_SIZE) -> np.ndarray:
    vectors = []
    total = len(texts)
    for start in range(0, total, batch_size):
        batch = texts[start:start + batch_size]
        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )
        encoded = {key: value.to("cuda") for key, value in encoded.items()}
        with torch.inference_mode():
            output = model(**encoded, output_hidden_states=True)
            hidden = getattr(output, "last_hidden_state", None)
            if hidden is None:
                hidden = output.hidden_states[-1]
            pooled = last_token_pool(hidden, encoded["attention_mask"])
            pooled = F.normalize(pooled.float(), p=2, dim=1)
        vectors.append(pooled.cpu().numpy())
        done = min(start + batch_size, total)
        if done == total or done % (batch_size * 20) == 0:
            print(f"  embedded {done}/{total}", flush=True)
    if not vectors:
        return np.zeros((0, 0), dtype=np.float32)
    return np.concatenate(vectors, axis=0)


def document_text(row: dict) -> str:
    title = str(row.get("title") or "").strip()
    text = str(row.get("text") or "").strip()
    if title and text:
        return f"{title}\n{text}"
    return title or text


def query_text(claim: str) -> str:
    return f"Instruct: {TASK}\nQuery:{claim}"


def build_one(name: str, tokenizer, model) -> None:
    corpus_path = CORPUS / f"{name}.jsonl"
    claim_path = CLAIM_FILES[name]
    documents = read_jsonl(corpus_path)
    claims = read_jsonl(claim_path)
    print(f"{name}: {len(documents)} docs, {len(claims)} claims", flush=True)
    doc_matrix = embed_texts(tokenizer, model, [document_text(row) for row in documents])
    query_matrix = embed_texts(tokenizer, model, [query_text(row["claim"]) for row in claims])
    out_dir = INDEX / name
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "embeddings.f16.npy", doc_matrix.astype(np.float16))
    (out_dir / "doc_ids.json").write_text(
        json.dumps([str(row["doc_id"]) for row in documents]),
        encoding="utf-8",
    )
    k = min(POOL_K, doc_matrix.shape[0])
    doc_tensor = torch.from_numpy(doc_matrix).to("cuda")
    with (out_dir / "pools.jsonl").open("w", encoding="utf-8") as handle:
        step = 64
        for start in range(0, len(claims), step):
            queries = torch.from_numpy(query_matrix[start:start + step]).to("cuda")
            scores = queries @ doc_tensor.T
            top = torch.topk(scores, k=k, dim=1).indices.cpu().tolist()
            for offset, indexes in enumerate(top):
                claim = claims[start + offset]
                handle.write(json.dumps({
                    "id": str(claim["id"]),
                    "claim": claim["claim"],
                    "doc_ids": [str(documents[index]["doc_id"]) for index in indexes],
                }, ensure_ascii=False) + "\n")
            print(f"  pooled {min(start + step, len(claims))}/{len(claims)}", flush=True)
    del doc_tensor
    torch.cuda.empty_cache()


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("build_index needs a CUDA device. Run it inside the compute job.")
    tokenizer, model = load_encoder()
    for name in CLAIM_FILES:
        build_one(name, tokenizer, model)
    print("index ready", INDEX, flush=True)


if __name__ == "__main__":
    main()
