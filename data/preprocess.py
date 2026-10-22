"""Build the fixed SciFact/FEVER/AVeriTeC/Climate-FEVER/HealthVer data.

Outputs:
  /root/autodl-tmp/benchmark_dsh/preprocessed/claims/<benchmark>/<split>.jsonl
  /root/autodl-tmp/benchmark_dsh/preprocessed/corpus/<benchmark>.jsonl

All claim records use:
  {"id": str, "claim": str, "label": str, "evidence": str}
All corpus records use:
  {"doc_id": str, "title": str, "text": str}
"""
from __future__ import annotations

import ast
import csv
import hashlib
import json
import random
import zipfile
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download

DATA = Path("/root/yxfeng/www27/data")
BENCH = Path("/root/autodl-tmp/benchmark_dsh")
OUT = BENCH / "preprocessed"
SAMPLE = BENCH / "samples"
WIKI = Path("/root/autodl-tmp/www27/benchmark/fever/wiki-pages/wiki-pages")
AVERITEC_ZIP = Path("/root/autodl-tmp/www27/benchmark/averitec/data_store/knowledge_store/dev_knowledge_store.zip")
CLIMATE_PARQUET = BENCH / "raw/climate_fever/data/test-00000-of-00001.parquet"
HEALTHVER_RAW = BENCH / "raw/HealthVer/data"


def write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_corpus(name: str, rows) -> None:
    path = OUT / f"corpus/{name}.jsonl"
    if path.exists() and path.stat().st_size > 0:
        print("skip existing corpus", path, flush=True)
        return
    write_jsonl(path, rows)


def load_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def scifact_labels(row: dict) -> str:
    evidence = row.get("evidence") or {}
    if not evidence:
        return "NOT ENOUGH INFO"
    labels = []
    for entries in evidence.values():
        for entry in entries or []:
            if isinstance(entry, dict) and entry.get("label"):
                labels.append(entry["label"])
    unique = list(dict.fromkeys(labels))
    return unique[0] if len(unique) == 1 else "NOT ENOUGH INFO"


def scifact_corpus_map() -> dict[str, dict]:
    return {str(row["doc_id"]): row for row in load_jsonl(DATA / "scifact/corpus.jsonl")}


def scifact_claims(corpus: dict[str, dict], path: Path) -> list[dict]:
    records = []
    for row in load_jsonl(path):
        chunks = []
        for doc_id, entries in (row.get("evidence") or {}).items():
            doc = corpus.get(str(doc_id))
            if not doc:
                continue
            abstract = doc.get("abstract") or []
            sentences = []
            for entry in entries or []:
                for index in entry.get("sentences") or []:
                    try:
                        sentences.append(str(abstract[int(index)]))
                    except (IndexError, TypeError, ValueError):
                        pass
            if sentences:
                chunks.append(f"{doc.get('title') or ''}\n" + " ".join(sentences))
        records.append({
            "id": str(row.get("id")),
            "claim": str(row.get("claim") or ""),
            "label": scifact_labels(row),
            "evidence": "\n\n".join(chunks),
        })
    return records


def fever_evidence(claim: dict, pages: dict[str, dict]) -> str:
    if str(claim.get("label") or "") == "NOT ENOUGH INFO":
        return ""
    chunks = []
    for group in claim.get("evidence") or []:
        if not isinstance(group, list):
            continue
        for item in group:
            if not isinstance(item, list) or not item:
                continue
            title = str(item[2]) if len(item) > 2 and item[2] else ""
            page = pages.get(title) if title else None
            if page is None and item[0] is not None:
                page = pages.get(str(item[0]))
            if page:
                chunks.append(f"{page['title']}\n{page['text']}")
            elif title:
                chunks.append(title)
            elif item[0] is not None:
                chunks.append(str(item[0]))
    return "\n\n".join(dict.fromkeys(chunks))


def fever_pages(needed: set[str] | None = None):
    if not WIKI.exists():
        return
    for path in sorted(WIKI.glob("*.jsonl")):
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                page_id = str(row.get("id") or "")
                if not page_id:
                    continue
                if needed is not None and page_id not in needed:
                    continue
                yield page_id, {
                    "doc_id": page_id,
                    "title": str(row.get("title") or ""),
                    "text": str(row.get("text") or ""),
                }


def averitec_evidence(claim: dict) -> str:
    chunks = []
    for question in claim.get("questions") or []:
        for answer in question.get("answers") or []:
            text = str(answer.get("answer") or "").strip()
            if text:
                chunks.append(text)
    if not chunks:
        chunks.append(str(claim.get("justification") or ""))
    return "\n\n".join(dict.fromkeys(chunks))


def climate_label(value) -> str:
    return {0: "SUPPORTS", 1: "REFUTES", 2: "NOT ENOUGH INFO"}.get(int(value), "NOT ENOUGH INFO")


def climate_evidence(value) -> str:
    try:
        items = ast.literal_eval(value) if isinstance(value, str) else value
    except (ValueError, SyntaxError):
        return ""
    if items is None:
        return ""
    if hasattr(items, "tolist"):
        items = items.tolist()
    chunks = []
    for item in items:
        if isinstance(item, dict):
            text = str(item.get("evidence") or "").strip()
            article = str(item.get("article") or "").strip()
            if text:
                chunks.append(f"{article}\n{text}" if article else text)
    return "\n\n".join(dict.fromkeys(chunks))


def healthver_label(value: str) -> str:
    normalized = str(value).strip().casefold()
    if normalized in {"supports", "support"}:
        return "SUPPORTS"
    if normalized in {"refutes", "refute", "contradicts"}:
        return "REFUTES"
    return "NOT ENOUGH INFO"


def build_claims() -> dict:
    counts = {}

    # SciFact
    corpus = scifact_corpus_map()
    for split, path in [("dev", DATA / "scifact/claims_dev.jsonl"), ("train", DATA / "scifact/claims_train.jsonl")]:
        rows = scifact_claims(corpus, path)
        write_jsonl(OUT / "claims/scifact" / f"{split}.jsonl", rows)
        counts[f"scifact/{split}"] = len(rows)
    write_corpus(
        "scifact",
        ({"doc_id": str(row["doc_id"]), "title": str(row.get("title") or ""),
          "text": " ".join(str(x) for x in (row.get("abstract") or []))}
         for row in load_jsonl(DATA / "scifact/corpus.jsonl")),
    )

    # FEVER
    for split, path in [("dev", SAMPLE / "fever_dev1000.jsonl"), ("train", DATA / "fever/train.jsonl")]:
        rows = list(load_jsonl(path))
        needed = set()
        for row in rows:
            for group in row.get("evidence") or []:
                for item in group if isinstance(group, list) else []:
                    if isinstance(item, list) and item:
                        if item[0] is not None:
                            needed.add(str(item[0]))
                        if len(item) > 2 and item[2]:
                            needed.add(str(item[2]))
        pages = dict(fever_pages(needed))
        out = []
        for row in rows:
            out.append({
                "id": str(row.get("id")),
                "claim": str(row.get("claim") or ""),
                "label": str(row.get("label") or "NOT ENOUGH INFO"),
                "evidence": fever_evidence(row, pages),
            })
        write_jsonl(OUT / "claims/fever" / f"{split}.jsonl", out)
        counts[f"fever/{split}"] = len(out)
    write_corpus("fever", (page for _, page in fever_pages()))

    # AVeriTeC
    for split, path in [("dev", SAMPLE / "averitec_dev300.json"), ("train", DATA / "averitec/train.json")]:
        rows = json.loads(path.read_text(encoding="utf-8"))
        out = []
        for index, row in enumerate(rows):
            out.append({
                "id": str(row.get("id") or row.get("claim_id") or index),
                "claim": str(row.get("claim") or ""),
                "label": str(row.get("label") or "Not Enough Evidence"),
                "evidence": averitec_evidence(row),
            })
        write_jsonl(OUT / "claims/averitec" / f"{split}.jsonl", out)
        counts[f"averitec/{split}"] = len(out)
    with zipfile.ZipFile(AVERITEC_ZIP) as archive:
        def averitec_docs():
            for name in archive.namelist():
                if not name.endswith(".json"):
                    continue
                with archive.open(name) as handle:
                    for line in handle:
                        line = line.strip()
                        if not line:
                            continue
                        row = json.loads(line)
                        text = "\n".join(str(x) for x in (row.get("url2text") or []))
                        if not text:
                            continue
                        yield {
                            "doc_id": f"{row.get('claim_id')}:{row.get('url')}",
                            "title": str(row.get("query") or ""),
                            "text": text,
                        }
        write_corpus("averitec", averitec_docs())

    # Climate-FEVER
    climate = pd.read_parquet(CLIMATE_PARQUET)
    climate_rows = []
    for row in climate.to_dict(orient="records"):
        climate_rows.append({
            "id": str(row.get("claim_id")),
            "claim": str(row.get("claim") or ""),
            "label": climate_label(row.get("claim_label")),
            "evidence": climate_evidence(row.get("evidences")),
        })
    rng = random.Random(42)
    climate_rows = rng.sample(climate_rows, min(500, len(climate_rows)))
    write_jsonl(OUT / "claims/climate_fever/test.jsonl", climate_rows)
    counts["climate_fever/test"] = len(climate_rows)
    corpus_file = hf_hub_download(
        repo_id="mteb/climate-fever",
        repo_type="dataset",
        filename="corpus.jsonl",
        local_dir=str(BENCH / "raw/climate_fever_beir"),
    )
    write_corpus(
        "climate_fever",
        ({"doc_id": str(row.get("_id") or row.get("doc_id")),
          "title": str(row.get("title") or ""),
          "text": str(row.get("text") or "")}
         for row in load_jsonl(Path(corpus_file))),
    )

    # HealthVer
    for split, path, limit in [
        ("test", HEALTHVER_RAW / "healthver_test.csv", 1000),
        ("train", HEALTHVER_RAW / "healthver_train.csv", -1),
    ]:
        with path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        if limit > 0:
            rows = random.Random(42).sample(rows, min(limit, len(rows)))
        out = [{
            "id": str(row.get("id")),
            "claim": str(row.get("claim") or ""),
            "label": healthver_label(row.get("label")),
            "evidence": str(row.get("evidence") or ""),
        } for row in rows]
        write_jsonl(OUT / "claims/healthver" / f"{split}.jsonl", out)
        counts[f"healthver/{split}"] = len(out)
    evidence_texts = {}
    for split in ["train", "dev", "test"]:
        path = HEALTHVER_RAW / f"healthver_{split}.csv"
        if not path.exists():
            continue
        with path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                text = str(row.get("evidence") or "").strip()
                if text:
                    doc_id = hashlib.sha1(text.encode("utf-8")).hexdigest()
                    evidence_texts[doc_id] = {"doc_id": doc_id, "title": "", "text": text}
    write_corpus("healthver", evidence_texts.values())

    (OUT / "manifest.json").write_text(json.dumps(counts, indent=2), encoding="utf-8")
    print(json.dumps(counts, indent=2), flush=True)
    return counts


def main() -> None:
    build_claims()


if __name__ == "__main__":
    main()
