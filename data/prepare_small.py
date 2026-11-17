"""Build the small WWW27 corpora. Does not download FEVER or the climate wiki.

SciFact and HealthVer archives are a few megabytes. The script refuses any
download whose Content-Length is above 50MB, and it is meant to run on a
compute node.
"""
from __future__ import annotations

import hashlib
import json
import random
import tarfile
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path("/data3/yxfeng/www27")
RAW = ROOT / "raw"
ARCHIVES = RAW / "archives"
EXTRACTED = RAW / "extracted"
CLAIMS = ROOT / "preprocessed" / "claims"
CORPUS = ROOT / "preprocessed" / "corpus"
MAX_DOWNLOAD_BYTES = 50_000_000

SCIFACT_URL = "https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz"
HEALTHVER_URL = "https://scifact.s3.us-west-2.amazonaws.com/longchecker/latest/data.tar.gz"

CLIMATE_LABELS = {
    0: "SUPPORTS",
    1: "REFUTES",
    2: "NOT ENOUGH INFO",
    3: "DISPUTED",
}


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def to_py(value):
    if isinstance(value, dict):
        return {str(key): to_py(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_py(item) for item in value]
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes)):
        return to_py(value.tolist())
    if isinstance(value, float) and value != value:
        return None
    return value


def sha1_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def download(url: str, dest: Path) -> None:
    request = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(request, timeout=60) as response:
        length = int(response.headers.get("Content-Length") or 0)
    if length <= 0 or length > MAX_DOWNLOAD_BYTES:
        raise SystemExit(f"refuse download {url} content-length={length}")
    if dest.is_file() and dest.stat().st_size == length:
        print(f"keep {dest} ({length} bytes)", flush=True)
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"download {url} ({length} bytes) -> {dest}", flush=True)
    urllib.request.urlretrieve(url, dest)
    if dest.stat().st_size != length:
        raise SystemExit(f"short download {dest} {dest.stat().st_size} != {length}")


def extract(archive: Path, dest: Path) -> None:
    marker = dest / ".extracted"
    if marker.exists():
        return
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:gz") as handle:
        handle.extractall(dest)
    marker.write_text("ok\n", encoding="utf-8")


def find_named(root: Path, name: str, must_contain: str | None = None) -> Path:
    matches = [
        path for path in root.rglob(name)
        if must_contain is None or must_contain in path.as_posix()
    ]
    if not matches:
        raise FileNotFoundError(f"{name} under {root} contain={must_contain}")
    matches.sort(key=lambda path: len(path.as_posix()))
    return matches[0]


def as_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return " ".join(str(item).strip() for item in value if str(item).strip())
    return str(value).strip()


def evidence_labels(evidence) -> list[str]:
    labels = []
    if not isinstance(evidence, dict):
        return labels
    for entries in evidence.values():
        if isinstance(entries, dict):
            entries = [entries]
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if isinstance(entry, dict) and entry.get("label"):
                labels.append(str(entry["label"]).upper())
    return labels


def one_label(labels: list[str], support: str, refute: str) -> str:
    unique = set(labels)
    support_keys = {"SUPPORT", "SUPPORTS", "SUPPORTED"}
    refute_keys = {"CONTRADICT", "CONTRADICTS", "REFUTE", "REFUTES", "REFUTED"}
    if unique and unique <= support_keys:
        return support
    if unique and unique <= refute_keys:
        return refute
    if not unique:
        return "NOT ENOUGH INFO"
    return "DISPUTED"


def cited_text(claim: dict, corpus: dict[str, dict]) -> str:
    evidence = claim.get("evidence") or {}
    if not isinstance(evidence, dict):
        return ""
    parts = []
    for doc_id, entries in evidence.items():
        document = corpus.get(str(doc_id))
        sentences = []
        if document is not None:
            abstract = document.get("abstract") or document.get("text") or []
            if isinstance(abstract, str):
                abstract = [abstract]
            if isinstance(entries, dict):
                entries = [entries]
            indexes = []
            for entry in entries or []:
                if isinstance(entry, dict):
                    indexes.extend(int(idx) for idx in entry.get("sentences") or [])
            for index in indexes:
                if 0 <= index < len(abstract):
                    sentences.append(str(abstract[index]))
        if sentences:
            parts.append(" ".join(sentences))
    return "\n".join(parts)


def corpus_from_abstracts(rows: list[dict]) -> list[dict]:
    documents = []
    seen = set()
    for row in rows:
        doc_id = str(row.get("doc_id") or row.get("_id") or row.get("id"))
        if doc_id in seen:
            continue
        seen.add(doc_id)
        documents.append({
            "doc_id": doc_id,
            "title": as_text(row.get("title")),
            "text": as_text(row.get("abstract") or row.get("text")),
        })
    return documents


def claims_from_scifact_like(rows: list[dict], corpus: dict[str, dict], kind: str) -> list[dict]:
    support = "SUPPORT" if kind == "scifact" else "SUPPORTS"
    refute = "CONTRADICT" if kind == "scifact" else "REFUTES"
    claims = []
    for row in rows:
        label = one_label(evidence_labels(row.get("evidence")), support, refute)
        if kind != "scifact" and label == "NOT ENOUGH INFO":
            label = "NOT ENOUGH INFO"
        claims.append({
            "id": str(row.get("id")),
            "claim": as_text(row.get("claim")),
            "label": label,
            "evidence": "",
            "gold_evidence": cited_text(row, corpus),
        })
    return claims


def prepare_scifact(root: Path) -> dict:
    corpus_rows = read_jsonl(find_named(root, "corpus.jsonl"))
    corpus = corpus_from_abstracts(corpus_rows)
    original = {
        str(row.get("doc_id") or row.get("_id") or row.get("id")): row
        for row in corpus_rows
    }
    dev = claims_from_scifact_like(read_jsonl(find_named(root, "claims_dev.jsonl")), original, "scifact")
    train = claims_from_scifact_like(read_jsonl(find_named(root, "claims_train.jsonl")), original, "scifact")
    write_jsonl(CLAIMS / "scifact" / "dev.jsonl", dev)
    write_jsonl(CLAIMS / "scifact" / "train.jsonl", train)
    write_jsonl(CORPUS / "scifact.jsonl", corpus)
    return {"scifact/dev": len(dev), "scifact/train": len(train), "scifact/corpus": len(corpus)}


def prepare_healthver(root: Path) -> dict:
    corpus_path = find_named(root, "corpus.jsonl", must_contain="healthver")
    corpus_rows = read_jsonl(corpus_path)
    corpus = corpus_from_abstracts(corpus_rows)
    original = {
        str(row.get("doc_id") or row.get("_id") or row.get("id")): row
        for row in corpus_rows
    }
    claims_dir = corpus_path.parent
    train = claims_from_scifact_like(read_jsonl(claims_dir / "claims_train.jsonl"), original, "healthver")
    dev_path = claims_dir / "claims_dev.jsonl"
    test_path = claims_dir / "claims_test.jsonl"
    dev = claims_from_scifact_like(read_jsonl(dev_path), original, "healthver") if dev_path.exists() else []
    test = claims_from_scifact_like(read_jsonl(test_path), original, "healthver") if test_path.exists() else []
    eval_rows = test or dev
    if len(eval_rows) > 1000:
        eval_rows = random.Random(42).sample(eval_rows, 1000)
    write_jsonl(CLAIMS / "healthver" / "train.jsonl", train)
    write_jsonl(CLAIMS / "healthver" / "test.jsonl", eval_rows)
    write_jsonl(CORPUS / "healthver.jsonl", corpus)
    return {
        "healthver/train": len(train),
        "healthver/test": len(eval_rows),
        "healthver/corpus": len(corpus),
    }


def prepare_averitec() -> dict:
    frames = {
        "train": pd.read_parquet(RAW / "averitec/data/train-00000-of-00001.parquet"),
        "dev": pd.read_parquet(RAW / "averitec/data/dev-00000-of-00001.parquet"),
    }
    documents: dict[str, dict] = {}
    counts = {}
    for split, frame in frames.items():
        claims = []
        for index, record in frame.iterrows():
            row = to_py(record.to_dict())
            questions = row.get("questions") or []
            snippets = []
            for question in questions:
                if not isinstance(question, dict):
                    continue
                question_text = as_text(question.get("question"))
                for answer in question.get("answers") or []:
                    if not isinstance(answer, dict):
                        continue
                    text = as_text(answer.get("answer"))
                    if not text:
                        continue
                    snippets.append(text)
                    doc_id = sha1_text(text)
                    documents.setdefault(doc_id, {
                        "doc_id": doc_id,
                        "title": question_text[:240],
                        "text": text,
                        "url": as_text(answer.get("cached_source_url") or answer.get("source_url")),
                    })
            claim_id = row.get("claim_id") or row.get("id") or f"{split}-{index}"
            claims.append({
                "id": str(claim_id),
                "claim": as_text(row.get("claim")),
                "label": as_text(row.get("label")),
                "evidence": "",
                "gold_evidence": "\n".join(snippets),
            })
        write_jsonl(CLAIMS / "averitec" / f"{split}.jsonl", claims)
        counts[f"averitec/{split}"] = len(claims)
    write_jsonl(CORPUS / "averitec.jsonl", list(documents.values()))
    counts["averitec/corpus"] = len(documents)
    return counts


def prepare_climate() -> dict:
    frame = pd.read_parquet(RAW / "climate_fever/data/test-00000-of-00001.parquet")
    documents: dict[str, dict] = {}
    claims = []
    for _, record in frame.iterrows():
        row = to_py(record.to_dict())
        snippets = []
        for evidence in row.get("evidences") or []:
            if not isinstance(evidence, dict):
                continue
            text = as_text(evidence.get("evidence")).strip().strip('"')
            if not text:
                continue
            snippets.append(text)
            doc_id = str(evidence.get("evidence_id") or sha1_text(text))
            documents.setdefault(doc_id, {
                "doc_id": doc_id,
                "title": as_text(evidence.get("article")),
                "text": text,
            })
        label_id = int(row["claim_label"])
        claims.append({
            "id": str(row["claim_id"]),
            "claim": as_text(row.get("claim")),
            "label": CLIMATE_LABELS.get(label_id, "NOT ENOUGH INFO"),
            "evidence": "",
            "gold_evidence": "\n".join(snippets),
        })
    write_jsonl(CLAIMS / "climate_fever" / "test.jsonl", claims)
    write_jsonl(CORPUS / "climate_fever.jsonl", list(documents.values()))
    return {"climate_fever/test": len(claims), "climate_fever/corpus": len(documents)}


def main() -> None:
    download(SCIFACT_URL, ARCHIVES / "scifact_release.tar.gz")
    download(HEALTHVER_URL, ARCHIVES / "longchecker_data.tar.gz")
    extract(ARCHIVES / "scifact_release.tar.gz", EXTRACTED / "scifact")
    extract(ARCHIVES / "longchecker_data.tar.gz", EXTRACTED / "longchecker")
    counts = {}
    counts.update(prepare_scifact(EXTRACTED / "scifact"))
    counts.update(prepare_healthver(EXTRACTED / "longchecker"))
    counts.update(prepare_averitec())
    counts.update(prepare_climate())
    manifest = ROOT / "preprocessed" / "manifest.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(counts, indent=2), encoding="utf-8")
    print(json.dumps(counts, indent=2), flush=True)


if __name__ == "__main__":
    main()
