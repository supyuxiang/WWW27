"""Prepare AVeriTeC, FEVER, and SciFact claims/corpora for JARV BM25 runs.

The prepared files use the same schema as ``src.data.load_examples`` and
``src.data.load_corpus``.  For AVeriTeC we use the per-claim knowledge store
as the candidate-document pool.  For FEVER we build a closed candidate pool
from the sampled claims' gold pages plus a bounded number of Wikipedia pages.
SciFact uses its full 5,183-abstract corpus.
"""

from __future__ import annotations

import argparse
import io
import json
import random
import sys
import zipfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

BENCH_ROOT = Path("/root/autodl-tmp/www27/benchmark")
RAW_SCI = BENCH_ROOT / "scifact" / "data"
RAW_FEVER = BENCH_ROOT / "fever"
RAW_AVERITEC = BENCH_ROOT / "averitec"


def normalize_label(value: object) -> str | None:
    text = str(value or "").strip().lower()
    if text in {"supported", "supports", "support", "true", "real"}:
        return "real"
    if text in {"refuted", "refutes", "refute", "false", "fake", "contradict"}:
        return "fake"
    return None


def scifact_label(record: dict) -> str | None:
    labels: set[str] = set()

    def visit(value: object) -> None:
        if isinstance(value, dict):
            label = value.get("label")
            if isinstance(label, str):
                labels.add(label.strip().upper())
            for item in value.values():
                visit(item)
        elif isinstance(value, list):
            for item in value:
                visit(item)

    visit(record.get("evidence") or {})
    if labels == {"SUPPORT"}:
        return "real"
    if labels and labels <= {"CONTRADICT", "REFUTE"}:
        return "fake"
    return None


def jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def sample(rows: list[dict], limit: int, seed: int) -> list[dict]:
    if limit > 0 and len(rows) > limit:
        rows = random.Random(seed).sample(rows, limit)
        rows.sort(key=lambda item: str(item["id"]))
    return rows


def write(out_dir: Path, claims: list[dict], corpus: list[dict], stats: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "claims.jsonl").open("w", encoding="utf-8") as handle:
        for row in claims:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (out_dir / "corpus.jsonl").open("w", encoding="utf-8") as handle:
        for row in corpus:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(stats, indent=2, sort_keys=True), flush=True)


def prepare_scifact(limit: int, seed: int, out_root: Path, max_doc_chars: int) -> None:
    rows = list(jsonl(RAW_SCI / "claims_dev.jsonl"))
    claims = []
    for row in rows:
        label = scifact_label(row)
        if not label:
            continue
        claims.append(
            {
                "id": f"scifact-{row['id']}",
                "claim": str(row["claim"]),
                "label": label,
                "domain": "scifact",
                "title": "",
                "text": "",
                "source_id": f"scifact-{row['id']}",
            }
        )
    claims = sample(claims, limit, seed)
    corpus = []
    for row in jsonl(RAW_SCI / "corpus.jsonl"):
        title = str(row.get("title") or "")
        abstract = row.get("abstract") or []
        text = " ".join(str(part) for part in abstract)
        corpus.append(
            {
                "doc_id": f"scifact:{row['doc_id']}",
                "title": title,
                "text": (title + "\n" + text).strip()[:max_doc_chars],
            }
        )
    write(
        out_root / "scifact",
        claims,
        corpus,
        {"benchmark": "scifact", "claims": len(claims), "corpus": len(corpus), "source": str(RAW_SCI)},
    )


def prepare_fever(limit: int, seed: int, out_root: Path, max_doc_chars: int, distractors: int) -> None:
    candidates = []
    for index, row in enumerate(jsonl(RAW_FEVER / "shared_task_dev.jsonl")):
        label = normalize_label(row.get("label"))
        if not label:
            continue
        gold_pages: set[str] = set()
        for group in row.get("evidence") or []:
            for item in group or []:
                if isinstance(item, list) and len(item) >= 3 and isinstance(item[2], str):
                    gold_pages.add(item[2])
        page_id = str(row.get("id") or index)
        candidates.append(
            {
                "id": f"fever-{page_id}",
                "claim": str(row["claim"]),
                "label": label,
                "domain": "fever",
                "title": str(row.get("claim") or ""),
                "text": str(row.get("claim") or ""),
                "source_id": f"fever-{page_id}",
                "gold_pages": sorted(gold_pages),
            }
        )
    claims = sample(candidates, limit, seed)
    targets = {page for row in claims for page in row["gold_pages"]}
    found: dict[str, dict] = {}
    filler: list[dict] = []
    files = sorted((RAW_FEVER / "wiki-pages").glob("wiki-pages/wiki-*.jsonl"))
    for file_index, path in enumerate(files, start=1):
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                try:
                    page = json.loads(line)
                except json.JSONDecodeError:
                    continue
                page_id = str(page.get("id") or "")
                text = str(page.get("text") or "")
                if not page_id or not text:
                    continue
                if page_id in targets:
                    found[page_id] = {
                        "doc_id": f"fever:{page_id}",
                        "title": page_id,
                        "text": text[:max_doc_chars],
                    }
                elif len(filler) < distractors:
                    filler.append(
                        {
                            "doc_id": f"fever:{page_id}",
                            "title": page_id,
                            "text": text[:max_doc_chars],
                        }
                    )
        if file_index % 10 == 0:
            print(f"fever wiki files {file_index}/{len(files)}, gold {len(found)}/{len(targets)}", flush=True)
    corpus = list(filler) + list(found.values())
    for row in claims:
        row.pop("gold_pages", None)
    write(
        out_root / "fever",
        claims,
        corpus,
        {
            "benchmark": "fever",
            "claims": len(claims),
            "corpus": len(corpus),
            "gold_pages_found": len(found),
            "gold_pages_missing": len(targets) - len(found),
            "distractors": len(filler),
            "source": str(RAW_FEVER),
        },
    )


def prepare_averitec(limit: int, seed: int, out_root: Path, max_doc_chars: int) -> None:
    rows = json.loads((RAW_AVERITEC / "data" / "dev.json").read_text(encoding="utf-8"))
    if isinstance(rows, dict):
        rows = rows.get("data") or rows.get("claims") or []
    candidates = []
    for index, row in enumerate(rows):
        label = normalize_label(row.get("label"))
        if not label:
            continue
        candidates.append(
            {
                "id": f"averitec-{index}",
                "claim": str(row["claim"]),
                "label": label,
                "domain": "averitec",
                "title": "",
                "text": "",
                "source_id": f"averitec-{index}",
                "_claim_index": index,
            }
        )
    claims = sample(candidates, limit, seed)
    store = RAW_AVERITEC / "data_store" / "knowledge_store" / "dev_knowledge_store.zip"
    corpus = []
    missing = 0
    with zipfile.ZipFile(store) as archive:
        names = set(archive.namelist())
        for row in claims:
            name = f"output_dev/{row['_claim_index']}.json"
            if name not in names:
                missing += 1
                continue
            with archive.open(name) as raw:
                with io.TextIOWrapper(raw, encoding="utf-8") as handle:
                    for line_no, line in enumerate(handle):
                        try:
                            item = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        text = item.get("url2text") or item.get("text") or ""
                        if isinstance(text, list):
                            text = " ".join(str(part) for part in text)
                        text = str(text).strip()
                        if not text:
                            continue
                        corpus.append(
                            {
                                "doc_id": f"averitec:{row['_claim_index']}:{line_no}",
                                "title": str(item.get("url") or ""),
                                "text": text[:max_doc_chars],
                            }
                        )
    for row in claims:
        row.pop("_claim_index", None)
    write(
        out_root / "averitec",
        claims,
        corpus,
        {
            "benchmark": "averitec",
            "claims": len(claims),
            "corpus": len(corpus),
            "missing_knowledge_files": missing,
            "source": str(RAW_AVERITEC),
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare benchmark claims and BM25 corpora")
    parser.add_argument("--benchmarks", nargs="+", default=["averitec", "fever", "scifact"])
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-root", default="/root/www27/data/benchmarks_prepared")
    parser.add_argument("--max-doc-chars", type=int, default=2000)
    parser.add_argument("--fever-distractors", type=int, default=20000)
    args = parser.parse_args()
    out_root = Path(args.out_root)
    for name in args.benchmarks:
        if name == "scifact":
            prepare_scifact(args.limit, args.seed, out_root, args.max_doc_chars)
        elif name == "fever":
            prepare_fever(args.limit, args.seed, out_root, args.max_doc_chars, args.fever_distractors)
        elif name == "averitec":
            prepare_averitec(args.limit, args.seed, out_root, args.max_doc_chars)
        else:
            raise SystemExit(f"Unknown benchmark {name}")


if __name__ == "__main__":
    main()
