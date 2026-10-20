"""Build a full AVeriTeC dev knowledge-store FTS5/BM25 index.

The official dev knowledge store is a zip of per-claim JSONL files.  Each
document is stored with its claim id, so the retriever can restrict search to
the candidate documents of the claim being evaluated.
"""

from __future__ import annotations

import argparse
import io
import json
import sqlite3
import time
import zipfile
from pathlib import Path


def build(store: Path, output: Path, max_chars: int, max_claims: int, batch_size: int) -> None:
    if output.exists():
        raise SystemExit(f"Output exists: {output}; remove it or choose another path")
    output.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(output)
    conn.execute("PRAGMA journal_mode=OFF")
    conn.execute("PRAGMA synchronous=OFF")
    conn.execute("PRAGMA temp_store=MEMORY")
    conn.execute("PRAGMA cache_size=-200000")
    conn.execute(
        "CREATE VIRTUAL TABLE docs USING fts5("
        "claim_id UNINDEXED, doc_id UNINDEXED, title, text, tokenize='porter unicode61')"
    )
    with zipfile.ZipFile(store) as archive:
        names = sorted(
            name for name in archive.namelist()
            if name.startswith("output_dev/") and name.endswith(".json")
        )
        if max_claims:
            names = names[:max_claims]
        started = time.time()
        total = 0
        batch: list[tuple[str, str, str, str]] = []
        for file_index, name in enumerate(names, start=1):
            claim_id = Path(name).stem
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
                        doc_id = f"{claim_id}:{line_no}"
                        batch.append(
                            (claim_id, doc_id, str(item.get("url") or ""), text[:max_chars])
                        )
                        total += 1
                        if len(batch) >= batch_size:
                            conn.executemany(
                                "INSERT INTO docs(claim_id,doc_id,title,text) VALUES (?,?,?,?)",
                                batch,
                            )
                            conn.commit()
                            batch.clear()
            if file_index % 25 == 0:
                print(
                    f"claims {file_index}/{len(names)} docs={total} "
                    f"elapsed={time.time() - started:.1f}s",
                    flush=True,
                )
        if batch:
            conn.executemany(
                "INSERT INTO docs(claim_id,doc_id,title,text) VALUES (?,?,?,?)", batch
            )
            conn.commit()
    conn.execute("INSERT INTO docs(docs) VALUES('optimize')")
    conn.commit()
    conn.close()
    print(json.dumps({"db": str(output), "documents": total, "claims": len(names),
                      "source": str(store), "seconds": round(time.time() - started, 1)}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--store",
        default="/root/autodl-tmp/www27/benchmark/averitec/data_store/knowledge_store/dev_knowledge_store.zip",
    )
    parser.add_argument("--output", default="/root/autodl-tmp/www27/index/averitec_dev_fts.sqlite")
    parser.add_argument("--max-chars", type=int, default=2000)
    parser.add_argument("--max-claims", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=2000)
    args = parser.parse_args()
    build(Path(args.store), Path(args.output), args.max_chars, args.max_claims, args.batch_size)


if __name__ == "__main__":
    main()
