"""Build a full FEVER Wikipedia FTS5/BM25 index.

CPU-only: stores the index on disk under /root/autodl-tmp/www27/index.
Each wiki page becomes one row (page id, title, full text).  The existing
in-memory BM25 in src/retrieval.py is not suitable for 5.4M pages.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import time
from pathlib import Path


def build(source: Path, output: Path, max_chars: int, limit: int, batch_size: int) -> None:
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
        "doc_id UNINDEXED, title, text, tokenize='porter unicode61')"
    )
    files = sorted(source.rglob("wiki-*.jsonl"))
    if not files:
        raise SystemExit(f"No wiki-*.jsonl under {source}")
    started = time.time()
    total = 0
    batch: list[tuple[str, str, str]] = []
    for file_index, path in enumerate(files, start=1):
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if limit and total >= limit:
                    break
                try:
                    page = json.loads(line)
                except json.JSONDecodeError:
                    continue
                doc_id = str(page.get("id") or "").strip()
                text = str(page.get("text") or "").strip()
                if not doc_id or not text:
                    continue
                text = text[:max_chars]
                batch.append((doc_id, doc_id.replace("_", " "), text))
                total += 1
                if len(batch) >= batch_size:
                    conn.executemany("INSERT INTO docs(doc_id,title,text) VALUES (?,?,?)", batch)
                    conn.commit()
                    batch.clear()
                    if total % 100000 == 0:
                        print(f"indexed {total} pages in {time.time() - started:.1f}s", flush=True)
        print(f"file {file_index}/{len(files)} total={total}", flush=True)
        if limit and total >= limit:
            break
    if batch:
        conn.executemany("INSERT INTO docs(doc_id,title,text) VALUES (?,?,?)", batch)
        conn.commit()
    conn.execute("INSERT INTO docs(docs) VALUES('optimize')")
    conn.commit()
    conn.close()
    print(json.dumps({"db": str(output), "documents": total, "source": str(source),
                      "seconds": round(time.time() - started, 1)}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="/root/autodl-tmp/www27/benchmark/fever")
    parser.add_argument("--output", default="/root/autodl-tmp/www27/index/fever_fts.sqlite")
    parser.add_argument("--max-chars", type=int, default=4000)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=5000)
    args = parser.parse_args()
    build(Path(args.source), Path(args.output), args.max_chars, args.limit, args.batch_size)


if __name__ == "__main__":
    main()
