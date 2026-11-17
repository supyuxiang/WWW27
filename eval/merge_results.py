"""Merge shard jsonl files and print binary accuracy."""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def main() -> None:
    directory = Path(sys.argv[1])
    grouped: dict[str, dict[str, dict]] = defaultdict(dict)
    for path in sorted(directory.glob("*.jsonl")):
        for row in load_jsonl(path):
            benchmark = str(row.get("benchmark") or "unknown")
            grouped[benchmark][str(row.get("id"))] = row
    report = {}
    for benchmark, rows_by_id in grouped.items():
        rows = list(rows_by_id.values())
        scored = [row for row in rows if row.get("prediction") is not None and row.get("gold") is not None]
        correct = sum(row["prediction"] == row["gold"] for row in scored)
        calls = [int(row.get("n_calls") or 0) for row in scored]
        latencies = [float(row["latency_s"]) for row in scored if row.get("latency_s") is not None]
        errors = sum(1 for row in rows if row.get("error"))
        report[benchmark] = {
            "n": len(scored),
            "n_errors": errors,
            "accuracy": correct / len(scored) if scored else 0.0,
            "mean_calls": sum(calls) / len(calls) if calls else 0.0,
            "mean_latency_s": sum(latencies) / len(latencies) if latencies else 0.0,
        }
        print(
            f"{benchmark}\tn={len(scored)}\tacc={report[benchmark]['accuracy']:.4f}"
            f"\tcalls={report[benchmark]['mean_calls']:.2f}"
            f"\tlatency_s={report[benchmark]['mean_latency_s']:.1f}"
            f"\terrors={errors}",
            flush=True,
        )
    out = directory / "merged.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("wrote", out, flush=True)


if __name__ == "__main__":
    main()
