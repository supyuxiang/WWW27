"""Recompute accuracy and macro-F1 from standard_eval JSONL outputs.

The shared `src.utils.metrics.macro_f1` defaults to the binary labels
("real", "fake"); the standard-label harness has 3/4 classes. This script
therefore computes the macro-F1 from the observed label set explicitly.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

LABELS = ("supported", "refuted", "nei", "conflicting")


def metrics(gold: list[str], pred: list[str]) -> dict:
    labels = sorted(set(gold) | set(pred))
    n = len(gold)
    correct = sum(g == p for g, p in zip(gold, pred))
    per_class = {}
    for label in labels:
        tp = sum(g == label and p == label for g, p in zip(gold, pred))
        fp = sum(g != label and p == label for g, p in zip(gold, pred))
        fn = sum(g == label and p != label for g, p in zip(gold, pred))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = {"precision": precision, "recall": recall, "f1": f1}
    return {
        "n": n,
        "accuracy": correct / n if n else 0.0,
        "macro_f1": sum(v["f1"] for v in per_class.values()) / len(per_class)
        if per_class
        else 0.0,
        "per_class": per_class,
        "gold_counts": dict(Counter(gold)),
        "prediction_counts": dict(Counter(pred)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.input)
    groups: dict[tuple[str, str], list[dict]] = {}
    for path in sorted(root.glob("*.jsonl")):
        if ".distill." in path.name:
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("error") or row.get("prediction") not in LABELS:
                continue
            groups.setdefault((row["method"], row["benchmark"]), []).append(row)

    rows = []
    for (method, benchmark), items in sorted(groups.items()):
        gold = [r["gold"] for r in items]
        pred = [r["prediction"] for r in items]
        m = metrics(gold, pred)
        rows.append({"method": method, "benchmark": benchmark, **m})
        print(
            f"{method:16s} {benchmark:10s} n={m['n']:4d} "
            f"acc={m['accuracy']:.4f} macroF1={m['macro_f1']:.4f}"
        )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    lines = [
        "# Standard-label comparison",
        "",
        "| Method | Benchmark | n | Accuracy | Macro-F1 |",
        "|---|---|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['method']} | {row['benchmark']} | {row['n']} | "
            f"{row['accuracy']:.4f} | {row['macro_f1']:.4f} |"
        )
    out.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out.with_suffix(".md"))


if __name__ == "__main__":
    main()
