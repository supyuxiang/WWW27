"""Summarize prediction files written by src.run."""

from __future__ import annotations

import argparse
import json

from src.data import read_jsonl
from src.utils.metrics import summarize


def _fmt(summary: dict) -> str:
    ece = summary.get("ece")
    ece_text = "n/a" if ece is None else f"{ece:.3f}"
    return (
        f"n={summary['n']:4d}  acc={100 * summary['accuracy']:6.2f}%  "
        f"macro-f1={100 * summary['macro_f1']:6.2f}%  "
        f"calls={summary['calls_per_item']:.3f}  "
        f"rounds={summary['rounds_per_item']:.2f}  "
        f"stop_gate={100 * summary['stop_gate_rate']:.1f}%  "
        f"stop_budget={100 * summary['stop_budget_rate']:.1f}%  "
        f"forwards={summary['forward_passes']:.2f}  "
        f"latency={summary['latency_s']:.2f}s  ece={ece_text} (n={summary['n_calibration']})  "
        f"errors={summary['n_errors']}"
    )


def print_report(name: str, rows: list[dict]) -> dict:
    summary = summarize(rows)
    print(f"\n{name}")
    print(f"  overall  {_fmt(summary)}")
    for domain, domain_summary in summary.get("by_domain", {}).items():
        print(f"  {domain:16s} {_fmt(domain_summary)}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate JARV prediction jsonl files")
    parser.add_argument("predictions", nargs="+")
    parser.add_argument("--json-out", default="")
    args = parser.parse_args()
    reports = {}
    for path in args.predictions:
        reports[path] = print_report(path, read_jsonl(path))
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as handle:
            json.dump(reports, handle, indent=2)
            handle.write("\n")


if __name__ == "__main__":
    main()
