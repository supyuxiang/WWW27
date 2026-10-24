"""Shared I/O and benchmark adapters for native three-way baselines."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import statistics
import sys
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

CODE_ROOT = Path(__file__).resolve().parents[1]
VENDOR_ROOT = CODE_ROOT / "vendor" / "www27"
if str(VENDOR_ROOT) not in sys.path:
    sys.path.insert(0, str(VENDOR_ROOT))

from baselines import run_cfever_adapted as cf  # noqa: E402
from baselines import run_quantemp_plus_adapted as qt  # noqa: E402


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_json(path: Path, value: Any) -> None:
    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_benchmark(benchmark: str, split_file: str, limit: int, seed: int):
    if benchmark == "cfever":
        rows = [
            json.loads(line)
            for line in Path(split_file).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        return cf, cf.sample_rows(rows, limit, seed)
    source = json.loads(Path(split_file).read_text(encoding="utf-8"))
    rows = [dict(row, __split_index=index) for index, row in enumerate(source)]
    return qt, qt.proportional_sample(rows, limit, seed)


def make_retriever(module, index: str):
    return module.CFEVERRetriever(index) if module is cf else module.QuanTempRetriever(index)


def prepare_retriever(module, retriever, row: dict[str, Any]) -> None:
    if module is qt:
        retriever.set_cutoff(row.get("published"))


def gold_label(module, row: dict[str, Any]) -> str:
    return module.normalize_label(row["label"])


def metric_summary(rows: list[dict[str, Any]], labels: tuple[str, ...]) -> dict[str, Any]:
    # Every expected row remains in the denominator. Empty/invalid predictions
    # therefore count as errors rather than disappearing from the metric.
    accuracy = sum(row.get("prediction") == row["gold"] for row in rows) / len(rows) if rows else 0.0
    f1_values = []
    per_class = {}
    for label in labels:
        tp = sum(row["gold"] == label and row.get("prediction") == label for row in rows)
        fp = sum(row["gold"] != label and row.get("prediction") == label for row in rows)
        fn = sum(row["gold"] == label and row.get("prediction") != label for row in rows)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        f1_values.append(f1)
        per_class[label] = {"precision": precision, "recall": recall, "f1": f1, "support": tp + fn}
    return {
        "n": len(rows),
        "accuracy": accuracy,
        "macro_f1": statistics.fmean(f1_values) if f1_values else 0.0,
        "per_class": per_class,
        "prediction_counts": dict(Counter(str(row.get("prediction") or "INVALID") for row in rows)),
        "failures": sum(bool(row.get("error")) for row in rows),
        "coverage": sum(row.get("prediction") in labels for row in rows) / len(rows) if rows else 0.0,
        "mean_search_calls": statistics.fmean(row.get("search_calls", 0) for row in rows) if rows else 0.0,
        "mean_latency_s": statistics.fmean(row.get("latency_s", 0.0) for row in rows) if rows else 0.0,
    }


def config_payload(args, module, selected: list[dict[str, Any]], method: str, source: Path, extra=None):
    payload = {
        "track": f"{args.benchmark}_native_three_way_adapted",
        "method": method,
        "model": args.model,
        "model_frozen": True,
        "chat_template": "tokenizer.apply_chat_template",
        "enable_thinking": False,
        "seed": args.seed,
        "limit": args.limit,
        "n_selected": len(selected),
        "sample_ids": [str(row["id"]) for row in selected],
        "labels": list(module.LABELS),
        "split_file": args.split_file,
        "split_sha256": sha256(Path(args.split_file)),
        "index": args.index,
        "index_sha256": sha256(Path(args.index)),
        "gold_evidence_used_for_retrieval": False,
        "source_file": str(source.resolve()),
        "source_sha256": sha256(source.resolve()),
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    payload.update(extra or {})
    stable = {key: value for key, value in payload.items() if key not in {"started_at_utc", "finished_at_utc", "status"}}
    payload["config_fingerprint"] = hashlib.sha256(json.dumps(stable, sort_keys=True).encode()).hexdigest()
    return payload


def initialize_output(output: Path, config: dict[str, Any], selected: list[dict[str, Any]]) -> None:
    output.mkdir(parents=True, exist_ok=True)
    config_path = output / "config.json"
    if config_path.exists():
        prior = json.loads(config_path.read_text(encoding="utf-8"))
        if prior.get("config_fingerprint") != config["config_fingerprint"]:
            raise RuntimeError("Output directory contains an incompatible experiment configuration")
        config["started_at_utc"] = prior.get("started_at_utc", config["started_at_utc"])
    atomic_json(config_path, config)
    atomic_json(output / "sample_manifest.json", [{"id": str(row["id"]), "label": str(row["label"])} for row in selected])


def finalize_output(output: Path, config: dict[str, Any], rows: list[dict[str, Any]], labels: tuple[str, ...]) -> dict[str, Any]:
    summary = metric_summary(rows, labels)
    summary.update({"method": config["method"], "benchmark": config["track"].split("_native_")[0], "complete": len(rows) == config["n_selected"]})
    atomic_json(output / "summary.json", summary)
    flat = {key: value for key, value in summary.items() if key != "per_class"}
    flat["per_class"] = json.dumps(summary["per_class"], ensure_ascii=False, sort_keys=True)
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(flat))
    writer.writeheader()
    writer.writerow(flat)
    atomic_text(output / "summary.csv", buffer.getvalue())
    atomic_text(output / "failures.jsonl", "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows if row.get("error")))
    config["status"] = "complete" if not summary["failures"] else "complete_with_failures"
    config["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    atomic_json(output / "config.json", config)
    return summary
