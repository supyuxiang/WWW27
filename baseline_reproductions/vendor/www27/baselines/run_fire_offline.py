"""FIRE baseline adapted to the fixed WWW27 offline retrieval stores.

The decision/search prompts and iterative control flow follow the official
FIRE implementation (Findings of NAACL 2025).  Only the search backend is
changed from Serper/Google to the same benchmark-specific offline indexes used
by JARV.  Outputs are atomic and resumable at sample granularity.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
import platform
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baselines.jev_only import load_benchmark
from baselines.run_main_experiment import RetrieverBundle, dataset_digest, sample_path
from src.llm import LocalLLM
from src.utils.metrics import accuracy, macro_f1

LOGGER = logging.getLogger("fire_offline")
BENCHMARKS = ("averitec", "fever", "scifact")
METHOD = "fire_offline"
FIRE_SYSTEM_PROMPT = "You are a fact-checking agent responsible for verifying the accuracy of claims."
REUSE_CONFIG_KEYS = (
    "model",
    "model_frozen",
    "seed",
    "chat_template",
    "enable_thinking",
    "temperature",
    "search_cap",
    "official_max_steps",
    "top_k",
    "max_passage_chars",
    "max_evidence_chars",
    "max_new_tokens",
    "final_retries",
    "repeat_model",
    "repeat_threshold",
    "repeat_window",
)

FIRE_PROMPT = """Instructions:
1. You are provided with a STATEMENT and relevant KNOWLEDGE points.
2. Based on the KNOWLEDGE, assess the factual accuracy of the STATEMENT.
3. Before presenting your conclusion, think through the process step-by-step.
   Include a summary of the key points from the KNOWLEDGE as part of your reasoning.
4. If the KNOWLEDGE allows you to confidently make a decision, output the final
   answer as a JSON object in the following format:
   {{
     "final_answer": "True" or "False"
   }}
5. If the KNOWLEDGE is insufficient to make a judgment, issue ONE Google Search
   query that could provide additional evidence. Output the search query in JSON
   format, as follows:
   {{
     "search_query": "Your Google search query here"
   }}
6. The query should aim to obtain new information not already present in the
   KNOWLEDGE, specifically helpful for verifying the STATEMENT's accuracy.

KNOWLEDGE:
{knowledge}

STATEMENT:
{claim}"""

FINAL_PROMPT = """Instructions:
1. You are provided with a STATEMENT and relevant KNOWLEDGE points.
2. Based on the KNOWLEDGE, assess the factual accuracy of the STATEMENT.
3. Before presenting your final answer, think step-by-step and show your reasoning.
   Include a summary of the key points from the KNOWLEDGE as part of your reasoning.
4. Your final answer should be either "True" or "False".
5. Format your final answer as a JSON object in the following structure:
   {{
     "final_answer": "True" or "False"
   }}

KNOWLEDGE:
{knowledge}

STATEMENT:
{claim}"""


def atomic_write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_text(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def extract_json(text: str) -> dict[str, Any] | None:
    """Return the last valid JSON object, matching FIRE's answer-at-end style."""
    candidates = re.findall(r"\{[^{}]*\}", text, flags=re.DOTALL)
    for candidate in reversed(candidates):
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None


def normalized_query(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", value.lower()))


class SemanticRepeatDetector:
    """FIRE's all-MiniLM-L6-v2 cosine-similarity early stopper."""

    def __init__(self, model_path: str, threshold: float = 0.9):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.threshold = threshold
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        self.model = AutoModel.from_pretrained(model_path, local_files_only=True)
        self.model.eval()

    def encode(self, texts: list[str]):
        torch = self.torch
        encoded = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        with torch.inference_mode():
            hidden = self.model(**encoded).last_hidden_state
        mask = encoded["attention_mask"].unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
        return torch.nn.functional.normalize(pooled, p=2, dim=1)

    def repeated(self, candidate: str, history: list[str]) -> bool:
        if not history:
            return False
        embeddings = self.encode([candidate, *history])
        similarities = embeddings[0:1] @ embeddings[1:].T
        return bool((similarities[0] > self.threshold).all().item())


def format_hits(hits: list[Any]) -> str:
    if not hits:
        return "No relevant result was returned."
    blocks = []
    for index, hit in enumerate(hits, start=1):
        title = str(hit.title or "").strip()
        prefix = f"[{index}] {title}" if title else f"[{index}]"
        blocks.append(f"{prefix}\n{str(hit.text).strip()}")
    return "\n\n".join(blocks)


def knowledge_text(searches: list[dict[str, Any]], max_chars: int) -> str:
    value = "\n\n".join(item["result_text"] for item in searches).strip()
    if not value:
        return "N/A"
    return value[-max_chars:]


def answer_value(value: object) -> str | None:
    text = str(value or "").strip().lower()
    if text == "true":
        return "real"
    if text == "false":
        return "fake"
    return None


def run_one(
    raw: dict[str, Any],
    benchmark: str,
    llm: LocalLLM,
    retriever: Any,
    repeat_detector: SemanticRepeatDetector | None,
    args: argparse.Namespace,
) -> dict[str, Any]:
    started = time.perf_counter()
    searches: list[dict[str, Any]] = []
    generations: list[dict[str, Any]] = []
    seen_queries: set[str] = set()
    prediction: str | None = None
    stop_reason = ""
    parse_failures = 0
    input_tokens = 0
    output_tokens = 0

    # FIRE defaults to five reasoning steps.  The common experiment protocol
    # independently caps actual retrieval calls at three.
    for step in range(1, args.max_steps + 1):
        prompt = FIRE_PROMPT.format(
            knowledge=knowledge_text(searches, args.max_evidence_chars),
            claim=raw["claim"],
        )
        generation = llm.generate(FIRE_SYSTEM_PROMPT, prompt, args.max_new_tokens)
        input_tokens += generation.usage.input_tokens
        output_tokens += generation.usage.output_tokens
        parsed = extract_json(generation.text)
        generations.append({"step": step, "forced_final": False, "raw": generation.text, "parsed": parsed})
        if parsed:
            prediction = answer_value(parsed.get("final_answer"))
            if prediction:
                stop_reason = "model_final_answer"
                break
            query = str(parsed.get("search_query") or "").strip()
        else:
            query = ""
        if not query:
            parse_failures += 1
            continue
        normalized = normalized_query(query)
        query_history = [item["query"] for item in searches]
        result_history = [item["result_text"] for item in searches]
        window = max(1, args.repeat_window - 1)
        query_repeated = (
            repeat_detector.repeated(query, query_history[-window:])
            if repeat_detector
            else normalized_query(query) in seen_queries
        )
        result_repeated = bool(
            repeat_detector
            and len(result_history) >= args.repeat_window
            and repeat_detector.repeated(result_history[-1], result_history[-args.repeat_window:-1])
        )
        if query_repeated or result_repeated:
            stop_reason = "semantic_repeat_query" if query_repeated else "semantic_repeat_results"
            break
        if len(searches) >= args.search_cap:
            stop_reason = "search_budget_exhausted"
            break
        seen_queries.add(normalized)
        retrieval_started = time.perf_counter()
        hits = retriever.search(
            query,
            k=args.top_k,
            exclude_ids=set(),
            max_passage_chars=args.max_passage_chars,
        )
        searches.append(
            {
                "query": query,
                "latency_seconds": time.perf_counter() - retrieval_started,
                "result_text": format_hits(hits),
                "hits": [
                    {
                        "doc_id": hit.doc_id,
                        "title": hit.title,
                        "score": hit.score,
                        "text": hit.text,
                    }
                    for hit in hits
                ],
            }
        )

    if prediction is None:
        for retry in range(args.final_retries + 1):
            prompt = FINAL_PROMPT.format(
                knowledge=knowledge_text(searches, args.max_evidence_chars),
                claim=raw["claim"],
            )
            generation = llm.generate(FIRE_SYSTEM_PROMPT, prompt, args.max_new_tokens)
            input_tokens += generation.usage.input_tokens
            output_tokens += generation.usage.output_tokens
            parsed = extract_json(generation.text)
            generations.append(
                {"step": len(generations) + 1, "forced_final": True, "retry": retry, "raw": generation.text, "parsed": parsed}
            )
            prediction = answer_value((parsed or {}).get("final_answer"))
            if prediction:
                stop_reason = stop_reason or "forced_final"
                break
            parse_failures += 1

    error = None if prediction else "FIRE returned no valid True/False final answer"
    return {
        "id": str(raw["id"]),
        "benchmark": benchmark,
        "method": METHOD,
        "claim": raw["claim"],
        "gold": raw["gold"],
        "prediction": prediction,
        "correct": prediction == raw["gold"] if prediction else False,
        "stop_reason": stop_reason or "invalid_output",
        "search_calls": len(searches),
        "retrieval_rounds": len(searches),
        "searches": searches,
        "generations": generations,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_generated_tokens": output_tokens,
        "parse_failures": parse_failures,
        "latency_seconds": time.perf_counter() - started,
        "error": error,
    }


def config_fingerprint(config: dict[str, Any]) -> str:
    stable = {key: value for key, value in config.items() if key not in {"started_at", "output_root"}}
    return hashlib.sha256(json.dumps(stable, sort_keys=True).encode()).hexdigest()


def import_compatible_predictions(
    source_root: Path,
    output_root: Path,
    datasets: dict[str, list[dict[str, Any]]],
    config: dict[str, Any],
    fingerprint: str,
) -> dict[str, Any]:
    """Reuse successful rows when every sample-level inference setting matches."""
    source_config_path = source_root / "config.json"
    if not source_config_path.exists():
        raise RuntimeError(f"Reuse source has no config.json: {source_root}")
    source_config = json.loads(source_config_path.read_text(encoding="utf-8"))
    mismatched = [
        key for key in REUSE_CONFIG_KEYS if source_config.get(key) != config.get(key)
    ]
    if mismatched:
        raise RuntimeError(
            f"Incompatible reuse source {source_root}; mismatched: {', '.join(mismatched)}"
        )
    imported: dict[str, int] = {}
    for benchmark, expected_rows in datasets.items():
        expected_ids = {str(row["id"]) for row in expected_rows}
        count = 0
        directory = source_root / "predictions" / METHOD / benchmark
        for path in directory.glob("*.json") if directory.exists() else []:
            try:
                row = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            item_id = str(row.get("id", ""))
            if (
                item_id not in expected_ids
                or row.get("error") not in {None, ""}
                or row.get("prediction") not in {"real", "fake"}
            ):
                continue
            target = sample_path(output_root, METHOD, benchmark, item_id)
            if target.exists():
                continue
            row["config_fingerprint"] = fingerprint
            row["checkpoint_reused_from"] = str(source_root)
            atomic_write_json(target, row)
            count += 1
        imported[benchmark] = count
    return {"source": str(source_root), "imported": imported}


def build_outputs(root: Path, datasets: dict[str, list[dict]], config: dict[str, Any]) -> None:
    summaries = []
    failures = []
    for benchmark, expected in datasets.items():
        rows = []
        directory = root / "predictions" / METHOD / benchmark
        for path in directory.glob("*.json") if directory.exists() else []:
            row = json.loads(path.read_text(encoding="utf-8"))
            rows.append(row)
            if row.get("error"):
                failures.append({"benchmark": benchmark, "id": row.get("id"), "error": row["error"]})
        valid = [row for row in rows if row.get("prediction") in {"real", "fake"}]
        gold = [row["gold"] for row in valid]
        pred = [row["prediction"] for row in valid]
        summaries.append(
            {
                "method": METHOD,
                "benchmark": benchmark,
                "n_expected": len(expected),
                "n_completed": len(valid),
                "accuracy": accuracy(gold, pred) if valid else 0.0,
                "macro_f1": macro_f1(gold, pred) if valid else 0.0,
                "mean_search_calls": sum(row["search_calls"] for row in valid) / len(valid) if valid else 0.0,
                "mean_latency_seconds": sum(row["latency_seconds"] for row in valid) / len(valid) if valid else 0.0,
                "parse_failures": sum(row.get("parse_failures", 0) for row in rows),
                "complete": len(valid) == len(expected),
            }
        )
    fields = list(summaries[0]) if summaries else []
    lines = []
    if fields:
        import io
        buffer = io.StringIO()
        writer = csv.DictWriter(buffer, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summaries)
        lines = buffer.getvalue()
    atomic_write_text(root / "summary.csv", lines)
    atomic_write_text(root / "failures.jsonl", "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in failures))
    report = [
        "# FIRE-offline smoke report",
        "",
        "FIRE's published iterative prompt/control flow was retained. Serper was replaced by the fixed offline benchmark indexes; retrieval was capped at three calls and top-k=3 for budget parity with JARV.",
        "",
        "| Benchmark | N | Accuracy | Macro-F1 | Mean searches | Mean latency (s) | Complete |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in summaries:
        report.append(
            f"| {row['benchmark']} | {row['n_completed']}/{row['n_expected']} | {row['accuracy']:.4f} | {row['macro_f1']:.4f} | {row['mean_search_calls']:.3f} | {row['mean_latency_seconds']:.3f} | {row['complete']} |"
        )
    report += ["", "## Protocol deviations", "", *[f"- {item}" for item in config["protocol_deviations"]]]
    atomic_write_text(root / "report.md", "\n".join(report) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--benchmarks", nargs="+", choices=BENCHMARKS, default=list(BENCHMARKS))
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--search-cap", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--max-passage-chars", type=int, default=800)
    parser.add_argument("--max-evidence-chars", type=int, default=6000)
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--final-retries", type=int, default=2)
    parser.add_argument("--repeat-model", default="")
    parser.add_argument("--repeat-threshold", type=float, default=0.9)
    parser.add_argument("--repeat-window", type=int, default=2)
    parser.add_argument("--reuse-root", type=Path, action="append", default=[])
    parser.add_argument("--sample-manifest", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    log_path = args.output_root / "logs" / "run.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(log_path, encoding="utf-8"), logging.StreamHandler()],
    )
    datasets = {name: load_benchmark(name, args.limit, args.seed) for name in args.benchmarks}
    if args.sample_manifest:
        manifest = json.loads(args.sample_manifest.read_text(encoding="utf-8"))
        for name in args.benchmarks:
            wanted = [str(item) for item in manifest[name]["ids"]]
            by_id = {str(row["id"]): row for row in load_benchmark(name, 0, args.seed)}
            missing = [item for item in wanted if item not in by_id]
            if missing:
                raise SystemExit(f"Manifest contains unknown {name} IDs: {missing[:10]}")
            datasets[name] = [by_id[item] for item in wanted]
    config = {
        "objective": "FIRE published baseline adapted to WWW27 fixed offline retrieval",
        "paper": "FIRE: Fact-checking with Iterative Retrieval and Verification (Findings NAACL 2025)",
        "official_code": "https://github.com/mbzuai-nlp/fire",
        "model": args.model,
        "model_frozen": True,
        "seed": args.seed,
        "chat_template": True,
        "enable_thinking": False,
        "temperature": 0.0,
        "method": METHOD,
        "benchmarks": args.benchmarks,
        "limit": args.limit,
        "search_cap": args.search_cap,
        "official_max_steps": args.max_steps,
        "top_k": args.top_k,
        "max_passage_chars": args.max_passage_chars,
        "max_evidence_chars": args.max_evidence_chars,
        "max_new_tokens": args.max_new_tokens,
        "final_retries": args.final_retries,
        "repeat_model": args.repeat_model,
        "repeat_threshold": args.repeat_threshold,
        "repeat_window": args.repeat_window,
        "sampling": ("explicit sample_manifest IDs in manifest order" if args.sample_manifest else "random.Random(seed).sample(binary_split, limit), then sort by id"),
        "sample_manifest": str(args.sample_manifest) if args.sample_manifest else None,
        "dataset_counts": {name: len(rows) for name, rows in datasets.items()},
        "dataset_sha256": {name: dataset_digest(rows) for name, rows in datasets.items()},
        "protocol_deviations": [
            "Serper/Google search replaced by the same fixed offline index used by JARV for each benchmark.",
            "Actual retrieval calls capped at 3 and each call returns top-k=3 for comparison-budget parity.",
            "The frozen local Qwen backbone replaces FIRE's hosted/model-specific backbones.",
        ],
        "environment": {"python": sys.version, "platform": platform.platform()},
        "output_root": str(args.output_root),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    fingerprint = config_fingerprint(config)
    config["config_fingerprint"] = fingerprint
    config_path = args.output_root / "config.json"
    if config_path.exists():
        old = json.loads(config_path.read_text(encoding="utf-8"))
        if old.get("config_fingerprint") != fingerprint:
            raise SystemExit("Output root contains a different experiment configuration")
    else:
        atomic_write_json(config_path, config)

    reuse_manifest = [
        import_compatible_predictions(
            source_root, args.output_root, datasets, config, fingerprint
        )
        for source_root in args.reuse_root
    ]
    if reuse_manifest:
        atomic_write_json(args.output_root / "checkpoint_reuse_manifest.json", reuse_manifest)

    llm = LocalLLM(args.model, use_chat_template=True)
    repeat_detector = (
        SemanticRepeatDetector(args.repeat_model, args.repeat_threshold)
        if args.repeat_model
        else None
    )
    retrievers = RetrieverBundle(args)
    try:
        total = sum(len(rows) for rows in datasets.values())
        done = 0
        failed = 0
        for benchmark in args.benchmarks:
            for raw in datasets[benchmark]:
                target = sample_path(args.output_root, METHOD, benchmark, str(raw["id"]))
                if target.exists():
                    try:
                        old = json.loads(target.read_text(encoding="utf-8"))
                        if old.get("config_fingerprint") == fingerprint and old.get("prediction") in {"real", "fake"}:
                            done += 1
                            continue
                    except Exception:
                        pass
                try:
                    retriever = retrievers.for_example(benchmark, str(raw["id"]))
                    result = run_one(raw, benchmark, llm, retriever, repeat_detector, args)
                except Exception as exc:
                    LOGGER.exception("Failure %s/%s", benchmark, raw["id"])
                    result = {
                        "id": str(raw["id"]), "benchmark": benchmark, "method": METHOD,
                        "claim": raw["claim"], "gold": raw["gold"], "prediction": None,
                        "correct": False, "search_calls": 0, "retrieval_rounds": 0,
                        "searches": [], "generations": [], "parse_failures": 0,
                        "latency_seconds": 0.0, "error": f"{type(exc).__name__}: {exc}",
                    }
                result["config_fingerprint"] = fingerprint
                atomic_write_json(target, result)
                done += 1
                failed += int(bool(result.get("error")))
                LOGGER.info("%s %d/%d failed=%d remaining=%d id=%s searches=%s", benchmark, done, total, failed, total - done, raw["id"], result.get("search_calls"))
    finally:
        retrievers.close()
    build_outputs(args.output_root, datasets, config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
