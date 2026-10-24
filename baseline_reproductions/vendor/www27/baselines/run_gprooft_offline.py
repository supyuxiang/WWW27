"""GProofT adapted to the fixed WWW27 binary/offline protocol.

The claim decomposition, binary-question generation, snippet-level Yes/No
voting, and per-question verdict aggregation follow the official GProofT
repository (FEVER 2024). Hosted Azure ChatGPT and Google Custom Search are
replaced by a frozen local model and the benchmark-specific offline indexes.
Every completed sample is atomically checkpointed and can be resumed.
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
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baselines.jev_only import load_benchmark
from baselines.run_main_experiment import RetrieverBundle, dataset_digest, sample_path
from src.llm import LocalLLM
from src.utils.metrics import accuracy, macro_f1

LOGGER = logging.getLogger("gprooft_offline")
BENCHMARKS = ("averitec", "fever", "scifact")
METHOD = "gprooft_offline"
SYSTEM_PROMPT = "You are a helpful assistant."
OFFICIAL_REPO = "https://github.com/HKUST-KnowComp/GProofT"
OFFICIAL_COMMIT = "18de5d570805e7333ae2e514696b52a5603bb910"
REUSE_CONFIG_KEYS = (
    "official_commit",
    "model",
    "model_frozen",
    "seed",
    "chat_template",
    "enable_thinking",
    "temperature",
    "method",
    "search_cap",
    "top_k",
    "max_subclaims",
    "max_passage_chars",
    "generation_limits",
)

SPLIT_PROMPT = '''Now I have a mission, and please help me deal with it: I have a claim: {claim},
and I need to split it into different subclaims according to THE FACT it contains.
For example, if I have a claim: "Trump is a student born in 2005", then I want to split it into two parts
(since there are two facts in it):"Trump is a student" and "Trump was born in 2005".
For this special case, I need the response to be: "Trump is a student. Trump was born in 2005.".
There are several RULES for the spliting process:
(1)VERY IMPORTANT!!! PLEASE RETURN THE SUBCLAIMS ONLY (DO NOT INCLUDE ANYTHING ELSE!!!)
and please separate the subclaims ONLY BY PERIOD instead of numbers.
(2)VERY IMPORTANT: DO NOT GENERATE DUPLICATE SUBCLAIMS!!!!!!!
(3)TRY TO BE MORE SPECIFIC and CLEAR(for example, if you want to generate "the orgarnization",
try to generate the orgarnization's name), and AVOID USING PRONOUNS.
(4)In most cases, the length of subclaims should be LESS THAN the length of the original claim.
And in most cases, each subclaims SHOULD NOT BE LONGER THAN 10 words.
(5)Do not expand the meaning of the original claim or generate subclaims that do not exist in the original claim.
(6)DO NOT generate a subclaim that is totally the same as the original claim UNLESS there is only one fact to check in the original claim.
(7)For example: for the claim "BJP MP Sushil Modi claims first five Indian education ministers were Muslims",
You should recognize that there is ONLY ONE FACT in the claim, which is whether BJP MP Sushil Modi really states the following claim,
so the subclaim should be itself. At the same time, if there are several facts in the claim,
you should split the claim into same amount of subclaim, each representing a fact.
(8)If the claim is more that 30 words, try to generate at least 3 subclaims.
(9)Here are some EXAMPLES:
If the claim is "Lionel Messi is 36-year-old football player who has a long career.",
then according to the claim, there are three facts introducing Lionel Messi, which are:
Lionel Messi is 36-year-old, Lionel Messi is a football player, LionelMessi has a long career.
So what you should generate is: "Lionel Messi is 36-year-old. Lionel Messi is a football player. Lionel Messi has a long career.".'''

QUESTION_PROMPT = '''according to the claim below, generate a binary question to CHECK THE FACTS in the claim: {subclaim}.
Note that (1)ONLY REPLY THE QUESTION ITSELF!!! DO NOT INCLUDE ANYTHING ELSE!!!
(2)Try to be more SPECIFIC, for example, if the claim is "Trump was a student.",
then you should AVOID GENERATING QUESTIONS CONTAINING PRONOUNS like "Was he a student?"
and instead generate "Was Trump a student?"
(3)Try to NOTICE THE FACT in the claim and generate the binary question to CHECK THE FACT.
For example: for a claim: "BJP MP Sushil Modi claims first five Indian education ministers were Muslims", the fact
to be checked will be whether BJP MP Sushil really states the claim, instead of whether the first five Indian
education ministers are Muslims.
Thus, you should generate "Did BJP MP Sushil Modi claim that the first five Indian education ministers were Muslims?"
(4)Here are some EXAMPLES:
If the claim is "Lionel Messi is loyal to FC Barcelona",
then the binary question should be "Is Lionel Messi loyal to FC Barcelona?".
If the claim is "Biden has been to Beijing twice.", then the binary question should be "Has Biden been to Beijing twice?".'''

ANSWER_PROMPT = "According to the question: {question} and the approximate answer: {snippet}, give me a yes or no answer.(only a word is needed)"

VERDICT_PROMPT = "Claim: {claim} Question: {question} Answer: {answer} Based on the above question and answer, provide a verdict to the claim(can only be supported, refuted, irrelevant to the claim), remember to add a punct like format 'supported.' The verdict is:"

JUSTIFICATION_PROMPT = "Claim: {claim} {qa_text}Based on the above questions and answers, we provide a final verdict to the claim: {verdict}. Justify this verdict with a one sentence explanation. The justification is:"


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


def clean_generation(text: str) -> str:
    value = text.strip()
    value = re.sub(r"^```(?:json|text)?\s*", "", value, flags=re.IGNORECASE)
    value = re.sub(r"\s*```$", "", value)
    return value.strip()


def parse_subclaims(text: str, original: str, maximum: int) -> list[str]:
    value = clean_generation(text)
    value = re.sub(r"^\s*(?:subclaims?|answer)\s*:\s*", "", value, flags=re.IGNORECASE)
    parts = [re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", item).strip(" \t\r\n.\"")
             for item in re.split(r"(?:\r?\n)+|(?<=[.!?])\s+", value)]
    unique: list[str] = []
    seen: set[str] = set()
    for part in parts:
        key = " ".join(part.lower().split())
        if len(part) < 3 or key in seen:
            continue
        seen.add(key)
        unique.append(part)
        if len(unique) >= maximum:
            break
    return unique or [original.strip()]


def parse_yes_no(text: str) -> str | None:
    match = re.search(r"\b(yes|no)\b", clean_generation(text), flags=re.IGNORECASE)
    return match.group(1).title() if match else None


def parse_verdict(text: str) -> str:
    first = clean_generation(text).split(".", 1)[0].lower()
    if "supported" in first:
        return "supported"
    if "refuted" in first:
        return "refuted"
    return "irrelevant to the claim"


def aggregate_verdict(verdicts: list[str]) -> str:
    if "supported" in verdicts and "refuted" in verdicts:
        return "Conflicting Evidence/Cherrypicking"
    if verdicts and all(item == "supported" for item in verdicts):
        return "supported"
    if verdicts and all(item == "refuted" for item in verdicts):
        return "refuted"
    return "not enough evidence"


def binary_projection(native_verdict: str, verdicts: list[str]) -> tuple[str | None, str]:
    """Project native GProofT labels onto the fixed binary benchmark.

    Supported and Refuted are exact mappings. A mixed supported/refuted result
    means the complete claim is not fully supported and is therefore projected
    to fake. NEI remains invalid instead of silently inventing a binary label.
    """
    if native_verdict == "supported":
        return "real", "native_supported"
    if native_verdict == "refuted":
        return "fake", "native_refuted"
    if native_verdict == "Conflicting Evidence/Cherrypicking" and "refuted" in verdicts:
        return "fake", "conflicting_projects_to_not_fully_supported"
    return None, "native_not_enough_evidence"


def hit_dict(hit: Any) -> dict[str, Any]:
    return {"doc_id": hit.doc_id, "title": hit.title, "score": hit.score, "text": hit.text}


def run_one(raw: dict[str, Any], benchmark: str, llm: LocalLLM, retriever: Any, args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    input_tokens = output_tokens = parse_failures = search_calls = 0
    generations: list[dict[str, Any]] = []
    qa_rows: list[dict[str, Any]] = []

    split = llm.generate(SYSTEM_PROMPT, SPLIT_PROMPT.format(claim=raw["claim"]), args.split_max_new_tokens)
    input_tokens += split.usage.input_tokens
    output_tokens += split.usage.output_tokens
    subclaims = parse_subclaims(split.text, raw["claim"], args.max_subclaims)
    generations.append({"stage": "split_claim", "raw": split.text, "parsed": subclaims})

    question_outputs = llm.generate_many(
        SYSTEM_PROMPT,
        [QUESTION_PROMPT.format(subclaim=item) for item in subclaims],
        args.question_max_new_tokens,
    )
    questions: list[str] = []
    for subclaim, generated in zip(subclaims, question_outputs):
        input_tokens += generated.usage.input_tokens
        output_tokens += generated.usage.output_tokens
        question = clean_generation(generated.text).splitlines()[0].strip()
        if not question:
            parse_failures += 1
            question = subclaim
        questions.append(question)
        generations.append({"stage": "generate_binary", "subclaim": subclaim, "raw": generated.text, "parsed": question})

    # The original method searches once for every generated question. The fixed
    # comparison protocol caps total calls, so questions after the cap remain
    # recorded but are not searched.
    for subclaim, question in zip(subclaims, questions):
        if search_calls >= args.search_cap:
            qa_rows.append({"subclaim": subclaim, "question": question, "skipped": "search_cap", "hits": [], "answers": [], "answer": None})
            continue
        retrieval_started = time.perf_counter()
        hits = retriever.search(question, k=args.top_k, exclude_ids=set(), max_passage_chars=args.max_passage_chars)
        search_calls += 1
        answer_outputs = llm.generate_many(
            SYSTEM_PROMPT,
            [ANSWER_PROMPT.format(question=question, snippet=hit.text) for hit in hits],
            args.answer_max_new_tokens,
        )
        answers: list[dict[str, Any]] = []
        votes: list[str] = []
        for hit, generated in zip(hits, answer_outputs):
            input_tokens += generated.usage.input_tokens
            output_tokens += generated.usage.output_tokens
            parsed = parse_yes_no(generated.text)
            if parsed is None:
                parse_failures += 1
            else:
                votes.append(parsed)
            answers.append({"doc_id": hit.doc_id, "raw": generated.text, "parsed": parsed})
        majority = None
        if votes:
            # Official GProofT resolves ties toward Yes.
            majority = "Yes" if votes.count("Yes") >= votes.count("No") else "No"
        qa_rows.append({
            "subclaim": subclaim,
            "question": question,
            "retrieval_latency_seconds": time.perf_counter() - retrieval_started,
            "hits": [hit_dict(hit) for hit in hits],
            "answers": answers,
            "answer": majority,
        })

    judged = [row for row in qa_rows if row.get("answer") in {"Yes", "No"}]
    verdict_outputs = llm.generate_many(
        SYSTEM_PROMPT,
        [VERDICT_PROMPT.format(claim=raw["claim"], question=row["question"], answer=row["answer"]) for row in judged],
        args.verdict_max_new_tokens,
    )
    verdicts: list[str] = []
    for row, generated in zip(judged, verdict_outputs):
        input_tokens += generated.usage.input_tokens
        output_tokens += generated.usage.output_tokens
        verdict = parse_verdict(generated.text)
        verdicts.append(verdict)
        row["verdict_raw"] = generated.text
        row["verdict"] = verdict
        generations.append({"stage": "qa_verdict", "question": row["question"], "raw": generated.text, "parsed": verdict})

    native_verdict = aggregate_verdict(verdicts)
    prediction, projection_reason = binary_projection(native_verdict, verdicts)
    justification = ""
    if judged:
        qa_text = "".join(f"Question {index}: {row['question']} Answer {index}: {row['answer']} " for index, row in enumerate(judged, 1))
        generated = llm.generate(
            SYSTEM_PROMPT,
            JUSTIFICATION_PROMPT.format(claim=raw["claim"], qa_text=qa_text, verdict=native_verdict),
            args.justification_max_new_tokens,
        )
        input_tokens += generated.usage.input_tokens
        output_tokens += generated.usage.output_tokens
        justification = clean_generation(generated.text).split(".", 1)[0].strip()
        if justification:
            justification += "."
        generations.append({"stage": "justification", "raw": generated.text, "parsed": justification})

    error = None if prediction else "GProofT native verdict could not be projected to binary"
    return {
        "id": str(raw["id"]),
        "benchmark": benchmark,
        "method": METHOD,
        "claim": raw["claim"],
        "gold": raw["gold"],
        "prediction": prediction,
        "correct": prediction == raw["gold"] if prediction else False,
        "native_verdict": native_verdict,
        "binary_projection_reason": projection_reason,
        "justification": justification,
        "subclaims": subclaims,
        "qa_pairs": qa_rows,
        "generations": generations,
        "search_calls": search_calls,
        "retrieval_rounds": search_calls,
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
    source_config_path = source_root / "config.json"
    if not source_config_path.exists():
        raise RuntimeError(f"Reuse source has no config.json: {source_root}")
    source_config = json.loads(source_config_path.read_text(encoding="utf-8"))
    mismatched = [key for key in REUSE_CONFIG_KEYS if source_config.get(key) != config.get(key)]
    if mismatched:
        raise RuntimeError(f"Incompatible reuse source {source_root}; mismatched: {', '.join(mismatched)}")
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
            terminal = row.get("prediction") in {"real", "fake"} or row.get("native_verdict") in {
                "supported", "refuted", "Conflicting Evidence/Cherrypicking", "not enough evidence"
            }
            if item_id not in expected_ids or not terminal:
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


def build_outputs(root: Path, datasets: dict[str, list[dict[str, Any]]], config: dict[str, Any]) -> None:
    summaries: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for benchmark, expected in datasets.items():
        directory = root / "predictions" / METHOD / benchmark
        rows = [json.loads(path.read_text(encoding="utf-8")) for path in directory.glob("*.json")] if directory.exists() else []
        failures.extend({"benchmark": benchmark, "id": row.get("id"), "error": row["error"]} for row in rows if row.get("error"))
        predicted = [row for row in rows if row.get("prediction") in {"real", "fake"}]
        gold = [row["gold"] for row in predicted]
        pred = [row["prediction"] for row in predicted]
        summaries.append({
            "method": METHOD,
            "benchmark": benchmark,
            "n_expected": len(expected),
            "n_processed": len(rows),
            "n_predicted": len(predicted),
            "accuracy_on_predicted": accuracy(gold, pred) if predicted else 0.0,
            "macro_f1_on_predicted": macro_f1(gold, pred) if predicted else 0.0,
            "coverage": len(predicted) / len(expected) if expected else 0.0,
            "mean_search_calls": sum(row.get("search_calls", 0) for row in rows) / len(rows) if rows else 0.0,
            "mean_latency_seconds": sum(row.get("latency_seconds", 0.0) for row in rows) / len(rows) if rows else 0.0,
            "input_tokens": sum(row.get("input_tokens", 0) for row in rows),
            "output_tokens": sum(row.get("output_tokens", 0) for row in rows),
            "parse_failures": sum(row.get("parse_failures", 0) for row in rows),
            "complete": len(rows) == len(expected),
        })
    if summaries:
        import io
        buffer = io.StringIO()
        writer = csv.DictWriter(buffer, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
        atomic_write_text(root / "summary.csv", buffer.getvalue())
    atomic_write_text(root / "failures.jsonl", "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in failures))
    report = [
        "# GProofT native-offline adapted report",
        "",
        "Official GProofT decomposition, binary-question generation, snippet voting, and local-model verdict aggregation are retained. Hosted services are replaced by the fixed local model and offline benchmark indexes.",
        "",
        "| Benchmark | Processed | Predicted | Accuracy* | Macro-F1* | Coverage | Mean searches | Mean latency (s) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        report.append(f"| {row['benchmark']} | {row['n_processed']}/{row['n_expected']} | {row['n_predicted']} | {row['accuracy_on_predicted']:.4f} | {row['macro_f1_on_predicted']:.4f} | {row['coverage']:.4f} | {row['mean_search_calls']:.3f} | {row['mean_latency_seconds']:.3f} |")
    report += ["", "* Accuracy and Macro-F1 are computed on valid binary projections; coverage is reported separately.", "", "## Protocol deviations", "", *[f"- {item}" for item in config["protocol_deviations"]]]
    atomic_write_text(root / "report.md", "\n".join(report) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--benchmarks", nargs="+", choices=BENCHMARKS, default=list(BENCHMARKS))
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--search-cap", type=int, default=3)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--max-subclaims", type=int, default=3)
    parser.add_argument("--max-passage-chars", type=int, default=800)
    parser.add_argument("--split-max-new-tokens", type=int, default=192)
    parser.add_argument("--question-max-new-tokens", type=int, default=96)
    parser.add_argument("--answer-max-new-tokens", type=int, default=16)
    parser.add_argument("--verdict-max-new-tokens", type=int, default=64)
    parser.add_argument("--justification-max-new-tokens", type=int, default=128)
    parser.add_argument("--sample-manifest", type=Path)
    parser.add_argument("--reuse-root", type=Path, action="append", default=[])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    log_path = args.output_root / "logs" / "run.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", handlers=[logging.FileHandler(log_path, encoding="utf-8"), logging.StreamHandler()])
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
        "objective": "GProofT official mechanism adapted to the WWW27 fixed binary/offline protocol",
        "paper": "GProofT: A Multi-dimension Multi-round Fact Checking Framework Based on Claim Fact Extraction (FEVER 2024)",
        "official_code": OFFICIAL_REPO,
        "official_commit": OFFICIAL_COMMIT,
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
        "top_k": args.top_k,
        "max_subclaims": args.max_subclaims,
        "max_passage_chars": args.max_passage_chars,
        "generation_limits": {"split": args.split_max_new_tokens, "question": args.question_max_new_tokens, "answer": args.answer_max_new_tokens, "verdict": args.verdict_max_new_tokens, "justification": args.justification_max_new_tokens},
        "sample_manifest": str(args.sample_manifest) if args.sample_manifest else None,
        "dataset_counts": {name: len(rows) for name, rows in datasets.items()},
        "dataset_sha256": {name: dataset_digest(rows) for name, rows in datasets.items()},
        "protocol_deviations": [
            "Azure ChatGPT is replaced by the frozen local Qwen2.5-3B-Instruct model with its chat template and deterministic decoding.",
            "Google Custom Search is replaced by the same benchmark-specific offline indexes used by JARV.",
            "The original one-search-per-question flow is capped at three total calls with top-k=3 for the fixed comparison budget.",
            "The original spaCy sentence boundary pass is replaced by deterministic punctuation/newline parsing because the prompt itself mandates period-separated subclaims.",
            "Native four-way verdicts are separately retained; only Supported, Refuted, and mixed Supported/Refuted have deterministic binary projections. NEI stays invalid and reduces coverage.",
        ],
        "environment": {"python": sys.version, "platform": platform.platform()},
        "command": [sys.executable, *sys.argv],
        "source_file": str(Path(__file__).resolve()),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
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
    atomic_write_json(
        args.output_root / "sample_manifest.json",
        {name: {"count": len(rows), "ids": [str(row["id"]) for row in rows], "sha256": dataset_digest(rows)} for name, rows in datasets.items()},
    )

    reuse_manifest = [
        import_compatible_predictions(source_root, args.output_root, datasets, config, fingerprint)
        for source_root in args.reuse_root
    ]
    if reuse_manifest:
        atomic_write_json(args.output_root / "checkpoint_reuse_manifest.json", reuse_manifest)

    llm = LocalLLM(args.model, use_chat_template=True)
    retrievers = RetrieverBundle(args)
    try:
        total = sum(len(rows) for rows in datasets.values())
        done = failed = 0
        for benchmark in args.benchmarks:
            for raw in datasets[benchmark]:
                target = sample_path(args.output_root, METHOD, benchmark, str(raw["id"]))
                if target.exists():
                    try:
                        previous = json.loads(target.read_text(encoding="utf-8"))
                        deterministic_terminal = (
                            previous.get("prediction") in {"real", "fake"}
                            or previous.get("native_verdict") in {
                                "supported",
                                "refuted",
                                "Conflicting Evidence/Cherrypicking",
                                "not enough evidence",
                            }
                        )
                        if previous.get("config_fingerprint") == fingerprint and deterministic_terminal:
                            done += 1
                            continue
                    except Exception:
                        pass
                try:
                    result = run_one(raw, benchmark, llm, retrievers.for_example(benchmark, str(raw["id"])), args)
                except Exception as exc:
                    LOGGER.exception("Failure %s/%s", benchmark, raw["id"])
                    result = {"id": str(raw["id"]), "benchmark": benchmark, "method": METHOD, "claim": raw["claim"], "gold": raw["gold"], "prediction": None, "correct": False, "native_verdict": None, "search_calls": 0, "retrieval_rounds": 0, "subclaims": [], "qa_pairs": [], "generations": [], "input_tokens": 0, "output_tokens": 0, "parse_failures": 0, "latency_seconds": 0.0, "error": f"{type(exc).__name__}: {exc}"}
                result["config_fingerprint"] = fingerprint
                atomic_write_json(target, result)
                done += 1
                failed += int(bool(result.get("error")))
                LOGGER.info("%s %d/%d failed=%d remaining=%d id=%s searches=%s native=%s prediction=%s", benchmark, done, total, failed, total - done, raw["id"], result.get("search_calls"), result.get("native_verdict"), result.get("prediction"))
    finally:
        retrievers.close()
    build_outputs(args.output_root, datasets, config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
