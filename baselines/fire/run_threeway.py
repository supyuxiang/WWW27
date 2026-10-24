#!/usr/bin/env python3
"""FIRE control flow adapted to CFEVER and QuanTemp++ native labels."""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT))
sys.path.insert(0, str(CODE_ROOT / "vendor" / "www27"))

from common.threeway_support import (  # noqa: E402
    atomic_json,
    config_payload,
    finalize_output,
    gold_label,
    initialize_output,
    load_benchmark,
    make_retriever,
    prepare_retriever,
)
from baselines.run_fire_offline import (  # noqa: E402
    FIRE_SYSTEM_PROMPT,
    SemanticRepeatDetector,
    extract_json,
    format_hits,
    normalized_query,
)
from src.llm import LocalLLM  # noqa: E402

METHOD = "fire_adapted"


def label_spec(benchmark: str) -> str:
    if benchmark == "cfever":
        return '"SUPPORTS", "REFUTES", or "NOT ENOUGH INFO"'
    return '"True", "False", or "Half True/False"'


def parse_answer(benchmark: str, value: object) -> str | None:
    key = re.sub(r"\s+", " ", str(value or "").strip().lower().replace("_", " "))
    if benchmark == "cfever":
        mapping = {
            "supports": "supports", "supported": "supports", "true": "supports",
            "refutes": "refutes", "refuted": "refutes", "false": "refutes",
            "not enough info": "not_enough_info", "not enough evidence": "not_enough_info",
            "nei": "not_enough_info",
        }
    else:
        mapping = {
            "true": "true", "supported": "true", "supports": "true",
            "false": "false", "refuted": "false", "refutes": "false",
            "half true/false": "conflicting", "half true half false": "conflicting",
            "conflicting": "conflicting", "conflicting evidence": "conflicting",
        }
    return mapping.get(key)


def prompt(claim: str, knowledge: str, benchmark: str, forced: bool) -> str:
    final_rule = (
        f'Output a JSON object exactly like {{"final_answer": {label_spec(benchmark)}}}.'
    )
    if forced:
        action = f"You must make the final three-way decision now. {final_rule}"
    else:
        action = (
            f"If the knowledge permits a decision, {final_rule} "
            'Otherwise output exactly one useful new query as {"search_query": "..."}.'
        )
    return f"""Instructions:
1. You are provided with a STATEMENT and relevant KNOWLEDGE points.
2. Assess the factual accuracy of the STATEMENT from the KNOWLEDGE.
3. Think step-by-step and summarize the key evidence before the JSON object.
4. {action}
5. A search query must seek new information not already in the KNOWLEDGE.

KNOWLEDGE:
{knowledge or 'N/A'}

STATEMENT:
{claim}"""


def knowledge_text(searches: list[dict[str, Any]], maximum: int) -> str:
    text = "\n\n".join(item["result_text"] for item in searches).strip()
    return text[-maximum:] if text else "N/A"


def run_one(row, benchmark, module, llm, retriever, repeat_detector, args):
    started = time.perf_counter()
    searches: list[dict[str, Any]] = []
    generations: list[dict[str, Any]] = []
    seen_queries: set[str] = set()
    prediction = None
    stop_reason = ""
    parse_failures = input_tokens = output_tokens = 0
    prepare_retriever(module, retriever, row)

    for step in range(1, args.max_steps + 1):
        generated = llm.generate(
            FIRE_SYSTEM_PROMPT,
            prompt(row["claim"], knowledge_text(searches, args.max_evidence_chars), benchmark, False),
            args.max_new_tokens,
        )
        input_tokens += generated.usage.input_tokens
        output_tokens += generated.usage.output_tokens
        parsed = extract_json(generated.text)
        generations.append({"step": step, "forced_final": False, "raw": generated.text, "parsed": parsed})
        if parsed:
            prediction = parse_answer(benchmark, parsed.get("final_answer"))
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
        history = [item["query"] for item in searches]
        window = max(1, args.repeat_window - 1)
        repeated = repeat_detector.repeated(query, history[-window:]) if repeat_detector else normalized in seen_queries
        if repeated:
            stop_reason = "semantic_repeat_query"
            break
        if len(searches) >= args.search_cap:
            stop_reason = "search_budget_exhausted"
            break
        seen_queries.add(normalized)
        retrieval_started = time.perf_counter()
        hits = retriever.search(query, args.top_k, set(), args.max_passage_chars)
        searches.append({
            "query": query,
            "latency_s": time.perf_counter() - retrieval_started,
            "result_text": format_hits(hits),
            "hits": [{"doc_id": hit.doc_id, "title": hit.title, "score": hit.score, "text": hit.text} for hit in hits],
        })

    if prediction is None:
        for retry in range(args.final_retries + 1):
            generated = llm.generate(
                FIRE_SYSTEM_PROMPT,
                prompt(row["claim"], knowledge_text(searches, args.max_evidence_chars), benchmark, True),
                args.max_new_tokens,
            )
            input_tokens += generated.usage.input_tokens
            output_tokens += generated.usage.output_tokens
            parsed = extract_json(generated.text)
            generations.append({"step": len(generations) + 1, "forced_final": True, "retry": retry, "raw": generated.text, "parsed": parsed})
            prediction = parse_answer(benchmark, (parsed or {}).get("final_answer"))
            if prediction:
                stop_reason = stop_reason or "forced_final"
                break
            parse_failures += 1

    return {
        "id": str(row["id"]), "claim": row["claim"], "gold": gold_label(module, row),
        "prediction": prediction, "method": METHOD, "correct": prediction == gold_label(module, row),
        "searches": searches, "generations": generations, "search_calls": len(searches),
        "input_tokens": input_tokens, "output_tokens": output_tokens,
        "parse_failures": parse_failures, "stop_reason": stop_reason or "invalid_output",
        "latency_s": time.perf_counter() - started,
        "error": None if prediction else "FIRE produced no valid native three-way label",
        **({"split_index": row["__split_index"]} if "__split_index" in row else {}),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", choices=["cfever", "quantemp_plus"], required=True)
    parser.add_argument("--split-file", required=True)
    parser.add_argument("--index", required=True)
    parser.add_argument("--model", default="/root/autodl-tmp/models/Qwen2.5-3B-Instruct")
    parser.add_argument("--repeat-model", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--search-cap", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--max-passage-chars", type=int, default=800)
    parser.add_argument("--max-evidence-chars", type=int, default=6000)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--final-retries", type=int, default=10)
    parser.add_argument("--repeat-threshold", type=float, default=0.9)
    parser.add_argument("--repeat-window", type=int, default=2)
    args = parser.parse_args()

    module, selected = load_benchmark(args.benchmark, args.split_file, args.limit, args.seed)
    source = Path(__file__)
    config = config_payload(args, module, selected, METHOD, source, {
        "official_method": "FIRE iterative retrieval and verification",
        "official_code": "https://github.com/mbzuai-nlp/fire",
        "adaptation": "Serper is replaced by the fixed offline index; final vocabulary is the benchmark native three-way space",
        "search_cap": args.search_cap, "official_max_steps": args.max_steps, "top_k": args.top_k,
        "max_new_tokens": args.max_new_tokens, "final_retries": args.final_retries,
        "repeat_model": args.repeat_model, "repeat_threshold": args.repeat_threshold,
        "repeat_window": args.repeat_window,
    })
    output = Path(args.output_root)
    initialize_output(output, config, selected)
    llm = LocalLLM(args.model, use_chat_template=True)
    repeat_detector = SemanticRepeatDetector(args.repeat_model, args.repeat_threshold)
    retriever = make_retriever(module, args.index)
    rows = []
    for position, row in enumerate(selected, 1):
        destination = output / "predictions" / METHOD / f"{row['id']}.json"
        if destination.exists():
            previous = json.loads(destination.read_text(encoding="utf-8"))
            if previous.get("config_fingerprint") == config["config_fingerprint"] and not previous.get("error"):
                rows.append(previous)
                continue
        try:
            result = run_one(row, args.benchmark, module, llm, retriever, repeat_detector, args)
        except Exception as exc:
            result = {"id": str(row["id"]), "claim": row["claim"], "gold": gold_label(module, row), "prediction": None, "method": METHOD, "error": f"{type(exc).__name__}: {exc}"}
        result["config_fingerprint"] = config["config_fingerprint"]
        atomic_json(destination, result)
        rows.append(result)
        print(json.dumps({"method": METHOD, "benchmark": args.benchmark, "processed": position, "total": len(selected), "failed": sum(bool(item.get('error')) for item in rows)}), flush=True)
    retriever.close()
    summary = finalize_output(output, config, rows, module.LABELS)
    print(json.dumps({"event": "complete", "summary": summary}, ensure_ascii=False), flush=True)
    return int(bool(summary["failures"]))


if __name__ == "__main__":
    raise SystemExit(main())
