from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.prompts import (
    prompt4llm_judge,
    build_few_shot,
)
from src.utils import (
    Benchmark,
    compute_accuracy,
    compute_f1,
    load_benchmark,
)
from src.worker import VLLMGenerator


MATH_VERIFY_ROOT = ROOT / "third_party" / "Math-Verify"
if (MATH_VERIFY_ROOT / "src").is_dir():
    sys.path.insert(0, str(MATH_VERIFY_ROOT / "src"))

from math_verify import parse as math_verify_parse
from math_verify.parser import LatexExtractionConfig

_LATEX_WRAPPER_RE = re.compile(
    r"\\(?:text|textnormal|mathrm|textbf|operatorname|mbox)\s*\{([^{}]*)\}"
)
_BOXED_EXTRACTION = [LatexExtractionConfig(boxed_match_priority=0)]
_SUPPORT_ALIASES = {"support", "supports", "supported", "s"}
_REFUTE_ALIASES = {
    "refute",
    "refutes",
    "refuted",
    "contradict",
    "contradicts",
    "contradicted",
    "r",
}
_NEI_ALIASES = {
    "nei",
    "not enough info",
    "not enough information",
    "not enough evidence",
    "insufficient evidence",
    "insufficient information",
}


def _normalize_label_text(value: str) -> str:
    value = str(value).strip()
    for _ in range(3):
        value = _LATEX_WRAPPER_RE.sub(r"\1", value)
    value = value.strip().strip("{}").strip()
    value = value.strip("$").strip()
    return " ".join(value.split())


def _semantic_key(value: str) -> str:
    key = _normalize_label_text(value).casefold()
    key = re.sub(r"[^a-z0-9]+", " ", key).strip()
    if key in _SUPPORT_ALIASES:
        return "SUPPORTS"
    if key in _REFUTE_ALIASES:
        return "REFUTES"
    if key in _NEI_ALIASES:
        return "NOT ENOUGH INFO"
    return key


def _extract_answer_candidates(text: str) -> list[str]:
    if not text:
        return []
    try:
        parsed = math_verify_parse(
            text,
            extraction_config=_BOXED_EXTRACTION,
            # first_match also exposes the raw extracted text (needed for
            # textual labels); no_fallback only returns the SymPy object.
            fallback_mode="first_match",
        )
    except Exception:
        return []
    raw_strings = [item for item in parsed if isinstance(item, str)]
    if raw_strings:
        return raw_strings
    return [str(item) for item in parsed]


def _match_label(value: str, labels: list[str]) -> str | None:
    normalized = _normalize_label_text(value)
    folded = normalized.casefold()

    exact = {label.casefold(): label for label in labels}
    if folded in exact:
        return exact[folded]

    semantic = {_semantic_key(label): label for label in labels}
    semantic_key = _semantic_key(normalized)
    if semantic_key in semantic:
        return semantic[semantic_key]

    hits: list[tuple[int, int, str]] = []
    for label in labels:
        pattern = re.compile(rf"(?<!\w){re.escape(label)}(?!\w)", re.IGNORECASE)
        for match in pattern.finditer(normalized):
            hits.append((match.start(), len(label), label))
    if not hits:
        return None
    hits.sort(key=lambda item: (item[0], item[1]))
    return hits[-1][2]


def parse_generation(text: str, labels: list[str]) -> str | None:
    candidates = _extract_answer_candidates(text)
    for value in reversed(candidates):
        matched = _match_label(value, labels)
        if matched is not None:
            return matched
    return None


def run_one(
    name: str,
    examples: list[dict],
    bench: Benchmark,
    generator: VLLMGenerator,
    few_shot: str,
    output_dir: Path,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    labels = list(bench.choice_dict)
    filtered = [
        item for item in examples
        if (item.get("evidence") or "").strip()
    ]
    prompts = [
        prompt4llm_judge(
            benchmark_cls=bench,
            claim=item["claim"],
            few_shot=few_shot,
            evidence=item["evidence"],
        )
        for item in filtered
    ]
    user_prompts = [prompt[1] for prompt in prompts]
    generations = generator.generate_batch(user_prompts)

    gold: list[str] = []
    pred: list[str] = []
    invalid = 0
    trajectory_path = output_dir / f"llm_gen_only_{name}.jsonl"
    with trajectory_path.open("w", encoding="utf-8") as handle:
        for item, generation in zip(filtered, generations):
            prediction = parse_generation(generation, labels)
            if prediction is None:
                prediction = "invalid"
                invalid += 1
            gold.append(item["label"])
            pred.append(prediction)
            handle.write(json.dumps({
                "benchmark": name,
                "id": item.get("id"),
                "claim": item["claim"],
                "gold": item["label"],
                "evidence": item["evidence"],
                "generation": generation,
                "prediction": prediction,
            }, ensure_ascii=False) + "\n")

    summary = {
        "benchmark": name,
        "n": len(gold),
        "invalid": invalid,
        "accuracy": compute_accuracy(gold, pred),
        "f1": compute_f1(gold, pred, bench),
        "gold_counts": dict(Counter(gold)),
        "prediction_counts": dict(Counter(pred)),
        "few_shot": few_shot,
        "output": str(trajectory_path),
    }
    (output_dir / f"llm_gen_only_{name}.summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM generation with gold evidence")
    parser.add_argument("--benchmarks", nargs="+", default=["averitec", "fever", "scifact"])
    parser.add_argument("--model", default="/root/autodl-tmp/models/Qwen3-8B")
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--output-dir", default="/root/autodl-tmp/www27/llm_gen_only")
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.92)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--fs-limit", type=int, default=0)
    parser.add_argument(
        "--enable-thinking",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    generator = VLLMGenerator(
        model_path=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_new_tokens=args.max_new_tokens,
        enable_thinking=args.enable_thinking,
    )

    summaries: dict[str, dict] = {}
    for name in args.benchmarks:
        examples, bench = load_benchmark(name, limit=args.limit)
        print(f"{name}: {len(examples)} claims", flush=True)
        few_shot = build_few_shot(
            bench_cls=bench,
            limit=args.fs_limit,
            split="train",
        )
        summaries[name] = run_one(
            name=name,
            examples=examples,
            bench=bench,
            generator=generator,
            few_shot=few_shot,
            output_dir=output_dir,
        )

    (output_dir / "llm_gen_only_summary.json").write_text(
        json.dumps(summaries, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    lines = [
        "# LLM generation with gold evidence",
        "",
        "| Benchmark | n | Accuracy | F1 | Invalid |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, summary in summaries.items():
        lines.append(
            f"| {name} | {summary['n']} | {summary['accuracy']:.4f} | "
            f"{summary['f1']:.4f} | {summary['invalid']} |"
        )
    (output_dir / "llm_gen_only_summary.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    print("wrote", output_dir, flush=True)


if __name__ == "__main__":
    main()
