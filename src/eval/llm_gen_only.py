from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.prompts import SYSTEM_PROMPT
from src.utils import (
    Benchmark,
    compute_accuracy,
    compute_f1,
    load_benchmark,
    load_tokenizer
)
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
from vllm import LLM, SamplingParams

_LATEX_WRAPPER_RE = re.compile(r"\\(?:text|mathrm|textbf)\s*\{([^{}]*)\}")


def _boxed_values(text: str) -> list[str]:
    """Extract balanced \\boxed{...} contents, including nested \\text{...}."""
    values: list[str] = []
    cursor = 0
    while True:
        marker = text.find("\\boxed", cursor)
        if marker < 0:
            break
        opening = text.find("{", marker)
        if opening < 0:
            break
        depth = 0
        for index in range(opening, len(text)):
            char = text[index]
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    values.append(text[opening + 1 : index])
                    cursor = index + 1
                    break
        else:
            break
    return values


def build_prompt(bench: Benchmark, claim: str) -> str:
    labels = list(bench.choice_dict)
    lines = []
    lines.append(f"Claim: {claim}")
    lines.append("Decide which verdict applies to the claim.")
    for label, description in bench.choice_dict.items():
        lines.append(f"- {label}: {description}")
    lines.append(
        "Put your final answer in \\boxed{VERDICT}, "
        "where VERDICT is exactly one of: " + ", ".join(labels) + "."
    )
    return "\n".join(lines)


def _match_label(value: str, labels: list[str]) -> str | None:
    folded = " ".join(value.split())
    exact = {label.casefold(): label for label in labels}
    if folded.casefold() in exact:
        return exact[folded.casefold()]
    hits: list[tuple[int, int, str]] = []
    for label in labels:
        pattern = re.compile(rf"(?<!\w){re.escape(label)}(?!\w)", re.IGNORECASE)
        for match in pattern.finditer(folded):
            hits.append((match.start(), len(label), label))
    if not hits:
        return None
    hits.sort(key=lambda item: (item[0], item[1]))
    return hits[-1][2]


def parse_generation(text: str, labels: list[str]) -> str | None:
    """Read the final \\boxed{...} verdict.

    Only a boxed final answer is accepted; free-form reasoning frequently
    mentions several labels and must not be mistaken for the conclusion.
    """
    boxed = _boxed_values(text or "")
    if not boxed:
        return None
    value = boxed[-1]
    for _ in range(2):
        value = _LATEX_WRAPPER_RE.sub(r"\1", value)
    return _match_label(value, labels)


class VLLMGenerator:
    def __init__(
        self,
        model_path: str,
        tensor_parallel_size: int,
        gpu_memory_utilization: float,
        max_model_len: int,
        enable_thinking: bool = False,
    ):
        self.enable_thinking = enable_thinking
        self.llm = LLM(
            model=model_path,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            dtype="bfloat16",
        )
        self.tokenizer = load_tokenizer(model_path)

    def _format_one(self, user: str) -> str:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ]
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=self.enable_thinking,
            )

    def generate_batch(self, prompts: list[str], max_new_tokens: int) -> list[str]:
        formatted = [
            self._format_one(prompt) for prompt in prompts
        ]
        sp = SamplingParams(
            temperature=0.0,
            max_tokens=max_new_tokens,
            skip_special_tokens=True,
        )
        outputs = self.llm.generate(formatted, sp)
        return [
            output.outputs[0].text.strip() for output in outputs
        ]


def run_one(
    name: str,
    examples: list[dict],
    bench: Benchmark,
    model_path: str,
    output_dir: Path,
    generator: VLLMGenerator,
    batch_size: int,
    max_new_tokens: int,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"llm_gen_only_{name}.jsonl"
    started = time.perf_counter()
    labels = list(bench.choice_dict)
    gold: list[str] = []
    pred: list[str] = []
    invalid = 0
    latencies: list[float] = []
    with output_path.open("w", encoding="utf-8") as handle:
        for batch_start in range(0, len(examples), batch_size):
            batch = examples[batch_start : batch_start + batch_size]
            prompts = [build_prompt(bench, item["claim"]) for item in batch]
            batch_started = time.perf_counter()
            texts = generator.generate_batch(prompts, max_new_tokens)
            
            batch_elapsed = time.perf_counter() - batch_started
            per_item_latency = batch_elapsed / len(batch)

            for offset, (example, text) in enumerate(zip(batch, texts)):
                index = batch_start + offset + 1
                prediction = parse_generation(text, labels)
                if prediction is None:
                    prediction = 'invalid'
                    invalid +=1
                gold.append(example["label"])
                pred.append(prediction)
                latencies.append(per_item_latency)
                handle.write(
                    json.dumps(
                        {
                            "benchmark": name,
                            "id": example.get("id"),
                            "claim": example["claim"],
                            "gold": example["label"],
                            "prediction": prediction,
                            "generation": text or None,
                            "latency_s": per_item_latency,
                            "retrieved": False,
                            "search_calls": 0,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                handle.flush()
                if index % 25 == 0 or index == len(examples):
                    print(f"{name}: {index}/{len(examples)}", flush=True)
    elapsed = time.perf_counter() - started
    summary = {
        "benchmark": name,
        "model": model_path,
        "retrieval": "none",
        "backend": "vllm",
        "decoding": (
            f"greedy, max_new_tokens={max_new_tokens}, "
            f"submit_batch_size={batch_size}, vLLM continuous batching"
        ),
        "n": len(gold),
        "invalid": invalid,
        "accuracy": compute_accuracy(gold, pred),
        "f1": compute_f1(gold, pred, bench),
        "gold_counts": dict(Counter(gold)),
        "prediction_counts": dict(Counter(pred)),
        "elapsed_s": elapsed,
        "avg_latency_s": sum(latencies) / len(latencies) if latencies else 0.0,
        "output": str(output_path),
    }
    (output_dir / f"llm_gen_only_{name}.summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="No-retrieval generation baseline")
    parser.add_argument("--benchmarks", nargs="+", default=["averitec", "fever", "scifact"])
    parser.add_argument("--model", default="/root/autodl-tmp/models/Qwen3-8B")
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", default="/root/www27/output")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="request-submission chunk size; 0 submits all prompts in one vLLM generate() call",
    )
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.92)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument(
        "--enable-thinking",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable Qwen3 thinking mode in the chat template (default: on)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    generator = VLLMGenerator(
        model_path=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        enable_thinking=args.enable_thinking,
    )
    summaries = {}
    for name in args.benchmarks:
        examples, bench = load_benchmark(name, args.limit)
        print(f"{name}: {len(examples)} claims", flush=True)
        summaries[name] = run_one(
            name,
            examples,
            bench,
            args.model,
            output_dir,
            generator,
            args.batch_size,
            args.max_new_tokens,
        )
    (output_dir / "llm_gen_only_summary.json").write_text(
        json.dumps(summaries, indent=2), encoding="utf-8"
    )
    lines = [
        "# No-retrieval LLM-Gen baseline",
        "",
        "| Benchmark | n | Accuracy | F1 | Invalid |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, summary in summaries.items():
        lines.append(
            f"| {name} | {summary['n']} | {summary['accuracy']:.4f} | "
            f"{summary['f1']:.4f} | {summary['invalid']} |"
        )
    (output_dir / "llm_gen_only_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("wrote", output_dir / "llm_gen_only_summary.json", flush=True)


if __name__ == "__main__":
    main()
