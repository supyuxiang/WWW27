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
    SYSTEM_PROMPT,
    prompt4llm_judge,
    build_few_shot,
)
from src.utils import (
    Benchmark,
    compute_accuracy,
    compute_f1,
    load_benchmark,
    load_tokenizer,
)

os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
from vllm import LLM, SamplingParams

_LATEX_WRAPPER_RE = re.compile(r"\\(?:text|mathrm|textbf)\s*\{([^{}]*)\}")


def _boxed_values(text: str) -> list[str]:
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
        tensor_parallel_size: int = torch.cuda.device_count(),
        gpu_memory_utilization: float = 0.92,
        max_model_len: int = 8192,
        max_new_tokens: int = 1024,
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
        self.llm_name = str(model_path).split("/")[-1]
        self.tokenizer = load_tokenizer(model_path, is_training=False)
        self.sp = SamplingParams(
            temperature=0.0,
            max_tokens=max_new_tokens,
            skip_special_tokens=True,
        )

    def _format_one(self, user_prompt: str) -> str:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=self.enable_thinking,
        )

    def generate_batch(self, prompts: list[str]) -> list[str]:
        formatted = [self._format_one(prompt) for prompt in prompts]
        outputs = self.llm.generate(formatted, self.sp)
        return [output.outputs[0].text.strip() for output in outputs]


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
    parser.add_argument("--fs-limit", type=int, default=2)
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
