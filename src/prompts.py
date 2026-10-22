from __future__ import annotations

import sys
from pathlib import Path
from transformers import AutoTokenizer

ROOT = Path(__file__).parent.parent
sys.path.insert(0,str(ROOT))

from src.aspects import Aspect, ASPECTS
from src.utils import (
    BENCHMARKS,
    Benchmark,
    load_benchmark
)


###############    prompts    #####################
SYSTEM_PROMPT = (
    'You are a helpful assistant for factual verification.'
)

ASSISTANT_PREFILL = "Best answer: ["

WHETHER_RETRIEVAL = {
    'retrieval': 'Retrieve external evidence.',
    'skip':'Skip external retireval.'
}
WHAT_ASPECT = {
    item.name: item.definition for item in ASPECTS
}
WHICH_CHOICE = {
    bench.name: bench.choice_dict for bench in BENCHMARKS
}

#######################    RQ1     ###########################

# format single few-shot example
def format_fs_one(
    claim: str,
    evidence: str,
    label: str,
):
    evidence_text = (evidence or "").strip()
    lines = [
        f'Claim: {claim}',
        'Decide which verdict applies to the claim.',
        f'Current evidence:\n{evidence_text}',
        f'Final answer: \\boxed{{{label}}}',
    ]
    return '\n'.join(lines)


def build_few_shot(
    bench_cls: Benchmark,
    limit: int = 3,
    split: str = 'train',
) -> str:
    """Build few-shot demonstrations from a normalized claim split.

    Only examples with non-empty gold evidence are used.  Benchmarks without
    a training split fall back to their dev split.
    """
    try:
        full_data, _ = load_benchmark(
            benchmark_name=bench_cls.name,
            limit=-1,
            split=split,
        )
    except FileNotFoundError:
        full_data, _ = load_benchmark(
            benchmark_name=bench_cls.name,
            limit=-1,
            split='dev',
        )

    selected_data = []
    for data in full_data:
        evidence = (data.get('evidence') or '').strip()
        if not evidence:
            continue
        selected_data.append({
            'claim': data['claim'],
            'evidence': evidence,
            'label': data['label'],
        })
        if len(selected_data) >= limit:
            break

    return '\n'.join(format_fs_one(**item) for item in selected_data)


# for llm-gen
def prompt4llm_judge(
    benchmark_cls: Benchmark,
    claim: str,
    few_shot: str | None = None,
    evidence: str | None = None,
) -> tuple[str, str, list[str]]:
    system_prompt = SYSTEM_PROMPT
    options = [f'{i+1}' for i in range(len(benchmark_cls.choice_dict))]

    labels = list(benchmark_cls.choice_dict)
    lines=[f'Claim: {claim}']
    lines = [
        f'Claim: {claim}',
        'Decide which verdict applies to the claim.',
    ]

    for i, (k,v) in enumerate(benchmark_cls.choice_dict.items()):
        lines.append(
            f'[{i+1}] {k} (description: {v}).'
        )
    if evidence is not None and isinstance(evidence, str):
        lines.append(f'Current evidence:\n{evidence}')

    if few_shot is not None and isinstance(few_shot, str):
        lines.append(
            f'-'*50 +
            f'\nFor example:\n{few_shot}\n' +
            f'-'*50
        )

    lines.append('\n')
    lines.append(
        'Put your final answer in \\boxed{VERDICT}, '
        'where VERDICT is exactly one of: ' + ', '.join(labels) + '.'
    )
    user_prompt = '\n'.join(lines)
    return system_prompt, user_prompt, options



# for llm_as_jeev
def prompt4jev_judge(
    benchmark_cls: Benchmark,
    claim: str,
    few_shot: str | None = None,
    evidence: str | None = None
) -> tuple[str, str, list[str]]:
    system_prompt = SYSTEM_PROMPT
    options = [f'{i+1}' for i in range(len(benchmark_cls.choice_dict))]

    lines = [
        f'Claim: {claim}',
        'Decide which verdict applies to the claim.',
    ]
    for i, (k,v) in enumerate(benchmark_cls.choice_dict.items()):
        lines.append(
            f'[{i+1}] {k} (description: {v}).'
        )
    if evidence is not None and isinstance(evidence, str):
        lines.append(f'Current Evidence: {evidence}')
    
    if few_shot is not None and isinstance(few_shot, str):
        lines.append(
            f'-'*50 +
            f'\nFor example:\n{few_shot}\n' +
            f'-'*50
        )
    user_prompt = '\n'.join(lines)
    return system_prompt, user_prompt, options


# for llm_as_jev
def format_prefix4jev(
    tokenizer: AutoTokenizer,
    system_prompt:str,
    user_prompt:str,
    prefill: str | None = None,
    enable_thinking: bool = False
) -> str:
    msg = [
        {'role':'system','content':system_prompt},
        {'role':'user', 'content':user_prompt},
    ]
    kwargs = dict(
        tokenize=False,
        add_generation_prompt=True,
    )

    # 我们项目不用这个
    if enable_thinking:
        kwargs['enable_thinking'] = True

    formatted = tokenizer.apply_chat_template(
        msg,
        **kwargs,
    )
    if prefill:
        formatted += prefill
    return formatted


############################################################


def prompt4jev_whether_retrieval(
    claim: str,
    evidence: str | None
) -> tuple[str, str, list[str]]: # system_prompt, user_prompt, options
    system_prompt = SYSTEM_PROMPT
    options = [f'{i+1}' for i in range(len(WHETHER_RETRIEVAL))]

    lines = [f'Claim: {claim}']
    if evidence is not None:
        lines.append(f'Current evidence: {evidence}')
    lines.append(
        'Determine whether additional external evidence is required to verify the claim, then select one option:'
    )
    for i, (k,v) in enumerate(WHETHER_RETRIEVAL.items()):
        lines.append(
            f'[{i+1}] {k} (description: {v}).'
        )
    user_prompt = '\n'.join(lines)
    return system_prompt, user_prompt, options


def prompt4jev_what_aspect(
    claim: str,
    evidence: str | None = None
) -> tuple[str, str, list[str]]:
    system_prompt=SYSTEM_PROMPT
    options = [f'{i+1}' for i in range(len(WHAT_ASPECT.items()))]
    
    lines = [f'Claim: {claim}']
    if evidence is not None:
        lines.append(f'Current evidence: {evidence}')

    lines.append('Which aspect of evidence requires further retrieval? Choose one option:')
    for i, (name, definition) in enumerate(WHAT_ASPECT.items()):
        lines.append(
            f'[{i+1}] {name} (definition: {definition}).'
        )
    user_prompt = '\n'.join(lines)
    return system_prompt, user_prompt, options


def construct_text4rerankng(claim: str, aspect: Aspect) -> str:
    """Rerank query."""
    return (
        f"Aspect: {aspect.name}\n"
        f"Definition: {aspect.definition}\n"
        f"Claim: {claim}"
    )



def main():
    pass



if __name__ == '__main__':
    main()
    