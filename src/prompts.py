from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0,str(ROOT))

from src.aspects import Aspect, ASPECTS
from src.utils import BENCHMARKS, Benchmark

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



def prompt4whether_retrieval(
    claim:str, current_evidence:str | None=None
) -> tuple[str, str, list[str]]: # system_prompt, user_prompt, options
    system_prompt = SYSTEM_PROMPT
    user_prompt_ls = [f'Claim: {claim}']
    if current_evidence is not None:
        user_prompt_ls.append(f'Current evidence: {current_evidence}')
    user_prompt_ls.append(
        'Decide whether more external evidence is needed to assess the central factual content, then select one:'
    )
    for i, (k,v) in enumerate(WHETHER_RETRIEVAL.items()):
        user_prompt_ls.append(
            f'[{i+1}] {k} (description: {v}).'
        )
    user_prompt = '\n'.join(user_prompt_ls)
    options = [f'{i+1}' for i in range(len(WHETHER_RETRIEVAL))]
    return system_prompt, user_prompt, options


def prompt4what_aspect(
    claim: str, current_evidence: str | None=None
) -> tuple[str, str, list[str]]:
    system_prompt=SYSTEM_PROMPT
    options = []
    user_prompt_ls = [f'Claim: {claim}']
    if current_evidence is not None:
        user_prompt_ls.append(f'Current evidence: {current_evidence}')
    user_prompt_ls.append('Which evidence of aspect should be retrieval:')
    for i, (name, definition) in enumerate(WHAT_ASPECT.items()):
        user_prompt_ls.append(
            f'[{i+1}] {name} (definition:{definition}).'
        )
        options.append(f'{i+1}')
    user_prompt = '\n'.join(user_prompt_ls)
    return system_prompt, user_prompt, options


def prompt4aspect_check( # 第二次开始筛选aspect用这个prompt
    claim: str,
    aspect: Aspect,
    current_evidence: str | None = None,
) -> tuple[str, str, list[str]]:
    system_prompt = SYSTEM_PROMPT
    user_prompt = f'Claim: {claim}.'
    if current_evidence is not None:
        user_prompt += f'\nCurrent evidence: {current_evidence}.'
    user_prompt += (
        f'\nCandidate aspect: {aspect.to_str()}'
        '\nDoes assessing this uncovered aspect require additional external retrieval?'
        '\n[1] check (description: Check this aspect using external evidence).'
        '\n[2] skip (description: Skip investigation of this aspect).'
    )
    options = ['1', '2']
    return system_prompt, user_prompt, options


def prompt4generate_choice(benchmark: Benchmark, claim: str) -> str:
    labels = list(benchmark.choice_dict)
    lines = [
        f'Claim: {claim}',
        'Decide which verdict applies to the claim.',
    ]
    for label, description in benchmark.choice_dict.items():
        lines.append(f'- {label}: {description}')
    lines.append(
        'Put your final answer in \\boxed{VERDICT}, '
        'where VERDICT is exactly one of: ' + ', '.join(labels) + '.'
    )
    return '\n'.join(lines)


def prompt4what_choice(
    benchmark: Benchmark,
    claim: str,
    current_evidence: str | None = None
) -> tuple[str, str, list[str]]:
    system_prompt = SYSTEM_PROMPT
    options = [f'{i+1}' for i in range(len(benchmark.choice_dict))]
    user_prompt_ls=[f'Claim: {claim}']
    if current_evidence is not None:
        user_prompt_ls.append(f'Current evidence: {current_evidence}')
    user_prompt_ls.append('Which choice is correct:')
    for i, (k,v) in enumerate(benchmark.choice_dict.items()):
        user_prompt_ls.append(
            f'[{i+1}] {k} (description: {v}).'
        )
    user_prompt = '\n'.join(user_prompt_ls)
    return system_prompt, user_prompt, options


def construct_text4rerankng(claim: str, aspect: Aspect) -> str:
    """Rerank query. BM25 itself searches with the raw claim."""
    return (
        f"Aspect: {aspect.name}\n"
        f"Definition: {aspect.definition}\n"
        f"Claim: {claim}"
    )





def main():
    pass


if __name__ == '__main__':
    main()
    