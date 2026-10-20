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
    user_prompt = f'Claim: {claim}.'
    if current_evidence is not None:
        user_prompt += f'\nCurrent evidence:{current_evidence}'
    user_prompt += (
        "\nDecide whether more external evidence is needed to assess the central factual content, then select one:\n "
    )
    for i, (k,v) in enumerate(WHETHER_RETRIEVAL.items()):
        user_prompt += f'\n[{i+1}] {k} (description: {v}).'
    options = [f'{i+1}' for i in range(len(WHETHER_RETRIEVAL))]
    return system_prompt, user_prompt, options


def prompt4what_aspect(
    claim: str, current_evidence: str | None=None
) -> tuple[str, str, list[str]]:
    system_prompt=SYSTEM_PROMPT
    options = []
    user_prompt = f'Claim: {claim}.'
    if current_evidence is not None:
        user_prompt += f'\nCurrent evidence: {current_evidence}.'
    user_prompt += ('\nWhich evidence of aspect should be retrieval:'
    )
    for i, (name, definition) in enumerate(WHAT_ASPECT.items()):
        user_prompt += f'\n[{i+1}] {name} (definition:{definition}).'
        options.append(f'{i+1}')
    return system_prompt, user_prompt, options


def prompt4what_choice(
    benchmark: Benchmark,
    claim: str,
    current_evidence: str | None = None
) -> tuple[str, str, list[str]]:
    system_prompt = SYSTEM_PROMPT
    options = [f'{i+1}' for i in range(len(benchmark.choice_dict))]
    user_prompt = f'Claim: {claim}.'
    if current_evidence is not None:
        user_prompt += f'\nCurrent evidence: {current_evidence}.'
    user_prompt += '\nWhich choice is correct:'
    for i, (k,v) in enumerate(benchmark.choice_dict.items()):
        user_prompt += f'\n[{i+1}] {k} (description: {v}).'
    return system_prompt, user_prompt, options


def construct_text4rerankng(
    retrieval_corpus: list[str],
    selected_aspect: str,
    top_k:int=3,
) -> list[str]:
    pass



# 
def jev_retrieval():
    pass


def main():
    pass


if __name__ == '__main__':
    main()
    