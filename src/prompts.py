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
    limit: int = 0, # 0 means ``len(labels) + 1``
    split: str = 'train',
) -> str:
    """Build few-shot demonstrations with complete label coverage.

    Args:
        bench_cls: Benchmark definition.
        limit: Number of demonstrations. 0 means ``len(labels) + 1``.  It must
            be strictly greater than the number of label categories so that
            every label can appear at least once and one extra example can be
            added.
        split: Preferred split; benchmarks without a training split fall back
            to their dev split.

    Every label in ``bench_cls.choice_dict`` is represented at least once.
    Examples with non-empty gold evidence are preferred within a label, but
    empty-evidence examples are kept for labels such as NEI.
    """
    labels = list(bench_cls.choice_dict)
    required = len(labels) + 1
    if limit <= 0:
        limit = required
    elif limit <= len(labels):
        raise ValueError(
            f'fs_limit={limit} must be greater than the number of labels '
            f'({len(labels)}) to cover every label'
        )

    def _load(preferred_split: str) -> list[dict]:
        rows, _ = load_benchmark(
            benchmark_name=bench_cls.name,
            limit=-1,
            split=preferred_split,
        )
        return [
            row for row in rows
            if row.get('label') in labels
        ]

    pool = _load(split)
    missing = [label for label in labels if not any(
        row.get('label') == label for row in pool
    )]
    assert not missing, missing

    selected: list[dict] = []
    selected_positions: set[int] = set()
    for label in labels:
        candidates = [
            (position, row)
            for position, row in enumerate(pool)
            if row.get('label') == label
        ]
        if not candidates:
            raise ValueError(
                f'{bench_cls.name}: no example available for label {label!r}'
            )
        position, row = next(
            (
                candidate for candidate in candidates
                if (candidate[1].get('evidence') or '').strip()
            ),
            candidates[0],
        )
        selected_positions.add(position)
        selected.append({
            'claim': row['claim'],
            'evidence': row.get('evidence') or '',
            'label': row['label'],
        })

    for position, row in enumerate(pool):
        if len(selected) >= limit:
            break
        if position in selected_positions:
            continue
        selected_positions.add(position)
        selected.append({
            'claim': row['claim'],
            'evidence': row.get('evidence') or '',
            'label': row['label'],
        })

    if len(selected) < limit:
        raise ValueError(
            f'{bench_cls.name}: only {len(selected)} examples available, '
            f'requested {limit}'
        )

    return '\n'.join(format_fs_one(**item) for item in selected)


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
    