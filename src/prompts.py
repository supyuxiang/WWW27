from __future__ import annotations

import sys
from pathlib import Path

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
RERANK_INSTRUCTION =  (
    "Retrieve passages that contain evidence relevant to verifying the claim."
)
#######################    RQ1     ###########################

# format single few-shot example
def format_fs_one(
    claim: str,
    evidence: str,
    label: str,
) -> str:
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
    limit: int = 0, # 0 means ``len(labels) `
    split: str = 'train',
) -> str:
    '''
    构建few_shot prompt, 本质是由一系列(claim, evidence, label)三元组拼接而成的prompt,
    其中，我们刻意要求evidence非空(若某个label的NEI数据的evidence为空,则随机用一个存在的evidence当作这个数据的evidence)
    这么做的原因是要对齐推理过程————总是要做检索(on/offline)并得到evidence(或相关或不相关)。
    同时,我们保证每个label被采样的fs_example数目是相同的(通过 num_per_label)
    '''
    labels = list(bench_cls.choice_dict)
    required = len(labels)
    if limit <= 0:
        limit = required
    elif limit < len(labels):
        raise ValueError(
            f'fs_limit={limit} must be greater than the number of labels '
            f'({len(labels)}) to cover every label'
        )

    num_per_label = limit // required

    # load benchmark
    def _load(preferred_split: str) -> list[dict]:
        rows, _ = load_benchmark(
            benchmark_name=bench_cls.name,
            limit=-1,
            split=preferred_split,
        )
        return [row for row in rows if row['label'] in labels]

    pool = _load(split)

    # checking missing
    missing = [label for label in labels if not any(
        row.get('label') == label for row in pool
    )]
    assert not missing, missing

    # 建立 label -> valid fs_examples的mapping
    mapping = dict()
    for label in labels:
        candidates: list[tuple[int,dict]] = [
            (position, row) for position, row in enumerate(pool)
            if row.get('label') == label
        ]
        assert candidates, f'label:{label},candidates:{candidates}'
        mapping[label] = candidates


    selected: list[dict] = []
    selected_positions: set[int] = set()

    # prepare evidence for NEI (if NEI evidence is {})
    evidence_for_nei: list[str] = []
    valid_label = [
        label for label in labels if mapping[label][0][1]['evidence']
    ][0]
    for _, row in mapping[valid_label][num_per_label:]:
        evidence_for_nei.append(row['evidence'])
        if len(evidence_for_nei) == num_per_label:
            break

    # 正式抽所有label的fs数据
    for i in range(num_per_label):
        for label in labels:
            position, row = mapping[label][i]
            selected_positions.add(position)
            evidence = row['evidence']
            if not evidence:
                evidence = evidence_for_nei[i]
            selected.append({
                'claim': row['claim'],
                'evidence': evidence,
                'label': row['label']
            })
    assert len(selected) >= num_per_label * len(labels), f'selected:{selected}, limit:{num_per_label}, num_label:{len(labels)}'

    lines = [
        (f'---Example {i}---\n{format_fs_one(**item)}')
        for i, item in enumerate(selected)
    ]
    return '\n'.join(lines)

def test_build_few_shot():
    all_p = []
    for bench_cls in BENCHMARKS:
        fs_prompt = build_few_shot(bench_cls=bench_cls,limit=0,split='train')
        all_p.append(fs_prompt)
        breakpoint()
    print(all_p)


# for llm-gen
def prompt4llm_judge(
    benchmark_cls: Benchmark,
    claim: str,
    few_shot: str | None = None,
    evidence: str | None = None,
    is_cot: bool = False,
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
    if is_cot: 
        lines.append(
            'Let\' think step by step.'
        )
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
    tokenizer,
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


# def prompt4whether_retrieval(
#     claim: str,
#     evidence: str | None,
# ) -> tuple[str, str, list[str]]:
#     return prompt4jev_whether_retrieval(claim, evidence)


# def prompt4aspect_check(
#     claim: str,
#     aspect: Aspect,
#     evidence: str | None = None,
# ) -> tuple[str, str, list[str]]:
#     """Binary check. Option 1 is the positive class read from probs[0]."""
#     system_prompt = SYSTEM_PROMPT
#     options = ["1", "2"]
#     lines = [f"Claim: {claim}"]
#     if evidence is not None:
#         lines.append(f"Current evidence: {evidence}")
#     lines.extend([
#         f"Aspect: {aspect.name}",
#         f"Definition: {aspect.definition}",
#         "Does this aspect of the claim still need external evidence?",
#         "[1] yes (description: This aspect is unresolved and needs retrieval.)",
#         "[2] no (description: Current evidence is enough for this aspect, or it does not apply.)",
#     ])
#     return system_prompt, "\n".join(lines), options


# def prompt4what_choice(
#     benchmark_cls: Benchmark,
#     claim: str,
#     evidence: str | None = None,
# ) -> tuple[str, str, list[str]]:
#     return prompt4jev_judge(
#         benchmark_cls,
#         claim,
#         few_shot=None,
#         evidence=evidence,
#     )


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




def main():
    test_build_few_shot()



if __name__ == '__main__':
    main()
    