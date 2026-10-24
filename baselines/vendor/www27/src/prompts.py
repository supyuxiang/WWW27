'''Prompts for the three JARV decisions in the retrieval loop.'''

from __future__ import annotations

from src.aspects import Aspect
from src.types import Option

SYSTEM_PROMPT = (
    'You are a helpful assistant for factual verification.'
)

ASSISTANT_PREFILL = 'Best answer: ['

_GATE_PAIRS = (
    ('skip', 'Skip external retrieval.'),
    ('retrieve', 'Retrieve external evidence.'),
)
_ASPECT_PAIRS = (
    ('check', 'Check this aspect using external evidence.'),
    ('skip', 'Skip investigation of this aspect.'),
)
_JUDGE_PAIRS = (
    (
        'real',
        'Real: the central factual content is substantially accurate under the task definition.',
    ),
    (
        'fake',
        'Fake: the central factual content is materially false or misleading under the task definition.',
    ),
)


def _options(
    pairs: tuple[tuple[str, str], ...] | list[tuple[str, str]],
    reverse: bool,
) -> tuple[str, list[Option]]:
    ordered = list(reversed(pairs)) if reverse else list(pairs)
    lines = []
    options = []
    for index, (key, text) in enumerate(ordered, start=1):
        lines.append(f'[{index}] {text}')
        options.append(Option(key=key, suffix=f'{index}]', text=text))
    return '\n'.join(lines), options


def gate_prompt(
    claim: str,
    evidence: str = 'none',
    inspected: str = 'none',
    reverse: bool = False,
) -> tuple[str, list[Option]]:
    listing, options = _options(_GATE_PAIRS, reverse)
    user = (
        f'Claim: {claim}\n'
        f'Current evidence: {evidence}\n'
        f'Already inspected aspects: {inspected}\n'
        'Decide whether more external evidence is needed to assess the central factual content. '
        'Consider unresolved attribution, temporal sensitivity, numerical context, and factual relationships. '
        'A number, named entity, or date alone does not make retrieval necessary. '
        'Choose Skip only when the current evidence and existing knowledge are sufficient for the '
        'main factual content, or when another search is unlikely to resolve a material uncertainty.\n'
        f'{listing}\n'
        'Select one option. Answer only with its bracketed numeric identifier.'
    )
    return user, options


def aspect_prompt(
    claim: str,
    aspect: Aspect,
    evidence: str = 'none',
    inspected: str = 'none',
    reverse: bool = False,
) -> tuple[str, list[Option]]:
    listing, options = _options(_ASPECT_PAIRS, reverse)
    user = (
        f'Claim: {claim}\n'
        f'Current evidence: {evidence}\n'
        f'Already inspected aspects: {inspected}\n'
        f'Candidate aspect: {aspect.brief()}\n'
        'Does assessing this uncovered aspect require additional external investigation to resolve '
        'a material factual uncertainty in the claim? Relevance to the topic alone is insufficient.\n'
        f'{listing}\n'
        'Select one option. Answer only with its bracketed numeric identifier.'
    )
    return user, options


def judge_prompt(
    claim: str,
    subclaims: str,
    evidence: str,
    reverse: bool = False,
) -> tuple[str, list[Option]]:
    listing, options = _options(_JUDGE_PAIRS, reverse)
    user = (
        f'Original claim: {claim}\n'
        f'Inspected sub-claims: {subclaims}\n'
        f'Retrieved evidence with source identifiers and dates: {evidence}\n'
        'Assess the original claim using the available evidence and your existing knowledge. '
        'Distinguish support, refutation, and insufficient information. '
        'Check that evidence matches the entities, time, population, quantities, and attribution. '
        'A missing result does not establish falsehood; source count or recency alone does not establish truth. '
        'Judge material factual content rather than tone or political agreement.\n'
        f'{listing}\n'
        'Select the best binary option. Answer only with its bracketed numeric identifier.'
    )
    return user, options


def query_prompt(claim: str, aspect: Aspect, max_queries: int, evidence: str = 'none') -> str:
    return (
        f'Claim: {claim}\n'
        f'Selected aspect: {aspect.brief()}\n'
        f'Evidence already collected: {evidence}\n'
        f'Identify at most {max_queries} atomic factual propositions relevant to this aspect. '
        'Preserve the original factual scope and qualifiers. For each proposition, write one concise, '
        'neutral search query that resolves references and identifies the relevant entities, time, and context. '
        'Do not assume that the proposition is true or false and do not invent supporting information. '
        'Return a JSON list containing sub_claim and query fields. '
        'Return an empty list if no applicable factual proposition can be identified.'
    )
