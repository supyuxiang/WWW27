from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0,str(ROOT))

from src.aspects import Aspect, ASPECTS
from src.utils import BENCHMARKS

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
    user_prompt = f'Claim"{claim}.'
    if current_evidence is not None:
        user_prompt += f'\nCurrent evidence:{current_evidence}'
    user_prompt += ("Decide whether more external evidence is needed to assess the central factual content, then select one:\n "
                    '[1] Need more evidence.\n'
                    '[2] Do not need more evidence.'
                    )
    options = ['1','2']
    return system_prompt, user_prompt, options

def prompt4what_aspect():
    pass

def prompt4what_choice():
    pass


def _options(
    pairs: tuple[tuple[str, str], ...] | list[tuple[str, str]],
    reverse: bool,
) -> tuple[str, list[str]]:
    ordered = list(reversed(pairs)) if reverse else list(pairs)
    lines = []
    options = []
    for index, (key, text) in enumerate(ordered, start=1):
        lines.append(f"[{index}] {text}")
        options.append(Option(key=key, suffix=f"{index}]", text=text))
    return "\n".join(lines), options


def gate_prompt(
    claim: str,
    evidence: str = "none",
    inspected: str = "none",
    reverse: bool = False,
):
    listing, options = _options(_GATE_PAIRS, reverse)
    user = (
        f"Claim: {claim}\n"
        f"Current evidence: {evidence}\n"
        f"Already inspected aspects: {inspected}\n"
        "Decide whether more external evidence is needed to assess the central factual content. "
        "Consider unresolved attribution, temporal sensitivity, numerical context, and factual relationships. "
        "A number, named entity, or date alone does not make retrieval necessary. "
        "Choose Skip only when the current evidence and existing knowledge are sufficient for the "
        "main factual content, or when another search is unlikely to resolve a material uncertainty.\n"
        f"{listing}\n"
        "Select one option. Answer only with its bracketed numeric identifier."
    )
    return user, options


def aspect_prompt(
    claim: str,
    aspect: Aspect,
    evidence: str = "none",
    inspected: str = "none",
    reverse: bool = False,
):
    listing, options = _options(_ASPECT_PAIRS, reverse)
    user = (
        f"Claim: {claim}\n"
        f"Current evidence: {evidence}\n"
        f"Already inspected aspects: {inspected}\n"
        f"Candidate aspect: {aspect.to_str()}\n"
        "Does assessing this uncovered aspect require additional external investigation to resolve "
        "a material factual uncertainty in the claim? Relevance to the topic alone is insufficient.\n"
        f"{listing}\n"
        "Select one option. Answer only with its bracketed numeric identifier."
    )
    return user, options


def judge_prompt(
    claim: str,
    subclaims: str,
    evidence: str,
    reverse: bool = False,
):
    listing, options = _options(_JUDGE_PAIRS, reverse)
    user = (
        f"Original claim: {claim}\n"
        f"Inspected sub-claims: {subclaims}\n"
        f"Retrieved evidence with source identifiers and dates: {evidence}\n"
        "Assess the original claim using the available evidence and your existing knowledge. "
        "Distinguish support, refutation, and insufficient information. "
        "Check that evidence matches the entities, time, population, quantities, and attribution. "
        "A missing result does not establish falsehood; source count or recency alone does not establish truth. "
        "Judge material factual content rather than tone or political agreement.\n"
        f"{listing}\n"
        "Select the best binary option. Answer only with its bracketed numeric identifier."
    )
    return user, options


def query_prompt(claim: str, aspect: Aspect, max_queries: int, evidence: str = "none") -> str:
    return (
        f"Claim: {claim}\n"
        f"Selected aspect: {aspect.brief()}\n"
        f"Evidence already collected: {evidence}\n"
        f"Identify at most {max_queries} atomic factual propositions relevant to this aspect. "
        "Preserve the original factual scope and qualifiers. For each proposition, write one concise, "
        "neutral search query that resolves references and identifies the relevant entities, time, and context. "
        "Do not assume that the proposition is true or false and do not invent supporting information. "
        "Return a JSON list containing sub_claim and query fields. "
        "Return an empty list if no applicable factual proposition can be identified."
    )
