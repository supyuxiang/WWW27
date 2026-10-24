'''Aspect inventory from the paper appendix.'''

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Aspect:
    key: str
    name: str
    definition: str

    def brief(self) -> str:
        return f'{self.name}: {self.definition}'


ASPECTS: tuple[Aspect, ...] = (
    Aspect(
        'event_context',
        'Event/context',
        'Whether the described event, entities, and contextual association match the available record.',
    ),
    Aspect(
        'temporal_scope',
        'Temporal scope',
        'Whether dates, ordering, and the reuse of historical material are applicable to the claim.',
    ),
    Aspect(
        'quantities',
        'Quantities',
        'Whether values, units, denominators, populations, and comparison periods are correctly specified.',
    ),
    Aspect(
        'attribution_actions',
        'Attribution / actions',
        'Whether a quotation, statement, policy, or institutional action is attributed to the correct actor and context.',
    ),
    Aspect(
        'causality',
        'Causality',
        'Whether the claimed causal relationship is supported, rather than inferred solely from co-occurrence or sequence.',
    ),
)

ASPECT_BY_KEY: dict[str, Aspect] = {aspect.key: aspect for aspect in ASPECTS}


def select_aspects(
    scores: dict[str, float],
    tau: float,
    limit: int,
    keep_best_if_empty: bool,
) -> list[str]:
    '''Keep aspects with score >= tau, at most `limit`, highest first.

    When the retrieval gate has already fired and nothing passes `tau`,
    the highest-scoring aspect is kept. Scores across aspects are not
    renormalized to sum to one.
    '''
    ranked = sorted(scores, key=lambda key: (-float(scores[key]), key))
    chosen = [key for key in ranked if float(scores[key]) >= tau]
    if limit >= 0:
        chosen = chosen[:limit]
    if not chosen and keep_best_if_empty and ranked:
        chosen = ranked[:1]
    return chosen
