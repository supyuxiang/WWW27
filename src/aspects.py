from __future__ import annotations

from dataclasses import dataclass
import torch



@dataclass(frozen=True)
class Aspect:
    name: str
    definition: str

    def to_str(self):
        return f'Name:{self.name}, definition:{self.definition}.'


ASPECTS: tuple[Aspect, ...] = (
    Aspect(
        "event_context",
        "Whether the described event, entities, and contextual association match the available record.",
    ),
    Aspect(
        "temporal_scope",
        "Whether dates, ordering, and the reuse of historical material are applicable to the claim.",
    ),
    Aspect(
        "quantities",
        "Whether values, units, denominators, populations, and comparison periods are correctly specified.",
    ),
    Aspect(
        "attribution_actions",
        "Whether a quotation, statement, policy, or institutional action is attributed to the correct actor and context.",
    ),
    Aspect(
        "causality",
        "Whether the claimed causal relationship is supported, rather than inferred solely from co-occurrence or sequence.",
    ),
)


ASPECT_BY_NAME: dict[str, Aspect] = {aspect.name: aspect for aspect in ASPECTS}




def select_aspects(
    scores: torch.Tensor,
    tau_aspect: float,
    limit: int=-1,
) -> list[str]:
    _, ranked_idx = torch.sort(input=scores, dim=-1)
    chosen = [idx for idx in ranked_idx if float(scores[idx]) >= tau_aspect]
    if limit >= 0:
        chosen = chosen[:limit]
    return chosen
