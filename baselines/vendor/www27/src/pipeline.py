"""JARV fact-checking pipeline: a bounded gate -> retrieve -> gate loop."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import TypeVar

from src.aspects import ASPECTS, ASPECT_BY_KEY, select_aspects
from src.config import METHODS
from src.utils.parsing import parse_query_list
from src.prompts import (
    SYSTEM_PROMPT,
    aspect_prompt,
    gate_prompt,
    judge_prompt,
    query_prompt,
)
from src.utils.text import clip, format_evidence, format_subclaims, is_leakage
from src.types import AtomicQuery, Choice, Cost, Example, Passage, add_usage

logger = logging.getLogger("src.pipeline")

T = TypeVar("T")


class FactChecker:
    def __init__(self, llm, retriever, config, system_prompt: str = SYSTEM_PROMPT):
        self.llm = llm
        self.retriever = retriever
        self.config = config
        self.system_prompt = system_prompt
        self.collected: list[Passage] = []
        self.seen_docs: set[str] = set()

    def generate(self, user: str, max_new_tokens: int, cost: Cost):
        generation = self.llm.generate(self.system_prompt, user, max_new_tokens)
        add_usage(cost, generation.usage)
        return generation

    def generate_json(
        self,
        user: str,
        parser: Callable[[str, int], T | None],
        limit: int,
        max_new_tokens: int,
        cost: Cost,
        retry_hint: str,
    ) -> T | None:
        """Generate a JSON value with a bounded number of retries."""
        attempts = self.config.json_retries + 1
        for attempt in range(attempts):
            prompt = user
            if attempt:
                prompt += f"\n{retry_hint}"
            generation = self.generate(prompt, max_new_tokens, cost)
            parsed = parser(generation.text, limit)
            if parsed is not None:
                cost.retries += attempt
                return parsed
            cost.parse_failures += 1
        cost.retries += self.config.json_retries
        return None

    def choose(self, user: str, options, cost: Cost) -> Choice:
        choice = self.llm.score_options(self.system_prompt, user, options)
        add_usage(cost, choice.usage)
        return choice

    def retrieve(self, example: Example, queries: list[AtomicQuery], cost: Cost) -> list[Passage]:
        added: list[Passage] = []
        blocked = {example.id, example.source_id}
        for query in queries:
            if cost.search_calls >= self.config.search_cap:
                cost.search_truncated = True
                break
            if not query.query or not query.query.strip():
                continue
            hits = self.retriever.search(
                query.query,
                k=self.config.passages_per_query,
                exclude_ids=blocked,
                max_passage_chars=self.config.max_passage_chars,
            )
            cost.search_calls += 1
            for hit in hits:
                if hit.doc_id in self.seen_docs:
                    continue
                if is_leakage(example.claim, blocked, hit.doc_id, hit.text):
                    continue
                self.seen_docs.add(hit.doc_id)
                hit.query = query.query
                hit.sub_claim = query.sub_claim
                hit.aspect = query.aspect
                self.collected.append(hit)
                added.append(hit)
                if sum(len(item.text) for item in self.collected) >= self.config.max_evidence_chars:
                    break
        return added

    def _fallback_query(self, claim: str, aspect_key: str) -> AtomicQuery:
        name = ASPECT_BY_KEY[aspect_key].name if aspect_key in ASPECT_BY_KEY else aspect_key
        return AtomicQuery(aspect_key, clip(claim, 240), clip(f"{name} {claim}", 180), True)

    def _aspect_queries(
        self,
        claim: str,
        aspect_key: str,
        evidence_text: str,
        cost: Cost,
    ) -> list[AtomicQuery]:
        aspect = ASPECT_BY_KEY[aspect_key]
        parsed = self.generate_json(
            query_prompt(
                claim,
                aspect,
                self.config.queries_per_aspect,
                evidence_text,
            ),
            parse_query_list,
            self.config.queries_per_aspect,
            self.config.query_max_new_tokens,
            cost,
            "Your previous reply was not a valid JSON list. "
            "Return only a JSON list of objects with sub_claim and query fields.",
        )
        if parsed is not None:
            return [AtomicQuery(aspect.key, sub, query, False) for sub, query in parsed]
        cost.fallbacks += 1
        return [self._fallback_query(claim, aspect.key)]

    def _select_aspects(
        self,
        claim: str,
        evidence_text: str,
        inspected: set[str],
        cost: Cost,
    ) -> tuple[list[str], dict[str, float]]:
        inspected_text = ", ".join(sorted(inspected)) or "none"
        scores: dict[str, float] = {}
        for aspect in ASPECTS:
            if aspect.key in inspected:
                continue
            user, options = aspect_prompt(
                claim,
                aspect,
                evidence_text,
                inspected_text,
                self.config.reverse_options,
            )
            choice = self.choose(user, options, cost)
            scores[aspect.key] = float(choice.probs["check"])
        selected = select_aspects(
            scores,
            self.config.tau_a,
            self.config.max_aspects,
            keep_best_if_empty=True,
        )
        return selected, scores

    def _decide_retrieve(
        self,
        claim: str,
        evidence_text: str,
        inspected: set[str],
        cost: Cost,
    ) -> float:
        inspected_text = ", ".join(sorted(inspected)) or "none"
        user, options = gate_prompt(
            claim,
            evidence_text,
            inspected_text,
            self.config.reverse_options,
        )
        choice = self.choose(user, options, cost)
        return float(choice.probs["retrieve"])

    def _search_selected(
        self,
        example: Example,
        claim: str,
        aspects: list[str],
        evidence_text: str,
        cost: Cost,
    ) -> list[AtomicQuery]:
        queries: list[AtomicQuery] = []
        for aspect_key in aspects:
            if cost.search_calls >= self.config.search_cap:
                cost.search_truncated = True
                break
            got = self._aspect_queries(claim, aspect_key, evidence_text, cost)
            queries.extend(got)
            if got:
                self.retrieve(example, got, cost)
        if not queries and self.config.search_cap > cost.search_calls:
            cost.fallbacks += 1
            fallback = AtomicQuery("holistic", clip(claim, 240), clip(claim, 180), True)
            queries.append(fallback)
            self.retrieve(example, [fallback], cost)
        return queries

    def _judge(self, claim: str, subclaims: list[str], evidence: list[Passage], cost: Cost):
        user, options = judge_prompt(
            claim,
            format_subclaims(subclaims),
            format_evidence(evidence, self.config.max_evidence_chars),
            self.config.reverse_options,
        )
        choice = self.choose(user, options, cost)
        return choice.key, choice.probs

    def run(self, example: Example) -> dict:
        if self.config.method not in METHODS:
            raise ValueError(f"Unknown method {self.config.method}")
        started = time.perf_counter()
        cost = Cost()
        self.collected = []
        self.seen_docs = set()
        set_claim = getattr(self.retriever, "set_claim", None)
        if callable(set_claim):
            set_claim(example.id)
        claim = clip(example.claim, self.config.max_claim_chars)
        prediction = ""
        judge_probs: dict[str, float] = {}
        gate_probs: list[float] = []
        aspects: list[str] = []
        aspect_probs: dict[str, float] = {}
        queries: list[AtomicQuery] = []
        inspected: set[str] = set()
        stop_reason = ""
        rounds = 0
        error = None
        try:
            while cost.search_calls < self.config.search_cap:
                rounds += 1
                evidence_text = format_evidence(self.collected, self.config.max_evidence_chars)
                probability = self._decide_retrieve(
                    claim, evidence_text, inspected, cost
                )
                gate_probs.append(probability)
                if probability < self.config.tau_g:
                    stop_reason = "gate"
                    break
                selected, scores = self._select_aspects(
                    claim, evidence_text, inspected, cost
                )
                aspect_probs.update(scores)
                selected = [key for key in selected if key not in inspected]
                if not selected:
                    stop_reason = "no_uncovered_aspect"
                    break
                before = cost.search_calls
                new_queries = self._search_selected(
                    example, claim, selected, evidence_text, cost
                )
                if cost.search_calls == before:
                    stop_reason = "no_query"
                    break
                inspected.update(selected)
                aspects.extend(selected)
                queries.extend(new_queries)
            if not stop_reason:
                stop_reason = "budget" if cost.search_calls >= self.config.search_cap else "gate"
            subclaims: list[str] = []
            for query in queries:
                if query.sub_claim and query.sub_claim not in subclaims:
                    subclaims.append(query.sub_claim)
            prediction, judge_probs = self._judge(claim, subclaims, self.collected, cost)
        except (ValueError, KeyError) as exc:
            logger.error("Failed on example %s: %s", example.id, exc)
            error = f"{type(exc).__name__}: {exc}"
        except Exception as exc:
            logger.exception("Failed on example %s", example.id)
            error = f"{type(exc).__name__}: {exc}"
        latency = time.perf_counter() - started
        return {
            "id": example.id,
            "domain": example.domain,
            "gold": example.label,
            "prediction": prediction,
            "method": self.config.method,
            "retrieved": cost.search_calls > 0,
            "gate_prob_retrieve": gate_probs[-1] if gate_probs else None,
            "gate_probs": gate_probs,
            "gate_source": "model" if gate_probs else "none",
            "retrieval_rounds": rounds,
            "stop_reason": stop_reason,
            "aspects": aspects,
            "aspect_probs": aspect_probs,
            "queries": [
                {
                    "aspect": query.aspect,
                    "sub_claim": query.sub_claim,
                    "query": query.query,
                    "fallback": query.fallback,
                }
                for query in queries
            ],
            "evidence": [
                {
                    "doc_id": passage.doc_id,
                    "score": passage.score,
                    "title": passage.title,
                    "url": passage.url,
                    "date": passage.date,
                    "query": passage.query,
                    "sub_claim": passage.sub_claim,
                    "aspect": passage.aspect,
                    "text": passage.text,
                }
                for passage in self.collected
            ],
            "judge_probs": judge_probs,
            "search_calls": cost.search_calls,
            "forward_passes": cost.forward_passes,
            "input_tokens": cost.input_tokens,
            "output_tokens": cost.output_tokens,
            "latency_s": latency,
            "parse_failures": cost.parse_failures,
            "fallbacks": cost.fallbacks,
            "retries": cost.retries,
            "search_truncated": cost.search_truncated,
            "error": error,
        }
