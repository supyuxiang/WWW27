"""Run the JARV retrieval loop with a local model and a local BM25 index."""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

from src.config import JarvConfig, load_config
from src.data import corpus_from_examples, load_corpus, load_examples, read_jsonl
from src.llm import LocalLLM
from src.utils.metrics import summarize
from src.pipeline import FactChecker
from src.retrieval import BM25Retriever

logger = logging.getLogger("src.run")


def _configure_runtime() -> None:
    """Set process limits before torch or transformers are imported."""
    if os.environ.get("MKL_NUM_THREADS", "1") in {"", "0"}:
        os.environ["MKL_NUM_THREADS"] = "1"
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def _apply_overrides(config: JarvConfig, args) -> None:
    mapping = {
        "model": "model_path",
        "data": "data_path",
        "corpus": "corpus_path",
        "output": "output_path",
        "claim_mode": "claim_mode",
        "domain": "domain",
        "limit": "limit",
        "offset": "offset",
        "seed": "seed",
        "tau_g": "tau_g",
        "tau_a": "tau_a",
        "max_aspects": "max_aspects",
        "queries_per_aspect": "queries_per_aspect",
        "passages_per_query": "passages_per_query",
        "search_cap": "search_cap",
    }
    for arg_name, field_name in mapping.items():
        value = getattr(args, arg_name)
        if value is not None and value != "":
            setattr(config, field_name, value)
    if args.reverse_options:
        config.reverse_options = True
    if args.resume:
        config.resume = True


def _done_ids(path: str) -> set[str]:
    if not os.path.exists(path):
        return set()
    return {str(row.get("id")) for row in read_jsonl(path) if not row.get("error")}


def _print_summary(rows: list[dict]) -> None:
    summary = summarize(rows)
    print(
        f"{summary['n']} claims: acc={100.0 * summary['accuracy']:.2f}% "
        f"calls/item={summary['calls_per_item']:.3f} "
        f"rounds/item={summary['rounds_per_item']:.2f} "
        f"stop_gate={100.0 * summary['stop_gate_rate']:.1f}% "
        f"stop_budget={100.0 * summary['stop_budget_rate']:.1f}% "
        f"errors={summary['n_errors']}"
    )


def run_examples(checker: FactChecker, examples, output_path: str, resume: bool) -> list[dict]:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    done = _done_ids(output_path) if resume else set()
    mode = "a" if resume and path.exists() else "w"
    rows = read_jsonl(path) if resume and path.exists() else []
    try:
        from tqdm import tqdm
    except ImportError:
        def tqdm(iterable, **_kwargs):
            return iterable
    with path.open(mode, encoding="utf-8") as handle:
        for example in tqdm(examples, desc="jarv"):
            if example.id in done:
                continue
            row = checker.run(example)
            rows.append(row)
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
    return rows


def main() -> None:
    _configure_runtime()
    parser = argparse.ArgumentParser(description="Run the JARV retrieval loop with a local LLM")
    parser.add_argument("--config", default=None, help="YAML config; defaults to configs/default.yaml")
    parser.add_argument("--model", default=None)
    parser.add_argument("--data", default=None)
    parser.add_argument("--corpus", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--claim-mode", default=None, choices=("stored", "headline", "headline_lead", "full"))
    parser.add_argument("--domain", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--offset", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--tau-g", dest="tau_g", type=float, default=None)
    parser.add_argument("--tau-a", dest="tau_a", type=float, default=None)
    parser.add_argument("--max-aspects", dest="max_aspects", type=int, default=None)
    parser.add_argument("--queries-per-aspect", dest="queries_per_aspect", type=int, default=None)
    parser.add_argument("--passages-per-query", dest="passages_per_query", type=int, default=None)
    parser.add_argument("--search-cap", dest="search_cap", type=int, default=None)
    parser.add_argument("--retrieval-backend", choices=("bm25", "sqlite"), default="bm25")
    parser.add_argument("--index-path", default="")
    parser.add_argument("--claim-scoped", action="store_true")
    parser.add_argument("--reverse-options", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config_path = args.config
    if config_path is None:
        default_path = Path("configs/default.yaml")
        if default_path.is_file():
            config_path = str(default_path)
        else:
            logger.warning("configs/default.yaml not found; using built-in defaults")
    elif not Path(config_path).is_file():
        raise SystemExit(f"Config file not found: {config_path}")
    config = load_config(config_path)
    _apply_overrides(config, args)
    examples = load_examples(
        config.data_path,
        claim_mode=config.claim_mode,
        domain=config.domain,
        offset=config.offset,
        limit=config.limit,
    )
    if not examples:
        raise SystemExit(f"No examples in {config.data_path}")
    if args.retrieval_backend == "sqlite":
        if not args.index_path or not os.path.exists(args.index_path):
            raise SystemExit(f"SQLite index not found: {args.index_path}")
        from src.retrieval_sqlite import SQLiteBM25Retriever

        retriever = SQLiteBM25Retriever(
            args.index_path,
            claim_scoped=args.claim_scoped,
        )
        documents = []
        logger.info("Using SQLite FTS5 index %s", args.index_path)
    else:
        if config.corpus_path and os.path.exists(config.corpus_path):
            documents = load_corpus(config.corpus_path)
        else:
            logger.warning("Corpus file missing; indexing the claim texts themselves")
            documents = corpus_from_examples(examples)
        retriever = BM25Retriever(documents, k1=config.bm25_k1, b=config.bm25_b)
        logger.info("Indexed %s documents", len(documents))
    llm = LocalLLM(config.model_path)
    checker = FactChecker(llm, retriever, config)
    rows = run_examples(checker, examples, config.output_path, config.resume)
    meta_path = str(Path(config.output_path).with_suffix(".meta.json"))
    meta = {
        "config": config.to_dict(),
        "model_path": config.model_path,
        "precision": "bfloat16" if llm.device.type == "cuda" else "float32",
        "decision": "full candidate-suffix log-likelihood, no length normalization, no fitted calibration",
        "loop": "gate -> retrieve -> update evidence -> gate; stop when gate declines or search_cap is reached",
        "tau_g": config.tau_g,
        "tau_a": config.tau_a,
        "max_aspects": config.max_aspects,
        "queries_per_aspect": config.queries_per_aspect,
        "passages_per_query": config.passages_per_query,
        "search_cap": config.search_cap,
        "reverse_options": config.reverse_options,
        "retrieval_backend": args.retrieval_backend,
        "index_path": args.index_path,
        "claim_scoped": args.claim_scoped,
        "n_examples": len(examples),
        "n_documents": len(documents),
    }
    Path(meta_path).write_text(json.dumps(meta, indent=2), encoding="utf-8")
    _print_summary(rows)
    print(f"wrote {config.output_path}")
    print(f"wrote {meta_path}")


if __name__ == "__main__":
    main()
