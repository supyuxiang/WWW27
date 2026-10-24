"""Build claim and corpus jsonl files from the local news dump."""

from __future__ import annotations

import argparse
import json

from src.data import prepare_local_news


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare local fact-checking jsonl files")
    parser.add_argument("--source", default="/root/autodl-tmp/jev/validation.jsonl")
    parser.add_argument("--claims", default="data/claims.jsonl")
    parser.add_argument("--corpus", default="data/corpus.jsonl")
    parser.add_argument(
        "--claim-mode",
        default="headline_lead",
        choices=("headline", "headline_lead", "full"),
    )
    args = parser.parse_args()
    counts = prepare_local_news(args.source, args.claims, args.corpus, claim_mode=args.claim_mode)
    print(json.dumps(counts, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
