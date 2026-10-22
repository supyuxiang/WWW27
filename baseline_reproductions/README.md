# FIRE and GProofT baseline reproductions

This subtree contains the code used for the two added published baselines. It
is isolated from the repository's existing `src/` and `eval/` code so the
reproductions can be reviewed without changing the original experiments.

## Layout

- `fire/run_binary.py`: FIRE on AVeriTeC, FEVER, and SciFact.
- `fire/run_threeway.py`: FIRE adapted to CFEVER and QuanTemp++ native labels.
- `gprooft_adapted/run_binary.py`: GProofT on the original binary protocol.
- `gprooft_adapted/run_threeway.py`: GProofT adapted to CFEVER and QuanTemp++.
- `common/`: atomic checkpoints, dataset loading, and full-scope metrics.
- `vendor/www27/`: frozen dependency snapshot used by the completed runs.

Large models, benchmark corpora, SQLite indexes, and result checkpoints are not
committed. They are passed by absolute command-line paths.

## Reproduction policy

- Frozen local model with `tokenizer.apply_chat_template()`.
- Deterministic decoding and `enable_thinking=False` where supported.
- Top-k 3 and at most three actual search calls.
- Per-sample atomic JSON checkpoints; compatible successful rows are resumed.
- Accuracy and Macro-F1 include every expected sample. Invalid predictions are
  errors and are not silently removed from the denominator.
- No gold evidence or prediction label is used to make retrieval decisions.

## Method fidelity

FIRE retains its iterative decision/search loop, forced final-answer retries,
and semantic-repeat stopping. Serper is replaced by the fixed offline corpus.

GProofT retains claim decomposition, binary-question generation, passage-level
Yes/No voting, verdict aggregation, and justification. Hosted Azure/OpenAI and
Google Search dependencies are replaced by the frozen local model and fixed
offline corpus, so it is explicitly reported as **adapted**.

## Example commands

```bash
python baseline_reproductions/fire/run_binary.py --help
python baseline_reproductions/fire/run_threeway.py --help
python baseline_reproductions/gprooft_adapted/run_binary.py --help
python baseline_reproductions/gprooft_adapted/run_threeway.py --help
```

The three-way entry points require `--benchmark`, `--split-file`, `--index`,
`--model`, and `--output-root`. FIRE additionally requires `--repeat-model`.

## Provenance

- FIRE paper implementation: <https://github.com/mbzuai-nlp/fire>
- GProofT official repository: <https://github.com/HKUST-KnowComp/GProofT>
- GProofT pinned revision: `18de5d570805e7333ae2e514696b52a5603bb910`

The frozen dependency files are included to preserve the exact prompts,
teacher-forced scoring helpers, retrievers, and trace schema used in the runs.
