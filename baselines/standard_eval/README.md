# Standard-label fact-checking comparison

Code:

```text
baselines/standard_eval/run_standard_eval.py
```

Evaluates on the original label spaces:

```text
FEVER:    Supported / Refuted / Not Enough Evidence
SciFact:  Support / Refute / Not Enough Evidence
AVeriTeC: Supported / Refuted / Conflicting / Not Enough Evidence
```

Methods:

```text
jev_no_retrieval
jev_retrieval
averitec       question generation + retrieval + QA + verdict
programfc      claim decomposition + sub-claim checking + aggregation
sled           denoising + PRO/CON debate + verdict
```

Prompts follow the task definitions of the original papers as closely as
possible; the papers' exact prompts/trained checkpoints are not public.

## Evaluation

```bash
cd /root/www27
/root/autodl-tmp/miniconda3/envs/verl1/bin/python \
  baselines/standard_eval/run_standard_eval.py \
  --benchmarks averitec scifact fever \
  --methods jev_no_retrieval jev_retrieval averitec programfc sled \
  --limit 300 \
  --model /root/autodl-tmp/models/Qwen2.5-3B-Instruct
```

`--limit 0` uses the full dev split.
