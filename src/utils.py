import json
import re
import sys
from pathlib import Path
from dataclasses import dataclass

ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0,str(ROOT))


@dataclass
class Benchmark:
    name: str
    path: str | Path
    choice_dict: dict
    doc: str | None = None
    train_path: str | Path | None = None


PREPROCESSED_ROOT = Path('/root/autodl-tmp/benchmark_dsh/preprocessed')
CORPUS_ROOT = PREPROCESSED_ROOT / 'corpus'


# averitec, fever, scifact, healthver
BENCHMARKS = [
    Benchmark(
        name='averitec',
        path=PREPROCESSED_ROOT / 'claims/averitec/dev.jsonl',
        train_path=PREPROCESSED_ROOT / 'claims/averitec/train.jsonl',
        choice_dict={
            'Supported': (
                'The evidence is sufficient to conclude that the central factual content of the claim is true.'
            ),
            'Refuted': (
                'The evidence is sufficient to conclude that the central factual content of the claim is false.'
            ),
            'Not Enough Evidence': (
                'The evidence is insufficient to support or refute the claim, '
                'including missing, off-topic, or unreliable sources.'
            ),
            'Conflicting Evidence/Cherrypicking': (
                'The claim is misleading: reliable sources conflict, or the claim '
                'cherry-picks true facts while omitting context needed for a fair reading.'
            ),
        },
        doc=(
            'AVeriTeC is a real-world fact-checking benchmark. Claims come from fact-checking '
            'articles and are verified against open-web evidence. Each claim receives one of four '
            'verdicts: Supported, Refuted, Not Enough Evidence, or Conflicting Evidence/Cherrypicking.'
        ),
    ),

    Benchmark(
        name='fever',
        path=PREPROCESSED_ROOT / 'claims/fever/dev.jsonl',
        train_path=PREPROCESSED_ROOT / 'claims/fever/train.jsonl',
        choice_dict={
            'SUPPORTS': 'The evidence supports the claim.',
            'REFUTES': 'The evidence refutes the claim.',
            'NOT ENOUGH INFO': (
                'The evidence does not contain enough information to support or refute the claim.'
            ),
        },
        doc=(
            'FEVER checks claims against Wikipedia. Labels are SUPPORTS, REFUTES, '
            'and NOT ENOUGH INFO.'
        ),
    ),

    Benchmark(
        name='scifact',
        path=PREPROCESSED_ROOT / 'claims/scifact/dev.jsonl',
        train_path=PREPROCESSED_ROOT / 'claims/scifact/train.jsonl',
        choice_dict={
            'SUPPORT': 'The scientific evidence supports the claim.',
            'CONTRADICT': 'The scientific evidence contradicts the claim.',
            'NOT ENOUGH INFO': (
                'The evidence does not contain enough information to support or contradict the claim.'
            ),
        },
        doc=(
            'SciFact checks scientific claims against research-paper abstracts. '
            'Labels are SUPPORT, CONTRADICT, and NOT ENOUGH INFO.'
        ),
    ),
    
    Benchmark(
        name='healthver',
        path=PREPROCESSED_ROOT / 'claims/healthver/test.jsonl',
        train_path=PREPROCESSED_ROOT / 'claims/healthver/train.jsonl',
        choice_dict={
            'SUPPORTS': 'The evidence supports the health claim.',
            'REFUTES': 'The evidence refutes the health claim.',
            'NOT ENOUGH INFO': (
                'The evidence does not contain enough information to support or refute the claim.'
            ),
        },
        doc=(
            'HealthVer checks real-world health claims against scientific evidence. '
            'This is the 1000-claim test sample.'
        ),
    ),
]


def load_tokenizer(model_path: str | Path, is_training: bool=False):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    if is_training:
        tok.padding_side='right'
    else:
        tok.padding_side='left'
    return tok



def _read_jsonl(path: str | Path):
    with open(path, 'r', encoding='utf-8') as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)

def load_benchmark(
    benchmark_name: str,
    limit: int = -1,
    split: str = 'dev',
) -> tuple[list[dict], Benchmark]:
    """Load a normalized claim split.

    The preprocessed files already contain ``id``, ``claim``, ``label`` and
    ``evidence``.  No evidence filtering or evidence injection is done here;
    that belongs to the evaluation script.
    """
    for bench in BENCHMARKS:
        if benchmark_name != bench.name:
            continue

        if split == 'train':
            path = bench.train_path
            
        elif split in ('dev', 'test'):
            path = bench.path
        else:
            raise ValueError(f'unknown split {split!r}')
        
        assert Path(path).exists(), path

        benchmark_data = []
        for item in _read_jsonl(path):
            benchmark_data.append({
                'id': item.get('id'),
                'claim': item['claim'],
                'label': item['label'],
                'evidence': item.get('evidence', ''),
            })
        if limit > 0:
            benchmark_data = benchmark_data[:limit]
        return benchmark_data, bench
    
    raise ValueError(f'{benchmark_name} not found!')


def load_corpus(
    benchmark_name: str,
    limit: int = -1,
    split: str | None = None,
) -> list:
    """Load a normalized corpus as ``Document`` objects.

    ``split`` is accepted for interface compatibility; corpus files are shared
    across splits.
    """
    from src.config import Document

    path = CORPUS_ROOT / f'{benchmark_name}.jsonl'
    assert path.exists(), path
    
    documents = []
    for item in _read_jsonl(path):
        documents.append(Document(
            doc_id=str(item.get('doc_id')),
            title=str(item.get('title') or ''),
            text=str(item.get('text') or ''),
            url=str(item.get('url') or ''),
            date=str(item.get('date') or ''),
        ))
        if limit > 0 and len(documents) >= limit:
            break
    return documents



# metrics

def compute_accuracy(gold_ls: list[str], pred_ls: list[str]) -> float:
    assert len(gold_ls) == len(pred_ls)
    correct = sum(truth == guess for truth, guess in zip(gold_ls, pred_ls))
    return correct / max(len(gold_ls),1)

def compute_f1(
    gold_ls: list[str],
    pred_ls: list[str],
    benchmark: Benchmark,
) -> float:
    assert len(gold_ls) == len(pred_ls)
    labels = list(benchmark.choice_dict)
    if not labels:
        return 0.0
    f1_sum = 0.0
    for label in labels:
        tp = fp = fn = 0
        for truth, guess in zip(gold_ls, pred_ls):
            if guess == label and truth == label:
                tp += 1
            elif guess == label and truth != label:
                fp += 1
            elif truth == label and guess != label:
                fn += 1
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1_sum += (2 * precision * recall / (precision + recall)) if precision + recall else 0.0
    return f1_sum / len(labels)


def per_class_scores(
    gold: list[str],
    pred: list[str],
    labels: list[str],
) -> dict[str, dict[str, float]]:
    """Precision, recall, and F1 for each official label.

    Predictions outside ``labels`` (such as ``invalid``) count as false
    negatives for the gold label and are not a class of their own.
    """
    scores: dict[str, dict[str, float]] = {}
    for label in labels:
        tp = fp = fn = 0
        for truth, guess in zip(gold, pred):
            if guess == label and truth == label:
                tp += 1
            elif guess == label and truth != label:
                fp += 1
            elif truth == label and guess != label:
                fn += 1
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        scores[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    return scores




########   normalize  generated text    ##########
MATH_VERIFY_ROOT = ROOT / "third_party" / "Math-Verify"

_LATEX_WRAPPER_RE = re.compile(
    r"\\(?:text|textnormal|mathrm|textbf|operatorname|mbox)\s*\{([^{}]*)\}"
)
_SUPPORT_ALIASES = {"support", "supports", "supported", "s"}
_REFUTE_ALIASES = {
    "refute",
    "refutes",
    "refuted",
    "contradict",
    "contradicts",
    "contradicted",
    "r",
}
_NEI_ALIASES = {
    "nei",
    "not enough info",
    "not enough information",
    "not enough evidence",
    "insufficient evidence",
    "insufficient information",
}

def _normalize_label_text(value: str) -> str:
    value = str(value).strip()
    for _ in range(3):
        value = _LATEX_WRAPPER_RE.sub(r"\1", value)
    value = value.strip().strip("{}").strip()
    value = value.strip("$").strip()
    return " ".join(value.split())

def _semantic_key(value: str) -> str:
    key = _normalize_label_text(value).casefold()
    key = re.sub(r"[^a-z0-9]+", " ", key).strip()
    if key in _SUPPORT_ALIASES:
        return "SUPPORTS"
    if key in _REFUTE_ALIASES:
        return "REFUTES"
    if key in _NEI_ALIASES:
        return "NOT ENOUGH INFO"
    return key

def _extract_answer_candidates(text: str) -> list[str]:
    if not text:
        return []
    vendor = str(MATH_VERIFY_ROOT / "src")
    if vendor not in sys.path:
        sys.path.insert(0, vendor)
    from math_verify import parse as math_verify_parse
    from math_verify.parser import LatexExtractionConfig

    try:
        parsed = math_verify_parse(
            text,
            extraction_config=[LatexExtractionConfig(boxed_match_priority=0)],
            # first_match also exposes the raw extracted text (needed for
            # textual labels); no_fallback only returns the SymPy object.
            fallback_mode="first_match",
        )
    except Exception:
        return []
    raw_strings = [item for item in parsed if isinstance(item, str)]
    if raw_strings:
        return raw_strings
    return [str(item) for item in parsed]


def _match_label(value: str, labels: list[str]) -> str | None:
    normalized = _normalize_label_text(value)
    folded = normalized.casefold()

    exact = {label.casefold(): label for label in labels}
    if folded in exact:
        return exact[folded]

    semantic = {_semantic_key(label): label for label in labels}
    semantic_key = _semantic_key(normalized)
    if semantic_key in semantic:
        return semantic[semantic_key]

    hits: list[tuple[int, int, str]] = []
    for label in labels:
        pattern = re.compile(rf"(?<!\w){re.escape(label)}(?!\w)", re.IGNORECASE)
        for match in pattern.finditer(normalized):
            hits.append((match.start(), len(label), label))
    if not hits:
        return None
    hits.sort(key=lambda item: (item[0], item[1]))
    return hits[-1][2]


# 提取predicted label from generated text
def parse_generation(text: str, labels: list[str]) -> str | None:
    candidates = _extract_answer_candidates(text)
    for value in reversed(candidates):
        matched = _match_label(value, labels)
        if matched is not None:
            return matched
    return None
################################################################




TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?", re.IGNORECASE)
STOPWORDS = {
    "a",
    "an",
    "the",
    "of",
    "to",
    "and",
    "in",
    "on",
    "for",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "it",
    "this",
    "that",
    "with",
    "as",
    "at",
    "by",
    "from",
    "or",
}
_BAD_MARKERS = (
    "<|im_start|>",
    "<|im_end|>",
    "<|endoftext|>",
    "<|object_ref_start|>",
    "<|object_ref_end|>",
)


def sanitize(text: str | None) -> str:
    value = text or ""
    for marker in _BAD_MARKERS:
        value = value.replace(marker, " ")
    return value.replace("\x00", " ")


def tokenize(text: str | None) -> list[str]:
    return TOKEN_RE.findall((text or "").lower())


def clip(text: str | None, max_chars: int) -> str:
    value = sanitize(text).strip()
    if max_chars <= 0 or len(value) <= max_chars:
        return value
    suffix = " ..."
    if max_chars <= len(suffix):
        return value[:max_chars]
    return value[: max_chars - len(suffix)].rstrip() + suffix


def _inline(value: str | None, default: str = "") -> str:
    """Collapse arbitrary text to one safe header line."""
    text = " ".join(sanitize(value).split())
    return text or default


def best_passage(text: str, query: str, max_chars: int) -> str:
    body = sanitize(text).strip()
    if max_chars <= 0 or len(body) <= max_chars:
        return body
    terms = [token for token in tokenize(query) if token not in STOPWORDS] or tokenize(query)
    lowered = body.lower()
    positions = []
    for term in terms:
        found = lowered.find(term)
        if term and found >= 0:
            positions.append(found)
    if not positions:
        return body[:max_chars].rstrip()
    center = positions[len(positions) // 2]
    start = max(0, center - max_chars // 3)
    if start > 0:
        space = body.rfind(" ", 0, start)
        if space >= 0 and start - space < 40:
            start = space + 1
    return body[start : start + max_chars].rstrip()


def near_duplicate(left: str, right: str, threshold: float = 0.9) -> bool:
    left_terms = set(tokenize(left)[:300])
    right_terms = set(tokenize(right)[:300])
    if not left_terms or not right_terms:
        return False
    overlap = len(left_terms & right_terms)
    return overlap / min(len(left_terms), len(right_terms)) >= threshold


def is_leakage(claim: str, source_ids: set[str], doc_id: str, doc_text: str) -> bool:
    if doc_id in source_ids:
        return True
    claim_text = (claim or "").strip().lower()
    if len(claim_text) >= 40 and claim_text in (doc_text or "").lower():
        return True
    return near_duplicate(claim, doc_text)


def format_evidence(passages, max_chars: int = 3600) -> str:
    if not passages:
        return "none"
    blocks = []
    used = 0
    for index, passage in enumerate(passages, start=1):
        source = _inline(passage.doc_id, "unknown")
        date = _inline(passage.date, "unavailable")
        url = _inline(passage.url, "unavailable")
        aspect = _inline(passage.aspect, "n/a")
        query = _inline(passage.query)
        header = (
            f"[{index}] source={source}; date={date}; url={url}; "
            f"aspect={aspect}; query={query}"
        )
        body = sanitize(passage.text).strip()
        remaining = max_chars - used
        if remaining <= 0:
            break
        piece = header + "\n" + body
        if len(piece) > remaining:
            piece = piece[:remaining].rstrip()
        blocks.append(piece)
        used += len(piece) + 2
    return "\n\n".join(blocks) if blocks else "none"


def main():
    pass

if __name__ == '__main__':
    main()
    










