'''Claim and corpus loading. Domain tags may be supplied or assigned by keyword.'''

from __future__ import annotations

import json
import re
from pathlib import Path

from src.aspects import ASPECT_BY_KEY
from src.types import Document, Example

_HEALTH = re.compile(
    r'covid|vaccine|virus|cancer|hospital|doctor|disease|health|medical|patient|'
    r'flu|epidemic|symptom|infection|clinic|diabetes|physician|mortality',
    re.IGNORECASE,
)
_SCIENCE = re.compile(
    r'scientist|researcher|nasa|climate|physics|biology|genome|quantum|fossil|'
    r'astronomy|laboratory|scientific|astronomer|physicist|chemist|geologist',
    re.IGNORECASE,
)
_SOCIAL = re.compile(
    r'facebook|twitter|instagram|viral|youtube|social media|tweet|tiktok|reddit|influencer|hashtag',
    re.IGNORECASE,
)
_POLITICAL = re.compile(
    r'election|president|congress|senate|government|minister|democrat|republican|'
    r'parliament|white house|campaign|senator|governor|trump|obama|biden|putin|'
    r'nato|military|legislation|diplomat',
    re.IGNORECASE,
)


def assign_domain(text: str) -> str:
    '''Keyword domain tag for local news text that has no gold domain field.

    Health is applied first, then science, then social media, then politics.
    Remaining news is labeled political because this local corpus is mostly
    political reporting. Records that already contain `domain` keep it.
    '''
    if _HEALTH.search(text or ''):
        return 'health'
    if _SCIENCE.search(text or ''):
        return 'scientific'
    if _SOCIAL.search(text or ''):
        return 'social_media'
    return 'political'


def normalize_label(value) -> str:
    if isinstance(value, bool) or value is None:
        raise ValueError(f'Unrecognized label: {value!r}')
    if isinstance(value, (int, float)):
        if int(value) == 1:
            return 'real'
        if int(value) == 0:
            return 'fake'
        raise ValueError(f'Unrecognized label: {value!r}')
    text = str(value).strip().lower()
    if text in {'real', 'true', 'mostly-true', 'mostly_true', 'reliable'}:
        return 'real'
    if text in {'fake', 'false', 'mostly-false', 'mostly_false', 'unreliable', 'rumor', 'rumour'}:
        return 'fake'
    raise ValueError(f'Unrecognized label: {value!r}')


def make_claim(title: str, text: str, mode: str, lead_chars: int = 500) -> str:
    title = (title or '').strip()
    body = (text or '').strip()
    if mode == 'headline':
        return title or body
    if mode == 'full':
        return f'{title}\n{body}'.strip() if title and body else (title or body)
    lead = body[:lead_chars].strip()
    if title and lead:
        return f'{title}\n{lead}'
    return title or lead


def _optional_bool(value):
    if value is None or value == '':
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {'1', 'true', 'yes', 'retrieve'}:
        return True
    if text in {'0', 'false', 'no', 'skip'}:
        return False
    raise ValueError(f'Unrecognized needs_retrieval value: {value!r}')


def example_from_row(row: dict, claim_mode: str = 'stored') -> Example:
    title = row.get('title') or ''
    text = row.get('text') or ''
    if claim_mode == 'stored' and row.get('claim'):
        claim = str(row['claim'])
    else:
        mode = 'headline_lead' if claim_mode == 'stored' else claim_mode
        claim = make_claim(title, text, mode)
    if not claim.strip():
        raise ValueError('Example has an empty claim')
    label_value = row['label'] if 'label' in row else row.get('label_hf')
    label = normalize_label(label_value)
    domain = (row.get('domain') or '').strip() or assign_domain(f'{title}\n{text}\n{claim}')
    example_id = str(row.get('id') or row.get('doc_id'))
    source_id = str(row.get('source_id') or example_id)
    gold_aspects = row.get('gold_aspects')
    if gold_aspects is not None:
        gold_aspects = [str(item) for item in gold_aspects if str(item) in ASPECT_BY_KEY]
    return Example(
        id=example_id,
        claim=claim.strip(),
        label=label,
        domain=domain,
        title=title,
        text=text,
        source_id=source_id,
        needs_retrieval=_optional_bool(row.get('needs_retrieval')),
        gold_aspects=gold_aspects,
    )


def read_jsonl(path: str | Path) -> list[dict]:
    rows = []
    with open(path, 'r', encoding='utf-8') as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f'{path}:{line_number} is not valid JSON') from exc
    return rows


def iter_jsonl_paths(path: str | Path) -> list[Path]:
    target = Path(path)
    if target.is_dir():
        files = sorted(target.glob('*.jsonl'))
        if not files:
            raise FileNotFoundError(f'No jsonl files in {target}')
        return files
    if not target.exists():
        raise FileNotFoundError(target)
    return [target]


def load_examples(
    path: str | Path,
    claim_mode: str = 'stored',
    domain: str = '',
    offset: int = 0,
    limit: int = 0,
) -> list[Example]:
    examples: list[Example] = []
    for file_path in iter_jsonl_paths(path):
        fallback_domain = file_path.stem if Path(path).is_dir() else ''
        for row in read_jsonl(file_path):
            if fallback_domain and not row.get('domain'):
                row = dict(row)
                row['domain'] = fallback_domain
            example = example_from_row(row, claim_mode=claim_mode)
            if domain and example.domain != domain:
                continue
            examples.append(example)
    if offset:
        examples = examples[offset:]
    if limit and limit > 0:
        examples = examples[:limit]
    return examples


def load_corpus(path: str | Path) -> list[Document]:
    documents = []
    for file_path in iter_jsonl_paths(path):
        for row in read_jsonl(file_path):
            doc_id = str(row.get('doc_id') or row.get('id'))
            title = row.get('title') or ''
            text = row.get('text') or row.get('contents') or ''
            if title and text and not str(text).startswith(str(title)):
                body = f'{title}\n{text}'
            else:
                body = text or title
            documents.append(
                Document(
                    doc_id=doc_id,
                    text=body,
                    title=title,
                    url=row.get('url') or '',
                    date=str(row.get('date') or row.get('publication_date') or ''),
                    domain=row.get('domain') or '',
                )
            )
    return documents


def corpus_from_examples(examples: list[Example]) -> list[Document]:
    documents = []
    for example in examples:
        body = example.text or example.claim
        if example.title and body and not body.startswith(example.title):
            body = f'{example.title}\n{body}'
        documents.append(
            Document(
                doc_id=example.source_id or example.id,
                text=body,
                title=example.title,
                domain=example.domain,
            )
        )
    return documents


def write_jsonl(path: str | Path, rows: list[dict]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open('w', encoding='utf-8') as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + '\n')


def prepare_local_news(
    source: str,
    claims_path: str,
    corpus_path: str,
    claim_mode: str = 'headline_lead',
) -> dict[str, int]:
    '''Convert a local jsonl news file into claim and corpus files.'''
    claims = []
    corpus = []
    counts: dict[str, int] = {}
    for row in read_jsonl(source):
        title = row.get('title') or ''
        text = row.get('text') or ''
        claim = make_claim(title, text, claim_mode)
        domain = (row.get('domain') or '').strip() or assign_domain(f'{title}\n{text[:800]}')
        label = normalize_label(row['label'] if 'label' in row else row.get('label_hf'))
        example_id = str(row.get('id') or f'row-{len(claims)}')
        counts[domain] = counts.get(domain, 0) + 1
        counts[f'label:{label}'] = counts.get(f'label:{label}', 0) + 1
        claims.append(
            {
                'id': example_id,
                'claim': claim,
                'label': label,
                'domain': domain,
                'title': title,
                'text': text,
                'source_id': example_id,
            }
        )
        corpus.append(
            {
                'doc_id': example_id,
                'title': title,
                'text': text,
                'url': '',
                'date': '',
                'domain': domain,
            }
        )
    write_jsonl(claims_path, claims)
    write_jsonl(corpus_path, corpus)
    counts['n'] = len(claims)
    return counts
