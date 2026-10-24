'''GProofT official mechanism adapted to native three-way offline tasks.'''

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(CODE_ROOT))
sys.path.insert(0,str(CODE_ROOT / 'vendor' / 'www27'))

from common.threeway_support import (
    atomic_json,
    config_payload,
    finalize_output,
    gold_label,
    initialize_output,
    load_benchmark,
    make_retriever,
    prepare_retriever,
)
from baselines.run_gprooft_offline import (
    ANSWER_PROMPT,
    JUSTIFICATION_PROMPT,
    OFFICIAL_COMMIT,
    OFFICIAL_REPO,
    QUESTION_PROMPT,
    SPLIT_PROMPT,
    SYSTEM_PROMPT,
    VERDICT_PROMPT,
    aggregate_verdict,
    clean_generation,
    parse_subclaims,
    parse_verdict,
    parse_yes_no,
)
from src.llm import LocalLLM

METHOD = 'gprooft_adapted'


def project(benchmark: str, native: str) -> tuple[str | None, str]:
    if benchmark == 'cfever':
        mapping = {
            'supported': 'supports',
            'refuted': 'refutes',
            'Conflicting Evidence/Cherrypicking': 'refutes',
            'not enough evidence': 'not_enough_info',
        }
        reason = {
            'supported': 'native_supported',
            'refuted': 'native_refuted',
            'Conflicting Evidence/Cherrypicking': 'mixed_claim_not_fully_supported',
            'not enough evidence': 'native_nei',
        }
        return mapping.get(native), reason.get(native, 'unmapped')
    mapping = {
        'supported': 'true',
        'refuted': 'false',
        'Conflicting Evidence/Cherrypicking': 'conflicting',
    }
    prediction = mapping.get(native)
    return prediction, 'exact_native_projection' if prediction else 'native_nei_unmapped'


def hit_dict(hit):
    return {'doc_id': hit.doc_id, 'title': hit.title, 'score': hit.score, 'text': hit.text}


def run_one(row, benchmark, module, llm, retriever, args):
    started = time.perf_counter()
    input_tokens = output_tokens = parse_failures = search_calls = 0
    generations = []
    qa_rows = []
    prepare_retriever(module, retriever, row)

    split = llm.generate(SYSTEM_PROMPT, SPLIT_PROMPT.format(claim=row['claim']), args.split_max_new_tokens)
    input_tokens += split.usage.input_tokens
    output_tokens += split.usage.output_tokens
    subclaims = parse_subclaims(split.text, row['claim'], args.max_subclaims)
    generations.append({'stage': 'split_claim', 'raw': split.text, 'parsed': subclaims})

    question_outputs = llm.generate_many(
        SYSTEM_PROMPT,
        [QUESTION_PROMPT.format(subclaim=item) for item in subclaims],
        args.question_max_new_tokens,
    )
    questions = []
    for subclaim, generated in zip(subclaims, question_outputs):
        input_tokens += generated.usage.input_tokens
        output_tokens += generated.usage.output_tokens
        question = clean_generation(generated.text).splitlines()[0].strip()
        if not question:
            parse_failures += 1
            question = subclaim
        questions.append(question)
        generations.append({'stage': 'generate_binary', 'subclaim': subclaim, 'raw': generated.text, 'parsed': question})

    for subclaim, question in zip(subclaims, questions):
        if search_calls >= args.search_cap:
            qa_rows.append({'subclaim': subclaim, 'question': question, 'skipped': 'search_cap', 'hits': [], 'answers': [], 'answer': None})
            continue
        retrieval_started = time.perf_counter()
        hits = retriever.search(question, args.top_k, set(), args.max_passage_chars)
        search_calls += 1
        answer_outputs = llm.generate_many(
            SYSTEM_PROMPT,
            [ANSWER_PROMPT.format(question=question, snippet=hit.text) for hit in hits],
            args.answer_max_new_tokens,
        )
        votes = []
        answers = []
        for hit, generated in zip(hits, answer_outputs):
            input_tokens += generated.usage.input_tokens
            output_tokens += generated.usage.output_tokens
            parsed = parse_yes_no(generated.text)
            if parsed is None:
                parse_failures += 1
            else:
                votes.append(parsed)
            answers.append({'doc_id': hit.doc_id, 'raw': generated.text, 'parsed': parsed})
        majority = ('Yes' if votes.count('Yes') >= votes.count('No') else 'No') if votes else None
        qa_rows.append({
            'subclaim': subclaim, 'question': question,
            'retrieval_latency_s': time.perf_counter() - retrieval_started,
            'hits': [hit_dict(hit) for hit in hits], 'answers': answers, 'answer': majority,
        })

    judged = [item for item in qa_rows if item.get('answer') in {'Yes', 'No'}]
    verdict_outputs = llm.generate_many(
        SYSTEM_PROMPT,
        [VERDICT_PROMPT.format(claim=row['claim'], question=item['question'], answer=item['answer']) for item in judged],
        args.verdict_max_new_tokens,
    )
    verdicts = []
    for item, generated in zip(judged, verdict_outputs):
        input_tokens += generated.usage.input_tokens
        output_tokens += generated.usage.output_tokens
        verdict = parse_verdict(generated.text)
        verdicts.append(verdict)
        item['verdict_raw'] = generated.text
        item['verdict'] = verdict
        generations.append({'stage': 'qa_verdict', 'question': item['question'], 'raw': generated.text, 'parsed': verdict})

    native = aggregate_verdict(verdicts)
    prediction, projection_reason = project(benchmark, native)
    justification = ''
    if judged:
        qa_text = ''.join(f'Question {index}: {item['question']} Answer {index}: {item['answer']} ' for index, item in enumerate(judged, 1))
        generated = llm.generate(
            SYSTEM_PROMPT,
            JUSTIFICATION_PROMPT.format(claim=row['claim'], qa_text=qa_text, verdict=native),
            args.justification_max_new_tokens,
        )
        input_tokens += generated.usage.input_tokens
        output_tokens += generated.usage.output_tokens
        justification = clean_generation(generated.text).split('.', 1)[0].strip()
        generations.append({'stage': 'justification', 'raw': generated.text, 'parsed': justification})

    return {
        'id': str(row['id']), 'claim': row['claim'], 'gold': gold_label(module, row),
        'prediction': prediction, 'correct': prediction == gold_label(module, row), 'method': METHOD,
        'native_verdict': native, 'projection_reason': projection_reason,
        'subclaims': subclaims, 'qa_pairs': qa_rows, 'justification': justification,
        'generations': generations, 'search_calls': search_calls,
        'input_tokens': input_tokens, 'output_tokens': output_tokens,
        'parse_failures': parse_failures, 'latency_s': time.perf_counter() - started,
        'error': None if prediction else 'GProofT native NEI cannot be mapped to QuanTemp++ three-way labels',
        **({'split_index': row['__split_index']} if '__split_index' in row else {}),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--benchmark', choices=['cfever', 'quantemp_plus'], required=True)
    parser.add_argument('--split-file', required=True)
    parser.add_argument('--index', required=True)
    parser.add_argument('--model', default='/root/autodl-tmp/models/Qwen2.5-3B-Instruct')
    parser.add_argument('--output-root', required=True)
    parser.add_argument('--limit', type=int, default=20)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--search-cap', type=int, default=3)
    parser.add_argument('--top-k', type=int, default=3)
    parser.add_argument('--max-subclaims', type=int, default=3)
    parser.add_argument('--max-passage-chars', type=int, default=800)
    parser.add_argument('--split-max-new-tokens', type=int, default=192)
    parser.add_argument('--question-max-new-tokens', type=int, default=96)
    parser.add_argument('--answer-max-new-tokens', type=int, default=16)
    parser.add_argument('--verdict-max-new-tokens', type=int, default=64)
    parser.add_argument('--justification-max-new-tokens', type=int, default=128)
    args = parser.parse_args()

    module, selected = load_benchmark(args.benchmark, args.split_file, args.limit, args.seed)
    source = Path(__file__)
    config = config_payload(args, module, selected, METHOD, source, {
        'paper': 'GProofT (FEVER 2024)', 'official_code': OFFICIAL_REPO,
        'official_commit': OFFICIAL_COMMIT,
        'adaptation': 'Hosted Azure/Google services are replaced by Qwen2.5-3B and fixed offline retrieval',
        'search_cap': args.search_cap, 'top_k': args.top_k, 'max_subclaims': args.max_subclaims,
        'generation_limits': {'split': args.split_max_new_tokens, 'question': args.question_max_new_tokens, 'answer': args.answer_max_new_tokens, 'verdict': args.verdict_max_new_tokens, 'justification': args.justification_max_new_tokens},
        'invalid_policy': 'Unmappable native NEI remains invalid and counts wrong in full-scope Accuracy/Macro-F1',
    })
    output = Path(args.output_root)
    initialize_output(output, config, selected)
    llm = LocalLLM(args.model, use_chat_template=True)
    retriever = make_retriever(module, args.index)
    rows = []
    for position, row in enumerate(selected, 1):
        destination = output / 'predictions' / METHOD / f'{row['id']}.json'
        if destination.exists():
            previous = json.loads(destination.read_text(encoding='utf-8'))
            if previous.get('config_fingerprint') == config['config_fingerprint']:
                rows.append(previous)
                continue
        try:
            result = run_one(row, args.benchmark, module, llm, retriever, args)
        except Exception as exc:
            result = {'id': str(row['id']), 'claim': row['claim'], 'gold': gold_label(module, row), 'prediction': None, 'method': METHOD, 'error': f'{type(exc).__name__}: {exc}'}
        result['config_fingerprint'] = config['config_fingerprint']
        atomic_json(destination, result)
        rows.append(result)
        print(json.dumps({'method': METHOD, 'benchmark': args.benchmark, 'processed': position, 'total': len(selected), 'invalid_or_failed': sum(bool(item.get('error')) for item in rows)}), flush=True)
    retriever.close()
    summary = finalize_output(output, config, rows, module.LABELS)
    print(json.dumps({'event': 'complete', 'summary': summary}, ensure_ascii=False), flush=True)
    # Native NEI is a valid completed method state even though it lowers coverage.
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
