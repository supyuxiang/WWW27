'''Local causal LM wrapper. Scoring implements LLM-as-Jev suffix likelihoods.'''

from __future__ import annotations

import logging
from contextlib import nullcontext

from src.prompts import ASSISTANT_PREFILL
from src.utils.text import softmax
from src.types import Choice, Generation, Option, Usage

logger = logging.getLogger('src.llm')


def render_chat(
    tokenizer,
    system: str,
    user: str,
    prefill: str | None,
    enable_thinking: bool = False,
    use_chat_template: bool = True,
) -> str:
    '''Render a chat prompt or plain-text prompt for Base models.'''
    if not use_chat_template:
        text = f'{system}\n\n{user}'
        if prefill:
            text += prefill
        return text

    messages = [
        {'role': 'system', 'content': system},
        {'role': 'user', 'content': user},
    ]
    try:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
    except TypeError:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    if prefill:
        text += prefill
    return text


class LocalLLM:
    '''Frozen local model. Candidate decisions do not sample a rationale.'''

    def __init__(self, model_path: str, use_chat_template: bool = True):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.use_chat_template = use_chat_template
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dtype = torch.bfloat16 if self.device.type == 'cuda' else torch.float32
        logger.info('Loading tokenizer and model from %s on %s', model_path, self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                dtype=self.dtype,
                attn_implementation='sdpa',
                local_files_only=True,
            )
        except TypeError:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=self.dtype,
                local_files_only=True,
            )
        self.model.to(self.device)
        self.model.eval()
        self._alignment_warned = False

    def encode(self, text: str) -> list[int]:
        encoded = self.tokenizer(text, add_special_tokens=False)['input_ids']
        if encoded and isinstance(encoded[0], list):
            encoded = encoded[0]
        return list(encoded)

    def _option_sequences(
        self, system: str, user: str, options: list[Option]
    ) -> tuple[list[list[int]], list[tuple[int, int]]]:
        prefix = render_chat(
            self.tokenizer, system, user, prefill=ASSISTANT_PREFILL, enable_thinking=False,
            use_chat_template=self.use_chat_template,
        )
        prefix_ids = self.encode(prefix)
        sequences = []
        spans = []
        for option in options:
            full_ids = self.encode(prefix + option.suffix)
            start = 0
            limit = min(len(prefix_ids), len(full_ids))
            while start < limit and prefix_ids[start] == full_ids[start]:
                start += 1
            if start != len(prefix_ids):
                if not self._alignment_warned:
                    logger.warning(
                        'Suffix tokenization crossed the prompt boundary for %r; scoring the unmatched tail.',
                        option.suffix,
                    )
                    self._alignment_warned = True
            if start >= len(full_ids):
                raise RuntimeError(f'Candidate suffix {option.suffix!r} produced no tokens')
            sequences.append(full_ids)
            spans.append((start, len(full_ids)))
        return sequences, spans

    def _forward_scores(
        self,
        sequences: list[list[int]],
        spans: list[tuple[int, int]],
        differentiable: bool,
    ):
        torch = self.torch
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id
        width = max(len(seq) for seq in sequences)
        input_ids = torch.full((len(sequences), width), pad_id, dtype=torch.long, device=self.device)
        attention = torch.zeros_like(input_ids)
        for row, seq in enumerate(sequences):
            input_ids[row, : len(seq)] = torch.tensor(seq, dtype=torch.long, device=self.device)
            attention[row, : len(seq)] = 1
        context = nullcontext() if differentiable else torch.inference_mode()
        with context:
            output = self.model(input_ids=input_ids, attention_mask=attention, use_cache=False)
            logits = output.logits
            scores = []
            for row, (start, end) in enumerate(spans):
                if start <= 0:
                    raise RuntimeError('Candidate suffix has no conditioning token')
                total = None
                for position in range(start, end):
                    token_id = int(input_ids[row, position].item())
                    log_prob = torch.log_softmax(logits[row, position - 1].float(), dim=-1)[token_id]
                    total = log_prob if total is None else total + log_prob
                scores.append(total)
            stacked = torch.stack(scores)
        usage = Usage(
            forward_passes=1,
            input_tokens=int(attention.sum().item()),
            output_tokens=sum(end - start for start, end in spans),
        )
        return stacked, usage

    def score_options(self, system: str, user: str, options: list[Option]) -> Choice:
        sequences, spans = self._option_sequences(system, user, options)
        scores, usage = self._forward_scores(sequences, spans, differentiable=False)
        values = [float(score.detach().cpu()) for score in scores]
        probabilities = softmax(values)
        best = max(range(len(options)), key=lambda index: (values[index], -index))
        return Choice(
            key=options[best].key,
            probs={option.key: probabilities[index] for index, option in enumerate(options)},
            scores={option.key: values[index] for index, option in enumerate(options)},
            parsed=True,
            raw_text='',
            usage=usage,
        )

    def generate(self, system: str, user: str, max_new_tokens: int) -> Generation:
        torch = self.torch
        prefix = render_chat(
            self.tokenizer, system, user, prefill=None, enable_thinking=False,
            use_chat_template=self.use_chat_template,
        )
        encoded = self.tokenizer(prefix, return_tensors='pt', add_special_tokens=False)
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        prompt_len = int(encoded['input_ids'].shape[1])
        with torch.inference_mode():
            output = self.model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        generated = output[0, prompt_len:].tolist()
        text = self.tokenizer.decode(generated, skip_special_tokens=True).strip()
        usage = Usage(
            forward_passes=1,
            input_tokens=prompt_len,
            output_tokens=len(generated),
        )
        return Generation(text=text, usage=usage)

    def generate_many(
        self, system: str, users: list[str], max_new_tokens: int
    ) -> list[Generation]:
        '''Greedily generate several independent prompts in one padded batch.'''
        if not users:
            return []
        torch = self.torch
        prefixes = [
            render_chat(
                self.tokenizer,
                system,
                user,
                prefill=None,
                enable_thinking=False,
                use_chat_template=self.use_chat_template,
            )
            for user in users
        ]
        old_padding_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = 'left'
        try:
            encoded = self.tokenizer(
                prefixes,
                return_tensors='pt',
                padding=True,
                add_special_tokens=False,
            )
        finally:
            self.tokenizer.padding_side = old_padding_side
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        prompt_width = int(encoded['input_ids'].shape[1])
        with torch.inference_mode():
            output = self.model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        generations = []
        for row in range(len(users)):
            generated = output[row, prompt_width:].tolist()
            while generated and generated[-1] == self.tokenizer.pad_token_id:
                generated.pop()
            text = self.tokenizer.decode(generated, skip_special_tokens=True).strip()
            generations.append(
                Generation(
                    text=text,
                    usage=Usage(
                        forward_passes=1,
                        input_tokens=int(encoded['attention_mask'][row].sum().item()),
                        output_tokens=len(generated),
                    ),
                )
            )
        return generations
