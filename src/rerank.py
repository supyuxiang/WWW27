from __future__ import annotations

from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.config import Passage

_INSTRUCTION = (
    "Retrieve passages that contain evidence about the specified aspect of the claim."
)
_PREFIX = (
    "<|im_start|>system\n"
    "Judge whether the Document meets the requirements based on the Query and the "
    "Instruct provided. Note that the answer can only be \"yes\" or \"no\"."
    "<|im_end|>\n<|im_start|>user\n"
)
_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def _pair(query: str, passage: Passage) -> str:
    document = passage.text.strip()
    if passage.title:
        document = f"{passage.title}\n{document}"
    return (
        f"<Instruct>: {_INSTRUCTION}\n"
        f"<Query>: {query}\n"
        f"<Document>: {document}"
    )


class Qwen3Reranker:
    def __init__(self, model_path: str, batch_size: int = 16, max_length: int = 2048):
        self.batch_size = batch_size
        self.max_length = max_length
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.device.type == 'cpu': print('using cpu!!!')
        dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            padding_side="left",
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.prefix_tokens = self.tokenizer.encode(_PREFIX, add_special_tokens=False)
        self.suffix_tokens = self.tokenizer.encode(_SUFFIX, add_special_tokens=False)
        self.true_id = self.tokenizer.convert_tokens_to_ids("yes")
        self.false_id = self.tokenizer.convert_tokens_to_ids("no")
        unk = getattr(self.tokenizer, "unk_token_id", None)
        if self.true_id is None or self.false_id is None or self.true_id == unk or self.false_id == unk:
            raise ValueError(f"{model_path} tokenizer has no single yes/no tokens")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=dtype,
            local_files_only=True,
            trust_remote_code=True,
        )
        self.model.to(self.device)
        self.model.eval()

    
    def score(
        self,
        query: str,
        passages: list[Passage]
    ) -> list[float]:
        if not passages:
            return []
        scores: list[float] = []
        for start in range(0, len(passages), self.batch_size):
            chunk = passages[start : start + self.batch_size]
            scores.extend(self._score_batch(query, chunk))
        return scores

    def rerank(
        self,
        query: str,
        passages: list[Passage],
        top_k:int
    ) -> list[tuple[Passage, float]]:
        if top_k <= 0 or not passages:
            return []
        scores = self.score(query, passages)
        order = sorted(
            range(len(passages)),
            key=lambda idx: scores[idx],
            reverse=True
        )
        return [(passages[idx], scores[idx]) for idx in order[: top_k]]

    def _score_batch(
        self,
        query: str,
        passages: list[Passage]
    ) -> list[float]:
        room = self.max_length - len(self.prefix_tokens) - len(self.suffix_tokens)
        encoded = self.tokenizer(
            [_pair(query, passage) for passage in passages],
            padding=False,
            truncation=True,
            max_length=max(room,8),
            add_special_tokens=False,
            return_attention_mask=False
        )
        for idx, token_ids in enumerate(encoded['input_ids']):
            encoded['input_ids'][idx] = self.prefix_tokens + list(token_ids) + self.suffix_tokens
        batch = self.tokenizer.pad(
            encoded,
            padding=True,
            return_tensors='pt'
        )
        batch = {k:v.to(self.device) for k,v in batch.items()}
        with torch.inference_mode():
            logits = self.model(**batch).logits[:, -1, :]
        true = logits[:, self.true_id]
        false = logits[:, self.false_id]
        probs = torch.softmax(
            torch.stack([false, true], dim=1).float(), dim=1
        )[:,1]
        return [float(item) for item in probs]


def main():
    reranker = Qwen3Reranker(model_path='/root/autodl-tmp/models/Qwen3-Reranker-0.6B')
    breakpoint()

if __name__ == '__main__':
    main()
    