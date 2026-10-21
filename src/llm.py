from __future__ import annotations

import logging
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

import sys
from pathlib import Path

ROOT = Path(__file__).parents[1].resolve()
sys.path.insert(0,str(ROOT))

# from src.prompts import ASSISTANT_PREFILL

from src.utils import (
    load_tokenizer
)

ASSISTANT_PREFILL = "\nBest answer: ["

logger = logging.getLogger(__file__)



def format_prefix(
    tokenizer: AutoTokenizer,
    system_prompt:str,
    user_prompt:str,
    prefill: str | None = None,
    enable_thinking: bool = False
) -> str:
    msg = [
        {'role':'system','content':system_prompt},
        {'role':'user', 'content':user_prompt},
    ]
    kwargs = dict(
        tokenize=False,
        add_generation_prompt=True,
    )
    if enable_thinking:
        kwargs['enable_thinking'] = True
    formatted = tokenizer.apply_chat_template(
        msg,
        **kwargs,
    )
    if prefill:
        formatted += prefill
    return formatted


class LocalLLM:
    def __init__(self, model_path: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.device.type == 'cpu': print('using cpu!!!')
        self.dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32

        self.tokenizer = load_tokenizer(model_path)
        self.model_path = model_path 
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=self.dtype,
            local_files_only=True,
        )
        self.model.to(self.device)
        self.model.eval()

    def encode(self, text: str) -> list[int]:
        encoded = self.tokenizer(
            text,
            add_special_tokens=False,
            return_tensors='pt',
        )['input_ids'].to(self.device)
        return encoded # torch.Tensor

    def _option_sequences(
        self,
        system_prompt: str,
        user_prompt:str,
        options: list[str],
    ) -> tuple[list[list[int]], list[tuple[int,int]]]:
        prefix = format_prefix(
            self.tokenizer,
            system_prompt,
            user_prompt,
            prefill=ASSISTANT_PREFILL,
        )
        prefix_ids: torch.Tensor = self.encode(prefix).to(torch.long).view(1,-1)
        start = int(prefix_ids.numel())
        sequences:list[torch.Tensor] = []
        spans:list[tuple[int,int]] = []
        for option in options:
            full_ids: torch.Tensor = self.encode(prefix + option).to(torch.long).view(1,-1)
            end = int(full_ids.numel())
            sequences.append(full_ids)
            spans.append(
                (start, end)
            )
        return sequences, spans

    def _forward_scores(
        self,
        sequences: list[torch.Tensor],
        spans: list[tuple[int,int]]
    ):
        pad_id = self.tokenizer.pad_token_id
        width = max(end for _, end in spans)
        input_ids = torch.full(
            (len(sequences), width),
            pad_id,
            dtype=torch.long,
            device=self.device
        )
        attention_mask = torch.zeros_like(input_ids)
        for idx, seq in enumerate(sequences):
            seq_len = int(seq.numel())
            input_ids[idx,:seq_len] = seq
            attention_mask[idx, :seq_len] = 1
        with torch.inference_mode():
            output = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cahe=False
            )
            logits = output.logits  # (num_seq, seq_len, vocab_size)
            scores:list[torch.Tensor] = []
            for idx, (start, end) in enumerate(spans):
                total = None
                for i in range(start, end):
                    token_id = int(input_ids[idx,i].item())
                    logp = torch.log_softmax(logits[idx,i-1].float(),dim=-1)[token_id].view(-1)
                    total = logp if total is None else (total + logp)
                scores.append(total)
            stacked: torch.Tensor = torch.stack(scores,dim=0)
        return stacked.view(-1) # (num_choice,)

    def score_options(
        self,
        system_prompt: str,
        user_prompt: str,
        options: list[str]
    ) -> dict:
        sequences, spans = self._option_sequences(system_prompt, user_prompt, options)
        scores = self._forward_scores(sequences, spans)
        prob_distribution = F.softmax(scores,dim=-1)
        best_idx = int(torch.argmax(prob_distribution,dim=-1))
        out = dict(
            probs=prob_distribution,
            scores=scores,
            best_idx=best_idx
        )
        return out

def main():
    model = LocalLLM(model_path = '/root/autodl-tmp/models/Qwen2.5-7B-Instruct')
    system_prompt = '你是一个新闻的判断者，善于判断新闻的正确性'
    user_prompt = '新闻：美国总统特朗普出生于2024年12月11日。 \n这个新闻是对的吗？给出你的选择: [1]True,[2]False'
    options=['1','2']
    out = model.score_options(
        system_prompt,
        user_prompt,
        options
    )
    
    breakpoint()


#########TODO

from abc import ABC, abstractmethod

class Jev_Base(ABC):
    
    @abstractmethod
    def encode(self):
        pass

    @abstractmethod
    def score_options(self):
        pass


class Laya(Jev_Base):
    def __init__(
        self,
    ):
        pass
    def encode(self):
        pass
    def score_options(self):
        pass


class Kev(Jev_Base):
    def encode(self):
        pass
    def score_options(self):
        pass


if __name__ == '__main__':
    main()
