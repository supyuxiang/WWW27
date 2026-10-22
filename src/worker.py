from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

import os
import sys
from pathlib import Path

ROOT = Path(__file__).parents[1].resolve()
sys.path.insert(0,str(ROOT))

from src.prompts import ASSISTANT_PREFILL

from src.utils import (
    load_tokenizer
)
from src.prompts import format_prefix4jev, SYSTEM_PROMPT

from vllm import LLM, SamplingParams

os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")


logger = logging.getLogger(__file__)


class VLLMGenerator:
    def __init__(
        self,
        model_path: str,
        tensor_parallel_size: int = torch.cuda.device_count(),
        gpu_memory_utilization: float = 0.92,
        max_model_len: int = 8192,
        max_new_tokens: int = 1024,
        enable_thinking: bool = False,
    ):
        self.enable_thinking = enable_thinking
        self.llm = LLM(
            model=model_path,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            dtype="bfloat16",
        )
        self.llm_name = str(model_path).split("/")[-1]
        self.tokenizer = load_tokenizer(model_path, is_training=False)
        self.sp = SamplingParams(
            temperature=0.0,
            max_tokens=max_new_tokens,
            skip_special_tokens=True,
        )

    def _format_one(self, user_prompt: str) -> str:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=self.enable_thinking,
        )

    def generate_batch(self, prompts: list[str]) -> list[str]:
        formatted = [self._format_one(prompt) for prompt in prompts]
        outputs = self.llm.generate(formatted, self.sp)
        return [output.outputs[0].text.strip() for output in outputs]



################    base     #################

class Base(ABC):
    
    @abstractmethod
    def encode(self, text:str) -> Any:
        pass

    @abstractmethod
    def score_options(
        self,
        system_prompt: str,
        user_prompt: str,
        options: list[str]
    ) -> Any:
        pass


#############################################

class LLM_as_Jev(Base):

    _cls = 'llm_as_jev'
    
    def __init__(
        self,
        model_path: str,
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        dtype=torch.bfloat16
    ):
        super().__init__()
        self.device = device
        if self.device.type == 'cpu': print('using cpu!!!')
        self.dtype = dtype

        self.tokenizer = load_tokenizer(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=self.dtype,
            local_files_only=True,
        )
        self.model.to(self.device)
        self.model.eval()

    def encode(self, text: str) -> torch.Tensor:
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
        prefix = format_prefix4jev(
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
                use_cache=False
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


    # TODO：主要优化
    def score_batch(
        self,
        system_prompt: str,
        batch_user_prompt: list[str],
        options: list[str]
    ) -> list[dict]:
        num_options = len(options)
        all_sequences = []   # 所有选项的完整序列（tensor）
        all_spans = []       # 每个选项在序列中的起止位置 (start, end)

        # 1. 收集所有 prompt 的选项序列和 spans
        for user_prompt in batch_user_prompt:
            sequences, spans = self._option_sequences(system_prompt, user_prompt, options)
            all_sequences.extend(sequences)   # sequences 是 list[tensor]
            all_spans.extend(spans)           # spans 是 list[tuple]

        # 2. 一次前向传播，得到所有选项的分数
        scores = self._forward_scores(all_sequences, all_spans)  # shape: [total_options]

        # 3. 按 prompt 分组
        batch_size = len(batch_user_prompt)
        scores = scores.view(batch_size, num_options)            # [batch, num_options]
        probs = F.softmax(scores, dim=-1)                        # [batch, num_options]
        best_indices = torch.argmax(probs, dim=-1)               # [batch]

        # 4. summary
        results = []
        for i in range(batch_size):
            results.append({
                'probs': probs[i],               # Tensor, shape [num_options]
                'scores': scores[i],             # Tensor, shape [num_options]
                'best_idx': best_indices[i].item()
            })
        return results



    def count_params(self, trainable=False):
        if trainable:
            return sum(
                [p.numel() for p in self.model.parameters() if p.requires_grad]
            )
        return sum(
                [p.numel() for p in self.model.parameters()]
            )


def main_test():
    model = LLM_as_Jev(model_path = '/root/autodl-tmp/models/Qwen2.5-7B-Instruct')
    system_prompt = '你是一个新闻的判断者，善于判断新闻的正确性'
    user_prompt = '新闻：美国总统特朗普出生于2024年12月11日。 \n这个新闻是对的吗？给出你的选择: [1]True,[2]False'
    options=['1','2']
    out = model.score_options(
        system_prompt,
        user_prompt,
        options
    )
    
    breakpoint()



#########TODO?

class Laya(Base):
    _cls = 'laya'

    def __init__(
        self,
    ):
        super().__init__()
        pass
    def encode(self):
        pass
    def score_options(self):
        pass


class Kev(Base):
    _cls = 'kev'

    def __init__(
        self,
    ):
        super().__init__()
        pass
    def encode(self):
        pass
    def score_options(self):
        pass


if __name__ == '__main__':
    main_test()
