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

    def _format_one(self, user_prompt: str, add_generation_prompt: bool=True) -> str:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=self.enable_thinking,
        )

    def generate_batch(self, prompts: list[str]) -> list[str]:
        # greedy decoding
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
            # 每一个sequence分别用tokenizer进行encode，最终存在list中以便后续整体padding
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
        spans: list[tuple[int, int]],
        is_training: bool = False,
    ) -> torch.Tensor:
        if not sequences or len(sequences) != len(spans):
            raise ValueError("sequences 和 spans 必须非空且长度一致")

        pad_id = self.tokenizer.pad_token_id

        num_sequences = len(sequences)
        width = max(int(seq.numel()) for seq in sequences)

        input_ids = torch.full(
            (num_sequences, width),
            pad_id,
            dtype=torch.long,
            device=self.device
        )
        attention_mask = torch.zeros_like(input_ids)
        option_mask = torch.zeros_like(input_ids, dtype=torch.bool)

        for row, (seq, (start, end)) in enumerate(zip(sequences, spans)):
            seq = seq.reshape(-1) # (seq_len,)
            length = seq.numel()
            assert (0 < start < end <= length)

            input_ids[row, :length] = seq
            attention_mask[row, :length] = 1
            option_mask[row, start:end] = True

        if is_training:
            self.model.train()
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                return_dict=True
            )
        else:
            self.model.eval()
            with torch.inference_mode():
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                    return_dict=True
                )
                
        rows, positions = option_mask.nonzero(as_tuple=True)

        selected_logits = outputs.logits[rows, positions - 1].float() # (N, vocab_size), N 是所有被选 token 的总数量
        targets = input_ids[rows, positions] # (N,), N 是所有被选 token 的总数量

        token_log_probs = -F.cross_entropy(
            selected_logits,
            targets,
            reduction="none", # 不聚合,返回 (N,)
        )

        # 将每个 token 的 log-prob 累加回所属序列。
        scores = torch.zeros(
            num_sequences,
            dtype=torch.float32,
            device=self.device,
        )
        scores.index_add_(0, rows, token_log_probs)

        return scores # (num_cpotions,)


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


    def score_batch(
        self,
        system_prompt: str,
        batch_user_prompt: list[str],
        options: list[str],
        max_sequences: int = 8,
        is_training: bool = False,
    ) -> list[dict]:
        """
        max_sequences: 每次 forward 最多处理多少条 prompt-option 序列。
        返回顺序与 batch_user_prompt 一致。
        """
        assert options and batch_user_prompt and max_sequences > 0

        sequences: list[torch.Tensor] = []
        spans: list[tuple[int, int]] = []

        # 顺序：
        # prompt_0 + option_0, prompt_0 + option_1, ...
        # prompt_1 + option_0, prompt_1 + option_1, ...
        for user_prompt in batch_user_prompt:
            seqs, seq_spans = self._option_sequences(
                system_prompt,
                user_prompt,
                options,
            )
            sequences.extend(seqs)
            spans.extend(seq_spans)

        # 分块控制显存；分块边界可以位于同一 prompt 的选项之间。
        score_chunks = []
        for start in range(0, len(sequences), max_sequences):
            end = start + max_sequences
            chunk_sequences = sequences[start:end]
            chunk_spans = spans[start:end]
            chunk_scores = self._forward_scores(
                chunk_sequences,
                chunk_spans,
                is_training=is_training
            ) # (num_seqs,)
            score_chunks.append(chunk_scores)

        # [B*K] -> [B, K]
        scores = torch.cat(score_chunks).reshape(
            len(batch_user_prompt),
            len(options),
        )

        probs = F.softmax(scores, dim=-1) # (B,K)
        best_indices = scores.argmax(dim=-1).tolist() # list[int]

        return [
            {
                "probs": probs[i],
                "scores": scores[i],
                "best_idx": best_indices[i],
            }
            for i in range(len(batch_user_prompt))
        ]



    def count_params(self, trainable=False):
        if trainable:
            return sum(
                [p.numel() for p in self.model.parameters() if p.requires_grad]
            )
        return sum(
                [p.numel() for p in self.model.parameters()]
            )

    # def _forward_scores(
    #     self,
    #     sequences: list[torch.Tensor],
    #     spans: list[tuple[int,int]]
    # ):
    #     pad_id = self.tokenizer.pad_token_id
    #     width = max(end for _, end in spans)
    #     input_ids = torch.full(
    #         (len(sequences), width),
    #         pad_id,
    #         dtype=torch.long,
    #         device=self.device
    #     )
    #     attention_mask = torch.zeros_like(input_ids)
    #     for idx, seq in enumerate(sequences):
    #         seq_len = int(seq.numel())
    #         input_ids[idx,:seq_len] = seq
    #         attention_mask[idx, :seq_len] = 1
    #     with torch.inference_mode():
    #         output = self.model(
    #             input_ids=input_ids,
    #             attention_mask=attention_mask,
    #             use_cache=False
    #         )
    #         logits = output.logits  # (num_seq, seq_len, vocab_size)
    #         scores:list[torch.Tensor] = []
    #         for idx, (start, end) in enumerate(spans):
    #             total = None
    #             for i in range(start, end):
    #                 token_id = int(input_ids[idx,i].item())
    #                 logp = torch.log_softmax(logits[idx,i-1].float(),dim=-1)[token_id].view(-1)
    #                 total = logp if total is None else (total + logp)
    #             scores.append(total)
    #         stacked: torch.Tensor = torch.stack(scores,dim=0)
    #     return stacked.view(-1) # (num_choice,)



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
