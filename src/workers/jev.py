from __future__ import annotations

import os
import sys
from pathlib import Path
import logging
from abc import ABC, abstractmethod
from typing import Any
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

ROOT = Path(__file__).parents[2].resolve()
sys.path.insert(0,str(ROOT))

from src.utils import (
    load_tokenizer,
)
from src.prompts import (
    format_prefix4jev,
    ASSISTANT_PREFILL
)

logger = logging.getLogger(__file__)



################    base     #################

class JevBase(ABC):

    @abstractmethod
    def score_options(
        self,
        system_prompt: str,
        user_prompt: str,
        options: list[str]
    ) -> dict[str,Any]:
        pass


#############################################

class LLM_as_Jev(JevBase):
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

    def _option_token_ids(self, options: list[str]) -> torch.Tensor:
        token_ids: list[int] = []
        for option in options:
            ids = self.tokenizer.encode(option, add_special_tokens=False) # tensor([[token_id]]), torch.Size([1,1])
            assert len(ids) == 1, ids
            token_ids.append(ids[0]) # torch.Size([1])
        return torch.tensor(token_ids, dtype=torch.long, device=self.device) # torch.Size([num_token_ids])

    def score_options(
        self,
        system_prompt: str,
        user_prompt: str,
        options: list[str]
    ) -> dict:
        assert options, options
        
        prefix = format_prefix4jev(
            self.tokenizer,
            system_prompt,
            user_prompt,
            prefill=ASSISTANT_PREFILL,
        )
        prefix_ids = self.encode(prefix).to(torch.long).view(1, -1)
        option_ids = self._option_token_ids(options)
        with torch.inference_mode():
            logits = self.model(
                input_ids=prefix_ids,
                use_cache=False,
                return_dict=True,
            ).logits[:, -1, :].float() # 1, vocab_size
        scores = F.log_softmax(logits, dim=-1)[0, option_ids] # (num_options,)
        probs = F.softmax(scores, dim=-1) # (num_options,)
        return dict(
            probs=probs,
            scores=scores,
            best_idx =int(torch.argmax(probs)),
        )

    def decode(self, tokens: list[int]) -> str:
        return self.tokenizer.decode(tokens, skip_special_tokens=True)

    def count_params(self, trainable=False):
        if trainable:
            return sum(
                [p.numel() for p in self.model.parameters() if p.requires_grad]
            )
        return sum(
                [p.numel() for p in self.model.parameters()]
            )


    # def get_sequences(
    #     self,
    #     system_prompt: str,
    #     user_prompt:str,
    #     options: list[str],
    # ) -> tuple[list[list[int]], list[tuple[int,int]]]:
    #     prefix = format_prefix4jev(
    #         self.tokenizer,
    #         system_prompt,
    #         user_prompt,
    #         prefill=ASSISTANT_PREFILL,
    #     )
    #     prefix_ids: torch.Tensor = self.encode(prefix).to(torch.long).view(1,-1)
    #     start: int = prefix_ids.numel()
    #     sequences : list[torch.Tensor] = []
    #     spans : list[tuple[int,int]] = []
    #     for option in options:
    #         # 每一个sequence分别用tokenizer进行encode，最终存在list中以便后续整体padding
    #         full_seq = prefix + option
    #         full_ids: torch.Tensor = self.encode(full_seq).to(torch.long).view(1,-1)
    #         sequences.append(full_ids)
    #         end: int = full_ids.numel()
    #         spans.append(
    #             (start, end)
    #         )
    #     return sequences, spans


    # def _forward_scores(
    #     self,
    #     sequences: list[torch.Tensor],
    #     spans: list[tuple[int, int]],
    #     is_training: bool = False,
    # ) -> torch.Tensor:
    #     assert sequences and len(sequences) == len(spans), f'{sequences}_{len(sequences)}_{len(spans)}'

    #     num_sequences = len(sequences)
    #     max_len = max(seq.numel() for seq in sequences)
    #     pad_id = self.tokenizer.pad_token_id; assert pad_id is not None
        
    #     input_ids = torch.full(
    #         (num_sequences, max_len),
    #         pad_id,
    #         dtype=torch.long,
    #         device=self.device
    #     )
    #     attention_mask = torch.zeros_like(input_ids)
    #     option_mask = torch.zeros_like(input_ids, dtype=torch.bool)

    #     for row, (seq, (start, end)) in enumerate(zip(sequences, spans)):
    #         seq = seq.reshape(-1) # (seq_len,)
    #         length = seq.numel()
    #         assert (0 < start < end <= length)

    #         input_ids[row, :length] = seq
    #         attention_mask[row, :length] = 1
    #         option_mask[row, start:end] = True

    #     if is_training:
    #         self.model.train()
    #         outputs = self.model(
    #             input_ids=input_ids,
    #             attention_mask=attention_mask,
    #             use_cache=False,
    #             return_dict=True
    #         )
    #     else:
    #         self.model.eval()
    #         with torch.inference_mode():
    #             outputs = self.model(
    #                 input_ids=input_ids,
    #                 attention_mask=attention_mask,
    #                 use_cache=False,
    #                 return_dict=True
    #             )
                
    #     rows, positions = option_mask.nonzero(as_tuple=True)

    #     selected_logits = outputs.logits[rows, positions - 1].float() # (N, vocab_size), N 是所有被选 token 的总数量
    #     targets = input_ids[rows, positions] # (N,), N 是所有被选 token 的总数量

    #     token_log_probs = -F.cross_entropy(
    #         selected_logits,
    #         targets,
    #         reduction="none", # 不聚合,返回 (N,)
    #     )

    #     # 将每个 token 的 log-prob 累加回所属序列。
    #     scores = torch.zeros(
    #         num_sequences,
    #         dtype=torch.float32,
    #         device=self.device,
    #     )
    #     scores.index_add_(dim=0, index=rows, source=token_log_probs)

    #     return scores # (num_cpotions,)


    # def score_batch(
    #     self,
    #     system_prompt: str,
    #     batch_user_prompt: list[str],
    #     options: list[str],
    #     max_sequences: int = 8,
    #     is_training: bool = False,
    # ) -> list[dict]:
    #     """
    #     max_sequences: 每次 forward 最多处理多少条 prompt-option 序列。
    #     返回顺序与 batch_user_prompt 一致。
    #     """
    #     assert options and batch_user_prompt and max_sequences > 0

    #     sequences: list[torch.Tensor] = []
    #     spans: list[tuple[int, int]] = []

    #     # 顺序：
    #     # prompt_0 + option_0, prompt_0 + option_1, ...
    #     # prompt_1 + option_0, prompt_1 + option_1, ...
    #     for user_prompt in batch_user_prompt:
    #         seqs, seq_spans = self.get_sequences(
    #             system_prompt,
    #             user_prompt,
    #             options,
    #         )
    #         sequences.extend(seqs)
    #         spans.extend(seq_spans)

    #     # 分块控制显存；分块边界可以位于同一 prompt 的选项之间。
    #     score_chunks = []
    #     for start in range(0, len(sequences), max_sequences):
    #         end = start + max_sequences
    #         chunk_sequences = sequences[start:end]
    #         chunk_spans = spans[start:end]
    #         chunk_scores = self._forward_scores(
    #             chunk_sequences,
    #             chunk_spans,
    #             is_training=is_training
    #         ) # (num_seqs,)
    #         score_chunks.append(chunk_scores)

    #     # [B*K] -> [B, K]
    #     scores = torch.cat(score_chunks).reshape(
    #         len(batch_user_prompt),
    #         len(options),
    #     )

    #     probs = F.softmax(scores, dim=-1) # (B,K)
    #     best_indices = scores.argmax(dim=-1).tolist() # list[int]

    #     return [
    #         {
    #             "probs": probs[i],
    #             "scores": scores[i],
    #             "best_idx": best_indices[i],
    #         }
    #         for i in range(len(batch_user_prompt))
    #     ]



    # def _forward_scores_vanilla(
    #     self,
    #     sequences: list[torch.Tensor],
    #     spans: list[tuple[int,int]],
    #     is_training: bool = False
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

    #     if is_training:
    #         self.model.train()
    #         output = self.model(
    #             input_ids=input_ids,
    #             attention_mask=attention_mask,
    #             use_cache=False,
    #         )
    #     else:
    #         with torch.inference_mode():
    #             self.model.eval()
    #             output = self.model(
    #                 input_ids=input_ids,
    #                 attention_mask=attention_mask,
    #                 use_cache=False
    #             )
                
    #     logits = output.logits  # (num_seq, seq_len, vocab_size)
    #     scores:list[torch.Tensor] = []
    #     for idx, (start, end) in enumerate(spans):
    #         total = None
    #         for i in range(start, end):
    #             token_id = int(input_ids[idx,i].item())
    #             logp = torch.log_softmax(logits[idx,i-1].float(),dim=-1)[token_id].view(-1)
    #             total = logp if total is None else (total + logp)
    #             scores.append(total)
    #         stacked: torch.Tensor = torch.stack(scores,dim=0)
    #     return stacked.view(-1) # (num_choice,)



class API_Jev(JevBase):
    def __init__(
        self,
    ):
        super().__init__()
        pass

    def score_options(
        self,
    ):
        pass





###TODO?

class Laya(JevBase):
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


class Kev(JevBase):
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
    def score_batch(self):
        pass





def main():
    model = LLM_as_Jev(model_path = '/root/autodl-tmp/models/Qwen2.5-7B-Instruct',device=torch.device('cpu'))
    system_prompt = '你是一个新闻的判断者，善于判断新闻的正确性'
    user_prompt = '新闻：美国总统特朗普出生于2024年12月11日。 \n这个新闻是对的吗？给出你的选择: [1]True,[2]False'
    options=['1','2']
    result = model.score_options(system_prompt, user_prompt, options)
    print(f"Options: {options}")
    print(f"Scores: {result['scores'].tolist()}")
    print(f"Probs:  {result['probs'].tolist()}")
    print(f"Best index: {result['best_idx']} -> '{options[result['best_idx']]}'")
    
    breakpoint()

    # print("\n" + "=" * 50)
    # print("Test 2: score_batch (multiple prompts)")
    # batch_prompts = [
    #     '新闻：地球是平的。\n这个新闻是对的吗？给出你的选择: [1]True,[2]False',
    #     '新闻：水在100摄氏度沸腾。\n这个新闻是对的吗？给出你的选择: [1]True,[2]False',
    # ]
    # batch_results = model.score_batch(system_prompt, batch_prompts, options, max_sequences=4)
    # for i, res in enumerate(batch_results):
    #     print(f"\nPrompt {i}: {batch_prompts[i][:30]}...")
    #     print(f"  Scores: {res['scores'].tolist()}")
    #     print(f"  Probs:  {res['probs'].tolist()}")
    #     print(f"  Best index: {res['best_idx']} -> '{options[res['best_idx']]}'")
    # breakpoint()


if __name__ == '__main__':
    main()
