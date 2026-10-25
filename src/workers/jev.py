from __future__ import annotations

import json
import logging
import math
import os
import re
import sys
import time
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import torch
import torch.nn.functional as F
from contextlib import nullcontext

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoTokenizer
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

    def tokenize(self, text: str) -> list[int]:
        return _tokenize(text)
    
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
        return _distribution(scores)
    
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


def _distribution(scores: torch.Tensor) -> dict[str, Any]:
    scores = scores.float().reshape(-1)
    probs = F.softmax(scores, dim=-1)
    return {
        "probs": probs,
        "scores": scores,
        "best_idx": int(torch.argmax(probs).item()),
    }

def _id_list(encoded: Any) -> list[int]:
    ids = encoded["input_ids"] if not isinstance(encoded, list) else encoded
    if isinstance(ids, torch.Tensor):
        ids = ids.reshape(-1).tolist()
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return [int(token_id) for token_id in ids]


def _tokenize(tokenizer, text: str) -> list[int]:
    return _id_list(tokenizer(text, add_special_tokens=False))



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




# prompt4jev_* 把候选写成 "[1] LABEL (description: ...)." 或 "(definition: ...)."
# Laya / Kev / API 吃的是选项正文，不是编号本身。编号顺序保持不变。
_OPTION_LINE = re.compile(
    r"^\[(?P<key>[^\]]+)\]\s+(?P<name>.+?)\s+\((?:description|definition):\s*(?P<body>.*)\)\.\s*$"
)


def _option_descriptions(user_prompt: str, options: list[str]) -> dict[str, str | None]:
    found: dict[str, str] = {}
    for line in user_prompt.splitlines():
        match = _OPTION_LINE.match(line.strip())
        if match is None:
            continue
        body = match.group("body").strip()
        text = match.group("name").strip()
        if body:
            text = f"{text}: {body}"
        found[match.group("key")] = text
    described: dict[str, str | None] = {}
    for option in options:
        described[str(option)] = found.get(str(option))
    return described


def _unique_options(options: list[str]) -> list[str]:
    keys = [str(option) for option in options]
    if not keys or len(set(keys)) != len(keys):
        raise ValueError(f"options must be non-empty and unique, got {options!r}")
    return keys


def _clamp_temperature(value: Any, default: float = 1.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(number):
        return default
    return min(5.0, max(0.5, number))


def _checkpoint_dir(model_path: str) -> Path:
    path = Path(model_path)
    if path.is_dir():
        return path
    if re.fullmatch(r"[\w.-]+/[\w.-]+(?:@[\w.-]+)?", model_path):
        from huggingface_hub import snapshot_download
        repo, _, revision = model_path.partition("@")
        return Path(snapshot_download(repo, revision=revision or None))
    raise FileNotFoundError(f"checkpoint not found: {model_path}")


def _load_state_dict(path: Path) -> dict[str, torch.Tensor]:
    safetensors_path = path / "model.safetensors"
    if safetensors_path.is_file():
        from safetensors.torch import load_file
        state = load_file(safetensors_path)
    else:
        binary = next(
            (path / name for name in ("pytorch_model.bin", "model.pt", "model.bin") if (path / name).is_file()),
            None,
        )
        if binary is None:
            raise FileNotFoundError(f"{path} has no model.safetensors or pytorch_model.bin")
        try:
            state = torch.load(binary, map_location="cpu", weights_only=True)
        except TypeError:
            state = torch.load(binary, map_location="cpu")
    if state and all(str(key).startswith("module.") for key in state):
        state = {str(key)[len("module."):]: value for key, value in state.items()}
    return state


################    api jev     #################
# TypeSafe System One: POST /v1/systemone
# https://docs.typesafe.ai/primitives/choice
# 开源客户端与字段约定同 jaredpalmer/kev 的 kev/api.py、kev/predictors.py。

class API_Jev(JevBase):
    """闭源 Jev。一次 choice，返回与 LLM_as_Jev 相同的 probs / scores / best_idx。"""

    _cls = "api_jev"

    def __init__(
        self,
        base_url: str = "https://api.typesafe.ai",
        api_key: str | None = None,
        model: str = "",
        max_retries: int = 3,
        timeout: float = 60.0,
    ):
        super().__init__()
        resolved_key = api_key or os.environ.get("TYPESAFE_API_KEY") or os.environ.get("JEV_API_KEY")
        if not resolved_key:
            raise ValueError("Missing API key: pass api_key or set TYPESAFE_API_KEY")
        self.api_key = resolved_key
        self.model = model or os.environ.get("TYPESAFE_DEFAULT_MODEL") or "jev-latest"
        self.max_retries = max_retries
        self.timeout = timeout
        root = base_url.rstrip("/")
        self.url = root if root.endswith("/v1/systemone") else f"{root}/v1/systemone"

    def score_options(
        self,
        system_prompt: str,
        user_prompt: str,
        options: list[str],
    ) -> dict[str, Any]:
        keys = _unique_options(options)
        described = _option_descriptions(user_prompt, keys)
        payload = {
            "model": self.model,
            "state": user_prompt,
            "questions": {
                "verdict": {
                    "type": "choice",
                    "instructions": system_prompt,
                    "criteria": {key: described[key] for key in keys},
                },
            },
        }
        answer = self._post(payload)
        probabilities = answer.get("probabilities")
        if not isinstance(probabilities, dict):
            raise RuntimeError(f"Jev API choice is missing probabilities: {answer}")
        missing = [key for key in keys if key not in probabilities]
        if missing:
            raise RuntimeError(
                f"Jev API probabilities missing {missing}; got {list(probabilities)}"
            )
        values = torch.tensor(
            [float(probabilities[key]) for key in keys],
            dtype=torch.float32,
        ).clamp_min(0)
        total = float(values.sum())
        if total <= 0:
            values = torch.ones_like(values) / values.numel()
        else:
            values = values / total
        return _distribution(torch.log(values.clamp_min(1e-12)))

    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            request = urllib.request.Request(
                self.url,
                data=data,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                method="POST",
            )
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    body = json.loads(response.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")
                last_error = RuntimeError(f"Jev API HTTP {exc.code}: {detail[:800]}")
                if exc.code != 429 and exc.code < 500:
                    raise last_error
                logger.warning(
                    "Jev API request failed (%s/%s): %s",
                    attempt,
                    self.max_retries,
                    last_error,
                )
                if attempt < self.max_retries:
                    self._sleep(attempt, exc.headers.get("Retry-After"))
                continue
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
                last_error = exc
                logger.warning(
                    "Jev API request failed (%s/%s): %s",
                    attempt,
                    self.max_retries,
                    exc,
                )
                if attempt < self.max_retries:
                    self._sleep(attempt, None)
                continue
            answers = body.get("answers", body)
            verdict = answers.get("verdict") if isinstance(answers, dict) else None
            if not isinstance(verdict, dict):
                raise RuntimeError(f"Jev API response has no verdict answer: {body}")
            return verdict
        assert last_error is not None
        raise last_error

    @staticmethod
    def _sleep(attempt: int, retry_after: str | None) -> None:
        delay = min(2 ** (attempt - 1), 8)
        if retry_after:
            try:
                delay = max(delay, float(retry_after))
            except ValueError:
                pass
        time.sleep(delay)


################    laya     #################
# NandhaKishorM/laya 与 convaiinnovations/laya 的选项标记打分。
# 序列: [CLS] choice question: instructions [SEP] [MASK] opt0 [MASK] opt1 ... [SEP] state [SEP]
# 决策头只读每个 [MASK] 的隐状态，再在选项上 softmax。

_LAYA_QTYPES = {"choice": 0, "score": 1, "noul": 2}


def _laya_temp_bucket(qtype: str, n_options: int) -> str:
    if n_options <= 2:
        size = "2"
    elif n_options <= 5:
        size = "3-5"
    elif n_options <= 10:
        size = "6-10"
    else:
        size = "11+"
    return f"{qtype}:{size}"


def _laya_render_options(criteria: dict[str, str | None]) -> list[str]:
    return [
        key if not value else f"{key}: {value}"
        for key, value in criteria.items()
    ]


def _laya_build_sequence(
    tokenizer,
    state: str,
    instructions: str,
    criteria: dict[str, str | None],
    max_len: int,
    head_max_len: int,
    truncate_left: bool = False,
) -> tuple[list[int], list[int]]:
    """与 laya rl_common.build_sequence 的 choice 路径一致。"""
    mask_token = tokenizer.mask_token
    if mask_token is None or tokenizer.cls_token_id is None or tokenizer.sep_token_id is None:
        raise ValueError("Laya tokenizer is missing [CLS], [SEP], or [MASK]")
    if tokenizer.mask_token_id is None:
        raise ValueError("Laya tokenizer is missing a mask token id")

    question = {
        "t": "choice",
        "ins": instructions,
        "crit": criteria,
    }
    options = _laya_render_options(criteria)
    instructions = str(question["ins"]).replace(mask_token, " ")
    head_ids = _tokenize(tokenizer, f"{question['t']} question: {instructions}")
    option_ids: list[list[int]] = []
    for option in options:
        body = _tokenize(tokenizer, " " + option.replace(mask_token, " "))[:48]
        option_ids.append([int(tokenizer.mask_token_id)] + body)

    budget = head_max_len - sum(len(piece) for piece in option_ids)
    if budget < 16:
        per_option = max(4, (head_max_len - 16) // max(1, len(option_ids)))
        option_ids = [piece[:per_option] for piece in option_ids]
        budget = head_max_len - sum(len(piece) for piece in option_ids)
    head_ids = head_ids[:max(8, budget)]

    ids = [int(tokenizer.cls_token_id)] + head_ids + [int(tokenizer.sep_token_id)]
    markers: list[int] = []
    for piece in option_ids:
        markers.append(len(ids))
        ids.extend(piece)
    ids.append(int(tokenizer.sep_token_id))

    room = max(0, max_len - len(ids) - 1)
    state_ids = _tokenize(tokenizer, state.replace(mask_token, " "))
    state_ids = state_ids[-room:] if truncate_left else state_ids[:room]
    ids = ids + state_ids + [int(tokenizer.sep_token_id)]
    ids = ids[:max_len]
    return ids, [marker for marker in markers if marker < max_len]


def _laya_collate(items: list[dict[str, Any]], pad_id: int) -> dict[str, torch.Tensor]:
    width = max(len(item["ids"]) for item in items)
    n_options = max(len(item["markers"]) for item in items)
    ids = torch.full((len(items), width), pad_id, dtype=torch.long)
    attention = torch.zeros((len(items), width), dtype=torch.long)
    marker_pos = torch.zeros((len(items), n_options), dtype=torch.long)
    marker_mask = torch.zeros((len(items), n_options), dtype=torch.bool)
    for row, item in enumerate(items):
        length = len(item["ids"])
        ids[row, :length] = torch.tensor(item["ids"], dtype=torch.long)
        attention[row, :length] = 1
        count = len(item["markers"])
        marker_pos[row, :count] = torch.tensor(item["markers"], dtype=torch.long)
        marker_mask[row, :count] = True
    return {
        "input_ids": ids,
        "attention_mask": attention,
        "marker_pos": marker_pos,
        "marker_mask": marker_mask,
        "qtype": torch.tensor([item["qtype"] for item in items], dtype=torch.long),
    }


class _LayaDecisionModel(torch.nn.Module):
    """双向编码器 + 选项 [MASK] 打分头。结构对齐 laya DecisionModel，以便加载其权重。"""

    def __init__(self, encoder: torch.nn.Module, head_layers: int = 2, n_act: int = 2, dropout: float = 0.1):
        super().__init__()
        self.encoder = encoder
        hidden = int(encoder.config.hidden_size)
        heads = max(1, hidden // 64)
        layer = torch.nn.TransformerEncoderLayer(
            hidden,
            heads,
            4 * hidden,
            dropout,
            batch_first=True,
            norm_first=True,
        )
        if head_layers > 0:
            self.head = torch.nn.TransformerEncoder(layer, head_layers, enable_nested_tensor=False)
        else:
            self.head = None
        self.type_emb = torch.nn.Embedding(3, hidden)
        self.scorer = torch.nn.Sequential(
            torch.nn.LayerNorm(hidden),
            torch.nn.Linear(hidden, hidden),
            torch.nn.GELU(),
            torch.nn.Linear(hidden, 1),
        )
        self.act_head = torch.nn.Sequential(
            torch.nn.Linear(hidden + 4, 256),
            torch.nn.GELU(),
            torch.nn.Linear(256, n_act),
        )
        self.register_buffer("temperature", torch.ones(3))

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        marker_pos: torch.Tensor,
        marker_mask: torch.Tensor,
        qtype: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        hidden = hidden + self.type_emb(qtype)[:, None, :]
        if self.head is not None:
            padding = ~attention_mask.bool()
            for layer in self.head.layers:
                hidden = layer(hidden, src_key_padding_mask=padding)
        index = marker_pos.clamp(min=0)[:, :, None].expand(-1, -1, hidden.size(-1))
        gathered = torch.gather(hidden, 1, index)
        logits = self.scorer(gathered).squeeze(-1).float()
        logits = logits.masked_fill(~marker_mask, -1e4)
        # act head 不参与 verdict，但权重在 checkpoint 里，前向必须在。
        # 单个选项时 topk(2) 越界，按仓库后续的修复把第二名补 0。
        distribution = torch.softmax(logits.detach(), dim=-1)
        width = distribution.size(-1)
        if width >= 2:
            top2 = distribution.topk(2, dim=-1).values
        else:
            top2 = torch.cat([distribution, torch.zeros_like(distribution)], dim=-1)
        n_valid = marker_mask.sum(-1).clamp(min=2).float()
        entropy = -(distribution * torch.log(distribution.clamp_min(1e-9))).sum(-1) / torch.log(n_valid)
        features = torch.stack(
            [top2[:, 0], top2[:, 0] - top2[:, 1], entropy, n_valid / 255.0],
            dim=-1,
        )
        pooled = hidden[:, 0].float()
        act_input = torch.cat([pooled, features], dim=-1)
        act_dtype = self.act_head[0].weight.dtype
        act_logits = self.act_head(act_input.to(dtype=act_dtype))
        return logits, act_logits


class Laya(JevBase):
    _cls = "laya"

    def __init__(
        self,
        model_path: str,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        max_len: int | None = None,
        head_max_len: int | None = None,
        local_files_only: bool = True,
        truncate_left: bool = False,
    ):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.device.type == "cpu":
            print("using cpu!!!")
        self.dtype = dtype or (torch.bfloat16 if self.device.type == "cuda" else torch.float32)
        self.truncate_left = truncate_left
        self.local_files_only = local_files_only

        path = _checkpoint_dir(model_path)
        config_path = path / "rl_agent_config.json"
        config: dict[str, Any] = {}
        if config_path.is_file():
            config = json.loads(config_path.read_text(encoding="utf-8"))
        self.max_len = int(max_len or config.get("max_len") or 512)
        self.head_max_len = int(head_max_len or config.get("head_max_len") or 192)
        self.temperature = [
            _clamp_temperature(value)
            for value in list(config.get("temperature") or [1.0, 1.0, 1.0])[:3]
        ]
        while len(self.temperature) < 3:
            self.temperature.append(1.0)
        buckets = config.get("temperature_by_options") or {}
        self.temperature_by_options = {
            str(key): _clamp_temperature(value)
            for key, value in buckets.items()
            if _clamp_temperature(value, default=-1) > 0
        }

        encoder_dir = path / "encoder" if (path / "encoder" / "config.json").is_file() else path
        tokenizer_dir = path if (path / "tokenizer_config.json").is_file() else encoder_dir
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_dir,
            local_files_only=local_files_only,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = 0

        state = _load_state_dict(path)
        n_act = int(state["act_head.2.weight"].shape[0]) if "act_head.2.weight" in state else 2
        head_layers = 0
        for key in state:
            if key.startswith("head.layers."):
                head_layers = max(head_layers, int(key.split(".")[2]) + 1)
        if head_layers == 0 and "head_layers" in config and not any(key.startswith("scorer.") for key in state):
            head_layers = int(config["head_layers"])

        encoder_config = AutoConfig.from_pretrained(encoder_dir, local_files_only=local_files_only)
        try:
            encoder = AutoModel.from_config(encoder_config, attn_implementation="sdpa")
        except TypeError:
            encoder = AutoModel.from_config(encoder_config)
        self.model = _LayaDecisionModel(encoder, head_layers=head_layers, n_act=n_act)
        self.model.to(dtype=self.dtype)
        self.model.load_state_dict(state)
        if getattr(self.model.encoder, "config", None) is not None:
            if getattr(self.model.encoder.config, "reference_compile", None):
                self.model.encoder.config.reference_compile = False
        self.model.to(self.device)
        self.model.eval()

    def encode(
        self,
        system_prompt: str,
        user_prompt: str,
        options: list[str],
    ) -> dict[str, Any]:
        keys = _unique_options(options)
        criteria = _option_descriptions(user_prompt, keys)
        ids, markers = _laya_build_sequence(
            self.tokenizer,
            state=user_prompt,
            instructions=system_prompt,
            criteria=criteria,
            max_len=self.max_len,
            head_max_len=self.head_max_len,
            truncate_left=self.truncate_left,
        )
        if len(markers) != len(keys):
            raise RuntimeError(
                f"Laya fit {len(markers)} of {len(keys)} options into head_max_len={self.head_max_len}"
            )
        return {
            "ids": ids,
            "markers": markers,
            "qtype": _LAYA_QTYPES["choice"],
        }

    def _temperature_for(self, n_options: int) -> float:
        bucket = _laya_temp_bucket("choice", n_options)
        if bucket in self.temperature_by_options:
            return self.temperature_by_options[bucket]
        return self.temperature[_LAYA_QTYPES["choice"]]

    def score_options(
        self,
        system_prompt: str,
        user_prompt: str,
        options: list[str],
    ) -> dict[str, Any]:
        encoded = self.encode(system_prompt, user_prompt, options)
        batch = _laya_collate([encoded], int(self.tokenizer.pad_token_id))
        batch = {key: value.to(self.device) for key, value in batch.items()}
        with torch.inference_mode():
            logits, _ = self.model(
                batch["input_ids"],
                batch["attention_mask"],
                batch["marker_pos"],
                batch["marker_mask"],
                batch["qtype"],
            )
        scores = logits[0, :len(encoded["markers"])]
        scores = scores / self._temperature_for(int(scores.numel()))
        return _distribution(scores)


################    kev     #################
# jaredpalmer/kev: Qwen 骨干 + 指针头。
# <|fim_prefix|> state <|fim_middle|> instruction <|box_start|> option <|box_end|> ... <|fim_suffix|>
# 指针用 <decide> 的隐状态去点每个选项 </opt> 的隐状态。
# 当前发布的 Kev 都是 Qwen3.5 混合骨干，单题用一条因果行，和 packed mask 等价。

_KEV_SPECIAL = (
    "<|fim_prefix|>",  # state
    "<|fim_middle|>",  # question
    "<|box_start|>",   # option
    "<|box_end|>",     # /option
    "<|fim_suffix|>",  # decide
)
_KEV_OPT_NONE = -1
_KEV_OPT_DECIDE = -2
_KEV_SPECIAL_RE = re.compile(r"<\|([A-Za-z0-9_]+)\|>")


class _KevPointerHead(torch.nn.Module):
    def __init__(self, hidden: int, pointer_dim: int = 256):
        super().__init__()
        self.q = torch.nn.Linear(hidden, pointer_dim)
        self.k = torch.nn.Linear(hidden, pointer_dim)
        self.scale = 1 / math.sqrt(pointer_dim)
        self.temperature = 1.0

    def forward(self, hidden_decide: torch.Tensor, hidden_options: torch.Tensor) -> torch.Tensor:
        logits = (self.k(hidden_options) @ self.q(hidden_decide)) * self.scale
        if self.training or self.temperature == 1.0:
            return logits
        return logits / self.temperature


def _kev_user_tokens(tokenizer, text: str) -> list[int]:
    """用户文本里的 <|...|> 先换掉，避免伪造 state / option 边界。"""
    safe = _KEV_SPECIAL_RE.sub(r"<¦\1¦>", text)
    return _tokenize(tokenizer, safe)


def _kev_special_ids(tokenizer) -> list[int]:
    unknown = tokenizer.unk_token_id
    ids: list[int] = []
    for token in _KEV_SPECIAL:
        token_id = tokenizer.convert_tokens_to_ids(token)
        if token_id is None or token_id == unknown:
            raise ValueError(f"tokenizer is missing Kev delimiter {token}")
        ids.append(int(token_id))
    return ids


def _kev_encode(
    tokenizer,
    state: str,
    instructions: str,
    options: list[str],
    max_state: int,
    max_branch: int,
    option_isolation: bool = False,
) -> dict[str, Any]:
    """一条因果行: state + 这一题的 branch。布局同 kev.model.encode 的单题情形。"""
    state_id, question_id, option_id, close_id, decide_id = _kev_special_ids(tokenizer)
    state_tokens = _kev_user_tokens(tokenizer, state)
    truncated = len(state_tokens) + 1 > max_state
    state_ids = [state_id] + state_tokens[:max_state - 1]
    instruction = [question_id] + _kev_user_tokens(tokenizer, instructions)
    spans = [
        [option_id] + _kev_user_tokens(tokenizer, option) + [close_id]
        for option in options
    ]
    if not spans or any(len(span) < 2 for span in spans):
        raise ValueError(f"Kev options produced an empty span: {options!r}")
    branch = instruction + [token for span in spans for token in span] + [decide_id]
    if len(branch) > max_branch - len(state_ids):
        raise RuntimeError(
            f"Kev branch has {len(branch)} tokens after a {len(state_ids)}-token state "
            f"(row limit {max_branch})"
        )
    if option_isolation:
        longest = max(len(span) for span in spans)
        instruction_pos = list(range(len(state_ids), len(state_ids) + len(instruction)))
        option_pos = [
            len(state_ids) + len(instruction) + offset
            for span in spans
            for offset in range(len(span))
        ]
        branch_pos = instruction_pos + option_pos + [len(state_ids) + len(instruction) + longest]
    else:
        branch_pos = list(range(len(state_ids), len(state_ids) + len(branch)))
    branch_opt = (
        [_KEV_OPT_NONE] * len(instruction)
        + [index for index, span in enumerate(spans) for _ in span]
        + [_KEV_OPT_DECIDE]
    )
    ids = state_ids + branch
    pos = list(range(len(state_ids))) + branch_pos
    opt = [_KEV_OPT_NONE] * len(state_ids) + branch_opt
    cursor = len(instruction)
    option_ends: list[int] = []
    for span in spans:
        cursor += len(span)
        option_ends.append(len(state_ids) + cursor - 1)
    return {
        "ids": ids,
        "pos": pos,
        "opt": opt,
        "decide": len(ids) - 1,
        "opts": option_ends,
        "state_truncated": truncated,
        "state_tokens": len(state_tokens) + 1,
    }


def _kev_isolation_mask(
    option_rows: list[list[int]],
    width: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """选项之间互不可见；<decide> 能看见本题全部 token。加性 mask，形状 [B, 1, L, L]。"""
    labels = torch.full((len(option_rows), width), -3, dtype=torch.long, device=device)
    for row, opt in enumerate(option_rows):
        labels[row, :len(opt)] = torch.tensor(opt, dtype=torch.long, device=device)
    causal = torch.tril(torch.ones(width, width, dtype=torch.bool, device=device))
    key_is_option = labels[:, None, :] >= 0
    query_is_decide = labels[:, :, None] == _KEV_OPT_DECIDE
    same_option = labels[:, None, :] == labels[:, :, None]
    valid_key = torch.zeros((len(option_rows), width), dtype=torch.bool, device=device)
    for row, opt in enumerate(option_rows):
        valid_key[row, :len(opt)] = True
    allow = causal[None] & valid_key[:, None, :] & (~key_is_option | query_is_decide | same_option)
    allow = allow | torch.eye(width, dtype=torch.bool, device=device)[None]
    mask = torch.zeros((len(option_rows), width, width), dtype=dtype, device=device)
    return mask.masked_fill(~allow, torch.finfo(dtype).min)[:, None]


class Kev(torch.nn.Module, JevBase):
    _cls = "kev"

    def __init__(
        self,
        model_path: str,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        max_state: int = 8192,
        max_branch: int | None = None,
        local_files_only: bool = True,
        merge_lora: bool | None = None,
    ):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.device.type == "cpu":
            print("using cpu!!!")
        self.dtype = dtype or (torch.bfloat16 if self.device.type == "cuda" else torch.float32)
        self.max_state = max_state
        self.max_branch = max_branch or (max_state + 1024)
        self.local_files_only = local_files_only
        self._truncation_warned = False

        path = _checkpoint_dir(model_path)
        head_path = path / "head.pt"
        if not head_path.is_file():
            raise FileNotFoundError(f"Kev checkpoint {path} has no head.pt")
        try:
            meta = torch.load(head_path, map_location="cpu", weights_only=False)
        except TypeError:
            meta = torch.load(head_path, map_location="cpu")
        if not isinstance(meta, dict) or not isinstance(meta.get("head"), dict):
            raise ValueError(f"{head_path} is not a Kev head.pt (expected a dict with 'head')")

        self.option_isolation = bool(meta.get("option_isolation", False))
        head_state = {
            key: value for key, value in meta["head"].items()
            if isinstance(value, torch.Tensor)
        }
        head_dim = int(head_state["q.weight"].shape[0]) if "q.weight" in head_state else int(meta.get("head_dim", 256))
        temperature = float(meta.get("temperature", 1.0))
        if not math.isfinite(temperature) or temperature <= 0:
            temperature = 1.0
        full_weights = (path / "adapter_config.json").is_file() is False and (path / "config.json").is_file()
        if str(meta.get("weights", "lora" if not full_weights else "full")) not in ("lora", "full"):
            raise ValueError(f"unsupported Kev weights field: {meta.get('weights')!r}")
        if bool(meta.get("weights", "lora") == "full") != full_weights and "weights" in meta:
            found = "full" if full_weights else "lora"
            raise ValueError(
                f"{path}: head.pt says weights={meta.get('weights')!r} but the directory looks like {found}"
            )

        base_name = str(meta.get("base") or model_path)
        revision = meta.get("base_revision") or None
        config_source = path if full_weights else base_name
        text_config = AutoConfig.from_pretrained(
            config_source,
            revision=None if full_weights else revision,
            local_files_only=local_files_only,
            trust_remote_code=True,
        )
        if hasattr(text_config, "get_text_config"):
            text_config = text_config.get_text_config()
        self.hybrid = "linear_attention" in set(getattr(text_config, "layer_types", None) or [])
        if self.hybrid and self.option_isolation:
            raise ValueError("option_isolation needs the packed mask; not available on hybrid backbones")

        attn = "sdpa" if self.device.type == "cuda" else "eager"
        load_kwargs: dict[str, Any] = {
            "dtype": self.dtype,
            "local_files_only": local_files_only,
            "trust_remote_code": True,
        }
        if not full_weights and revision:
            load_kwargs["revision"] = revision
        source = path if full_weights else base_name
        if full_weights:
            try:
                self.lm = AutoModel.from_pretrained(source, attn_implementation=attn, **load_kwargs)
            except TypeError:
                self.lm = AutoModel.from_pretrained(source, **load_kwargs)
        else:
            try:
                causal = AutoModelForCausalLM.from_pretrained(source, attn_implementation=attn, **load_kwargs)
            except TypeError:
                causal = AutoModelForCausalLM.from_pretrained(source, **load_kwargs)
            self.lm = causal.model
            del causal
            from peft import PeftModel
            self.lm = PeftModel.from_pretrained(self.lm, str(path))
            adapter_config = json.loads((path / "adapter_config.json").read_text(encoding="utf-8"))
            should_merge = merge_lora if merge_lora is not None else (
                not adapter_config.get("trainable_token_indices") and meta.get("weights_dtype") != "bf16"
            )
            if should_merge:
                self.lm = self.lm.merge_and_unload()

        hidden = getattr(self.lm.config, "hidden_size", None)
        if hidden is None and hasattr(self.lm.config, "get_text_config"):
            hidden = self.lm.config.get_text_config().hidden_size
        if hidden is None:
            hidden = text_config.hidden_size
        self.head = _KevPointerHead(int(hidden), pointer_dim=head_dim)
        self.head.load_state_dict(head_state)
        self.head.temperature = temperature
        self.pad_id = int(self._load_tokenizer(path, base_name, revision).pad_token_id or 0)
        self.lm.to(device=self.device, dtype=self.dtype)
        self.head.to(self.device)
        self.eval()

    def _load_tokenizer(self, path: Path, base_name: str, revision: str | None):
        tokenizer_dir: str | Path = path if (path / "tokenizer_config.json").is_file() else base_name
        kwargs: dict[str, Any] = {
            "local_files_only": self.local_files_only,
            "trust_remote_code": True,
        }
        if tokenizer_dir == base_name and revision:
            kwargs["revision"] = revision
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, **kwargs)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        return self.tokenizer

    def encode(
        self,
        system_prompt: str,
        user_prompt: str,
        options: list[str],
    ) -> dict[str, Any]:
        keys = _unique_options(options)
        described = _option_descriptions(user_prompt, keys)
        option_text = [
            key if not described[key] else f"{key}: {described[key]}"
            for key in keys
        ]
        encoded = _kev_encode(
            self.tokenizer,
            state=user_prompt,
            instructions=system_prompt,
            options=option_text,
            max_state=self.max_state,
            max_branch=self.max_branch,
            option_isolation=self.option_isolation,
        )
        if encoded["state_truncated"] and not self._truncation_warned:
            logger.warning(
                "Kev state truncated from %s tokens to max_state=%s",
                encoded["state_tokens"],
                self.max_state,
            )
            self._truncation_warned = True
        return encoded

    def _hidden_rows(self, rows: list[dict[str, Any]]) -> list[torch.Tensor]:
        width = max(len(row["ids"]) for row in rows)
        ids = torch.full((len(rows), width), self.pad_id, dtype=torch.long, device=self.device)
        pos = torch.zeros((len(rows), width), dtype=torch.long, device=self.device)
        attention = torch.zeros((len(rows), width), dtype=torch.long, device=self.device)
        for index, row in enumerate(rows):
            length = len(row["ids"])
            ids[index, :length] = torch.tensor(row["ids"], dtype=torch.long, device=self.device)
            pos[index, :length] = torch.tensor(row["pos"], dtype=torch.long, device=self.device)
            attention[index, :length] = 1
        forward_kwargs: dict[str, Any] = {
            "input_ids": ids,
            "position_ids": pos,
            "use_cache": False,
        }
        if self.option_isolation:
            forward_kwargs["attention_mask"] = _kev_isolation_mask(
                [row["opt"] for row in rows],
                width,
                self.device,
                self.dtype,
            )
        else:
            forward_kwargs["attention_mask"] = attention
        output = self.lm(**forward_kwargs)
        hidden = output.last_hidden_state if hasattr(output, "last_hidden_state") else output[0]
        hidden = hidden.float()
        return [hidden[index, :len(row["ids"])] for index, row in enumerate(rows)]

    def _row_logits(self, rows: list[dict[str, Any]]) -> list[torch.Tensor]:
        logits: list[torch.Tensor] = []
        for hidden, row in zip(self._hidden_rows(rows), rows):
            option_index = torch.tensor(row["opts"], dtype=torch.long, device=hidden.device)
            logits.append(self.head(hidden[row["decide"]], hidden[option_index]))
        return logits

    def score_options(
        self,
        system_prompt: str,
        user_prompt: str,
        options: list[str],
    ) -> dict[str, Any]:
        encoded = self.encode(system_prompt, user_prompt, options)
        with torch.inference_mode():
            scores = self._row_logits([encoded])[0]
        return _distribution(scores)

    def score_batch(
        self,
        system_prompt: str,
        batch_user_prompt: list[str],
        options: list[str],
        max_sequences: int = 8,
        is_training: bool = False,
    ) -> list[dict[str, Any]]:
        if not batch_user_prompt:
            return []
        if max_sequences <= 0:
            raise ValueError(f"max_sequences must be positive, got {max_sequences}")
        encoded = [
            self.encode(system_prompt, user_prompt, options)
            for user_prompt in batch_user_prompt
        ]
        was_training = self.training
        self.train(is_training)
        chunks: list[torch.Tensor] = []
        context = nullcontext() if is_training else torch.inference_mode()
        try:
            with context:
                for start in range(0, len(encoded), max_sequences):
                    chunks.extend(self._row_logits(encoded[start:start + max_sequences]))
        finally:
            self.train(was_training)
        return [_distribution(scores) for scores in chunks]










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
