from __future__ import annotations

import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import logging
from abc import ABC, abstractmethod
from typing import Any
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import torch
import torch.nn.functional as F
from vllm import LLM, SamplingParams
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
from openai import OpenAI

ROOT = Path(__file__).parents[2].resolve()
sys.path.insert(0,str(ROOT))

from src.utils import load_tokenizer
from src.prompts import SYSTEM_PROMPT

logger = logging.getLogger(__file__)



class GeneratorBase(ABC):

    @abstractmethod
    def generate_batch(self):
        pass



class VLLMGenerator(GeneratorBase):
    def __init__(
        self,
        model_path: str,
        tensor_parallel_size: int = torch.cuda.device_count(),
        gpu_memory_utilization: float = 0.92,
        max_model_len: int = 8192,
        max_tokens: int = 1024,
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
            max_tokens=max_tokens,
            skip_special_tokens=True,
        )

    def _format_one(self, user_prompt: str, add_generation_prompt: bool=True) -> str:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        extra_kwargs = dict()
        if self.enable_thinking:
            extra_kwargs['enable_thinking'] = True
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            **extra_kwargs
        )

    def generate_batch(self, prompts: list[str]) -> list[str]:
        # greedy decoding
        formatted = [self._format_one(prompt) for prompt in prompts]
        outputs = self.llm.generate(formatted, self.sp)
        return [output.outputs[0].text.strip() for output in outputs]


class APIGenerator(GeneratorBase):
    def __init__(
        self,
        base_url: str = "https://api.deepseek.com",
        api_key: str | None = None,
        model: str = "deepseek-flash",
        max_tokens: int | None = None,
        enable_thinking: bool = True,
        reasoning_effort: str = "high",
        max_workers: int = 8,
        max_retries: int = 3,
    ):
        resolved_key = api_key or os.environ.get("DEEPSEEK_API_KEY")
        if not resolved_key:
            raise ValueError(
                "Missing API key: pass api_key or set DEEPSEEK_API_KEY"
            )
        self.client = OpenAI(
            api_key=resolved_key,
            base_url=base_url or "https://api.deepseek.com",
        )
        self.model = model
        self.max_tokens = max_tokens
        self.enable_thinking = enable_thinking
        self.reasoning_effort = reasoning_effort
        self.max_workers = max_workers
        self.max_retries = max_retries
        self.llm_name = model

    def _format_one(self, user_prompt: str) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

    def _generate_one(self, user_prompt: str) -> str:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": self._format_one(user_prompt),
            "stream": False,
        }
        if self.max_tokens is not None:
            kwargs["max_tokens"] = self.max_tokens
        if self.enable_thinking:
            kwargs["reasoning_effort"] = self.reasoning_effort
            kwargs["extra_body"] = {"thinking": {"type": "enabled"}}

        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.client.chat.completions.create(**kwargs)
                content = response.choices[0].message.content or ""
                return content.strip()
            except Exception as exc:
                last_error = exc
                logger.warning(
                    "API request failed (%s/%s): %s",
                    attempt,
                    self.max_retries,
                    exc,
                )
                if attempt < self.max_retries:
                    time.sleep(min(2 ** (attempt - 1), 8))
        assert last_error is not None
        raise last_error

    def generate_batch(self, prompts: list[str]) -> list[str]:
        if not prompts:
            return []
        workers = max(1, min(self.max_workers, len(prompts)))
        if workers == 1:
            return [self._generate_one(prompt) for prompt in prompts]

        results: list[str] = [""] * len(prompts)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(self._generate_one, prompt): index
                for index, prompt in enumerate(prompts)
            }
            for future in as_completed(futures):
                results[futures[future]] = future.result()
        return results