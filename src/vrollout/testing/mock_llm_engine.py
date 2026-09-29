"""Mock LLM engine for testing."""

import asyncio
from collections.abc import Mapping
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any
from typing import AsyncIterator
from typing import Dict
from typing import List
from typing import Optional

import torch
from torch import nn
from torch import Tensor
from torch.nn import functional as F

from vrollout.testing import models
from vrollout.types import SamplingParams

MIN_COMPLETION_TOKENS = 16
ASCII_CHAR_COUNT = 128


@dataclass(frozen=True)
class CompletionOutput:
    index: int
    text: str
    token_ids: List[int]
    cumulative_logprob: float
    logprobs: Optional[List[Dict[int, Any]]] = None
    finish_reason: Optional[str] = "stop"


@dataclass(frozen=True)
class RequestOutput:
    request_id: str
    prompt: str
    prompt_token_ids: List[int]
    outputs: List[CompletionOutput]
    finished: bool


def _sample_next_token(next_token_logits: Tensor, temperature: float) -> int:
    if temperature < 1e-4:
        return int(torch.argmax(next_token_logits).item())

    probs = F.softmax(next_token_logits / temperature, dim=-1)
    return int(torch.multinomial(probs, num_samples=1).item())


class AsciiTokenizer:
    """Minimal, deterministic ASCII tokenizer with vocab-size bucketing."""

    def __init__(self, vocab_size: int = ASCII_CHAR_COUNT) -> None:
        self.vocab_size = vocab_size

    def encode(self, text: str) -> list[int]:
        """Encodes text to token IDs bucketed by vocab_size."""
        return [ord(c) % self.vocab_size for c in text]

    def decode(self, token_ids: Sequence[int]) -> str:
        """Decodes token IDs back to a string."""
        return "".join(chr(t) for t in token_ids)


def _infer_vocab_size(model_name: str) -> int:
    if model_name in models.TEST_MODEL_SPECS:
        return models.TEST_MODEL_SPECS[model_name]["vocab_size"]

    raise ValueError(f"Could not infer vocab_size from model_name: {model_name}")


class MockAsyncLLMEngine:
    """Drop-in mock for vllm.AsyncLLMEngine for local testing."""

    def __init__(
        self,
        model: nn.Module | None = None,
        vocab_size: int | None = None,
        pad_token_id: int = 0,
        model_name: str | None = None,
    ):
        self.model = model or models.create_model(model_name)
        self.vocab_size = vocab_size or _infer_vocab_size(model_name)
        self.pad_token_id = pad_token_id
        self.model_name = model_name or type(model).__name__

        self.tokenizer = AsciiTokenizer(self.vocab_size)

    async def generate(
        self, prompt: str, sampling_params: SamplingParams, request_id: str
    ) -> AsyncIterator[RequestOutput]:
        prompt_tokens = self.tokenizer.encode(prompt)
        curr_tokens = prompt_tokens.copy()

        completion_tokens: list[int] = []
        logprobs_list: list[dict[int, float]] = []
        cumulative_logprob = 0.0

        max_tokens = min(sampling_params.max_tokens, MIN_COMPLETION_TOKENS)
        temperature = max(sampling_params.temperature, 0.0)

        for step in range(1, max_tokens + 1):
            # generate new tokens
            inputs = torch.tensor([curr_tokens], dtype=torch.long)
            logits = self.model(inputs)
            next_token_logits = logits[0, -1, :]  # (1, vocab_size)
            next_token = _sample_next_token(next_token_logits, temperature)
            logprobs = F.log_softmax(next_token_logits, dim=-1)
            next_token_logprob = float(logprobs[next_token].item())

            # collect token generation data
            completion_tokens.append(next_token)
            curr_tokens.append(next_token)
            logprobs_list.append({next_token: next_token_logprob})
            cumulative_logprob += next_token_logprob

            # yield progress to asyncio event loop
            await asyncio.sleep(0.001)

            # form output
            is_finished = step == max_tokens
            yield RequestOutput(
                request_id=request_id,
                prompt=prompt,
                prompt_token_ids=prompt_tokens,
                outputs=[
                    CompletionOutput(
                        index=0,
                        text=self.tokenizer.decode(completion_tokens),
                        token_ids=completion_tokens,
                        cumulative_logprob=cumulative_logprob,
                        logprobs=logprobs_list,
                        finish_reason="stop" if is_finished else None,
                    )
                ],
                finished=is_finished,
            )

    def update_weights(self, weights: Mapping[str, Tensor]) -> None:
        assert self.model is not None
        with torch.no_grad():
            self.model.load_state_dict(weights, strict=False)
