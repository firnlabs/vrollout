"""Mock LLM engine for testing."""

import asyncio
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any
from typing import AsyncIterator
from typing import Dict
from typing import List
from typing import Optional

from torch import Tensor

from vrollout.types import SamplingParams


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


class MockAsyncLLMEngine:
    """Drop-in mock for vllm.AsyncLLMEngine for local testing."""

    def __init__(self, model_name: str = "mock-model", vocab_size: int = 32):
        self.model_name = model_name
        self.vocab_size = vocab_size
        self.weights: Mapping[str, Tensor] | None = None

    async def generate(
        self, prompt: str, sampling_params: SamplingParams, request_id: str
    ) -> AsyncIterator[RequestOutput]:
        prompt_tokens = [hash(w) % self.vocab_size for w in prompt.split()]
        num_gen = min(sampling_params.max_tokens, 8)
        gen_tokens = [(i + 1) % self.vocab_size for i in range(num_gen)]

        logprobs_list = []
        for tok in gen_tokens:
            logprobs_list.append({tok: -0.25})

        for step in range(1, num_gen + 1):
            await asyncio.sleep(0.01)
            is_finished = step == num_gen
            yield RequestOutput(
                request_id=request_id,
                prompt=prompt,
                prompt_token_ids=prompt_tokens,
                outputs=[
                    CompletionOutput(
                        index=0,
                        text=f"Mock completion for: {prompt[:15]}...",
                        token_ids=gen_tokens[:step],
                        cumulative_logprob=-0.25 * step,
                        logprobs=logprobs_list[:step],
                        finish_reason="stop" if is_finished else None,
                    )
                ],
                finished=is_finished,
            )

    def update_weights(self, weights: Mapping[str, Tensor]) -> None:
        self.weights = weights
