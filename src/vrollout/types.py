"""Common distributed, model, and generation type definitions."""

from collections.abc import Mapping
from collections.abc import Sequence
from dataclasses import dataclass
from typing import AsyncGenerator
from typing import Protocol
from typing import runtime_checkable

type Rank = int
type WorldSize = int


class SamplingParams(Protocol):
    """LLM sampling parameters for generation.

    Reference:
    - https://docs.vllm.ai/en/v0.6.0/dev/sampling_params.html
    """

    temperature: float
    max_tokens: int
    logprobs: int | None


@runtime_checkable
class SampleLogprob(Protocol):
    logprob: float
    rank: int | None = None
    decoded_token: str | None = None


class CompletionOutput(Protocol):
    index: int
    text: str
    token_ids: Sequence[int]
    cumulative_logprob: float
    logprobs: Sequence[Mapping[int, SampleLogprob | float]] | None = None
    finish_reason: str | None = "stop"


class RequestOutput(Protocol):
    request_id: str
    prompt: str
    prompt_token_ids: Sequence[int]
    outputs: Sequence[CompletionOutput]
    finished: bool


class AsyncLLMEngine(Protocol):
    def generate(
        self,
        prompt: str,
        sampling_params: SamplingParams,
        request_id: str,
        *args,
        **kwargs,
    ) -> AsyncGenerator[RequestOutput, None]: ...


@dataclass
class RolloutConfig:
    """Configuration for distributed rollout generation."""

    model_name: str = "mock-model"
    num_workers: int = 2
    gpus_per_worker: float = 0.0  # default 0.0 for local testing
    max_batch_size: int = 64
    max_seq_len: int = 512
    gpu_memory_utilization: float = 0.60


@dataclass(frozen=True, slots=True)
class RolloutTrajectory:
    """Individual generated rollout trajectory and metadata."""

    request_id: str
    prompt: str
    prompt_token_ids: Sequence[int]
    completion_token_ids: Sequence[int]
    logprobs: Sequence[float]
    finish_reason: str
    text: str
    worker_id: int
