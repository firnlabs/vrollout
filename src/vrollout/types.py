"""Common distributed, model, and generation type definitions."""

from collections.abc import Mapping
from collections.abc import Sequence
from dataclasses import dataclass
from typing import AsyncIterator
from typing import Protocol
from typing import runtime_checkable

from jaxtyping import Shaped
from torch import Tensor

type Rank = int
type WorldSize = int


@runtime_checkable
class SamplingParams(Protocol):
    """LLM sampling parameters for generation.

    Reference:
    - https://docs.vllm.ai/en/v0.6.0/dev/sampling_params.html
    """

    temperature: float
    max_tokens: int
    logprobs: int | None
    group_size: int = 1


@runtime_checkable
class SampleLogprob(Protocol):
    """Token-level log-probability metadata protocol."""

    logprob: float
    rank: int | None = None
    decoded_token: str | None = None


@runtime_checkable
class CompletionOutput(Protocol):
    """Candidate completion output protocol emitted by an LLM engine."""

    @property
    def index(self) -> int: ...
    @property
    def text(self) -> str: ...
    @property
    def token_ids(self) -> Sequence[int]: ...
    @property
    def cumulative_logprob(self) -> float | None: ...
    @property
    def logprobs(
        self,
    ) -> Sequence[Mapping[int, SampleLogprob | float] | None] | None: ...
    @property
    def finish_reason(self) -> str | None: ...


@runtime_checkable
class RequestOutput(Protocol):
    """Request-level generation result protocol emitted by an LLM engine."""

    @property
    def request_id(self) -> str: ...
    @property
    def prompt(self) -> str | None: ...
    @property
    def prompt_token_ids(self) -> Sequence[int] | None: ...
    @property
    def outputs(self) -> Sequence[CompletionOutput]: ...
    @property
    def finished(self) -> bool: ...


@runtime_checkable
class AsyncLLMEngine(Protocol):
    """Asynchronous LLM generation engine protocol (e.g. vLLM engine)."""

    def generate(
        self,
        prompt: str,
        sampling_params: SamplingParams,
        request_id: str,
        *args,
        **kwargs,
    ) -> AsyncIterator[RequestOutput]: ...

    async def update_weights(
        self, weights: Mapping[str, Shaped[Tensor, "..."]]
    ) -> None: ...


@dataclass
class RolloutConfig:
    """Configuration for distributed rollout generation."""

    model_name: str = "test-model"
    backend: str | None = None
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
