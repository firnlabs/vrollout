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
    """Configuration for distributed rollout generation.

    Attributes:
        model_name: Hugging Face model ID or local path loaded by each worker.
        backend: Engine backend. `"mock"` forces the mock engine; `None`
            selects the mock engine when `model_name` starts with `"test"`
            and vLLM otherwise.
        num_workers: Number of Ray rollout worker actors to launch.
        gpus_per_worker: Ray GPU reservation per worker. Fractional values share
            a GPU across workers; `0.0` schedules workers without GPUs (local
            testing).
        max_num_seqs: Maximum number of sequences each vLLM engine schedules at
            once; `None` uses vLLM's default.
        max_seq_len: Maximum model context length (prompt plus completion
            tokens), passed to vLLM as `max_model_len`.
        gpu_memory_utilization: Fraction of GPU memory vLLM may reserve for
            weights and KV cache.
        enforce_eager: Whether vLLM runs in eager mode, skipping CUDA graph
            capture for faster startup at the cost of throughput.
        seed: Base sampling seed. Each worker offsets it by its worker ID so
            workers draw distinct samples; `None` leaves RNGs unseeded.
    """

    model_name: str = "test-model"
    backend: str | None = None
    num_workers: int = 2
    gpus_per_worker: float = 0.0
    max_num_seqs: int | None = None
    max_seq_len: int = 512
    gpu_memory_utilization: float = 0.60
    enforce_eager: bool = True
    seed: int | None = None


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
