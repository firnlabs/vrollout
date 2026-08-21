"""Rollout worker."""

import asyncio
from collections.abc import Mapping
from collections.abc import Sequence
from dataclasses import dataclass
from typing import AsyncGenerator
from typing import Protocol
from typing import runtime_checkable

import ray
from ray.actor import ActorClass


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
    model_name: str = "mock-model"
    num_workers: int = 2
    gpus_per_worker: float = 0.0  # default 0.0 for local testing
    max_batch_size: int = 64
    max_seq_len: int = 512
    temperature: float = 1.0
    gpu_memory_utilization: float = 0.60


@dataclass(frozen=True)
class RolloutTrajectory:
    request_id: str
    prompt: str
    prompt_token_ids: Sequence[int]
    completion_token_ids: Sequence[int]
    logprobs: Sequence[float]
    finish_reason: str
    text: str
    worker_id: int


def _parse_vllm_output_to_rollout(
    request_output: RequestOutput,
    worker_id: int,
) -> RolloutTrajectory:
    """Extracts RL training tensors and metadata from a finished vLLM RequestOutput."""
    r = request_output
    c = r.outputs[0]
    logprobs = []
    if c.logprobs is None:
        logprobs = [0.0 for _ in range(len(c.token_ids))]
    else:
        for lp_dict, token_id in zip(c.logprobs, c.token_ids, strict=True):
            assert token_id in lp_dict
            logprob = lp_dict[token_id]
            if isinstance(logprob, SampleLogprob):
                logprob = logprob.logprob
            logprobs.append(logprob)

    return RolloutTrajectory(
        request_id=r.request_id,
        prompt=r.prompt,
        prompt_token_ids=r.prompt_token_ids,
        completion_token_ids=c.token_ids,
        logprobs=logprobs,
        finish_reason=c.finish_reason or "stop",
        text=c.text,
        worker_id=worker_id,
    )


def _create_default_llm_engine(config: RolloutConfig) -> AsyncLLMEngine:
    raise NotImplementedError("To be implemented")


class RolloutWorker:
    """Worker for continuous rollout integration."""

    def __init__(
        self,
        worker_id: int,
        config: RolloutConfig,
        llm_engine: AsyncLLMEngine | None = None,
    ) -> None:
        self._worker_id = worker_id
        self._config = config
        self._llm_engine = (
            llm_engine
            if llm_engine is not None
            else _create_default_llm_engine(self._config)
        )

    async def _generate_single(
        self, prompt: str, sampling_params: SamplingParams, request_id: str
    ) -> RolloutTrajectory:
        assert self._llm_engine is not None
        generator = self._llm_engine.generate(prompt, sampling_params, request_id)
        final_output = None
        async for output in generator:
            final_output = output

        return _parse_vllm_output_to_rollout(final_output, self._worker_id)

    @ray.method
    async def generate_batch(
        self, prompts: Sequence[str], sampling_params: SamplingParams
    ) -> list[RolloutTrajectory]:
        tasks = [
            self._generate_single(p, sampling_params, f"request_{i}")
            for i, p in enumerate(prompts)
        ]
        return await asyncio.gather(*tasks)


# use ray.remote function style to aid type checking
ActorRolloutWorker: ActorClass[RolloutWorker] = ray.remote(RolloutWorker)
