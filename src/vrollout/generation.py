"""Distributed LLM rollout generation across Ray actors."""

import asyncio
from collections.abc import Callable
from collections.abc import Mapping
from collections.abc import Sequence
from dataclasses import dataclass
import itertools
from typing import Any
from typing import cast

from beartype import beartype
from jaxtyping import jaxtyped
from jaxtyping import Shaped
import ray
from ray.actor import ActorClass
from ray.actor import ActorProxy
from torch import Tensor

from vrollout.testing import MockAsyncLLMEngine
from vrollout.types import AsyncLLMEngine
from vrollout.types import RequestOutput
from vrollout.types import RolloutConfig
from vrollout.types import RolloutTrajectory
from vrollout.types import SampleLogprob
from vrollout.types import SamplingParams


def _parse_vllm_output_to_rollout(
    request_output: RequestOutput,
    worker_id: int,
) -> RolloutTrajectory:
    """Extracts RL training tensors and metadata from a finished vLLM RequestOutput."""
    if not request_output.outputs:
        raise ValueError(
            f"RequestOutput contains no completion outputs: {request_output.request_id}"
        )
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


def _create_llm_engine(config: RolloutConfig) -> AsyncLLMEngine:
    """Creates a default vLLM AsyncLLMEngine instance from configuration."""
    if config.model_name.startswith("test"):
        return MockAsyncLLMEngine(model_name=config.model_name)

    raise NotImplementedError("To be implemented.")


@dataclass(frozen=True)
class RolloutWorkerStatus:
    """Rollout worker status for inspection."""

    worker_id: int
    llm_engine_type: str
    weights_version: int


class RolloutWorker:
    """Worker for continuous rollout generation via LLM engine."""

    def __init__(
        self,
        worker_id: int,
        config: RolloutConfig,
        llm_engine: AsyncLLMEngine | None = None,
    ) -> None:
        self._worker_id = worker_id
        self._config = config
        self._llm_engine = (
            _create_llm_engine(self._config) if llm_engine is None else llm_engine
        )
        self._weights_version = 0

    async def _generate_single(
        self, prompt: str, sampling_params: SamplingParams, request_id: str
    ) -> RolloutTrajectory:
        assert self._llm_engine is not None
        generator = self._llm_engine.generate(prompt, sampling_params, request_id)
        final_output = None
        async for output in generator:
            final_output = output

        if final_output is None:
            raise RuntimeError(f"Engine produced no output for request {request_id}")

        return _parse_vllm_output_to_rollout(final_output, self._worker_id)

    async def generate_batch(
        self, prompts: Sequence[str], sampling_params: SamplingParams
    ) -> list[RolloutTrajectory]:
        """Generates rollout trajectories asynchronouslyfor a batch of prompts."""
        tasks = [
            self._generate_single(p, sampling_params, f"request_{i}")
            for i, p in enumerate(prompts)
        ]
        return await asyncio.gather(*tasks)

    @jaxtyped(typechecker=beartype)
    def update_weights(self, weights: Mapping[str, Shaped[Tensor, "..."]]) -> None:
        """Updates engine weights, bumping weights version."""
        if self._llm_engine is not None and hasattr(self._llm_engine, "update_weights"):
            update_fn = cast(Callable[[Any], None], self._llm_engine.update_weights)
            update_fn(weights)
        self._weights_version += 1

    def get_status(self) -> RolloutWorkerStatus:
        """Returns snapshot of worker status."""
        return RolloutWorkerStatus(
            worker_id=self._worker_id,
            llm_engine_type=type(self._llm_engine).__name__,
            weights_version=self._weights_version,
        )


# use ray.remote function style to aid type checking
ActorRolloutWorker: ActorClass[RolloutWorker] = ray.remote(RolloutWorker)


class RolloutManager:
    """Manages a pool of Ray `RolloutWorker` actors and distributes prompt batches."""

    def __init__(
        self,
        config: RolloutConfig,
    ) -> None:
        self._config = config
        self._workers = [
            ActorRolloutWorker.remote(i, config) for i in range(config.num_workers)
        ]
        self._worker_cycler = itertools.cycle(self._workers)

    async def generate(
        self,
        prompts: Sequence[str],
        sampling_params: SamplingParams,
        chunk_size: int = 8,
    ) -> list[RolloutTrajectory]:
        """Dispatches prompt chunks in round-robin order to Ray rollout workers."""
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {chunk_size}")

        tasks = []
        for chunk in itertools.batched(prompts, chunk_size):
            worker = next(self._worker_cycler)
            task = worker.generate_batch.remote(chunk, sampling_params)  # ty: ignore[unresolved-attribute]
            tasks.append(task)

        results = await asyncio.gather(*tasks)
        # flatten results
        results = [r for chunk in results for r in chunk]
        return results

    @jaxtyped(typechecker=beartype)
    def update_weights(self, weights: Mapping[str, Shaped[Tensor, "..."]]) -> None:
        """Broadcasts updated weights to all remote Ray workers via Ray object store."""
        ref = ray.put(weights)
        ray.get([w.update_weights.remote(ref) for w in self._workers])  # ty: ignore[unresolved-attribute]

    @property
    def workers(self) -> list[ActorProxy[RolloutWorker]]:
        return self._workers
