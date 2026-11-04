"""Unit tests for RolloutPipeline chunk dispatch and ordering."""

import asyncio
from collections.abc import Awaitable
from collections.abc import Sequence
import itertools
from typing import Any

import pytest

from vrollout import pipeline
from vrollout import types


def _create_trajectory(prompt: str) -> types.RolloutTrajectory:
    return types.RolloutTrajectory(
        request_id=prompt,
        prompt=prompt,
        prompt_token_ids=[1],
        completion_token_ids=[2],
        logprobs=[-0.1],
        finish_reason="stop",
        text="",
        worker_id=0,
    )


class _SlowRolloutManager:
    """Fake manager whose earlier chunks finish last, tracking chunks in flight."""

    def __init__(self) -> None:
        self.in_flight = 0
        self.max_in_flight = 0

    async def _generate(
        self, chunk: Sequence[str], delay: float
    ) -> list[types.RolloutTrajectory]:
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        await asyncio.sleep(delay)
        self.in_flight -= 1
        return [_create_trajectory(p) for p in chunk]

    def submit(
        self, prompts: Sequence[str], sampling_params: Any, chunk_size: int
    ) -> list[Awaitable[list[types.RolloutTrajectory]]]:
        _ = sampling_params  # unused
        chunks = list(itertools.batched(prompts, chunk_size))
        return [
            asyncio.ensure_future(self._generate(c, 0.01 * (len(chunks) - i)))
            for i, c in enumerate(chunks)
        ]


async def _constant_verifier(rollout: types.RolloutTrajectory) -> float:
    _ = rollout  # unused
    return 1.0


@pytest.mark.asyncio
async def test_generate_and_score_dispatches_all_chunks_concurrently() -> None:
    manager = _SlowRolloutManager()
    config = pipeline.PipelineConfig(chunk_size=2)
    pipe = pipeline.RolloutPipeline(manager, _constant_verifier, config)
    prompts = [f"Prompt {i}" for i in range(8)]

    await pipe.generate_and_score(prompts, None)

    assert manager.max_in_flight == 4


@pytest.mark.asyncio
async def test_generate_and_score_preserves_prompt_order() -> None:
    manager = _SlowRolloutManager()
    config = pipeline.PipelineConfig(chunk_size=2, scoring_batch_size=3)
    pipe = pipeline.RolloutPipeline(manager, _constant_verifier, config)
    prompts = [f"Prompt {i}" for i in range(8)]

    results = await pipe.generate_and_score(prompts, None)

    assert [r.trajectory.prompt for r in results] == prompts


async def _failing_verifier(rollout: types.RolloutTrajectory) -> float:
    raise KeyError(rollout.prompt)


@pytest.mark.asyncio
async def test_generate_and_score_propagates_verifier_error() -> None:
    manager = _SlowRolloutManager()
    config = pipeline.PipelineConfig(chunk_size=2, scoring_batch_size=1)
    pipe = pipeline.RolloutPipeline(manager, _failing_verifier, config)
    prompts = [f"Prompt {i}" for i in range(8)]

    with pytest.raises(KeyError):
        async with asyncio.timeout(5):
            await pipe.generate_and_score(prompts, None)
