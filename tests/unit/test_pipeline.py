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
        self.chunks: list[asyncio.Future[list[types.RolloutTrajectory]]] = []

    async def _generate(
        self, chunk: Sequence[str], delay: float
    ) -> list[types.RolloutTrajectory]:
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        await asyncio.sleep(delay)
        self.in_flight -= 1
        return [_create_trajectory(p) for p in chunk]

    def _delay(self, index: int, num_chunks: int) -> float:
        return 0.01 * (num_chunks - index)

    def submit(
        self, prompts: Sequence[str], sampling_params: Any, chunk_size: int
    ) -> list[Awaitable[list[types.RolloutTrajectory]]]:
        _ = sampling_params  # unused
        chunks = list(itertools.batched(prompts, chunk_size))
        self.chunks = [
            asyncio.ensure_future(self._generate(c, self._delay(i, len(chunks))))
            for i, c in enumerate(chunks)
        ]
        return list(self.chunks)

    def cancel(self, refs: Sequence[Awaitable[Any]]) -> None:
        for ref in refs:
            assert isinstance(ref, asyncio.Future)
            ref.cancel()


class _ChunkError(RuntimeError):
    pass


class _FailingChunkManager(_SlowRolloutManager):
    """Fake manager whose second chunk fails fast while the others run for long."""

    def _delay(self, index: int, num_chunks: int) -> float:
        _ = num_chunks  # unused
        return 0.0 if index == 0 else 60.0

    async def _generate(
        self, chunk: Sequence[str], delay: float
    ) -> list[types.RolloutTrajectory]:
        if chunk[0] == "Prompt 2":
            raise _ChunkError(chunk[0])
        return await super()._generate(chunk, delay)


def _pending_tasks() -> set[asyncio.Task[Any]]:
    return {t for t in asyncio.all_tasks() if t is not asyncio.current_task()}


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


@pytest.mark.asyncio
async def test_generate_and_score_cleans_up_after_chunk_error() -> None:
    manager = _FailingChunkManager()
    config = pipeline.PipelineConfig(chunk_size=2)
    pipe = pipeline.RolloutPipeline(manager, _constant_verifier, config)
    prompts = [f"Prompt {i}" for i in range(8)]

    with pytest.raises(_ChunkError):
        async with asyncio.timeout(5):
            await pipe.generate_and_score(prompts, None)

    assert [c.cancelled() for c in manager.chunks] == [False, False, True, True]
    assert _pending_tasks() == set()


@pytest.mark.asyncio
async def test_generate_and_score_cancels_chunks_after_verifier_error() -> None:
    manager = _FailingChunkManager()
    config = pipeline.PipelineConfig(chunk_size=2, scoring_batch_size=1)
    pipe = pipeline.RolloutPipeline(manager, _failing_verifier, config)
    # prompts 2 and 3 are skipped to avoid triggering _ChunkError
    prompts = [f"Prompt {i}" for i in [0, 1, 4, 5, 6, 7]]

    with pytest.raises(KeyError):
        async with asyncio.timeout(5):
            await pipe.generate_and_score(prompts, None)

    assert [c.cancelled() for c in manager.chunks] == [False, True, True]
    assert _pending_tasks() == set()


async def _wait_for_first_chunk(manager: _SlowRolloutManager) -> None:
    async with asyncio.timeout(5):
        while not (manager.chunks and manager.chunks[0].done()):
            await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_generate_and_score_cleans_up_when_cancelled() -> None:
    manager = _FailingChunkManager()
    config = pipeline.PipelineConfig(chunk_size=2)
    pipe = pipeline.RolloutPipeline(manager, _constant_verifier, config)
    # prompts 2 and 3 are skipped to avoid triggering _ChunkError
    prompts = [f"Prompt {i}" for i in [0, 1, 4, 5, 6, 7]]
    task = asyncio.create_task(pipe.generate_and_score(prompts, None))
    await _wait_for_first_chunk(manager)

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert [c.cancelled() for c in manager.chunks] == [False, True, True]
    assert _pending_tasks() == set()


class _RejectingManager(_SlowRolloutManager):
    """Fake manager whose submit rejects the batch before dispatching anything."""

    def submit(
        self, prompts: Sequence[str], sampling_params: Any, chunk_size: int
    ) -> list[Awaitable[list[types.RolloutTrajectory]]]:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")


@pytest.mark.asyncio
async def test_generate_and_score_leaves_no_tasks_when_submit_fails() -> None:
    pipe = pipeline.RolloutPipeline(_RejectingManager(), _constant_verifier)

    with pytest.raises(ValueError):
        await pipe.generate_and_score(["Prompt 0"], None)

    assert _pending_tasks() == set()
