"""Unit tests for RolloutRunner background generation."""

import asyncio
from collections.abc import Sequence
import threading
from typing import Any

import pytest

from vrollout import rollout_runner
from vrollout import scorer
from vrollout import types


def _scored_rollout(prompt: str) -> scorer.ScoredRollout:
    return scorer.ScoredRollout(
        reward=1.0,
        trajectory=types.RolloutTrajectory(
            request_id=prompt,
            prompt=prompt,
            prompt_token_ids=[1],
            completion_token_ids=[2],
            logprobs=[-0.1],
            finish_reason="stop",
            text="",
            worker_id=0,
        ),
    )


class _GatedPipeline:
    """Fake pipeline that finishes only once `release` is set."""

    def __init__(self) -> None:
        self.release = threading.Event()
        self.thread_ids: list[int] = []
        self.cancelled = False

    async def generate_and_score(
        self, prompts: Sequence[str], sampling_params: Any
    ) -> list[scorer.ScoredRollout]:
        _ = sampling_params  # unused
        self.thread_ids.append(threading.get_ident())
        try:
            while not self.release.is_set():
                await asyncio.sleep(0.001)
        except asyncio.CancelledError:
            self.cancelled = True
            raise
        return [_scored_rollout(p) for p in prompts]


class _FailingPipeline:
    async def generate_and_score(
        self, prompts: Sequence[str], sampling_params: Any
    ) -> list[scorer.ScoredRollout]:
        raise KeyError(prompts[0])


def test_submit_returns_while_batch_generates_off_main_thread() -> None:
    fake = _GatedPipeline()

    with rollout_runner.RolloutRunner(fake) as runner:
        future = runner.submit(["a", "b"], None)
        running_after_submit = not future.done()
        fake.release.set()
        batch = future.result(timeout=5)

    assert running_after_submit
    assert fake.thread_ids == [runner._thread.ident]
    assert [sr.trajectory.prompt for sr in batch.scored_rollouts] == ["a", "b"]
    assert batch.generation_time_s > 0.0


def test_future_raises_pipeline_error() -> None:
    with rollout_runner.RolloutRunner(_FailingPipeline()) as runner:
        future = runner.submit(["a"], None)

        with pytest.raises(KeyError):
            future.result(timeout=5)


def test_exit_cancels_in_flight_batch_and_stops_thread() -> None:
    fake = _GatedPipeline()
    runner = rollout_runner.RolloutRunner(fake)

    with runner:
        future = runner.submit(["a"], None)
        while not fake.thread_ids:
            threading.Event().wait(0.001)

    assert fake.cancelled
    assert future.cancelled()
    assert not runner._thread.is_alive()
