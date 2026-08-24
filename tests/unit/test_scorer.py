"""Tests for scorer.py."""

import asyncio

import pytest

from vrollout import async_queue
from vrollout import scorer
from vrollout import worker


@pytest.fixture
def queue() -> async_queue.AsyncQueue:
    return async_queue.AsyncQueue()


@pytest.mark.asyncio
async def test_score_rollouts(queue: async_queue.AsyncQueue, mock_verifier):
    stop_event = asyncio.Event()
    batch_size = 2
    num_rollouts = 8
    items = [
        worker.RolloutTrajectory(
            request_id=f"req-{i}",
            prompt="",
            prompt_token_ids=[],
            completion_token_ids=[],
            logprobs=[],
            finish_reason="stop",
            text="",
            worker_id=i,
        )
        for i in range(num_rollouts)
    ]
    await queue.put_items(items)

    score_task = asyncio.create_task(
        scorer.score_rollouts(
            reward_queue=queue,
            verifier=mock_verifier,
            stop_event=stop_event,
            batch_size=batch_size,
        )
    )

    await queue.join()
    stop_event.set()
    results = await score_task

    assert len(results) == num_rollouts
    for r in results:
        exp_reward = float(r.trajectory.worker_id % 2)
        assert r.reward == exp_reward
