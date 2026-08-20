"""Tests for queue.py."""

import asyncio

import pytest

from vrollout import queue


@pytest.fixture
def queue_size() -> int:
    return 10


@pytest.fixture
def queue_(queue_size: int) -> queue.AsyncRewardQueue:
    return queue.AsyncRewardQueue(maxsize=queue_size)


class TestAsyncRewardQueue:
    """Tests for AsyncRewardQueue class."""

    @pytest.mark.asyncio
    async def test_put_rollouts(self, queue_: queue.AsyncRewardQueue):
        rollouts = [1, 2, 3]

        await queue_.put_rollouts(rollouts)

        results = []
        for _ in range(len(rollouts)):
            item = await asyncio.wait_for(queue_._queue.get(), timeout=2)
            results.append(item)
        assert results == rollouts

    @pytest.mark.asyncio
    async def test_get_batch_for_scoring(
        self, queue_: queue.AsyncRewardQueue, queue_size: int
    ):
        rollouts = list(range(queue_size))
        batch_size = 4
        num_batches = len(rollouts) // batch_size + 1
        for rollout in rollouts:
            queue_._queue.put_nowait(rollout)

        batches = []
        for _ in range(num_batches):
            batch = await queue_.get_batch_for_scoring(batch_size=batch_size)
            batches.append(batch)

        assert len(batches) == num_batches
        assert batches[0] == [0, 1, 2, 3]
        assert batches[1] == [4, 5, 6, 7]
        assert batches[2] == [8, 9]
