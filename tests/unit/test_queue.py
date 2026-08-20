"""Tests for queue.py."""

import asyncio

import pytest

from vrollout import queue


@pytest.fixture
def queue_size() -> int:
    return 10


@pytest.fixture
def queue_(queue_size: int) -> queue.AsyncQueue:
    return queue.AsyncQueue(maxsize=queue_size)


class TestAsyncQueue:
    """Tests for AsyncQueue class."""

    @pytest.mark.asyncio
    async def test_put_rollouts(self, queue_: queue.AsyncQueue):
        items = [1, 2, 3]

        await queue_.put_items(items)

        results = []
        for _ in range(len(items)):
            item = await asyncio.wait_for(queue_._queue.get(), timeout=2)
            results.append(item)
        assert results == items

    @pytest.mark.asyncio
    async def test_get_batch_for_scoring(
        self, queue_: queue.AsyncQueue, queue_size: int
    ):
        items = list(range(queue_size))
        batch_size = 4
        num_batches = len(items) // batch_size + 1
        for rollout in items:
            queue_._queue.put_nowait(rollout)

        batches = []
        for _ in range(num_batches):
            batch = await queue_.get_batch(batch_size=batch_size)
            batches.append(batch)

        assert len(batches) == num_batches
        assert batches[0] == [0, 1, 2, 3]
        assert batches[1] == [4, 5, 6, 7]
        assert batches[2] == [8, 9]
