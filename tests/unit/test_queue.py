"""Tests for queue.py."""

import asyncio
import logging
import time

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
    async def test_get_batch(self, queue_: queue.AsyncQueue, queue_size: int):
        items = list(range(queue_size))
        batch_size = 4
        num_batches = len(items) // batch_size + 1
        for item in items:
            queue_._queue.put_nowait(item)

        batches = [
            await queue_.get_batch(batch_size=batch_size) for _ in range(num_batches)
        ]

        assert len(batches) == num_batches
        assert batches[0] == [0, 1, 2, 3]
        assert batches[1] == [4, 5, 6, 7]
        assert batches[2] == [8, 9]

    @pytest.mark.asyncio
    async def test_get_batch_timeout_on_empty_queue(self, queue_: queue.AsyncQueue):
        timeout_sec = 0.05
        start = time.time()

        batch = await queue_.get_batch(batch_size=4, timeout_sec=timeout_sec)

        elapsed = time.time() - start
        assert batch == []
        assert elapsed >= timeout_sec

    @pytest.mark.asyncio
    async def test_get_batch_partial_batch_on_timeout(
        self, queue_: queue.AsyncQueue, caplog: pytest.LogCaptureFixture
    ):
        caplog.set_level(logging.DEBUG)
        await queue_.put_items([1, 2])
        timeout_sec = 0.05
        start_time = time.time()

        batch = await queue_.get_batch(batch_size=5, timeout_sec=timeout_sec)

        elapsed = time.time() - start_time
        assert batch == [1, 2]
        assert elapsed >= timeout_sec
        assert "Timeout reached" in caplog.text

    def test_qsize(self, queue_: queue.AsyncQueue):
        queue_._queue.put_nowait(0)
        queue_._queue.put_nowait(1)

        assert queue_.qsize() == 2

    @pytest.mark.asyncio
    async def test_join_all_tasks_done(self, queue_: queue.AsyncQueue):
        for i in range(3):
            await queue_._queue.put(i)

        # Start join() in the background
        join_task = asyncio.create_task(queue_.join())

        # Yield control: join_task starts and blocks because unfinished_tasks == 3
        await asyncio.sleep(0)
        assert not join_task.done()

        # Mark 2 tasks done: still 1 remaining, join_task should still be blocked
        queue_.task_done(n=2)
        await asyncio.sleep(0)
        assert not join_task.done()

        # Mark final task done: join_task should complete
        queue_.task_done(n=1)
        await asyncio.wait_for(join_task, timeout=1.0)
        assert join_task.done()

    @pytest.mark.asyncio
    async def test_join_on_empty_queue(self, queue_: queue.AsyncQueue):
        """Tests that join() immediately returns when queue has no unfinished tasks."""
        timeout_sec = 0.1
        start_time = time.time()

        # Should complete immediately without blocking/timing out
        await asyncio.wait_for(queue_.join(), timeout=timeout_sec)

        elapsed = time.time() - start_time
        assert elapsed <= timeout_sec
