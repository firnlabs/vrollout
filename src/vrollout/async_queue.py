"""Asynchronous bounded queue with batched retrieval and backpressure."""

import asyncio
from collections.abc import Sequence
import logging
from typing import Any


class AsyncQueue:
    """Asynchronous queue with backpressure handling for rollout pipelining."""

    def __init__(self, maxsize: int = 1000):
        self._maxsize = maxsize
        self._queue = asyncio.Queue(maxsize=self._maxsize)

    async def put_items(self, items: Sequence[Any]) -> None:
        """Enqueues items sequentially, applying backpressure when the queue is full."""
        for item in items:
            await self._queue.put(item)

    async def get_batch(
        self, batch_size: int = 32, timeout_sec: float = 0.1
    ) -> list[Any]:
        """Collects up to `batch_size` items within timeout window."""
        batches = []
        try:
            async with asyncio.timeout(timeout_sec):
                while len(batches) < batch_size:
                    item = await self._queue.get()
                    batches.append(item)
        except asyncio.TimeoutError:
            logging.debug("Timeout reached")

        return batches

    def qsize(self) -> int:
        """Returns the number of items currently in the queue."""
        return self._queue.qsize()

    def task_done(self, n: int = 1) -> None:
        """Marks `n` processed items as finished for queue task accounting."""
        for _ in range(n):
            self._queue.task_done()

    async def join(self) -> None:
        """Blocks until all enqueued items have been marked finished via `task_done`."""
        await self._queue.join()
