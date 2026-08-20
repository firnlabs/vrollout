"""Implements asynchronous queue."""

import asyncio
from collections.abc import Sequence
import logging
from typing import Any


class AsyncQueue:
    """Asynchronous queue with backpressure handling."""

    def __init__(self, maxsize: int = 1000):
        self._maxsize = maxsize
        self._queue = asyncio.Queue(maxsize=self._maxsize)

    async def put_items(self, items: Sequence[Any]) -> None:
        """Enqueues a batch of completed rollouts."""
        for item in items:
            await self._queue.put(item)

    async def get_batch(
        self, batch_size: int = 32, timeout_sec: float = 0.1
    ) -> list[Any]:
        """Returns a batch of items from the queue."""
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
        """Returns the number of items in the queue."""
        return self._queue.qsize()

    def task_done(self, n: int = 1) -> None:
        """Marks a set of items retrieved from the queue as 'done'."""
        for _ in range(n):
            self._queue.task_done()

    async def join(self) -> None:
        """Block and wait until all unfinished tasks (items) are done."""
        await self._queue.join()
