"""Implements asynchronous queue."""

import asyncio
from collections.abc import Sequence
import logging
import time
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

    async def get_batch(self, batch_size: int = 32, timeout: float = 0.1) -> list[Any]:
        batch = []
        deadline = time.time() + timeout

        while len(batch) < batch_size:
            # Try to get any items in the queue immediately
            try:
                item = self._queue.get_nowait()
            except asyncio.QueueEmpty:
                logging.debug("Queue is empty.")
            else:
                batch.append(item)
                continue

            time_left = deadline - time.time()
            if time_left <= 0:
                break

            try:
                item = await asyncio.wait_for(self._queue.get(), timeout=time_left)
            except asyncio.TimeoutError:
                logging.debug("Timeout reached.")
                break
            else:
                batch.append(item)

        return batch

    def qsize(self) -> int:
        raise NotImplementedError

    def task_done(self, n: int = 1) -> None:
        raise NotImplementedError

    async def join(self) -> None:
        raise NotImplementedError
