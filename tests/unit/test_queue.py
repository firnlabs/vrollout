"""Tests for queue.py."""

import asyncio

import pytest

from vrollout import queue


@pytest.fixture
def queue_() -> queue.AsyncRewardQueue:
    return queue.AsyncRewardQueue(maxsize=3)


class TestAsyncRewardQueue:
    """Tests for AsyncRewardQueue class."""

    @pytest.mark.asyncio
    async def test_put_rollouts(self, queue_):
        rollouts = [1, 2, 3]

        await queue_.put_rollouts(rollouts)

        results = []
        for _ in range(len(rollouts)):
            item = await asyncio.wait_for(queue_._queue.get(), timeout=2)
            results.append(item)

        assert results == rollouts
