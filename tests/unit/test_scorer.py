import asyncio

import pytest

from vrollout import async_queue
from vrollout import scorer
from vrollout import types


@pytest.fixture
def queue() -> async_queue.AsyncQueue:
    return async_queue.AsyncQueue()


@pytest.mark.asyncio
async def test_score_rollouts(queue: async_queue.AsyncQueue, mock_verifier):
    stop_event = asyncio.Event()
    batch_size = 2
    num_rollouts = 8
    items = [
        types.RolloutTrajectory(
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


@pytest.mark.asyncio
async def test_score_rollouts_handle_verifier_failure(
    queue: async_queue.AsyncQueue,
):
    stop_event = asyncio.Event()

    async def failing_verifier(rollout: types.RolloutTrajectory) -> float:
        raise RuntimeError("Simulated verifier failure (e.g., parse/OOM error)")

    item = types.RolloutTrajectory(
        request_id="req-fail-0",
        prompt="test",
        prompt_token_ids=[],
        completion_token_ids=[],
        logprobs=[],
        finish_reason="stop",
        text="",
        worker_id=0,
    )
    await queue.put_items([item])

    scorer_task = asyncio.create_task(
        scorer.score_rollouts(
            reward_queue=queue,
            verifier=failing_verifier,
            stop_event=stop_event,
            batch_size=1,
        )
    )

    async with asyncio.timeout(0.5):
        await queue.join()

    assert queue.qsize() == 0
    assert scorer_task.done()
    with pytest.raises(RuntimeError, match="Simulated verifier failure"):
        await scorer_task
