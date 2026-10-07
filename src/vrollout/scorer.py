"""Trajectory evaluation and reward scoring routines."""

import asyncio
from dataclasses import dataclass
from typing import Any

from vrollout import async_queue
from vrollout import types


@dataclass(frozen=True, slots=True)
class ScoredRollout:
    """Rollout trajectory with corresponding reward score."""

    reward: float
    trajectory: types.RolloutTrajectory


async def score_rollouts(
    reward_queue: async_queue.AsyncQueue,
    verifier: Any,
    stop_event: asyncio.Event,
    batch_size: int,
    timeout_sec: float = 0.1,
) -> list[ScoredRollout]:
    """Consumes rollouts from queue and scores them using a verifier."""
    scored_rollouts = []
    while not stop_event.is_set() or reward_queue.qsize() > 0:
        batch = await reward_queue.get_batch(batch_size, timeout_sec)
        if not batch:
            continue

        tasks = [verifier(rollout) for rollout in batch]
        rewards = await asyncio.gather(*tasks)

        for rollout, reward in zip(batch, rewards, strict=True):
            scored_rollout = ScoredRollout(reward=reward, trajectory=rollout)
            scored_rollouts.append(scored_rollout)

        reward_queue.task_done(len(batch))

    return scored_rollouts
