"""Rollout worker."""

import ray
from ray.actor import ActorClass


class RolloutWorker:
    """Worker for continuous rollout integration."""

    @ray.method
    async def generate_batch(self, prompts: list[str]) -> list[str]:
        return prompts


# use ray.remote function style to aid type checking
ActorRolloutWorker: ActorClass[RolloutWorker] = ray.remote(RolloutWorker)
