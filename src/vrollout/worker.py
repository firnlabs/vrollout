"""Rollout worker."""

from dataclasses import dataclass

import ray
from ray.actor import ActorClass


@dataclass
class RolloutConfig:
    model_name: str = "mock-model"  # default use of mock LLM engine
    num_workers: int = 2
    gpus_per_worker: float = 0.0  # default 0.0 for local testing
    max_batch_size: int = 64
    max_seq_len: int = 512
    temperature: float = 1.0
    gpu_memory_utilization: float = 0.60
    use_mock_engine: bool = True


class RolloutWorker:
    """Worker for continuous rollout integration."""

    def __init__(self, worker_id: int, config: RolloutConfig) -> None:
        self._worker_id = worker_id
        self._config = config

    @ray.method
    async def generate_batch(self, prompts: list[str]) -> list[str]:
        return prompts


# use ray.remote function style to aid type checking
ActorRolloutWorker: ActorClass[RolloutWorker] = ray.remote(RolloutWorker)
