"""Rollout worker."""

from dataclasses import dataclass
from typing import Protocol

import ray
from ray.actor import ActorClass


class LLMEngine(Protocol):
    """Protocol for defiining interface of an LLM engine."""

    async def generate(self, prompt: str, *args, **kwargs): ...


@dataclass
class RolloutConfig:
    model_name: str = "mock-model"
    num_workers: int = 2
    gpus_per_worker: float = 0.0  # default 0.0 for local testing
    max_batch_size: int = 64
    max_seq_len: int = 512
    temperature: float = 1.0
    gpu_memory_utilization: float = 0.60


def _create_default_llm_engine(config: RolloutConfig) -> LLMEngine:
    raise NotImplementedError("To be implemented")


class RolloutWorker:
    """Worker for continuous rollout integration."""

    def __init__(
        self, worker_id: int, config: RolloutConfig, llm_engine: LLMEngine | None = None
    ) -> None:
        self._worker_id = worker_id
        self._config = config
        self._llm_engine = (
            llm_engine
            if llm_engine is not None
            else _create_default_llm_engine(self._config)
        )

    @ray.method
    async def generate_batch(self, prompts: list[str]) -> list[str]:
        return prompts


# use ray.remote function style to aid type checking
ActorRolloutWorker: ActorClass[RolloutWorker] = ray.remote(RolloutWorker)
