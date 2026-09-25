"""Distributed RL training engine."""

from dataclasses import dataclass

import torch
from torch import distributed as dist
from torch import nn
from torch import optim

from vrollout import buffer
from vrollout import sync
from vrollout import trainer
from vrollout.scorer import ScoredRollout
from vrollout.sync import WeightsReceiver
from vrollout.types import Rank
from vrollout.types import WorldSize


@dataclass(frozen=True)
class RLEngineConfig:
    pad_token_id: int = 0
    beta: float = 0.1
    train_steps_per_iteration: int = 1


@dataclass(frozen=True)
class TrainIterationMetrics:
    avg_loss: float
    rank: int


class DistributedRLEngine:
    """Distributed RL training engine."""

    def __init__(
        self,
        *,
        actor: nn.Module,
        reference: nn.Module,
        optimizer: optim.Optimizer,
        receivers: list[WeightsReceiver] | None = None,
        rank: Rank | None = None,
        world_size: WorldSize | None = None,
        config: RLEngineConfig | None = None,
    ):
        if rank is None:
            self._rank = dist.get_rank() if dist.is_initialized() else 0
        else:
            self._rank = rank
        if world_size is None:
            self._world_size = dist.get_world_size() if dist.is_initialized() else 1
        else:
            self._world_size = world_size

        self._config = config or RLEngineConfig()
        self._actor = actor
        self._reference = reference
        self._optimizer = optimizer
        self._receivers = receivers or []

        self._trainer = trainer.ModelTrainer(
            actor=self._actor,
            reference=self._reference,
            optimizer=self._optimizer,
            beta=self._config.beta,
        )

        self._broadcaster = sync.WeightBroadcaster(
            rank=self._rank, receivers=self._receivers
        )

    def train_iteration(
        self, scored_rollouts: list[ScoredRollout], device: torch.device | str = "cpu"
    ) -> TrainIterationMetrics:
        if not scored_rollouts:
            raise ValueError("`scored_rollouts` cannot be empty.")

        dist_buffer = buffer.DistributedRolloutBuffer(
            scored_rollouts, pad_token_id=self._config.pad_token_id
        )
        local_batch = dist_buffer.get_rank_batch(
            rank=self._rank, world_size=self._world_size, device=device
        )

        losses: list[float] = []
        for _ in range(self._config.train_steps_per_iteration):
            loss = self._trainer.train_step(local_batch)
            losses.append(loss)

        self._broadcaster.extract_and_broadcast(self._actor)

        return TrainIterationMetrics(
            avg_loss=sum(losses) / len(losses) if losses else 0.0,
            rank=self._rank,
        )
