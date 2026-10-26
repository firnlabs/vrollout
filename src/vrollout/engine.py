"""Distributed RL training engine coordinating rollout batching and weight broadcast."""

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
    """Configuration for distributed RL training iterations."""

    pad_token_id: int = 0
    beta: float = 0.04
    clip_eps: float = 0.2
    group_size: int = 1
    train_steps_per_iteration: int = 1
    micro_batch_size: int = 4
    logit_chunk_size: int = 128
    max_grad_norm: float | None = 1.0
    autocast_dtype: torch.dtype | None = None


@dataclass(frozen=True)
class TrainIterationMetrics:
    """Per-rank training metrics for a single optimization iteration."""

    avg_loss: float
    rank: int
    avg_kl: float = 0.0
    avg_clip_fraction: float = 0.0


class DistributedRLEngine:
    """Distributed training engine for batch sharding, training, and weights sync."""

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
            clip_eps=self._config.clip_eps,
            microbatch_size=self._config.micro_batch_size,
            logit_chunk_size=self._config.logit_chunk_size,
            max_grad_norm=self._config.max_grad_norm,
            autocast_dtype=self._config.autocast_dtype,
        )

        self._broadcaster = sync.WeightBroadcaster(
            rank=self._rank, receivers=self._receivers
        )

    def train_iteration(
        self, scored_rollouts: list[ScoredRollout], device: torch.device | str = "cpu"
    ) -> TrainIterationMetrics:
        """Runs training iteration over scored rollouts and broadcasts actor weights."""
        if not scored_rollouts:
            raise ValueError("`scored_rollouts` cannot be empty.")

        dist_buffer = buffer.DistributedRolloutBuffer(
            scored_rollouts,
            pad_token_id=self._config.pad_token_id,
            group_size=self._config.group_size,
        )
        local_batch = dist_buffer.get_rank_batch(
            rank=self._rank, world_size=self._world_size, device=device
        )

        metrics_list = []
        for _ in range(self._config.train_steps_per_iteration):
            metrics = self._trainer.train_step(local_batch)
            metrics_list.append(metrics)

        self._broadcaster.extract_and_broadcast(self._actor)

        avg_loss = (
            sum(m.loss for m in metrics_list) / len(metrics_list)
            if metrics_list
            else 0.0
        )
        avg_kl = (
            sum(m.mean_kl for m in metrics_list) / len(metrics_list)
            if metrics_list
            else 0.0
        )
        avg_clip_fraction = (
            sum(m.clip_fraction for m in metrics_list) / len(metrics_list)
            if metrics_list
            else 0.0
        )

        return TrainIterationMetrics(
            avg_loss=avg_loss,
            rank=self._rank,
            avg_kl=avg_kl,
            avg_clip_fraction=avg_clip_fraction,
        )
