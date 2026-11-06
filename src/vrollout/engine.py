"""Distributed RL training engine coordinating rollout batching and weight broadcast."""

from dataclasses import dataclass
import time

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
    """Configuration for distributed RL training iterations.

    Attributes:
        pad_token_id: Token ID used to right-pad rollouts to a common length.
        beta: Coefficient of the KL penalty against the reference policy.
        clip_eps: Clipping range epsilon for the PPO-style importance ratio.
        group_size: Number of rollouts sampled per prompt; rewards are normalized
            within each group to compute advantages when greater than 1.
        train_steps_per_iteration: Number of optimizer steps taken on the same
            rank-local batch per iteration.
        micro_batch_size: Number of sequences per gradient accumulation
            micro-batch.
        logit_chunk_size: Sequence chunk length used when computing log-probs to
            bound peak logit memory.
        max_grad_norm: Maximum global L2 gradient norm for clipping, or None to
            disable clipping.
        autocast_dtype: Dtype for autocast in the forward pass, or None to run in
            the model's native precision.
        temperature: Sampling temperature of the training rollouts; the trainer
            divides logits by this value so its log-probs match the sampler's.
    """

    pad_token_id: int = 0
    beta: float = 0.04
    clip_eps: float = 0.2
    group_size: int = 1
    train_steps_per_iteration: int = 1
    micro_batch_size: int = 4
    logit_chunk_size: int = 128
    max_grad_norm: float | None = 1.0
    autocast_dtype: torch.dtype | None = None
    temperature: float = 1.0


@dataclass(frozen=True)
class TrainIterationMetrics:
    """Training metrics for a single optimization iteration, as reported by `rank`.

    Each 'avg_*' value is averaged over the iteration's train steps.

    Attributes:
        avg_loss: Mean GRPO loss per completion token.
        rank: Rank of the process that produced these metrics.
        avg_kl: Mean per-token KL divergence between the actor and the reference
            policy.
        avg_clip_fraction: Fraction of completion tokens whose importance ratio
            fell outside ``[1 - clip_eps, 1 + clip_eps]``.
        avg_logprob_abs_diff: Mean per-token absolute difference between the
            current policy's and the old policy's log-probs.
        avg_mismatch_kl: Mean per-token k3 estimate of KL(pi_old || pi) between
            the old and current policies.
        avg_grad_norm: Global L2 gradient norm before clipping.
        train_time_s: Wall-clock seconds spent across all train steps in the
            iteration.
        weight_sync_time_s: Wall-clock seconds spent extracting and broadcasting
            actor weights to the rollout receivers.
        lr: Learning rate used by the iteration's last train step.
    """

    avg_loss: float
    rank: int
    avg_kl: float = 0.0
    avg_clip_fraction: float = 0.0
    avg_logprob_abs_diff: float = 0.0
    avg_mismatch_kl: float = 0.0
    avg_grad_norm: float = 0.0
    train_time_s: float = 0.0
    weight_sync_time_s: float = 0.0
    lr: float = 0.0


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


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
        lr_scheduler: optim.lr_scheduler.LRScheduler | None = None,
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
            temperature=self._config.temperature,
            lr_scheduler=lr_scheduler,
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

        t_train = time.perf_counter()
        metrics_list = []
        for _ in range(self._config.train_steps_per_iteration):
            metrics = self._trainer.train_step(local_batch)
            metrics_list.append(metrics)
        train_time_s = time.perf_counter() - t_train

        t_sync = time.perf_counter()
        self._broadcaster.extract_and_broadcast(self._actor)
        weight_sync_time_s = time.perf_counter() - t_sync

        return TrainIterationMetrics(
            avg_loss=_mean([m.loss for m in metrics_list]),
            rank=self._rank,
            avg_kl=_mean([m.mean_kl for m in metrics_list]),
            avg_clip_fraction=_mean([m.clip_fraction for m in metrics_list]),
            avg_logprob_abs_diff=_mean([m.logprob_abs_diff for m in metrics_list]),
            avg_mismatch_kl=_mean([m.mismatch_kl for m in metrics_list]),
            avg_grad_norm=_mean([m.grad_norm for m in metrics_list]),
            train_time_s=train_time_s,
            weight_sync_time_s=weight_sync_time_s,
            lr=metrics_list[-1].lr,
        )
