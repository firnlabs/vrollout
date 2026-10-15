"""Distributed experience buffer and batch collation across process ranks."""

from dataclasses import dataclass

from beartype import beartype
from jaxtyping import Bool
from jaxtyping import Float
from jaxtyping import Int64
from jaxtyping import jaxtyped
import torch
from torch import Tensor

from vrollout import grpo
from vrollout.scorer import ScoredRollout
from vrollout.types import Rank
from vrollout.types import WorldSize


@jaxtyped(typechecker=beartype)
@dataclass(frozen=True, slots=True, kw_only=True)
class DistributedBatch:
    """Padded batch of scored rollouts for distributed training."""

    input_ids: Int64[Tensor, "batch seq_len"]
    loss_mask: Bool[Tensor, "batch seq_len"]
    attention_mask: Bool[Tensor, "batch seq_len"]
    rewards: Float[Tensor, "batch"]
    advantages: Float[Tensor, "batch"]
    old_log_probs: Float[Tensor, "batch seq_len_minus_1"]


def _get_max_seq_len(scored_rollouts: list[ScoredRollout]) -> int:
    max_seq_len = 0
    for sr in scored_rollouts:
        seq_len = len(sr.trajectory.prompt_token_ids) + len(
            sr.trajectory.completion_token_ids
        )
        max_seq_len = max(max_seq_len, seq_len)
    return max_seq_len


def _compute_global_advantages(
    scored_rollouts: list[ScoredRollout], group_size: int
) -> torch.Tensor:
    rewards = torch.tensor([sr.reward for sr in scored_rollouts], dtype=torch.float32)
    if group_size > 1 and len(scored_rollouts) % group_size == 0:
        global_advantages = grpo.compute_group_advantages(
            rewards, group_size=group_size
        )
    else:
        global_advantages = rewards
    return global_advantages


class DistributedRolloutBuffer:
    """Partitions and collates scored rollouts across distributed training ranks."""

    def __init__(
        self,
        scored_rollouts: list[ScoredRollout],
        pad_token_id: int = 0,
        group_size: int = 1,
    ) -> None:
        self._scored_rollouts = scored_rollouts
        self._num_rollouts = len(scored_rollouts)
        self._max_seq_len = _get_max_seq_len(scored_rollouts)
        self._pad_token_id = pad_token_id
        self._group_size = group_size
        self._global_advantages = (
            _compute_global_advantages(scored_rollouts, group_size)
            if scored_rollouts
            else torch.empty(0, dtype=torch.float32)
        )

    @jaxtyped(typechecker=beartype)
    def _create_distributed_batch(
        self,
        batch: list[ScoredRollout],
        advantages: Float[Tensor, "batch"] | None = None,
        device: str | torch.device = "cpu",
    ) -> DistributedBatch:
        batch_size = len(batch)

        max_len = max(self._max_seq_len, 2)
        input_ids = torch.full(
            (batch_size, max_len), self._pad_token_id, dtype=torch.int64, device=device
        )
        attention_mask = torch.zeros(
            (batch_size, max_len), dtype=torch.bool, device=device
        )
        loss_mask = torch.zeros((batch_size, max_len), dtype=torch.bool, device=device)
        rewards = torch.zeros((batch_size,), dtype=torch.float32, device=device)
        old_log_probs = torch.zeros(
            (batch_size, max_len - 1), dtype=torch.float32, device=device
        )

        for i, sr in enumerate(batch):
            traj = sr.trajectory
            prompt_len = len(traj.prompt_token_ids)
            completion_len = len(traj.completion_token_ids)
            total_len = prompt_len + completion_len
            input_ids[i, :total_len] = torch.tensor(
                [*traj.prompt_token_ids, *traj.completion_token_ids],
                dtype=torch.int64,
                device=device,
            )
            attention_mask[i, :total_len] = True
            loss_mask[i, prompt_len:total_len] = True
            rewards[i] = sr.reward

            if completion_len > 0 and traj.logprobs:
                start_idx = max(prompt_len - 1, 0)
                end_idx = start_idx + len(traj.logprobs)
                old_log_probs[i, start_idx:end_idx] = torch.tensor(
                    traj.logprobs, dtype=torch.float32, device=device
                )

        if advantages is None:
            batch_adv = rewards
        else:
            batch_adv = advantages.to(device)

        return DistributedBatch(
            input_ids=input_ids,
            loss_mask=loss_mask,
            attention_mask=attention_mask,
            rewards=rewards,
            advantages=batch_adv,
            old_log_probs=old_log_probs,
        )

    @jaxtyped(typechecker=beartype)
    def get_rank_batch(
        self, rank: Rank, world_size: WorldSize, device: str | torch.device = "cpu"
    ) -> DistributedBatch:
        """Collates and returns padded `DistributedBatch` assigned to `rank`."""
        if world_size <= 0:
            raise ValueError(f"world_size must be positive, got {world_size}")
        if not 0 <= rank < world_size:
            raise IndexError(f"rank {rank} out of bounds for world_size {world_size}")
        if not self._scored_rollouts:
            raise ValueError("Cannot get rank batch from an empty buffer")
        if self._num_rollouts % world_size != 0:
            raise ValueError(
                "Rollout count can't be divided evenly across ranks: "
                f"{self._num_rollouts=} % {world_size=} != 0"
            )

        batch_size = self._num_rollouts // world_size
        start_idx = rank * batch_size
        end_idx = (rank + 1) * batch_size
        batch = self._scored_rollouts[start_idx:end_idx]
        rank_adv = self._global_advantages[start_idx:end_idx]
        return self._create_distributed_batch(batch, rank_adv, device=device)
