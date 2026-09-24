"""Prepares scored rollouts for FSDP processing."""

from dataclasses import dataclass

import torch
from torch import Tensor

from vrollout.scorer import ScoredRollout

type Rank = int
type WorldSize = int


# TODO: add jaxtyping tensor type, shape
@dataclass(frozen=True, slots=True, kw_only=True)
class DistributedBatch:
    input_ids: Tensor
    loss_mask: Tensor
    attention_mask: Tensor
    rewards: Tensor


def _get_max_seq_len(scored_rollouts: list[ScoredRollout]) -> int:
    max_seq_len = 0
    for sr in scored_rollouts:
        seq_len = len(sr.trajectory.prompt_token_ids) + len(
            sr.trajectory.completion_token_ids
        )
        max_seq_len = max(max_seq_len, seq_len)
    return max_seq_len


class DistributedRolloutBuffer:
    """Batches scored rollouts based on FSDP process rank."""

    def __init__(
        self, scored_rollouts: list[ScoredRollout], pad_token_id: int = 0
    ) -> None:
        self._scored_rollouts = scored_rollouts
        self._num_rollouts = len(scored_rollouts)
        self._max_seq_len = _get_max_seq_len(scored_rollouts)
        self._pad_token_id = pad_token_id

    def _create_distributed_batch(
        self, batch: list[ScoredRollout], device: str | torch.device = "cpu"
    ) -> DistributedBatch:
        batch_size = len(batch)

        max_len = self._max_seq_len
        input_ids = torch.full(
            (batch_size, max_len), self._pad_token_id, dtype=torch.int64, device=device
        )
        attention_mask = torch.zeros(
            (batch_size, max_len), dtype=torch.bool, device=device
        )
        loss_mask = torch.zeros((batch_size, max_len), dtype=torch.bool, device=device)
        rewards = torch.zeros((batch_size,), dtype=torch.float32, device=device)

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

        return DistributedBatch(
            input_ids=input_ids,
            loss_mask=loss_mask,
            attention_mask=attention_mask,
            rewards=rewards,
        )

    def get_rank_batch(
        self, rank: Rank, world_size: WorldSize, device: str | torch.device = "cpu"
    ) -> DistributedBatch:
        """Returns collated DistributedBatch from rollouts based on rank."""
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
        batch = self._scored_rollouts[rank * batch_size : (rank + 1) * batch_size]
        return self._create_distributed_batch(batch, device=device)
