"""Rollout processing buffer for GAE computation."""

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor

from vrollout import scorer


def _get_sequence_lengths(scored_rollouts: Sequence[scorer.ScoredRollout]) -> list[int]:
    return [len(sr.trajectory.completion_token_ids) for sr in scored_rollouts]


@dataclass(frozen=True, slots=True, kw_only=True)
class RolloutBatch:
    """Collated and padded batch of trajectory data for GAE calculation."""

    completion_token_ids: Tensor  # (B, T_max), int64
    rewards: Tensor  # (B, T_max), float32
    values: Tensor  # (B, T_max), float32
    next_values: Tensor  # (B, ), float32
    dones: Tensor  # (B, T_max), float32
    attention_mask: Tensor  # (B, T_max), float32
    seq_lens: Tensor  # (B,), int64


def collate_rollouts_for_gae(
    scored_rollouts: Sequence[scorer.ScoredRollout],
    values: Sequence[Sequence[float]] | None = None,
    next_values: Sequence[float] | None = None,
    pad_token_id: int = 0,
    device: str | torch.device = "cpu",
) -> RolloutBatch:
    if not scored_rollouts:
        raise ValueError("`scored_rollouts` is empty")

    batch_size = len(scored_rollouts)

    if values is not None and len(values) < batch_size:
        raise ValueError(
            "Number of `values` must be at least the number of `scored_rollouts`"
        )
    if next_values is not None and len(next_values) < batch_size:
        raise ValueError(
            "Number of `next_values` must be at least the number of `scored_rollouts`"
        )

    seq_lens = _get_sequence_lengths(scored_rollouts)
    max_seq_len = max(max(seq_lens), 1)  # max_seq_len >= 1

    completion_token_ids = torch.full(
        (batch_size, max_seq_len), pad_token_id, dtype=torch.int64, device=device
    )
    rewards = torch.zeros_like(completion_token_ids, dtype=torch.float32)
    values_t = torch.zeros_like(rewards)
    next_values_t = torch.zeros((batch_size,), dtype=torch.float32, device=device)
    dones = torch.ones_like(rewards)
    attention_mask = torch.zeros_like(rewards)
    seq_lens_t = torch.tensor(seq_lens, dtype=torch.int64, device=device)

    for i in range(batch_size):
        seq_len = seq_lens[i]
        if next_values is not None:
            next_values_t[i] = float(next_values[i])

        if seq_len <= 0:
            continue

        tokens = scored_rollouts[i].trajectory.completion_token_ids
        completion_token_ids[i, :seq_len] = torch.tensor(
            tokens, dtype=torch.int64, device=device
        )
        attention_mask[i, :seq_len] = 1.0

        if seq_len > 1:
            dones[i, : seq_len - 1] = 0.0
        dones[i, seq_len - 1] = 1.0

        rewards[i, seq_len - 1] = float(scored_rollouts[i].reward)

        if values is not None:
            value_slice = values[i][:seq_len]
            values_t[i, : len(value_slice)] = torch.tensor(
                value_slice, dtype=torch.float32, device=device
            )

    return RolloutBatch(
        completion_token_ids=completion_token_ids,
        rewards=rewards,
        values=values_t,
        next_values=next_values_t,
        dones=dones,
        attention_mask=attention_mask,
        seq_lens=seq_lens_t,
    )
