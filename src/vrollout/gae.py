"""Generalized advantage estimation (GAE)."""

import torch
from torch import Tensor


def compute_gae(
    rewards: Tensor,  # (B, T)
    values: Tensor,  # (B, T)
    next_values: Tensor,  # (B, )
    dones: Tensor,  # (B, T)
    gamma: float = 0.99,
    lam: float = 0.95,
) -> tuple[Tensor, Tensor]:  # (B, T),  (B, T)
    """Computes generalized advantage estimation (GAE)."""
    seq_len = rewards.shape[-1]
    last_gae = 0

    advantages = torch.zeros_like(rewards)
    dones = dones.to(torch.float32)

    for t in reversed(range(seq_len)):
        next_value = next_values if t == seq_len - 1 else values[..., t + 1]
        delta = (
            rewards[..., t] + gamma * (1 - dones[..., t]) * next_value - values[..., t]
        )
        last_gae = delta + gamma * lam * (1 - dones[..., t]) * last_gae
        advantages[..., t] = last_gae

    target_values = advantages + values
    return advantages, target_values
