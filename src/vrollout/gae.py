"""Generalized advantage estimation (GAE)."""

import torch
from torch import Tensor

try:
    import triton
    import triton.language as tl
except ImportError:
    TRITON_AVAILABLE = False
else:
    TRITON_AVAILABLE = True


def compute_gae(
    rewards: Tensor,  # (B, T)
    values: Tensor,  # (B, T)
    next_values: Tensor,  # (B, )
    dones: Tensor,  # (B, T)
    seq_lens: Tensor,  # (B, )
    gamma: float = 0.99,
    lam: float = 0.95,
) -> tuple[Tensor, Tensor]:  # (B, T),  (B, T)
    """Computes generalized advantage estimation (GAE)."""
    max_seq_len = rewards.shape[-1]
    if max_seq_len == 0:
        raise ValueError("Maximum sequence length is zero.")

    if (seq_lens > max_seq_len).any():
        raise ValueError("Maximum value in seq_lens exceeds rewards sequence length.")

    last_gae = torch.zeros_like(rewards[..., 0])
    advantages = torch.zeros_like(rewards)
    dones = dones.to(dtype=rewards.dtype)
    seq_lens = seq_lens.to(device=rewards.device, dtype=torch.int64)

    # Append next_values to values and move former's values to proper positions
    # given by seq_lens
    values_ext = torch.cat([values, next_values.unsqueeze(1)], dim=1)
    values_ext.scatter_(
        dim=1, index=seq_lens.unsqueeze(1), src=next_values.unsqueeze(1)
    )

    for t in reversed(range(max_seq_len)):
        is_valid = t < seq_lens

        delta = (
            rewards[..., t]
            + gamma * (1 - dones[..., t]) * values_ext[..., t + 1]
            - values[..., t]
        )

        last_gae = (delta + gamma * lam * (1 - dones[..., t]) * last_gae) * is_valid
        advantages[..., t] = last_gae

    mask = torch.arange(max_seq_len, device=rewards.device) < seq_lens.unsqueeze(1)
    target_values = (advantages + values) * mask
    return advantages, target_values
