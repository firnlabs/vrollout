"""Generalized advantage estimation (GAE)."""

from beartype import beartype
from jaxtyping import Float
from jaxtyping import Int64
from jaxtyping import jaxtyped
import torch
from torch import Tensor


@jaxtyped(typechecker=beartype)
def compute_gae_torch(
    rewards: Float[Tensor, "batch completion_len"],
    values: Float[Tensor, "batch completion_len"],
    next_values: Float[Tensor, "batch"],
    dones: Float[Tensor, "batch completion_len"],
    seq_lens: Int64[Tensor, "batch"],
    gamma: float = 0.99,
    lam: float = 0.95,
) -> tuple[
    Float[Tensor, "batch completion_len"],
    Float[Tensor, "batch completion_len"],
]:
    """Computes Generalized Advantage Estimation (GAE) using PyTorch primitives.

    Follows Schulman et al. (2015), Eq. (11)-(12) with sequence masking.

    Returns a tuple of `(advantages, target_values)` where:
        - advantages: Advantage estimates masked beyond completion lengths.
        - target_values: TD(lambda) return targets (`advantages + values`) for critic
          training.
    """
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


@jaxtyped(typechecker=beartype)
def compute_gae(
    rewards: Float[Tensor, "batch completion_len"],
    values: Float[Tensor, "batch completion_len"],
    next_values: Float[Tensor, "batch"],
    dones: Float[Tensor, "batch completion_len"],
    seq_lens: Int64[Tensor, "batch"],
    gamma: float = 0.99,
    lam: float = 0.95,
) -> tuple[
    Float[Tensor, "batch completion_len"],
    Float[Tensor, "batch completion_len"],
]:
    """Computes Generalized Advantage Estimation (GAE) across variable-length rollouts.

    Uses PyTorch primitives for advantage and target return computation.

    Returns a tuple of `(advantages, target_values)` where:
        - advantages: Advantage estimates masked beyond completion lengths.
        - target_values: TD(lambda) return targets (`advantages + values`)
          for critic training.
    """
    return compute_gae_torch(
        rewards=rewards,
        values=values,
        next_values=next_values,
        dones=dones,
        seq_lens=seq_lens,
        gamma=gamma,
        lam=lam,
    )
