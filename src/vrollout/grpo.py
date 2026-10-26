"""Implements the GRPO (Generalized Reward Policy Optimization) algorithm.

Reference:
    - Shao, et al. (2024). DeepSeekMath: Pushing the Limits of Mathematical Reasoning
      in Open Language Models. https://arxiv.org/pdf/2402.03300
"""

from dataclasses import dataclass

from jaxtyping import Bool
from jaxtyping import Float
from jaxtyping import Int
import torch
from torch import Tensor


@dataclass(frozen=True, slots=True)
class GRPOLossMetrics:
    loss: float
    mean_kl: float
    clip_fraction: float


def compute_group_advantages(
    rewards: Float[Tensor, "batch_size"], group_size: int, eps: float = 1e-6
) -> Float[Tensor, "batch_size"]:
    """Computes group advantages for the given rewards."""
    rewards_ = rewards.view(-1, group_size)
    std = rewards_.std(dim=-1, keepdim=True, correction=0)
    advantages = torch.where(
        std < eps,
        torch.zeros_like(rewards_),
        (rewards_ - rewards_.mean(dim=-1, keepdim=True)) / (std + eps),
    )
    return advantages.view(-1)


def compute_kl_divergence(
    log_probs: Float[Tensor, "batch_size seq_len"],
    ref_log_probs: Float[Tensor, "batch_size seq_len"],
) -> Float[Tensor, "batch_size seq_len"]:
    """Computes approximate KL divergence between current and reference log probs.

    Uses the 'k3' method described in Schulman (2020).
    http://joschu.net/blog/kl-approx.html
    """
    log_ratios = ref_log_probs - log_probs
    return torch.expm1(log_ratios) - (log_ratios)


def compute_grpo_loss(
    log_probs: Float[Tensor, "batch_size seq_len"],
    old_log_probs: Float[Tensor, "batch_size seq_len"],
    ref_log_probs: Float[Tensor, "batch_size seq_len"],
    advantages: Float[Tensor, "batch_size"],
    completion_mask: Bool[Tensor, "batch_size seq_len"],
    clip_eps: float = 0.2,
    beta: float = 0.04,
    num_valid_tokens: Int[Tensor, ""] | int | None = None,
) -> tuple[Tensor, GRPOLossMetrics]:
    """Computes the GRPO loss and associated metrics.

    Args:
        log_probs: Current log probabilities of the policy.
        old_log_probs: Log probabilities of the policy from the previous iteration.
        ref_log_probs: Log probabilities of the reference policy.
        advantages: Advantage estimates for each sequence.
        completion_mask: Boolean mask indicating True forvalid tokens; False otherwise.
        clip_eps: Clipping epsilon for the policy surrogate.
        beta: Coefficient for the KL divergence term.
        num_valid_tokens: Number of valid tokens in the completion mask.
            If None, it will be computed from the mask.

    Returns:
        A tuple containing the computed loss and metrics.

    Raises:
        ValueError: if there are no valid tokens in the completion mask.
    """
    total_valid_tokens = (
        completion_mask.sum() if num_valid_tokens is None else num_valid_tokens
    )
    if total_valid_tokens == 0:
        raise ValueError("No valid tokens found.")

    ratios = torch.exp(log_probs - old_log_probs)
    advantages = advantages.unsqueeze(-1)

    surr1 = ratios * advantages
    surr2 = torch.clamp(ratios, 1 - clip_eps, 1 + clip_eps) * advantages
    policy_surrogate = torch.min(surr1, surr2)
    kl = compute_kl_divergence(log_probs, ref_log_probs)

    per_token_losses = -(policy_surrogate - beta * kl)
    masked_losses = per_token_losses * completion_mask
    loss = masked_losses.sum() / total_valid_tokens

    # compute metrics
    with torch.no_grad():
        clipped_mask = (ratios < 1 - clip_eps) | (ratios > 1 + clip_eps)
        clip_fraction = (clipped_mask & completion_mask).sum() / total_valid_tokens
        mean_kl = (kl * completion_mask).sum() / total_valid_tokens

    metrics = GRPOLossMetrics(
        loss=loss.item(),
        mean_kl=mean_kl.item(),
        clip_fraction=clip_fraction.item(),
    )

    return loss, metrics
