"""Generalized advantage estimation (GAE)."""

from beartype import beartype
from jaxtyping import Float
from jaxtyping import Int64
from jaxtyping import jaxtyped
import torch
from torch import Tensor

try:
    import triton  # noqa: F401
    import triton.language as tl  # noqa: F401
except ImportError:
    TRITON_AVAILABLE = False
else:
    TRITON_AVAILABLE = True


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
    """Computes generalized advantage estimation (GAE) using PyTorch primitives."""
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


if TRITON_AVAILABLE:

    @triton.jit
    def _fused_gae_kernel(
        rewards_ptr,
        values_ptr,
        next_values_ptr,
        dones_ptr,
        seq_lens_ptr,
        adv_ptr,
        target_values_ptr,
        stride_b,
        gamma: tl.constexpr,
        lam: tl.constexpr,
    ):  # pragma: no cover
        """Fused Triton GAE Kernel.

        Program instance per batch row, backward pass in registers.
        """
        # retrieve 1D program ID and compute row offset using stride_b
        pid = tl.program_id(axis=0)
        offset = pid * stride_b

        # load initial bootstrap value from next_values
        next_value = tl.load(next_values_ptr + pid)

        # load sequence length
        seq_len = tl.load(seq_lens_ptr + pid)

        # initialize GAE accumulator and decay factor (gamma * lam)
        last_gae = 0.0
        gamma_lam = gamma * lam

        # iterate backwards through time steps (seq_len - 1 down to 0):
        for t in range(seq_len - 1, -1, -1):
            #  compute memory index for current step
            offset_t = offset + t

            #  load reward, value, and done flag for current time step
            reward = tl.load(rewards_ptr + offset_t)
            value = tl.load(values_ptr + offset_t)
            done = tl.load(dones_ptr + offset_t)

            #  compute non-terminal mask and 1-step TD-error delta
            non_terminal_mask = 1 - done
            delta = reward + gamma * next_value * non_terminal_mask - value

            #  compute GAE
            last_gae = delta + gamma_lam * non_terminal_mask * last_gae

            #  compute target value for critic training
            target_value = last_gae + value

            #  store advantage and return to global memory output pointers
            tl.store(adv_ptr + offset_t, last_gae)
            tl.store(target_values_ptr + offset_t, target_value)

            #  update bootstrap value with current step's value for previous time step
            next_value = value

    @jaxtyped(typechecker=beartype)
    def compute_gae_triton(
        rewards: Float[Tensor, "batch completion_len"],
        values: Float[Tensor, "batch completion_len"],
        next_values: Float[Tensor, "batch"],
        dones: Float[Tensor, "batch completion_len"],
        seq_lens: Int64[Tensor, "batch"],
        gamma: float = 0.99,
        lam: float = 0.95,
    ) -> tuple[  # pragma: no cover
        Float[Tensor, "batch completion_len"],
        Float[Tensor, "batch completion_len"],
    ]:
        """Computes generalized advantage estimation (GAE) using Triton fused kernel."""
        B, T = rewards.shape
        if values.shape != (B, T):
            raise ValueError("Shape mismatch: values must have shape (B, T).")
        if next_values.shape != (B,):
            raise ValueError("Shape mismatch: next_values must have shape (B,).")
        if dones.shape != (B, T):
            raise ValueError("Shape mismatch: dones must have shape (B, T).")

        assert rewards.dtype == torch.float32
        rewards = rewards.contiguous().cuda()
        values = values.contiguous().cuda()
        next_values = next_values.contiguous().cuda()
        dones = dones.contiguous().cuda()
        seq_lens = seq_lens.contiguous().cuda()

        # pre-allocate empty output tensors for advantages and returns
        advantages = torch.zeros_like(rewards)
        target_values = torch.zeros_like(rewards)

        # calculate batch row stride and launch 1D Triton grid
        stride_b = rewards.stride(0)

        # compute GAE using Triton kernel
        grid = (B,)
        _fused_gae_kernel[grid](
            rewards,
            values,
            next_values,
            dones,
            seq_lens,
            advantages,
            target_values,
            stride_b,
            gamma,
            lam,
        )

        return advantages, target_values
else:

    @jaxtyped(typechecker=beartype)
    def compute_gae_triton(
        rewards: Float[Tensor, "batch completion_len"],
        values: Float[Tensor, "batch completion_len"],
        next_values: Float[Tensor, "batch"],
        dones: Float[Tensor, "batch completion_len"],
        seq_lens: Int64[Tensor, "batch"],
        gamma: float = 0.99,
        lam: float = 0.95,
    ) -> tuple[  # pragma: no cover
        Float[Tensor, "batch completion_len"],
        Float[Tensor, "batch completion_len"],
    ]:
        """Computes generalized advantage estimation (GAE) using Triton fused kernel."""
        raise RuntimeError("Triton is not available in the current environment.")


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
    """Computes generalized advantage estimation (GAE)."""
    if TRITON_AVAILABLE and torch.cuda.is_available():
        return compute_gae_triton(
            rewards, values, next_values, dones, seq_lens, gamma, lam
        )

    return compute_gae_torch(rewards, values, next_values, dones, seq_lens, gamma, lam)
