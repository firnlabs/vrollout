"""Policy gradient optimization with reference model regularization."""

from beartype import beartype
from jaxtyping import Float
from jaxtyping import Int
from jaxtyping import jaxtyped
import torch
from torch import nn
from torch import optim
from torch import Tensor
from torch.utils import checkpoint

from vrollout import buffer as vb
from vrollout import grpo
from vrollout.buffer import DistributedBatch


def _get_target_log_probs(
    logits_chunk: Float[Tensor, "batch chunk vocab"],
    targets_chunk: Int[Tensor, "batch chunk"],
) -> Float[Tensor, "batch chunk"]:
    # upcast to float32 for numerical stability in logsumexp
    logits_chunk = logits_chunk.to(
        torch.promote_types(logits_chunk.dtype, torch.float32)
    )
    target_logits = logits_chunk.gather(
        dim=-1, index=targets_chunk.unsqueeze(-1)
    ).squeeze(-1)
    return target_logits - torch.logsumexp(logits_chunk, dim=-1)


@jaxtyped(typechecker=beartype)
def _compute_logprobs(
    model: nn.Module,
    dist_batch: DistributedBatch,
    chunk_size: int = 128,
) -> Float[Tensor, "batch seq_len_minus_1"]:
    outputs = model(dist_batch.input_ids, attention_mask=dist_batch.attention_mask)
    logits = outputs.logits if hasattr(outputs, "logits") else outputs

    shifted_logits = logits[:, :-1, :]  # (B, T-1, V)
    shifted_targets = dist_batch.input_ids[..., 1:]  # (B, T-1)

    use_ckpt = torch.is_grad_enabled() and logits.requires_grad

    chunks = []
    for lc, tc in zip(
        shifted_logits.split(chunk_size, dim=1),
        shifted_targets.split(chunk_size, dim=1),
        strict=True,
    ):
        if use_ckpt:
            chunk = checkpoint.checkpoint(
                _get_target_log_probs, lc, tc, use_reentrant=False
            )
        else:
            chunk = _get_target_log_probs(lc, tc)
        chunks.append(chunk)

    return torch.cat(chunks, dim=1)


class ModelTrainer:
    """Dual actor-reference policy trainer with GRPO loss."""

    def __init__(
        self,
        *,
        actor: nn.Module,
        reference: nn.Module,
        optimizer: optim.Optimizer,
        beta: float = 0.04,
        clip_eps: float = 0.2,
        microbatch_size: int | None = None,
        logit_chunk_size: int = 128,
        autocast_dtype: torch.dtype | None = None,
        max_grad_norm: float | None = None,
    ) -> None:
        self._actor = actor
        self._reference = reference
        self._optimizer = optimizer
        self._beta = beta
        self._clip_eps = clip_eps
        self._microbatch_size = microbatch_size
        self._logit_chunk_size = logit_chunk_size
        self._max_grad_norm = max_grad_norm
        self._autocast_dtype = autocast_dtype

    def _autocast(self) -> torch.autocast:
        device_type = next(self._actor.parameters()).device.type
        return torch.autocast(
            device_type=device_type,
            dtype=self._autocast_dtype,
            enabled=self._autocast_dtype is not None,
        )

    @jaxtyped(typechecker=beartype)
    def train_step(self, dist_batch: DistributedBatch) -> grpo.GRPOLossMetrics:
        """Executes optimization step computing GRPO loss with KL penalty."""
        completion_mask = dist_batch.loss_mask[:, 1:]
        num_valid_tokens = completion_mask.sum()
        if num_valid_tokens == 0:
            raise ValueError("No valid tokens found in completion mask.")

        # create microbatches
        microbatches = (
            vb.split_distributed_batch(dist_batch, self._microbatch_size)
            if self._microbatch_size is not None
            else [dist_batch]
        )

        # reference model forward
        with torch.no_grad(), self._autocast():
            ref_logprobs_list = [
                _compute_logprobs(self._reference, mb, self._logit_chunk_size)
                for mb in microbatches
            ]

        # actor model forward, loss, and gradient step
        self._optimizer.zero_grad()
        accum_loss = 0.0
        accum_kl = 0.0
        accum_clip_frac = 0.0

        for mb, ref_lps in zip(microbatches, ref_logprobs_list, strict=True):
            with self._autocast():
                actor_lps = _compute_logprobs(self._actor, mb, self._logit_chunk_size)

            loss, metrics = grpo.compute_grpo_loss(
                log_probs=actor_lps,
                old_log_probs=mb.old_log_probs,
                ref_log_probs=ref_lps,
                advantages=mb.advantages,
                completion_mask=mb.loss_mask[:, 1:],
                clip_eps=self._clip_eps,
                beta=self._beta,
                num_valid_tokens=num_valid_tokens,
            )

            loss.backward()

            accum_loss += loss.item()
            accum_kl += metrics.mean_kl
            accum_clip_frac += metrics.clip_fraction

        if self._max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                self._actor.parameters(), self._max_grad_norm
            )

        self._optimizer.step()

        return grpo.GRPOLossMetrics(
            loss=accum_loss,
            mean_kl=accum_kl,
            clip_fraction=accum_clip_frac,
        )
