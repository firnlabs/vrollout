"""Policy gradient optimization with reference model regularization."""

from dataclasses import dataclass

from beartype import beartype
from jaxtyping import Float
from jaxtyping import Int
from jaxtyping import jaxtyped
import torch
from torch import distributed as dist
from torch import nn
from torch import optim
from torch import Tensor
from torch.distributed import fsdp
from torch.distributed.tensor import DTensor
from torch.utils import checkpoint

from vrollout import buffer as vb
from vrollout import grpo
from vrollout.buffer import DistributedBatch


@dataclass(frozen=True, slots=True)
class TrainStepMetrics:
    """Loss and diagnostic metrics for one optimizer step, averaged over all tokens.

    Token-level metrics are means over all completion tokens in the batch, across all
    microbatches and, under data parallelism, across all ranks.

    Attributes:
        loss: GRPO loss.
        mean_kl: k3 estimate of KL(pi || pi_ref) between the current and reference
            policies.
        clip_fraction: Fraction of tokens whose importance ratio pi / pi_old fell
            outside `[1 - clip_eps, 1 + clip_eps]`.
        logprob_abs_diff: Mean of `|log pi - log pi_old|`, the log-prob gap between
            the current policy and the old policy that generated the batch.
        mismatch_kl: k3 estimate of KL(pi_old || pi), using pi_old as the sampling
            policy.
        grad_norm: Global L2 norm of the actor gradient before clipping.
    """

    loss: float
    mean_kl: float
    clip_fraction: float
    logprob_abs_diff: float
    mismatch_kl: float
    grad_norm: float


def _get_target_log_probs(
    logits_chunk: Float[Tensor, "batch chunk vocab"],
    targets_chunk: Int[Tensor, "batch chunk"],
    temperature: float = 1.0,
) -> Float[Tensor, "batch chunk"]:
    # upcast to float32 for numerical stability in logsumexp
    logits_chunk = logits_chunk.to(
        torch.promote_types(logits_chunk.dtype, torch.float32)
    )
    # score tokens under softmax(z / T), the distribution the sampler drew them from
    if temperature != 1.0:
        logits_chunk = logits_chunk / temperature
    target_logits = logits_chunk.gather(
        dim=-1, index=targets_chunk.unsqueeze(-1)
    ).squeeze(-1)
    return target_logits - torch.logsumexp(logits_chunk, dim=-1)


@jaxtyped(typechecker=beartype)
def _compute_logprobs(
    model: nn.Module,
    dist_batch: DistributedBatch,
    chunk_size: int = 128,
    temperature: float = 1.0,
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
                _get_target_log_probs, lc, tc, temperature, use_reentrant=False
            )
        else:
            chunk = _get_target_log_probs(lc, tc, temperature)
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
        temperature: float = 1.0,
    ) -> None:
        if temperature <= 0.0:
            raise ValueError(f"temperature must be positive, got {temperature}")
        self._actor = actor
        self._reference = reference
        self._optimizer = optimizer
        self._beta = beta
        self._clip_eps = clip_eps
        self._microbatch_size = microbatch_size
        self._logit_chunk_size = logit_chunk_size
        self._max_grad_norm = max_grad_norm
        self._autocast_dtype = autocast_dtype
        self._temperature = temperature

    def _logprobs(
        self, model: nn.Module, mb: DistributedBatch
    ) -> Float[Tensor, "batch seq_len_minus_1"]:
        return _compute_logprobs(
            model, mb, self._logit_chunk_size, temperature=self._temperature
        )

    def _autocast(self) -> torch.autocast:
        device_type = next(self._actor.parameters()).device.type
        return torch.autocast(
            device_type=device_type,
            dtype=self._autocast_dtype,
            enabled=self._autocast_dtype is not None,
        )

    @jaxtyped(typechecker=beartype)
    def train_step(self, dist_batch: DistributedBatch) -> TrainStepMetrics:
        """Executes optimization step computing GRPO loss with KL penalty."""
        completion_mask = dist_batch.loss_mask[:, 1:]
        local_tokens = completion_mask.sum().to(torch.float32)

        # with FSDP2, gradients are averaged across ranks (divided by world size W) in
        # reduce-scatter. To get the exact global token mean sum_all / N_all instead of
        # a mean of per-rank means, each rank normalizes its local sum by N_all / W.
        data_parallel = (
            isinstance(self._actor, fsdp.FSDPModule)
            and dist.is_initialized()
            and dist.get_world_size() > 1
        )
        world_size = dist.get_world_size() if data_parallel else 1
        global_tokens = local_tokens.clone()
        if data_parallel:
            dist.all_reduce(global_tokens, op=dist.ReduceOp.SUM)
        if global_tokens.item() == 0:
            raise ValueError("No valid tokens found in completion mask.")
        num_valid_tokens = global_tokens / world_size

        # create microbatches
        microbatches = (
            vb.split_distributed_batch(dist_batch, self._microbatch_size)
            if self._microbatch_size is not None
            else [dist_batch]
        )

        # reference model forward
        with torch.no_grad(), self._autocast():
            ref_logprobs_list = [
                self._logprobs(self._reference, mb) for mb in microbatches
            ]

        # actor model forward, loss, and gradient step. Every microbatch's forward runs
        # before the optimizer step, so the log-prob mismatch metrics compare the
        # sampler with the same weights that generated the batch (staleness 0).
        self._optimizer.zero_grad()
        accum_loss = 0.0
        accum_kl = 0.0
        accum_clip_frac = 0.0
        accum_lp_diff = 0.0
        accum_mismatch_kl = 0.0

        for mb, ref_lps in zip(microbatches, ref_logprobs_list, strict=True):
            with self._autocast():
                actor_lps = self._logprobs(self._actor, mb)

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
            accum_lp_diff += metrics.logprob_abs_diff
            accum_mismatch_kl += metrics.mismatch_kl

        # an infinite max norm leaves gradients unchanged but still returns the norm
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self._actor.parameters(),
            self._max_grad_norm if self._max_grad_norm is not None else float("inf"),
        )
        if isinstance(grad_norm, DTensor):
            grad_norm = grad_norm.full_tensor()

        self._optimizer.step()

        if data_parallel:
            # average per-rank metrics: mean_r(sum_r * W / N_all) == sum_all / N_all
            # where:
            # - sum_r: sum of per-token metrics on rank r
            # - W: world size
            # - N_all: total number of valid tokens across all ranks
            # - sum_all: sum of per-token metrics across all ranks
            device = local_tokens.device
            stats = torch.tensor(
                [
                    accum_loss,
                    accum_kl,
                    accum_clip_frac,
                    accum_lp_diff,
                    accum_mismatch_kl,
                ],
                dtype=torch.float32,
                device=device,
            )
            dist.all_reduce(stats, op=dist.ReduceOp.SUM)
            stats /= world_size
            (
                accum_loss,
                accum_kl,
                accum_clip_frac,
                accum_lp_diff,
                accum_mismatch_kl,
            ) = stats.tolist()

        return TrainStepMetrics(
            loss=accum_loss,
            mean_kl=accum_kl,
            clip_fraction=accum_clip_frac,
            logprob_abs_diff=accum_lp_diff,
            mismatch_kl=accum_mismatch_kl,
            grad_norm=grad_norm.item(),
        )
