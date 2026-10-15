"""Policy gradient optimization with reference model regularization."""

from beartype import beartype
from jaxtyping import Float
from jaxtyping import jaxtyped
import torch
from torch import nn
from torch import optim
from torch import Tensor
from torch.nn import functional as F

from vrollout import grpo
from vrollout.buffer import DistributedBatch


@jaxtyped(typechecker=beartype)
def _compute_logprobs(
    model: nn.Module, dist_batch: DistributedBatch, detach: bool = False
) -> Float[Tensor, "batch seq_len_minus_1"]:
    outputs = model(dist_batch.input_ids, attention_mask=dist_batch.attention_mask)
    logits = outputs.logits if hasattr(outputs, "logits") else outputs
    log_probs = F.log_softmax(logits, dim=-1)
    targets = dist_batch.input_ids[..., 1:].unsqueeze(-1)
    token_log_probs = torch.gather(
        log_probs[..., :-1, :],
        dim=-1,
        index=targets,
    ).squeeze(-1)
    return token_log_probs.detach() if detach else token_log_probs


class ModelTrainer:
    """Dual actor-reference policy trainer with GRPO loss."""

    def __init__(
        self,
        actor: nn.Module,
        reference: nn.Module,
        optimizer: optim.Optimizer,
        beta: float = 0.04,
        clip_eps: float = 0.2,
    ) -> None:
        self._actor = actor
        self._reference = reference
        self._optimizer = optimizer
        self._beta = beta
        self._clip_eps = clip_eps

    @jaxtyped(typechecker=beartype)
    def train_step(self, dist_batch: DistributedBatch) -> grpo.GRPOLossMetrics:
        """Executes optimization step computing GRPO loss with KL penalty."""
        self._optimizer.zero_grad()

        actor_token_log_probs = _compute_logprobs(self._actor, dist_batch)

        with torch.no_grad():
            reference_token_log_probs = _compute_logprobs(
                self._reference, dist_batch, detach=True
            )

        completion_mask = dist_batch.loss_mask[:, 1:]

        loss, metrics = grpo.compute_grpo_loss(
            log_probs=actor_token_log_probs,
            old_log_probs=dist_batch.old_log_probs,
            ref_log_probs=reference_token_log_probs,
            advantages=dist_batch.advantages,
            completion_mask=completion_mask,
            clip_eps=self._clip_eps,
            beta=self._beta,
        )

        loss.backward()
        self._optimizer.step()

        return metrics
