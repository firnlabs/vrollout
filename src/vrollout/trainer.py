"""Policy gradient optimization with reference model regularization."""

from beartype import beartype
from jaxtyping import Float
from jaxtyping import jaxtyped
import torch
from torch import nn
from torch import optim
from torch import Tensor
from torch.nn import functional as F

from vrollout.buffer import DistributedBatch


@jaxtyped(typechecker=beartype)
def _compute_logprobs(
    model: nn.Module, dist_batch: DistributedBatch, detach: bool = False
) -> Float[Tensor, "batch seq_len_minus_1"]:
    logits = model(dist_batch.input_ids, attention_mask=dist_batch.attention_mask)
    log_probs = F.log_softmax(logits, dim=-1)
    targets = dist_batch.input_ids[..., 1:].unsqueeze(-1)
    token_log_probs = torch.gather(
        log_probs[..., :-1, :],
        dim=-1,
        index=targets,
    ).squeeze(-1)
    return token_log_probs.detach() if detach else token_log_probs


class ModelTrainer:
    """Dual actor-reference policy trainer with KL-divergence penalty."""

    def __init__(
        self,
        actor: nn.Module,
        reference: nn.Module,
        optimizer: optim.Optimizer,
        beta: float = 0.1,
    ) -> None:
        self._actor = actor
        self._reference = reference
        self._optimizer = optimizer
        self._beta = beta

    @jaxtyped(typechecker=beartype)
    def train_step(self, dist_batch: DistributedBatch) -> float:
        """Executes optimization step computing REINFORCE loss with KL penalty."""
        self._optimizer.zero_grad()

        actor_token_log_probs = _compute_logprobs(self._actor, dist_batch)

        with torch.no_grad():
            reference_token_log_probs = _compute_logprobs(
                self._reference, dist_batch, detach=True
            )

        mask = dist_batch.loss_mask[:, 1:].float()
        actor_seq_log_probs = (actor_token_log_probs * mask).sum(dim=-1)
        kl_div = (
            (actor_token_log_probs.detach() - reference_token_log_probs) * mask
        ).sum(dim=-1)
        advantages = dist_batch.rewards - self._beta * kl_div
        loss = -(actor_seq_log_probs * advantages).mean()

        loss.backward()
        self._optimizer.step()

        return loss.item()
