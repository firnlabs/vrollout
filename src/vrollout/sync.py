"""Weight extraction and broadcasting from FSDP2 to inference receivers."""

from collections.abc import Mapping
from typing import Protocol
from typing import runtime_checkable

from beartype import beartype
from jaxtyping import jaxtyped
from jaxtyping import Shaped
from torch import nn
from torch import Tensor
from torch.distributed.tensor import DTensor

from vrollout.types import Rank


@runtime_checkable
class WeightsReceiver(Protocol):
    """Protocol for entities capable of receiving updated model weights."""

    def update_weights(self, weights: Mapping[str, Shaped[Tensor, "..."]]) -> None: ...


class WeightBroadcaster:
    """Broadcasts gathered FSDP parameters from rank 0 to receivers."""

    def __init__(
        self, rank: Rank, receivers: list[WeightsReceiver] | None = None
    ) -> None:
        self._rank = rank
        self._receivers = receivers or []

    @jaxtyped(typechecker=beartype)
    def extract_and_broadcast(self, model: nn.Module) -> None:
        """Extracts full parameters and dispatches detached weights to receivers."""
        weights = {}
        for n, p in model.named_parameters():
            full_p = p.full_tensor() if isinstance(p, DTensor) else p
            if self._rank == 0:
                weights[n] = full_p.detach()

        if self._rank == 0 and self._receivers:
            for r in self._receivers:
                r.update_weights(weights)
