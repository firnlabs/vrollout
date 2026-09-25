"""Broadcasts unshared weights multiple receivers."""

from collections.abc import Mapping
from typing import Protocol

from torch import nn
from torch import Tensor
from torch.distributed.tensor import DTensor

from vrollout.types import Rank


class WeightsReceiver(Protocol):
    def update_weights(self, weights: Mapping[str, Tensor]) -> None: ...


class WeightBroadcaster:
    """Broadcasts unshared weights from FSDP to Ray workers.

    This should be created and run within an PyTorch FSDP2 process.
    """

    def __init__(
        self, rank: Rank, receivers: list[WeightsReceiver] | None = None
    ) -> None:
        self._rank = rank
        self._receivers = receivers or []

    def extract_and_broadcast(self, model: nn.Module) -> None:
        """Extracts model unsharded parameters and broadcasts them to Ray workers."""
        weights = {}
        for n, p in model.named_parameters():
            full_p = p.full_tensor() if isinstance(p, DTensor) else p
            weights[n] = full_p.detach()

        if self._rank == 0 and self._receivers:
            for r in self._receivers:
                r.update_weights(weights)
