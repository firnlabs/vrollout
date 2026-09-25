"""Unit tests for engine.py."""

from collections.abc import Callable

import pytest
from torch import optim

from vrollout.engine import DistributedRLEngine
from vrollout.scorer import ScoredRollout
from vrollout.types import Rank
from vrollout.types import WorldSize


@pytest.fixture
def engine_builder(
    actor_model, ref_model
) -> Callable[[Rank, WorldSize], DistributedRLEngine]:
    def _engine_builder(rank=None, world_size=None):
        optimizer = optim.AdamW(actor_model.parameters())
        return DistributedRLEngine(
            actor=actor_model,
            reference=ref_model,
            optimizer=optimizer,
            rank=rank,
            world_size=world_size,
        )

    return _engine_builder


class TestDistributedRLEngine:
    """Tests for DistributedRLEngine."""

    @pytest.mark.parametrize(
        "rank, world_size",
        [
            (None, None),
            (1, 2),
        ],
    )
    def test_train_iteration_local(
        self,
        rank: Rank | None,
        world_size: WorldSize | None,
        engine_builder: Callable[[Rank, WorldSize], DistributedRLEngine],
        scored_rollouts: list[ScoredRollout],
    ):
        dist_engine = engine_builder(rank, world_size)

        metrics = dist_engine.train_iteration(scored_rollouts)

        assert metrics.avg_loss > 0.0
        assert metrics.rank == (rank or 0)
        assert dist_engine._world_size == (world_size or 1)
        assert dist_engine._receivers == []

    def test_error_train_iteration(self, engine_builder):
        dist_engine = engine_builder()

        with pytest.raises(ValueError, match="`scored_rollouts` cannot be empty."):
            dist_engine.train_iteration([])
