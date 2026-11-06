from collections.abc import Callable
import math
import time

import pytest
from torch import optim

from vrollout import trainer
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

        assert math.isfinite(metrics.avg_loss)
        assert math.isfinite(metrics.avg_kl)
        assert 0.0 <= metrics.avg_clip_fraction <= 1.0
        assert metrics.avg_logprob_abs_diff >= 0.0
        assert metrics.avg_mismatch_kl >= 0.0
        assert metrics.avg_grad_norm > 0.0
        assert metrics.rank == (rank or 0)
        assert dist_engine._world_size == (world_size or 1)
        assert dist_engine._receivers == []

    def test_train_iteration_times_weight_sync_separately(
        self, actor_model, ref_model, scored_rollouts: list[ScoredRollout]
    ):
        sync_delay_s = 0.2

        class _SlowReceiver:
            def update_weights(self, weights) -> None:
                _ = weights  # unused
                time.sleep(sync_delay_s)

        dist_engine = DistributedRLEngine(
            actor=actor_model,
            reference=ref_model,
            optimizer=optim.AdamW(actor_model.parameters()),
            receivers=[_SlowReceiver()],
        )

        metrics = dist_engine.train_iteration(scored_rollouts)

        assert metrics.weight_sync_time_s >= sync_delay_s
        assert 0.0 < metrics.train_time_s < sync_delay_s

    def test_train_iteration_forwards_lr_scheduler(
        self, actor_model, ref_model, scored_rollouts: list[ScoredRollout]
    ):
        optimizer = optim.AdamW(actor_model.parameters(), lr=1e-3)
        dist_engine = DistributedRLEngine(
            actor=actor_model,
            reference=ref_model,
            optimizer=optimizer,
            lr_scheduler=trainer.linear_warmup_scheduler(optimizer, warmup_steps=2),
        )

        lrs = [dist_engine.train_iteration(scored_rollouts).lr for _ in range(3)]

        assert lrs == pytest.approx([5e-4, 1e-3, 1e-3])

    def test_error_train_iteration(self, engine_builder):
        dist_engine = engine_builder()

        with pytest.raises(ValueError, match="`scored_rollouts` cannot be empty."):
            dist_engine.train_iteration([])
