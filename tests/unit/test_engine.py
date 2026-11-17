from collections.abc import Callable
import dataclasses
import math

import pytest
from torch import optim

from vrollout import trainer
from vrollout.engine import DistributedRLEngine
from vrollout.engine import RLEngineConfig
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


def _with_version(
    scored_rollouts: list[ScoredRollout], weights_version: int
) -> list[ScoredRollout]:
    return [
        ScoredRollout(
            reward=sr.reward,
            trajectory=dataclasses.replace(
                sr.trajectory, weights_version=weights_version
            ),
        )
        for sr in scored_rollouts
    ]


class _CountingReceiver:
    def __init__(self) -> None:
        self.calls = 0

    def update_weights(self, weights) -> None:
        _ = weights  # unused
        self.calls += 1


def _engine_at_version(
    actor_model, ref_model, max_staleness: int | None, policy_version: int
) -> DistributedRLEngine:
    dist_engine = DistributedRLEngine(
        actor=actor_model,
        reference=ref_model,
        optimizer=optim.AdamW(actor_model.parameters()),
        config=RLEngineConfig(max_staleness=max_staleness),
    )
    for _ in range(policy_version):
        dist_engine.sync_weights()
    return dist_engine


class TestStaleness:
    @pytest.mark.parametrize(
        ("max_staleness", "policy_version", "weights_version"),
        [
            pytest.param(0, 1, 0, id="sync_one_behind"),
            pytest.param(1, 2, 0, id="async_two_behind"),
            pytest.param(1, 0, 1, id="ahead_of_policy"),
        ],
    )
    def test_train_iteration_rejects_rollouts_outside_staleness_limit(
        self,
        actor_model,
        ref_model,
        scored_rollouts: list[ScoredRollout],
        max_staleness: int,
        policy_version: int,
        weights_version: int,
    ):
        dist_engine = _engine_at_version(
            actor_model, ref_model, max_staleness, policy_version
        )
        rollouts = _with_version(scored_rollouts, weights_version)

        with pytest.raises(ValueError, match="outside the allowed staleness"):
            dist_engine.train_iteration(rollouts)

    @pytest.mark.parametrize(
        ("max_staleness", "policy_version", "weights_version"),
        [
            pytest.param(0, 3, 3, id="sync_current"),
            pytest.param(1, 3, 3, id="async_current"),
            pytest.param(1, 3, 2, id="async_one_behind"),
            pytest.param(None, 3, 0, id="unchecked"),
        ],
    )
    def test_train_iteration_accepts_rollouts_within_staleness_limit(
        self,
        actor_model,
        ref_model,
        scored_rollouts: list[ScoredRollout],
        max_staleness: int | None,
        policy_version: int,
        weights_version: int,
    ):
        dist_engine = _engine_at_version(
            actor_model, ref_model, max_staleness, policy_version
        )
        rollouts = _with_version(scored_rollouts, weights_version)

        metrics = dist_engine.train_iteration(rollouts)

        assert math.isfinite(metrics.avg_loss)

    def test_train_iteration_leaves_version_and_receivers(
        self, actor_model, ref_model, scored_rollouts: list[ScoredRollout]
    ):
        receiver = _CountingReceiver()
        dist_engine = DistributedRLEngine(
            actor=actor_model,
            reference=ref_model,
            optimizer=optim.AdamW(actor_model.parameters()),
            receivers=[receiver],
        )

        dist_engine.train_iteration(scored_rollouts)

        assert receiver.calls == 0
        assert dist_engine.policy_version == 0

    def test_sync_weights_updates_receivers_and_bumps_version(
        self, actor_model, ref_model
    ):
        receiver = _CountingReceiver()
        dist_engine = DistributedRLEngine(
            actor=actor_model,
            reference=ref_model,
            optimizer=optim.AdamW(actor_model.parameters()),
            receivers=[receiver],
        )

        dist_engine.sync_weights()

        assert receiver.calls == 1
        assert dist_engine.policy_version == 1
