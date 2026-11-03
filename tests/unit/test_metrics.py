"""Unit tests for rollout diagnostic metrics."""

from collections.abc import Callable

import pytest
import torch

from vrollout import metrics
from vrollout import scorer
from vrollout import types


def _create_scored_rollout(
    reward: float,
    logprobs: list[float],
    finish_reason: str = "stop",
) -> scorer.ScoredRollout:
    return scorer.ScoredRollout(
        reward=reward,
        trajectory=types.RolloutTrajectory(
            request_id="r",
            prompt="p",
            prompt_token_ids=[1, 2],
            completion_token_ids=list(range(len(logprobs))),
            logprobs=logprobs,
            finish_reason=finish_reason,
            text="",
            worker_id=0,
        ),
    )


@pytest.fixture
def batch() -> list[scorer.ScoredRollout]:
    # group 0 is all-correct (zero advantage), group 1 has reward variance
    return [
        _create_scored_rollout(1.0, [-1.0, -1.0]),
        _create_scored_rollout(1.0, [-1.0, -1.0, -1.0, -1.0], finish_reason="length"),
        _create_scored_rollout(0.0, [-0.5]),
        _create_scored_rollout(1.0, [-0.5, -0.5, -0.5]),
    ]


def test_compute_rollout_metrics(batch: list[scorer.ScoredRollout]) -> None:
    advantages = torch.tensor([0.0, 0.0, -1.0, 1.0])

    result = metrics.compute_rollout_metrics(batch, advantages, group_size=2)

    assert result.response_length_mean == pytest.approx(2.5)
    assert result.response_length_max == 4
    assert result.truncation_rate == pytest.approx(0.25)
    assert result.zero_advantage_group_frac == pytest.approx(0.5)
    assert result.entropy_proxy == pytest.approx(8.0 / 10)


def test_zero_advantage_group_requires_all_zero() -> None:
    group_size = 3
    # first group has zero advantage (rewards all equal)
    batch = [_create_scored_rollout(r, [-1.0]) for r in (1.0, 1.0, 1.0, 1.0, 0.5, 0.0)]
    advantages = torch.tensor([0.0, 0.0, 0.0, 1.2, 0.0, -1.2])

    result = metrics.compute_rollout_metrics(batch, advantages, group_size=group_size)

    assert result.zero_advantage_group_frac == pytest.approx(0.5)


@pytest.mark.parametrize(
    "group_size",
    [pytest.param(1, id="no_groups"), pytest.param(3, id="indivisible")],
)
def test_compute_rollout_metrics_omits_group_metric_without_groups(
    batch: list[scorer.ScoredRollout], group_size: int
) -> None:
    advantages = torch.zeros(len(batch))

    result = metrics.compute_rollout_metrics(batch, advantages, group_size)

    assert result.zero_advantage_group_frac is None
    assert "zero_advantage_group_frac" not in result.to_dict()


def test_compute_rollout_metrics_empty_completions_have_zero_entropy() -> None:
    batch = [_create_scored_rollout(0.0, []), _create_scored_rollout(1.0, [])]

    result = metrics.compute_rollout_metrics(batch, torch.zeros(2), group_size=2)

    assert result.response_length_mean == 0.0
    assert result.entropy_proxy == 0.0


def test_compute_rollout_metrics_rejects_empty_batch() -> None:
    with pytest.raises(ValueError, match="empty"):
        metrics.compute_rollout_metrics([], torch.zeros(0), group_size=2)


def test_compute_rollout_metrics_rejects_mismatched_advantages(
    batch: list[scorer.ScoredRollout],
) -> None:
    with pytest.raises(ValueError, match="advantages"):
        metrics.compute_rollout_metrics(batch, torch.zeros(3), group_size=2)


@pytest.mark.parametrize(
    "compute",
    [
        pytest.param(metrics.compute_rollout_metrics, id="rollout_metrics"),
        pytest.param(metrics.sequence_rows, id="sequence_rows"),
    ],
)
@pytest.mark.parametrize("group_size", [0, -1])
def test_rejects_non_positive_group_size(
    batch: list[scorer.ScoredRollout],
    compute: Callable[..., object],
    group_size: int,
) -> None:
    advantages = torch.zeros(len(batch))

    with pytest.raises(ValueError, match="group_size"):
        compute(batch, advantages, group_size)


def test_sequence_rows(batch: list[scorer.ScoredRollout]) -> None:
    advantages = torch.tensor([0.0, 0.0, -1.0, 1.0])

    rows = metrics.sequence_rows(batch, advantages, group_size=2)

    assert rows[1] == metrics.SequenceRow(
        group=0, reward=1.0, advantage=0.0, length=4, finish_reason="length"
    )
    assert rows[2] == metrics.SequenceRow(
        group=1, reward=0.0, advantage=-1.0, length=1, finish_reason="stop"
    )
    assert len(rows) == len(batch)


def test_sequence_row_to_list_matches_columns() -> None:
    row = metrics.SequenceRow(
        group=3, reward=0.5, advantage=-0.2, length=7, finish_reason="stop"
    )

    result = row.to_list()

    assert dict(zip(metrics.SEQUENCE_COLUMNS, result, strict=True)) == {
        "group": 3,
        "reward": 0.5,
        "advantage": -0.2,
        "length": 7,
        "finish_reason": "stop",
    }
