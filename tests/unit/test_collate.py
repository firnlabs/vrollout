"""Tests for collate.py."""

import pytest
import torch

from vrollout import collate
from vrollout import scorer
from vrollout import worker


def _create_scored_rollout(
    completed_tokens: list[int], reward: float = 1.0
) -> scorer.ScoredRollout:
    traj = worker.RolloutTrajectory(
        request_id="req_1",
        prompt="Prompt",
        prompt_token_ids=[1],
        completion_token_ids=completed_tokens,
        logprobs=[-0.2] * len(completed_tokens),
        finish_reason="stop",
        text="text",
        worker_id=0,
    )
    return scorer.ScoredRollout(trajectory=traj, reward=reward)


@pytest.fixture
def scored_rollouts() -> list[scorer.ScoredRollout]:
    return [
        _create_scored_rollout([10, 20, 30]),
        _create_scored_rollout([10, 20, 30, 40, 50], reward=0.0),
    ]


@pytest.fixture
def batch_size(scored_rollouts) -> int:
    return len(scored_rollouts)


@pytest.fixture
def max_seq_len(scored_rollouts) -> int:
    return max([len(sr.trajectory.completion_token_ids) for sr in scored_rollouts])


@pytest.fixture
def values(scored_rollouts) -> list[list[float]]:
    out = []
    for sr in scored_rollouts:
        row = [float(x) for x in range(len(sr.trajectory.completion_token_ids))]
        out.append(row)
    return out


@pytest.fixture
def next_values(batch_size) -> list[float]:
    return [0.2 * i for i in range(1, batch_size + 1)]


def test_collate_rollouts_for_gae(
    batch_size, max_seq_len, scored_rollouts, values, next_values
):
    result = collate.collate_rollouts_for_gae(scored_rollouts, values, next_values)

    assert isinstance(result, collate.RolloutBatch)

    # check shapes
    assert result.completion_token_ids.shape == (batch_size, max_seq_len)
    assert result.rewards.shape == (batch_size, max_seq_len)
    assert result.values.shape == (batch_size, max_seq_len)
    assert result.next_values.shape == (batch_size,)
    assert result.dones.shape == (batch_size, max_seq_len)
    assert result.seq_lens.shape == (batch_size,)

    # check dtypes
    assert result.completion_token_ids.dtype == torch.int64
    assert result.rewards.dtype == torch.float32
    assert result.values.dtype == torch.float32
    assert result.next_values.dtype == torch.float32
    assert result.dones.dtype == torch.float32
    assert result.seq_lens.dtype == torch.int64

    # check terminal values (batch 0)
    seq_len = result.seq_lens[0]
    assert result.rewards[0, seq_len - 1].item() == 1.0
    assert result.dones[0, 0].item() == pytest.approx(0.0, abs=1e-5)
    assert result.dones[0, seq_len - 1].item() == 1.0  # terminal
    assert result.dones[0, -1].item() == 1.0  # padded
    torch.testing.assert_close(
        result.completion_token_ids[0], torch.tensor([10, 20, 30, 0, 0])
    )
    torch.testing.assert_close(
        result.attention_mask[0], torch.tensor([1.0, 1.0, 1.0, 0.0, 0.0])
    )
    torch.testing.assert_close(result.values[0], torch.tensor(values[0] + [0, 0]))
    assert result.next_values[0].item() == pytest.approx(0.2)


def test_collate_empty_rollouts_raises_error():
    with pytest.raises(ValueError, match="`scored_rollouts` is empty"):
        collate.collate_rollouts_for_gae([])


def test_collate_insufficient_values_raises_error(scored_rollouts):
    with pytest.raises(ValueError, match="Number of `values`"):
        collate.collate_rollouts_for_gae(scored_rollouts, values=[[1.0]])


def test_collate_insufficient_next_values_raises_error(scored_rollouts):
    with pytest.raises(ValueError, match="Number of `next_values`"):
        collate.collate_rollouts_for_gae(scored_rollouts, next_values=[0.5])


def test_collate_defaults_none_values(scored_rollouts, batch_size, max_seq_len):
    result = collate.collate_rollouts_for_gae(scored_rollouts)

    torch.testing.assert_close(
        result.values, torch.zeros((batch_size, max_seq_len), dtype=torch.float32)
    )
    torch.testing.assert_close(
        result.next_values, torch.zeros((batch_size,), dtype=torch.float32)
    )


def test_collate_single_token_and_empty_token_rollouts():
    rollouts = [
        _create_scored_rollout([42], reward=1.0),  # seq_len == 1
        _create_scored_rollout([], reward=0.0),  # seq_len == 0
    ]
    result = collate.collate_rollouts_for_gae(rollouts)

    assert result.completion_token_ids.shape == (2, 1)
    # Row 0: single token
    assert result.completion_token_ids[0, 0].item() == 42
    assert result.dones[0, 0].item() == 1.0
    assert result.attention_mask[0, 0].item() == 1.0
    assert result.rewards[0, 0].item() == 1.0

    # Row 1: empty token rollout
    assert result.dones[1, 0].item() == 1.0
    assert result.attention_mask[1, 0].item() == 0.0
    assert result.rewards[1, 0].item() == 0.0
