"""Tests for buffer.py."""

import pytest
import torch

from vrollout import buffer
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
    result = buffer.collate_rollouts_for_gae(scored_rollouts, values, next_values)

    assert isinstance(result, buffer.RolloutBatch)

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
