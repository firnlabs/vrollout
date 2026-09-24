"""Tests for buffer.py."""

import random

import pytest
import torch
from torch.distributed.device_mesh import DeviceMesh

from vrollout import buffer as vb
from vrollout.buffer import DistributedBatch
from vrollout.scorer import ScoredRollout
from vrollout.worker import RolloutTrajectory


def _create_rollout_trajectory(
    *,
    request_id: int,
    worker_id: int,
    prompt_len: int,
    completion_len: int,
    vocab_size: int,
) -> RolloutTrajectory:
    vocab = list(range(vocab_size))

    return RolloutTrajectory(
        request_id=request_id,
        prompt=f"Prompt {request_id}",
        prompt_token_ids=random.choices(vocab, k=prompt_len),
        completion_token_ids=random.choices(vocab, k=completion_len),
        logprobs=[random.random() for _ in range(completion_len)],
        finish_reason="stop",
        text=f"Text {request_id}",
        worker_id=worker_id,
    )


@pytest.fixture
def vocab_size() -> int:
    return 16


@pytest.fixture
def num_rollouts() -> int:
    return 8


@pytest.fixture
def prompt_len() -> int:
    return 2


@pytest.fixture
def scored_rollouts(
    num_rollouts: int, prompt_len: int, vocab_size: int
) -> list[ScoredRollout]:
    scored_rollouts = []
    for i in range(num_rollouts):
        val = i % 2  # 0 or 1
        sr = ScoredRollout(
            reward=val,
            trajectory=_create_rollout_trajectory(
                request_id=i,
                worker_id=i,
                prompt_len=prompt_len,
                completion_len=3 + val,  # length is 3 or 4
                vocab_size=vocab_size,
            ),
        )
        scored_rollouts.append(sr)
    return scored_rollouts


def _test_get_rank_batch_worker(
    rank: int,
    world_size: int,
    mesh: DeviceMesh,
    scored_rollouts: list[ScoredRollout],
    prompt_len: int,
) -> None:
    _ = mesh  # unused
    buffer = vb.DistributedRolloutBuffer(scored_rollouts)

    batch = buffer.get_rank_batch(rank, world_size)

    assert isinstance(batch, DistributedBatch)
    exp_max_seq_len = 6
    exp_batch_size = len(scored_rollouts) // world_size
    assert batch.input_ids.shape == (exp_batch_size, exp_max_seq_len)
    assert batch.loss_mask.shape == (exp_batch_size, exp_max_seq_len)
    assert batch.attention_mask.shape == (exp_batch_size, exp_max_seq_len)
    assert batch.rewards.shape == (exp_batch_size,)
    assert torch.all(~batch.loss_mask[:, :prompt_len]).item()


class TestDistributedRolloutBuffer:
    """Tests for DistributedRolloutBuffer."""

    def test_get_rank_batch(
        self, scored_rollouts: list[ScoredRollout], prompt_len: int, run_multiprocess
    ):

        run_multiprocess(_test_get_rank_batch_worker, scored_rollouts, prompt_len)
