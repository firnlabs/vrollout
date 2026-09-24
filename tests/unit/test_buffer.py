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
    request_id: int | str,
    worker_id: int,
    prompt_len: int,
    completion_len: int,
    vocab_size: int,
) -> RolloutTrajectory:
    vocab = list(range(vocab_size))

    return RolloutTrajectory(
        request_id=str(request_id),
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

    def test_get_rank_batch_local(
        self, scored_rollouts: list[ScoredRollout], prompt_len: int
    ) -> None:
        buffer = vb.DistributedRolloutBuffer(scored_rollouts, pad_token_id=0)
        world_size = 2
        per_rank = len(scored_rollouts) // world_size
        exp_max_seq_len = 6

        for rank in range(world_size):
            batch = buffer.get_rank_batch(rank, world_size)
            assert isinstance(batch, DistributedBatch)
            assert batch.input_ids.shape == (per_rank, exp_max_seq_len)
            assert batch.loss_mask.shape == (per_rank, exp_max_seq_len)
            assert batch.attention_mask.shape == (per_rank, exp_max_seq_len)
            assert batch.rewards.shape == (per_rank,)

            rank_rollouts = scored_rollouts[rank * per_rank : (rank + 1) * per_rank]
            for i, sr in enumerate(rank_rollouts):
                p_ids = list(sr.trajectory.prompt_token_ids)
                c_ids = list(sr.trajectory.completion_token_ids)
                total_len = len(p_ids) + len(c_ids)

                expected_ids = torch.tensor(p_ids + c_ids, dtype=torch.int64)
                assert torch.equal(batch.input_ids[i, :total_len], expected_ids)
                assert not torch.any(batch.loss_mask[i, :prompt_len]).item()
                assert torch.all(batch.loss_mask[i, prompt_len:total_len]).item()
                assert torch.all(batch.attention_mask[i, :total_len]).item()

                if total_len < exp_max_seq_len:
                    assert torch.all(batch.input_ids[i, total_len:] == 0).item()
                    assert torch.all(~batch.attention_mask[i, total_len:]).item()
                    assert torch.all(~batch.loss_mask[i, total_len:]).item()

                assert batch.rewards[i].item() == pytest.approx(sr.reward)

    def test_get_rank_batch_device(self, scored_rollouts: list[ScoredRollout]) -> None:
        buffer = vb.DistributedRolloutBuffer(scored_rollouts)
        batch = buffer.get_rank_batch(rank=0, world_size=2, device="cpu")
        assert batch.input_ids.device == torch.device("cpu")
        assert batch.loss_mask.device == torch.device("cpu")

    def test_get_rank_batch_validation(
        self, scored_rollouts: list[ScoredRollout]
    ) -> None:
        buffer = vb.DistributedRolloutBuffer(scored_rollouts)

        with pytest.raises(ValueError, match="world_size must be positive"):
            buffer.get_rank_batch(rank=0, world_size=0)

        with pytest.raises(IndexError, match="out of bounds"):
            buffer.get_rank_batch(rank=2, world_size=2)

        with pytest.raises(IndexError, match="out of bounds"):
            buffer.get_rank_batch(rank=-1, world_size=2)

        with pytest.raises(ValueError, match="can't be divided evenly"):
            buffer.get_rank_batch(rank=0, world_size=3)

        empty_buffer = vb.DistributedRolloutBuffer([])
        with pytest.raises(ValueError, match="empty buffer"):
            empty_buffer.get_rank_batch(rank=0, world_size=2)

    def test_get_rank_batch_multiprocess(
        self, scored_rollouts: list[ScoredRollout], prompt_len: int, run_multiprocess
    ) -> None:
        run_multiprocess(_test_get_rank_batch_worker, scored_rollouts, prompt_len)
