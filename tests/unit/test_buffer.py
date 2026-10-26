import math
from typing import Callable

import pytest
import torch
from torch.distributed.device_mesh import DeviceMesh

from vrollout import buffer as vb
from vrollout.buffer import DistributedBatch
from vrollout.scorer import ScoredRollout


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


@pytest.mark.parametrize(
    "batch_size, split_size, exp_splits",
    [
        (8, 2, [2, 2, 2, 2]),
        (8, 3, [3, 3, 2]),
    ],
)
def test_split_distributed_batch(
    batch_size: int,
    split_size: int,
    exp_splits: list[int],
    distributed_batch_factory: Callable[[int, int], vb.DistributedBatch],
):
    seq_len = 2
    batch = distributed_batch_factory(batch_size, seq_len)

    result = vb.split_distributed_batch(batch, split_size)

    assert len(result) == math.ceil(batch_size / split_size)
    splits = [r.input_ids.shape[0] for r in result]
    assert splits == exp_splits


@pytest.mark.parametrize("split_size", [0, -1])
def test_split_distributed_batch_error_split_size(
    split_size,
    distributed_batch_factory: Callable[[int, int], vb.DistributedBatch],
):
    batch_size = 8
    seq_len = 2
    batch = distributed_batch_factory(batch_size, seq_len)

    with pytest.raises(ValueError, match="split_size must be greater than 0"):
        vb.split_distributed_batch(batch, split_size)
