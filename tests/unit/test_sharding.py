"""Tests for FSDP2 sharding helpers and data-parallel loss normalization."""

from collections.abc import Iterator
import copy
import dataclasses

import pytest
import torch
from torch import distributed as dist
from torch import nn
from torch import optim
from torch.distributed import fsdp
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor

from vrollout import sharding
from vrollout import testing
from vrollout import trainer as tr
from vrollout.buffer import DistributedBatch
from vrollout.types import Rank
from vrollout.types import WorldSize


def test_find_transformer_blocks_toy_model() -> None:
    model = testing.create_model("test-model")

    blocks = sharding.find_transformer_blocks(model)

    assert len(blocks) == testing.TEST_MODEL_SPECS["test-model"]["num_layers"]
    assert all(isinstance(b, testing.ToyBlock) for b in blocks)


def test_find_transformer_blocks_hf_style_path() -> None:
    num_layers = 3

    class Inner(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layers = nn.ModuleList([nn.Linear(2, 2) for _ in range(num_layers)])

    class Outer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = Inner()

    blocks = sharding.find_transformer_blocks(Outer())

    assert len(blocks) == num_layers


def test_find_transformer_blocks_none() -> None:
    assert sharding.find_transformer_blocks(nn.Linear(2, 2)) == []


@pytest.fixture
def single_rank_mesh() -> Iterator[DeviceMesh]:
    """Single-rank mesh for calling `shard_model` in-process, where coverage traces it.

    Spawned ranks in `run_multiprocess` are not traced by coverage.
    """
    dist.init_process_group("gloo", store=dist.HashStore(), rank=0, world_size=1)
    try:
        yield init_device_mesh("cpu", (1,))
    finally:
        dist.destroy_process_group()


def test_shard_model_wraps_blocks_and_root(single_rank_mesh: DeviceMesh) -> None:
    model = testing.create_model("test-model")

    sharded = sharding.shard_model(model, single_rank_mesh)

    assert sharded is model
    assert isinstance(sharded, fsdp.FSDPModule)
    assert all(
        isinstance(b, fsdp.FSDPModule)
        for b in sharding.find_transformer_blocks(sharded)
    )
    assert all(isinstance(p, DTensor) for p in sharded.parameters())


def test_shard_model_without_blocks_wraps_root_only(
    single_rank_mesh: DeviceMesh,
) -> None:
    model = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2))

    sharded = sharding.shard_model(model, single_rank_mesh)

    assert isinstance(sharded, fsdp.FSDPModule)
    assert not any(isinstance(m, fsdp.FSDPModule) for m in sharded.children())
    assert all(isinstance(p, DTensor) for p in sharded.parameters())


def _make_batch(vocab_size: int) -> DistributedBatch:
    """Builds a 4-sequence batch whose two halves have very different token counts.

    Rank 0 receives short completions and rank 1 long ones, so a mean of per-rank token
    means differs from the global token mean.
    """
    completion_lens = [2, 3, 10, 9]
    batch_size = 4
    seq_len = 12
    prompt_len = 2
    gen = torch.Generator().manual_seed(42)
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), generator=gen)
    loss_mask = torch.zeros((batch_size, seq_len), dtype=torch.bool)

    for i, n in enumerate(completion_lens):
        loss_mask[i, prompt_len : prompt_len + n] = True

    attention_mask = torch.ones((batch_size, seq_len), dtype=torch.bool)
    rewards = torch.tensor([1.0, 0.0, 0.0, 1.0])
    advantages = torch.tensor([1.0, -1.0, -1.0, 1.0])
    old_log_probs = torch.full((batch_size, seq_len - 1), -3.5)

    return DistributedBatch(
        input_ids=input_ids,
        loss_mask=loss_mask,
        attention_mask=attention_mask,
        rewards=rewards,
        advantages=advantages,
        old_log_probs=old_log_probs,
    )


def _slice_batch(batch: DistributedBatch, start: int, end: int) -> DistributedBatch:
    return DistributedBatch(
        **{f.name: getattr(batch, f.name)[start:end] for f in dataclasses.fields(batch)}
    )


def _make_trainer(
    actor: nn.Module, reference: nn.Module, max_grad_norm: float | None
) -> tr.ModelTrainer:
    # SGD (not Adam) so the update is proportional to the gradient: Adam's first step
    # is ~lr * sign(grad) and would hide a wrong loss normalization
    return tr.ModelTrainer(
        actor=actor,
        reference=reference,
        optimizer=optim.SGD(actor.parameters(), lr=0.5),
        microbatch_size=1,
        max_grad_norm=max_grad_norm,
    )


def _fsdp_matches_single_process_worker(
    rank: Rank,
    world_size: WorldSize,
    mesh: DeviceMesh,
    vocab_size: int,
    max_grad_norm: float | None,
) -> None:
    torch.manual_seed(42)
    base_actor = testing.create_model("test-model")
    base_ref = copy.deepcopy(base_actor)
    for p in base_ref.parameters():
        p.requires_grad = False

    full_batch = _make_batch(vocab_size)

    # single-process baseline on the full batch (identical on every rank)
    local_actor = copy.deepcopy(base_actor)
    local_metrics = _make_trainer(
        local_actor, copy.deepcopy(base_ref), max_grad_norm
    ).train_step(full_batch)

    # FSDP2 data parallel: each rank trains on its half of the batch
    sharded_actor = sharding.shard_model(copy.deepcopy(base_actor), mesh)
    sharded_ref = sharding.shard_model(copy.deepcopy(base_ref), mesh)
    per_rank = full_batch.input_ids.shape[0] // world_size
    rank_batch = _slice_batch(full_batch, rank * per_rank, (rank + 1) * per_rank)
    fsdp_metrics = _make_trainer(sharded_actor, sharded_ref, max_grad_norm).train_step(
        rank_batch
    )

    assert fsdp_metrics.loss == pytest.approx(local_metrics.loss, rel=1e-5, abs=1e-6)
    assert fsdp_metrics.mean_kl == pytest.approx(
        local_metrics.mean_kl, rel=1e-5, abs=1e-6
    )
    assert fsdp_metrics.clip_fraction == pytest.approx(
        local_metrics.clip_fraction, rel=1e-5, abs=1e-6
    )

    for (n1, p1), (n2, p2) in zip(
        sharded_actor.named_parameters(), local_actor.named_parameters(), strict=True
    ):
        assert n1 == n2
        full = p1.full_tensor() if isinstance(p1, DTensor) else p1
        torch.testing.assert_close(full.detach(), p2.detach(), atol=1e-5, rtol=1e-5)

    # the update must actually move parameters for the comparison to be meaningful
    params_changed = any(
        not torch.allclose(p1.detach(), p2.detach())
        for p1, p2 in zip(
            local_actor.parameters(), base_actor.parameters(), strict=True
        )
    )
    assert params_changed


# the test batch has a gradient norm of ~0.62, so 0.1 scales the update by ~6x and a
# per-rank (instead of global) norm would change the result
@pytest.mark.parametrize("max_grad_norm", [None, 0.1])
def test_fsdp_train_step_matches_single_process(
    run_multiprocess, vocab_size, max_grad_norm
) -> None:
    run_multiprocess(_fsdp_matches_single_process_worker, vocab_size, max_grad_norm)
