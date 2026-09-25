"""Tests for trainer.py."""

from collections.abc import Sequence
import functools
import math
from typing import Callable

import pytest
import torch
from torch import nn
from torch import optim
from torch import Tensor
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor

from tests.conftest import ToyModel
from vrollout import trainer as tr
from vrollout.buffer import DistributedBatch
from vrollout.types import Rank
from vrollout.types import WorldSize

torch.manual_seed(42)


@pytest.fixture
def batch_size() -> int:
    return 4


@pytest.fixture
def max_seq_len() -> int:
    return 8


@pytest.fixture
def create_batch(
    batch_size: int, max_seq_len: int, vocab_size: int
) -> Callable[[], DistributedBatch]:
    input_ids = torch.randint(0, vocab_size, (batch_size, max_seq_len))
    prompt_len = 2
    loss_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    loss_mask[:, prompt_len:] = True
    attention_mask = torch.ones_like(input_ids, dtype=torch.bool)

    return functools.partial(
        DistributedBatch,
        input_ids=input_ids,
        loss_mask=loss_mask,
        attention_mask=attention_mask,
        rewards=torch.randint(0, 2, (batch_size,)),
    )


def _to_full(t: Tensor) -> Tensor:
    return t.full_tensor() if isinstance(t, DTensor) else t


def _get_params_copy(model: nn.Module) -> dict[str, Tensor]:
    return {n: _to_full(p).detach().clone() for n, p in model.named_parameters()}


def _assert_params_equal(
    named_params1: Sequence[tuple[str, Tensor]],
    named_params2: Sequence[tuple[str, Tensor]],
) -> None:
    for (n1, p1), (n2, p2) in zip(named_params1, named_params2, strict=True):
        assert n1 == n2
        torch.testing.assert_close(_to_full(p1), _to_full(p2), atol=1e-5, rtol=1e-5)


def _assert_params_not_equal(
    named_params1: Sequence[tuple[str, Tensor]],
    named_params2: Sequence[tuple[str, Tensor]],
) -> None:
    for (n1, p1), (n2, p2) in zip(named_params1, named_params2, strict=True):
        assert n1 == n2
        assert not torch.allclose(_to_full(p1), _to_full(p2), atol=1e-5, rtol=1e-5)


def _test_train_step_helper(
    create_batch: Callable[[], DistributedBatch],
    actor_model: nn.Module,
    ref_model: nn.Module,
):
    batch = create_batch()
    optimizer = optim.AdamW(actor_model.parameters())
    trainer = tr.ModelTrainer(actor_model, ref_model, optimizer)

    init_actor_params = _get_params_copy(actor_model)
    init_ref_params = _get_params_copy(ref_model)

    for p in ref_model.parameters():
        assert p.grad is None

    init_loss = None
    for _ in range(5):
        loss = trainer.train_step(batch)
        if init_loss is None:
            init_loss = loss

    for p in ref_model.parameters():
        assert p.grad is None

    assert math.isfinite(loss)
    assert init_loss is not None
    assert loss < init_loss
    _assert_params_not_equal(actor_model.named_parameters(), init_actor_params.items())
    _assert_params_equal(ref_model.named_parameters(), init_ref_params.items())


def _test_train_step_worker(
    rank: Rank,
    world_size: WorldSize,
    mesh: DeviceMesh,
    create_batch: Callable[[], DistributedBatch],
    actor_model: ToyModel,
    ref_model: ToyModel,
    shard_model: Callable[[nn.Module, DeviceMesh], nn.Module],
) -> None:
    _ = rank  # unused
    _ = world_size  # unused
    sharded_actor = shard_model(actor_model, mesh)
    sharded_ref = shard_model(ref_model, mesh)
    _test_train_step_helper(create_batch, sharded_actor, sharded_ref)


class TestFSDPTrainer:
    """Tests for FSDPTrainer."""

    def test_train_step_local(self, create_batch, actor_model, ref_model):
        _test_train_step_helper(create_batch, actor_model, ref_model)

    def test_train_step(
        self, create_batch, actor_model, ref_model, shard_model, run_multiprocess
    ):
        run_multiprocess(
            _test_train_step_worker, create_batch, actor_model, ref_model, shard_model
        )
