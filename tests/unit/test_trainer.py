from collections.abc import Sequence
import copy
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

from vrollout import trainer as tr
from vrollout.buffer import DistributedBatch
from vrollout.testing.models import ToyModel
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
    rewards = torch.randint(0, 2, (batch_size,)).float()
    advantages = rewards - rewards.mean()
    old_log_probs = torch.zeros((batch_size, max_seq_len - 1), dtype=torch.float32)

    return functools.partial(
        DistributedBatch,
        input_ids=input_ids,
        loss_mask=loss_mask,
        attention_mask=attention_mask,
        rewards=rewards,
        advantages=advantages,
        old_log_probs=old_log_probs,
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
    microbatch_size: int | None = 2,
):
    batch = create_batch()
    optimizer = optim.AdamW(actor_model.parameters())
    trainer = tr.ModelTrainer(
        actor=actor_model,
        reference=ref_model,
        optimizer=optimizer,
        microbatch_size=microbatch_size,
    )

    init_actor_params = _get_params_copy(actor_model)
    init_ref_params = _get_params_copy(ref_model)

    for p in ref_model.parameters():
        assert p.grad is None

    init_loss = None
    metrics = None
    for _ in range(5):
        metrics = trainer.train_step(batch)
        if init_loss is None:
            init_loss = metrics.loss

    for p in ref_model.parameters():
        assert p.grad is None

    assert metrics is not None
    assert math.isfinite(metrics.loss)
    assert math.isfinite(metrics.mean_kl)
    assert 0.0 <= metrics.clip_fraction <= 1.0
    assert init_loss is not None
    assert metrics.loss < init_loss
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
    def test_train_step_local(self, create_batch, actor_model, ref_model):
        _test_train_step_helper(create_batch, actor_model, ref_model)

    def test_train_step(
        self, create_batch, actor_model, ref_model, shard_model, run_multiprocess
    ):
        run_multiprocess(
            _test_train_step_worker, create_batch, actor_model, ref_model, shard_model
        )


@pytest.mark.parametrize("use_checkpoint", [True, False])
def test_compute_logprobs(
    use_checkpoint: bool,
    actor_model: ToyModel,
    ref_model: ToyModel,
    distributed_batch_factory: Callable[[int, int], DistributedBatch],
    mocker,
):
    batch_size = 2
    seq_len = 12
    chunk_size = 3
    dist_batch = distributed_batch_factory(batch_size, seq_len)
    model = actor_model if use_checkpoint else ref_model
    mock_checkpoint = mocker.patch.object(
        tr.checkpoint, "checkpoint", wraps=tr.checkpoint.checkpoint
    )

    if use_checkpoint:
        result = tr._compute_logprobs(model, dist_batch, chunk_size)
    else:
        with torch.no_grad():
            result = tr._compute_logprobs(model, dist_batch, chunk_size)

    assert result.shape == (batch_size, seq_len - 1)
    if use_checkpoint:
        assert mock_checkpoint.call_count == seq_len // chunk_size
    else:
        mock_checkpoint.assert_not_called()


def _run_train_step(
    actor_base: nn.Module,
    ref_model: nn.Module,
    batch: DistributedBatch,
    microbatch_size: int | None,
    logit_chunk_size: int,
) -> tuple[tr.grpo.GRPOLossMetrics, dict[str, Tensor], dict[str, Tensor]]:
    actor = copy.deepcopy(actor_base)
    optimizer = optim.AdamW(actor.parameters(), lr=1e-3)
    trainer = tr.ModelTrainer(
        actor=actor,
        reference=ref_model,
        optimizer=optimizer,
        microbatch_size=microbatch_size,
        logit_chunk_size=logit_chunk_size,
    )
    metrics = trainer.train_step(batch)
    grads = {
        n: p.grad.detach().clone()
        for n, p in actor.named_parameters()
        if p.grad is not None
    }
    params = _get_params_copy(actor)
    return metrics, grads, params


def test_train_step_microbatch_chunk_equivalence(
    create_batch: Callable[[], DistributedBatch],
    actor_model: ToyModel,
    ref_model: ToyModel,
) -> None:
    batch = create_batch()
    # mask out tokens in decreasing manner (simulates variable length completions)
    for i in range(batch.loss_mask.shape[0]):
        batch.loss_mask[i, 2 + i + 1 :] = False

    # full batch, no chunking
    full_metrics, full_grads, full_params = _run_train_step(
        actor_model, ref_model, batch, microbatch_size=None, logit_chunk_size=128
    )

    # microbatches, chunking
    mb_metrics, mb_grads, mb_params = _run_train_step(
        actor_model, ref_model, batch, microbatch_size=2, logit_chunk_size=3
    )

    assert math.isclose(mb_metrics.loss, full_metrics.loss, rel_tol=1e-5, abs_tol=1e-6)
    assert math.isclose(
        mb_metrics.mean_kl, full_metrics.mean_kl, rel_tol=1e-5, abs_tol=1e-6
    )
    assert math.isclose(
        mb_metrics.clip_fraction,
        full_metrics.clip_fraction,
        rel_tol=1e-5,
        abs_tol=1e-6,
    )
    _assert_params_equal(list(mb_grads.items()), list(full_grads.items()))
    _assert_params_equal(list(mb_params.items()), list(full_params.items()))
