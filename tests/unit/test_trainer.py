from collections.abc import Sequence
import copy
import dataclasses
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
from torch.nn import functional as F

from vrollout import buffer as vb
from vrollout import generation
from vrollout import trainer as tr
from vrollout.buffer import DistributedBatch
from vrollout.scorer import ScoredRollout
from vrollout.testing import MockAsyncLLMEngine
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
) -> tuple[tr.TrainStepMetrics, dict[str, Tensor], dict[str, Tensor]]:
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


def _grad_norm(model: nn.Module) -> float:
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    return torch.linalg.vector_norm(torch.stack([g.norm() for g in grads])).item()


@pytest.mark.parametrize(
    "max_grad_norm",
    [pytest.param(None, id="unclipped"), pytest.param(1e-4, id="clipped")],
)
def test_train_step_reports_pre_clip_grad_norm(
    create_batch: Callable[[], DistributedBatch],
    actor_model: ToyModel,
    ref_model: ToyModel,
    max_grad_norm: float | None,
) -> None:
    trainer = tr.ModelTrainer(
        actor=actor_model,
        reference=ref_model,
        optimizer=optim.SGD(actor_model.parameters(), lr=0.0),
        max_grad_norm=max_grad_norm,
    )

    metrics = trainer.train_step(create_batch())

    applied_norm = _grad_norm(actor_model)
    assert metrics.grad_norm > 0.0
    if max_grad_norm is not None:
        # clip_grad_norm_ scales by max_norm / (total_norm + 1e-6)
        expected = max_grad_norm * metrics.grad_norm / (metrics.grad_norm + 1e-6)
        assert applied_norm == pytest.approx(expected, rel=1e-5)
        assert metrics.grad_norm > applied_norm
    else:
        assert metrics.grad_norm == pytest.approx(applied_norm, rel=1e-5)


def test_train_step_mismatch_is_zero_when_old_policy_matches_current(
    create_batch: Callable[[], DistributedBatch],
    actor_model: ToyModel,
    ref_model: ToyModel,
) -> None:
    batch = create_batch()
    with torch.no_grad():
        current_policy_lps = tr._compute_logprobs(actor_model, batch)
    batch = dataclasses.replace(batch, old_log_probs=current_policy_lps)
    trainer = tr.ModelTrainer(
        actor=actor_model,
        reference=ref_model,
        optimizer=optim.AdamW(actor_model.parameters(), lr=1e-2),
        microbatch_size=2,
    )

    metrics = trainer.train_step(batch)

    assert metrics.logprob_abs_diff == pytest.approx(0.0, abs=1e-6)
    assert metrics.mismatch_kl == pytest.approx(0.0, abs=1e-6)
    assert metrics.clip_fraction == 0.0


@pytest.mark.parametrize("temperature", [0.5, 1.0, 1.7])
def test_compute_logprobs_scores_tokens_under_tempered_softmax(
    temperature: float,
    ref_model: ToyModel,
    distributed_batch_factory: Callable[[int, int], DistributedBatch],
) -> None:
    dist_batch = distributed_batch_factory(3, 11)

    with torch.no_grad():
        result = tr._compute_logprobs(
            ref_model, dist_batch, chunk_size=4, temperature=temperature
        )

    with torch.no_grad():
        logits = ref_model(dist_batch.input_ids)[:, :-1, :]
    expected = (
        F.log_softmax(logits / temperature, dim=-1)
        .gather(-1, dist_batch.input_ids[:, 1:].unsqueeze(-1))
        .squeeze(-1)
    )
    torch.testing.assert_close(result, expected)


@pytest.mark.parametrize("temperature", [0.0, -1.0])
def test_trainer_rejects_non_positive_temperature(
    temperature: float, actor_model: ToyModel, ref_model: ToyModel
) -> None:
    optimizer = optim.SGD(actor_model.parameters(), lr=0.0)

    with pytest.raises(ValueError, match="temperature"):
        tr.ModelTrainer(
            actor=actor_model,
            reference=ref_model,
            optimizer=optimizer,
            temperature=temperature,
        )


@dataclasses.dataclass
class _SamplingParams:
    temperature: float
    max_tokens: int = 6
    logprobs: int = 1
    group_size: int = 3


async def _sample_batch(model: nn.Module, temperature: float) -> DistributedBatch:
    engine = MockAsyncLLMEngine(model=model, model_name="test-model")
    output = await anext(
        engine.generate("prompt", _SamplingParams(temperature=temperature), "req-0")
    )
    rollouts = [
        ScoredRollout(reward=float(i), trajectory=traj)
        for i, traj in enumerate(generation._parse_vllm_output_to_rollouts(output, 0))
    ]
    return vb.DistributedRolloutBuffer(rollouts, group_size=3).get_rank_batch(0, 1)


def _mismatch_kl(
    actor: nn.Module, ref: nn.Module, batch: DistributedBatch, temperature: float
) -> float:
    trainer = tr.ModelTrainer(
        actor=actor,
        reference=ref,
        optimizer=optim.SGD(actor.parameters(), lr=0.0),
        temperature=temperature,
    )
    return trainer.train_step(batch).mismatch_kl


@pytest.mark.asyncio
@pytest.mark.parametrize("temperature", [0.6, 1.0, 1.5])
async def test_trainer_matches_sampler_log_probs_at_same_temperature(
    temperature: float, actor_model: ToyModel, ref_model: ToyModel
) -> None:
    torch.manual_seed(0)
    batch = await _sample_batch(actor_model, temperature)

    matched = _mismatch_kl(actor_model, ref_model, batch, temperature)
    unmatched = _mismatch_kl(actor_model, ref_model, batch, temperature * 2)

    assert matched == pytest.approx(0.0, abs=1e-6)
    assert unmatched > 1e-4


@pytest.mark.parametrize(
    ("warmup_steps", "expected_factors"),
    [
        pytest.param(4, [0.25, 0.5, 0.75, 1.0, 1.0, 1.0], id="warmup_4"),
        pytest.param(1, [1.0, 1.0, 1.0], id="warmup_1"),
        pytest.param(0, [1.0, 1.0, 1.0], id="no_warmup"),
    ],
)
def test_linear_warmup_scheduler_ramps_then_holds(
    warmup_steps: int, expected_factors: list[float]
) -> None:
    base_lr = 0.1
    optimizer = optim.SGD([nn.Parameter(torch.zeros(1))], lr=base_lr)
    scheduler = tr.linear_warmup_scheduler(optimizer, warmup_steps)

    lrs = []
    for _ in expected_factors:
        lrs.append(optimizer.param_groups[0]["lr"])
        optimizer.step()
        scheduler.step()

    assert lrs == pytest.approx([base_lr * f for f in expected_factors])


def test_linear_warmup_scheduler_rejects_negative_steps() -> None:
    optimizer = optim.SGD([nn.Parameter(torch.zeros(1))], lr=0.1)

    with pytest.raises(ValueError, match="warmup_steps"):
        tr.linear_warmup_scheduler(optimizer, -1)


def test_train_step_steps_lr_scheduler_and_reports_lr(
    create_batch: Callable[[], DistributedBatch],
    actor_model: ToyModel,
    ref_model: ToyModel,
) -> None:
    base_lr = 0.1
    optimizer = optim.SGD(actor_model.parameters(), lr=base_lr)
    trainer = tr.ModelTrainer(
        actor=actor_model,
        reference=ref_model,
        optimizer=optimizer,
        lr_scheduler=tr.linear_warmup_scheduler(optimizer, warmup_steps=3),
    )

    lrs = [trainer.train_step(create_batch()).lr for _ in range(4)]

    assert lrs == pytest.approx([base_lr / 3, 2 * base_lr / 3, base_lr, base_lr])


def test_train_step_reports_constant_lr_without_scheduler(
    create_batch: Callable[[], DistributedBatch],
    actor_model: ToyModel,
    ref_model: ToyModel,
) -> None:
    trainer = tr.ModelTrainer(
        actor=actor_model,
        reference=ref_model,
        optimizer=optim.SGD(actor_model.parameters(), lr=0.1),
    )

    lrs = [trainer.train_step(create_batch()).lr for _ in range(2)]

    assert lrs == pytest.approx([0.1, 0.1])
