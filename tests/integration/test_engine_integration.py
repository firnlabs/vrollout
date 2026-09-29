"""Tests for engine.py."""

from collections.abc import Awaitable
from typing import Callable

import pytest
import ray
import torch
from torch import distributed as dist
from torch import nn
from torch import optim
from torch.distributed.device_mesh import DeviceMesh

from vrollout import engine
from vrollout import generation
from vrollout import pipeline
from vrollout import types
from vrollout.generation import RolloutManager
from vrollout.types import Rank
from vrollout.types import WorldSize


class LazyPipelineBuilder:
    """Builder that defers RolloutManager instantiation to inside the rank process."""

    def __init__(
        self,
        rollout_config: types.RolloutConfig,
        pipeline_config: pipeline.PipelineConfig,
        verifier: Callable[[types.RolloutTrajectory], Awaitable[float]],
    ) -> None:
        self.rollout_config = rollout_config
        self.pipeline_config = pipeline_config
        self.verifier = verifier

    def __call__(self) -> pipeline.RolloutPipeline:
        manager = generation.RolloutManager(self.rollout_config)
        return pipeline.RolloutPipeline(manager, self.verifier, self.pipeline_config)


@pytest.fixture
def pipeline_config() -> pipeline.PipelineConfig:
    return pipeline.PipelineConfig(chunk_size=4)


@pytest.fixture
def pipeline_builder(
    rollout_config, pipeline_config, mock_verifier
) -> Callable[[], pipeline.RolloutPipeline]:
    return LazyPipelineBuilder(rollout_config, pipeline_config, mock_verifier)


@pytest.fixture
def num_prompts() -> int:
    return 8


@pytest.fixture
def num_iterations() -> int:
    return 3


async def _test_train_iteration_worker(
    rank: Rank,
    world_size: WorldSize,
    mesh: DeviceMesh,
    pipeline_builder: Callable[[], pipeline.RolloutPipeline],
    prompts: list[str],
    sampling_params: types.SamplingParams,
    actor_model: nn.Module,
    ref_model: nn.Module,
    shard_model: Callable[[nn.Module, DeviceMesh], nn.Module],
    num_iterations: int,
):
    torch.manual_seed(42 + rank)

    if rank == 0:
        rollout_pipeline = pipeline_builder()
        scored_rollouts = await rollout_pipeline.generate_and_score(
            prompts, sampling_params
        )
        receivers = [rollout_pipeline.manager]
        container = [scored_rollouts]
    else:
        receivers = []
        container = [None]

    dist.broadcast_object_list(container, src=0)
    global_rollouts = container[0]

    ref_model.load_state_dict(actor_model.state_dict())
    sharded_actor = shard_model(actor_model, mesh)
    sharded_ref = shard_model(ref_model, mesh)
    optimizer = optim.AdamW(sharded_actor.parameters())
    dist_engine = engine.DistributedRLEngine(
        actor=sharded_actor,
        reference=sharded_ref,
        optimizer=optimizer,
        receivers=receivers,
        rank=rank,
        world_size=world_size,
    )

    losses = []
    for _ in range(num_iterations):
        metrics = dist_engine.train_iteration(global_rollouts)
        assert metrics.rank == rank
        losses.append(metrics.avg_loss)

    losses_t = torch.tensor(losses)
    assert not any(torch.isnan(losses_t))
    assert all(torch.isfinite(losses_t))

    if rank == 0:
        assert len(receivers) == 1
        receiver = receivers[0]
        assert isinstance(receiver, RolloutManager)
        statuses = ray.get([w.get_status.remote() for w in receiver.workers])  # ty: ignore[unresolved-attribute]
        for s in statuses:
            assert s.weights_version == num_iterations


class TestDistributedRLEngine:
    """Tests for DistributedRLEngine."""

    def test_train_iteration(
        self,
        pipeline_builder: Callable[[], pipeline.RolloutPipeline],
        prompts: list[str],
        sampling_params: types.SamplingParams,
        actor_model: nn.Module,
        ref_model: nn.Module,
        shard_model: Callable[[nn.Module, DeviceMesh], nn.Module],
        num_iterations: int,
        run_multiprocess,
    ):
        run_multiprocess(
            _test_train_iteration_worker,
            pipeline_builder,
            prompts,
            sampling_params,
            actor_model,
            ref_model,
            shard_model,
            num_iterations,
        )
