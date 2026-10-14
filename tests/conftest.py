"""Pytest commons."""

import asyncio
from collections.abc import Awaitable
from dataclasses import dataclass
import inspect
import os
import random
import tempfile
from typing import Any
from typing import Callable
from typing import Concatenate
from typing import ParamSpec
from unittest.mock import MagicMock

import pytest
from pytest_mock import MockerFixture
import ray
from torch import distributed as dist
from torch import multiprocessing as mp
from torch import nn
from torch.distributed import device_mesh
from torch.distributed import fsdp
from torch.distributed.device_mesh import DeviceMesh

from vrollout import generation
from vrollout import pipeline
from vrollout import types
from vrollout.scorer import ScoredRollout
from vrollout.testing import create_model
from vrollout.testing import MockAsyncLLMEngine
from vrollout.testing import TEST_MODEL_SPECS
from vrollout.testing import ToyBlock
from vrollout.types import Rank
from vrollout.types import RolloutTrajectory
from vrollout.types import WorldSize

P = ParamSpec("P")
type MultiProcessFn[**P] = Callable[
    Concatenate[Rank, WorldSize, DeviceMesh, P], None | Awaitable[None]
]


@pytest.fixture
def model_name() -> str:
    return "test-model"


@pytest.fixture
def ref_model(model_name: str) -> nn.Module:
    model = create_model(model_name)
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    return model


@pytest.fixture
def actor_model(model_name: str) -> nn.Module:
    return create_model(model_name)


def _shard_model(model: nn.Module, mesh: DeviceMesh) -> nn.Module:
    for m in model.modules():
        if isinstance(m, ToyBlock):
            fsdp.fully_shard(m, mesh=mesh)
    fsdp.fully_shard(model, mesh=mesh)
    return model


@pytest.fixture
def vocab_size(model_name: str) -> int:
    return TEST_MODEL_SPECS[model_name]["vocab_size"]


@pytest.fixture
def mock_llm_engine(model_name, vocab_size) -> MockAsyncLLMEngine:
    return MockAsyncLLMEngine(model_name=model_name, vocab_size=vocab_size)


async def _mock_verifier(rollout: types.RolloutTrajectory) -> float:
    return float(rollout.worker_id % 2)


@pytest.fixture
def mock_verifier() -> Callable[[types.RolloutTrajectory], Awaitable[float]]:
    return _mock_verifier


@dataclass
class SamplingParams_:
    temperature: float = 0.7
    max_tokens: int = 32
    logprobs: int = 1


@pytest.fixture
def sampling_params() -> SamplingParams_:
    return SamplingParams_()


@pytest.fixture
def world_size() -> WorldSize:
    return 2


def _worker_wrapper(
    rank: Rank,
    world_size: WorldSize,
    init_method: str,
    fn: MultiProcessFn[Any],
    args: tuple[Any, ...],
) -> None:
    try:
        dist.init_process_group(
            backend="gloo",
            init_method=init_method,
            rank=rank,
            world_size=world_size,
        )
        mesh = device_mesh.init_device_mesh("cpu", (world_size,))

        if inspect.iscoroutinefunction(fn):
            asyncio.run(fn(rank, world_size, mesh, *args))
        else:
            fn(rank, world_size, mesh, *args)

    finally:
        if ray.is_initialized():
            ray.shutdown()
        if dist.is_initialized():
            dist.destroy_process_group()


@pytest.fixture
def run_multiprocess(
    world_size: WorldSize,
) -> Callable[Concatenate[MultiProcessFn[P], P], None]:
    def _run_multiprocess(
        fn: MultiProcessFn[P],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> None:
        """Multiprocessing harness."""
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        init_method = f"file://{temp_file.name}"
        try:
            mp.spawn(
                _worker_wrapper,
                args=(world_size, init_method, fn, args),
                nprocs=world_size,
                join=True,
            )
        finally:
            if os.path.exists(temp_file.name):
                os.remove(temp_file.name)

    return _run_multiprocess


@pytest.fixture
def num_prompts() -> int:
    return 32


@pytest.fixture
def prompts(num_prompts) -> list[str]:
    return [f"Prompt {i}" for i in range(num_prompts)]


@pytest.fixture
def rollout_config(model_name: str) -> types.RolloutConfig:
    return types.RolloutConfig(model_name=model_name)


@pytest.fixture
def rollout_manager(rollout_config) -> generation.RolloutManager:
    return generation.RolloutManager(rollout_config)


@pytest.fixture
def rollout_pipeline(rollout_manager, mock_verifier) -> pipeline.RolloutPipeline:
    return pipeline.RolloutPipeline(rollout_manager, mock_verifier)


@pytest.fixture
def shard_model() -> Callable[[nn.Module, DeviceMesh], nn.Module]:
    return _shard_model


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
def prompt_len() -> int:
    return 2


@pytest.fixture
def num_rollouts() -> int:
    return 8


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


@pytest.fixture(autouse=True)
def mock_async_llm_engine(mocker: MockerFixture) -> MagicMock:
    mock_engine_cls = mocker.patch("vllm.AsyncLLMEngine", autospec=True)

    mock_instance = mock_engine_cls.return_value
    mock_engine_cls.from_engine_args.return_value = mock_instance

    async def mock_generate(*args, **kwargs):
        for item in getattr(mock_instance, "mock_outputs", []):
            yield item

    mock_instance.generate = mock_generate

    return mock_instance
