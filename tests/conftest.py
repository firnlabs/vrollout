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

import pytest
from torch import distributed as dist
from torch import multiprocessing as mp
from torch import nn
from torch import Tensor
from torch.distributed import device_mesh
from torch.distributed import fsdp
from torch.distributed.device_mesh import DeviceMesh

from vrollout import generation
from vrollout import pipeline
from vrollout import types
from vrollout.scorer import ScoredRollout
from vrollout.testing import MockAsyncLLMEngine
from vrollout.types import Rank
from vrollout.types import RolloutTrajectory
from vrollout.types import WorldSize

P = ParamSpec("P")
type MultiProcessFn[**P] = Callable[
    Concatenate[Rank, WorldSize, DeviceMesh, P], None | Awaitable[None]
]


@pytest.fixture
def vocab_size() -> int:
    return 32


@pytest.fixture
def mock_llm_engine(vocab_size) -> MockAsyncLLMEngine:
    return MockAsyncLLMEngine(vocab_size=vocab_size)


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
def rollout_config() -> types.RolloutConfig:
    return types.RolloutConfig()


@pytest.fixture
def rollout_manager(rollout_config, mock_llm_engine) -> generation.RolloutManager:
    return generation.RolloutManager(rollout_config, mock_llm_engine)


@pytest.fixture
def rollout_pipeline(rollout_manager, mock_verifier) -> pipeline.RolloutPipeline:
    return pipeline.RolloutPipeline(rollout_manager, mock_verifier)


class Block(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.ln = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, 4 * d_model)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(4 * d_model, d_model)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.fc2(self.act(self.fc1(self.ln(x))))


class ToyModel(nn.Module):
    def __init__(self, *, d_model: int, num_layers: int, vocab_size: int = 64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([Block(d_model) for _ in range(num_layers)])
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(
        self, input_ids: Tensor, attention_mask: Tensor | None = None
    ) -> Tensor:
        _ = attention_mask  # unused: kept for API compatiblity
        x = self.embedding(input_ids)
        for layer in self.layers:
            x = layer(x)
        return self.head(x)


@pytest.fixture
def num_layers() -> int:
    return 2


@pytest.fixture
def d_model() -> int:
    return 16


@pytest.fixture
def model_builder(
    d_model: int, num_layers: int, vocab_size: int
) -> Callable[[], ToyModel]:
    def _model_factory():
        return ToyModel(d_model=d_model, num_layers=num_layers, vocab_size=vocab_size)

    return _model_factory


@pytest.fixture
def ref_model(model_builder: Callable[[], ToyModel]) -> ToyModel:
    model = model_builder()
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    return model


@pytest.fixture
def actor_model(model_builder: Callable[[], ToyModel]) -> ToyModel:
    return model_builder()


def _shard_model(model: nn.Module, mesh: DeviceMesh) -> nn.Module:
    for m in model.modules():
        if isinstance(m, Block):
            fsdp.fully_shard(m, mesh=mesh)
    fsdp.fully_shard(model, mesh=mesh)
    return model


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
