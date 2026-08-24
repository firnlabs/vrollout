"""Tests for worker.py."""

import itertools

import pytest
import ray
from ray.actor import ActorProxy

from tests.conftest import MockAsyncLLMEngine
from vrollout import worker


@pytest.fixture(scope="session", autouse=True)
def ray_session():
    """Start Ray once for the entire test session."""
    ray.init(
        ignore_reinit_error=True,
        include_dashboard=False,  # Speed up startup & reduce memory
    )
    yield
    ray.shutdown()


@pytest.fixture
def worker_id() -> int:
    return 1


@pytest.fixture
def num_workers() -> int:
    return 3


@pytest.fixture
def rollout_config(num_workers: int) -> worker.RolloutConfig:
    return worker.RolloutConfig(num_workers=num_workers)


@pytest.fixture
def worker_(
    worker_id: int,
    rollout_config: worker.RolloutConfig,
    mock_llm_engine: MockAsyncLLMEngine,
) -> worker.RolloutWorker:
    return worker.RolloutWorker(
        worker_id=worker_id,
        config=rollout_config,
        llm_engine=mock_llm_engine,
    )


@pytest.fixture
def actor_worker(
    worker_id: int,
    rollout_config: worker.RolloutConfig,
    mock_llm_engine: MockAsyncLLMEngine,
) -> ActorProxy[worker.RolloutWorker]:
    return worker.ActorRolloutWorker.remote(
        worker_id=worker_id,
        config=rollout_config,
        llm_engine=mock_llm_engine,
    )


class TestRolloutWorker:
    """Tests for RolloutWorker class."""

    @pytest.mark.asyncio
    async def test_generate_batch(
        self,
        worker_: worker.RolloutWorker,
        worker_id: int,
        sampling_params: worker.SamplingParams,
    ):
        prompts = ["Prompt 1", "Prompt 2"]

        results = await worker_.generate_batch(prompts, sampling_params)

        assert len(results) == len(prompts)
        for i, (r, p) in enumerate(zip(results, prompts, strict=True)):
            assert r.prompt == p
            assert r.request_id == f"request_{i}"
            assert r.finish_reason == "stop"
            assert r.worker_id == worker_id

    @pytest.mark.asyncio
    async def test_actor_rollout_worker_remote(
        self,
        actor_worker: ActorProxy[worker.RolloutWorker],
        sampling_params: worker.SamplingParams,
    ):
        prompts = ["Remote prompt"]
        results = await actor_worker.generate_batch.remote(prompts, sampling_params)  # ty: ignore[unresolved-attribute]
        assert len(results) == 1
        assert results[0].prompt == "Remote prompt"


@pytest.fixture
def manager(
    rollout_config: worker.RolloutConfig,
    mock_llm_engine: MockAsyncLLMEngine,
) -> worker.RolloutManager:
    return worker.RolloutManager(rollout_config, mock_llm_engine)


class TestRolloutManager:
    """Tests for RolloutManager class."""

    @pytest.mark.asyncio
    async def test_generate(
        self,
        manager: worker.RolloutManager,
        sampling_params: worker.SamplingParams,
        num_workers: int,
    ):
        num_prompts = 12
        chunk_size = 3
        prompts = ["Prompt {i}" for i in range(num_prompts)]

        results = await manager.generate(prompts, sampling_params, chunk_size)

        assert len(results) == num_prompts
        worker_id_cycler = itertools.cycle(range(num_workers))
        for chunk in itertools.batched(results, chunk_size):
            exp_worker_id = next(worker_id_cycler)
            for r in chunk:
                assert r.worker_id == exp_worker_id
