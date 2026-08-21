"""Tests for worker.py."""

from dataclasses import dataclass

import pytest
import ray
from ray.actor import ActorProxy

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


@dataclass
class SamplingParams:
    temperature: float = 0.7
    max_tokens: int = 32
    logprobs: int = 1


@pytest.fixture
def sampling_params() -> SamplingParams:
    return SamplingParams()


@pytest.fixture
def worker_id() -> int:
    return 1


@pytest.fixture
def worker_(worker_id: int, mock_llm_engine) -> ActorProxy[worker.RolloutWorker]:
    return worker.ActorRolloutWorker.remote(
        worker_id=worker_id,
        config=worker.RolloutConfig(),
        llm_engine=mock_llm_engine,
    )


class TestRolloutWorker:
    """Tests for RolloutWorker class."""

    @pytest.mark.asyncio
    async def test_generate_batch(
        self,
        worker_: ActorProxy[worker.RolloutWorker],
        worker_id: int,
        sampling_params: SamplingParams,
    ):
        prompts = ["Prompt 1", "Prompt 2"]

        results = await worker_.generate_batch.remote(prompts, sampling_params)

        assert len(results) == len(prompts)
        for i, (r, p) in enumerate(zip(results, prompts, strict=True)):
            assert r.prompt == p
            assert r.request_id == f"request_{i}"
            assert r.finish_reason == "stop"
            assert r.worker_id == worker_id
