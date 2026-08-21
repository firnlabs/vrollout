"""Tests for worker.py."""

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


@pytest.fixture
def worker_(mock_llm_engine) -> ActorProxy[worker.RolloutWorker]:
    return worker.ActorRolloutWorker.remote(
        worker_id=1,
        config=worker.RolloutConfig(),
        llm_engine=mock_llm_engine,
    )


class TestRolloutWorker:
    """Tests for RolloutWorker class."""

    @pytest.mark.asyncio
    async def test_generate_batch(
        self, worker_: ActorProxy[worker.RolloutWorker], sampling_params
    ):
        prompts = ["Prompt 1", "Prompt 2"]

        results = await worker_.generate_batch.remote(prompts, sampling_params)

        assert len(results) == len(prompts)
