"""Integration test for pipeline.py."""

import pytest

from vrollout import pipeline
from vrollout import worker


@pytest.fixture
def rollout_config() -> worker.RolloutConfig:
    return worker.RolloutConfig()


@pytest.fixture
def manager(rollout_config, mock_llm_engine) -> worker.RolloutManager:
    return worker.RolloutManager(rollout_config, mock_llm_engine)


@pytest.fixture
def pipeline_(manager, mock_verifier) -> pipeline.RolloutPipeline:
    return pipeline.RolloutPipeline(manager, mock_verifier)


@pytest.fixture
def prompts() -> list[str]:
    return [f"Prompt {i}" for i in range(4)]


class TestRolloutPipeline:
    """Tests for RolloutPipeline class."""

    @pytest.mark.asyncio
    async def test_generate_and_score(self, pipeline_, prompts, sampling_params):
        results = await pipeline_.generate_and_score(prompts, sampling_params)

        assert len(results) == len(prompts)
