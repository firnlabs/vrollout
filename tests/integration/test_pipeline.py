"""Integration test for pipeline.py."""

import collections
import time

import pytest

from vrollout import pipeline
from vrollout import scorer
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
def num_prompts() -> int:
    return 32


@pytest.fixture
def prompts(num_prompts) -> list[str]:
    return [f"Prompt {i}" for i in range(num_prompts)]


class TestRolloutPipeline:
    """Tests for RolloutPipeline class."""

    @pytest.mark.asyncio
    async def test_generate_and_score(
        self, pipeline_, num_prompts, prompts, sampling_params, rollout_config
    ):
        results: list[scorer.ScoredRollout] = await pipeline_.generate_and_score(
            prompts, sampling_params
        )

        assert len(results) == num_prompts
        worker_ids = collections.Counter([r.trajectory.worker_id for r in results])
        assert sum(worker_ids.values()) == num_prompts
        assert len(worker_ids) == rollout_config.num_workers

    @pytest.mark.asyncio
    async def test_generate_and_score_empty_prompts(self, pipeline_):
        results = await pipeline_.generate_and_score([])
        assert results == []

    @pytest.mark.asyncio
    async def test_generate_and_score_single_prompt(self, pipeline_, sampling_params):
        results = await pipeline_.generate_and_score(["one prompt"], sampling_params)
        assert len(results) == 1
        assert isinstance(results[0], scorer.ScoredRollout)

    @pytest.mark.asyncio
    async def test_scored_rollout_content(self, pipeline_, prompts, sampling_params):
        results = await pipeline_.generate_and_score(prompts, sampling_params)

        for r in results:
            assert isinstance(r, scorer.ScoredRollout)
            assert isinstance(r.trajectory, worker.RolloutTrajectory)
            # mock_verifier returns float(worker_id % 2)
            assert r.reward == float(r.trajectory.worker_id % 2)
            assert r.trajectory.finish_reason == "stop"
            assert len(r.trajectory.completion_token_ids) > 0
            assert len(r.trajectory.logprobs) == len(r.trajectory.completion_token_ids)

    @pytest.mark.asyncio
    async def test_generate_and_score_uneven_chunks(
        self, manager, mock_verifier, sampling_params
    ):
        # chunk_size=5 with 12 prompts -> chunks of [5, 5, 2]
        config = pipeline.PipelineConfig(chunk_size=5, scoring_batch_size=3)
        pipe = pipeline.RolloutPipeline(manager, mock_verifier, config)
        prompts = [f"Prompt {i}" for i in range(12)]

        results = await pipe.generate_and_score(prompts, sampling_params)

        assert len(results) == 12

    @pytest.mark.asyncio
    async def test_generate_and_score_fewer_prompts_than_chunk_size(
        self, manager, mock_verifier, sampling_params
    ):
        config = pipeline.PipelineConfig(chunk_size=16)
        pipe = pipeline.RolloutPipeline(manager, mock_verifier, config)
        prompts = [f"Prompt {i}" for i in range(3)]

        results = await pipe.generate_and_score(prompts, sampling_params)

        assert len(results) == 3

    @pytest.mark.asyncio
    async def test_scoring_overlaps_with_generation(self, manager, sampling_params):
        """Verify the scorer starts processing before all chunks finish generating."""
        score_timestamps: list[float] = []

        async def timestamping_verifier(
            rollout: worker.RolloutTrajectory,
        ) -> float:
            _ = rollout  # unused
            score_timestamps.append(time.monotonic())
            return 1.0

        config = pipeline.PipelineConfig(chunk_size=4, scoring_batch_size=4)
        pipe = pipeline.RolloutPipeline(manager, timestamping_verifier, config)
        prompts = [f"Prompt {i}" for i in range(16)]  # 4 chunks of 4

        await pipe.generate_and_score(prompts, sampling_params)

        # If pipelining works, the first scored items arrive well before
        # the last ones — timestamps are spread out, not clustered.
        assert len(score_timestamps) == 16
        first_chunk = score_timestamps[:4]
        last_chunk = score_timestamps[-4:]
        assert max(first_chunk) < min(last_chunk), (
            "Scoring did not overlap with generation — "
            "all items were scored at the same time"
        )
