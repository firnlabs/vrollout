from collections.abc import AsyncGenerator
from dataclasses import dataclass
import itertools
from typing import Sequence

import pytest
import ray
from ray.actor import ActorProxy
import torch
from torch import Tensor

from vrollout import generation
from vrollout import types
from vrollout.testing import MockAsyncLLMEngine


@pytest.fixture(autouse=True)
def _ensure_ray(ray_session):
    """Ensure Ray is initialized for rollout worker and actor tests."""
    pass


@pytest.fixture
def worker_id() -> int:
    return 1


@pytest.fixture
def num_workers() -> int:
    return 3


@pytest.fixture
def rollout_config(model_name: str, num_workers: int) -> types.RolloutConfig:
    return types.RolloutConfig(model_name=model_name, num_workers=num_workers)


@pytest.fixture
def worker(
    worker_id: int,
    rollout_config: types.RolloutConfig,
    mock_llm_engine: MockAsyncLLMEngine,
) -> generation.RolloutWorker:
    return generation.RolloutWorker(
        worker_id=worker_id,
        config=rollout_config,
        llm_engine=mock_llm_engine,
    )


@pytest.fixture
def actor_worker(
    worker_id: int,
    rollout_config: types.RolloutConfig,
    mock_llm_engine: MockAsyncLLMEngine,
) -> ActorProxy[generation.RolloutWorker]:
    return generation.ActorRolloutWorker.remote(
        worker_id=worker_id,
        config=rollout_config,
        llm_engine=mock_llm_engine,
    )


@pytest.fixture
def weights() -> dict[str, Tensor]:
    return {f"weight_{i}": torch.rand(2, 3) for i in range(2)}


class TestRolloutWorker:
    @pytest.mark.asyncio
    async def test_generate_batch(
        self,
        worker: generation.RolloutWorker,
        worker_id: int,
        sampling_params: types.SamplingParams,
    ):
        prompts = ["Prompt 1", "Prompt 2"]

        results = await worker.generate_batch(prompts, sampling_params)

        assert len(results) == len(prompts)
        for i, (r, p) in enumerate(zip(results, prompts, strict=True)):
            assert r.prompt == p
            assert r.request_id == f"request_{i}"
            assert r.finish_reason == "stop"
            assert r.worker_id == worker_id

    @pytest.mark.asyncio
    async def test_actor_rollout_worker_remote(
        self,
        actor_worker: ActorProxy[generation.RolloutWorker],
        sampling_params: types.SamplingParams,
    ):
        prompts = ["Remote prompt"]
        results = await actor_worker.generate_batch.remote(prompts, sampling_params)  # ty: ignore[unresolved-attribute]
        assert len(results) == 1
        assert results[0].prompt == "Remote prompt"

    @pytest.mark.asyncio
    async def test_generate_single_empty_engine_raises_error(
        self,
        worker_id: int,
        rollout_config: types.RolloutConfig,
        sampling_params: types.SamplingParams,
    ):
        class EmptyEngine:
            async def generate(
                self,
                prompt: str,
                sampling_params: types.SamplingParams,
                request_id: str,
                *args,
                **kwargs,
            ) -> AsyncGenerator[types.RequestOutput, None]:
                if False:
                    yield  # pragma: no cover

        worker = generation.RolloutWorker(
            worker_id=worker_id,
            config=rollout_config,
            llm_engine=EmptyEngine(),
        )
        with pytest.raises(RuntimeError, match="Engine produced no output"):
            await worker._generate_single("prompt", sampling_params, "req_1")

    def test_update_weights(
        self,
        worker: generation.RolloutWorker,
        weights: dict[str, Tensor],
        mock_llm_engine,
    ):
        worker.update_weights(weights)

        assert worker._weights_version == 1

    def test_use_vllm_engine_from_config(self, mock_async_llm_engine):
        config = generation.RolloutConfig(model_name="vllm-test-model")
        worker = generation.RolloutWorker(1, config, llm_engine=None)

        from vrollout.vllm_engine import AsyncVLLMEngine

        assert isinstance(worker._llm_engine, AsyncVLLMEngine)


@pytest.fixture
def manager(
    rollout_config: types.RolloutConfig,
) -> generation.RolloutManager:
    return generation.RolloutManager(rollout_config)


class TestRolloutManager:
    @pytest.mark.asyncio
    async def test_generate(
        self,
        manager: generation.RolloutManager,
        sampling_params: types.SamplingParams,
        num_workers: int,
    ):
        num_prompts = 12
        chunk_size = 3
        prompts = [f"Prompt {i}" for i in range(num_prompts)]

        results = await manager.generate(prompts, sampling_params, chunk_size)

        assert len(results) == num_prompts
        worker_id_cycler = itertools.cycle(range(num_workers))
        for chunk in itertools.batched(results, chunk_size):
            exp_worker_id = next(worker_id_cycler)
            for r in chunk:
                assert r.worker_id == exp_worker_id

    @pytest.mark.asyncio
    async def test_generate_invalid_chunk_size(
        self,
        manager: generation.RolloutManager,
        sampling_params: types.SamplingParams,
    ):
        with pytest.raises(ValueError, match="chunk_size must be positive"):
            await manager.generate(["Prompt 1"], sampling_params, chunk_size=0)

    def test_update_weights(
        self, manager: generation.RolloutManager, weights: list[Tensor]
    ):
        manager.update_weights(weights)

        worker_statuses = ray.get(
            [w.get_status.remote() for w in manager.workers]  # ty: ignore[unresolved-attribute]
        )
        assert len(worker_statuses) == len(manager.workers)
        for s in worker_statuses:
            assert isinstance(s, generation.RolloutWorkerStatus)
            assert s.llm_engine_type == "MockAsyncLLMEngine"
            assert s.weights_version == 1


class TestParseVllmOutputToRollout:
    @dataclass
    class MockSampleLogprob:
        logprob: float
        rank: int | None = 1
        decoded_token: str | None = "token"

    @dataclass
    class MockCompletion:
        index: int
        text: str
        token_ids: Sequence[int]
        cumulative_logprob: float
        logprobs: (
            Sequence[
                dict[
                    int,
                    "TestParseVllmOutputToRollout.MockSampleLogprob | float",
                ]
            ]
            | None
        )
        finish_reason: str | None = "stop"

    @dataclass
    class MockRequestOutput:
        request_id: str
        prompt: str
        prompt_token_ids: Sequence[int]
        outputs: Sequence["TestParseVllmOutputToRollout.MockCompletion"]
        finished: bool = True

    def test_parse_with_sample_logprob_objects(self):
        sample_lp = self.MockSampleLogprob(logprob=-0.5)
        completion = self.MockCompletion(
            index=0,
            text="hello",
            token_ids=[42],
            cumulative_logprob=-0.5,
            logprobs=[{42: sample_lp}],
        )
        req_out = self.MockRequestOutput(
            request_id="req_test",
            prompt="hi",
            prompt_token_ids=[1],
            outputs=[completion],
        )

        traj = generation._parse_vllm_output_to_rollout(req_out, worker_id=2)
        assert traj.logprobs == [-0.5]
        assert traj.worker_id == 2
        assert traj.text == "hello"

    def test_parse_with_none_logprobs(self):
        completion = self.MockCompletion(
            index=0,
            text="hello",
            token_ids=[42, 43],
            cumulative_logprob=0.0,
            logprobs=None,
        )
        req_out = self.MockRequestOutput(
            request_id="req_test2",
            prompt="hi",
            prompt_token_ids=[1],
            outputs=[completion],
        )

        traj = generation._parse_vllm_output_to_rollout(req_out, worker_id=0)
        assert traj.logprobs == [0.0, 0.0]

    def test_parse_with_empty_outputs_raises_error(self):
        req_out = self.MockRequestOutput(
            request_id="req_empty",
            prompt="hi",
            prompt_token_ids=[1],
            outputs=[],
        )
        with pytest.raises(ValueError, match="RequestOutput contains no completion"):
            generation._parse_vllm_output_to_rollout(req_out, worker_id=0)
