from unittest.mock import MagicMock

import pytest
import torch
from vllm import CompletionOutput
from vllm import RequestOutput
from vllm import SamplingParams
from vllm.logprobs import Logprob

from vrollout.vllm_engine import AsyncVLLMEngine


@pytest.fixture
def sampling_params() -> SamplingParams:
    return SamplingParams(temperature=0.0)


@pytest.fixture
def async_vllm_engine() -> AsyncVLLMEngine:
    return AsyncVLLMEngine(model_name="vllm-test-model")


class TestAsyncVLLMEngine:
    @pytest.mark.asyncio
    async def test_generate(
        self,
        mock_async_llm_engine: MagicMock,
        async_vllm_engine: AsyncVLLMEngine,
        sampling_params: SamplingParams,
    ):
        prompt = "Hello world!"
        request_id = "1"
        mock_outputs = [
            RequestOutput(
                request_id=request_id,
                prompt=prompt,
                prompt_token_ids=[1, 2],
                prompt_logprobs=[{1: Logprob(0.1), 2: Logprob(0.2)}],
                finished=True,
                outputs=[
                    CompletionOutput(
                        index=0,
                        text="Goodbye world!",
                        token_ids=[4, 5],
                        cumulative_logprob=0.5,
                        logprobs=[{4: Logprob(0.2), 5: Logprob(0.3)}],
                    ),
                ],
            )
        ]
        mock_async_llm_engine.mock_outputs = mock_outputs

        generator = async_vllm_engine.generate(prompt, sampling_params, request_id)
        final_output = None
        async for output in generator:
            assert output is not None
            final_output = output

        assert final_output is not None
        assert isinstance(final_output, RequestOutput)

    @pytest.mark.asyncio
    async def test_update_weights(
        self,
        mock_async_llm_engine: MagicMock,
        async_vllm_engine: AsyncVLLMEngine,
    ):
        weights = {"w1": torch.tensor([1.0, 2.0]), "w2": torch.tensor([3.0, 4.0, 5.0])}

        await async_vllm_engine.update_weights(weights)

        mock_async_llm_engine.collective_rpc.assert_awaited_once_with(
            "reload_weights",
            kwargs={
                "weights_iterator": list(weights.items()),
                "is_checkpoint_format": True,
            },
        )

    @pytest.mark.asyncio
    async def test_update_weights_error_empty(
        self, mock_async_llm_engine: MagicMock, async_vllm_engine: AsyncVLLMEngine
    ):
        with pytest.raises(ValueError, match="`weights` cannot be empty"):
            await async_vllm_engine.update_weights({})

        mock_async_llm_engine.collective_rpc.assert_not_awaited()
