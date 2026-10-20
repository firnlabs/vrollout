import os
from unittest.mock import MagicMock

import pytest
import safetensors.torch
import torch
from vllm import CompletionOutput
from vllm import RequestOutput
from vllm import SamplingParams
from vllm.logprobs import Logprob
from vllm.sampling_params import RequestOutputKind

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
        shared = torch.tensor([1.0, 2.0])
        weights = {
            "w1": shared,
            "w2": shared,  # tied weight sharing storage with w1
            "w3": torch.tensor([3.0, 4.0, 5.0]),
        }
        observed_loaded: dict[str, torch.Tensor] = {}
        observed_path: str | None = None

        async def _capture_rpc(method: str, *, kwargs: dict[str, object]):
            nonlocal observed_loaded, observed_path
            assert method == "reload_weights"
            assert kwargs["is_checkpoint_format"] is True
            weights_path = kwargs["weights_path"]
            assert isinstance(weights_path, str)
            observed_path = weights_path
            st_file = os.path.join(observed_path, "model.safetensors")
            assert os.path.isfile(st_file)
            observed_loaded = safetensors.torch.load_file(st_file)

        mock_async_llm_engine.collective_rpc.side_effect = _capture_rpc

        await async_vllm_engine.update_weights(weights)

        mock_async_llm_engine.collective_rpc.assert_awaited_once()
        assert set(observed_loaded.keys()) == {"w1", "w2", "w3"}
        assert torch.equal(observed_loaded["w1"], shared)
        assert torch.equal(observed_loaded["w2"], shared)
        assert torch.equal(observed_loaded["w3"], weights["w3"])
        assert observed_path is not None
        assert not os.path.exists(observed_path)

    @pytest.mark.asyncio
    async def test_update_weights_error_empty(
        self, mock_async_llm_engine: MagicMock, async_vllm_engine: AsyncVLLMEngine
    ):
        with pytest.raises(ValueError, match="`weights` cannot be empty"):
            await async_vllm_engine.update_weights({})

        mock_async_llm_engine.collective_rpc.assert_not_awaited()

    def test_init_forwards_engine_args(self, mocker):
        mock_args = mocker.patch("vllm.AsyncEngineArgs")
        mocker.patch("vllm.AsyncLLMEngine.from_engine_args")

        AsyncVLLMEngine(
            model_name="Qwen/Qwen2.5-0.5B-Instruct",
            tensor_parallel_size=2,
            gpu_memory_utilization=0.4,
            max_model_len=512,
        )

        mock_args.assert_called_once_with(
            model="Qwen/Qwen2.5-0.5B-Instruct",
            tensor_parallel_size=2,
            gpu_memory_utilization=0.4,
            max_model_len=512,
        )

    @pytest.mark.asyncio
    async def test_generate_converts_request_output_kind_final_only(
        self,
        mock_async_llm_engine: MagicMock,
        async_vllm_engine: AsyncVLLMEngine,
    ):
        class CustomParams:
            temperature = 0.7
            max_tokens = 64
            logprobs = 1
            group_size = 4

        captured_params: list[SamplingParams] = []

        async def _spy_generate(prompt, params, req_id, *args, **kwargs):
            captured_params.append(params)
            # creates an async generator, output is not needed
            if False:
                yield  # pragma: no cover

        mock_async_llm_engine.generate = _spy_generate
        gen = async_vllm_engine.generate("prompt", CustomParams(), "req_1")
        async for _ in gen:
            pass

        assert len(captured_params) == 1
        assert captured_params[0].n == 4
        assert captured_params[0].output_kind == RequestOutputKind.FINAL_ONLY
