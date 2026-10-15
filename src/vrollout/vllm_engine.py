"""Implements a thin vLLM wrapper API for distributed rollout."""

from collections.abc import Mapping
from typing import AsyncIterator
import uuid

from jaxtyping import Shaped
from torch import Tensor
import vllm
from vllm import RequestOutput
from vllm import SamplingParams


class AsyncVLLMEngine:
    """Asynchronous vLLM engine for distributed rollout."""

    def __init__(self, model_name: str, tensor_parallel_size: int = 1) -> None:
        self._llm_engine = vllm.AsyncLLMEngine.from_engine_args(
            vllm.AsyncEngineArgs(
                model=model_name, tensor_parallel_size=tensor_parallel_size
            )
        )

    def generate(
        self,
        prompt: str,
        sampling_params: SamplingParams,
        request_id: str | None = None,
        *args,
        **kwargs,
    ) -> AsyncIterator[RequestOutput]:
        request_id = request_id or str(uuid.uuid4())

        generator = self._llm_engine.generate(
            prompt, sampling_params, request_id, *args, **kwargs
        )
        return generator

    async def update_weights(
        self, weights: Mapping[str, Shaped[Tensor, "..."]]
    ) -> None:
        if not weights:
            raise ValueError("`weights` cannot be empty")

        # convert to list since dict_items is not pickleable
        weights_list = list(weights.items())

        await self._llm_engine.collective_rpc(
            "reload_weights",
            kwargs={
                "weights_iterator": weights_list,
                "is_checkpoint_format": True,
            },
        )
