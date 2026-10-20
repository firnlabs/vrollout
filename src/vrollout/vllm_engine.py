"""Implements a thin vLLM wrapper API for distributed rollout."""

from collections.abc import Mapping
import os
import tempfile
from typing import AsyncIterator
import uuid

from jaxtyping import Shaped
import safetensors.torch
from torch import Tensor
import vllm
from vllm import RequestOutput
from vllm import SamplingParams
from vllm.sampling_params import RequestOutputKind


class AsyncVLLMEngine:
    """Asynchronous vLLM engine for distributed rollout."""

    def __init__(
        self,
        model_name: str,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.60,
        max_model_len: int | None = None,
        enforce_eager: bool = True,
    ) -> None:
        self._llm_engine = vllm.AsyncLLMEngine.from_engine_args(
            vllm.AsyncEngineArgs(
                model=model_name,
                tensor_parallel_size=tensor_parallel_size,
                gpu_memory_utilization=gpu_memory_utilization,
                max_model_len=max_model_len,
                enforce_eager=enforce_eager,
                distributed_executor_backend="uni"
                if tensor_parallel_size == 1
                else "mp",
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

        if not isinstance(sampling_params, vllm.SamplingParams):
            group_size = getattr(
                sampling_params, "group_size", getattr(sampling_params, "n", 1)
            )
            sampling_params = vllm.SamplingParams(
                temperature=sampling_params.temperature,
                max_tokens=sampling_params.max_tokens,
                logprobs=sampling_params.logprobs,
                n=group_size,
                output_kind=RequestOutputKind.FINAL_ONLY,
            )

        generator = self._llm_engine.generate(
            prompt, sampling_params, request_id, *args, **kwargs
        )
        return generator

    async def update_weights(
        self, weights: Mapping[str, Shaped[Tensor, "..."]]
    ) -> None:
        if not weights:
            raise ValueError("`weights` cannot be empty")

        # stage safetensors in shared memory (/dev/shm) because vLLM V1's untyped
        # ZMQ MsgpackDecoder decodes raw torch.Tensor RPC arguments as plain lists.
        shm_dir = "/dev/shm" if os.path.isdir("/dev/shm") else None
        with tempfile.TemporaryDirectory(
            prefix="vrollout_weights_", dir=shm_dir
        ) as tmp_dir:
            seen_ptrs: set[int] = set()
            tensors: dict[str, Tensor] = {}
            for name, tensor in weights.items():
                t = tensor.detach().cpu().contiguous()
                # ensure each tensor has a unique data pointer to avoid aliasing issues
                # this can happen when weights share the same memory
                if t.data_ptr() in seen_ptrs:
                    t = t.clone()
                seen_ptrs.add(t.data_ptr())
                tensors[name] = t

            safetensors.torch.save_file(
                tensors, os.path.join(tmp_dir, "model.safetensors")
            )
            await self._llm_engine.collective_rpc(
                "reload_weights",
                kwargs={
                    "weights_path": tmp_dir,
                    "is_checkpoint_format": True,
                },
            )
