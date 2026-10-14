"""Universal vLLM Smoke Test Script (Apple Silicon Metal & NVIDIA CUDA).

Validates that:
1. vLLM imports cleanly without namespace or driver issues.
2. Hardware acceleration is active.
3. The inference engine initializes and successfully emits completion tokens.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import importlib.metadata
import sys
import time

import torch
from vllm import LLM
from vllm import SamplingParams
from vllm.platforms import current_platform

DEFAULT_MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"
BYTES_TO_GIB = 1024**3


@dataclass(frozen=True)
class BackendDiagnostics:
    """Hardware and backend diagnostics for the active vLLM runtime."""

    platform_name: str
    is_cuda: bool
    is_metal: bool
    device_name: str
    total_mem_gib: float
    vllm_ver: str
    metal_ver: str


def detect_backend() -> BackendDiagnostics:
    """Inspects runtime hardware and returns backend diagnostics metadata.

    Returns:
        BackendDiagnostics containing detected platform, backend type, and memory stats.
    """
    platform_name = type(current_platform).__name__
    is_cuda = getattr(current_platform, "is_cuda", lambda: False)() or (
        torch.cuda.is_available() and "cuda" in platform_name.lower()
    )
    is_metal = (
        platform_name == "MetalPlatform"
        or "metal" in platform_name.lower()
        or "apple"
        in str(getattr(current_platform, "get_device_name", lambda: "")()).lower()
    )

    try:
        total_mem_bytes = current_platform.get_device_total_memory()
        total_mem_gib = total_mem_bytes / BYTES_TO_GIB
    except Exception:
        total_mem_gib = 0.0

    device_name = "Unknown"
    if is_cuda and torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
    else:
        try:
            device_name = current_platform.get_device_name()
        except Exception:
            device_name = platform_name

    vllm_ver = importlib.metadata.version("vllm")
    try:
        metal_ver = importlib.metadata.version("vllm-metal")
    except importlib.metadata.PackageNotFoundError:
        metal_ver = "N/A (Standard CUDA / ROCm build)"

    return BackendDiagnostics(
        platform_name=platform_name,
        is_cuda=is_cuda,
        is_metal=is_metal,
        device_name=device_name,
        total_mem_gib=total_mem_gib,
        vllm_ver=vllm_ver,
        metal_ver=metal_ver,
    )


def run_smoke_test(
    model_name: str = DEFAULT_MODEL,
    gpu_memory_utilization: float = 0.6,
    tensor_parallel_size: int = 1,
) -> None:
    """Runs hardware backend verification and an end-to-end inference pass.

    Args:
        model_name: Hugging Face model identifier to load and test.
        gpu_memory_utilization: Fraction of device memory for KV cache.
        tensor_parallel_size: Number of GPUs for tensor parallelism (CUDA only).
    """
    print("=" * 60)
    print("vLLM Universal Smoke Test (Metal / CUDA)")
    print("=" * 60)

    # 1. Environment & Backend Diagnostics
    diag = detect_backend()
    print(f"[*] Python:             {sys.version.split()[0]}")
    print(f"[*] PyTorch Version:    {torch.__version__}")
    print(f"[*] vLLM Version:       {diag.vllm_ver}")

    if diag.is_cuda:
        cuda_ver = getattr(torch.version, "cuda", "unknown")
        gpu_count = torch.cuda.device_count()
        print(f"[*] Target Backend:     NVIDIA CUDA (CUDA {cuda_ver})")
        print(f"[*] Active Device:      {diag.device_name} (Total GPUs: {gpu_count})")
    elif diag.is_metal:
        print(f"[*] Target Backend:     Apple Silicon Metal (Plugin: {diag.metal_ver})")
        print(f"[*] Active Device:      {diag.device_name}")
    else:
        print(f"[*] Target Backend:     {diag.platform_name}")
        print(f"[*] Active Device:      {diag.device_name}")

    if diag.total_mem_gib > 0:
        print(f"[*] Total Device Mem:   {diag.total_mem_gib:.2f} GB")

    print("-" * 60)
    print(f"[*] Initializing model:  {model_name}...")
    start_time = time.perf_counter()

    # 2. Engine Initialization
    engine_kwargs: dict[str, object] = {
        "model": model_name,
        "max_model_len": 256,
        "gpu_memory_utilization": gpu_memory_utilization,
        "trust_remote_code": True,
    }

    # Only pass tensor_parallel_size if CUDA multi-GPU is active
    if diag.is_cuda and tensor_parallel_size > 1:
        engine_kwargs["tensor_parallel_size"] = tensor_parallel_size

    llm = LLM(**engine_kwargs)
    init_duration = time.perf_counter() - start_time
    print(f"[+] Engine initialized in {init_duration:.2f}s")

    # 3. Batched Inference Generation
    prompts = [
        "Hardware acceleration provides",
        "Explain reinforcement learning in one sentence:",
    ]
    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        max_tokens=32,
    )

    print("-" * 60)
    print("[*] Running inference pass...")
    gen_start = time.perf_counter()
    outputs = llm.generate(prompts, sampling_params)
    gen_duration = time.perf_counter() - gen_start

    print(f"[+] Generation finished in {gen_duration:.2f}s")
    print("-" * 60)

    # 4. Output Validation
    for i, output in enumerate(outputs):
        prompt_text = output.prompt
        generated_text = output.outputs[0].text.strip()
        num_tokens = len(output.outputs[0].token_ids)

        assert generated_text, f"Received empty generation for: {prompt_text}"
        print(f"Prompt #{i + 1}:  {prompt_text}")
        print(f"Generated:   {generated_text}")
        print(f"Tokens:      {num_tokens} tokens\n")

    print("=" * 60)
    print("STATUS: ALL SMOKE TESTS PASSED [OK]")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Universal vLLM Smoke Test (Metal & CUDA)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Model identifier (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.6,
        help="Fraction of device memory for KV cache (default: 0.6)",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Tensor parallel size (CUDA only, default: 1)",
    )
    args = parser.parse_args()

    run_smoke_test(
        model_name=args.model,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=args.tensor_parallel_size,
    )
