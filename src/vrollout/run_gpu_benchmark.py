"""GPU Capability and Memory Bandwidth Benchmark Suite.

This script benchmarks the compute and memory subsystem of the active GPU:
1. Hardware capabilities, compute capability, and theoretical peak bandwidth.
2. PCIe Host-to-Device (H2D) and Device-to-Host (D2H) bandwidth (pinned vs unpinned).
3. GPU High Bandwidth Memory (HBM) Device-to-Device (D2D) copy throughput.
4. Effective memory bandwidth for memory-bound kernels (Triton vector_add vs PyTorch).
5. Compute throughput baseline (GEMM FP32 / FP16 / BF16 TFLOPS).
"""

from typing import Any
from typing import Dict

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from vrollout import triton_smoke_test

    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False


# Known peak memory bandwidth specifications for reference (GB/s)
KNOWN_GPU_SPECS = {
    # Reference: https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/tesla-t4/t4-tensor-core-datasheet-951643.pdf
    "Tesla T4": {
        "bandwidth_gbs": 300.0,
        "fp32_tflops": 8.1,  # single-precision
        "fp16_tflops": 65.0,  # mixed-precision
        "arch": "Turing",
    },
}


def get_gpu_capabilities(device_id: int = 0) -> Dict[str, Any]:
    """Inspect and report GPU architectural and memory hardware specifications."""
    props = torch.cuda.get_device_properties(device_id)
    device_name = torch.cuda.get_device_name(device_id)
    total_memory_gb = props.total_memory / (1024**3)
    major, minor = props.major, props.minor
    sm_count = getattr(props, "multi_processor_count", "N/A")

    # Check if we have reference theoretical specs
    ref_spec = None
    for name, spec in KNOWN_GPU_SPECS.items():
        if name.lower() in device_name.lower():
            ref_spec = spec
            break

    return {
        "device_name": device_name,
        "compute_capability": f"sm_{major}{minor}",
        "total_memory_gb": total_memory_gb,
        "sm_count": sm_count,
        "major": major,
        "minor": minor,
        "bf16_supported": torch.cuda.is_bf16_supported(),
        "ref_spec": ref_spec,
    }


def benchmark_cuda_event(fn, warmup: int = 10, reps: int = 50) -> float:  # noqa: DOC108
    """Benchmark a callable GPU function using high-precision CUDA Events.

    Args:
        fn: Callable GPU workload function to benchmark.
        warmup: Number of warmup iterations before recording events.
        reps: Number of timed repetitions.

    Returns:
        float: Average execution time in milliseconds.
    """
    # Warmup
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(reps):
        fn()
    end_event.record()

    torch.cuda.synchronize()
    elapsed_ms = start_event.elapsed_time(end_event) / reps
    return elapsed_ms


def benchmark_pcie_transfers(size_mb: int = 256) -> Dict[str, float]:
    """Measure PCIe Host-to-Device (H2D) and Device-to-Host (D2H) bandwidth."""
    num_bytes = size_mb * 1024 * 1024
    num_floats = num_bytes // 4

    # 1. Non-pinned memory
    host_unpinned = torch.randn(num_floats, dtype=torch.float32)
    device_tensor = torch.empty(num_floats, dtype=torch.float32, device="cuda")

    # H2D unpinned
    h2d_unpinned_ms = benchmark_cuda_event(
        lambda: device_tensor.copy_(host_unpinned, non_blocking=False),
        warmup=5,
        reps=20,
    )
    h2d_unpinned_gbs = (num_bytes / (h2d_unpinned_ms * 1e-3)) / 1e9

    # D2H unpinned
    d2h_unpinned_ms = benchmark_cuda_event(
        lambda: host_unpinned.copy_(device_tensor, non_blocking=False),
        warmup=5,
        reps=20,
    )
    d2h_unpinned_gbs = (num_bytes / (d2h_unpinned_ms * 1e-3)) / 1e9

    # 2. Pinned (page-locked) host memory
    host_pinned = torch.randn(num_floats, dtype=torch.float32).pin_memory()

    # H2D pinned
    h2d_pinned_ms = benchmark_cuda_event(
        lambda: device_tensor.copy_(host_pinned, non_blocking=True), warmup=5, reps=20
    )
    h2d_pinned_gbs = (num_bytes / (h2d_pinned_ms * 1e-3)) / 1e9

    # D2H pinned
    d2h_pinned_ms = benchmark_cuda_event(
        lambda: host_pinned.copy_(device_tensor, non_blocking=True), warmup=5, reps=20
    )
    d2h_pinned_gbs = (num_bytes / (d2h_pinned_ms * 1e-3)) / 1e9

    return {
        "h2d_unpinned_gbs": h2d_unpinned_gbs,
        "d2h_unpinned_gbs": d2h_unpinned_gbs,
        "h2d_pinned_gbs": h2d_pinned_gbs,
        "d2h_pinned_gbs": d2h_pinned_gbs,
    }


def benchmark_hbm_copy(size_mb: int = 512) -> float:
    """Measure GPU High Bandwidth Memory (HBM) Device-to-Device copy bandwidth."""
    num_bytes = size_mb * 1024 * 1024
    num_floats = num_bytes // 4

    src = torch.randn(num_floats, dtype=torch.float32, device="cuda")
    dst = torch.empty(num_floats, dtype=torch.float32, device="cuda")

    # D2D copy reads 1 tensor and writes 1 tensor = 2 * num_bytes transferred
    ms = benchmark_cuda_event(lambda: dst.copy_(src), warmup=10, reps=30)
    total_bytes = 2 * num_bytes
    bandwidth_gbs = (total_bytes / (ms * 1e-3)) / 1e9
    return bandwidth_gbs


def benchmark_vector_add_bandwidth(
    n_elements: int = 50_000_000,
) -> Dict[str, Any]:
    """Measure effective memory bandwidth of elementwise vector add (PyTorch vs Triton).

    Formula for vector add (z = x + y):
        Total bytes transferred:
            2 * Read (x, y) + 1 * Write (z) = 3 * N * sizeof(float32).
    """
    total_bytes = 3 * n_elements * 4  # 3 tensors * N * 4 bytes
    x = torch.randn(n_elements, dtype=torch.float32, device="cuda")
    y = torch.randn(n_elements, dtype=torch.float32, device="cuda")

    # 1. PyTorch native baseline
    torch_ms = benchmark_cuda_event(lambda: torch.add(x, y), warmup=10, reps=30)
    torch_gbs = (total_bytes / (torch_ms * 1e-3)) / 1e9

    # 2. Triton kernel (if available)
    triton_gbs = None
    triton_ms = None
    if TRITON_AVAILABLE:
        triton_ms = benchmark_cuda_event(
            lambda: triton_smoke_test.triton_vector_add(x, y, block_size=1024),
            warmup=10,
            reps=30,
        )
        triton_gbs = (total_bytes / (triton_ms * 1e-3)) / 1e9

    return {
        "n_elements": n_elements,
        "size_mb": (n_elements * 4) / (1024 * 1024),
        "torch_ms": torch_ms,
        "torch_gbs": torch_gbs,
        "triton_ms": triton_ms,
        "triton_gbs": triton_gbs,
    }


def benchmark_gemm_compute(
    m: int = 4096, n: int = 4096, k: int = 4096
) -> Dict[str, Any]:
    """Measure raw compute throughput (TFLOPS) for Matrix Multiplication (GEMM).

    FLOPS formula for M x K * K x N:
      Operations = 2 * M * N * K
    """
    ops = 2.0 * m * n * k
    results = {}

    # 1. FP32 GEMM
    a32 = torch.randn(m, k, dtype=torch.float32, device="cuda")
    b32 = torch.randn(k, n, dtype=torch.float32, device="cuda")
    ms32 = benchmark_cuda_event(lambda: torch.matmul(a32, b32), warmup=10, reps=30)
    results["fp32_tflops"] = (ops / (ms32 * 1e-3)) / 1e12

    # 2. FP16 Tensor Core GEMM
    a16 = torch.randn(m, k, dtype=torch.float16, device="cuda")
    b16 = torch.randn(k, n, dtype=torch.float16, device="cuda")
    ms16 = benchmark_cuda_event(lambda: torch.matmul(a16, b16), warmup=10, reps=30)
    results["fp16_tflops"] = (ops / (ms16 * 1e-3)) / 1e12

    # 3. BF16 Tensor Core GEMM (if supported)
    if torch.cuda.is_bf16_supported():
        abf16 = torch.randn(m, k, dtype=torch.bfloat16, device="cuda")
        bbf16 = torch.randn(k, n, dtype=torch.bfloat16, device="cuda")
        msbf16 = benchmark_cuda_event(
            lambda: torch.matmul(abf16, bbf16), warmup=10, reps=30
        )
        results["bf16_tflops"] = (ops / (msbf16 * 1e-3)) / 1e12
    else:
        results["bf16_tflops"] = None

    return results


def run_full_gpu_benchmark() -> None:
    """Executes the complete GPU capability and memory bandwidth benchmark."""
    print("=" * 60)
    print("GPU capability & memory bandwidth benchmark")
    print("=" * 60)

    if not TORCH_AVAILABLE:
        print("❌ Error: PyTorch is not installed.")
        return

    if not torch.cuda.is_available():
        print("❌ Error: CUDA device not detected.")
        return

    caps = get_gpu_capabilities(0)
    print("\n1. GPU Architectural Specifications:")
    print(f"   • Device Name:         {caps['device_name']}")
    print(f"   • Compute Capability:  {caps['compute_capability']}")
    print(f"   • Total VRAM:          {caps['total_memory_gb']:.2f} GB")
    print(f"   • Streaming Multiprocessors (SMs): {caps['sm_count']}")
    print(f"   • Bfloat16 Support:    {'Yes' if caps['bf16_supported'] else 'No'}")

    ref = caps.get("ref_spec")
    if ref:
        print(f"   • Reference architecture: {ref['arch']}")
        print(f"   • Reference memory bandwidth: {ref['bandwidth_gbs']:.1f} GB/s")
        print(
            "   • Reference mixed-precision fp16 FLOPS:"
            f" {ref['fp16_tflops']:.1f} TFLOPS"
        )

    print("\n2. PCIe Host <-> Device Bus Bandwidth (256 MB transfer):")
    pcie = benchmark_pcie_transfers(size_mb=256)
    print(
        f"   • Host-to-Device (H2D) [Unpinned]:  {pcie['h2d_unpinned_gbs']:>6.2f} GB/s"
    )
    print(
        f"   • Host-to-Device (H2D) [Pinned]:    {pcie['h2d_pinned_gbs']:>6.2f} GB/s"
        "  ⚡ (Page-Locked)"
    )
    print(
        f"   • Device-to-Host (D2H) [Unpinned]:  {pcie['d2h_unpinned_gbs']:>6.2f} GB/s"
    )
    print(
        f"   • Device-to-Host (D2H) [Pinned]:    {pcie['d2h_pinned_gbs']:>6.2f} GB/s"
        "  ⚡ (Page-Locked)"
    )

    print("\n3. GPU On-Chip Memory (HBM) Bandwidth:")
    d2d_gbs = benchmark_hbm_copy(size_mb=512)
    print(f"   • Raw D2D Copy Bandwidth:           {d2d_gbs:>6.2f} GB/s")
    if ref:
        hbm_eff = (d2d_gbs / ref["bandwidth_gbs"]) * 100
        print(f"   • Efficiency vs Theoretical Peak:   {hbm_eff:>6.1f} %")

    print(
        "\n⚡ 4. Memory-Bound Elementwise Vector Add Bandwidth "
        "(N = 50M floats, ~200MB):"
    )
    vec = benchmark_vector_add_bandwidth(n_elements=50_000_000)
    print(
        f"   • PyTorch (torch.add):              {vec['torch_gbs']:>6.2f} GB/s "
        f"(Latency: {vec['torch_ms']:.3f} ms)"
    )
    if vec["triton_gbs"]:
        print(
            "   • OpenAI Triton (vector_add):       "
            f"{vec['triton_gbs']:>6.2f} GB/s (Latency: {vec['triton_ms']:.3f} ms)"
        )
        speedup = vec["torch_ms"] / vec["triton_ms"]
        print(f"   • Triton Speedup:                   {speedup:>6.2f}x")
        if ref:
            triton_eff = (vec["triton_gbs"] / ref["bandwidth_gbs"]) * 100
            print(f"   • Triton % of Peak Theoretical:     {triton_eff:>6.1f} %")

    print("\n5. Compute Throughput Baseline (GEMM 4096 x 4096 x 4096):")
    gemm = benchmark_gemm_compute(4096, 4096, 4096)
    print(f"   • FP32 Matrix Multiplication:       {gemm['fp32_tflops']:>6.2f} TFLOPS")
    print(f"   • FP16 Tensor Core MatMul:          {gemm['fp16_tflops']:>6.2f} TFLOPS")
    if gemm["bf16_tflops"] is not None:
        print(
            f"   • BF16 Tensor Core MatMul:          {gemm['bf16_tflops']:>6.2f} TFLOPS"
        )


if __name__ == "__main__":
    run_full_gpu_benchmark()
