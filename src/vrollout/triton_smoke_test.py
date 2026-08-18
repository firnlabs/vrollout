"""OpenAI Triton Vector Addition Smoke Test Kernel.

This module validates that the cloud GPU environment has properly configured:
1. NVIDIA GPU drivers and CUDA runtime.
2. PyTorch with CUDA device support.
3. OpenAI Triton JIT compilation pipeline (@triton.jit) targeting PTX/SASS.
4. Memory boundary masking logic (offsets < n_elements).
"""

import sys

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import triton
    import triton.language as tl

    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False


if TRITON_AVAILABLE:

    @triton.jit
    def vector_add_kernel(  # noqa: DOC108
        x_ptr,
        y_ptr,
        out_ptr,
        n_elements,
        BLOCK_SIZE: tl.constexpr,
    ):
        """Triton kernel for elementwise vector addition: out = x + y.

        Args:
            x_ptr: Pointer to first input vector in GPU HBM.
            y_ptr: Pointer to second input vector in GPU HBM.
            out_ptr: Pointer to output vector in GPU HBM.
            n_elements: Total number of scalar elements.
            BLOCK_SIZE: Number of elements processed per thread block.
        """
        # Program ID indicates which block in the 1D grid is executing
        pid = tl.program_id(axis=0)

        # Calculate memory offset range for this block
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)

        # Boundary guard: mask prevents out-of-bounds memory access
        mask = offsets < n_elements

        # Coalesced loads from GPU Global Memory (HBM) into SRAM/registers
        x = tl.load(x_ptr + offsets, mask=mask)
        y = tl.load(y_ptr + offsets, mask=mask)

        # SIMD vector addition inside registers
        output = x + y

        # Coalesced store from registers back to GPU Global Memory
        tl.store(out_ptr + offsets, output, mask=mask)


def triton_vector_add(x, y, block_size: int = 1024):  # noqa: DOC108
    """Wrapper function to invoke the Triton vector addition kernel on PyTorch tensors.

    Args:
        x: Input tensor on CUDA device.
        y: Input tensor on CUDA device (matching shape and dtype).
        block_size: Tile block size for Triton grid execution (default: 1024).

    Returns:
        torch.Tensor: Elementwise sum (x + y) computed via Triton kernel.

    Raises:
        ImportError: If Triton is not installed.
        ValueError: If the inputs are not CUDA float32 tensors with matching shapes.
        TypeError: If the input tensors are not float32 tensors.
    """
    if not TRITON_AVAILABLE:
        raise ImportError("OpenAI Triton is not installed in the current environment.")

    if not x.is_cuda or not y.is_cuda:
        raise ValueError("Inputs to triton_vector_add must be on a CUDA device.")

    if x.shape != y.shape:
        raise ValueError(f"Shape mismatch: {x.shape} vs {y.shape}")

    if x.dtype != torch.float32 or y.dtype != torch.float32:
        raise TypeError("triton_vector_add currently expects float32 tensors.")

    out = torch.empty_like(x)
    n_elements = x.numel()

    # 1D grid calculation: ceil(n_elements / block_size) blocks launched
    def grid(meta):
        return (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    vector_add_kernel[grid](
        x,
        y,
        out,
        n_elements,
        BLOCK_SIZE=block_size,  # ty: ignore[invalid-argument-type]
    )
    return out


def run_smoke_test() -> bool:
    """Runs a complete environment check and Triton vector addition verification.

    Tests:
    1. CUDA availability and GPU device inspection.
    2. Power-of-two vector length (N=1024).
    3. Non-power-of-two vector length (N=9843) to strictly verify boundary masking.

    Returns:
        True if all parity checks pass within tolerance.
    """
    print("=" * 60)
    print("🔍 [vRollout] GPU & Triton Environment Smoke Test")
    print("=" * 60)

    # 1. Check PyTorch availability
    if not TORCH_AVAILABLE:
        print("❌ PyTorch is NOT installed!")
        print(
            "Please install PyTorch with CUDA, e.g."
            " pip install torch --index-url https://download.pytorch.org/whl/cu124"
        )
        return False

    # 2. Check CUDA availability
    if not torch.cuda.is_available():
        print("❌ CUDA is NOT available to PyTorch!")
        print("   Please verify that you have attached an NVIDIA GPU to your machine.")
        return False

    device_name = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    cuda_cap = torch.cuda.get_device_capability(0)
    print(f"✅ GPU Device Detected: {device_name}")
    print(f"   Compute Capability:   sm_{cuda_cap[0]}{cuda_cap[1]}")
    print(f"   Total VRAM:           {vram_gb:.2f} GB")
    print(f"   PyTorch Version:      {torch.__version__}")
    print(f"   CUDA Runtime:         {torch.version.cuda}")

    if not TRITON_AVAILABLE:
        print("❌ OpenAI Triton is not installed. Run: pip install triton")
        return False

    print(f"   Triton Version:       {triton.__version__}")
    print("-" * 60)

    # Test cases: (name, length, block_size)
    test_cases = [
        ("Power-of-Two (N=1024)", 1024, 1024),
        ("Non-Power-of-Two (N=9843, Boundary Masking)", 9843, 1024),
        ("Large Vector (N=1,048,576)", 1024 * 1024, 1024),
    ]

    all_passed = True
    for name, n_elems, block_sz in test_cases:
        torch.manual_seed(42)
        x = torch.randn(n_elems, device="cuda", dtype=torch.float32)
        y = torch.randn(n_elems, device="cuda", dtype=torch.float32)

        # Triton execution (JIT compilation triggers on first invocation)
        triton_out = triton_vector_add(x, y, block_size=block_sz)

        # PyTorch reference baseline
        expected_out = x + y

        # Parity check
        is_close = torch.allclose(triton_out, expected_out, atol=1e-5)
        max_diff = (triton_out - expected_out).abs().max().item()

        if is_close:
            print(f"✅ Test '{name}': PASSED (max discrepancy = {max_diff:.2e})")
        else:
            print(f"❌ Test '{name}': FAILED (max discrepancy = {max_diff:.2e})")
            all_passed = False

    print("=" * 60)
    if all_passed:
        print("🎉 SUCCESS: GPU, CUDA, PyTorch, and Triton JIT are working perfectly!")
    else:
        print("❌ FAILURE: Discrepancies detected in Triton kernel execution.")
    print("=" * 60)
    return all_passed


if __name__ == "__main__":
    success = run_smoke_test()
    sys.exit(0 if success else 1)
