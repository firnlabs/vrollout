#!/bin/bash
# ==============================================================================
# Setup script for vRollout on a GPU-enabled machine
# Stack: CUDA + PyTorch + OpenAI Triton + Ray + vLLM
# ==============================================================================

set -e

echo "============================================================"
echo "Verifying stack installation..."
echo "============================================================"

echo "Checking NVIDIA GPU status..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi
else
    echo "⚠️ WARNING: nvidia-smi not found. Ensure you are running on a GPU instance."
fi

uv run python -c "
import torch
print('✅ PyTorch Version:   ', torch.__version__)
print('   CUDA Available:    ', torch.cuda.is_available())
if torch.cuda.is_available():
    print('   CUDA Device Name:  ', torch.cuda.get_device_name(0))
    print('   CUDA Capability:   ', torch.cuda.get_device_capability(0))
"

uv run python -c "
import triton
print('✅ OpenAI Triton:     ', triton.__version__)
"

uv run python -c "
import ray
print('✅ Ray Version:        ', ray.__version__)
"

uv run python -c "
import vllm
print('✅ vLLM Version:       ', vllm.__version__)
"

echo "============================================================"
echo "Running Triton vector_add_kernel smoke test..."
echo "============================================================"
uv run python src/vrollout/triton_smoke_test.py

echo "============================================================"
echo "Setup and verification complete"
echo "============================================================"
