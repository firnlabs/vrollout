# vRollout: Distributed RL post-training engine

[![Python Version](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![Coverage](https://img.shields.io/badge/coverage-97.8%25-brightgreen.svg)](#testing)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

A distributed reinforcement learning (RL) rollout generation and training pipeline designed to decouple high-throughput autoregressive inference from policy training.

This repository is an educational reference implementation to study and benchmark inference-training topologies used in LLM post-training.

---

> [!NOTE]
>
> **Project Status**
>
> - **Implemented**: End-to-end pipeline with distributed completion token rollout using Ray, reward scoring using asynchronous queues, model training, weight broadcasting; tested using mock LLM engine (`MockAsyncLLMEngine`).
> - **In progress**: vLLM backend integration (`AsyncLLMEngine`) and multi-GPU FSDP2 weight gathering for Linux/CUDA cluster environments.

---

## Overview

Post-training foundation models with reinforcement learning (e.g. PPO, GRPO) introduces severe tension between memory-bound autoregressive token generation and compute-bound model training. Running both within a monolithic loop creates CUDA memory fragmentation and GPU idle bubbles while trainers wait for completion tokens to be available.

`vRollout` resolves this tension by decoupling token generation, reward evaluation, and policy updates into isolated asynchronous actors communicating across bounded queues and Ray's Plasma object store for zero-copy weight synchronization.

---

## Getting started

Prerequisites: Python 3.12+ and [uv](https://github.com/astral-sh/uv).

```bash
# Install dependencies
uv sync
```

### Running the distributed engine benchmark

The following benchmark script profiles PyTorch FSDP2 model sharding, distributed training iterations, and parameter weight broadcasting across ranks (tested on CPU via `gloo` and GPU via `nccl`).

```bash
# Optional: set UV_LINK_MODE=copy if your environment uses mounted scratch volumes (e.g. Kaggle)
uv run torchrun --standalone --nproc_per_node=2 --local-addr 127.0.0.1 \
  benchmarks/run_engine_benchmark.py
```

---

## Directory structure

```text
.
├── src/vrollout/
│   ├── async_queue.py          # Asynchronous queue for reward scoring
│   ├── buffer.py               # Distributed rollout buffer and batch collation
│   ├── engine.py               # Training engine
│   ├── gae.py                  # Fused Triton GAE kernel with PyTorch reference
│   ├── generation.py           # Distributed token generation using Ray
│   ├── pipeline.py             # Rollout generation and scoring pipeline
│   └── sync.py                 # Weight broadcasting (back to actor model)
├── benchmarks/
│   └── run_engine_benchmark.py # Engine profiling & benchmarking
├── tests/
│   ├── unit/                   # Unit tests
│   └── integration/            # Integration tests
├── pyproject.toml              # uv-based Python project configuration
```

---

## Testing

Run the complete test suite:

```bash
uv run pytest
```

---

## Contributing

Contributions, feedback, and bug reports are welcome. To contribute:

1. Fork the repository and create a feature branch (e.g. `git checkout -b feat/my-feature`).
2. Verify all tests pass and linting succeeds (`uv run pytest && uv run ruff check`).
3. Submit a pull request describing the rationale, changes made, and invariants preserved.

---

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.
