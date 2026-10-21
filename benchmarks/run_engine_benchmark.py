"""Profiling tool for the distributed RL engine."""

from dataclasses import dataclass
import os
import random
import time

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
import torch
from torch import Tensor
import torch.distributed as dist
from torch.distributed import fsdp
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.device_mesh import init_device_mesh
import torch.nn as nn

from vrollout import engine
from vrollout import sync
from vrollout.scorer import ScoredRollout
from vrollout.types import Rank
from vrollout.types import RolloutTrajectory
from vrollout.types import WorldSize

BYTES_PER_FP32 = 4
BYTES_TO_MB = 1024**2
BYTES_TO_GB = 1024**3
DEFAULT_VOCAB_SIZE = 1024
DEFAULT_MASTER_ADDR = "127.0.0.1"
DEFAULT_MASTER_PORT = "29500"
DEFAULT_TRACE_DIR = "output/traces"
DEFAULT_TRACE_FILE_PREFIX = "distributed_trace"
DEFAULT_SWEEP_REPS = 5
DEFAULT_SWEEP_WARMUP_REPS = 1
MS_PER_SECOND = 1000.0
US_PER_MS = 1000.0


@dataclass(frozen=True, slots=True)
class ModelConfig:
    dim: int
    layers: int
    vocab_size: int = DEFAULT_VOCAB_SIZE

    @property
    def label(self) -> str:
        return f"d={self.dim}, L={self.layers}"


@dataclass(frozen=True, slots=True)
class DistributedContext:
    rank: Rank
    world_size: WorldSize
    mesh: DeviceMesh
    device: torch.device
    device_type: str
    backend: str


@dataclass(frozen=True, slots=True)
class SyntheticRolloutConfig:
    min_prompt_len: int = 16
    max_prompt_len: int = 32
    min_completion_len: int = 24
    max_completion_len: int = 48
    min_reward: float = -1.0
    max_reward: float = 1.0
    dummy_logprob: float = -0.2
    num_workers: int = 2
    seed: int = 42


@dataclass(frozen=True, slots=True)
class ProfilerConfig:
    warmup_steps: int = 2
    profiled_steps: int = 5
    rollout_multiplier: int = 4
    min_rollouts: int = 8
    synthetic_config: SyntheticRolloutConfig = SyntheticRolloutConfig()


@dataclass(frozen=True, slots=True)
class BenchmarkReport:
    backend: str
    world_size: WorldSize
    avg_step_ms: float
    comm_ms: float
    gemm_ms: float
    comm_to_compute_ratio: float
    peak_vram_mb: float | None
    sweep_results: list[dict[str, float | int | str]]
    trace_path: str | None


DEFAULT_MODEL_CONFIGS: list[ModelConfig] = [
    ModelConfig(dim=256, layers=4),
    ModelConfig(dim=512, layers=8),
    ModelConfig(dim=768, layers=12),
]


class BenchmarkTransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int = 4):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads, batch_first=True
        )
        self.norm1 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Linear(4 * dim, dim),
        )
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x: Tensor) -> Tensor:
        attn_out, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class BenchmarkPolicyModel(nn.Module):
    def __init__(
        self,
        vocab_size: int = DEFAULT_VOCAB_SIZE,
        dim: int = 256,
        num_layers: int = 4,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, dim)
        self.layers = nn.ModuleList(
            [BenchmarkTransformerBlock(dim) for _ in range(num_layers)]
        )
        self.head = nn.Linear(dim, vocab_size, bias=False)

    def forward(
        self, input_ids: Tensor, attention_mask: Tensor | None = None
    ) -> Tensor:
        _ = attention_mask  # Accepted for API compatibility with ModelTrainer
        h = self.embedding(input_ids)
        for layer in self.layers:
            h = layer(h)
        return self.head(h)


def _shard_model(model: BenchmarkPolicyModel, mesh: DeviceMesh) -> BenchmarkPolicyModel:
    for layer in model.layers:
        fsdp.fully_shard(layer, mesh=mesh)
    fsdp.fully_shard(model, mesh=mesh)
    return model


def _generate_synthetic_rollouts(
    num_rollouts: int,
    vocab_size: int = DEFAULT_VOCAB_SIZE,
    config: SyntheticRolloutConfig | None = None,
) -> list[ScoredRollout]:
    cfg = config or SyntheticRolloutConfig()
    rng = random.Random(cfg.seed)
    rollouts = []
    for i in range(num_rollouts):
        p_len = rng.randint(cfg.min_prompt_len, cfg.max_prompt_len)
        c_len = rng.randint(cfg.min_completion_len, cfg.max_completion_len)
        prompt_tokens = [rng.randint(1, vocab_size - 1) for _ in range(p_len)]
        completion_tokens = [rng.randint(1, vocab_size - 1) for _ in range(c_len)]
        reward = round(rng.uniform(cfg.min_reward, cfg.max_reward), 2)

        trajectory = RolloutTrajectory(
            request_id=f"prof-req-{i}",
            prompt=f"prompt text {i}",
            prompt_token_ids=prompt_tokens,
            completion_token_ids=completion_tokens,
            logprobs=[cfg.dummy_logprob] * c_len,
            finish_reason="stop",
            text=f"completion text {i}",
            worker_id=i % cfg.num_workers,
        )
        rollouts.append(ScoredRollout(reward=reward, trajectory=trajectory))
    return rollouts


def _run_profiling(
    ctx: DistributedContext,
    model_cfg: ModelConfig,
    prof_cfg: ProfilerConfig | None = None,
) -> tuple[list[float], torch.profiler.profile]:
    cfg = prof_cfg or ProfilerConfig()

    actor = BenchmarkPolicyModel(
        vocab_size=model_cfg.vocab_size,
        dim=model_cfg.dim,
        num_layers=model_cfg.layers,
    ).to(ctx.device)

    reference = BenchmarkPolicyModel(
        vocab_size=model_cfg.vocab_size,
        dim=model_cfg.dim,
        num_layers=model_cfg.layers,
    ).to(ctx.device)
    reference.load_state_dict(actor.state_dict())
    reference.eval()

    actor = _shard_model(actor, ctx.mesh)
    reference = _shard_model(reference, ctx.mesh)

    optimizer = torch.optim.AdamW(actor.parameters(), lr=1e-4)
    dist_engine = engine.DistributedRLEngine(
        actor=actor,
        reference=reference,
        optimizer=optimizer,
        rank=ctx.rank,
        world_size=ctx.world_size,
        config=engine.RLEngineConfig(
            pad_token_id=0, beta=0.1, train_steps_per_iteration=1
        ),
    )

    batch_size = max(ctx.world_size * cfg.rollout_multiplier, cfg.min_rollouts)
    warmup_rollouts = _generate_synthetic_rollouts(
        batch_size,
        vocab_size=model_cfg.vocab_size,
        config=cfg.synthetic_config,
    )
    eval_rollouts = _generate_synthetic_rollouts(
        batch_size,
        vocab_size=model_cfg.vocab_size,
        config=cfg.synthetic_config,
    )

    for _ in range(cfg.warmup_steps):
        dist_engine.train_iteration(warmup_rollouts, device=ctx.device)

    activities = [torch.profiler.ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    step_times: list[float] = []
    with torch.profiler.profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    ) as prof:
        for _ in range(cfg.profiled_steps):
            t0 = time.perf_counter()
            dist_engine.train_iteration(eval_rollouts, device=ctx.device)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            step_times.append(time.perf_counter() - t0)

    return step_times, prof


def _extract_comm_and_compute_latencies(
    prof: torch.profiler.profile,
) -> tuple[float, float, float]:
    comm_time_us = 0.0
    gemm_time_us = 0.0
    for evt in prof.key_averages():
        k = evt.key.lower()
        duration = (
            evt.cuda_time_total
            if (torch.cuda.is_available() and hasattr(evt, "cuda_time_total"))
            else evt.cpu_time_total
        )
        if any(
            term in k
            for term in ["all_gather", "reduce_scatter", "all_reduce", "c10d", "nccl"]
        ):
            comm_time_us += duration
        elif any(term in k for term in ["gemm", "mm", "bmm", "addmm", "cutlass"]):
            gemm_time_us += duration

    comm_ms = comm_time_us / US_PER_MS
    gemm_ms = gemm_time_us / US_PER_MS
    ratio = (comm_ms / gemm_ms) if gemm_ms > 0 else 0.0
    return comm_ms, gemm_ms, ratio


def _benchmark_weight_broadcast_scaling(
    ctx: DistributedContext,
    model_configs: list[ModelConfig],
    num_reps: int = DEFAULT_SWEEP_REPS,
    warmup_reps: int = DEFAULT_SWEEP_WARMUP_REPS,
) -> list[dict[str, float | int | str]]:
    sweep_results: list[dict[str, float | int | str]] = []
    broadcaster = sync.WeightBroadcaster(rank=ctx.rank)

    for cfg in model_configs:
        model = BenchmarkPolicyModel(
            vocab_size=cfg.vocab_size, dim=cfg.dim, num_layers=cfg.layers
        ).to(ctx.device)
        model = _shard_model(model, ctx.mesh)

        param_count = sum(p.numel() for p in model.parameters())
        param_bytes = param_count * BYTES_PER_FP32

        for _ in range(warmup_reps):
            broadcaster.extract_and_broadcast(model)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        if dist.is_initialized():
            dist.barrier()

        sync_latencies: list[float] = []
        for _ in range(num_reps):
            t_sync_start = time.perf_counter()
            broadcaster.extract_and_broadcast(model)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            sync_latencies.append(time.perf_counter() - t_sync_start)

        avg_sync_ms = (sum(sync_latencies) / len(sync_latencies)) * MS_PER_SECOND
        eff_bw_gbs = (param_bytes / (avg_sync_ms / MS_PER_SECOND)) / BYTES_TO_GB
        sweep_results.append(
            {
                "config": cfg.label,
                "params": param_count,
                "size_mb": param_bytes / BYTES_TO_MB,
                "sync_ms": avg_sync_ms,
                "bw_gbs": eff_bw_gbs,
            }
        )

    return sweep_results


def _export_chrome_trace(
    prof: torch.profiler.profile,
    *,
    output_dir: str,
    device_type: str,
    file_prefix: str = DEFAULT_TRACE_FILE_PREFIX,
) -> str:
    os.makedirs(output_dir, exist_ok=True)
    filename = f"{file_prefix}_{device_type}.json"
    trace_path = os.path.join(output_dir, filename)
    prof.export_chrome_trace(trace_path)
    return trace_path


def _print_benchmark_results(report: BenchmarkReport) -> None:
    console = Console()

    vram_info = (
        f"{report.peak_vram_mb:,.2f} MB (Peak per rank)"
        if report.peak_vram_mb is not None
        else f"CPU ({report.backend.upper()}) - CUDA VRAM tracking inactive"
    )

    summary_table = Table.grid(padding=(0, 2))
    summary_table.add_column(style="bold cyan")
    summary_table.add_column()
    summary_table.add_row(
        "Backend / World Size:", f"{report.backend} | {report.world_size} ranks"
    )
    summary_table.add_row("Avg Step Latency:", f"{report.avg_step_ms:.2f} ms")
    summary_table.add_row(
        "Collective Comm:",
        f"{report.comm_ms:.2f} ms (All-Gather / Reduce-Scatter)",
    )
    summary_table.add_row("GEMM Compute:", f"{report.gemm_ms:.2f} ms (Matmul / Linear)")
    summary_table.add_row(
        "Comm-to-Compute Ratio:", f"{report.comm_to_compute_ratio:.2f}x"
    )
    summary_table.add_row("Memory Footprint:", vram_info)

    console.print()
    console.print(
        Panel(
            summary_table,
            title=(
                "[bold green]vrollout Distributed RL Engine Profiling &"
                " Scaling[/bold green]"
            ),
            border_style="bright_blue",
            expand=False,
        )
    )

    table = Table(
        title="Weight Broadcast Latency vs. Model Size",
        box=box.ROUNDED,
        header_style="bold magenta",
    )
    table.add_column("Model Config", justify="left")
    table.add_column("Params", justify="right")
    table.add_column("Size (MB)", justify="right")
    table.add_column("Sync Latency", justify="right")
    table.add_column("Effective BW", justify="right")

    for r in report.sweep_results:
        table.add_row(
            str(r["config"]),
            f"{int(r['params']):,d}",
            f"{float(r['size_mb']):.2f}",
            f"{float(r['sync_ms']):.2f} ms",
            f"{float(r['bw_gbs']):.2f} GB/s",
        )

    console.print(table)
    if report.trace_path is not None:
        console.print(
            "\n[green]✓[/green] Chrome trace exported to: "
            f"[cyan]{report.trace_path}[/cyan]\n"
        )


def _setup_env_variables() -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    if "RANK" not in os.environ and not dist.is_initialized():
        os.environ["RANK"] = "0"
        os.environ["LOCAL_RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"
        os.environ["MASTER_ADDR"] = DEFAULT_MASTER_ADDR
        os.environ["MASTER_PORT"] = DEFAULT_MASTER_PORT


def _init_distributed() -> DistributedContext:
    backend = "nccl" if torch.cuda.is_available() else "gloo"

    if torch.cuda.is_available():
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        device_type = "cuda"
    else:
        device = torch.device("cpu")
        device_type = "cpu"

    dist.init_process_group(
        backend=backend,
        device_id=device if torch.cuda.is_available() else None,
    )

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    mesh = init_device_mesh(device_type, (world_size,))
    return DistributedContext(
        rank=rank,
        world_size=world_size,
        mesh=mesh,
        device=device,
        device_type=device_type,
        backend=backend,
    )


def _run_benchmark_suite(
    ctx: DistributedContext,
    model_configs: list[ModelConfig],
    prof_cfg: ProfilerConfig | None = None,
) -> BenchmarkReport:
    primary_cfg = model_configs[0]
    step_times, prof = _run_profiling(ctx, primary_cfg, prof_cfg)
    comm_ms, gemm_ms, ratio = _extract_comm_and_compute_latencies(prof)
    if dist.is_initialized():
        dist.barrier()
    sweep_results = _benchmark_weight_broadcast_scaling(ctx, model_configs)

    avg_step_ms = (
        sum(step_times) / len(step_times) * MS_PER_SECOND if step_times else 0.0
    )
    peak_vram_mb = (
        torch.cuda.max_memory_allocated(ctx.device) / BYTES_TO_MB
        if torch.cuda.is_available()
        else None
    )
    trace_path = (
        _export_chrome_trace(
            prof, output_dir=DEFAULT_TRACE_DIR, device_type=ctx.device_type
        )
        if ctx.rank == 0
        else None
    )

    return BenchmarkReport(
        backend=ctx.backend,
        world_size=ctx.world_size,
        avg_step_ms=avg_step_ms,
        comm_ms=comm_ms,
        gemm_ms=gemm_ms,
        comm_to_compute_ratio=ratio,
        peak_vram_mb=peak_vram_mb,
        sweep_results=sweep_results,
        trace_path=trace_path,
    )


def main() -> None:
    # Run command:
    # export UV_LINK_MODE=copy  # optional: avoid hardlink warnings across volumes
    # uv run torchrun --standalone --nproc_per_node=2 --local-addr 127.0.0.1 \
    #   benchmarks/run_engine_benchmark.py
    _setup_env_variables()
    ctx = _init_distributed()
    report = _run_benchmark_suite(ctx, DEFAULT_MODEL_CONFIGS)
    if ctx.rank == 0:
        _print_benchmark_results(report)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
