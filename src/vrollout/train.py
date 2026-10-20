import os

# prevent Ray from attempting to recreate virtual environment
os.environ.setdefault("RAY_ENABLE_UV_RUN_RUNTIME_ENV", "0")
os.environ.setdefault("MASTER_ADDR", "127.0.0.1")

import asyncio
from dataclasses import dataclass
import time
from typing import Any

from rich.console import Console
from rich.table import Table
import torch
from torch import distributed as dist
from torch import nn
from torch import optim
import typer

from vrollout import generation
from vrollout import pipeline
from vrollout import types
from vrollout.datasets import load_gsm8k_dataset
from vrollout.datasets import load_synthetic_dataset
from vrollout.engine import DistributedRLEngine
from vrollout.engine import RLEngineConfig
from vrollout.testing import models
from vrollout.verifiers import GSM8KVerifier
from vrollout.verifiers import synthetic_verifier

app = typer.Typer(pretty_exceptions_enable=False)


@dataclass
class TrainSamplingParams:
    temperature: float
    max_tokens: int
    logprobs: int | None = 1
    group_size: int = 1


def _create_models(
    model_name: str, backend: str, device: torch.device
) -> tuple[nn.Module, nn.Module]:
    """Creates actor and reference models."""
    if backend == "mock":
        actor = models.create_model(model_name).to(device)
        reference = models.create_model(model_name).to(device)
    else:
        from transformers import AutoModelForCausalLM  # type: ignore[import-untyped]

        dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
        actor = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype).to(device)
        reference = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype).to(
            device
        )

    for p in reference.parameters():
        p.requires_grad = False
    reference.eval()

    return actor, reference


def _run_evaluation(
    rollout_pipeline: pipeline.RolloutPipeline,
    eval_prompts: list[str],
    max_tokens: int,
) -> float:
    """Runs deterministic greedy evaluation on held-out prompts."""
    if not eval_prompts:
        return 0.0

    eval_sampling_params = TrainSamplingParams(
        temperature=0.0,
        max_tokens=max_tokens,
        group_size=1,
    )
    scored_eval = asyncio.run(
        rollout_pipeline.generate_and_score(eval_prompts, eval_sampling_params)
    )
    if not scored_eval:
        return 0.0
    return sum(sr.reward for sr in scored_eval) / len(scored_eval)


@app.command()
def main(
    model_name: str = typer.Option("test-model", help="Model name or checkpoint path"),
    backend: str = typer.Option("mock", help="Engine backend: 'mock' or 'vllm'"),
    dataset: str = typer.Option("synthetic", help="Dataset: 'synthetic' or 'gsm8k'"),
    group_size: int = typer.Option(
        4, "--group-size", "-G", help="Number of completions per prompt for GRPO"
    ),
    num_prompts: int = typer.Option(4, help="Number of prompts per iteration"),
    num_iterations: int = typer.Option(3, help="Number of training iterations"),
    eval_every: int = typer.Option(
        1, help="Run evaluation every N iterations (0 to disable)"
    ),
    num_eval_prompts: int = typer.Option(
        4, help="Number of held-out prompts for evaluation"
    ),
    lr: float = typer.Option(1e-5, help="Learning rate for AdamW"),
    beta: float = typer.Option(0.04, help="KL penalty coefficient (beta)"),
    clip_eps: float = typer.Option(0.2, help="Clipping epsilon for policy surrogate"),
    num_rollout_workers: int = typer.Option(2, help="Number of Ray rollout workers"),
    gpus_per_worker: float = typer.Option(
        0.0, help="GPUs allocated per rollout worker"
    ),
    gpu_memory_utilization: float = typer.Option(
        0.4, help="Fraction of GPU memory reserved for vLLM KV cache"
    ),
    max_tokens: int = typer.Option(32, help="Max completion tokens to generate"),
    max_seq_len: int = typer.Option(
        512, help="Maximum total sequence length (prompt + completion) for vLLM"
    ),
    temperature: float = typer.Option(0.8, help="Sampling temperature"),
    use_wandb: bool = typer.Option(False, "--wandb", help="Enable W&B tracking"),
    wandb_project: str = typer.Option("vrollout", help="W&B project name"),
) -> None:
    """Executes distributed GRPO training iterations with Ray rollout generation."""
    console = Console()

    # initialize distributed environment
    if not dist.is_initialized():
        backend_dist = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend_dist)

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    device = (
        torch.device(f"cuda:{rank}")
        if torch.cuda.is_available()
        else torch.device("cpu")
    )

    if rank == 0:
        console.rule("[bold cyan]vRollout Distributed GRPO Trainer[/bold cyan]")
        console.print(
            f"[dim]World Size:[/dim] {world_size} | "
            f"[dim]Backend:[/dim] {backend} | "
            f"[dim]Model:[/dim] {model_name} | "
            f"[dim]Group Size:[/dim] {group_size}"
        )

    # initialize Weights & Biases on rank 0 if requested
    wandb_run = None
    if rank == 0 and use_wandb:
        try:
            import wandb as wb  # type: ignore[import-not-found]

            wandb_run = wb.init(
                project=wandb_project,
                config={
                    "model_name": model_name,
                    "backend": backend,
                    "dataset": dataset,
                    "group_size": group_size,
                    "num_prompts": num_prompts,
                    "num_iterations": num_iterations,
                    "eval_every": eval_every,
                    "num_eval_prompts": num_eval_prompts,
                    "lr": lr,
                    "beta": beta,
                    "clip_eps": clip_eps,
                },
            )
        except ImportError:
            console.print("[yellow]Warning: wandb is not installed. Skipping.[/yellow]")

    # setup models and optimizer
    actor, reference = _create_models(model_name, backend, device)
    optimizer = optim.AdamW(actor.parameters(), lr=lr)

    # Ray rollout pipeline initialization (rank 0 manages Ray actors)
    rollout_pipeline: pipeline.RolloutPipeline | None = None
    all_prompts: list[str] = []
    eval_prompts: list[str] = []

    if rank == 0:
        import ray

        if not ray.is_initialized():
            os.environ["RAY_ENABLE_UV_RUN_RUNTIME_ENV"] = "0"
            ray.init(ignore_reinit_error=True, include_dashboard=False)

        total_prompts_needed = num_prompts * num_iterations
        run_eval = eval_every > 0 and num_eval_prompts > 0
        if dataset == "gsm8k":
            data = load_gsm8k_dataset(num_samples=total_prompts_needed, split="train")
            eval_data = (
                load_gsm8k_dataset(num_samples=num_eval_prompts, split="test")
                if run_eval
                else []
            )
            prompt_to_ans = {p: a for p, a in data + eval_data}
            verifier_fn = GSM8KVerifier(prompt_to_ans)
        else:
            data = load_synthetic_dataset(num_samples=total_prompts_needed, seed=42)
            eval_data = (
                load_synthetic_dataset(num_samples=num_eval_prompts, seed=43)
                if run_eval
                else []
            )
            verifier_fn = synthetic_verifier

        all_prompts = [p for p, _ in data]
        eval_prompts = [p for p, _ in eval_data]

        rollout_config = types.RolloutConfig(
            model_name=model_name,
            backend=backend,
            num_workers=num_rollout_workers,
            gpus_per_worker=gpus_per_worker,
            max_seq_len=max(max_seq_len, max_tokens * 2),
            gpu_memory_utilization=gpu_memory_utilization,
        )
        rollout_manager = generation.RolloutManager(rollout_config)
        rollout_pipeline = pipeline.RolloutPipeline(rollout_manager, verifier_fn)
        receivers = [rollout_manager]

    else:
        receivers = []

    sampling_params = TrainSamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
        group_size=group_size,
    )

    engine_config = RLEngineConfig(
        beta=beta,
        clip_eps=clip_eps,
        group_size=group_size,
    )

    dist_engine = DistributedRLEngine(
        actor=actor,
        reference=reference,
        optimizer=optimizer,
        receivers=receivers,
        rank=rank,
        world_size=world_size,
        config=engine_config,
    )

    # metric tracking table
    table = Table(
        "Iteration",
        "Loss",
        "KL Div",
        "Clip Frac",
        "Mean Reward",
        "Eval Reward",
        "Time (s)",
        title="GRPO training progress",
    )

    # baseline evaluation at iteration 0 (before training updates)
    if rank == 0 and eval_every > 0 and eval_prompts:
        assert rollout_pipeline is not None
        t_eval0 = time.perf_counter()
        eval_reward0 = _run_evaluation(rollout_pipeline, eval_prompts, max_tokens)
        eval0_elapsed = time.perf_counter() - t_eval0
        table.add_row(
            "0",
            "-",
            "-",
            "-",
            "-",
            f"{eval_reward0:.3f}",
            f"{eval0_elapsed:.2f}s",
        )
        if wandb_run:
            wandb_run.log({"eval_reward_mean": eval_reward0}, step=0)

    for it in range(num_iterations):
        t0 = time.perf_counter()

        # rollout & score on rank 0
        container: list[Any] = [None]
        if rank == 0:
            assert rollout_pipeline is not None
            iter_prompts = all_prompts[it * num_prompts : (it + 1) * num_prompts]
            scored = asyncio.run(
                rollout_pipeline.generate_and_score(iter_prompts, sampling_params)
            )
            container[0] = scored

        # broadcast scored rollouts to all ranks
        dist.broadcast_object_list(container, src=0)
        scored_rollouts = container[0]

        # train step on rank local batch & broadcast weights from rank 0
        metrics = dist_engine.train_iteration(scored_rollouts, device=device)

        # periodic evaluation on held-out prompts (rank 0)
        eval_reward: float | None = None
        if rank == 0 and eval_every > 0 and (it + 1) % eval_every == 0 and eval_prompts:
            assert rollout_pipeline is not None
            eval_reward = _run_evaluation(rollout_pipeline, eval_prompts, max_tokens)

        elapsed = time.perf_counter() - t0

        # metric reporting on rank 0
        if rank == 0:
            rewards = [sr.reward for sr in scored_rollouts]
            mean_reward = sum(rewards) / len(rewards) if rewards else 0.0
            eval_reward_str = f"{eval_reward:.3f}" if eval_reward is not None else "-"

            table.add_row(
                str(it + 1),
                f"{metrics.avg_loss:.4f}",
                f"{metrics.avg_kl:.4f}",
                f"{metrics.avg_clip_fraction:.2%}",
                f"{mean_reward:.3f}",
                eval_reward_str,
                f"{elapsed:.2f}s",
            )

            if wandb_run:
                log_payload: dict[str, float] = {
                    "loss": metrics.avg_loss,
                    "kl": metrics.avg_kl,
                    "clip_fraction": metrics.avg_clip_fraction,
                    "reward_mean": mean_reward,
                    "step_time_s": elapsed,
                }
                if eval_reward is not None:
                    log_payload["eval_reward_mean"] = eval_reward
                wandb_run.log(log_payload, step=it + 1)

    if rank == 0:
        console.print(table)
        if wandb_run:
            wandb_run.finish()

    dist.destroy_process_group()


if __name__ == "__main__":
    app()
