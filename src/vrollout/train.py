import os

# prevent Ray from attempting to recreate virtual environment
os.environ.setdefault("RAY_ENABLE_UV_RUN_RUNTIME_ENV", "0")
os.environ.setdefault("MASTER_ADDR", "127.0.0.1")

import asyncio
from dataclasses import dataclass
from datetime import datetime
from datetime import timedelta
import time
from typing import Any

from rich.console import Console
from rich.table import Table
import torch
from torch import distributed as dist
from torch import nn
from torch import optim
from torch.distributed.device_mesh import init_device_mesh
import typer

from vrollout import generation
from vrollout import pipeline
from vrollout import placement
from vrollout import scorer
from vrollout import sharding
from vrollout import types
from vrollout.datasets import DEFAULT_GSM8K_PROMPT_TEMPLATE
from vrollout.datasets import load_gsm8k_dataset
from vrollout.datasets import load_synthetic_dataset
from vrollout.engine import DistributedRLEngine
from vrollout.engine import RLEngineConfig
from vrollout.testing import models
from vrollout.verifiers import check_format_compliance
from vrollout.verifiers import compute_format_compliance_rate
from vrollout.verifiers import extract_numerical_answer
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

        ref_dtype = (
            torch.bfloat16
            if (device.type == "cuda" and torch.cuda.is_bf16_supported())
            else (torch.float16 if device.type == "cuda" else torch.float32)
        )
        reference = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=ref_dtype
        ).to(device)

        # set actor model dtype to float32 so that optimizer parameters work with that
        # dtype, since autocast_dtype may downcast actor weights (e.g. bfloat16)
        actor = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=torch.float32
        ).to(device)

        if getattr(actor, "supports_gradient_checkpointing", False):
            actor.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        actor.train()

    for p in reference.parameters():
        p.requires_grad = False
    reference.eval()

    return actor, reference


def _init_distributed(device: torch.device, timeout: timedelta) -> None:
    """Initializes the default process group unless one already exists."""
    if dist.is_initialized():
        return
    use_cuda = device.type == "cuda"
    dist.init_process_group(
        backend="nccl" if use_cuda else "gloo",
        device_id=device if use_cuda else None,
        timeout=timeout,
    )


def _run_evaluation(
    rollout_pipeline: pipeline.RolloutPipeline,
    eval_prompts: list[str],
    max_tokens: int,
) -> tuple[float, list[scorer.ScoredRollout]]:
    """Runs deterministic greedy evaluation on held-out prompts."""
    if not eval_prompts:
        return 0.0, []

    eval_sampling_params = TrainSamplingParams(
        temperature=0.0,
        max_tokens=max_tokens,
        group_size=1,
    )
    scored_eval = asyncio.run(
        rollout_pipeline.generate_and_score(eval_prompts, eval_sampling_params)
    )
    if not scored_eval:
        return 0.0, []
    mean_reward = sum(sr.reward for sr in scored_eval) / len(scored_eval)
    return mean_reward, scored_eval


def _format_eval_cell(
    sr: scorer.ScoredRollout | None,
    max_chars: int = 256,
    target_format: str = "gsm8k",
) -> str:
    """Formats a scored evaluation rollout cell with reward badge and snippet."""
    if sr is None:
        return "-"
    badge = (
        f"[green]✓ ({sr.reward:.1f})[/green]"
        if sr.reward >= 0.5
        else f"[red]✗ ({sr.reward:.1f})[/red]"
    )
    compliant = check_format_compliance(sr.trajectory.text, target_format=target_format)
    fmt_badge = "[cyan][fmt:✓][/cyan]" if compliant else "[yellow][fmt:✗][/yellow]"
    pred = extract_numerical_answer(sr.trajectory.text) or "N/A"
    text = sr.trajectory.text.strip().replace("\n", " ")
    if max_chars > 3 and len(text) > max_chars:
        text = f"...{text[-(max_chars - 3) :]}"
    return f"{badge} {fmt_badge} [dim](pred: {pred})[/dim]\n{text}"


def _build_qualitative_table(
    eval_prompts: list[str],
    prompt_to_ans: dict[str, str],
    baseline_eval: list[scorer.ScoredRollout],
    final_eval: list[scorer.ScoredRollout],
    max_samples: int = 3,
    max_chars: int = 256,
    target_format: str = "gsm8k",
) -> Table:
    """Builds a side-by-side table comparing baseline and final eval completions."""
    sample_table = Table(
        "Prompt",
        "Ground Truth",
        "Before (Iter 0)",
        "After (Final)",
        title="Qualitative evaluation samples (Before vs. After)",
        show_lines=True,
    )
    base_by_prompt = {sr.trajectory.prompt: sr for sr in baseline_eval}
    final_by_prompt = {sr.trajectory.prompt: sr for sr in final_eval}

    def _sort_key(prompt: str) -> tuple[bool, float]:
        base_sr = base_by_prompt.get(prompt)
        final_sr = final_by_prompt.get(prompt)
        base_r = base_sr.reward if base_sr is not None else 0.0
        final_r = final_sr.reward if final_sr is not None else 0.0
        return (final_r > base_r, abs(final_r - base_r))

    sorted_prompts = sorted(eval_prompts, key=_sort_key, reverse=True)

    for prompt in sorted_prompts[:max_samples]:
        gt_raw = prompt_to_ans.get(prompt, "")
        gt_ans = extract_numerical_answer(gt_raw) or gt_raw
        sample_table.add_row(
            prompt,
            str(gt_ans),
            _format_eval_cell(
                base_by_prompt.get(prompt),
                max_chars=max_chars,
                target_format=target_format,
            ),
            _format_eval_cell(
                final_by_prompt.get(prompt),
                max_chars=max_chars,
                target_format=target_format,
            ),
        )
    return sample_table


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
    num_qualitative_samples: int = typer.Option(
        3, help="Number of held-out eval completions to display (0 to disable)"
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
    eval_max_tokens: int | None = typer.Option(
        None,
        "--eval-max-tokens",
        "--eval-completion-length",
        help="Max completion tokens for evaluation (defaults to max_tokens)",
    ),
    eval_max_chars: int = typer.Option(
        256,
        "--eval-max-chars",
        help="Max chars per eval cell in qualitative table (0 for no limit)",
    ),
    max_seq_len: int = typer.Option(
        512, help="Maximum total sequence length (prompt + completion) for vLLM"
    ),
    temperature: float = typer.Option(0.8, help="Sampling temperature"),
    enforce_eager: bool = typer.Option(
        True, help="Disable CUDA graph capture and Inductor compilation in vLLM"
    ),
    use_wandb: bool = typer.Option(False, "--wandb", help="Enable W&B tracking"),
    wandb_project: str = typer.Option("vrollout", help="W&B project name"),
    wandb_run_name: str | None = typer.Option(
        None,
        "--wandb-run-name",
        help=(
            "W&B run display name (defaults to "
            "{model}-{backend}-{dataset}-G{group_size}-{timestamp})"
        ),
    ),
    use_fsdp: bool | None = typer.Option(
        None,
        "--fsdp/--no-fsdp",
        help=(
            "Shard actor and reference models with FSDP2 across trainer ranks "
            "(defaults to on when world size > 1)"
        ),
    ),
    dist_timeout_minutes: int = typer.Option(
        60,
        min=1,
        help=(
            "Timeout for collectives; must exceed one rollout + eval cycle, since "
            "non-zero ranks wait in a collective while rank 0 generates"
        ),
    ),
    micro_batch_size: int = typer.Option(4, help="Micro batch size for training"),
    logit_chunk_size: int = typer.Option(128, help="Logit chunk size for training"),
    max_grad_norm: float | None = typer.Option(
        1.0, help="Maximum gradient norm for training"
    ),
    prompt_template: str = typer.Option(
        DEFAULT_GSM8K_PROMPT_TEMPLATE,
        "--prompt-template",
        help="Reasoning prompt template with {question} placeholder for GSM8K",
    ),
    format_target: str = typer.Option(
        "gsm8k",
        "--format-target",
        help=(
            "Target format style for compliance evaluation ('gsm8k', 'boxed', or 'any')"
        ),
    ),
    strict_format: bool = typer.Option(
        False,
        "--strict-format",
        help="Whether reward scoring strictly enforces answer format tags",
    ),
) -> None:
    """Executes distributed GRPO training iterations with Ray rollout generation."""
    console = Console()

    eval_completion_len = eval_max_tokens if eval_max_tokens is not None else max_tokens

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")

    _init_distributed(device, timedelta(minutes=dist_timeout_minutes))

    # prevent Ray and vLLM child processes from inheriting torchelastic agent store
    os.environ["TORCHELASTIC_USE_AGENT_STORE"] = "False"

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    shard_models = world_size > 1 if use_fsdp is None else use_fsdp

    if world_size > 1 and not shard_models:
        # without FSDP2 nothing synchronizes gradients, so ranks would silently train
        # divergent model copies
        dist.destroy_process_group()
        raise typer.BadParameter(
            "Multi-rank training requires --fsdp (gradients are not synchronized "
            "otherwise). Use --nproc_per_node=1 to train without FSDP2."
        )

    # validated on every rank so a bad layout fails all ranks instead of leaving the
    # others blocked in a collective
    rollout_gpu_ids: list[str] | None = None
    rollout_shares_trainer_gpus = False
    if device.type == "cuda":
        try:
            visible = placement.visible_gpu_ids(
                os.environ.get("CUDA_VISIBLE_DEVICES"), torch.cuda.device_count()
            )
            local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", world_size))
            rollout_gpu_ids, rollout_shares_trainer_gpus = (
                placement.select_rollout_gpus(visible, local_world_size)
            )
            placement.check_rollout_capacity(
                len(rollout_gpu_ids), num_rollout_workers, gpus_per_worker
            )
        except ValueError as e:
            dist.destroy_process_group()
            raise typer.BadParameter(str(e)) from e

    if rank == 0:
        console.rule("[bold cyan]vRollout Distributed GRPO Trainer[/bold cyan]")
        console.print(
            f"[dim]World Size:[/dim] {world_size} | "
            f"[dim]Backend:[/dim] {backend} | "
            f"[dim]Model:[/dim] {model_name} | "
            f"[dim]Group Size:[/dim] {group_size} | "
            f"[dim]FSDP2:[/dim] {'on' if shard_models else 'off'}"
        )
        if rollout_gpu_ids is not None:
            console.print(f"[dim]Rollout GPUs:[/dim] {','.join(rollout_gpu_ids)}")
            if rollout_shares_trainer_gpus and gpus_per_worker > 0:
                console.print(
                    "[yellow]Warning: no spare GPU for rollouts; vLLM shares GPUs "
                    "with trainer ranks.[/yellow]"
                )

    # initialize Weights & Biases on rank 0 if requested
    wandb_run = None
    if rank == 0 and use_wandb:
        try:
            import wandb as wb  # type: ignore[import-not-found]

            short_model = model_name.split("/")[-1]
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            resolved_run_name = (
                wandb_run_name
                if wandb_run_name is not None
                else f"{short_model}-{backend}-{dataset}-G{group_size}-{timestamp}"
            )

            wandb_run = wb.init(
                project=wandb_project,
                name=resolved_run_name,
                config={
                    "model_name": model_name,
                    "backend": backend,
                    "dataset": dataset,
                    "group_size": group_size,
                    "num_prompts": num_prompts,
                    "num_iterations": num_iterations,
                    "eval_every": eval_every,
                    "num_eval_prompts": num_eval_prompts,
                    "eval_max_tokens": eval_completion_len,
                    "lr": lr,
                    "beta": beta,
                    "clip_eps": clip_eps,
                    "fsdp": shard_models,
                    "world_size": world_size,
                },
            )
        except ImportError:
            console.print("[yellow]Warning: wandb is not installed. Skipping.[/yellow]")

    # setup models and optimizer
    actor, reference = _create_models(model_name, backend, device)
    if shard_models:
        # shard before creating the optimizer: fully_shard swaps parameters for DTensors
        mesh = init_device_mesh(device.type, (world_size,))
        actor = sharding.shard_model(actor, mesh)
        reference = sharding.shard_model(reference, mesh)
    optimizer = optim.AdamW(actor.parameters(), lr=lr)

    # Ray rollout pipeline initialization (rank 0 manages Ray actors)
    rollout_pipeline: pipeline.RolloutPipeline | None = None
    all_prompts: list[str] = []
    eval_prompts: list[str] = []
    prompt_to_ans: dict[str, str] = {}

    if rank == 0:
        os.environ["RAY_ENABLE_UV_RUN_RUNTIME_ENV"] = "0"
        placement.init_ray(rollout_gpu_ids)

        total_prompts_needed = num_prompts * num_iterations
        run_eval = eval_every > 0 and num_eval_prompts > 0
        effective_template = (
            None
            if prompt_template.strip().lower() in ("", "none", "raw")
            else prompt_template
        )
        if dataset == "gsm8k":
            data = load_gsm8k_dataset(
                num_samples=total_prompts_needed,
                split="train",
                prompt_template=effective_template,
            )
            eval_data = (
                load_gsm8k_dataset(
                    num_samples=num_eval_prompts,
                    split="test",
                    prompt_template=effective_template,
                )
                if run_eval
                else []
            )
        else:
            data = load_synthetic_dataset(num_samples=total_prompts_needed, seed=42)
            eval_data = (
                load_synthetic_dataset(num_samples=num_eval_prompts, seed=43)
                if run_eval
                else []
            )

        prompt_to_ans = {p: a for p, a in data + eval_data}
        verifier_fn = (
            synthetic_verifier
            if dataset == "synthetic" and backend == "mock"
            else GSM8KVerifier(prompt_to_ans, strict_format=strict_format)
        )

        all_prompts = [p for p, _ in data]
        eval_prompts = [p for p, _ in eval_data]

        effective_max_tokens = max(max_tokens, eval_completion_len)
        rollout_config = types.RolloutConfig(
            model_name=model_name,
            backend=backend,
            num_workers=num_rollout_workers,
            gpus_per_worker=gpus_per_worker,
            max_seq_len=max(max_seq_len, effective_max_tokens * 2),
            gpu_memory_utilization=gpu_memory_utilization,
            enforce_eager=enforce_eager,
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
        micro_batch_size=micro_batch_size,
        logit_chunk_size=logit_chunk_size,
        max_grad_norm=max_grad_norm,
        autocast_dtype=(
            torch.bfloat16
            if (device.type == "cuda" and torch.cuda.is_bf16_supported())
            else None
        ),
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
        "Format Compliance Rate",
        "Eval Reward",
        "Time (s)",
        title="GRPO training progress",
    )

    baseline_eval_rollouts: list[scorer.ScoredRollout] = []
    final_eval_rollouts: list[scorer.ScoredRollout] = []

    # baseline evaluation at iteration 0 (before training updates)
    if rank == 0 and eval_every > 0 and eval_prompts:
        assert rollout_pipeline is not None
        t_eval0 = time.perf_counter()
        eval_reward0, baseline_eval_rollouts = _run_evaluation(
            rollout_pipeline, eval_prompts, eval_completion_len
        )
        eval_format0 = compute_format_compliance_rate(
            baseline_eval_rollouts, target_format=format_target
        )
        final_eval_rollouts = baseline_eval_rollouts
        eval0_elapsed = time.perf_counter() - t_eval0
        table.add_row(
            "0",
            "-",
            "-",
            "-",
            "-",
            f"{eval_format0:.1%}",
            f"{eval_reward0:.3f}",
            f"{eval0_elapsed:.2f}s",
        )
        if wandb_run:
            wandb_run.log(
                {
                    "eval_reward_mean": eval_reward0,
                    "eval_format_compliance_rate": eval_format0,
                },
                step=0,
            )

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
        dist.broadcast_object_list(container, src=0, device=device)
        scored_rollouts = container[0]

        # train step on rank local batch & broadcast weights from rank 0
        metrics = dist_engine.train_iteration(scored_rollouts, device=device)

        # periodic evaluation on held-out prompts (rank 0)
        eval_reward: float | None = None
        eval_format_rate: float | None = None
        if rank == 0 and eval_every > 0 and (it + 1) % eval_every == 0 and eval_prompts:
            assert rollout_pipeline is not None
            eval_reward, final_eval_rollouts = _run_evaluation(
                rollout_pipeline, eval_prompts, eval_completion_len
            )
            eval_format_rate = compute_format_compliance_rate(
                final_eval_rollouts, target_format=format_target
            )

        elapsed = time.perf_counter() - t0

        # metric reporting on rank 0
        if rank == 0:
            rewards = [sr.reward for sr in scored_rollouts]
            mean_reward = sum(rewards) / len(rewards) if rewards else 0.0
            train_format_rate = compute_format_compliance_rate(
                scored_rollouts, target_format=format_target
            )
            eval_reward_str = f"{eval_reward:.3f}" if eval_reward is not None else "-"
            format_rate_str = f"{train_format_rate:.1%}"
            if eval_format_rate is not None:
                format_rate_str = f"{train_format_rate:.1%} / {eval_format_rate:.1%}"

            table.add_row(
                str(it + 1),
                f"{metrics.avg_loss:.4f}",
                f"{metrics.avg_kl:.4f}",
                f"{metrics.avg_clip_fraction:.2%}",
                f"{mean_reward:.3f}",
                format_rate_str,
                eval_reward_str,
                f"{elapsed:.2f}s",
            )

            if wandb_run:
                log_payload: dict[str, float] = {
                    "loss": metrics.avg_loss,
                    "kl": metrics.avg_kl,
                    "clip_fraction": metrics.avg_clip_fraction,
                    "reward_mean": mean_reward,
                    "format_compliance_rate": train_format_rate,
                    "step_time_s": elapsed,
                }
                if eval_reward is not None:
                    log_payload["eval_reward_mean"] = eval_reward
                if eval_format_rate is not None:
                    log_payload["eval_format_compliance_rate"] = eval_format_rate
                wandb_run.log(log_payload, step=it + 1)

    if rank == 0:
        console.print(table)
        if eval_every > 0 and eval_prompts and num_qualitative_samples > 0:
            qualitative_table = _build_qualitative_table(
                eval_prompts=eval_prompts,
                prompt_to_ans=prompt_to_ans,
                baseline_eval=baseline_eval_rollouts,
                final_eval=final_eval_rollouts,
                max_samples=num_qualitative_samples,
                max_chars=eval_max_chars,
                target_format=format_target,
            )
            console.print(qualitative_table)
        if wandb_run:
            wandb_run.finish()

    dist.destroy_process_group()


if __name__ == "__main__":
    app()
