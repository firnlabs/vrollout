r"""Tests whether vLLM's prefix cache explains the in-training eval gap.

In E1 and E2, the step-200 in-training eval (weights reloaded in place into a vLLM
engine that had already run evals) scored below a separate eval of the saved
checkpoint. A weight reload leaves vLLM's prefix cache in place, so prompts whose
prefix is cached would reuse KV values computed by the old weights.

Each repetition evaluates the same greedy prompts under four conditions:

- `base`: a fresh engine on the base model, which fills the prefix cache.
- `inplace`: the checkpoint's weights reloaded in place into that engine (A).
- `reset`: the same engine after `reset_prefix_cache()` (A').
- `fresh`: a new engine that loads the checkpoint from disk (B).

If stale KV is the cause, `inplace` scores below `reset`, and `reset` matches
`fresh` up to the batch-invariance noise of repeated evals. Answers are saved as
`<output-dir>/rep-<r>/<condition>/eval-step-0.jsonl`, the layout of the training
script, so `scripts/report_stats.py outcomes` can compare any two of them.

Usage (one GPU):
    uv run python scripts/prefix_cache_eval_gap.py \
        --checkpoint /workspace/ckpt/e2-step-200 \
        --output-dir /workspace/prefix-cache/eval
"""

import os

os.environ.setdefault("RAY_ENABLE_UV_RUN_RUNTIME_ENV", "0")

import asyncio
from collections.abc import Sequence
import dataclasses
import itertools
import json
import pathlib
import time
from typing import Annotated
from typing import Any

import ray
import torch
import transformers  # type: ignore[import-untyped]
import typer

from vrollout import compare_evals
from vrollout import datasets
from vrollout import generation
from vrollout import metrics
from vrollout import pipeline
from vrollout import scorer
from vrollout import types
from vrollout import verifiers

app = typer.Typer(add_completion=False, pretty_exceptions_enable=False)

CONDITIONS = ("base", "inplace", "reset", "fresh")
PAIRS = (("inplace", "reset"), ("inplace", "fresh"), ("reset", "fresh"))


@dataclasses.dataclass
class _GreedyParams:
    temperature: float
    max_tokens: int
    logprobs: int | None = 1
    group_size: int = 1


def _eval_prompts(
    model_name: str, num_prompts: int, template: str
) -> list[tuple[str, str]]:
    """Builds the in-training eval prompts: the first GSM8K test problems."""
    data = datasets.load_gsm8k_dataset(
        num_samples=num_prompts,
        split="test",
        prompt_template=datasets.resolve_prompt_template(template),
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    return datasets.apply_chat_template(data, tokenizer)


def _checkpoint_weights(path: pathlib.Path) -> dict[str, torch.Tensor]:
    """Loads the checkpoint as the trainer sends it: fp32 named parameters on CPU."""
    model = transformers.AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32)
    return {n: p.detach().cpu() for n, p in model.named_parameters()}


def _wait_for_free_gpu(min_free_fraction: float, timeout_s: float = 180.0) -> None:
    """Blocks until a killed engine has released its GPU memory.

    Raises:
        TimeoutError: If the memory is not released within `timeout_s`.
    """
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        free, total = torch.cuda.mem_get_info()
        if free / total >= min_free_fraction:
            return
        time.sleep(2.0)
    raise TimeoutError("GPU memory was not released after killing the engine")


def _shutdown(manager: generation.RolloutManager, min_free_fraction: float) -> None:
    for w in manager.workers:
        ray.kill(w)
    _wait_for_free_gpu(min_free_fraction)


def _evaluate(
    rollout_pipeline: pipeline.RolloutPipeline,
    prompts: Sequence[str],
    prompt_to_ans: dict[str, str],
    max_tokens: int,
    out_dir: pathlib.Path,
) -> dict[str, float]:
    """Runs one greedy eval, saves its answers and returns its summary."""
    t0 = time.perf_counter()
    scored: list[scorer.ScoredRollout] = asyncio.run(
        rollout_pipeline.generate_and_score(
            prompts, _GreedyParams(temperature=0.0, max_tokens=max_tokens)
        )
    )
    if len(scored) != len(prompts):
        raise RuntimeError(f"Expected {len(prompts)} answers, got {len(scored)}")
    rows = metrics.eval_rows(scored, prompt_to_ans)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "eval-step-0.jsonl").open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row.to_dict()) + "\n")
    return {
        "accuracy_strict": verifiers.compute_accuracy(
            scored, prompt_to_ans, strict_format=True
        ),
        "accuracy_lenient": verifiers.compute_accuracy(
            scored, prompt_to_ans, strict_format=False
        ),
        **metrics.eval_length_metrics(rows),
        "eval_time_s": time.perf_counter() - t0,
    }


def _compare(base: pathlib.Path, candidate: pathlib.Path) -> dict[str, float]:
    """Paired comparison of two evals of the same prompts, scored strictly."""
    a, b = compare_evals.load_eval(base), compare_evals.load_eval(candidate)
    paired = compare_evals.summarize(compare_evals.pair_evals(a, b, strict=True))
    return {
        "difference": paired.delta,
        "ci95_half_width": paired.ci95_half_width,
        "gained": paired.gained,
        "lost": paired.lost,
        "mcnemar_p": paired.p_value,
        "text_differs": sum(a[k].completion != b[k].completion for k in a) / len(a),
    }


@app.command()
def main(
    checkpoint: Annotated[
        pathlib.Path,
        typer.Option(exists=True, file_okay=False, help="Saved step-N checkpoint"),
    ],
    output_dir: Annotated[pathlib.Path, typer.Option(help="Where evals are saved")],
    base_model: str = "Qwen/Qwen2.5-0.5B-Instruct",
    reps: Annotated[int, typer.Option(min=1)] = 3,
    num_eval_prompts: Annotated[int, typer.Option(min=1)] = 500,
    prompt_template: str = "qwen-boxed",
    max_tokens: int = 512,
    max_seq_len: int = 1024,
    gpu_memory_utilization: float = 0.85,
    seed: int = 0,
) -> None:
    """Evaluates base, in-place reload, reset prefix cache and fresh load."""
    eval_data = _eval_prompts(base_model, num_eval_prompts, prompt_template)
    prompts = [p for p, _ in eval_data]
    prompt_to_ans = dict(eval_data)
    verifier = verifiers.GSM8KVerifier(prompt_to_ans, strict_format=False)
    weights = _checkpoint_weights(checkpoint)
    # same engine settings as the training script's rollout worker
    config = types.RolloutConfig(
        model_name=base_model,
        backend="vllm",
        num_workers=1,
        gpus_per_worker=1.0,
        max_seq_len=max(max_seq_len, max_tokens * 2),
        gpu_memory_utilization=gpu_memory_utilization,
        enforce_eager=True,
        seed=seed,
    )
    min_free = gpu_memory_utilization + 0.05

    ray.init()
    summary: dict[str, Any] = {}
    for rep in range(reps):
        rep_dir = output_dir / f"rep-{rep}"
        results: dict[str, dict[str, float]] = {}

        manager = generation.RolloutManager(config)
        rollout_pipeline = pipeline.RolloutPipeline(manager, verifier)
        for condition in ("base", "inplace", "reset"):
            if condition == "inplace":
                manager.update_weights(weights)
            elif condition == "reset":
                manager.reset_prefix_cache()
            results[condition] = _evaluate(
                rollout_pipeline,
                prompts,
                prompt_to_ans,
                max_tokens,
                rep_dir / condition,
            )
            typer.echo(f"rep {rep} {condition}: {results[condition]}")
        _shutdown(manager, min_free)

        manager = generation.RolloutManager(
            dataclasses.replace(config, model_name=str(checkpoint))
        )
        rollout_pipeline = pipeline.RolloutPipeline(manager, verifier)
        results["fresh"] = _evaluate(
            rollout_pipeline, prompts, prompt_to_ans, max_tokens, rep_dir / "fresh"
        )
        typer.echo(f"rep {rep} fresh: {results['fresh']}")
        _shutdown(manager, min_free)

        pairs = {
            f"{a}_vs_{b}": _compare(
                rep_dir / a / "eval-step-0.jsonl", rep_dir / b / "eval-step-0.jsonl"
            )
            for a, b in PAIRS
        }
        summary[f"rep-{rep}"] = {"conditions": results, "pairs": pairs}
        for name, stats in pairs.items():
            typer.echo(f"rep {rep} {name}: {stats}")

    # the same condition across reps, to size the noise floor of a repeated eval
    summary["across_reps"] = {
        condition: {
            f"rep-{i}_vs_rep-{j}": _compare(
                output_dir / f"rep-{i}" / condition / "eval-step-0.jsonl",
                output_dir / f"rep-{j}" / condition / "eval-step-0.jsonl",
            )
            for i, j in itertools.combinations(range(reps), 2)
        }
        for condition in CONDITIONS
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    typer.echo(f"wrote {output_dir / 'summary.json'}")


if __name__ == "__main__":
    app()
