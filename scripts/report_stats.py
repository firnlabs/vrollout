"""Summarizes saved GRPO runs for experiment reports.

Works on two kinds of run output:

- Training history: a W&B history export, a JSON object `{"history": [...]}` with
  one row per logged step. Step 0 holds only the pre-training eval; training
  metrics start at step 1. `fetch-history` downloads one.
- Eval answers: `eval-step-<N>.jsonl` files written by `vrollout.train
  --eval-output-dir`, one row per problem.

Commands:

- `fetch-history`: download a run's history from W&B.
- `step-times`: median step and phase times, and throughput.
- `curves`: windowed means of the training metrics.
- `runs`: one line per run with its peak NLL, peak truncation and eval accuracy.
- `outcomes`: per-problem outcomes between a base and a candidate eval.
- `flips`: how much two evals of the same model disagree.

Example:
    uv run scripts/report_stats.py outcomes \
        eval/e0-full/eval-step-0.jsonl eval/e1-full-step-200/eval-step-0.jsonl
"""

from collections.abc import Callable
from collections.abc import Iterable
from collections.abc import Sequence
from dataclasses import dataclass
import json
import pathlib
import statistics
from typing import Annotated
from typing import Any

import typer

from vrollout import compare_evals
from vrollout import verifiers
from vrollout.compare_evals import Outcome
from vrollout.compare_evals import PairedProblem
from vrollout.compare_evals import ScoredAnswer

STEP_PHASES = ("rollout_time_s", "train_time_s", "weight_sync_time_s")
CURVE_METRICS = (
    "reward_mean",
    "entropy_proxy",
    "kl",
    "truncation_rate",
    "format_compliance_rate",
    "response_length_mean",
    "clip_fraction",
)
OUTCOME_ORDER = (Outcome.BOTH_RIGHT, Outcome.GAINED, Outcome.LOST, Outcome.BOTH_WRONG)

app = typer.Typer(add_completion=False, no_args_is_help=True)

ExistingFile = Annotated[pathlib.Path, typer.Argument(exists=True, dir_okay=False)]

Row = dict[str, Any]


@dataclass(frozen=True)
class History:
    """A run's training history, split into training steps and eval results.

    Attributes:
        train: Rows of training steps (step >= 1), sorted by step.
        evals: `(step, strict accuracy)` for every step with an eval, including
            the pre-training eval at step 0.
    """

    train: list[Row]
    evals: list[tuple[int, float]]


def load_history(path: pathlib.Path) -> History:
    """Loads a W&B history export.

    Args:
        path: JSON file with a `history` list of per-step rows.

    Returns:
        The run's training steps and eval results.
    """
    rows = sorted(
        json.loads(path.read_text(encoding="utf-8"))["history"],
        key=lambda r: r["_step"],
    )
    return History(
        train=[r for r in rows if r["_step"] >= 1 and "step_time_s" in r],
        evals=[
            (r["_step"], r["eval_accuracy_strict"])
            for r in rows
            if r.get("eval_accuracy_strict") is not None
        ],
    )


def _window_mean(rows: Sequence[Row], key: str, first: int, last: int) -> float:
    """Returns the mean of `key` over steps `first..last` inclusive."""
    return statistics.mean(r[key] for r in rows if first <= r["_step"] <= last)


def _peak(rows: Sequence[Row], key: str) -> Row:
    """Returns the row with the largest value of `key`."""
    return max(rows, key=lambda r: r[key])


def _median(rows: Iterable[Row], value: Callable[[Row], float]) -> float:
    return statistics.median(value(r) for r in rows)


def _completion_tokens(row: Row) -> float:
    """Returns the completion tokens a step generated, from its rollout rate."""
    return row["rollout_tokens_per_s"] * row["rollout_time_s"]


def _calc_steps(row: Any) -> int:
    """Counts the calculator steps (`<<...>>`) in a GSM8K reference solution."""
    return (row.ground_truth_raw or "").count("<<")


def _failure_kind(answer: ScoredAnswer) -> str:
    """Classifies a wrong answer as truncated, unmarked or a wrong final answer."""
    if answer.finish_reason == "length":
        return "truncated"
    if answer.answer is None:
        return "unmarked"
    return "wrong answer"


def _accuracy(answers: Sequence[ScoredAnswer]) -> float:
    return sum(a.correct for a in answers) / len(answers)


@app.command()
def fetch_history(
    run_path: Annotated[str, typer.Argument(help="W&B run as entity/project/run_id")],
    output: Annotated[pathlib.Path, typer.Option("--output", "-o")],
) -> None:
    """Downloads a run's full per-step history from W&B to JSON."""
    import wandb  # noqa: PLC0415 (only this command needs network access)

    run = wandb.Api().run(run_path)
    payload = {
        "id": run.id,
        "name": run.name,
        "state": run.state,
        "url": run.url,
        "history": list(run.scan_history()),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload), encoding="utf-8")
    typer.echo(f"wrote {len(payload['history'])} rows to {output}")


@app.command()
def step_times(history: ExistingFile) -> None:
    """Prints median step and phase times, and token throughput.

    Medians of the phases need not add up to the median step time. Step time
    excludes the periodic eval and checkpoint saves.
    """
    rows = load_history(history).train
    step = _median(rows, lambda r: r["step_time_s"])
    total_hours = sum(r["step_time_s"] for r in rows) / 3600
    typer.echo(f"{len(rows)} steps, median {step:.1f} s, total {total_hours:.2f} h")
    for key in STEP_PHASES:
        phase = _median(rows, lambda r, key=key: r[key])
        typer.echo(f"  {key:<20} {phase:5.1f} s ({phase / step:.0%})")

    tokens = _median(rows, _completion_tokens)
    rollout_rate = _median(rows, lambda r: r["rollout_tokens_per_s"])
    train_rate = _median(rows, lambda r: _completion_tokens(r) / r["train_time_s"])
    typer.echo(f"completion tokens per step: {tokens:,.0f}")
    typer.echo(f"rollout: {rollout_rate:,.0f} tokens/s, training: {train_rate:,.0f}")


@app.command()
def curves(
    history: ExistingFile,
    window: Annotated[int, typer.Option(min=1, help="Steps per window")] = 25,
) -> None:
    """Prints windowed means of the training metrics and the peak of each."""
    rows = load_history(history).train
    starts = range(1, rows[-1]["_step"] + 1, window)
    typer.echo(
        "steps".ljust(24) + " ".join(f"{s:>3}-{s + window - 1:<3}" for s in starts)
    )
    for key in CURVE_METRICS:
        means = (_window_mean(rows, key, s, s + window - 1) for s in starts)
        typer.echo(key.ljust(24) + " ".join(f"{m:7.4f}" for m in means))

    typer.echo("peaks:")
    for key in CURVE_METRICS:
        peak = _peak(rows, key)
        typer.echo(f"  {key:<22} {peak[key]:.4f} at step {peak['_step']}")


@app.command()
def runs(histories: Annotated[list[pathlib.Path], typer.Argument(exists=True)]) -> None:
    """Prints one line per run: peak sampling NLL and truncation, eval accuracy.

    Useful for comparing a collapsed run with healthy ones. Eval accuracy is the
    first and last strict eval of the run.
    """
    typer.echo(
        f"{'run':<40}{'steps':>6}{'max NLL (step)':>18}{'max trunc (step)':>18}"
        f"{'eval first -> last':>26}"
    )
    for path in histories:
        run = load_history(path)
        nll = _peak(run.train, "entropy_proxy")
        trunc = _peak(run.train, "truncation_rate")
        (_, first_acc), (last_step, last_acc) = run.evals[0], run.evals[-1]
        typer.echo(
            f"{path.stem:<40}{run.train[-1]['_step']:>6}"
            f"{nll['entropy_proxy']:>11.2f} ({nll['_step']:>3})"
            f"{trunc['truncation_rate']:>11.0%} ({trunc['_step']:>3})"
            f"{first_acc:>11.3f} -> {last_acc:.3f} ({last_step})"
        )


@app.command()
def outcomes(base: ExistingFile, candidate: ExistingFile) -> None:
    """Prints per-problem outcomes between two evals of the same problems.

    Problems are grouped by whether each eval answered them correctly (strict
    scoring), with the difficulty of each group measured in reference
    calculator steps, and the reasons the wrong answers failed.
    """
    base_rows = compare_evals.load_eval(base)
    problems = compare_evals.pair_evals(
        base_rows, compare_evals.load_eval(candidate), strict=True
    )
    summary = compare_evals.summarize(problems)
    typer.echo(
        compare_evals.format_summary(
            summary, base_label="base", candidate_label="candidate"
        )
    )
    _print_side_stats(problems, base_rows)

    groups: dict[Outcome, list[PairedProblem]] = {o: [] for o in OUTCOME_ORDER}
    for p in problems:
        groups[p.outcome].append(p)

    typer.echo(f"\n{'outcome':<12}{'problems':>9}{'share':>8}{'calc steps':>16}")
    for outcome, group in groups.items():
        steps = [_calc_steps(base_rows[p.prompt]) for p in group]
        typer.echo(
            f"{outcome:<12}{len(group):>9}{len(group) / len(problems):>8.1%}"
            f"{statistics.mean(steps):>10.2f} ± {statistics.stdev(steps):.2f}"
        )

    typer.echo("\nwhy the wrong answer failed:")
    for outcome, side in (
        (Outcome.GAINED, "base"),
        (Outcome.LOST, "candidate"),
        (Outcome.BOTH_WRONG, "candidate"),
    ):
        answers = [getattr(p, side) for p in groups[outcome]]
        kinds = [_failure_kind(a) for a in answers]
        both = sum(a.finish_reason == "length" and a.answer is None for a in answers)
        counts = ", ".join(
            f"{k} {kinds.count(k)}" for k in ("wrong answer", "truncated", "unmarked")
        )
        typer.echo(
            f"  {outcome} ({side}): {counts} (truncated with no marked answer: {both})"
        )


def _print_side_stats(
    problems: Sequence[PairedProblem], base_rows: dict[str, Any]
) -> None:
    """Prints accuracy, truncation, format compliance and length of each eval."""
    references = [base_rows[p.prompt].ground_truth_raw for p in problems]
    for side in ("base", "candidate"):
        answers: list[ScoredAnswer] = [getattr(p, side) for p in problems]
        lenient = sum(
            verifiers.verify_math_answer(a.completion, ref, strict_format=False) == 1.0
            for a, ref in zip(answers, references, strict=True)
        )
        truncated = sum(a.finish_reason == "length" for a in answers)
        compliant = sum(
            verifiers.check_format_compliance(a.completion, "any") for a in answers
        )
        n = len(answers)
        typer.echo(
            f"{side}: strict {_accuracy(answers):.3f}, lenient {lenient / n:.3f}, "
            f"truncated {truncated / n:.1%}, format-compliant {compliant / n:.1%}, "
            f"mean length {statistics.mean(a.length for a in answers):.0f} tokens"
        )


@app.command()
def flips(first: ExistingFile, second: ExistingFile) -> None:
    """Counts flips between two evals of the same model to measure eval noise.

    A flip is a problem whose verdict differs between the two evals
    (Dutta et al., 2024, https://arxiv.org/abs/2407.09141). Positive flips are
    wrong in the first eval and right in the second; negative flips the reverse.

    Pass evals from different vLLM engine instances, e.g. the in-training eval
    and a separate eval of the saved checkpoint. The second eval may cover more
    problems (e.g. the full test set), but only the first eval's problems are
    compared.
    """
    a = compare_evals.load_eval(first)
    b = {
        prompt: row
        for prompt, row in compare_evals.load_eval(second).items()
        if prompt in a
    }
    problems = compare_evals.pair_evals(a, b, strict=True)
    n = len(problems)
    positive = sum(p.outcome == Outcome.GAINED for p in problems)
    negative = sum(p.outcome == Outcome.LOST for p in problems)
    flipped = positive + negative
    texts = sum(p.base.completion != p.candidate.completion for p in problems)
    typer.echo(
        f"{n} problems; strict accuracy {_accuracy([p.base for p in problems]):.3f}"
        f" vs {_accuracy([p.candidate for p in problems]):.3f}"
    )
    typer.echo(
        f"flips: {flipped} ({flipped / n:.1%}); "
        f"positive {positive}, negative {negative}"
    )
    typer.echo(f"answer text differs: {texts} ({texts / n:.0%})")


if __name__ == "__main__":
    app()
