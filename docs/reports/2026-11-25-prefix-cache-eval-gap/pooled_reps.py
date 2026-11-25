r"""Pools the repetitions of `prefix_cache_eval_gap.py` into one comparison.

Each problem's strict correctness is averaged over the repetitions of a condition,
giving a value between 0 and 1 per problem. Two conditions are compared by the mean
of their per-problem differences, with a 95% normal confidence interval and a
two-sided sign-flip permutation test: if the conditions were interchangeable, each
difference would be as likely to be negative as positive.

Usage (from the repository root):
    uv run python docs/reports/2026-11-25-prefix-cache-eval-gap/pooled_reps.py \
        eval/prefix-cache/e1-step-200
"""

import pathlib
import random
import statistics
from typing import Annotated

import typer

from vrollout import compare_evals

CONDITIONS = ("base", "inplace", "reset", "fresh")
COMPARISONS = (("inplace", "reset"), ("inplace", "fresh"), ("reset", "fresh"))


def mean_correctness(output_dir: pathlib.Path, condition: str) -> dict[str, float]:
    """Returns each problem's strict correctness, averaged over the repetitions."""
    rep_dirs = sorted(output_dir.glob("rep-*"))
    correct: dict[str, int] = {}
    for rep_dir in rep_dirs:
        rows = compare_evals.load_eval(rep_dir / condition / "eval-step-0.jsonl")
        for prompt, row in rows.items():
            is_correct = compare_evals.score_answer(row, strict=True).correct
            correct[prompt] = correct.get(prompt, 0) + is_correct
    return {prompt: n / len(rep_dirs) for prompt, n in correct.items()}


def sign_flip_p(diffs: list[float], rounds: int = 20_000, seed: int = 0) -> float:
    """Returns the two-sided p-value of the summed differences under random signs."""
    rng = random.Random(seed)
    observed = abs(sum(diffs))
    extreme = 0
    for _ in range(rounds):
        flipped = sum(d if rng.random() < 0.5 else -d for d in diffs)
        extreme += abs(flipped) >= observed - 1e-12
    return (extreme + 1) / (rounds + 1)


def main(
    output_dir: Annotated[pathlib.Path, typer.Argument(exists=True, file_okay=False)],
) -> None:
    """Prints mean accuracy per condition and the pooled paired differences."""
    accuracy = {c: mean_correctness(output_dir, c) for c in CONDITIONS}
    for condition, per_problem in accuracy.items():
        mean_accuracy = statistics.mean(per_problem.values())
        typer.echo(f"{condition:<8} strict accuracy {mean_accuracy:.4f}")

    for before, after in COMPARISONS:
        diffs = [accuracy[after][p] - accuracy[before][p] for p in accuracy[before]]
        mean = statistics.mean(diffs)
        ci95 = 1.96 * statistics.stdev(diffs) / len(diffs) ** 0.5
        typer.echo(
            f"{after} - {before}: {mean * 100:+.2f} points "
            f"(95% CI ±{ci95 * 100:.2f}), p = {sign_flip_p(diffs):.4f}"
        )


if __name__ == "__main__":
    typer.run(main)
