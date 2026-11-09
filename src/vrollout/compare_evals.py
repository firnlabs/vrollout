"""Side-by-side comparison of two eval answer files (`eval-step-<N>.jsonl`).

Pairs the two evals by prompt, re-scores both from the saved reference solution, prints
a paired summary and writes a filterable HTML page with the prompt, ground truth and
both answers for every problem.
"""

import collections
from collections.abc import Sequence
from dataclasses import dataclass
import enum
import functools
import json
import math
import pathlib
import statistics
from typing import Annotated

import jinja2
import typer

from vrollout import metrics
from vrollout import verifiers

_Z_95 = statistics.NormalDist().inv_cdf(0.975)
_CHAT_USER_START = "<|im_start|>user\n"
_CHAT_TURN_END = "<|im_end|>"


class Outcome(enum.StrEnum):
    """How a problem's correctness changed from the base to the candidate eval."""

    GAINED = "gained"
    LOST = "lost"
    BOTH_RIGHT = "both-right"
    BOTH_WRONG = "both-wrong"


@dataclass(frozen=True)
class ScoredAnswer:
    r"""One eval answer, re-scored under a single answer-format setting.

    Attributes:
        completion: Full completion text the model generated.
        answer: Final answer extracted under the chosen setting, or `None` if no
            answer was found (e.g. no `\boxed{}` in strict mode).
        correct: Whether the answer matches the reference solution.
        finish_reason: Why the sampler stopped; `"length"` means truncation.
        length: Completion length in tokens.
    """

    completion: str
    answer: str | None
    correct: bool
    finish_reason: str
    length: int


@dataclass(frozen=True)
class PairedProblem:
    """One problem with the base and candidate answers side by side.

    Attributes:
        prompt: Rendered prompt, identical in both evals; the pairing key.
        ground_truth: Final answer parsed from the reference solution, for display,
            or `None` if it could not be parsed.
        base: Answer from the base eval (e.g. E0).
        candidate: Answer from the candidate eval (e.g. E1).
    """

    prompt: str
    ground_truth: str | None
    base: ScoredAnswer
    candidate: ScoredAnswer

    @property
    def outcome(self) -> Outcome:
        """Returns how correctness changed from base to candidate."""
        if self.base.correct and self.candidate.correct:
            return Outcome.BOTH_RIGHT
        if self.base.correct:
            return Outcome.LOST
        if self.candidate.correct:
            return Outcome.GAINED
        return Outcome.BOTH_WRONG


@dataclass(frozen=True)
class PairedSummary:
    """Paired accuracy comparison between two evals on the same problems.

    Attributes:
        num_problems: Number of problems both evals answered.
        base_accuracy: Fraction of problems the base eval got right.
        candidate_accuracy: Fraction of problems the candidate eval got right.
        gained: Problems wrong in the base eval and right in the candidate.
        lost: Problems right in the base eval and wrong in the candidate.
        ci95_half_width: Half-width of the 95% Wald confidence interval on
            `delta`, as a fraction; the interval is `delta ± ci95_half_width`.
        p_value: Two-sided exact McNemar p-value for `gained` vs `lost`.
    """

    num_problems: int
    base_accuracy: float
    candidate_accuracy: float
    gained: int
    lost: int
    ci95_half_width: float
    p_value: float

    @property
    def delta(self) -> float:
        """Returns candidate minus base accuracy, e.g. 0.039 for +3.9 points.

        The difference is absolute, not relative, and positive when the candidate
        is better. Problems both evals agree on cancel out, so it equals
        `(gained - lost) / num_problems`.
        """
        return self.candidate_accuracy - self.base_accuracy


def load_eval(path: pathlib.Path) -> dict[str, metrics.EvalRow]:
    """Loads an eval JSONL file keyed by prompt, in file order.

    Args:
        path: File written by training's `--eval-output-dir`.

    Returns:
        Mapping from prompt to its eval row.

    Raises:
        ValueError: If a prompt appears more than once, which would make the
            pairing ambiguous.
    """
    rows: dict[str, metrics.EvalRow] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = metrics.EvalRow(**json.loads(line))
            if row.prompt in rows:
                raise ValueError(f"{path}: duplicate prompt {row.prompt[:80]!r}")
            rows[row.prompt] = row
    return rows


def score_answer(row: metrics.EvalRow, strict: bool) -> ScoredAnswer:
    """Re-scores a saved answer against its reference solution.

    The saved `reward` follows the run's `--strict-format` flag, so it is
    recomputed here to compare both evals under the same rule.

    Args:
        row: Saved eval row.
        strict: If True, only answers marked with a format tag count.

    Returns:
        The answer with its extracted value and correctness.

    Raises:
        ValueError: If the row has no reference solution to score against.
    """
    if row.ground_truth_raw is None:
        raise ValueError(f"no reference solution for prompt {row.prompt[:80]!r}")
    return ScoredAnswer(
        completion=row.completion,
        answer=verifiers.extract_numerical_answer(row.completion, strict=strict),
        correct=verifiers.verify_math_answer(
            row.completion, row.ground_truth_raw, strict_format=strict
        )
        == 1.0,
        finish_reason=row.finish_reason,
        length=row.length,
    )


def pair_evals(
    base: dict[str, metrics.EvalRow],
    candidate: dict[str, metrics.EvalRow],
    strict: bool,
) -> list[PairedProblem]:
    """Pairs two evals problem by problem, in the base eval's order.

    Args:
        base: Base eval rows keyed by prompt.
        candidate: Candidate eval rows keyed by prompt.
        strict: If True, only answers marked with a format tag count.

    Returns:
        One paired problem per prompt.

    Raises:
        ValueError: If the two evals cover different prompts, since a paired
            comparison needs the same problems on both sides.
    """
    if base.keys() != candidate.keys():
        only_base = len(base.keys() - candidate.keys())
        only_candidate = len(candidate.keys() - base.keys())
        raise ValueError(
            f"evals cover different prompts: {only_base} only in base, "
            f"{only_candidate} only in candidate"
        )
    return [
        PairedProblem(
            prompt=prompt,
            ground_truth=row.ground_truth,
            base=score_answer(row, strict),
            candidate=score_answer(candidate[prompt], strict),
        )
        for prompt, row in base.items()
    ]


def mcnemar_exact_p(gained: int, lost: int) -> float:
    """Returns the two-sided exact McNemar p-value for discordant pair counts.

    Under the null, each discordant problem is equally likely to flip either way,
    so min(gained, lost) ~ Binomial(gained + lost, 1/2).

    Args:
        gained: Problems wrong in the base eval and right in the candidate.
        lost: Problems right in the base eval and wrong in the candidate.

    Returns:
        The p-value, capped at 1.
    """
    n = gained + lost
    tail = sum(math.comb(n, i) for i in range(min(gained, lost) + 1))
    return min(1.0, 2 * tail / 2**n)


def summarize(problems: Sequence[PairedProblem]) -> PairedSummary:
    """Computes paired accuracy, a 95% CI on the difference and McNemar's p.

    Args:
        problems: Paired problems; must not be empty.

    Returns:
        The paired summary.

    Raises:
        ValueError: If `problems` is empty.
    """
    n = len(problems)
    if n == 0:
        raise ValueError("no problems to compare")
    outcomes = [p.outcome for p in problems]
    gained = outcomes.count(Outcome.GAINED)
    lost = outcomes.count(Outcome.LOST)
    base_right = lost + outcomes.count(Outcome.BOTH_RIGHT)
    # Var of the paired difference in proportions: ((b + c) - (b - c)^2 / n) / n^2
    variance = ((gained + lost) - (gained - lost) ** 2 / n) / n**2
    return PairedSummary(
        num_problems=n,
        base_accuracy=base_right / n,
        candidate_accuracy=(base_right + gained - lost) / n,
        gained=gained,
        lost=lost,
        ci95_half_width=_Z_95 * math.sqrt(variance),
        p_value=mcnemar_exact_p(gained, lost),
    )


def question_text(prompt: str) -> str:
    """Returns the user turn of a ChatML prompt, or the prompt unchanged.

    Args:
        prompt: Rendered prompt as saved in the eval file.

    Returns:
        The user's question without chat-template markup.
    """
    start = prompt.rfind(_CHAT_USER_START)
    if start == -1:
        return prompt.strip()
    body = prompt[start + len(_CHAT_USER_START) :]
    return body.split(_CHAT_TURN_END, 1)[0].strip()


def format_summary(
    summary: PairedSummary, base_label: str, candidate_label: str
) -> str:
    """Formats the paired summary as one line of plain text.

    Args:
        summary: Paired summary.
        base_label: Display name of the base eval.
        candidate_label: Display name of the candidate eval.

    Returns:
        A human-readable summary line.
    """
    return (
        f"{base_label} {summary.base_accuracy:.3f} -> {candidate_label} "
        f"{summary.candidate_accuracy:.3f} ({summary.delta * 100:+.1f} pts, "
        f"95% CI ±{summary.ci95_half_width * 100:.1f}), "
        f"+{summary.gained}/-{summary.lost} on {summary.num_problems} problems, "
        f"exact McNemar p={summary.p_value:.2g}"
    )


@functools.cache
def _template() -> jinja2.Template:
    env = jinja2.Environment(
        loader=jinja2.PackageLoader("vrollout", "templates"),
        autoescape=True,
        undefined=jinja2.StrictUndefined,
    )
    env.filters["question_text"] = question_text
    return env.get_template("compare_evals.html")


def render_html(
    problems: Sequence[PairedProblem],
    summary: PairedSummary,
    base_label: str,
    candidate_label: str,
    scoring: str,
) -> str:
    """Renders a self-contained, filterable side-by-side HTML page.

    The markup lives in `templates/compare_evals.html`; autoescaping covers every
    model-generated string, since completions can contain arbitrary HTML.

    Args:
        problems: Paired problems to show, one table row each.
        summary: Paired summary shown in the header.
        base_label: Display name of the base eval.
        candidate_label: Display name of the candidate eval.
        scoring: Name of the scoring rule, shown in the header.

    Returns:
        The HTML document.
    """
    counts = collections.Counter(p.outcome for p in problems)
    return _template().render(
        problems=problems,
        counts={o: counts[o] for o in Outcome},
        headline=format_summary(summary, base_label, candidate_label),
        base_label=base_label,
        candidate_label=candidate_label,
        scoring=scoring,
    )


app = typer.Typer(add_completion=False)


@app.command()
def main(
    base: Annotated[
        pathlib.Path,
        typer.Argument(exists=True, dir_okay=False, help="Base eval JSONL (e.g. E0)"),
    ],
    candidate: Annotated[
        pathlib.Path,
        typer.Argument(
            exists=True, dir_okay=False, help="Candidate eval JSONL (e.g. E1)"
        ),
    ],
    output: Annotated[
        pathlib.Path,
        typer.Option("--output", "-o", help="Where to write the HTML page"),
    ] = pathlib.Path("compare-evals.html"),
    base_label: Annotated[
        str, typer.Option(help="Display name of the base eval")
    ] = "base",
    candidate_label: Annotated[
        str, typer.Option(help="Display name of the candidate eval")
    ] = "candidate",
    lenient: Annotated[
        bool,
        typer.Option(
            "--lenient", help="Also count unmarked final answers, not only marked ones"
        ),
    ] = False,
) -> None:
    """Compares two eval answer files side by side."""
    strict = not lenient
    problems = pair_evals(load_eval(base), load_eval(candidate), strict=strict)
    summary = summarize(problems)
    page = render_html(
        problems,
        summary,
        base_label,
        candidate_label,
        scoring="strict" if strict else "lenient",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(page, encoding="utf-8")
    typer.echo(format_summary(summary, base_label, candidate_label))
    typer.echo(f"wrote {output}")


if __name__ == "__main__":
    app()
