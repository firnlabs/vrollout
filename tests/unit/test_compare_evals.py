"""Unit tests for the side-by-side eval comparison."""

import json
import pathlib
from typing import Any

import pytest
from typer.testing import CliRunner

from vrollout import compare_evals
from vrollout import metrics

_REFERENCE = "16 - 3 - 4 = 9 eggs, 9 * 2 = 18.\n#### 18"


def _row(
    prompt: str, completion: str, reference: str | None = _REFERENCE
) -> dict[str, Any]:
    return metrics.EvalRow(
        prompt=prompt,
        ground_truth="18",
        ground_truth_raw=reference,
        completion=completion,
        extracted_answer=None,
        reward=0.0,
        finish_reason="stop",
        length=10,
    ).to_dict()


def _write(path: pathlib.Path, rows: list[dict[str, Any]]) -> pathlib.Path:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


def _evals(
    answers: list[tuple[str, str]],
) -> tuple[dict[str, metrics.EvalRow], dict[str, metrics.EvalRow]]:
    base = {
        f"q{i}": metrics.EvalRow(**_row(f"q{i}", a)) for i, (a, _) in enumerate(answers)
    }
    cand = {
        f"q{i}": metrics.EvalRow(**_row(f"q{i}", b)) for i, (_, b) in enumerate(answers)
    }
    return base, cand


def test_pair_evals_classifies_each_outcome() -> None:
    right, wrong = r"so \boxed{18}", r"so \boxed{17}"
    base, cand = _evals(
        [(wrong, right), (right, wrong), (right, right), (wrong, wrong)]
    )

    problems = compare_evals.pair_evals(base, cand, strict=True)

    assert [p.outcome for p in problems] == list(compare_evals.Outcome)


def test_pair_evals_strict_rejects_unmarked_answer_lenient_accepts() -> None:
    base, cand = _evals([("9 * 2 = 18", "9 * 2 = 18")])

    strict = compare_evals.pair_evals(base, cand, strict=True)
    lenient = compare_evals.pair_evals(base, cand, strict=False)

    assert not strict[0].base.correct
    assert lenient[0].base.correct
    assert lenient[0].base.answer == "18"


def test_pair_evals_rejects_different_prompt_sets() -> None:
    base, cand = _evals([("a", "a"), ("b", "b")])
    del cand["q1"]

    with pytest.raises(ValueError, match="1 only in base"):
        compare_evals.pair_evals(base, cand, strict=True)


def test_score_answer_rejects_missing_reference() -> None:
    row = metrics.EvalRow(**_row("q", r"\boxed{18}", reference=None))

    with pytest.raises(ValueError, match="no reference"):
        compare_evals.score_answer(row, strict=True)


def test_load_eval_rejects_duplicate_prompts(tmp_path: pathlib.Path) -> None:
    path = _write(tmp_path / "e.jsonl", [_row("q", "a"), _row("q", "b")])

    with pytest.raises(ValueError, match="duplicate prompt"):
        compare_evals.load_eval(path)


def test_load_eval_skips_blank_lines_and_keeps_order(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "e.jsonl"
    path.write_text(
        json.dumps(_row("q1", "a")) + "\n\n" + json.dumps(_row("q0", "b")) + "\n"
    )

    rows = compare_evals.load_eval(path)

    assert list(rows) == ["q1", "q0"]


@pytest.mark.parametrize(
    ("gained", "lost", "expected"),
    [
        (0, 0, 1.0),
        (5, 5, 1.0),
        (6, 0, 0.03125),
        (137, 86, 0.0007759),
        (86, 137, 0.0007759),
    ],
)
def test_mcnemar_exact_p(gained: int, lost: int, expected: float) -> None:
    p = compare_evals.mcnemar_exact_p(gained, lost)

    assert p == pytest.approx(expected, rel=1e-3)


def test_summarize_matches_hand_computed_paired_stats() -> None:
    right, wrong = r"\boxed{18}", r"\boxed{1}"
    pairs = (
        [(wrong, right)] * 3
        + [(right, wrong)]
        + [(right, right)] * 2
        + [(wrong, wrong)] * 2
    )
    problems = compare_evals.pair_evals(*_evals(pairs), strict=True)

    summary = compare_evals.summarize(problems)

    assert (summary.gained, summary.lost, summary.num_problems) == (3, 1, 8)
    assert summary.base_accuracy == pytest.approx(3 / 8)
    assert summary.candidate_accuracy == pytest.approx(5 / 8)
    assert summary.delta == pytest.approx(0.25)
    assert summary.ci95_half_width == pytest.approx(1.959964 * (4 - 4 / 8) ** 0.5 / 8)
    assert summary.p_value == pytest.approx(0.625)


def test_summarize_rejects_empty() -> None:
    with pytest.raises(ValueError, match="no problems"):
        compare_evals.summarize([])


@pytest.mark.parametrize(
    ("prompt", "expected"),
    [
        (
            "<|im_start|>system\nBe nice.<|im_end|>\n<|im_start|>user\n"
            "How many eggs?<|im_end|>\n<|im_start|>assistant\n",
            "How many eggs?",
        ),
        ("  Q: How many eggs?\nA:", "Q: How many eggs?\nA:"),
    ],
)
def test_question_text(prompt: str, expected: str) -> None:
    text = compare_evals.question_text(prompt)

    assert text == expected


def test_render_html_escapes_text_and_tags_rows() -> None:
    pairs = [
        (r"<script>x</script> \boxed{18}", r"\boxed{17}"),
        (r"\boxed{1}", r"\boxed{18}"),
    ]
    problems = compare_evals.pair_evals(*_evals(pairs), strict=True)
    summary = compare_evals.summarize(problems)

    page = compare_evals.render_html(problems, summary, "E0", "E1<b>", "strict")

    assert "<script>x</script>" not in page
    assert "&lt;script&gt;x&lt;/script&gt;" in page
    assert "E1&lt;b&gt;" in page
    assert page.count('data-outcome="lost"') == 1
    assert page.count('data-outcome="gained"') == 1


def test_render_html_inlines_stylesheet() -> None:
    problems = compare_evals.pair_evals(
        *_evals([(r"\boxed{18}", r"\boxed{1}")]), strict=True
    )
    summary = compare_evals.summarize(problems)

    page = compare_evals.render_html(problems, summary, "E0", "E1", "strict")

    style = page.split("<style>", 1)[1].split("</style>", 1)[0]
    assert "--accent:" in style
    assert "prefers-color-scheme: dark" in style
    assert "{%" not in page
    assert "<link" not in page


def test_cli_writes_page_and_prints_summary(tmp_path: pathlib.Path) -> None:
    base = _write(
        tmp_path / "e0.jsonl", [_row("q0", r"\boxed{1}"), _row("q1", r"\boxed{18}")]
    )
    cand = _write(
        tmp_path / "e1.jsonl", [_row("q0", r"\boxed{18}"), _row("q1", r"\boxed{18}")]
    )
    out = tmp_path / "out" / "cmp.html"

    result = CliRunner().invoke(
        compare_evals.app,
        [
            str(base),
            str(cand),
            "-o",
            str(out),
            "--base-label",
            "E0",
            "--candidate-label",
            "E1",
        ],
    )

    assert result.exit_code == 0
    assert out.exists()
    assert "E0 0.500 -> E1 1.000" in result.output


def test_cli_lenient_scores_unmarked_answers(tmp_path: pathlib.Path) -> None:
    base = _write(tmp_path / "e0.jsonl", [_row("q0", "9 * 2 = 18")])
    cand = _write(tmp_path / "e1.jsonl", [_row("q0", r"\boxed{18}")])

    result = CliRunner().invoke(
        compare_evals.app,
        [str(base), str(cand), "-o", str(tmp_path / "c.html"), "--lenient"],
    )

    assert result.exit_code == 0
    assert "base 1.000 -> candidate 1.000" in result.output


def test_cli_raises_on_mismatched_prompts(tmp_path: pathlib.Path) -> None:
    base = _write(tmp_path / "e0.jsonl", [_row("q0", "a")])
    cand = _write(tmp_path / "e1.jsonl", [_row("q1", "a")])

    with pytest.raises(ValueError):
        CliRunner().invoke(
            compare_evals.app,
            [str(base), str(cand), "-o", str(tmp_path / "c.html")],
            standalone_mode=False,
            catch_exceptions=False,
        )
