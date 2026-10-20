"""Unit tests for train CLI qualitative table and evaluation rendering."""

from vrollout import scorer
from vrollout import train
from vrollout import types


def _make_scored_rollout(prompt: str, text: str, reward: float) -> scorer.ScoredRollout:
    """Helper to construct a ScoredRollout with minimal trajectory fields."""
    return scorer.ScoredRollout(
        trajectory=types.RolloutTrajectory(
            request_id="req-test",
            prompt=prompt,
            prompt_token_ids=[],
            completion_token_ids=[],
            logprobs=[],
            finish_reason="stop",
            text=text,
            worker_id=0,
        ),
        reward=reward,
    )


def test_format_eval_cell_none() -> None:
    """Test formatting when rollout is None."""
    assert train._format_eval_cell(None) == "-"


def test_format_eval_cell_success() -> None:
    """Test formatting for successful rollout with reward >= 0.5."""
    sr = _make_scored_rollout("What is 2+2?", "The answer is #### 4", 1.0)
    formatted = train._format_eval_cell(sr)
    assert "[green]✓ (1.0)[/green]" in formatted
    assert "(pred: 4)" in formatted
    assert "The answer is #### 4" in formatted


def test_format_eval_cell_failure_truncation() -> None:
    """Test formatting for failed rollout and text truncation."""
    long_text = "wrong " * 50
    sr = _make_scored_rollout("What is 2+2?", long_text, 0.0)
    formatted = train._format_eval_cell(sr, max_chars=30)
    assert "[red]✗ (0.0)[/red]" in formatted
    assert "..." in formatted


def test_build_qualitative_table_prioritizes_improved_prompts() -> None:
    """Test that prompts with reward improvement (0.0 -> 1.0) are sorted first."""
    prompts = ["p_easy", "p_hard", "p_improved"]
    gt_map = {
        "p_easy": "#### 10",
        "p_hard": "#### 20",
        "p_improved": "#### 30",
    }

    baseline_eval = [
        _make_scored_rollout("p_easy", "#### 10", 1.0),
        _make_scored_rollout("p_hard", "#### 99", 0.0),
        _make_scored_rollout("p_improved", "#### 99", 0.0),
    ]
    final_eval = [
        _make_scored_rollout("p_easy", "#### 10", 1.0),
        _make_scored_rollout("p_hard", "#### 99", 0.0),
        _make_scored_rollout("p_improved", "#### 30", 1.0),
    ]

    table = train._build_qualitative_table(
        eval_prompts=prompts,
        prompt_to_ans=gt_map,
        baseline_eval=baseline_eval,
        final_eval=final_eval,
        max_samples=2,
    )

    # Table should have 2 rows (limited by max_samples)
    assert len(table.rows) == 2
    # The first row must be p_improved because it improved from 0.0 to 1.0
    row0_prompt = table.columns[0]._cells[0]
    assert row0_prompt == "p_improved"


def test_build_qualitative_table_preserves_order_when_rewards_equal() -> None:
    """Test stable tie-breaking when all prompts have identical reward deltas."""
    prompts = ["p1", "p2", "p3"]
    gt_map = {"p1": "1", "p2": "2", "p3": "3"}

    baseline_eval = [
        _make_scored_rollout("p1", "1", 1.0),
        _make_scored_rollout("p2", "2", 1.0),
        _make_scored_rollout("p3", "3", 1.0),
    ]
    final_eval = [
        _make_scored_rollout("p1", "1", 1.0),
        _make_scored_rollout("p2", "2", 1.0),
        _make_scored_rollout("p3", "3", 1.0),
    ]

    table = train._build_qualitative_table(
        eval_prompts=prompts,
        prompt_to_ans=gt_map,
        baseline_eval=baseline_eval,
        final_eval=final_eval,
        max_samples=3,
    )

    assert len(table.rows) == 3
    assert table.columns[0]._cells == ["p1", "p2", "p3"]
