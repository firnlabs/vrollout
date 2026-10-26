"""Unit tests for train CLI qualitative table and evaluation rendering."""

from unittest.mock import MagicMock

import pytest
import torch

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
    formatted = train._format_eval_cell(None)

    assert formatted == "-"


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


def test_format_eval_cell_tail_truncation() -> None:
    """Test that long completions preserve the tail ending with answer."""
    text = "beginning steps that are long and repetitive. The answer is #### 42"
    sr = _make_scored_rollout("What is the answer?", text, 1.0)

    formatted = train._format_eval_cell(sr, max_chars=35)

    assert "[green]✓ (1.0)[/green]" in formatted
    assert "(pred: 42)" in formatted
    assert formatted.endswith("The answer is #### 42")
    assert "..." in formatted


def test_build_qualitative_table_prioritizes_improved_prompts() -> None:
    """Test that prompts with reward improvement (0.0 -> 1.0) are sorted first."""
    prompts = ["p_easy", "p_hard", "p_improved"]
    gt_map = {"p_easy": "#### 10", "p_hard": "#### 20", "p_improved": "#### 30"}
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

    assert len(table.rows) == 2
    assert table.columns[0]._cells[0] == "p_improved"


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


def test_format_eval_cell_zero_max_chars_disables_truncation() -> None:
    """Test that max_chars=0 disables text truncation for long strings."""
    long_text = "step " * 100
    sr = _make_scored_rollout("What is 2+2?", long_text, 1.0)

    formatted = train._format_eval_cell(sr, max_chars=0)

    assert "[green]✓ (1.0)[/green]" in formatted
    assert "..." not in formatted
    assert long_text.strip() in formatted


def test_build_qualitative_table_custom_max_chars() -> None:
    """Test that build_qualitative_table respects custom max_chars per cell."""
    long_text = "explanation " * 20
    prompts = ["p1"]
    gt_map = {"p1": "42"}
    baseline_eval = [_make_scored_rollout("p1", long_text, 0.0)]
    final_eval = [_make_scored_rollout("p1", long_text, 1.0)]

    table = train._build_qualitative_table(
        eval_prompts=prompts,
        prompt_to_ans=gt_map,
        baseline_eval=baseline_eval,
        final_eval=final_eval,
        max_samples=1,
        max_chars=40,
    )

    cell_before = str(table.columns[2]._cells[0])
    cell_after = str(table.columns[3]._cells[0])
    assert "..." in cell_before
    assert "..." in cell_after


def test_cli_eval_completion_flags_registered() -> None:
    """Test that eval completion length CLI flags are registered in train app."""
    import typer

    click_cmd = typer.main.get_command(train.app)
    opts = {opt for p in click_cmd.params for opt in p.opts}

    assert "--eval-max-tokens" in opts
    assert "--eval-completion-length" in opts
    assert "--eval-max-chars" in opts


def test_cli_eval_completion_flags_accepted() -> None:
    """Test that eval completion flags are accepted without parse error."""
    from typer.testing import CliRunner

    runner = CliRunner()

    result = runner.invoke(
        train.app,
        ["--eval-completion-length", "64", "--eval-max-chars", "128", "--help"],
    )

    assert result.exit_code == 0


def test_create_models_falls_back_to_fp16_when_bf16_unsupported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_device = torch.device("cuda:0")
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: False)
    mock_from_pretrained = MagicMock()
    monkeypatch.setattr(
        "transformers.AutoModelForCausalLM.from_pretrained", mock_from_pretrained
    )

    train._create_models("test-model", backend="vllm", device=fake_device)

    assert mock_from_pretrained.call_count == 2
    ref_call, actor_call = mock_from_pretrained.call_args_list
    assert ref_call.kwargs["dtype"] == torch.float16
    assert actor_call.kwargs["dtype"] == torch.float32


def test_create_models_uses_bf16_when_supported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_device = torch.device("cuda:0")
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)
    mock_from_pretrained = MagicMock()
    monkeypatch.setattr(
        "transformers.AutoModelForCausalLM.from_pretrained", mock_from_pretrained
    )

    train._create_models("test-model", backend="vllm", device=fake_device)

    assert mock_from_pretrained.call_count == 2
    ref_call, actor_call = mock_from_pretrained.call_args_list
    assert ref_call.kwargs["dtype"] == torch.bfloat16
    assert actor_call.kwargs["dtype"] == torch.float32
