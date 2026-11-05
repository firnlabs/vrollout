"""Unit tests for train CLI qualitative table and evaluation rendering."""

from datetime import timedelta
import json
import random
from unittest.mock import MagicMock

import pytest
import torch
import typer
from typer.testing import CliRunner

from vrollout import metrics
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


@pytest.mark.parametrize(
    ("every", "num_iterations", "expected"),
    [
        pytest.param(0, 5, [], id="disabled"),
        pytest.param(2, 5, [2, 4, 5], id="periodic_plus_final"),
        pytest.param(2, 4, [2, 4], id="final_on_period"),
    ],
)
def test_should_evaluate(every: int, num_iterations: int, expected: list[int]):
    steps = [
        step
        for step in range(1, num_iterations + 1)
        if train._should_evaluate(step, every, num_iterations)
    ]

    assert steps == expected


def test_format_eval_cell_none() -> None:
    formatted = train._format_eval_cell(None)

    assert formatted == "-"


def test_format_eval_cell_success() -> None:
    sr = _make_scored_rollout("What is 2+2?", "The answer is #### 4", 1.0)

    formatted = train._format_eval_cell(sr)

    assert "[green]✓ (1.0)[/green]" in formatted
    assert "[cyan][fmt:✓][/cyan]" in formatted
    assert "(pred: 4)" in formatted
    assert "The answer is #### 4" in formatted


def test_format_eval_cell_format_non_compliant() -> None:
    sr = _make_scored_rollout("What is 2+2?", "The answer is 4", 1.0)

    formatted = train._format_eval_cell(sr)

    assert "[green]✓ (1.0)[/green]" in formatted
    assert "[yellow][fmt:✗][/yellow]" in formatted


def test_format_eval_cell_failure_truncation() -> None:
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


def test_build_qualitative_table_shows_display_prompts() -> None:
    """Test that rendered prompts are shown by their display text when mapped."""
    rendered = "<|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\n"
    prompts = [rendered, "p_unmapped"]
    gt_map = {rendered: "4", "p_unmapped": "5"}
    baseline_eval = [
        _make_scored_rollout(rendered, "#### 3", 0.0),
        _make_scored_rollout("p_unmapped", "#### 1", 0.0),
    ]
    final_eval = [
        _make_scored_rollout(rendered, "#### 4", 1.0),
        _make_scored_rollout("p_unmapped", "#### 1", 0.0),
    ]

    table = train._build_qualitative_table(
        eval_prompts=prompts,
        prompt_to_ans=gt_map,
        baseline_eval=baseline_eval,
        final_eval=final_eval,
        max_samples=2,
        display_prompts={rendered: "What is 2+2?"},
    )

    assert table.columns[0]._cells == ["What is 2+2?", "p_unmapped"]
    assert table.columns[1]._cells == ["4", "5"]


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
    click_cmd = typer.main.get_command(train.app)
    opts = {opt for p in click_cmd.params for opt in p.opts}

    assert "--eval-max-tokens" in opts
    assert "--eval-completion-length" in opts
    assert "--eval-max-chars" in opts


def test_cli_eval_completion_flags_accepted() -> None:
    """Test that eval completion flags are accepted without parse error."""
    runner = CliRunner()

    result = runner.invoke(
        train.app,
        ["--eval-completion-length", "64", "--eval-max-chars", "128", "--help"],
    )

    assert result.exit_code == 0


def test_cli_wandb_flags_registered() -> None:
    """Test that wandb CLI options are registered in train app."""
    click_cmd = typer.main.get_command(train.app)

    opts = {opt for p in click_cmd.params for opt in p.opts}

    assert "--wandb" in opts
    assert "--wandb-project" in opts
    assert "--wandb-run-name" in opts


def test_cli_wandb_flags_accepted() -> None:
    """Test that wandb options are accepted without CLI parse error."""
    runner = CliRunner()

    result = runner.invoke(
        train.app,
        [
            "--wandb",
            "--wandb-project",
            "test-proj",
            "--wandb-run-name",
            "run-1",
            "--help",
        ],
    )

    assert result.exit_code == 0


def test_cli_format_flags_registered() -> None:
    click_cmd = typer.main.get_command(train.app)
    opts = {opt for p in click_cmd.params for opt in p.opts}

    assert "--prompt-template" in opts
    assert "--format-target" in opts
    assert "--strict-format" in opts


def test_cli_format_flags_accepted() -> None:
    runner = CliRunner()

    result = runner.invoke(
        train.app,
        [
            "--prompt-template",
            "Custom template {question}",
            "--format-target",
            "boxed",
            "--strict-format",
            "--help",
        ],
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


def test_cli_dist_timeout_flag_registered() -> None:
    """Test that the distributed timeout CLI option is registered in train app."""
    click_cmd = typer.main.get_command(train.app)

    opts = {opt for p in click_cmd.params for opt in p.opts}

    assert "--dist-timeout-minutes" in opts


def test_cli_dist_timeout_flag_accepted() -> None:
    """Test that the distributed timeout flag is accepted without parse error."""
    runner = CliRunner()

    result = runner.invoke(train.app, ["--dist-timeout-minutes", "90", "--help"])

    assert result.exit_code == 0


@pytest.mark.parametrize("minutes", ["0", "-5"])
def test_cli_dist_timeout_rejects_non_positive(minutes: str) -> None:
    """Test that a timeout below one minute is rejected before training starts."""
    runner = CliRunner()

    result = runner.invoke(
        train.app, ["--dist-timeout-minutes", minutes], standalone_mode=False
    )

    assert isinstance(result.exception, typer.BadParameter)
    assert result.exception.param is not None
    assert result.exception.param.name == "dist_timeout_minutes"


def test_cli_num_iterations_rejects_negative() -> None:
    runner = CliRunner()

    result = runner.invoke(train.app, ["--num-iterations", "-1"], standalone_mode=False)

    assert isinstance(result.exception, typer.BadParameter)
    assert result.exception.param is not None
    assert result.exception.param.name == "num_iterations"


@pytest.mark.parametrize("group_size", ["0", "-1"])
def test_cli_group_size_rejects_non_positive(group_size: str) -> None:
    runner = CliRunner()

    result = runner.invoke(
        train.app, ["--group-size", group_size], standalone_mode=False
    )

    assert isinstance(result.exception, typer.BadParameter)
    assert result.exception.param is not None
    assert result.exception.param.name == "group_size"


def test_cli_log_sequences_every_rejects_negative() -> None:
    runner = CliRunner()

    result = runner.invoke(
        train.app, ["--log-sequences-every", "-1"], standalone_mode=False
    )

    assert isinstance(result.exception, typer.BadParameter)
    assert result.exception.param is not None
    assert result.exception.param.name == "log_sequences_every"


def test_cli_checkpoint_flags_accepted(tmp_path) -> None:
    runner = CliRunner()

    result = runner.invoke(
        train.app,
        [
            "--seed",
            "3",
            "--checkpoint-dir",
            str(tmp_path),
            "--checkpoint-every",
            "5",
            "--help",
        ],
    )

    assert result.exit_code == 0


def test_cli_checkpoint_every_rejects_negative() -> None:
    runner = CliRunner()

    result = runner.invoke(
        train.app, ["--checkpoint-every", "-1"], standalone_mode=False
    )

    assert isinstance(result.exception, typer.BadParameter)
    assert result.exception.param is not None
    assert result.exception.param.name == "checkpoint_every"


def test_seed_all_makes_torch_and_python_rngs_repeatable() -> None:
    train._seed_all(5)
    first = (torch.rand(3), random.random())

    train._seed_all(5)
    second = (torch.rand(3), random.random())

    torch.testing.assert_close(first[0], second[0])
    assert first[1] == second[1]


@pytest.mark.parametrize(
    ("device", "backend", "device_id"),
    [
        pytest.param(torch.device("cpu"), "gloo", None, id="cpu-gloo"),
        pytest.param(
            torch.device("cuda:1"), "nccl", torch.device("cuda:1"), id="cuda-nccl"
        ),
    ],
)
def test_init_distributed_passes_timeout(
    monkeypatch: pytest.MonkeyPatch,
    device: torch.device,
    backend: str,
    device_id: torch.device | None,
) -> None:
    """Test that the process group is created with the configured timeout."""
    monkeypatch.setattr(train.dist, "is_initialized", lambda: False)
    mock_init = MagicMock()
    monkeypatch.setattr(train.dist, "init_process_group", mock_init)
    timeout = timedelta(minutes=90)

    train._init_distributed(device, timeout)

    mock_init.assert_called_once_with(
        backend=backend, device_id=device_id, timeout=timeout
    )


def test_init_distributed_skips_when_initialized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that an existing process group is reused."""
    monkeypatch.setattr(train.dist, "is_initialized", lambda: True)
    mock_init = MagicMock()
    monkeypatch.setattr(train.dist, "init_process_group", mock_init)

    train._init_distributed(torch.device("cpu"), timedelta(minutes=90))

    mock_init.assert_not_called()


def test_eval_accuracies_reports_strict_and_lenient() -> None:
    prompt_to_ans = {"p1": "#### 4", "p2": "#### 5"}
    scored_eval = [
        _make_scored_rollout("p1", "#### 4", 1.0),
        _make_scored_rollout("p2", "truncated at 5", 1.0),
    ]

    accuracies = train._eval_accuracies(scored_eval, prompt_to_ans)

    assert accuracies == {"eval_accuracy_strict": 0.5, "eval_accuracy_lenient": 1.0}


def test_format_accuracies() -> None:
    accuracies = {"eval_accuracy_strict": 0.25, "eval_accuracy_lenient": 0.5}

    cell = train._format_accuracies(accuracies)

    assert cell == "25.0% / 50.0%"


def _eval_pipeline(scored: list[scorer.ScoredRollout]) -> MagicMock:
    """Returns a fake rollout pipeline whose generation yields `scored`."""

    async def generate_and_score(*_args, **_kwargs) -> list[scorer.ScoredRollout]:
        return scored

    return MagicMock(generate_and_score=generate_and_score)


def test_run_evaluation_returns_mean_reward() -> None:
    scored_rollouts = [
        _make_scored_rollout("a", "", 1.0),
        _make_scored_rollout("b", "", 0.0),
    ]
    rollout_pipeline = _eval_pipeline(scored_rollouts)

    mean_reward, rollouts = train._run_evaluation(rollout_pipeline, ["a", "b"], 8)

    assert mean_reward == 0.5
    assert rollouts == scored_rollouts


def test_run_evaluation_no_prompts_returns_empty() -> None:
    rollout_pipeline = _eval_pipeline([])

    mean_reward, rollouts = train._run_evaluation(rollout_pipeline, [], 8)

    assert (mean_reward, rollouts) == (0.0, [])


def test_run_evaluation_no_rollouts_raises() -> None:
    rollout_pipeline = _eval_pipeline([])

    with pytest.raises(RuntimeError):
        train._run_evaluation(rollout_pipeline, ["a"], 8)


def test_save_eval_rows_writes_jsonl(tmp_path) -> None:
    rows = metrics.eval_rows(
        [
            _make_scored_rollout("q1", "#### 4", 1.0),
            _make_scored_rollout("q2", "", 0.0),
        ],
        {"q1": "#### 4"},
    )

    path = train._save_eval_rows(rows, str(tmp_path / "evals"), step=3)

    expected_path = tmp_path / "evals" / "eval-step-3.jsonl"
    assert path == expected_path
    lines = expected_path.read_text(encoding="utf-8").splitlines()
    assert [json.loads(line) for line in lines] == [row.to_dict() for row in rows]


def test_save_eval_rows_skipped_without_dir() -> None:
    rows = metrics.eval_rows([_make_scored_rollout("q1", "#### 4", 1.0)], {})

    path = train._save_eval_rows(rows, None, step=0)

    assert path is None


def test_cli_eval_output_dir_flag_accepted(tmp_path) -> None:
    runner = CliRunner()

    result = runner.invoke(train.app, ["--eval-output-dir", str(tmp_path), "--help"])

    assert result.exit_code == 0


def test_cli_chat_template_flag_accepted() -> None:
    runner = CliRunner()

    result = runner.invoke(
        train.app,
        ["--chat-template", "--prompt-template", "qwen-boxed", "--help"],
    )

    assert result.exit_code == 0


def test_cli_chat_template_rejects_mock_backend() -> None:
    runner = CliRunner()

    result = runner.invoke(
        train.app, ["--backend", "mock", "--chat-template"], standalone_mode=False
    )

    assert isinstance(result.exception, typer.BadParameter)
