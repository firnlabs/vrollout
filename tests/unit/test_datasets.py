"""Unit tests for dataset loading routines."""

from unittest.mock import MagicMock

import pytest

from vrollout import datasets


@pytest.fixture
def default_num_samples() -> int:
    return 2


def test_format_gsm8k_prompt_default() -> None:
    question = "If a box has 12 apples and 4 are removed, how many remain?"

    default_formatted = datasets.format_gsm8k_prompt(question)

    expected_instruction = (
        "Please reason step by step, and put your final answer within #### <answer>."
    )
    assert "Question: If a box has 12 apples" in default_formatted
    assert expected_instruction in default_formatted


def test_format_gsm8k_prompt_custom() -> None:
    question = "If a box has 12 apples and 4 are removed, how many remain?"
    custom_template = "Solve this problem: {question} Result:"

    custom_formatted = datasets.format_gsm8k_prompt(
        question, prompt_template=custom_template
    )

    assert custom_formatted == f"Solve this problem: {question} Result:"


@pytest.mark.parametrize("split, num_samples", [("train", 4), ("test", None)])
def test_load_gsm8k_dataset(
    mocker, split: str, num_samples: int | None, default_num_samples: int
) -> None:
    exp_num_samples = num_samples or default_num_samples
    mock_rows = [
        {"question": f"Question {i}?", "answer": f"Step {i}\n#### {i}"}
        for i in range(exp_num_samples)
    ]
    mock_load = mocker.patch(
        "vrollout.datasets.load_dataset",
        return_value=MagicMock(__iter__=lambda self: iter(mock_rows)),
    )

    samples = datasets.load_gsm8k_dataset(num_samples=num_samples, split=split)

    exp_split = f"{split}[:{num_samples}]" if num_samples is not None else split
    mock_load.assert_called_once_with("openai/gsm8k", "main", split=exp_split)
    assert len(samples) == exp_num_samples
    expected_prompt = (
        "Question: Question 0?\n"
        "Please reason step by step, and put your final answer within #### <answer>."
    )
    assert samples[0] == (expected_prompt, "Step 0\n#### 0")


def test_load_gsm8k_dataset_zero_samples_skips_download(mocker) -> None:
    mock_load = mocker.patch("vrollout.datasets.load_dataset")

    samples = datasets.load_gsm8k_dataset(num_samples=0)

    assert samples == []
    mock_load.assert_not_called()


def test_load_gsm8k_dataset_raw_prompts_when_template_none(mocker) -> None:
    mock_rows = [{"question": "What is 5 x 5?", "answer": "25\n#### 25"}]
    mocker.patch(
        "vrollout.datasets.load_dataset",
        return_value=MagicMock(__iter__=lambda self: iter(mock_rows)),
    )

    samples = datasets.load_gsm8k_dataset(num_samples=1, prompt_template=None)

    assert len(samples) == 1
    assert samples[0] == ("What is 5 x 5?", "25\n#### 25")


def test_load_gsm8k_dataset_custom_prompt_template(mocker) -> None:
    mock_rows = [{"question": "What is 10 / 2?", "answer": "5\n#### 5"}]
    mocker.patch(
        "vrollout.datasets.load_dataset",
        return_value=MagicMock(__iter__=lambda self: iter(mock_rows)),
    )
    template = "Problem:\n{question}\nAnswer:"

    samples = datasets.load_gsm8k_dataset(num_samples=1, prompt_template=template)

    assert len(samples) == 1
    assert samples[0] == ("Problem:\nWhat is 10 / 2?\nAnswer:", "5\n#### 5")


def test_load_synthetic_dataset(default_num_samples: int) -> None:
    seed = 42

    synthetic_samples = datasets.load_synthetic_dataset(
        num_samples=default_num_samples, seed=seed
    )

    assert len(synthetic_samples) == default_num_samples
    for q, a in synthetic_samples:
        assert "What is" in q
        assert "####" in a


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        pytest.param("gsm8k", datasets.DEFAULT_GSM8K_PROMPT_TEMPLATE, id="preset"),
        pytest.param(
            " Qwen-Boxed ", datasets.QWEN_BOXED_PROMPT_TEMPLATE, id="preset_casing"
        ),
        pytest.param("Q: {question}", "Q: {question}", id="custom"),
        pytest.param("raw", None, id="raw"),
        pytest.param("None", None, id="none"),
        pytest.param("  ", None, id="blank"),
    ],
)
def test_resolve_prompt_template(value: str, expected: str | None) -> None:
    template = datasets.resolve_prompt_template(value)

    assert template == expected


def test_qwen_boxed_prompt_template() -> None:
    question = "What is 2 + 2?"

    prompt = datasets.format_gsm8k_prompt(question, datasets.QWEN_BOXED_PROMPT_TEMPLATE)

    assert prompt == (
        "What is 2 + 2?\n"
        "Please reason step by step, and put your final answer within \\boxed{}."
    )


class _FakeChatTokenizer:
    def apply_chat_template(
        self,
        conversation: list[dict[str, str]],
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> str:
        assert not tokenize
        assert add_generation_prompt
        (turn,) = conversation
        return f"<user>{turn['content']}</user><assistant>"


def test_apply_chat_template() -> None:
    samples = [("Q1?", "#### 1"), ("Q2?", "#### 2")]

    templated = datasets.apply_chat_template(samples, _FakeChatTokenizer())

    assert templated == [
        ("<user>Q1?</user><assistant>", "#### 1"),
        ("<user>Q2?</user><assistant>", "#### 2"),
    ]
