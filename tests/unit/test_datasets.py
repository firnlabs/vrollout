"""Unit tests for dataset loading routines."""

from unittest.mock import MagicMock

import pytest

from vrollout import datasets


@pytest.fixture
def default_num_samples() -> int:
    return 2


@pytest.mark.parametrize("split, num_samples", [("train", 4), ("test", None)])
def test_load_gsm8k_dataset(
    mocker, split: str, num_samples: int | None, default_num_samples: int
):
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
    assert samples[0] == ("Question 0?", "Step 0\n#### 0")


def test_load_synthetic_dataset(default_num_samples: int):
    seed = 42
    synthetic_samples = datasets.load_synthetic_dataset(
        num_samples=default_num_samples, seed=seed
    )
    assert len(synthetic_samples) == default_num_samples
    for q, a in synthetic_samples:
        assert "What is" in q
        assert "####" in a
