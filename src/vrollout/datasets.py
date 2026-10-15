"""Dataset loading routines for RL post-training."""

import random

from datasets import load_dataset


def load_gsm8k_dataset(
    num_samples: int | None = None,
    split: str = "train",
) -> list[tuple[str, str]]:
    """Loads GSM8K prompt-answer pairs from Hugging Face Hub.

    Args:
        num_samples: Optional maximum number of samples to load.
        split: Dataset split to load (e.g., "train" or "test").

    Returns:
        List of (question, answer) string tuples.
    """
    target_split = f"{split}[:{num_samples}]" if num_samples is not None else split
    ds = load_dataset("openai/gsm8k", "main", split=target_split)
    return [(str(row["question"]), str(row["answer"])) for row in ds]


def load_synthetic_dataset(
    num_samples: int = 16, seed: int = 42
) -> list[tuple[str, str]]:
    """Generates synthetic math problems for fast offline testing."""
    problems = []
    rng = random.Random(seed)
    for _ in range(num_samples):
        a = rng.randint(1, 50)
        b = rng.randint(1, 50)
        ans = a + b
        prompt = f"What is {a} + {b}?"
        answer = f"The sum of {a} and {b} is {ans}.\n#### {ans}"
        problems.append((prompt, answer))
    return problems
