"""Dataset loading routines for RL post-training."""

import random

from datasets import load_dataset

DEFAULT_GSM8K_PROMPT_TEMPLATE = (
    "Question: {question}\n"
    "Please reason step by step, and put your final answer within #### <answer>."
)


def format_gsm8k_prompt(
    question: str,
    prompt_template: str = DEFAULT_GSM8K_PROMPT_TEMPLATE,
) -> str:
    """Applies a reasoning prompt template to a raw GSM8K question.

    Args:
        question: The raw question text from the dataset.
        prompt_template: Formatting template containing '{question}' placeholder.

    Returns:
        The formatted question string.
    """
    return prompt_template.replace("{question}", question)


def load_gsm8k_dataset(
    num_samples: int | None = None,
    split: str = "train",
    prompt_template: str | None = DEFAULT_GSM8K_PROMPT_TEMPLATE,
) -> list[tuple[str, str]]:
    """Loads GSM8K prompt-answer pairs from Hugging Face Hub.

    Args:
        num_samples: Optional maximum number of samples to load.
        split: Dataset split to load (e.g., "train" or "test").
        prompt_template: Optional template to format the prompt. If None, the
            raw question is preserved.

    Returns:
        List of (prompt, answer) string tuples.
    """
    target_split = f"{split}[:{num_samples}]" if num_samples is not None else split
    ds = load_dataset("openai/gsm8k", "main", split=target_split)
    return [
        (
            format_gsm8k_prompt(str(row["question"]), prompt_template)
            if prompt_template is not None
            else str(row["question"]),
            str(row["answer"]),
        )
        for row in ds
    ]


def load_synthetic_dataset(
    num_samples: int = 16, seed: int = 42
) -> list[tuple[str, str]]:
    """Generates synthetic math problems for fast offline testing.

    Args:
        num_samples: Number of synthetic math problems to generate.
        seed: Random seed for deterministic generation.

    Returns:
        List of (prompt, answer) string tuples.
    """
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
