"""Dataset loading routines for RL post-training."""

from collections.abc import Sequence
import random
from typing import Protocol

from datasets import load_dataset

DEFAULT_GSM8K_PROMPT_TEMPLATE = (
    "Question: {question}\n"
    "Please reason step by step, and put your final answer within #### <answer>."
)
# Qwen2.5's chain-of-thought prompt, used for its published GSM8K and MATH scores
QWEN_BOXED_PROMPT_TEMPLATE = (
    "{question}\nPlease reason step by step, and put your final answer within "
    "\\boxed{}."
)
PROMPT_TEMPLATE_PRESETS: dict[str, str] = {
    "gsm8k": DEFAULT_GSM8K_PROMPT_TEMPLATE,
    "qwen-boxed": QWEN_BOXED_PROMPT_TEMPLATE,
}
_RAW_PROMPT_TEMPLATE_NAMES = ("", "none", "raw")


class ChatTokenizer(Protocol):
    """Tokenizer that renders a conversation with its chat template.

    Mirrors the subset of `transformers.PreTrainedTokenizerBase.apply_chat_template`
    used here, so Hugging Face tokenizers satisfy it directly.
    """

    def apply_chat_template(
        self,
        conversation: list[dict[str, str]],
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> str:
        """Renders `conversation` as a prompt string.

        Args:
            conversation: Chat turns, each a dict with 'role' (e.g. 'user') and
                'content' keys.
            tokenize: Whether to return token ids instead of a string. Callers
                pass False to get the rendered text.
            add_generation_prompt: Whether to append the assistant turn header
                so the model continues as the assistant.

        Returns:
            The conversation rendered with the tokenizer's chat template.
        """
        ...


def resolve_prompt_template(name_or_template: str) -> str | None:
    """Resolves a `--prompt-template` value to a template string.

    Args:
        name_or_template: A key of `PROMPT_TEMPLATE_PRESETS`, one of 'raw', 'none'
            or '' to keep the raw question, or a template with a '{question}'
            placeholder.

    Returns:
        The template string, or None to keep the raw question.
    """
    key = name_or_template.strip().lower()
    if key in _RAW_PROMPT_TEMPLATE_NAMES:
        return None
    return PROMPT_TEMPLATE_PRESETS.get(key, name_or_template)


def apply_chat_template(
    samples: Sequence[tuple[str, str]], tokenizer: ChatTokenizer
) -> list[tuple[str, str]]:
    """Wraps each prompt as a single user turn of the tokenizer's chat template.

    The rendered prompt ends with the assistant generation header, so it can be
    sent to the sampler as a raw string. Answers are left unchanged.

    Args:
        samples: (prompt, answer) pairs.
        tokenizer: Tokenizer whose chat template renders the prompts.

    Returns:
        (chat-formatted prompt, answer) pairs, in input order.
    """
    return [
        (
            tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            ),
            answer,
        )
        for prompt, answer in samples
    ]


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
    # HF rejects empty slices like "train[:0]"; eval-only runs request no train data
    if num_samples == 0:
        return []
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
