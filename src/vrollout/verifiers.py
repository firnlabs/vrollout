"""Reward verifiers for RL post-training."""

from collections.abc import Sequence
import re
from typing import Any

import math_verify

from vrollout.types import RolloutTrajectory

REGEX_PATTERN_GSM8K = re.compile(r"####\s*(?:\\?\$\s*)?-?[\d.,]*\d")
REGEX_PATTERN_BOXED = re.compile(r"\\boxed\{[^}]")

# math-verify only counts answers marked with \boxed{} or a phrase such as "final
# answer:" as anchored, so GSM8K's "####" tag is rewritten into such a phrase
_GSM8K_TAG = re.compile(r"####")
_GSM8K_TAG_AS_ANCHOR = "Final answer:"
# base models often keep generating past their answer and invent the next few-shot
# "Question:", whose answer must not count
_FOLLOW_UP_QUESTION = re.compile(r"\n\s*Question:")

# \boxed{} outranks other anchors; among answers of one kind the last one wins
_STRICT_EXTRACTION = (
    math_verify.LatexExtractionConfig(
        try_extract_without_anchor=False, boxed_match_priority=0
    ),
    math_verify.ExprExtractionConfig(try_extract_without_anchor=False),
)
_LENIENT_EXTRACTION = (
    math_verify.LatexExtractionConfig(boxed_match_priority=0),
    math_verify.ExprExtractionConfig(),
)


def _strip_follow_up_questions(text: str) -> str:
    """Drops everything from the first invented follow-up "Question:" onwards."""
    return _FOLLOW_UP_QUESTION.split(text, maxsplit=1)[0]


def parse_answer(text: str, strict: bool = False) -> list[Any]:
    r"""Parses the final answer of a completion or reference solution.

    Answers marked with '\boxed{}', '#### ', or a phrase such as 'The answer is'
    are found by math-verify, which also normalizes '$', thousands separators and
    LaTeX units. Text from an invented follow-up 'Question:' onwards is ignored.

    Args:
        text: The completion or reference solution to parse.
        strict: If True, only marked answers count; otherwise math-verify falls
            back to the last expression in the text.

    Returns:
        math-verify's parse result: the answer as a SymPy object followed by the
        matched string, or an empty list if no answer was found.
    """
    if not text:
        return []
    text = _GSM8K_TAG.sub(_GSM8K_TAG_AS_ANCHOR, _strip_follow_up_questions(text))
    return math_verify.parse(
        text, extraction_config=_STRICT_EXTRACTION if strict else _LENIENT_EXTRACTION
    )


def check_format_compliance(text: str, target_format: str = "gsm8k") -> bool:
    """Checks whether the completion text complies with expected answer formatting.

    Args:
        text: The model completion or reasoning trace to check.
        target_format: Expected format identifier ('gsm8k', 'boxed', or 'any').

    Returns:
        True if the text contains a compliant formatted answer tag, False otherwise.

    Raises:
        ValueError: If target_format is not recognized.
    """
    if not text:
        return False

    text = _strip_follow_up_questions(text)
    if target_format == "gsm8k":
        return bool(REGEX_PATTERN_GSM8K.search(text))
    if target_format == "boxed":
        return bool(REGEX_PATTERN_BOXED.search(text))
    if target_format == "any":
        return bool(
            REGEX_PATTERN_GSM8K.search(text) or REGEX_PATTERN_BOXED.search(text)
        )

    raise ValueError(f"Unknown target_format: {target_format!r}")


def compute_format_compliance_rate(
    rollouts: Sequence[Any], target_format: str = "gsm8k"
) -> float:
    """Computes the format compliance rate across a batch of rollouts.

    Args:
        rollouts: Sequence of rollout trajectories, scored rollouts, or text strings.
        target_format: Expected format identifier ('gsm8k', 'boxed', or 'any').

    Returns:
        Fraction of compliant rollouts as a float between 0.0 and 1.0.
    """
    if not rollouts:
        return 0.0

    compliant_count = 0
    for r in rollouts:
        if hasattr(r, "trajectory") and hasattr(r.trajectory, "text"):
            text = r.trajectory.text
        elif hasattr(r, "text"):
            text = r.text
        else:
            text = str(r)

        if check_format_compliance(text, target_format=target_format):
            compliant_count += 1

    return compliant_count / len(rollouts)


def extract_numerical_answer(text: str, strict: bool = False) -> str | None:
    """Returns the final answer of `text` as a string, for display and logging.

    Args:
        text: The completion or reference solution to parse.
        strict: If True, only answers marked with a format tag count.

    Returns:
        The answer string math-verify matched, or None if no answer was found.
    """
    parsed = parse_answer(text, strict=strict)
    if not parsed:
        return None
    return str(parsed[-1]).replace(",", "")


def verify_math_answer(
    prediction: str, ground_truth: str, strict_format: bool = False
) -> float:
    """Compares predicted mathematical answer against ground truth answer.

    Args:
        prediction: Candidate completion string.
        ground_truth: Ground truth reasoning trace containing answer.
        strict_format: If True, requires prediction to adhere to explicit format tags.

    Returns:
        1.0 if math-verify finds the answers equivalent, 0.0 otherwise.
    """
    gt = parse_answer(ground_truth)
    pred = parse_answer(prediction, strict=strict_format)
    if not gt or not pred:
        return 0.0
    return 1.0 if math_verify.verify(gt, pred) else 0.0


class GSM8KVerifier:
    """Async reward verifier for GSM8K math problems."""

    def __init__(
        self, prompt_to_answer: dict[str, str], strict_format: bool = False
    ) -> None:
        """Initializes GSM8K verifier with ground truth mapping.

        Args:
            prompt_to_answer: Dictionary mapping prompts to ground truth answers.
            strict_format: Whether to enforce explicit format tags for predictions.
        """
        self._prompt_to_answer = prompt_to_answer
        self._strict_format = strict_format

    async def __call__(self, rollout: RolloutTrajectory) -> float:
        """Scores a rollout trajectory against ground truth."""
        gt = self._prompt_to_answer.get(rollout.prompt)
        if gt is None:
            # Fallback: check stripped prompt
            gt = self._prompt_to_answer.get(rollout.prompt.strip(), "")
        return verify_math_answer(rollout.text, gt, strict_format=self._strict_format)


async def synthetic_verifier(rollout: RolloutTrajectory) -> float:
    """Deterministic verifier for synthetic and smoke testing."""
    return 1.0 if len(rollout.completion_token_ids) > 0 else 0.0
