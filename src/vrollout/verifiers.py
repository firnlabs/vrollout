"""Reward verifiers for RL post-training."""

from collections.abc import Sequence
import re
from typing import Any

from beartype import beartype
from jaxtyping import jaxtyped

from vrollout.types import RolloutTrajectory

REGEX_PATTERN_GSM8K = re.compile(r"####\s*([-\d\.,]+)")
REGEX_PATTERN_BOXED = re.compile(r"\\boxed\{([-\d\.,]+)\}")


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
    r"""Extracts final numerical answer from reasoning or completion text.

    Supports '#### <answer>', '\boxed{<answer>}', and 'The answer is <answer>'.
    When strict is False, falls back to the last standalone number.

    Args:
        text: The reasoning completion string to parse.
        strict: If True, disables the fallback to unformatted standalone numbers.

    Returns:
        Extracted answer string with thousands commas stripped, or None if no match.
    """
    if not text:
        return None

    # check for '#### <answer>'
    match = re.search(r"####\s*([-\d\.,]+)", text)
    if match:
        raw = match.group(1).replace(",", "").strip()
        return raw.rstrip(".")

    # check for '\boxed{<answer>}'
    match = re.search(r"\\boxed\{([-\d\.,]+)\}", text)
    if match:
        raw = match.group(1).replace(",", "").strip()
        return raw.rstrip(".")

    # check for 'The answer is <answer>'
    match = re.search(
        r"(?:the\s+answer\s+is|answer\s*[:=])\s*([-\d\.,]+)", text, re.IGNORECASE
    )
    if match:
        raw = match.group(1).replace(",", "").strip()
        return raw.rstrip(".")

    if strict:
        return None

    # fallback to the last standalone number
    matches = re.findall(r"[-+]?\d*\.?\d+", text.replace(",", ""))
    if matches:
        return matches[-1].rstrip(".")

    return None


@jaxtyped(typechecker=beartype)
def verify_math_answer(
    prediction: str, ground_truth: str, strict_format: bool = False
) -> float:
    """Compares predicted mathematical answer against ground truth answer.

    Args:
        prediction: Candidate completion string.
        ground_truth: Ground truth reasoning trace containing answer.
        strict_format: If True, requires prediction to adhere to explicit format tags.

    Returns:
        1.0 for exact numerical equivalence, 0.0 otherwise.
    """
    pred_ans = extract_numerical_answer(prediction, strict=strict_format)
    gt_ans = extract_numerical_answer(ground_truth, strict=False)

    if pred_ans is None or gt_ans is None:
        return 0.0

    if pred_ans == gt_ans:
        return 1.0

    try:
        if abs(float(pred_ans) - float(gt_ans)) < 1e-5:
            return 1.0
    except ValueError:
        pass

    return 0.0


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
