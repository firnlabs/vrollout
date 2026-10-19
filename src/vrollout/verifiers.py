"""Reward verifiers for RL post-training."""

import re

from beartype import beartype
from jaxtyping import jaxtyped

from vrollout.types import RolloutTrajectory


def extract_numerical_answer(text: str) -> str | None:
    r"""Extracts final numerical answer from reasoning or completion text.

    Supports '#### <answer>', '\boxed{<answer>}', and trailing numbers.
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

    # fallback to the last standalone number
    matches = re.findall(r"[-+]?\d*\.?\d+", text.replace(",", ""))
    if matches:
        return matches[-1].rstrip(".")

    return None


@jaxtyped(typechecker=beartype)
def verify_math_answer(prediction: str, ground_truth: str) -> float:
    """Compares predicted mathematical answer against ground truth answer.

    Returns 1.0 for exact numerical equivalence, 0.0 otherwise.
    """
    pred_ans = extract_numerical_answer(prediction)
    gt_ans = extract_numerical_answer(ground_truth)

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

    def __init__(self, prompt_to_answer: dict[str, str]) -> None:
        self._prompt_to_answer = prompt_to_answer

    async def __call__(self, rollout: RolloutTrajectory) -> float:
        """Scores a rollout trajectory against ground truth."""
        gt = self._prompt_to_answer.get(rollout.prompt)
        if gt is None:
            # Fallback: check stripped prompt
            gt = self._prompt_to_answer.get(rollout.prompt.strip(), "")
        return verify_math_answer(rollout.text, gt)


async def synthetic_verifier(rollout: RolloutTrajectory) -> float:
    """Deterministic verifier for synthetic and smoke testing."""
    return 1.0 if len(rollout.completion_token_ids) > 0 else 0.0
