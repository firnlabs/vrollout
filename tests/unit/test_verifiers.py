"""Unit tests for reward verifiers and format compliance metrics."""

import pytest

from vrollout import scorer
from vrollout import verifiers
from vrollout.types import RolloutTrajectory
from vrollout.verifiers import check_format_compliance
from vrollout.verifiers import compute_format_compliance_rate
from vrollout.verifiers import extract_numerical_answer
from vrollout.verifiers import GSM8KVerifier
from vrollout.verifiers import synthetic_verifier
from vrollout.verifiers import verify_math_answer


@pytest.mark.parametrize(
    ("answer", "expected"),
    [
        pytest.param("The answer is #### 42", "42", id="hash_tag"),
        pytest.param("#### -12.5", "-12.5", id="hash_tag_negative"),
        pytest.param("The result is \\boxed{1,250}", "1250", id="boxed"),
        pytest.param("The answer is 100.", "100", id="phrase"),
        pytest.param("There are 5 apples and 10 oranges.", "10", id="standalone"),
        pytest.param("", None, id="empty"),
        pytest.param("No numbers here", None, id="no_numbers"),
        pytest.param("#### $18", "18", id="hash_tag_dollar"),
        pytest.param("\\boxed{\\$18}", "18", id="boxed_escaped_dollar"),
        pytest.param("\\boxed{18 \\text{ apples}}", "18", id="boxed_text_units"),
        pytest.param("#### 5\nWait, fix: #### 7", "7", id="last_hash_tag"),
        pytest.param("The answer is 5. No, the answer is 7.", "7", id="last_phrase"),
        pytest.param("\\boxed{7} then #### 5", "7", id="boxed_outranks_hash_tag"),
        pytest.param(
            "So #### 18\nQuestion: Tom has 3 cats. #### 3",
            "18",
            id="ignores_follow_up_question",
        ),
    ],
)
def test_extract_numerical_answer(answer: str, expected: str | None) -> None:
    res = extract_numerical_answer(answer)

    assert res == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        pytest.param("First step then #### 42", "42", id="formatted"),
        pytest.param("First step then 42", None, id="unformatted"),
        pytest.param("Done.\nQuestion: next? #### 3", None, id="only_follow_up"),
    ],
)
def test_extract_numerical_answer_strict_mode(text: str, expected: str | None) -> None:
    ans = extract_numerical_answer(text, strict=True)

    assert ans == expected


@pytest.mark.parametrize(
    ("text", "target_format", "expected"),
    [
        pytest.param(
            "Step 1: 2+2=4. Final answer: #### 4", "gsm8k", True, id="gsm8k_valid_hash"
        ),
        pytest.param(
            "Step 1: 2+2=4. \\boxed{4}", "gsm8k", False, id="gsm8k_invalid_boxed"
        ),
        pytest.param("The answer is 4.", "gsm8k", False, id="gsm8k_invalid_raw"),
        pytest.param("", "gsm8k", False, id="gsm8k_empty"),
        pytest.param("Final: \\boxed{100}", "boxed", True, id="boxed_valid"),
        pytest.param("Final: #### 100", "boxed", False, id="boxed_invalid_hash"),
        pytest.param("Final: \\boxed{100}", "any", True, id="any_valid_boxed"),
        pytest.param("Final: #### 100", "any", True, id="any_valid_hash"),
        pytest.param("Final: 100", "any", False, id="any_invalid_plain"),
        pytest.param("Final: #### $18", "gsm8k", True, id="gsm8k_dollar"),
        pytest.param("Final: \\boxed{\\$18}", "boxed", True, id="boxed_dollar"),
        pytest.param(
            "18.\nQuestion: next? #### 3", "gsm8k", False, id="gsm8k_follow_up_only"
        ),
    ],
)
def test_check_format_compliance(text: str, target_format: str, expected: bool) -> None:
    compliance = check_format_compliance(text, target_format=target_format)

    assert compliance is expected


def test_check_format_compliance_unknown_format_raises() -> None:
    text = "#### 42"

    with pytest.raises(ValueError, match="Unknown target_format"):
        check_format_compliance(text, target_format="unsupported_format")


def test_compute_format_compliance_rate_empty_raises() -> None:
    rollouts = []

    with pytest.raises(ValueError):
        compute_format_compliance_rate(rollouts)


def test_compute_format_compliance_rate_trajectories() -> None:
    t1 = RolloutTrajectory(
        request_id="r1",
        prompt="p1",
        prompt_token_ids=[],
        completion_token_ids=[1],
        logprobs=[0.0],
        finish_reason="stop",
        text="Reasoning #### 42",
        worker_id=0,
    )
    t2 = RolloutTrajectory(
        request_id="r2",
        prompt="p2",
        prompt_token_ids=[],
        completion_token_ids=[2],
        logprobs=[0.0],
        finish_reason="stop",
        text="Reasoning \\boxed{42}",
        worker_id=0,
    )
    t3 = RolloutTrajectory(
        request_id="r3",
        prompt="p3",
        prompt_token_ids=[],
        completion_token_ids=[3],
        logprobs=[0.0],
        finish_reason="stop",
        text="Reasoning without tag 42",
        worker_id=0,
    )
    t4 = RolloutTrajectory(
        request_id="r4",
        prompt="p4",
        prompt_token_ids=[],
        completion_token_ids=[4],
        logprobs=[0.0],
        finish_reason="stop",
        text="Reasoning #### 100",
        worker_id=0,
    )

    rate_gsm8k = compute_format_compliance_rate([t1, t2, t3, t4], target_format="gsm8k")
    rate_any = compute_format_compliance_rate([t1, t2, t3, t4], target_format="any")

    assert rate_gsm8k == 0.5
    assert rate_any == 0.75


def test_compute_format_compliance_rate_scored_rollouts() -> None:
    sr_compliant = scorer.ScoredRollout(
        trajectory=RolloutTrajectory(
            request_id="r1",
            prompt="p1",
            prompt_token_ids=[],
            completion_token_ids=[1],
            logprobs=[0.0],
            finish_reason="stop",
            text="Sol: #### 10",
            worker_id=0,
        ),
        reward=1.0,
    )
    sr_non_compliant = scorer.ScoredRollout(
        trajectory=RolloutTrajectory(
            request_id="r2",
            prompt="p2",
            prompt_token_ids=[],
            completion_token_ids=[2],
            logprobs=[0.0],
            finish_reason="stop",
            text="Sol: 10",
            worker_id=0,
        ),
        reward=1.0,
    )

    rate = compute_format_compliance_rate([sr_compliant, sr_non_compliant])

    assert rate == 0.5


def test_compute_format_compliance_rate_strings() -> None:
    raw_strings = ["Solution: #### 10", "Solution: 20"]

    rate = compute_format_compliance_rate(raw_strings)

    assert rate == 0.5


def test_verify_math_answer_malformed_float_fallback() -> None:
    assert verify_math_answer("#### 1.2.3", "#### 4") == 0.0


@pytest.mark.asyncio
async def test_gsm8k_verifier_stripped_prompt_fallback() -> None:
    prompt_unstripped = "  What is 2 + 2?  \n"
    gt_answer = "2 + 2 = 4\n#### 4"
    verifier = GSM8KVerifier({prompt_unstripped.strip(): gt_answer})

    rollout = RolloutTrajectory(
        request_id="r1",
        prompt=prompt_unstripped,
        prompt_token_ids=[1, 2],
        completion_token_ids=[3, 4],
        logprobs=[-0.1, -0.2],
        finish_reason="stop",
        text="The answer is #### 4",
        worker_id=0,
    )
    score = await verifier(rollout)

    assert score == 1.0


@pytest.mark.asyncio
async def test_gsm8k_verifier_unknown_prompt_raises() -> None:
    verifier = GSM8KVerifier({"What is 2 + 2?": "#### 4"})
    rollout = RolloutTrajectory(
        request_id="r1",
        prompt="What is 3 + 3?",
        prompt_token_ids=[1, 2],
        completion_token_ids=[3, 4],
        logprobs=[-0.1, -0.2],
        finish_reason="stop",
        text="#### 6",
        worker_id=0,
    )

    with pytest.raises(KeyError):
        await verifier(rollout)


@pytest.mark.parametrize(
    ("pred", "gt", "expected"),
    [
        pytest.param("#### 42", "#### 42", 1.0, id="exact_match_hash"),
        pytest.param(
            "Result is \\boxed{42.0}", "#### 42", 1.0, id="boxed_float_equivalent"
        ),
        pytest.param("Result is 42", "#### 43", 0.0, id="mismatch"),
        pytest.param("Invalid", "#### 42", 0.0, id="invalid_format"),
        pytest.param("#### $42", "#### 42", 1.0, id="hash_tag_dollar"),
        pytest.param("\\boxed{\\$42.00}", "#### 42", 1.0, id="boxed_dollar"),
        pytest.param("\\boxed{1,250}", "#### 1250", 1.0, id="thousands_separator"),
        pytest.param("\\boxed{5B}", "#### 5", 0.0, id="boxed_glued_suffix"),
        pytest.param(
            "Reasoning.\n#### 72",
            "Natalia sold 48/2 = <<48/2=24>>24 clips.\n#### 72",
            1.0,
            id="gsm8k_reference_annotations",
        ),
    ],
)
def test_verify_math_answer(pred: str, gt: str, expected: float) -> None:
    score = verify_math_answer(pred, gt)

    assert score == expected


@pytest.mark.parametrize(
    ("strict_format", "expected"),
    [
        pytest.param(False, 1.0, id="not strict"),
        pytest.param(True, 0.0, id="strict"),
    ],
)
def test_verify_math_answer_strict_format(strict_format: bool, expected: float) -> None:
    pred_unformatted = "So in total it is 42."
    gt = "#### 42"

    score = verify_math_answer(pred_unformatted, gt, strict_format=strict_format)

    assert score == expected


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("strict_format", "expected_unformatted", "expected_formatted"),
    [
        pytest.param(False, 1.0, 1.0, id="not strict"),
        pytest.param(True, 0.0, 1.0, id="strict"),
    ],
)
async def test_gsm8k_verifier_format_enforcement(
    strict_format: bool, expected_unformatted: float, expected_formatted: float
) -> None:
    prompt = "What is 2 + 2?"
    gt_answer = "2 + 2 = 4\n#### 4"
    verifier = GSM8KVerifier({prompt: gt_answer}, strict_format=strict_format)
    rollout_unformatted = RolloutTrajectory(
        request_id="r1",
        prompt=prompt,
        prompt_token_ids=[1, 2],
        completion_token_ids=[3, 4],
        logprobs=[-0.1, -0.2],
        finish_reason="stop",
        text="The calculation ends with 4.",
        worker_id=0,
    )
    rollout_formatted = RolloutTrajectory(
        request_id="r2",
        prompt=prompt,
        prompt_token_ids=[1, 2],
        completion_token_ids=[3, 4],
        logprobs=[-0.1, -0.2],
        finish_reason="stop",
        text="The calculation ends with #### 4",
        worker_id=0,
    )

    score_unformatted = await verifier(rollout_unformatted)
    score_formatted = await verifier(rollout_formatted)

    assert score_unformatted == expected_unformatted
    assert score_formatted == expected_formatted


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("text", "expected"),
    [
        pytest.param("hello", 1.0, id="non_empty"),
        pytest.param("", 0.0, id="empty"),
    ],
)
async def test_synthetic_verifier(text: str, expected: float) -> None:
    rollout = RolloutTrajectory(
        request_id="r1",
        prompt="test",
        prompt_token_ids=[1],
        completion_token_ids=[2, 3] if text else [],
        logprobs=[-0.1, -0.2] if text else [],
        finish_reason="stop",
        text=text,
        worker_id=0,
    )

    score = await synthetic_verifier(rollout)

    assert score == expected


def _scored(prompt: str, text: str) -> scorer.ScoredRollout:
    return scorer.ScoredRollout(
        trajectory=RolloutTrajectory(
            request_id="r",
            prompt=prompt,
            prompt_token_ids=[],
            completion_token_ids=[1],
            logprobs=[0.0],
            finish_reason="stop",
            text=text,
            worker_id=0,
        ),
        reward=0.0,
    )


@pytest.mark.parametrize(
    ("strict_format", "expected"),
    [
        pytest.param(True, 0.25, id="strict"),
        pytest.param(False, 0.75, id="lenient"),
    ],
)
def test_compute_accuracy(strict_format: bool, expected: float) -> None:
    prompt_to_answer = {"a": "#### 4", "b": "#### 5", "c": "#### 6", "d": "#### 7"}
    rollouts = [
        _scored("a", "#### 4"),
        _scored("b", "so 5"),
        _scored("c", "cut off at 6"),
        _scored("d", "#### 8"),
    ]

    accuracy = verifiers.compute_accuracy(rollouts, prompt_to_answer, strict_format)

    assert accuracy == expected


def test_compute_accuracy_ignores_rollout_reward() -> None:
    rollouts = [_scored("a", "#### 4")]

    accuracy = verifiers.compute_accuracy(rollouts, {"a": "#### 4"}, True)

    assert accuracy == 1.0


def test_compute_accuracy_matches_stripped_prompt() -> None:
    rollouts = [_scored("  a \n", "#### 4")]

    accuracy = verifiers.compute_accuracy(rollouts, {"a": "#### 4"}, True)

    assert accuracy == 1.0


def test_compute_accuracy_empty_raises() -> None:
    rollouts: list[scorer.ScoredRollout] = []

    with pytest.raises(ValueError):
        verifiers.compute_accuracy(rollouts, {}, False)


def test_compute_accuracy_unknown_prompt_raises() -> None:
    rollouts = [_scored("b", "#### 4")]

    with pytest.raises(KeyError):
        verifiers.compute_accuracy(rollouts, {"a": "#### 4"}, False)
