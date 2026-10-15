"""Unit tests for reward verifiers."""

import pytest

from vrollout.types import RolloutTrajectory
from vrollout.verifiers import extract_numerical_answer
from vrollout.verifiers import GSM8KVerifier
from vrollout.verifiers import synthetic_verifier
from vrollout.verifiers import verify_math_answer


def test_extract_numerical_answer():
    assert extract_numerical_answer("Some text #### 42") == "42"
    assert extract_numerical_answer("#### -12.5") == "-12.5"
    assert extract_numerical_answer("The result is \\boxed{1,250}") == "1250"
    assert extract_numerical_answer("The answer is 100.") == "100"
    assert extract_numerical_answer("There are 5 apples and 10 oranges.") == "10"
    assert extract_numerical_answer("") is None
    assert extract_numerical_answer("No numbers here") is None


def test_verify_math_answer():
    assert verify_math_answer("#### 42", "#### 42") == 1.0
    assert verify_math_answer("Result is \\boxed{42.0}", "#### 42") == 1.0
    assert verify_math_answer("Result is 42", "The answer is 43") == 0.0
    assert verify_math_answer("Invalid", "#### 42") == 0.0


@pytest.mark.asyncio
async def test_gsm8k_verifier():
    prompt = "What is 2 + 2?"
    gt_answer = "2 + 2 = 4\n#### 4"
    verifier = GSM8KVerifier({prompt: gt_answer})

    rollout_correct = RolloutTrajectory(
        request_id="r1",
        prompt=prompt,
        prompt_token_ids=[1, 2],
        completion_token_ids=[3, 4],
        logprobs=[-0.1, -0.2],
        finish_reason="stop",
        text="The answer is 4.",
        worker_id=0,
    )
    score_correct = await verifier(rollout_correct)
    assert score_correct == 1.0

    rollout_wrong = RolloutTrajectory(
        request_id="r2",
        prompt=prompt,
        prompt_token_ids=[1, 2],
        completion_token_ids=[3, 4],
        logprobs=[-0.1, -0.2],
        finish_reason="stop",
        text="The answer is 5.",
        worker_id=0,
    )
    score_wrong = await verifier(rollout_wrong)
    assert score_wrong == 0.0


@pytest.mark.asyncio
async def test_synthetic_verifier():
    rollout = RolloutTrajectory(
        request_id="r1",
        prompt="test",
        prompt_token_ids=[1],
        completion_token_ids=[2, 3],
        logprobs=[-0.1, -0.2],
        finish_reason="stop",
        text="hello",
        worker_id=0,
    )
    assert await synthetic_verifier(rollout) == 1.0

    empty_rollout = RolloutTrajectory(
        request_id="r2",
        prompt="test",
        prompt_token_ids=[1],
        completion_token_ids=[],
        logprobs=[],
        finish_reason="stop",
        text="",
        worker_id=0,
    )
    assert await synthetic_verifier(empty_rollout) == 0.0
