"""The trainer must score rollouts under the distribution the sampler reports.

GRPO compares three log-probs per token: the sampler's (pi_old), the actor's (pi) and
the reference's (pi_ref). The sampler reports log-probs of softmax(z / T), the
distribution it draws from, so the actor and reference must use it too. With identical
weights and a zero learning rate, all three then agree at any T; a site that ignores T
only shows up at T != 1.

The mock engine stands in for the sampler, so this checks the trainer side (actor,
reference and the engine's temperature wiring) against the sampler's contract. That the
mock follows the contract is checked in `tests/unit/test_mock_llm_engine.py`; that vLLM
does is checked on GPU by `scripts/vllm_smoke_test.py`.
"""

import asyncio
import copy
from dataclasses import dataclass

import pytest
import torch
from torch import nn
from torch import optim

from vrollout import engine
from vrollout import generation
from vrollout import scorer
from vrollout.testing import create_model
from vrollout.testing import MockAsyncLLMEngine
from vrollout.testing import TEST_MODEL_SPECS

_MODEL_NAME = "test-model"
_GROUP_SIZE = 4


@dataclass
class _SamplingParams:
    temperature: float
    max_tokens: int = 8
    logprobs: int = 1
    group_size: int = _GROUP_SIZE


def _sample_scored_rollouts(
    policy: nn.Module, temperature: float
) -> list[scorer.ScoredRollout]:
    """Samples one group per prompt from `policy`, as the rollout workers do."""
    sampler = MockAsyncLLMEngine(
        model=copy.deepcopy(policy),
        vocab_size=TEST_MODEL_SPECS[_MODEL_NAME]["vocab_size"],
        model_name=_MODEL_NAME,
    )
    params = _SamplingParams(temperature=temperature)

    async def _generate(prompt: str, request_id: str):
        output = await anext(sampler.generate(prompt, params, request_id))
        return generation._parse_vllm_output_to_rollouts(output, worker_id=0)

    trajectories = [
        t
        for i, prompt in enumerate(["Hello", "World"])
        for t in asyncio.run(_generate(prompt, f"req-{i}"))
    ]
    # alternate rewards so every group has non-zero advantages
    return [
        scorer.ScoredRollout(reward=float(i % 2), trajectory=t)
        for i, t in enumerate(trajectories)
    ]


@pytest.mark.parametrize("temperature", [0.5, 1.0, 2.0])
def test_trainer_scores_rollouts_under_the_sampling_distribution(
    temperature: float,
) -> None:
    torch.manual_seed(0)
    policy = create_model(_MODEL_NAME)
    scored_rollouts = _sample_scored_rollouts(policy, temperature)
    rl_engine = engine.DistributedRLEngine(
        actor=policy,
        reference=copy.deepcopy(policy),
        optimizer=optim.SGD(policy.parameters(), lr=0.0),
        config=engine.RLEngineConfig(group_size=_GROUP_SIZE, temperature=temperature),
    )

    metrics = rl_engine.train_iteration(scored_rollouts)

    assert metrics.avg_logprob_abs_diff == pytest.approx(0.0, abs=1e-5)
    assert metrics.avg_mismatch_kl == pytest.approx(0.0, abs=1e-6)
    assert metrics.avg_kl == pytest.approx(0.0, abs=1e-6)
    assert metrics.avg_clip_fraction == 0.0
