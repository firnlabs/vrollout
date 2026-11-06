from dataclasses import dataclass

import pytest
import torch
from torch.nn import functional as F

from vrollout.testing import MockAsyncLLMEngine


@dataclass
class _SamplingParams:
    temperature: float
    max_tokens: int = 8
    logprobs: int = 1
    group_size: int = 2


def _sampling_log_probs(
    engine: MockAsyncLLMEngine,
    prompt_ids: list[int],
    completion_ids: list[int],
    temperature: float,
) -> torch.Tensor:
    """Log-probs of `completion_ids` under the distribution they were sampled from."""
    with torch.no_grad():
        logits = engine.model(torch.tensor([prompt_ids + completion_ids]))[0]
    next_token_logits = logits[len(prompt_ids) - 1 : -1]
    scale = temperature if temperature > 0 else 1.0
    log_probs = F.log_softmax(next_token_logits / scale, dim=-1)
    return log_probs.gather(-1, torch.tensor(completion_ids).unsqueeze(-1)).squeeze(-1)


@pytest.mark.asyncio
# 5e-5 is just above the mock's greedy cutoff (1e-5), so it is still sampled
@pytest.mark.parametrize("temperature", [0.0, 5e-5, 0.5, 1.0, 2.0])
async def test_generate_reports_temperature_scaled_log_probs(
    temperature: float,
) -> None:
    """Reported log-probs are log softmax(z / T), as in vLLM's processed_logprobs.

    At T = 0 decoding is greedy and, like vLLM, the mock reports log softmax(z).
    """
    torch.manual_seed(0)
    engine = MockAsyncLLMEngine(model_name="test-model")

    output = await anext(
        engine.generate("Hello", _SamplingParams(temperature=temperature), "req-0")
    )

    for completion in output.outputs:
        reported = torch.tensor(
            [
                lp[t]
                for lp, t in zip(completion.logprobs, completion.token_ids, strict=True)
            ]
        )
        expected = _sampling_log_probs(
            engine, output.prompt_token_ids, completion.token_ids, temperature
        )
        torch.testing.assert_close(reported, expected)
        assert completion.cumulative_logprob == pytest.approx(reported.sum().item())
