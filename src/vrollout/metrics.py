"""Rollout diagnostics for monitoring GRPO training batches."""

from collections.abc import Mapping
from collections.abc import Sequence
from dataclasses import asdict
from dataclasses import astuple
from dataclasses import dataclass
from dataclasses import fields

from jaxtyping import Float
from torch import Tensor

from vrollout import verifiers
from vrollout.scorer import ScoredRollout


def _validate(
    scored_rollouts: Sequence[ScoredRollout],
    advantages: Float[Tensor, "batch"],
    group_size: int,
) -> None:
    if not scored_rollouts:
        raise ValueError("`scored_rollouts` cannot be empty.")
    if advantages.numel() != len(scored_rollouts):
        raise ValueError(
            f"Expected {len(scored_rollouts)} advantages, got {advantages.numel()}"
        )
    if group_size < 1:
        raise ValueError(f"`group_size` must be at least 1, got {group_size}.")


@dataclass(frozen=True, slots=True, kw_only=True)
class RolloutMetrics:
    """Batch-level diagnostics of a GRPO rollout batch.

    Attributes:
        response_length_mean: Mean completion length in tokens over all sequences.
            Tracks how verbose the policy is; steady growth can signal length
            hacking, and a collapse can signal degenerate short answers.
        response_length_max: Longest completion in the batch, in tokens. Compared
            against the sampler's max tokens, it shows how close responses get to
            the generation budget.
        truncation_rate: Fraction of sequences that stopped on the length limit
            (`finish_reason == "length"`). Truncated responses usually cannot be
            scored as correct, so a rising rate wastes samples and biases rewards.
        entropy_proxy: Mean negative log-prob per sampled token, as reported by the
            sampler. Estimates the policy's per-token entropy without an extra
            forward pass, to detect entropy collapse (loss of exploration). vLLM
            reports log-probs before temperature scaling, so at T != 1 it is the
            cross-entropy of tempered samples under the untempered policy. It is
            0.0 when the batch has no completion tokens.
        zero_advantage_group_frac: Fraction of prompt groups whose advantages are
            all zero, i.e. every completion got the same reward. These groups give
            no policy-gradient signal, so a high value means most of the batch is
            wasted (prompts too easy or too hard). `None` when the batch does not
            split into groups of more than one completion.
    """

    response_length_mean: float
    response_length_max: float
    truncation_rate: float
    entropy_proxy: float
    zero_advantage_group_frac: float | None = None

    def to_dict(self) -> dict[str, float]:
        """Returns metric name to value, omitting metrics that are `None`."""
        return {k: v for k, v in asdict(self).items() if v is not None}


def compute_rollout_metrics(
    scored_rollouts: Sequence[ScoredRollout],
    advantages: Float[Tensor, "batch"],
    group_size: int,
) -> RolloutMetrics:
    """Computes batch-level response length, truncation and learning-signal metrics.

    Raises `ValueError` if the batch is empty, `advantages` has the wrong length or
    `group_size` is less than 1.

    Args:
        scored_rollouts: Scored rollouts in batch order, with each prompt's
            `group_size` completions contiguous.
        advantages: Per-sequence advantages used for training, in batch order.
        group_size: Number of completions per prompt.

    Returns:
        The batch's rollout metrics.
    """
    _validate(scored_rollouts, advantages, group_size)

    lengths = [len(sr.trajectory.completion_token_ids) for sr in scored_rollouts]
    total_tokens = sum(lengths)
    total_nll = -sum(sum(sr.trajectory.logprobs) for sr in scored_rollouts)
    num_truncated = sum(
        sr.trajectory.finish_reason == "length" for sr in scored_rollouts
    )

    zero_advantage_group_frac = None
    if group_size > 1 and len(scored_rollouts) % group_size == 0:
        groups = advantages.view(-1, group_size)
        zero_advantage_group_frac = (groups == 0).all(dim=-1).float().mean().item()

    return RolloutMetrics(
        response_length_mean=total_tokens / len(lengths),
        response_length_max=float(max(lengths)),
        truncation_rate=num_truncated / len(lengths),
        entropy_proxy=total_nll / total_tokens if total_tokens else 0.0,
        zero_advantage_group_frac=zero_advantage_group_frac,
    )


@dataclass(frozen=True, slots=True, kw_only=True)
class SequenceRow:
    """Per-sequence diagnostics, logged as one row of a W&B table.

    Logging these rows lets the length bias of token-level loss averaging be checked
    directly: with advantages summing to zero per group, the loss is non-zero only
    when advantage and length are correlated within groups.

    Attributes:
        group: Index of the sequence's prompt group within the batch. Rows sharing a
            group are completions of the same prompt, so within-group comparisons
            of advantage and length isolate the length bias.
        reward: Scalar reward the scorer assigned to the completion.
        advantage: Advantage used for training, i.e. the reward normalized against
            the other completions in its group.
        length: Completion length in tokens. Under token-level loss averaging,
            longer completions contribute more tokens to the loss.
        finish_reason: Why the sampler stopped, e.g. `"stop"` for a natural end or
            `"length"` for truncation at the token limit.
    """

    group: int
    reward: float
    advantage: float
    length: int
    finish_reason: str

    def to_list(self) -> list[int | float | str]:
        """Returns the field values in `SEQUENCE_COLUMNS` order."""
        return list(astuple(self))


SEQUENCE_COLUMNS: tuple[str, ...] = tuple(f.name for f in fields(SequenceRow))


def sequence_rows(
    scored_rollouts: Sequence[ScoredRollout],
    advantages: Float[Tensor, "batch"],
    group_size: int,
) -> list[SequenceRow]:
    """Builds one diagnostics row per sequence.

    Raises `ValueError` if the batch is empty, `advantages` has the wrong length or
    `group_size` is less than 1.

    Args:
        scored_rollouts: Scored rollouts in batch order.
        advantages: Per-sequence advantages used for training, in batch order.
        group_size: Number of completions per prompt.

    Returns:
        One row per sequence, in batch order.
    """
    _validate(scored_rollouts, advantages, group_size)
    return [
        SequenceRow(
            group=i // group_size,
            reward=sr.reward,
            advantage=float(adv),
            length=len(sr.trajectory.completion_token_ids),
            finish_reason=sr.trajectory.finish_reason,
        )
        for i, (sr, adv) in enumerate(zip(scored_rollouts, advantages, strict=True))
    ]


@dataclass(frozen=True, slots=True, kw_only=True)
class EvalRow:
    """One held-out evaluation answer, saved so runs can be compared per problem.

    Attributes:
        prompt: Prompt sent to the sampler; identifies the problem across runs.
        ground_truth: Numerical answer extracted from the reference solution, or
            `None` if the prompt has no known answer.
        ground_truth_raw: Reference solution exactly as stored in the dataset, so
            any verifier can be rerun offline on the same rows without depending
            on the parser that produced `ground_truth`.
        completion: Full completion text, so answer extraction can be audited.
        extracted_answer: Numerical answer the lenient extractor read from the
            completion, or `None` if it found none.
        reward: Reward the verifier assigned to the completion.
        finish_reason: Why the sampler stopped; `"length"` means truncation.
        length: Completion length in tokens.
    """

    prompt: str
    ground_truth: str | None
    ground_truth_raw: str | None
    completion: str
    extracted_answer: str | None
    reward: float
    finish_reason: str
    length: int

    def to_list(self) -> list[str | float | int | None]:
        """Returns the field values in `EVAL_COLUMNS` order."""
        return list(astuple(self))

    def to_dict(self) -> dict[str, str | float | int | None]:
        """Returns field name to value, for JSON serialization."""
        return asdict(self)


EVAL_COLUMNS: tuple[str, ...] = tuple(f.name for f in fields(EvalRow))


def eval_rows(
    scored_rollouts: Sequence[ScoredRollout], prompt_to_answer: Mapping[str, str]
) -> list[EvalRow]:
    """Builds one row per evaluation answer.

    Args:
        scored_rollouts: Scored evaluation rollouts.
        prompt_to_answer: Mapping from prompt to reference solution.

    Returns:
        One row per rollout, in input order.
    """
    rows = []
    for sr in scored_rollouts:
        reference = prompt_to_answer.get(sr.trajectory.prompt)
        rows.append(
            EvalRow(
                prompt=sr.trajectory.prompt,
                ground_truth=(
                    verifiers.extract_numerical_answer(reference)
                    if reference is not None
                    else None
                ),
                ground_truth_raw=reference,
                completion=sr.trajectory.text,
                extracted_answer=verifiers.extract_numerical_answer(sr.trajectory.text),
                reward=sr.reward,
                finish_reason=sr.trajectory.finish_reason,
                length=len(sr.trajectory.completion_token_ids),
            )
        )
    return rows


def eval_length_metrics(rows: Sequence[EvalRow]) -> dict[str, float]:
    """Computes the mean eval answer length and the fraction truncated.

    A truncated answer usually has no final answer tag, so a high truncation rate
    means the eval token budget, not the model, caps accuracy.

    Args:
        rows: Evaluation rows.

    Returns:
        `eval_response_length_mean` (tokens) and `eval_truncation_rate`, or an
        empty dict if there are no rows.
    """
    if not rows:
        return {}
    return {
        "eval_response_length_mean": sum(r.length for r in rows) / len(rows),
        "eval_truncation_rate": sum(r.finish_reason == "length" for r in rows)
        / len(rows),
    }
