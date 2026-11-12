"""Rollout pipeline orchestrating generation and concurrent reward scoring."""

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass
import logging
from typing import Any

from beartype import beartype
from jaxtyping import Float
from torch import Tensor

from vrollout import async_queue
from vrollout import collate
from vrollout import gae
from vrollout import generation
from vrollout import scorer
from vrollout import types

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Configuration parameters for the RolloutPipeline."""

    chunk_size: int = 8
    scoring_batch_size: int = 16
    queue_maxsize: int = 1000
    gae_gamma: float = 0.99
    gae_lambda: float = 0.95


class RolloutPipeline:
    """Rollout pipeline connecting rollout generation with scoring."""

    def __init__(
        self,
        manager: generation.RolloutManager,
        verifier: Any,
        config: PipelineConfig | None = None,
    ) -> None:
        self._manager = manager
        self._verifier = verifier
        self._config = config or PipelineConfig()

    async def generate_and_score(
        self,
        prompts: Sequence[str],
        sampling_params: types.SamplingParams | None = None,
    ) -> list[scorer.ScoredRollout]:
        """Generates rollouts and scores them concurrently via an asynchronous queue.

        Streams chunks of generated rollouts into an `AsyncQueue` consumed by a
        background scoring worker, returning reward-annotated trajectories. Empty
        `prompts` returns an empty list.

        If a chunk or the verifier raises, or the caller cancels this coroutine, the
        chunks still in flight are cancelled and the scorer is stopped before the error
        propagates, so no generation keeps holding engine capacity.
        """
        # awaiting in submission order keeps each prompt's group_size completions
        # contiguous and the batch order deterministic, while all chunks generate
        # concurrently
        chunk_refs = self._manager.submit(
            prompts, sampling_params, self._config.chunk_size
        )

        queue = async_queue.AsyncQueue(self._config.queue_maxsize)
        stop_event = asyncio.Event()

        async def produce() -> None:
            for ref in chunk_refs:
                await queue.put_items(await ref)
            await queue.join()

        scorer_task = asyncio.create_task(
            scorer.score_rollouts(
                queue,
                self._verifier,
                stop_event,
                batch_size=self._config.scoring_batch_size,
            )
        )
        producer_task = asyncio.create_task(produce())
        try:
            # if the scorer fails, nothing drains the queue and the producer would
            # block forever in put_items or join
            await asyncio.wait(
                {producer_task, scorer_task}, return_when=asyncio.FIRST_COMPLETED
            )
            if scorer_task.done() and not producer_task.done():
                # the scorer only exits before stop_event is set by raising
                exc = scorer_task.exception()
                assert exc is not None
                raise exc
            await producer_task
        except BaseException:
            # awaiting a Ray ObjectRef doesn't cancel the remote task, so cancel the
            # chunks explicitly; vLLM aborts a request whose generator is cancelled
            self._manager.cancel(chunk_refs)
            stop_event.set()
            producer_task.cancel()
            scorer_task.cancel()
            await asyncio.gather(producer_task, scorer_task, return_exceptions=True)
            raise

        stop_event.set()
        return await scorer_task

    @beartype
    async def generate_and_compute_gae(
        self,
        prompts: Sequence[str],
        sampling_params: types.SamplingParams | None = None,
        # TODO: add values, next_values, device parameters for GAE computation
    ) -> tuple[
        collate.RolloutBatch,
        Float[Tensor, "batch completion_len"],
        Float[Tensor, "batch completion_len"],
    ]:
        """Generates reward-scored rollouts and computes GAE advantages.

        Returns a tuple of `(rollout_batch, advantages, target_values)` where:
            - rollout_batch: Collated batch of rewards, values, dones, and token masks.
            - advantages: Masked Generalized Advantage Estimation tensor.
            - target_values: TD(lambda) return targets (`advantages + values`).
        """
        scored_rollouts = await self.generate_and_score(prompts, sampling_params)

        # Collate tensors for GAE computation
        rollout_batch = collate.collate_rollouts_for_gae(scored_rollouts)

        advantages, target_values = gae.compute_gae(
            rewards=rollout_batch.rewards,
            values=rollout_batch.values,
            next_values=rollout_batch.next_values,
            dones=rollout_batch.dones,
            seq_lens=rollout_batch.seq_lens,
            gamma=self._config.gae_gamma,
            lam=self._config.gae_lambda,
        )

        return rollout_batch, advantages, target_values

    @property
    def manager(self) -> generation.RolloutManager:
        return self._manager
