"""Rollout pipeline orchestrating generation and concurrent reward scoring."""

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass
import itertools
import logging
from typing import Any

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
        """Generates rollouts and scores them concurrently in a background task."""
        # TODO: check if the prompts list is empty; raise error if it is
        queue = async_queue.AsyncQueue(self._config.queue_maxsize)
        stop_event = asyncio.Event()

        scorer_task = asyncio.create_task(
            scorer.score_rollouts(
                queue,
                self._verifier,
                stop_event,
                batch_size=self._config.scoring_batch_size,
            )
        )

        for chunk in itertools.batched(prompts, self._config.chunk_size):
            rollout_chunk = await self._manager.generate(chunk, sampling_params)
            await queue.put_items(rollout_chunk)

        await queue.join()
        stop_event.set()

        return await scorer_task

    async def generate_and_compute_gae(
        self,
        prompts: Sequence[str],
        sampling_params: types.SamplingParams | None = None,
        # TODO: add values, next_values, device parameters for GAE computation
    ) -> tuple[collate.RolloutBatch, Tensor, Tensor]:
        """Generates batched rollouts, scores them and computes GAE for each batch."""
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
