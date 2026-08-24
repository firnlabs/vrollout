"""Rollout pipeline orchestrating generation and concurrent reward scoring."""

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass
import logging
from typing import Any

from vrollout import async_queue
from vrollout import scorer
from vrollout import worker

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Configuration parameters for the RolloutPipeline."""

    chunk_size: int = 8
    scoring_batch_size: int = 16
    queue_maxsize: int = 1000


class RolloutPipeline:
    """Rollout pipeline connecting rollout generation with scoring."""

    def __init__(
        self,
        manager: worker.RolloutManager,
        verifier: Any,
        config: PipelineConfig | None = None,
    ) -> None:
        self._manager = manager
        self._verifier = verifier
        self._config = config or PipelineConfig()

    async def generate_and_score(
        self,
        prompts: Sequence[str],
        sampling_params: worker.SamplingParams | None = None,
    ) -> list[scorer.ScoredRollout]:
        """Generates rollouts and scores them concurrently in a background task."""
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
        # FIXME: Should the manager inputs be batched first?
        rollouts = await self._manager.generate(
            prompts, sampling_params, self._config.chunk_size
        )

        await queue.put_items(rollouts)
        await queue.join()
        stop_event.set()
        results = await scorer_task

        return results
