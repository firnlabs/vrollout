"""Background event loop that generates and scores rollouts off the main thread."""

import asyncio
from collections.abc import Sequence
from concurrent import futures
from dataclasses import dataclass
import threading
import time
import types as builtin_types

from vrollout import pipeline
from vrollout import scorer
from vrollout import types


@dataclass(frozen=True)
class TimedRolloutBatch:
    """Scored rollouts for one batch and how long generating them took.

    Attributes:
        scored_rollouts: Reward-scored rollouts in prompt order.
        generation_time_s: Wall-clock seconds from dispatch to the last score.
    """

    scored_rollouts: list[scorer.ScoredRollout]
    generation_time_s: float


class RolloutRunner:
    """Runs `RolloutPipeline.generate_and_score` on a dedicated event-loop thread."""

    def __init__(self, rollout_pipeline: pipeline.RolloutPipeline) -> None:
        self._pipeline = rollout_pipeline
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._loop.run_forever, name="rollout-runner", daemon=True
        )

    def __enter__(self) -> "RolloutRunner":
        self._thread.start()
        return self

    async def _cancel_all(self) -> None:
        current = asyncio.current_task()
        tasks = [t for t in asyncio.all_tasks() if t is not current]
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: builtin_types.TracebackType | None,
    ) -> None:
        asyncio.run_coroutine_threadsafe(self._cancel_all(), self._loop).result()
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join()
        self._loop.close()

    def submit(
        self,
        prompts: Sequence[str],
        sampling_params: types.SamplingParams,
    ) -> futures.Future[TimedRolloutBatch]:
        """Starts generating and scoring `prompts` in the background.

        Args:
            prompts: Prompts to generate completions for.
            sampling_params: Sampling parameters shared by all prompts.

        Returns:
            A future resolving to the scored batch; it raises whatever
            `generate_and_score` raised.
        """
        return asyncio.run_coroutine_threadsafe(
            self._generate(prompts, sampling_params), self._loop
        )

    async def _generate(
        self, prompts: Sequence[str], sampling_params: types.SamplingParams
    ) -> TimedRolloutBatch:
        t0 = time.perf_counter()
        scored = await self._pipeline.generate_and_score(prompts, sampling_params)
        return TimedRolloutBatch(scored, time.perf_counter() - t0)
