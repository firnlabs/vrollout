"""Integration test placing Ray rollout workers on the GPUs left by trainer ranks.

Starts a real Ray instance with fake GPU IDs in a spawned interpreter, so it stays
isolated from the session-wide Ray instance used by other tests.
"""

from concurrent import futures
import multiprocessing
import os

import ray

from vrollout import placement


@ray.remote(num_gpus=1)
def _assigned_gpus() -> list[str]:
    return [str(i) for i in ray.get_gpu_ids()]


def _place_one_worker(
    visible: str, rollout_gpu_ids: list[str]
) -> tuple[float | None, list[str], str | None]:
    os.environ["CUDA_VISIBLE_DEVICES"] = visible
    placement.init_ray(rollout_gpu_ids)
    try:
        return (
            ray.cluster_resources().get("GPU"),
            ray.get(_assigned_gpus.remote(), timeout=60),
            os.environ.get("CUDA_VISIBLE_DEVICES"),
        )
    finally:
        ray.shutdown()


def test_init_ray_places_workers_on_rollout_gpus() -> None:
    spawn = multiprocessing.get_context("spawn")

    with futures.ProcessPoolExecutor(max_workers=1, mp_context=spawn) as pool:
        cluster_gpus, assigned, driver_env = pool.submit(
            _place_one_worker, "0,1,2", ["2"]
        ).result(timeout=180)

    assert cluster_gpus == 1.0
    assert assigned == ["2"]
    assert driver_env == "0,1,2"
