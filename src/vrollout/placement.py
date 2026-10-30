"""Single-node GPU placement between trainer ranks and Ray rollout workers."""

from collections.abc import Iterator
from collections.abc import Sequence
from contextlib import contextmanager
import math
import os

_CUDA_VISIBLE_DEVICES = "CUDA_VISIBLE_DEVICES"


def visible_gpu_ids(cuda_visible_devices: str | None, device_count: int) -> list[str]:
    """Returns the GPU IDs this process can see, in CUDA ordinal order.

    IDs are kept as strings because `CUDA_VISIBLE_DEVICES` may hold GPU UUIDs.
    """
    if cuda_visible_devices is None:
        return [str(i) for i in range(device_count)]
    return [d.strip() for d in cuda_visible_devices.split(",") if d.strip()]


def select_rollout_gpus(
    visible: Sequence[str], num_trainer_ranks: int
) -> tuple[list[str], bool]:
    """Returns the rollout GPU IDs and whether they are shared with trainer ranks.

    Trainer rank `r` uses `cuda:r`, i.e. `visible[r]`, so the GPUs after the first
    `num_trainer_ranks` are free for rollouts. With no spare GPU, rollouts share all
    visible GPUs with the trainers (colocated mode).

    Args:
        visible: GPU IDs visible to this process, in CUDA ordinal order.
        num_trainer_ranks: Number of trainer ranks on this node.

    Returns:
        The GPU IDs Ray may place rollout workers on, and True if those GPUs are
        also used by trainer ranks.

    Raises:
        ValueError: If `num_trainer_ranks` is not in [1, len(visible)].
    """
    if not 1 <= num_trainer_ranks <= len(visible):
        raise ValueError(
            f"Cannot place {num_trainer_ranks} trainer ranks on {len(visible)} "
            f"visible GPUs ({','.join(visible) or 'none'})"
        )
    spare = list(visible[num_trainer_ranks:])
    if spare:
        return spare, False
    return list(visible), True


def check_rollout_capacity(
    num_gpus: int, num_workers: int, gpus_per_worker: float
) -> None:
    """Fails fast when Ray could not schedule every rollout worker.

    Ray leaves unschedulable actors pending forever, which hangs training instead of
    erroring. A fractional worker must fit within a single GPU, so e.g. three 0.6-GPU
    workers do not fit on two GPUs even though 1.8 <= 2.

    Args:
        num_gpus: Number of GPUs Ray can place rollout workers on.
        num_workers: Number of rollout workers.
        gpus_per_worker: GPUs requested per worker; 0 means CPU-only workers.

    Raises:
        ValueError: If the workers do not fit on `num_gpus` GPUs.
    """
    if gpus_per_worker <= 0:
        return
    if gpus_per_worker < 1:
        # epsilon guards float error, e.g. 1 / 0.1 must count as 10 workers per GPU
        capacity = num_gpus * math.floor(1 / gpus_per_worker + 1e-9)
    else:
        capacity = math.floor(num_gpus / gpus_per_worker + 1e-9)
    if num_workers > capacity:
        raise ValueError(
            f"{num_workers} rollout workers x {gpus_per_worker} GPUs do not fit on "
            f"{num_gpus} rollout GPUs (capacity {capacity} workers)"
        )


@contextmanager
def cuda_visible_devices(gpu_ids: Sequence[str]) -> Iterator[None]:
    """Temporarily sets `CUDA_VISIBLE_DEVICES`, restoring the previous value on exit."""
    previous = os.environ.get(_CUDA_VISIBLE_DEVICES)
    os.environ[_CUDA_VISIBLE_DEVICES] = ",".join(gpu_ids)
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(_CUDA_VISIBLE_DEVICES, None)
        else:
            os.environ[_CUDA_VISIBLE_DEVICES] = previous


def init_ray(gpu_ids: Sequence[str] | None) -> None:
    """Starts a local Ray instance whose GPU actors only land on `gpu_ids`.

    Ray reads `CUDA_VISIBLE_DEVICES` once at startup and assigns actors from those IDs,
    so restoring the variable afterwards does not affect placement. The caller's own
    CUDA device is already bound and is unaffected. `None` starts Ray without GPUs.
    """
    import ray

    if ray.is_initialized():
        return
    if gpu_ids is None:
        ray.init(include_dashboard=False)
        return
    with cuda_visible_devices(gpu_ids):
        ray.init(include_dashboard=False, num_gpus=len(gpu_ids))
