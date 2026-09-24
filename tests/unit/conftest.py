"""Pytest commons."""

import os
import tempfile
from typing import Any
from typing import Callable
from typing import Concatenate
from typing import ParamSpec

import pytest
from torch import distributed as dist
from torch import multiprocessing as mp
from torch.distributed import device_mesh
from torch.distributed.device_mesh import DeviceMesh

P = ParamSpec("P")
WorldSize = int
Rank = int
MultiProcessFn = Callable[Concatenate[Rank, WorldSize, DeviceMesh, P], None]


def _worker_wrapper(
    rank: Rank,
    world_size: WorldSize,
    init_method: str,
    fn: MultiProcessFn,
    args: tuple[Any, ...],
) -> None:
    try:
        dist.init_process_group(
            backend="gloo",
            init_method=init_method,
            rank=rank,
            world_size=world_size,
        )
        mesh = device_mesh.init_device_mesh("cpu", (world_size,))
        fn(rank, world_size, mesh, *args)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


@pytest.fixture
def world_size() -> WorldSize:
    return 2


@pytest.fixture
def run_multiprocess(
    world_size: WorldSize,
) -> Callable[Concatenate[MultiProcessFn, P], None]:
    def _run_multiprocess(
        fn: MultiProcessFn,
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> None:
        """Multiprocessing harness."""
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        init_method = f"file://{temp_file.name}"
        try:
            mp.spawn(
                _worker_wrapper,
                args=(world_size, init_method, fn, args),
                nprocs=world_size,
                join=True,
            )
        finally:
            if os.path.exists(temp_file.name):
                os.remove(temp_file.name)

    return _run_multiprocess
