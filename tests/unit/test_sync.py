from collections.abc import Mapping
import functools
import os
import tempfile
from typing import Callable

import pytest
from torch import distributed as dist
from torch import multiprocessing as mp
from torch import nn
from torch import Tensor
from torch.distributed import device_mesh
from torch.distributed import fsdp
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import FSDPModule

from vrollout import sync


# TODO: use testing module model classes instead
class ToyBlock(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.ln = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, 4 * d_model)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(4 * d_model, d_model)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.fc2(self.act(self.fc1(self.ln(x))))


class ToyModel(nn.Module):
    def __init__(self, *, d_model: int, num_layers: int, vocab_size: int = 64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([ToyBlock(d_model) for _ in range(num_layers)])
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, idx: Tensor) -> Tensor:
        x = self.embedding(idx)
        for layer in self.layers:
            x = layer(x)
        return self.head(x)


@pytest.fixture
def model_dim() -> int:
    return 4


@pytest.fixture
def model_layers() -> int:
    return 2


@pytest.fixture
def model_factory(model_dim: int, model_layers: int) -> Callable[[], ToyModel]:
    return functools.partial(ToyModel, d_model=model_dim, num_layers=model_layers)


def _shard_model(model: ToyModel, mesh: DeviceMesh):
    for m in model.modules():
        if isinstance(m, ToyBlock) and not isinstance(m, FSDPModule):
            fsdp.fully_shard(m, mesh=mesh)
    fsdp.fully_shard(model, mesh=mesh)
    return model


class DummyWeightsReceiver:
    def __init__(self):
        self.weights = None

    def update_weights(self, weights: Mapping[str, Tensor]) -> None:
        self.weights = weights


def _assert_param_count_equal(
    receiver: DummyWeightsReceiver, ref_model: nn.Module
) -> None:
    assert receiver.weights is not None
    exp_param_count = sum(p.numel() for p in ref_model.parameters())
    param_count = sum(p.numel() for p in receiver.weights.values())
    assert param_count == exp_param_count


def _test_extract_and_broadcast_worker(
    rank: int, world_size: int, init_method: str, model_factory
):
    dist.init_process_group(
        "gloo", init_method=init_method, rank=rank, world_size=world_size
    )
    mesh = device_mesh.init_device_mesh("cpu", (world_size,))
    if rank == 0:
        receiver = DummyWeightsReceiver()
        receivers = [receiver]
    else:
        receiver = None
        receivers = None
    broadcaster = sync.WeightBroadcaster(rank, receivers)
    model = model_factory()
    sharded_model = _shard_model(model, mesh=mesh)

    broadcaster.extract_and_broadcast(sharded_model)

    if rank == 0:
        ref_model = model_factory()
        assert len(broadcaster._receivers) > 0
        assert receiver is not None
        _assert_param_count_equal(receiver, ref_model)
    else:
        assert len(broadcaster._receivers) == 0

    dist.barrier()
    dist.destroy_process_group()


class TestWeightBroadcaster:
    def test_extract_and_broadcast_local(self, model_factory):
        model = model_factory()
        receiver = DummyWeightsReceiver()
        broadcaster = sync.WeightBroadcaster(rank=0, receivers=[receiver])

        broadcaster.extract_and_broadcast(model)

        assert len(broadcaster._receivers) > 0
        _assert_param_count_equal(receiver, model)

    def test_extract_and_broadcast_non_zero_rank(self, model_factory):
        model = model_factory()
        receiver = DummyWeightsReceiver()
        broadcaster = sync.WeightBroadcaster(rank=1, receivers=[receiver])

        broadcaster.extract_and_broadcast(model)

        assert receiver.weights is None

    def test_extract_and_broadcast(self, model_factory):
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        world_size = 2

        try:
            mp.spawn(
                _test_extract_and_broadcast_worker,
                args=(world_size, f"file://{temp_file.name}", model_factory),
                nprocs=world_size,
                join=True,
            )
        finally:
            if os.path.exists(temp_file.name):
                os.remove(temp_file.name)
