"""Unit tests for saving full-model checkpoints."""

import copy
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch
from torch import nn
from torch.distributed.device_mesh import DeviceMesh

from vrollout import checkpoint
from vrollout import sharding
from vrollout import testing
from vrollout.types import Rank
from vrollout.types import WorldSize


def _assert_state_dicts_equal(
    actual: dict[str, torch.Tensor], expected: dict[str, torch.Tensor]
) -> None:
    assert actual.keys() == expected.keys()
    for name, tensor in expected.items():
        torch.testing.assert_close(actual[name], tensor)


def test_save_checkpoint_writes_state_dict(tmp_path: Path) -> None:
    model = testing.create_model("test-model")
    path = tmp_path / "step-1"

    saved = checkpoint.save_checkpoint(model, path, rank=0)

    assert saved == path
    loaded = torch.load(path / checkpoint.STATE_DICT_FILENAME, weights_only=True)
    _assert_state_dicts_equal(loaded, model.state_dict())


def test_save_checkpoint_uses_save_pretrained(tmp_path: Path) -> None:
    model = testing.create_model("test-model")
    model.save_pretrained = MagicMock()

    checkpoint.save_checkpoint(model, tmp_path, rank=0)

    model.save_pretrained.assert_called_once()
    args, kwargs = model.save_pretrained.call_args
    assert args == (tmp_path,)
    _assert_state_dicts_equal(kwargs["state_dict"], model.state_dict())


def test_save_checkpoint_skips_non_zero_rank(tmp_path: Path) -> None:
    model = testing.create_model("test-model")
    path = tmp_path / "step-1"

    saved = checkpoint.save_checkpoint(model, path, rank=1)

    assert saved is None
    assert not path.exists()


def _save_sharded_worker(
    rank: Rank,
    world_size: WorldSize,
    mesh: DeviceMesh,
    path: Path,
) -> None:
    _ = world_size  # unused
    torch.manual_seed(0)
    base = testing.create_model("test-model")
    sharded = sharding.shard_model(copy.deepcopy(base), mesh)

    saved = checkpoint.save_checkpoint(sharded, path, rank=rank)

    if rank == 0:
        assert saved == path
        loaded = torch.load(path / checkpoint.STATE_DICT_FILENAME, weights_only=True)
        _assert_state_dicts_equal(loaded, base.state_dict())
    else:
        assert saved is None


def test_save_checkpoint_gathers_fsdp_shards(run_multiprocess, tmp_path: Path) -> None:
    path = tmp_path / "fsdp"

    run_multiprocess(_save_sharded_worker, path)

    assert (path / checkpoint.STATE_DICT_FILENAME).exists()


@pytest.mark.parametrize(
    ("every", "num_iterations", "expected"),
    [
        pytest.param(0, 5, [5], id="final_only"),
        pytest.param(2, 5, [2, 4, 5], id="periodic_plus_final"),
        pytest.param(2, 4, [2, 4], id="final_on_period"),
    ],
)
def test_should_checkpoint(every: int, num_iterations: int, expected: list[int]):
    steps = [
        step
        for step in range(1, num_iterations + 1)
        if checkpoint.should_checkpoint(step, every, num_iterations)
    ]

    assert steps == expected


def test_save_checkpoint_rejects_file_path(tmp_path: Path) -> None:
    model: nn.Module = testing.create_model("test-model")
    path = tmp_path / "file"
    path.write_text("")

    with pytest.raises(FileExistsError):
        checkpoint.save_checkpoint(model, path, rank=0)
