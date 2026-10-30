"""Tests for GPU placement between trainer ranks and Ray rollout workers."""

import os
from unittest.mock import MagicMock

import pytest
import ray

from vrollout import placement


@pytest.mark.parametrize(
    ("cuda_visible_devices", "device_count", "expected"),
    [
        pytest.param(None, 3, ["0", "1", "2"], id="unset"),
        pytest.param(None, 0, [], id="no-cuda"),
        pytest.param("4,5,6", 3, ["4", "5", "6"], id="preset-ids"),
        pytest.param(" 1, 2 ", 2, ["1", "2"], id="whitespace"),
        pytest.param("GPU-a1,GPU-b2", 2, ["GPU-a1", "GPU-b2"], id="uuids"),
        pytest.param("", 0, [], id="empty"),
    ],
)
def test_visible_gpu_ids(
    cuda_visible_devices: str | None, device_count: int, expected: list[str]
) -> None:
    ids = placement.visible_gpu_ids(cuda_visible_devices, device_count)

    assert ids == expected


@pytest.mark.parametrize(
    ("visible", "num_trainer_ranks", "expected_ids", "expected_shared"),
    [
        pytest.param(["0", "1", "2"], 2, ["2"], False, id="spare-gpu"),
        pytest.param(["0", "1", "2", "3"], 2, ["2", "3"], False, id="two-spare"),
        pytest.param(["4", "5", "6"], 2, ["6"], False, id="preset-ids"),
        pytest.param(["0", "1"], 2, ["0", "1"], True, id="no-spare-shares-all"),
        pytest.param(["0"], 1, ["0"], True, id="single-gpu"),
    ],
)
def test_select_rollout_gpus(
    visible: list[str],
    num_trainer_ranks: int,
    expected_ids: list[str],
    expected_shared: bool,
) -> None:
    ids, shared = placement.select_rollout_gpus(visible, num_trainer_ranks)

    assert ids == expected_ids
    assert shared == expected_shared


@pytest.mark.parametrize(
    ("visible", "num_trainer_ranks"),
    [
        pytest.param(["0", "1"], 3, id="more-ranks-than-gpus"),
        pytest.param(["0"], 0, id="zero-ranks"),
    ],
)
def test_select_rollout_gpus_rejects_invalid_ranks(
    visible: list[str], num_trainer_ranks: int
) -> None:
    with pytest.raises(ValueError, match="trainer ranks"):
        placement.select_rollout_gpus(visible, num_trainer_ranks)


@pytest.mark.parametrize(
    ("num_gpus", "num_workers", "gpus_per_worker"),
    [
        pytest.param(1, 1, 1.0, id="one-per-gpu"),
        pytest.param(1, 2, 0.5, id="two-halves"),
        pytest.param(1, 10, 0.1, id="ten-tenths"),
        pytest.param(4, 2, 2.0, id="multi-gpu-workers"),
        pytest.param(0, 4, 0.0, id="cpu-workers"),
    ],
)
def test_check_rollout_capacity_fits(
    num_gpus: int, num_workers: int, gpus_per_worker: float
) -> None:
    placement.check_rollout_capacity(num_gpus, num_workers, gpus_per_worker)


@pytest.mark.parametrize(
    ("num_gpus", "num_workers", "gpus_per_worker"),
    [
        pytest.param(1, 2, 1.0, id="too-many-workers"),
        pytest.param(1, 3, 0.4, id="fractional-overflow"),
        pytest.param(2, 3, 0.6, id="fractions-do-not-pack"),
        pytest.param(3, 2, 2.0, id="multi-gpu-overflow"),
        pytest.param(0, 1, 0.5, id="no-gpus"),
    ],
)
def test_check_rollout_capacity_rejects_overflow(
    num_gpus: int, num_workers: int, gpus_per_worker: float
) -> None:
    with pytest.raises(ValueError, match="rollout workers"):
        placement.check_rollout_capacity(num_gpus, num_workers, gpus_per_worker)


def test_plan_rollout_gpus_returns_spare_gpus() -> None:
    ids, shared = placement.plan_rollout_gpus(
        ["0", "1", "2"], num_trainer_ranks=2, num_workers=2, gpus_per_worker=0.5
    )

    assert ids == ["2"]
    assert not shared


@pytest.mark.parametrize(
    ("num_trainer_ranks", "num_workers", "match"),
    [
        pytest.param(4, 1, "trainer ranks", id="too-many-ranks"),
        pytest.param(2, 2, "rollout workers", id="workers-overflow-spare-gpu"),
    ],
)
def test_plan_rollout_gpus_rejects_invalid_layout(
    num_trainer_ranks: int, num_workers: int, match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        placement.plan_rollout_gpus(
            ["0", "1", "2"], num_trainer_ranks, num_workers, gpus_per_worker=1.0
        )


@pytest.mark.parametrize(
    "previous", [pytest.param("0,1,2", id="set"), pytest.param(None, id="unset")]
)
def test_cuda_visible_devices_restores_env(
    monkeypatch: pytest.MonkeyPatch, previous: str | None
) -> None:
    if previous is None:
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    else:
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", previous)

    with placement.cuda_visible_devices(["2", "3"]):
        inside = os.environ.get("CUDA_VISIBLE_DEVICES")

    assert inside == "2,3"
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == previous


def test_cuda_visible_devices_restores_env_on_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2")

    with pytest.raises(RuntimeError), placement.cuda_visible_devices(["2"]):
        raise RuntimeError("ray.init failed")

    assert os.environ["CUDA_VISIBLE_DEVICES"] == "0,1,2"


@pytest.fixture
def mock_ray_init(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Records `ray.init` kwargs and the `CUDA_VISIBLE_DEVICES` seen at call time."""
    monkeypatch.setattr(ray, "is_initialized", lambda: False)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2")
    mock_init = MagicMock(
        side_effect=lambda **_: mock_init.env_at_call.append(
            os.environ.get("CUDA_VISIBLE_DEVICES")
        )
    )
    mock_init.env_at_call = []
    monkeypatch.setattr(ray, "init", mock_init)
    return mock_init


def test_init_ray_restricts_gpus_during_init(mock_ray_init: MagicMock) -> None:
    placement.init_ray(["2"])

    mock_ray_init.assert_called_once_with(include_dashboard=False, num_gpus=1)
    assert mock_ray_init.env_at_call == ["2"]
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "0,1,2"


def test_init_ray_without_gpus(mock_ray_init: MagicMock) -> None:
    placement.init_ray(None)

    mock_ray_init.assert_called_once_with(include_dashboard=False)
    assert mock_ray_init.env_at_call == ["0,1,2"]


def test_init_ray_skips_when_initialized(
    monkeypatch: pytest.MonkeyPatch, mock_ray_init: MagicMock
) -> None:
    monkeypatch.setattr(ray, "is_initialized", lambda: True)

    placement.init_ray(["2"])

    mock_ray_init.assert_not_called()
