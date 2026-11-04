from __future__ import annotations

import pytest

import benchmark_runner
from benchmark_runner import (
    DeviceResolutionError,
    list_available_devices,
    resolve_device,
)
from domain import ExperimentConfig, ModelConfig, TestConfig


class FakeMpsBackend:
    def __init__(self, available: bool) -> None:
        self._available = available

    def is_available(self) -> bool:
        return self._available


class FakeBackends:
    def __init__(self, mps_available: bool) -> None:
        self.mps = FakeMpsBackend(mps_available)


class FakeCuda:
    def __init__(self, available: bool) -> None:
        self._available = available

    def is_available(self) -> bool:
        return self._available


class FakeVersion:
    def __init__(self, hip: str | None) -> None:
        self.hip = hip


class FakeTorch:
    def __init__(
        self, mps: bool = False, cuda: bool = False, hip: str | None = None
    ) -> None:
        self.backends = FakeBackends(mps)
        self.cuda = FakeCuda(cuda)
        self.version = FakeVersion(hip)


def test_resolve_auto_prefers_mps(monkeypatch) -> None:
    monkeypatch.setattr(benchmark_runner, "torch", FakeTorch(mps=True, cuda=True))
    assert resolve_device("auto") == "mps"


def test_resolve_auto_falls_back_to_cuda_then_cpu(monkeypatch) -> None:
    monkeypatch.setattr(benchmark_runner, "torch", FakeTorch(cuda=True))
    assert resolve_device("auto") == "cuda"

    monkeypatch.setattr(benchmark_runner, "torch", FakeTorch())
    assert resolve_device("auto") == "cpu"


def test_resolve_auto_detects_rocm(monkeypatch) -> None:
    monkeypatch.setattr(
        benchmark_runner, "torch", FakeTorch(cuda=True, hip="6.2.41133")
    )
    assert resolve_device("auto") == "rocm"


def test_resolve_unavailable_device_raises(monkeypatch) -> None:
    monkeypatch.setattr(benchmark_runner, "torch", FakeTorch())

    with pytest.raises(DeviceResolutionError) as exc_info:
        resolve_device("cuda")
    assert exc_info.value.kind == "unavailable"

    with pytest.raises(DeviceResolutionError) as exc_info:
        resolve_device("mps")
    assert exc_info.value.kind == "unavailable"


def test_resolve_unsupported_device_raises(monkeypatch) -> None:
    monkeypatch.setattr(benchmark_runner, "torch", FakeTorch())

    with pytest.raises(DeviceResolutionError) as exc_info:
        resolve_device("tpu")
    assert exc_info.value.kind == "unsupported"


def test_resolve_without_torch_raises(monkeypatch) -> None:
    monkeypatch.setattr(benchmark_runner, "torch", None)

    with pytest.raises(DeviceResolutionError) as exc_info:
        resolve_device("cpu")
    assert exc_info.value.kind == "missing"


def test_list_available_devices(monkeypatch) -> None:
    monkeypatch.setattr(benchmark_runner, "torch", FakeTorch(mps=True, cuda=False))
    assert list_available_devices() == ["cpu", "mps"]

    monkeypatch.setattr(benchmark_runner, "torch", None)
    assert list_available_devices() == ["cpu"]


def _config_with_models() -> ExperimentConfig:
    return ExperimentConfig(
        device="cpu",
        models=[
            ModelConfig(name="model-a", model_id="org/model-a"),
            ModelConfig(name="model-b", model_id="org/model-b", dtype="bfloat16"),
        ],
        tests=[TestConfig(id="t1", prompt="prompt", seeds=[1])],
    )


def test_apply_overrides_replaces_prompt_and_device() -> None:
    cfg = _config_with_models()

    result = benchmark_runner._apply_overrides(cfg, "mps", None, "nowy prompt")

    assert result.device == "mps"
    assert all(test.prompt == "nowy prompt" for test in result.tests)


def test_apply_overrides_selects_existing_model_by_name() -> None:
    cfg = _config_with_models()

    result = benchmark_runner._apply_overrides(cfg, None, "model-b", None)

    assert len(result.models) == 1
    assert result.models[0].name == "model-b"
    assert result.models[0].dtype == "bfloat16"


def test_apply_overrides_builds_model_from_hub_id() -> None:
    cfg = _config_with_models()

    result = benchmark_runner._apply_overrides(cfg, None, "org/new-model", None)

    assert len(result.models) == 1
    assert result.models[0].name == "new-model"
    assert result.models[0].model_id == "org/new-model"
