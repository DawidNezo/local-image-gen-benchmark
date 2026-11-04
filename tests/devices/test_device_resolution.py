from __future__ import annotations

import pytest

from imagebench import devices
from imagebench.devices import (
    DeviceResolutionError,
    list_available_devices,
    resolve_device,
)

pytestmark = pytest.mark.unit


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
    def __init__(self, mps: bool = False, cuda: bool = False, hip: str | None = None) -> None:
        self.backends = FakeBackends(mps)
        self.cuda = FakeCuda(cuda)
        self.version = FakeVersion(hip)


def test_resolve_auto_prefers_mps(monkeypatch) -> None:
    monkeypatch.setattr(devices, "torch", FakeTorch(mps=True, cuda=True))
    assert resolve_device("auto") == "mps"


def test_resolve_auto_falls_back_to_cuda_then_cpu(monkeypatch) -> None:
    monkeypatch.setattr(devices, "torch", FakeTorch(cuda=True))
    assert resolve_device("auto") == "cuda"

    monkeypatch.setattr(devices, "torch", FakeTorch())
    assert resolve_device("auto") == "cpu"


def test_resolve_auto_detects_rocm(monkeypatch) -> None:
    monkeypatch.setattr(devices, "torch", FakeTorch(cuda=True, hip="6.2.41133"))
    assert resolve_device("auto") == "rocm"


def test_resolve_unavailable_device_raises(monkeypatch) -> None:
    monkeypatch.setattr(devices, "torch", FakeTorch())

    with pytest.raises(DeviceResolutionError) as exc_info:
        resolve_device("cuda")
    assert exc_info.value.kind == "unavailable"

    with pytest.raises(DeviceResolutionError) as exc_info:
        resolve_device("mps")
    assert exc_info.value.kind == "unavailable"


def test_resolve_unsupported_device_raises(monkeypatch) -> None:
    monkeypatch.setattr(devices, "torch", FakeTorch())

    with pytest.raises(DeviceResolutionError) as exc_info:
        resolve_device("tpu")
    assert exc_info.value.kind == "unsupported"


def test_resolve_without_torch_raises(monkeypatch) -> None:
    monkeypatch.setattr(devices, "torch", None)

    with pytest.raises(DeviceResolutionError) as exc_info:
        resolve_device("cpu")
    assert exc_info.value.kind == "missing"


def test_list_available_devices(monkeypatch) -> None:
    monkeypatch.setattr(devices, "torch", FakeTorch(mps=True, cuda=False))
    assert list_available_devices() == ["cpu", "mps"]

    monkeypatch.setattr(devices, "torch", None)
    assert list_available_devices() == ["cpu"]
