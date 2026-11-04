"""Compute device detection and resolution (auto, mps, cuda, rocm, cpu)."""

from __future__ import annotations

from typing import Literal

from imagebench._compat import torch

DeviceErrorKind = Literal["missing", "unavailable", "unsupported"]

_UNAVAILABLE_MESSAGES = {
    "mps": "Wybrano 'mps', ale Metal Performance Shaders nie jest dostępne.",
    "cuda": "Wybrano 'cuda', ale CUDA nie jest dostępna.",
    "rocm": "Wybrano 'rocm', ale ROCm nie jest dostępny.",
}


class DeviceResolutionError(RuntimeError):
    """`kind` is "missing" (no PyTorch), "unavailable" (no such hardware) or "unsupported"."""

    def __init__(self, message: str, kind: DeviceErrorKind) -> None:
        super().__init__(message)
        self.kind = kind


def to_torch_device(device: str) -> str:
    """PyTorch exposes ROCm GPUs through the CUDA API."""
    return "cuda" if device == "rocm" else device


def _is_rocm(torch_module) -> bool:
    version = getattr(torch_module, "version", None)
    return bool(getattr(version, "hip", None))


def _mps_available() -> bool:
    return hasattr(torch.backends, "mps") and torch.backends.mps.is_available()


def _gpu_kind() -> str:
    """Name of the GPU behind `torch.cuda`: "rocm" on AMD builds, "cuda" otherwise."""
    return "rocm" if _is_rocm(torch) else "cuda"


def list_available_devices() -> list[str]:
    devices = ["cpu"]
    if torch is None:
        return devices
    if _mps_available():
        devices.append("mps")
    if torch.cuda.is_available():
        devices.append(_gpu_kind())
    return devices


def resolve_device(requested_device: str) -> str:
    """Map a configured device to a usable one; `auto` prefers MPS, then GPU, then CPU."""
    if torch is None:
        raise DeviceResolutionError("PyTorch nie jest zainstalowany.", kind="missing")

    device = requested_device.lower().strip()
    if device == "cpu":
        return "cpu"
    if device == "auto":
        if _mps_available():
            return "mps"
        return _gpu_kind() if torch.cuda.is_available() else "cpu"
    if device == "mps" and _mps_available():
        return "mps"
    if device == "cuda" and torch.cuda.is_available():
        return _gpu_kind()
    if device == "rocm" and torch.cuda.is_available() and _is_rocm(torch):
        return "rocm"
    if device in _UNAVAILABLE_MESSAGES:
        raise DeviceResolutionError(_UNAVAILABLE_MESSAGES[device], kind="unavailable")
    raise DeviceResolutionError(
        "Nieobsługiwane urządzenie. Użyj: auto, mps, cuda, rocm lub cpu.",
        kind="unsupported",
    )
