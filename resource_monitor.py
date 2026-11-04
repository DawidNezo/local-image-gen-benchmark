from __future__ import annotations

import re
import shutil
import subprocess
import threading
import time
from typing import Protocol

import psutil

from domain import ResourceSample


class GpuAdapter(Protocol):
    def get_stats(self) -> tuple[float | None, float | None]:
        ...


class NullGpuAdapter:
    def get_stats(self) -> tuple[float | None, float | None]:
        return None, None


class MpsGpuAdapter:
    def __init__(self, torch_module) -> None:
        self._torch = torch_module

    def get_stats(self) -> tuple[float | None, float | None]:
        torch_module = self._torch
        if torch_module is None:
            return None, None
        backends = getattr(torch_module, "backends", None)
        if backends is None or not hasattr(backends, "mps"):
            return None, None
        if not backends.mps.is_available():
            return None, None
        mps = getattr(torch_module, "mps", None)
        if mps is None or not hasattr(mps, "current_allocated_memory"):
            return None, None
        try:
            used = float(mps.current_allocated_memory())
            total = float(psutil.virtual_memory().total)
            mem_percent = (used / total * 100.0) if total else None
            return None, mem_percent
        except Exception:
            return None, None


class NvidiaSmiGpuAdapter:
    def __init__(self, runner=None, available: bool | None = None) -> None:
        self._runner = runner or subprocess.check_output
        if available is None:
            self._available = bool(shutil.which("nvidia-smi"))
        else:
            self._available = available

    def get_stats(self) -> tuple[float | None, float | None]:
        if not self._available:
            return None, None
        try:
            command = [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ]
            output = self._runner(command, stderr=subprocess.DEVNULL, text=True).strip()
            if not output:
                return None, None
            util_str, used_str, total_str = [item.strip() for item in output.split(",")[:3]]
            util = float(util_str)
            used = float(used_str)
            total = float(total_str)
            mem_percent = (used / total * 100.0) if total else None
            return util, mem_percent
        except Exception:
            return None, None


class RocmSmiGpuAdapter:
    def __init__(self, runner=None, available: bool | None = None) -> None:
        self._runner = runner or subprocess.check_output
        if available is None:
            self._available = bool(shutil.which("rocm-smi"))
        else:
            self._available = available

    def get_stats(self) -> tuple[float | None, float | None]:
        if not self._available:
            return None, None
        try:
            command = ["rocm-smi", "--showuse", "--showmemuse"]
            output = self._runner(command, stderr=subprocess.DEVNULL, text=True)
            for line in output.splitlines():
                if not line.strip():
                    continue
                if line.lstrip().startswith("GPU") or line.lstrip().startswith("="):
                    continue
                if not line.lstrip()[0].isdigit():
                    continue
                percents = re.findall(r"(\d+(?:\.\d+)?)%", line)
                if len(percents) >= 2:
                    mem_percent = float(percents[-2])
                    util = float(percents[-1])
                    return util, mem_percent
            return None, None
        except Exception:
            return None, None


def build_gpu_adapter(device: str, torch_module=None, subprocess_runner=None) -> GpuAdapter:
    device_name = device.lower().strip()
    if device_name in {"auto", "cpu"}:
        return NullGpuAdapter()
    if device_name == "mps":
        return MpsGpuAdapter(torch_module)
    if device_name == "rocm":
        return RocmSmiGpuAdapter(subprocess_runner)
    if device_name == "null":
        return NullGpuAdapter()
    return NvidiaSmiGpuAdapter(subprocess_runner)


class ResourceMonitor:
    def __init__(
        self,
        interval: float = 0.5,
        device: str = "auto",
        gpu_adapter: GpuAdapter | None = None,
        torch_module=None,
        subprocess_runner=None,
    ) -> None:
        self.interval = interval
        self.device = device
        self.samples: list[ResourceSample] = []
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        if gpu_adapter is None:
            self._gpu_adapter = build_gpu_adapter(device, torch_module, subprocess_runner)
        else:
            self._gpu_adapter = gpu_adapter

    def _sample_loop(self) -> None:
        while not self._stop_event.is_set():
            cpu = psutil.cpu_percent(interval=None)
            ram = psutil.virtual_memory().percent
            gpu_util, gpu_mem = self._gpu_adapter.get_stats()
            self.samples.append(
                ResourceSample(
                    timestamp=time.time(),
                    cpu_percent=cpu,
                    ram_percent=ram,
                    gpu_util=gpu_util,
                    gpu_memory_percent=gpu_mem,
                )
            )
            time.sleep(self.interval)

    def start(self) -> None:
        self.samples.clear()
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
