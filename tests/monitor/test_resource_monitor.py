from __future__ import annotations

import time

import pytest

from imagebench.resource_monitor import (
    MpsGpuAdapter,
    NullGpuAdapter,
    NvidiaSmiGpuAdapter,
    ResourceMonitor,
    ResourceSample,
    ResourceStats,
    RocmSmiGpuAdapter,
    build_gpu_adapter,
)

pytestmark = pytest.mark.unit


def test_null_adapter_returns_no_stats() -> None:
    assert NullGpuAdapter().get_stats() == (None, None)


def test_nvidia_adapter_parses_smi_output() -> None:
    def fake_runner(command, stderr, text):
        return "55, 4000, 8000"

    adapter = NvidiaSmiGpuAdapter(runner=fake_runner, available=True)
    util, mem = adapter.get_stats()

    assert util == 55.0
    assert mem == 50.0


def test_nvidia_adapter_unavailable_returns_none() -> None:
    adapter = NvidiaSmiGpuAdapter(available=False)
    assert adapter.get_stats() == (None, None)


def test_nvidia_adapter_handles_malformed_output() -> None:
    def fake_runner(command, stderr, text):
        return "garbage"

    adapter = NvidiaSmiGpuAdapter(runner=fake_runner, available=True)
    assert adapter.get_stats() == (None, None)


def test_rocm_adapter_parses_percentages() -> None:
    def fake_runner(command, stderr, text):
        return "==== ROCm ====\nGPU  use  mem\n0    34.0%   55.0%\n"

    adapter = RocmSmiGpuAdapter(runner=fake_runner, available=True)
    util, mem = adapter.get_stats()

    assert util == 55.0
    assert mem == 34.0


def test_rocm_adapter_unavailable_returns_none() -> None:
    adapter = RocmSmiGpuAdapter(available=False)
    assert adapter.get_stats() == (None, None)


def test_mps_adapter_without_torch_returns_none() -> None:
    adapter = MpsGpuAdapter(torch_module=None)
    assert adapter.get_stats() == (None, None)


def test_build_gpu_adapter_mapping() -> None:
    assert isinstance(build_gpu_adapter("cpu"), NullGpuAdapter)
    assert isinstance(build_gpu_adapter("auto"), NullGpuAdapter)
    assert isinstance(build_gpu_adapter("mps"), MpsGpuAdapter)
    assert isinstance(build_gpu_adapter("rocm"), RocmSmiGpuAdapter)
    assert isinstance(build_gpu_adapter("cuda"), NvidiaSmiGpuAdapter)


def test_resource_monitor_collects_samples() -> None:
    monitor = ResourceMonitor(interval=0.01, device="cpu")

    monitor.start()
    time.sleep(0.08)
    monitor.stop()

    assert monitor.samples
    sample = monitor.samples[0]
    assert sample.cpu_percent >= 0.0
    assert sample.ram_percent >= 0.0
    assert sample.gpu_util is None
    assert sample.gpu_memory_percent is None


def test_resource_monitor_start_clears_previous_samples() -> None:
    monitor = ResourceMonitor(interval=0.01, device="cpu")

    monitor.start()
    time.sleep(0.03)
    monitor.stop()
    first_run_count = len(monitor.samples)
    assert first_run_count > 0

    monitor.start()
    monitor.stop()
    assert len(monitor.samples) <= first_run_count


def test_resource_stats_without_samples() -> None:
    stats = ResourceStats.from_samples([])

    assert stats == ResourceStats()
    assert stats.avg_cpu == 0.0
    assert stats.avg_gpu_util is None


def test_resource_stats_skips_missing_gpu_values() -> None:
    samples = [
        ResourceSample(
            0.0, cpu_percent=10.0, ram_percent=50.0, gpu_util=None, gpu_memory_percent=20.0
        ),
        ResourceSample(
            1.0, cpu_percent=30.0, ram_percent=70.0, gpu_util=None, gpu_memory_percent=40.0
        ),
    ]

    stats = ResourceStats.from_samples(samples)

    assert (stats.avg_cpu, stats.max_cpu) == (20.0, 30.0)
    assert (stats.avg_ram, stats.max_ram) == (60.0, 70.0)
    assert (stats.avg_gpu_util, stats.max_gpu_util) == (None, None)
    assert (stats.avg_gpu_mem, stats.max_gpu_mem) == (30.0, 40.0)
