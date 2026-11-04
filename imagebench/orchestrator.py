"""Benchmark loop: runs every model x test x seed and turns each run into a `RunResult`."""

from __future__ import annotations

import logging
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Protocol

from imagebench.config import (
    ExperimentConfig,
    ModelConfig,
    TestConfig,
    effective_guidance_scale,
    effective_steps,
)
from imagebench.resource_monitor import ResourceSample, ResourceStats
from imagebench.results import RunResult

LOGGER = logging.getLogger("model_benchmark")

MONITOR_INTERVAL_SEC = 0.5


class Runner(Protocol):
    """Generates images for one run; `pipeline.ModelRunner` is the real implementation."""

    last_load_time_sec: float

    def run(
        self,
        model_cfg: ModelConfig,
        test_cfg: TestConfig,
        seed: int,
        output_dir: Path,
        save_images: bool,
    ) -> tuple[float, str]: ...


class Monitor(Protocol):
    """Samples resource usage between `start()` and `stop()`."""

    samples: list[ResourceSample]

    def start(self) -> None: ...

    def stop(self) -> None: ...


RunnerFactory = Callable[[str], Runner]
MonitorFactory = Callable[[float, str], Monitor]


def _round(value: float | None) -> float | None:
    return round(value, 4) if value is not None else None


def run_benchmark_orchestrator(
    cfg: ExperimentConfig,
    device: str,
    run_dir: Path,
    runner_factory: RunnerFactory,
    monitor_factory: MonitorFactory,
    limit_models: int | None = None,
    limit_tests: int | None = None,
) -> list[RunResult]:
    models = cfg.models[:limit_models] if limit_models else cfg.models
    tests = cfg.tests[:limit_tests] if limit_tests else cfg.tests

    runner = runner_factory(device)
    return [
        _run_once(
            runner, monitor_factory(MONITOR_INTERVAL_SEC, device), cfg, run_dir, model, test, seed
        )
        for model in models
        for test in tests
        for seed in test.seeds
    ]


def _run_once(
    runner: Runner,
    monitor: Monitor,
    cfg: ExperimentConfig,
    run_dir: Path,
    model: ModelConfig,
    test: TestConfig,
    seed: int,
) -> RunResult:
    """Run one generation under the monitor. Errors are recorded in the result, not raised."""
    status, error, duration, image_path = "ok", "", 0.0, ""

    LOGGER.info("Start: model=%s test=%s seed=%s", model.name, test.id, seed)
    monitor.start()
    try:
        duration, image_path = runner.run(model, test, seed, run_dir, cfg.save_images)
    except Exception as ex:
        status, error = "error", str(ex)
        LOGGER.exception("Błąd uruchomienia testu")
    finally:
        monitor.stop()

    stats = ResourceStats.from_samples(monitor.samples)
    throughput = (test.num_images / duration) if duration > 0 else 0.0

    return RunResult(
        timestamp=datetime.now().isoformat(),
        model_name=model.name,
        model_id=model.source,
        test_id=test.id,
        prompt=test.prompt,
        seed=seed,
        scheduler=test.scheduler,
        steps=effective_steps(model, test),
        guidance_scale=effective_guidance_scale(model, test),
        width=test.width,
        height=test.height,
        load_time_sec=round(float(runner.last_load_time_sec or 0.0), 4),
        duration_sec=round(duration, 4),
        throughput_img_s=round(throughput, 4),
        avg_cpu_percent=round(stats.avg_cpu, 4),
        max_cpu_percent=round(stats.max_cpu, 4),
        avg_ram_percent=round(stats.avg_ram, 4),
        max_ram_percent=round(stats.max_ram, 4),
        avg_gpu_util=_round(stats.avg_gpu_util),
        max_gpu_util=_round(stats.max_gpu_util),
        avg_gpu_memory_percent=_round(stats.avg_gpu_mem),
        max_gpu_memory_percent=_round(stats.max_gpu_mem),
        status=status,
        error=error,
        image_path=image_path,
    )
