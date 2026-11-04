from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from domain import (
    ExperimentConfig,
    ResourceSample,
    RunResult,
    effective_guidance_scale,
    effective_steps,
)


def aggregate_samples(samples: list[ResourceSample]) -> dict[str, float | None]:
    if not samples:
        return {
            "avg_cpu": 0.0,
            "max_cpu": 0.0,
            "avg_ram": 0.0,
            "max_ram": 0.0,
            "avg_gpu_util": None,
            "max_gpu_util": None,
            "avg_gpu_mem": None,
            "max_gpu_mem": None,
        }

    cpus = [sample.cpu_percent for sample in samples]
    rams = [sample.ram_percent for sample in samples]
    gpus = [sample.gpu_util for sample in samples if sample.gpu_util is not None]
    gpu_mems = [
        sample.gpu_memory_percent
        for sample in samples
        if sample.gpu_memory_percent is not None
    ]

    return {
        "avg_cpu": sum(cpus) / len(cpus),
        "max_cpu": max(cpus),
        "avg_ram": sum(rams) / len(rams),
        "max_ram": max(rams),
        "avg_gpu_util": (sum(gpus) / len(gpus)) if gpus else None,
        "max_gpu_util": max(gpus) if gpus else None,
        "avg_gpu_mem": (sum(gpu_mems) / len(gpu_mems)) if gpu_mems else None,
        "max_gpu_mem": max(gpu_mems) if gpu_mems else None,
    }


def run_benchmark_orchestrator(
    cfg: ExperimentConfig,
    device: str,
    run_dir: Path,
    limit_models: int | None,
    limit_tests: int | None,
    runner_factory: Callable[[str], Any],
    monitor_factory: Callable[[float, str], Any],
    on_start: Callable[[str, str, int], None] | None = None,
    on_error: Callable[[Exception], None] | None = None,
) -> list[RunResult]:
    models = cfg.models[:limit_models] if limit_models else cfg.models
    tests = cfg.tests[:limit_tests] if limit_tests else cfg.tests

    runner = runner_factory(device)
    rows: list[RunResult] = []

    for model in models:
        for test in tests:
            for seed in test.seeds:
                monitor = monitor_factory(0.5, device)
                status = "ok"
                error = ""
                duration = 0.0
                image_path = ""

                if on_start:
                    on_start(model.name, test.id, seed)

                monitor.start()
                try:
                    duration, image_path = runner.run(
                        model, test, seed, run_dir, cfg.save_images
                    )
                except Exception as ex:
                    status = "error"
                    error = str(ex)
                    if on_error:
                        on_error(ex)
                finally:
                    monitor.stop()

                metrics = aggregate_samples(monitor.samples)
                throughput = (test.num_images / duration) if duration > 0 else 0.0
                load_time = float(getattr(runner, "last_load_time_sec", 0.0) or 0.0)

                model_ref = model.model_id or model.model_path or ""
                rows.append(
                    RunResult(
                        timestamp=datetime.now().isoformat(),
                        model_name=model.name,
                        model_id=model_ref,
                        test_id=test.id,
                        prompt=test.prompt,
                        seed=seed,
                        scheduler=test.scheduler,
                        steps=effective_steps(model, test),
                        guidance_scale=effective_guidance_scale(model, test),
                        width=test.width,
                        height=test.height,
                        load_time_sec=round(load_time, 4),
                        duration_sec=round(duration, 4),
                        throughput_img_s=round(throughput, 4),
                        avg_cpu_percent=round(float(metrics["avg_cpu"] or 0.0), 4),
                        max_cpu_percent=round(float(metrics["max_cpu"] or 0.0), 4),
                        avg_ram_percent=round(float(metrics["avg_ram"] or 0.0), 4),
                        max_ram_percent=round(float(metrics["max_ram"] or 0.0), 4),
                        avg_gpu_util=round(metrics["avg_gpu_util"], 4)
                        if metrics["avg_gpu_util"] is not None
                        else None,
                        max_gpu_util=round(metrics["max_gpu_util"], 4)
                        if metrics["max_gpu_util"] is not None
                        else None,
                        avg_gpu_memory_percent=round(metrics["avg_gpu_mem"], 4)
                        if metrics["avg_gpu_mem"] is not None
                        else None,
                        max_gpu_memory_percent=round(metrics["max_gpu_mem"], 4)
                        if metrics["max_gpu_mem"] is not None
                        else None,
                        status=status,
                        error=error,
                        image_path=image_path,
                    )
                )

    return rows
