"""The `run` use case: config -> device -> run directory -> benchmark loop -> result files."""

from __future__ import annotations

import logging
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

from imagebench.config import ConfigError, apply_overrides, load_config, validate_config
from imagebench.metadata import write_run_metadata
from imagebench.orchestrator import MonitorFactory, RunnerFactory, run_benchmark_orchestrator
from imagebench.results import write_outputs

LOGGER = logging.getLogger("model_benchmark")


def run_benchmarks(
    config_path: Path,
    *,
    runner_factory: RunnerFactory,
    monitor_factory: MonitorFactory,
    device_resolver: Callable[[str], str],
    limit_models: int | None = None,
    limit_tests: int | None = None,
    override_device: str | None = None,
    override_model: str | None = None,
    override_prompt: str | None = None,
) -> Path:
    """Run the whole benchmark and return the run directory.

    Raises `ConfigError` for an invalid configuration and lets `DeviceResolutionError` from
    `device_resolver` propagate; turning them into exit codes is the CLI's job.
    """
    cfg = apply_overrides(
        load_config(config_path),
        device=override_device,
        model=override_model,
        prompt=override_prompt,
    )
    errors = validate_config(cfg)
    if errors:
        raise ConfigError(errors)

    device = device_resolver(cfg.device)
    LOGGER.info("Wybrane urządzenie: %s", device)

    run_dir = Path(cfg.output_dir) / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = write_run_metadata(run_dir, device, config_path)
    LOGGER.info("Metadane środowiska: %s", metadata_path)

    rows = run_benchmark_orchestrator(
        cfg,
        device,
        run_dir,
        runner_factory=runner_factory,
        monitor_factory=monitor_factory,
        limit_models=limit_models,
        limit_tests=limit_tests,
    )

    jsonl_path, csv_path, summary_path = write_outputs(rows, run_dir)
    LOGGER.info("Wyniki JSONL: %s", jsonl_path)
    LOGGER.info("Wyniki CSV: %s", csv_path)
    LOGGER.info("Podsumowanie: %s", summary_path)
    return run_dir
