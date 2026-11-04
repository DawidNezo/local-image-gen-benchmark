from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from imagebench.benchmark import run_benchmarks
from imagebench.config import ConfigError


class DummyMonitor:
    def __init__(self, interval: float, device: str, **_kwargs) -> None:
        self.samples = []

    def start(self) -> None:
        return None

    def stop(self) -> None:
        return None


class DummyRunner:
    def __init__(self, device: str) -> None:
        self.device = device
        self.last_load_time_sec = 0.0

    def run(self, model_cfg, test_cfg, seed, output_dir: Path, save_images: bool):
        return 0.5, ""


DUMMY_DEPENDENCIES = {
    "runner_factory": DummyRunner,
    "monitor_factory": DummyMonitor,
    "device_resolver": lambda _requested: "cpu",
}


@pytest.mark.contract
@pytest.mark.integration
def test_run_debug_outputs(minimal_config) -> None:
    run_benchmarks(minimal_config, limit_models=1, limit_tests=1, **DUMMY_DEPENDENCIES)

    payload = json.loads(Path(minimal_config).read_text(encoding="utf-8"))
    output_dir = Path(payload["output_dir"])
    run_dirs = sorted(output_dir.glob("run_*"))
    assert len(run_dirs) == 1

    run_dir = run_dirs[0]
    jsonl_path = run_dir / "results.jsonl"
    csv_path = run_dir / "results.csv"
    summary_path = run_dir / "summary_by_model.csv"

    assert jsonl_path.exists()
    assert csv_path.exists()
    assert summary_path.exists()

    jsonl_lines = jsonl_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(jsonl_lines) == 1
    row = json.loads(jsonl_lines[0])
    assert row["model_name"] == "model-a"
    assert row["test_id"] == "t1"
    assert row["status"] in {"ok", "error"}

    with csv_path.open("r", encoding="utf-8", newline="") as csv_file:
        csv_rows = list(csv.DictReader(csv_file))
    assert len(csv_rows) == 1
    assert csv_rows[0]["model_name"] == "model-a"

    with summary_path.open("r", encoding="utf-8", newline="") as csv_file:
        summary_rows = list(csv.DictReader(csv_file))
    assert len(summary_rows) == 1
    assert summary_rows[0]["model_name"] == "model-a"


@pytest.mark.contract
@pytest.mark.integration
def test_run_overrides_model_path_and_prompt(minimal_config, tmp_path) -> None:
    local_model_dir = tmp_path / "local-model"
    local_model_dir.mkdir()

    run_benchmarks(
        minimal_config,
        limit_models=1,
        limit_tests=1,
        override_model=str(local_model_dir),
        override_prompt="override prompt",
        **DUMMY_DEPENDENCIES,
    )

    payload = json.loads(Path(minimal_config).read_text(encoding="utf-8"))
    output_dir = Path(payload["output_dir"])
    run_dirs = sorted(output_dir.glob("run_*"))
    assert len(run_dirs) == 1

    jsonl_path = run_dirs[0] / "results.jsonl"
    row = json.loads(jsonl_path.read_text(encoding="utf-8").strip())
    assert row["model_id"] == str(local_model_dir)
    assert row["prompt"] == "override prompt"


@pytest.mark.integration
def test_run_returns_config_errors(tmp_path) -> None:
    config_path = tmp_path / "bad.json"
    config_path.write_text(json.dumps({"models": [], "tests": []}), encoding="utf-8")

    with pytest.raises(ConfigError) as excinfo:
        run_benchmarks(config_path, **DUMMY_DEPENDENCIES)

    assert excinfo.value.errors == ["Brak modeli w konfiguracji.", "Brak testów w konfiguracji."]
