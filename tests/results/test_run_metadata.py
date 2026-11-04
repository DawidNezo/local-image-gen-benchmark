from __future__ import annotations

import json
from pathlib import Path

import pytest

from imagebench.metadata import TRACKED_PACKAGES, collect_environment_metadata, write_run_metadata

pytestmark = pytest.mark.unit


def test_collect_environment_metadata_contains_versions() -> None:
    metadata = collect_environment_metadata("cpu", Path("config.example.json"))

    assert metadata["device"] == "cpu"
    assert metadata["config_path"] == "config.example.json"
    assert metadata["python_version"]
    assert metadata["platform"]
    assert set(TRACKED_PACKAGES) == set(metadata["packages"])


def test_write_run_metadata_creates_json(tmp_path: Path) -> None:
    metadata_path = write_run_metadata(tmp_path, "mps", Path("cfg.json"))

    assert metadata_path == tmp_path / "run_meta.json"
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert payload["device"] == "mps"
    assert payload["config_path"] == "cfg.json"
    assert "packages" in payload
