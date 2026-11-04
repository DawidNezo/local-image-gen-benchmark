from __future__ import annotations

import json
from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def sample_results_path(repo_root: Path) -> Path:
    return repo_root / "tests" / "fixtures" / "sample_results.jsonl"


@pytest.fixture(scope="session")
def minimal_results_path(repo_root: Path) -> Path:
    return repo_root / "tests" / "fixtures" / "minimal_results.jsonl"


@pytest.fixture()
def minimal_config(tmp_path: Path) -> Path:
    payload = {
        "output_dir": str(tmp_path / "results"),
        "device": "cpu",
        "save_images": False,
        "models": [
            {
                "name": "model-a",
                "model_id": "dummy/model",
            }
        ],
        "tests": [
            {
                "id": "t1",
                "prompt": "test prompt",
                "seeds": [1],
            }
        ],
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(payload), encoding="utf-8")
    return config_path
