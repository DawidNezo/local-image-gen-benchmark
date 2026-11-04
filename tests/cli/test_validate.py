from __future__ import annotations

import subprocess
import sys

import pytest


@pytest.mark.contract
@pytest.mark.integration
def test_validate_example_config(repo_root) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "app.py",
            "validate",
            "--config",
            "config.example.json",
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
