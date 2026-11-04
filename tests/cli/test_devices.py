from __future__ import annotations

import subprocess
import sys

import pytest


@pytest.mark.contract
@pytest.mark.integration
def test_devices_command(repo_root) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "app.py",
            "devices",
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    combined = result.stdout + result.stderr
    assert "Dostępne urządzenia" in combined
