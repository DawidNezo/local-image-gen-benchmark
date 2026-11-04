from __future__ import annotations

import subprocess
import sys

import pytest


@pytest.mark.contract
@pytest.mark.integration
def test_report_from_sample_results(repo_root, sample_results_path, tmp_path) -> None:
    output_path = tmp_path / "report.md"
    result = subprocess.run(
        [
            sys.executable,
            "app.py",
            "report",
            "--input",
            str(sample_results_path),
            "--output",
            str(output_path),
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert output_path.exists()

    content = output_path.read_text(encoding="utf-8")
    lines = content.splitlines()
    assert lines[0].startswith("# Raport")
    assert any(line.startswith("Data: ") for line in lines)
    assert "# Raport" in content
    assert "| Model |" in content
    assert "## Podsumowanie" in content
    assert "## Szczeg" in content
    assert any(line.strip().startswith("| sd15 |") for line in lines)
    assert any(line.strip().startswith("| sdxl |") for line in lines)
    assert any("model=sd15" in line for line in lines)
    assert any("status=error" in line for line in lines)
    assert "sd15" in content
    assert "sdxl" in content


@pytest.mark.contract
@pytest.mark.integration
def test_report_from_minimal_results(repo_root, minimal_results_path, tmp_path) -> None:
    output_path = tmp_path / "report.md"
    result = subprocess.run(
        [
            sys.executable,
            "app.py",
            "report",
            "--input",
            str(minimal_results_path),
            "--output",
            str(output_path),
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert output_path.exists()

    content = output_path.read_text(encoding="utf-8")
    lines = content.splitlines()
    assert lines[0].startswith("# Raport")
    assert "## Podsumowanie" in content
    assert "## Szczeg" in content
    assert any(line.strip().startswith("| mini |") for line in lines)
    assert any("model=mini" in line for line in lines)
    assert any("status=error" in line for line in lines)
