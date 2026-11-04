"""Environment metadata (`run_meta.json`) that makes a benchmark run reproducible."""

from __future__ import annotations

import json
import platform
import sys
from datetime import datetime
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any

TRACKED_PACKAGES = (
    "torch",
    "diffusers",
    "transformers",
    "accelerate",
    "safetensors",
    "psutil",
)


def _package_version(package: str) -> str | None:
    try:
        return importlib_metadata.version(package)
    except importlib_metadata.PackageNotFoundError:
        return None


def collect_environment_metadata(device: str, config_path: Path | None = None) -> dict[str, Any]:
    return {
        "timestamp": datetime.now().isoformat(),
        "device": device,
        "config_path": str(config_path) if config_path else None,
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": {package: _package_version(package) for package in TRACKED_PACKAGES},
    }


def write_run_metadata(run_dir: Path, device: str, config_path: Path | None = None) -> Path:
    metadata_path = run_dir / "run_meta.json"
    metadata = collect_environment_metadata(device, config_path)
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return metadata_path
