from __future__ import annotations

import csv
import json
import logging
import platform
import sys
from dataclasses import asdict
from datetime import datetime
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any

from domain import RunResult

LOGGER = logging.getLogger("model_benchmark")

RowLike = RunResult | dict[str, Any]


def _get_field(row: RowLike, key: str) -> Any:
    if isinstance(row, dict):
        return row.get(key)
    return getattr(row, key)


def _format_optional(value: float | None) -> float | str:
    return round(value, 4) if value is not None else ""


TRACKED_PACKAGES = (
    "torch",
    "diffusers",
    "transformers",
    "accelerate",
    "safetensors",
    "psutil",
)


def collect_environment_metadata(device: str, config_path: Path | None = None) -> dict[str, Any]:
    packages: dict[str, str | None] = {}
    for package in TRACKED_PACKAGES:
        try:
            packages[package] = importlib_metadata.version(package)
        except importlib_metadata.PackageNotFoundError:
            packages[package] = None

    return {
        "timestamp": datetime.now().isoformat(),
        "device": device,
        "config_path": str(config_path) if config_path else None,
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": packages,
    }


def write_run_metadata(
    run_dir: Path, device: str, config_path: Path | None = None
) -> Path:
    metadata_path = run_dir / "run_meta.json"
    metadata = collect_environment_metadata(device, config_path)
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return metadata_path


def load_results_jsonl(input_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with input_path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def summarize_by_model(rows: list[RowLike]) -> list[dict[str, Any]]:
    per_model: dict[str, list[RowLike]] = {}
    for row in rows:
        model_name = _get_field(row, "model_name")
        per_model.setdefault(model_name, []).append(row)

    summaries: list[dict[str, Any]] = []
    for model_name, model_rows in per_model.items():
        success = [row for row in model_rows if _get_field(row, "status") == "ok"]

        avg_duration = (
            sum(float(_get_field(row, "duration_sec")) for row in success) / len(success)
            if success
            else None
        )
        avg_throughput = (
            sum(float(_get_field(row, "throughput_img_s")) for row in success) / len(success)
            if success
            else None
        )

        gpu_vals: list[float] = []
        gpu_mem_vals: list[float] = []
        for row in success:
            gpu_util = _get_field(row, "avg_gpu_util")
            if gpu_util is not None:
                gpu_vals.append(float(gpu_util))

            gpu_mem = _get_field(row, "avg_gpu_memory_percent")
            if gpu_mem is not None:
                gpu_mem_vals.append(float(gpu_mem))

        summaries.append(
            {
                "model_name": model_name,
                "runs": len(model_rows),
                "success_runs": len(success),
                "avg_duration_sec": avg_duration,
                "avg_throughput_img_s": avg_throughput,
                "avg_gpu_util": (sum(gpu_vals) / len(gpu_vals)) if gpu_vals else None,
                "avg_gpu_memory_percent": (sum(gpu_mem_vals) / len(gpu_mem_vals)) if gpu_mem_vals else None,
            }
        )

    return summaries


def write_outputs(rows: list[RunResult], output_dir: Path) -> tuple[Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "results.jsonl"
    csv_path = output_dir / "results.csv"
    summary_path = output_dir / "summary_by_model.csv"

    with jsonl_path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(asdict(row), ensure_ascii=False) + "\n")

    if rows:
        fieldnames = list(asdict(rows[0]).keys())
        with csv_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(asdict(row) for row in rows)

    summaries = summarize_by_model(rows)
    with summary_path.open("w", newline="", encoding="utf-8") as file:
        fieldnames = [
            "model_name",
            "runs",
            "success_runs",
            "avg_duration_sec",
            "avg_throughput_img_s",
            "avg_gpu_util",
            "avg_gpu_memory_percent",
        ]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for summary in summaries:
            writer.writerow(
                {
                    "model_name": summary["model_name"],
                    "runs": summary["runs"],
                    "success_runs": summary["success_runs"],
                    "avg_duration_sec": _format_optional(summary["avg_duration_sec"]),
                    "avg_throughput_img_s": _format_optional(summary["avg_throughput_img_s"]),
                    "avg_gpu_util": _format_optional(summary["avg_gpu_util"]),
                    "avg_gpu_memory_percent": _format_optional(summary["avg_gpu_memory_percent"]),
                }
            )

    return jsonl_path, csv_path, summary_path


def build_markdown_report(input_path: Path, output_path: Path) -> None:
    rows = load_results_jsonl(input_path)
    summaries = summarize_by_model(rows)

    lines: list[str] = [
        "# Raport porównawczy modeli generatywnych",
        "",
        f"Data: {datetime.now().isoformat()}",
        "",
        "## Podsumowanie",
        "",
        "| Model | Uruchomienia | Sukcesy | Śr. czas [s] | Śr. throughput [img/s] | Śr. GPU util [%] |",
        "|---|---:|---:|---:|---:|---:|",
    ]

    for summary in summaries:
        avg_time = summary["avg_duration_sec"] if summary["avg_duration_sec"] is not None else 0.0
        avg_thr = (
            summary["avg_throughput_img_s"] if summary["avg_throughput_img_s"] is not None else 0.0
        )
        avg_gpu = summary["avg_gpu_util"] if summary["avg_gpu_util"] is not None else 0.0

        lines.append(
            "| {model} | {runs} | {success} | {avg_time:.4f} | {avg_thr:.4f} | {avg_gpu:.2f} |".format(
                model=summary["model_name"],
                runs=summary["runs"],
                success=summary["success_runs"],
                avg_time=avg_time,
                avg_thr=avg_thr,
                avg_gpu=avg_gpu,
            )
        )

    lines.extend(["", "## Szczegóły", ""])
    for row in rows:
        lines.append(
            "- "
            f"model={row.get('model_name')} test={row.get('test_id')} seed={row.get('seed')} "
            f"status={row.get('status')} time={row.get('duration_sec')}s "
            f"gpu={row.get('avg_gpu_util')}%"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    LOGGER.info("Raport zapisany: %s", output_path)
