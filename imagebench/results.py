"""Per-run results: the `RunResult` record, JSONL/CSV output and per-model summaries."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class RunResult:
    timestamp: str
    model_name: str
    model_id: str
    test_id: str
    prompt: str
    seed: int
    scheduler: str
    steps: int
    guidance_scale: float
    width: int
    height: int
    load_time_sec: float
    duration_sec: float
    throughput_img_s: float
    avg_cpu_percent: float
    max_cpu_percent: float
    avg_ram_percent: float
    max_ram_percent: float
    avg_gpu_util: float | None
    max_gpu_util: float | None
    avg_gpu_memory_percent: float | None
    max_gpu_memory_percent: float | None
    status: str
    error: str = ""
    image_path: str = ""


# Rows come either from a live run (RunResult) or from a results.jsonl file (dict).
RowLike = RunResult | dict[str, Any]

SUMMARY_FIELDS = [
    "model_name",
    "runs",
    "success_runs",
    "avg_duration_sec",
    "avg_throughput_img_s",
    "avg_gpu_util",
    "avg_gpu_memory_percent",
]


def _get_field(row: RowLike, key: str) -> Any:
    if isinstance(row, dict):
        return row.get(key)
    return getattr(row, key)


def _floats(rows: list[RowLike], key: str, *, skip_missing: bool = False) -> list[float]:
    values = (_get_field(row, key) for row in rows)
    return [float(v) for v in values if not (skip_missing and v is None)]


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _format_optional(value: float | None) -> float | str:
    return round(value, 4) if value is not None else ""


def load_results_jsonl(input_path: Path) -> list[dict[str, Any]]:
    with input_path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in (raw.strip() for raw in file) if line]


def summarize_by_model(rows: list[RowLike]) -> list[dict[str, Any]]:
    """Aggregate rows per model; averages only use successful runs (None if there are none)."""
    per_model: dict[str, list[RowLike]] = {}
    for row in rows:
        per_model.setdefault(_get_field(row, "model_name"), []).append(row)

    summaries: list[dict[str, Any]] = []
    for model_name, model_rows in per_model.items():
        success = [row for row in model_rows if _get_field(row, "status") == "ok"]
        summaries.append(
            {
                "model_name": model_name,
                "runs": len(model_rows),
                "success_runs": len(success),
                "avg_duration_sec": _mean(_floats(success, "duration_sec")),
                "avg_throughput_img_s": _mean(_floats(success, "throughput_img_s")),
                # GPU metrics are missing on some backends, so only present values count.
                "avg_gpu_util": _mean(_floats(success, "avg_gpu_util", skip_missing=True)),
                "avg_gpu_memory_percent": _mean(
                    _floats(success, "avg_gpu_memory_percent", skip_missing=True)
                ),
            }
        )
    return summaries


def write_outputs(rows: list[RunResult], output_dir: Path) -> tuple[Path, Path, Path]:
    """Write results.jsonl, results.csv (only if there are rows) and summary_by_model.csv."""
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "results.jsonl"
    csv_path = output_dir / "results.csv"
    summary_path = output_dir / "summary_by_model.csv"

    records = [asdict(row) for row in rows]

    with jsonl_path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")

    if records:
        with csv_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)

    with summary_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for summary in summarize_by_model(rows):
            writer.writerow(
                {
                    key: _format_optional(value) if key.startswith("avg_") else value
                    for key, value in summary.items()
                }
            )

    return jsonl_path, csv_path, summary_path
