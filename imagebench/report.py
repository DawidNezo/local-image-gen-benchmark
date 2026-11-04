"""Markdown comparison report built from a `results.jsonl` file."""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from imagebench.results import load_results_jsonl, summarize_by_model

LOGGER = logging.getLogger("model_benchmark")


def _summary_line(summary: dict[str, Any]) -> str:
    def value(key: str) -> float:
        return summary[key] if summary[key] is not None else 0.0

    return (
        f"| {summary['model_name']} | {summary['runs']} | {summary['success_runs']} "
        f"| {value('avg_duration_sec'):.4f} | {value('avg_throughput_img_s'):.4f} "
        f"| {value('avg_gpu_util'):.2f} |"
    )


def _detail_line(row: dict[str, Any]) -> str:
    return (
        "- "
        f"model={row.get('model_name')} test={row.get('test_id')} seed={row.get('seed')} "
        f"status={row.get('status')} time={row.get('duration_sec')}s "
        f"gpu={row.get('avg_gpu_util')}%"
    )


def build_markdown_report(input_path: Path, output_path: Path) -> None:
    rows = load_results_jsonl(input_path)

    lines = [
        "# Raport porównawczy modeli generatywnych",
        "",
        f"Data: {datetime.now().isoformat()}",
        "",
        "## Podsumowanie",
        "",
        "| Model | Uruchomienia | Sukcesy | Śr. czas [s] "
        "| Śr. throughput [img/s] | Śr. GPU util [%] |",
        "|---|---:|---:|---:|---:|---:|",
        *(_summary_line(summary) for summary in summarize_by_model(rows)),
        "",
        "## Szczegóły",
        "",
        *(_detail_line(row) for row in rows),
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    LOGGER.info("Raport zapisany: %s", output_path)
