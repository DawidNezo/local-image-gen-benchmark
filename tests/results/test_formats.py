from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import pytest

from imagebench.results import RunResult, write_outputs

pytestmark = pytest.mark.unit


EXPECTED_FIELDS = [
    "timestamp",
    "model_name",
    "model_id",
    "test_id",
    "prompt",
    "seed",
    "scheduler",
    "steps",
    "guidance_scale",
    "width",
    "height",
    "load_time_sec",
    "duration_sec",
    "throughput_img_s",
    "avg_cpu_percent",
    "max_cpu_percent",
    "avg_ram_percent",
    "max_ram_percent",
    "avg_gpu_util",
    "max_gpu_util",
    "avg_gpu_memory_percent",
    "max_gpu_memory_percent",
    "status",
    "error",
    "image_path",
]

SUMMARY_FIELDS = [
    "model_name",
    "runs",
    "success_runs",
    "avg_duration_sec",
    "avg_throughput_img_s",
    "avg_gpu_util",
    "avg_gpu_memory_percent",
]


def _build_result(
    model_name: str,
    status: str,
    duration: float,
    throughput: float,
    avg_gpu_util: float | None,
    avg_gpu_memory_percent: float | None,
    *,
    error: str = "",
) -> RunResult:
    return RunResult(
        timestamp="2026-05-26T12:00:00",
        model_name=model_name,
        model_id="test/model",
        test_id="t1",
        prompt="test prompt",
        seed=1,
        scheduler="euler_a",
        steps=30,
        guidance_scale=7.5,
        width=512,
        height=512,
        load_time_sec=0.0,
        duration_sec=duration,
        throughput_img_s=throughput,
        avg_cpu_percent=10.0,
        max_cpu_percent=20.0,
        avg_ram_percent=30.0,
        max_ram_percent=40.0,
        avg_gpu_util=avg_gpu_util,
        max_gpu_util=avg_gpu_util,
        avg_gpu_memory_percent=avg_gpu_memory_percent,
        max_gpu_memory_percent=avg_gpu_memory_percent,
        status=status,
        error=error,
        image_path="",
    )


def _assert_decimal(value: str) -> None:
    if value == "":
        return
    assert re.fullmatch(r"-?\d+(\.\d{1,4})?", value)


def test_write_outputs_formats(tmp_path: Path) -> None:
    rows = [
        _build_result(
            model_name="model-b",
            status="ok",
            duration=1.0,
            throughput=0.5,
            avg_gpu_util=15.0,
            avg_gpu_memory_percent=50.0,
        ),
        _build_result(
            model_name="model-a",
            status="ok",
            duration=1.23456,
            throughput=0.33333,
            avg_gpu_util=10.11111,
            avg_gpu_memory_percent=55.12345,
        ),
        _build_result(
            model_name="model-a",
            status="ok",
            duration=2.34567,
            throughput=0.44444,
            avg_gpu_util=20.22222,
            avg_gpu_memory_percent=65.98765,
        ),
        _build_result(
            model_name="model-a",
            status="error",
            duration=9.0,
            throughput=0.0,
            avg_gpu_util=0.0,
            avg_gpu_memory_percent=0.0,
            error="boom",
        ),
    ]

    jsonl_path, csv_path, summary_path = write_outputs(rows, tmp_path)

    json_lines = jsonl_path.read_text(encoding="utf-8").splitlines()
    assert len(json_lines) == len(rows)
    first_keys = list(json.loads(json_lines[0]).keys())
    assert first_keys == EXPECTED_FIELDS

    with csv_path.open("r", encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        assert reader.fieldnames == EXPECTED_FIELDS
        csv_rows = list(reader)

    assert [row["model_name"] for row in csv_rows] == [
        "model-b",
        "model-a",
        "model-a",
        "model-a",
    ]

    with summary_path.open("r", encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        assert reader.fieldnames == SUMMARY_FIELDS
        summary_rows = list(reader)

    assert [row["model_name"] for row in summary_rows] == ["model-b", "model-a"]

    model_b = summary_rows[0]
    assert model_b["runs"] == "1"
    assert model_b["success_runs"] == "1"
    assert model_b["avg_duration_sec"] == "1.0"
    assert model_b["avg_throughput_img_s"] == "0.5"
    assert model_b["avg_gpu_util"] == "15.0"
    assert model_b["avg_gpu_memory_percent"] == "50.0"

    model_a = summary_rows[1]
    assert model_a["runs"] == "3"
    assert model_a["success_runs"] == "2"
    assert model_a["avg_duration_sec"] == "1.7901"
    assert model_a["avg_throughput_img_s"] == "0.3889"
    assert model_a["avg_gpu_util"] == "15.1667"
    # (55.12345 + 65.98765) / 2 = 60.55555, którego reprezentacja binarna leży
    # minimalnie poniżej połowy — round() daje 60.5555, nie 60.5556.
    assert model_a["avg_gpu_memory_percent"] == "60.5555"

    for row in summary_rows:
        _assert_decimal(row["avg_duration_sec"])
        _assert_decimal(row["avg_throughput_img_s"])
        _assert_decimal(row["avg_gpu_util"])
        _assert_decimal(row["avg_gpu_memory_percent"])
