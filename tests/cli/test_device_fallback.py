from __future__ import annotations

import pytest

import app
from benchmark_runner import DeviceResolutionError


def _raise_unavailable(requested: str) -> str:
    if requested == "cpu":
        return "cpu"
    raise DeviceResolutionError(f"Urządzenie '{requested}' niedostępne.", kind="unavailable")


def test_fallback_assume_yes_switches_to_cpu(monkeypatch) -> None:
    monkeypatch.setattr(app, "resolve_device", _raise_unavailable)

    device = app._resolve_device_with_fallback(
        "cuda", non_interactive=False, assume_yes=True
    )

    assert device == "cpu"


def test_fallback_non_interactive_raises(monkeypatch) -> None:
    monkeypatch.setattr(app, "resolve_device", _raise_unavailable)

    with pytest.raises(DeviceResolutionError):
        app._resolve_device_with_fallback(
            "cuda", non_interactive=True, assume_yes=False
        )


def test_fallback_does_not_catch_other_errors(monkeypatch) -> None:
    def raise_missing(requested: str) -> str:
        raise DeviceResolutionError("PyTorch nie jest zainstalowany.", kind="missing")

    monkeypatch.setattr(app, "resolve_device", raise_missing)

    with pytest.raises(DeviceResolutionError):
        app._resolve_device_with_fallback(
            "cuda", non_interactive=False, assume_yes=True
        )


def test_cli_parses_run_arguments() -> None:
    parser = app.build_cli()
    args = parser.parse_args(
        [
            "run",
            "--config",
            "config.example.json",
            "--device",
            "cpu",
            "--limit-models",
            "1",
            "--limit-tests",
            "2",
            "--assume-yes",
        ]
    )

    assert args.command == "run"
    assert str(args.config) == "config.example.json"
    assert args.device == "cpu"
    assert args.limit_models == 1
    assert args.limit_tests == 2
    assert args.assume_yes is True
    assert args.non_interactive is False


def test_cli_requires_command() -> None:
    parser = app.build_cli()
    with pytest.raises(SystemExit):
        parser.parse_args([])
