"""Command-line interface. This is the only place that wires real dependencies together
and turns errors into exit codes."""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import NoReturn

from imagebench._compat import torch
from imagebench.benchmark import run_benchmarks
from imagebench.config import ConfigError, load_config, validate_config
from imagebench.devices import DeviceResolutionError, list_available_devices, resolve_device
from imagebench.pipeline import ModelRunner
from imagebench.report import build_markdown_report
from imagebench.resource_monitor import ResourceMonitor

LOGGER = logging.getLogger("model_benchmark")

EXIT_INVALID = 2


def setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def _exit_with_errors(errors: Sequence[str]) -> NoReturn:
    for error in errors:
        LOGGER.error(error)
    raise SystemExit(EXIT_INVALID)


def resolve_device_with_fallback(
    requested_device: str,
    *,
    non_interactive: bool,
    assume_yes: bool,
) -> str:
    """Resolve the device; if it is unavailable, offer CPU (or take it with --assume-yes)."""
    try:
        return resolve_device(requested_device)
    except DeviceResolutionError as ex:
        if ex.kind != "unavailable":
            raise
        if not assume_yes:
            if non_interactive or not sys.stdin.isatty():
                raise
            answer = input(f"{ex} Czy przejść na CPU? [y/N]: ").strip().lower()
            if answer not in {"y", "yes", "t", "tak"}:
                raise
        LOGGER.warning("Używam CPU jako fallback dla urządzenia '%s'.", requested_device)
        return resolve_device("cpu")


def _monitor_factory(interval: float, device: str) -> ResourceMonitor:
    return ResourceMonitor(interval=interval, device=device, torch_module=torch)


def _cmd_validate(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    errors = validate_config(cfg)
    if errors:
        _exit_with_errors(errors)
    try:
        resolve_device(cfg.device)
    except DeviceResolutionError as ex:
        LOGGER.error(str(ex))
        raise SystemExit(EXIT_INVALID) from ex
    LOGGER.info("Konfiguracja poprawna.")


def _cmd_devices(_args: argparse.Namespace) -> None:
    LOGGER.info("Dostępne urządzenia: %s", ", ".join(list_available_devices()))
    if torch is None:
        LOGGER.warning("PyTorch nie jest zainstalowany; lista może być niepełna.")


def _cmd_run(args: argparse.Namespace) -> None:
    if not args.config.exists():
        raise SystemExit(f"Nie znaleziono konfiguracji: {args.config}")
    if args.config.suffix.lower() != ".json":
        raise SystemExit("Obsługiwane są tylko pliki JSON.")
    if args.non_interactive and args.assume_yes:
        raise SystemExit("Nie można użyć jednocześnie --non-interactive i --assume-yes.")

    def device_resolver(requested: str) -> str:
        return resolve_device_with_fallback(
            requested, non_interactive=args.non_interactive, assume_yes=args.assume_yes
        )

    try:
        run_benchmarks(
            args.config,
            runner_factory=ModelRunner,
            monitor_factory=_monitor_factory,
            device_resolver=device_resolver,
            limit_models=args.limit_models,
            limit_tests=args.limit_tests,
            override_device=args.device,
            override_model=args.model,
            override_prompt=args.prompt,
        )
    except ConfigError as ex:
        _exit_with_errors(ex.errors)
    except DeviceResolutionError as ex:
        LOGGER.error(str(ex))
        raise SystemExit(EXIT_INVALID) from ex


def _cmd_report(args: argparse.Namespace) -> None:
    if not args.input.exists():
        raise SystemExit(f"Nie znaleziono pliku wejściowego: {args.input}")
    build_markdown_report(args.input, args.output)


def build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="imagebench",
        description="Aplikacja do lokalnego porównywania modeli generatywnych obrazów",
    )
    parser.add_argument("--verbose", action="store_true", help="Szczegółowe logowanie")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_parser = subparsers.add_parser("validate", help="Walidacja pliku konfiguracji")
    validate_parser.add_argument(
        "--config", required=True, type=Path, help="Ścieżka do config.json"
    )
    validate_parser.set_defaults(handler=_cmd_validate)

    devices_parser = subparsers.add_parser("devices", help="Lista dostępnych urządzeń")
    devices_parser.set_defaults(handler=_cmd_devices)

    run_parser = subparsers.add_parser("run", help="Uruchomienie testów porównawczych")
    run_parser.add_argument("--config", required=True, type=Path, help="Ścieżka do config.json")
    run_parser.add_argument(
        "--device",
        default=None,
        help="Nadpisz urządzenie z konfiguracji (auto, mps, cuda, rocm, cpu)",
    )
    run_parser.add_argument(
        "--model", default=None, help="Nadpisz model (name/model_id/model_path)"
    )
    run_parser.add_argument("--prompt", default=None, help="Nadpisz prompt we wszystkich testach")
    run_parser.add_argument(
        "--limit-models", type=int, default=None, help="Ogranicz liczbę modeli (debug)"
    )
    run_parser.add_argument(
        "--limit-tests", type=int, default=None, help="Ogranicz liczbę testów (debug)"
    )
    run_parser.add_argument(
        "--non-interactive",
        action="store_true",
        help="Nie pytaj o fallback urządzenia",
    )
    run_parser.add_argument(
        "--assume-yes",
        action="store_true",
        help="Automatyczny fallback do CPU, jeśli urządzenie jest niedostępne",
    )
    run_parser.set_defaults(handler=_cmd_run)

    report_parser = subparsers.add_parser("report", help="Budowa raportu markdown")
    report_parser.add_argument("--input", required=True, type=Path, help="Plik results.jsonl")
    report_parser.add_argument("--output", required=True, type=Path, help="Plik raportu markdown")
    report_parser.set_defaults(handler=_cmd_report)

    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_cli().parse_args(argv)
    setup_logging(args.verbose)
    args.handler(args)
