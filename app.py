from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from benchmark_runner import (
	DeviceResolutionError,
	ModelRunner,
	list_available_devices,
	resolve_device,
	run_benchmarks as _run_benchmarks,
)
from domain import load_config, validate_config
from outputs import build_markdown_report
from resource_monitor import ResourceMonitor

try:
	import torch
except Exception:  # pragma: no cover - optional dependency at runtime
	torch = None


LOGGER = logging.getLogger("model_benchmark")


def setup_logging(verbose: bool) -> None:
	level = logging.DEBUG if verbose else logging.INFO
	logging.basicConfig(
		level=level,
		format="%(asctime)s | %(levelname)s | %(message)s",
		datefmt="%H:%M:%S",
	)



def _resolve_device_with_fallback(
	requested_device: str,
	*,
	non_interactive: bool,
	assume_yes: bool,
) -> str:
	try:
		return resolve_device(requested_device)
	except DeviceResolutionError as ex:
		if ex.kind != "unavailable":
			raise
		if assume_yes:
			LOGGER.warning("Używam CPU jako fallback dla urządzenia '%s'.", requested_device)
			return resolve_device("cpu")
		if non_interactive or not sys.stdin.isatty():
			raise
		answer = input(
			f"{ex} Czy przejść na CPU? [y/N]: "
		).strip().lower()
		if answer in {"y", "yes", "t", "tak"}:
			LOGGER.warning("Używam CPU jako fallback dla urządzenia '%s'.", requested_device)
			return resolve_device("cpu")
		raise


def run_benchmarks(
	config_path: Path,
	limit_models: int | None,
	limit_tests: int | None,
	*,
	override_device: str | None = None,
	override_model: str | None = None,
	override_prompt: str | None = None,
	non_interactive: bool = False,
	assume_yes: bool = False,
) -> None:
	def monitor_factory(interval: float, device_name: str) -> ResourceMonitor:
		return ResourceMonitor(interval=interval, device=device_name, torch_module=torch)

	def device_resolver(requested: str) -> str:
		return _resolve_device_with_fallback(
			requested,
			non_interactive=non_interactive,
			assume_yes=assume_yes,
		)

	try:
		_run_benchmarks(
			config_path,
			limit_models,
			limit_tests,
			runner_factory=ModelRunner,
			monitor_factory=monitor_factory,
			device_resolver=device_resolver,
			override_device=override_device,
			override_model=override_model,
			override_prompt=override_prompt,
		)
	except DeviceResolutionError as ex:
		LOGGER.error(str(ex))
		raise SystemExit(2)


def build_cli() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(
		description="Aplikacja do lokalnego porównywania modeli generatywnych obrazów"
	)
	parser.add_argument("--verbose", action="store_true", help="Szczegółowe logowanie")

	subparsers = parser.add_subparsers(dest="command", required=True)

	validate_parser = subparsers.add_parser("validate", help="Walidacja pliku konfiguracji")
	validate_parser.add_argument("--config", required=True, type=Path, help="Ścieżka do config.json")

	subparsers.add_parser("devices", help="Lista dostępnych urządzeń")

	run_parser = subparsers.add_parser("run", help="Uruchomienie testów porównawczych")
	run_parser.add_argument("--config", required=True, type=Path, help="Ścieżka do config.json")
	run_parser.add_argument(
		"--device",
		default=None,
		help="Nadpisz urządzenie z konfiguracji (auto, mps, cuda, rocm, cpu)",
	)
	run_parser.add_argument("--model", default=None, help="Nadpisz model (name/model_id/model_path)")
	run_parser.add_argument("--prompt", default=None, help="Nadpisz prompt we wszystkich testach")
	run_parser.add_argument(
		"--limit-models", type=int, default=None, help="Ogranicz liczbę modeli (debug)"
	)
	run_parser.add_argument("--limit-tests", type=int, default=None, help="Ogranicz liczbę testów (debug)")
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

	report_parser = subparsers.add_parser("report", help="Budowa raportu markdown")
	report_parser.add_argument("--input", required=True, type=Path, help="Plik results.jsonl")
	report_parser.add_argument("--output", required=True, type=Path, help="Plik raportu markdown")

	return parser


def main() -> None:
	parser = build_cli()
	args = parser.parse_args()
	setup_logging(args.verbose)

	if args.command == "validate":
		cfg = load_config(args.config)
		errors = validate_config(cfg)
		if errors:
			for error in errors:
				LOGGER.error(error)
			raise SystemExit(2)
		try:
			resolve_device(cfg.device)
		except DeviceResolutionError as ex:
			LOGGER.error(str(ex))
			raise SystemExit(2)
		LOGGER.info("Konfiguracja poprawna.")
		return

	if args.command == "devices":
		devices = list_available_devices()
		LOGGER.info("Dostępne urządzenia: %s", ", ".join(devices))
		if torch is None:
			LOGGER.warning("PyTorch nie jest zainstalowany; lista może być niepełna.")
		return

	if args.command == "run":
		if args.config.exists() is False:
			raise SystemExit(f"Nie znaleziono konfiguracji: {args.config}")
		if args.config.suffix.lower() != ".json":
			raise SystemExit("Obsługiwane są tylko pliki JSON.")
		if args.non_interactive and args.assume_yes:
			raise SystemExit("Nie można użyć jednocześnie --non-interactive i --assume-yes.")
		run_benchmarks(
			args.config,
			args.limit_models,
			args.limit_tests,
			override_device=args.device,
			override_model=args.model,
			override_prompt=args.prompt,
			non_interactive=args.non_interactive,
			assume_yes=args.assume_yes,
		)
		return

	if args.command == "report":
		if not args.input.exists():
			raise SystemExit(f"Nie znaleziono pliku wejściowego: {args.input}")
		build_markdown_report(args.input, args.output)
		return

	raise SystemExit("Nieznana komenda.")


if __name__ == "__main__":
	if "HF_HOME" not in os.environ:
		os.environ["HF_HOME"] = str(Path.cwd() / ".hf_cache")
	main()
