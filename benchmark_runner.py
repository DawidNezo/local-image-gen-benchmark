from __future__ import annotations

import gc
import inspect
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from domain import (
    ModelConfig,
    TestConfig,
    effective_guidance_scale,
    effective_steps,
    load_config,
    validate_config,
)
from orchestrator import run_benchmark_orchestrator
from outputs import write_outputs, write_run_metadata
from resource_monitor import ResourceMonitor

try:
    import torch
except Exception:  # pragma: no cover - optional dependency at runtime
    torch = None

try:
    from diffusers import DiffusionPipeline
except Exception:  # pragma: no cover - optional dependency at runtime
    DiffusionPipeline = None


LOGGER = logging.getLogger("model_benchmark")


class DeviceResolutionError(RuntimeError):
    def __init__(self, message: str, kind: str) -> None:
        super().__init__(message)
        self.kind = kind


class ModelRunner:
    def __init__(self, device: str) -> None:
        self.device = device
        self._loaded: dict[str, Any] = {}
        self.last_load_time_sec: float = 0.0
        self._warned_dropped_kwargs: set[tuple[str, str]] = set()

    @staticmethod
    def _torch_device_name(device: str) -> str:
        return "cuda" if device == "rocm" else device

    @staticmethod
    def _model_source(model_cfg: ModelConfig) -> str:
        if model_cfg.model_id:
            return model_cfg.model_id
        if model_cfg.model_path:
            return model_cfg.model_path
        raise RuntimeError("Model nie ma zdefiniowanego 'model_id' ani 'model_path'.")

    @staticmethod
    def _torch_dtype(dtype_name: str) -> Any:
        if torch is None:
            raise RuntimeError("PyTorch nie jest zainstalowany.")
        mapping = {
            "float16": torch.float16,
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
        }
        return mapping.get(dtype_name, torch.float16)

    def _set_scheduler(self, pipeline: Any, scheduler_name: str) -> None:
        # "default" (lub nazwa spoza mapy) pozostawia natywny scheduler modelu;
        # modele flow-matching (SD3, FLUX) nie współpracują z klasycznymi schedulerami.
        scheduler_map = {
            "ddim": "DDIMScheduler",
            "euler": "EulerDiscreteScheduler",
            "euler_a": "EulerAncestralDiscreteScheduler",
            "dpmpp": "DPMSolverMultistepScheduler",
            "pndm": "PNDMScheduler",
        }
        scheduler_class_name = scheduler_map.get(scheduler_name.lower())
        if not scheduler_class_name:
            return
        try:
            scheduler_class = getattr(
                __import__("diffusers", fromlist=[scheduler_class_name]), scheduler_class_name
            )
            pipeline.scheduler = scheduler_class.from_config(pipeline.scheduler.config)
        except Exception as ex:
            LOGGER.warning("Nie udało się ustawić schedulera '%s': %s", scheduler_name, ex)

    def _release_loaded_models(self) -> None:
        if not self._loaded:
            return
        LOGGER.info("Zwalnianie pamięci po modelach: %s", ", ".join(self._loaded))
        self._loaded.clear()
        gc.collect()
        if torch is None:
            return
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            mps = getattr(torch, "mps", None)
            if mps is not None and hasattr(mps, "empty_cache"):
                mps.empty_cache()
        except Exception:  # pragma: no cover - zależne od platformy
            pass

    def load(self, model_cfg: ModelConfig) -> Any:
        if DiffusionPipeline is None:
            raise RuntimeError("Diffusers nie jest zainstalowany.")
        if model_cfg.name in self._loaded:
            self.last_load_time_sec = 0.0
            return self._loaded[model_cfg.name]

        # Benchmark porównuje modele sekwencyjnie; trzymanie kilku pipeline'ów
        # naraz wyczerpuje pamięć na maszynach z 16 GB RAM/unified memory.
        self._release_loaded_models()

        kwargs: dict[str, Any] = {
            "torch_dtype": self._torch_dtype(model_cfg.dtype),
            "use_safetensors": model_cfg.use_safetensors,
        }
        if model_cfg.variant:
            kwargs["variant"] = model_cfg.variant

        model_source = self._model_source(model_cfg)
        LOGGER.info("Ładowanie modelu '%s' (%s)", model_cfg.name, model_source)
        load_start = time.perf_counter()
        try:
            pipe = DiffusionPipeline.from_pretrained(model_source, **kwargs)
        except Exception as ex:
            raise self._wrap_load_error(model_cfg, model_source, ex) from ex
        pipe.to(self._torch_device_name(self.device))
        if model_cfg.enable_attention_slicing and hasattr(pipe, "enable_attention_slicing"):
            pipe.enable_attention_slicing()
        if model_cfg.enable_vae_tiling and hasattr(pipe, "enable_vae_tiling"):
            pipe.enable_vae_tiling()
        self.last_load_time_sec = time.perf_counter() - load_start
        LOGGER.info(
            "Model '%s' załadowany w %.1f s", model_cfg.name, self.last_load_time_sec
        )
        self._loaded[model_cfg.name] = pipe
        return pipe

    @staticmethod
    def _wrap_load_error(
        model_cfg: ModelConfig, model_source: str, ex: Exception
    ) -> Exception:
        # Modele bramkowane (gated) na Hugging Face zwracają 401/403 lub
        # GatedRepoError, dopóki użytkownik nie zaakceptuje licencji i nie ustawi
        # tokenu. Zamieniamy techniczny wyjątek na czytelną wskazówkę.
        text = f"{type(ex).__name__}: {ex}"
        markers = ("gatedrepo", "401", "403", "unauthorized", "access to model", "awaiting")
        looks_gated = model_cfg.gated or any(m in text.lower() for m in markers)
        if looks_gated:
            return RuntimeError(
                f"Nie udało się pobrać modelu bramkowanego '{model_source}'. "
                "Zaakceptuj licencję na stronie modelu (https://huggingface.co/"
                f"{model_source}) i ustaw token dostępu w zmiennej środowiskowej "
                "HF_TOKEN (albo wykonaj `hf auth login`). Szczegóły: "
                f"{text}"
            )
        return ex

    def _filter_supported_kwargs(
        self, pipe: Any, model_name: str, call_kwargs: dict[str, Any]
    ) -> dict[str, Any]:
        try:
            parameters = inspect.signature(pipe.__call__).parameters
        except (TypeError, ValueError):
            return call_kwargs
        if any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in parameters.values()
        ):
            return call_kwargs
        supported = {key: value for key, value in call_kwargs.items() if key in parameters}
        for dropped in sorted(set(call_kwargs) - set(supported)):
            warn_key = (model_name, dropped)
            if warn_key not in self._warned_dropped_kwargs:
                self._warned_dropped_kwargs.add(warn_key)
                LOGGER.warning(
                    "Pipeline modelu '%s' nie obsługuje parametru '%s' — pomijam.",
                    model_name,
                    dropped,
                )
        return supported

    def run(
        self,
        model_cfg: ModelConfig,
        test_cfg: TestConfig,
        seed: int,
        output_dir: Path,
        save_images: bool,
    ) -> tuple[float, str]:
        pipe = self.load(model_cfg)
        self._set_scheduler(pipe, test_cfg.scheduler)

        if torch is None:
            raise RuntimeError("PyTorch nie jest zainstalowany.")

        generator_device = (
            "cpu" if self.device == "mps" else self._torch_device_name(self.device)
        )
        generator = torch.Generator(device=generator_device).manual_seed(seed)

        call_kwargs: dict[str, Any] = {
            "prompt": test_cfg.prompt,
            "num_inference_steps": effective_steps(model_cfg, test_cfg),
            "guidance_scale": effective_guidance_scale(model_cfg, test_cfg),
            "width": test_cfg.width,
            "height": test_cfg.height,
            "num_images_per_prompt": test_cfg.num_images,
            "generator": generator,
        }
        if test_cfg.negative_prompt:
            call_kwargs["negative_prompt"] = test_cfg.negative_prompt
        call_kwargs = self._filter_supported_kwargs(pipe, model_cfg.name, call_kwargs)

        start_time = time.perf_counter()
        result = pipe(**call_kwargs)
        duration = time.perf_counter() - start_time

        image_path = ""
        if save_images and result.images:
            filename = f"{model_cfg.name}_{test_cfg.id}_seed{seed}.png".replace(" ", "_")
            full_path = output_dir / "images" / filename
            full_path.parent.mkdir(parents=True, exist_ok=True)
            result.images[0].save(full_path)
            image_path = str(full_path)

        return duration, image_path


def _is_rocm(torch_module) -> bool:
    version = getattr(torch_module, "version", None)
    return bool(getattr(version, "hip", None))


def list_available_devices() -> list[str]:
    devices = ["cpu"]
    if torch is None:
        return devices
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        devices.append("mps")
    if torch.cuda.is_available():
        devices.append("rocm" if _is_rocm(torch) else "cuda")
    return devices


def resolve_device(requested_device: str) -> str:
    if torch is None:
        raise DeviceResolutionError("PyTorch nie jest zainstalowany.", kind="missing")

    device = requested_device.lower().strip()
    if device == "auto":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "rocm" if _is_rocm(torch) else "cuda"
        return "cpu"

    if device == "mps":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        raise DeviceResolutionError(
            "Wybrano 'mps', ale Metal Performance Shaders nie jest dostępne.",
            kind="unavailable",
        )

    if device == "cuda":
        if torch.cuda.is_available():
            return "rocm" if _is_rocm(torch) else "cuda"
        raise DeviceResolutionError("Wybrano 'cuda', ale CUDA nie jest dostępna.", kind="unavailable")

    if device == "rocm":
        if torch.cuda.is_available() and _is_rocm(torch):
            return "rocm"
        raise DeviceResolutionError(
            "Wybrano 'rocm', ale ROCm nie jest dostępny.",
            kind="unavailable",
        )

    if device == "cpu":
        return "cpu"

    raise DeviceResolutionError(
        "Nieobsługiwane urządzenie. Użyj: auto, mps, cuda, rocm lub cpu.",
        kind="unsupported",
    )


def _build_model_override(cfg: Any, override_value: str) -> ModelConfig:
    candidate = override_value.strip()
    for model in cfg.models:
        if model.name == candidate:
            return model

    base = cfg.models[0] if cfg.models else None
    defaults: dict[str, Any] = {
        "dtype": base.dtype if base else "float16",
        "variant": base.variant if base else None,
        "use_safetensors": base.use_safetensors if base else True,
    }

    override_path = Path(candidate).expanduser()
    if override_path.exists():
        name = override_path.name or "local-model"
        return ModelConfig(name=name, model_path=str(override_path), **defaults)

    name = candidate.split("/")[-1] or "model"
    return ModelConfig(name=name, model_id=candidate, **defaults)


def _apply_overrides(
    cfg: Any,
    override_device: str | None,
    override_model: str | None,
    override_prompt: str | None,
) -> Any:
    if override_device:
        cfg.device = override_device

    if override_prompt is not None:
        for test in cfg.tests:
            test.prompt = override_prompt

    if override_model:
        cfg.models = [_build_model_override(cfg, override_model)]

    return cfg


def run_benchmarks(
    config_path: Path,
    limit_models: int | None,
    limit_tests: int | None,
    runner_factory: Callable[[str], Any] | None = None,
    monitor_factory: Callable[[float, str], Any] | None = None,
    device_resolver: Callable[[str], str] | None = None,
    override_device: str | None = None,
    override_model: str | None = None,
    override_prompt: str | None = None,
) -> None:
    cfg = load_config(config_path)
    cfg = _apply_overrides(cfg, override_device, override_model, override_prompt)
    errors = validate_config(cfg)
    if errors:
        for error in errors:
            LOGGER.error(error)
        raise SystemExit(2)

    resolve = device_resolver or resolve_device
    device = resolve(cfg.device)
    LOGGER.info("Wybrane urządzenie: %s", device)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(cfg.output_dir) / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = write_run_metadata(run_dir, device, config_path)
    LOGGER.info("Metadane środowiska: %s", metadata_path)

    def on_start(model_name: str, test_id: str, seed: int) -> None:
        LOGGER.info("Start: model=%s test=%s seed=%s", model_name, test_id, seed)

    def on_error(ex: Exception) -> None:
        LOGGER.exception("Błąd uruchomienia testu")

    def default_monitor_factory(interval: float, device_name: str) -> ResourceMonitor:
        return ResourceMonitor(interval=interval, device=device_name, torch_module=torch)

    rows = run_benchmark_orchestrator(
        cfg=cfg,
        device=device,
        run_dir=run_dir,
        limit_models=limit_models,
        limit_tests=limit_tests,
        runner_factory=runner_factory or ModelRunner,
        monitor_factory=monitor_factory or default_monitor_factory,
        on_start=on_start,
        on_error=on_error,
    )

    jsonl_path, csv_path, summary_path = write_outputs(rows, run_dir)
    LOGGER.info("Wyniki JSONL: %s", jsonl_path)
    LOGGER.info("Wyniki CSV: %s", csv_path)
    LOGGER.info("Podsumowanie: %s", summary_path)
