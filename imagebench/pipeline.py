"""Hugging Face Diffusers adapter: loads pipelines and generates images for one run."""

from __future__ import annotations

import gc
import importlib
import inspect
import logging
import time
from pathlib import Path
from typing import Any

from imagebench._compat import torch
from imagebench.config import (
    ModelConfig,
    TestConfig,
    effective_guidance_scale,
    effective_steps,
)
from imagebench.devices import to_torch_device

try:
    from diffusers import DiffusionPipeline
except Exception:  # pragma: no cover - optional dependency at runtime
    DiffusionPipeline = None

LOGGER = logging.getLogger("model_benchmark")

# Nazwa z konfiguracji -> klasa schedulera w diffusers. "default" (lub nazwa spoza
# mapy) pozostawia natywny scheduler modelu; modele flow-matching (SD3, FLUX)
# nie współpracują z klasycznymi schedulerami.
SCHEDULERS = {
    "ddim": "DDIMScheduler",
    "euler": "EulerDiscreteScheduler",
    "euler_a": "EulerAncestralDiscreteScheduler",
    "dpmpp": "DPMSolverMultistepScheduler",
    "pndm": "PNDMScheduler",
}

_GATED_ERROR_MARKERS = ("gatedrepo", "401", "403", "unauthorized", "access to model", "awaiting")


class ModelRunner:
    """Keeps at most one pipeline in memory and runs a single generation per `run()` call."""

    def __init__(self, device: str) -> None:
        self.device = device
        self._loaded: dict[str, Any] = {}
        self.last_load_time_sec: float = 0.0
        self._warned_dropped_kwargs: set[tuple[str, str]] = set()

    def run(
        self,
        model_cfg: ModelConfig,
        test_cfg: TestConfig,
        seed: int,
        output_dir: Path,
        save_images: bool,
    ) -> tuple[float, str]:
        """Generate the images for one seed; returns (generation time in s, saved image path)."""
        pipe = self.load(model_cfg)
        self._set_scheduler(pipe, test_cfg.scheduler)

        if torch is None:
            raise RuntimeError("PyTorch nie jest zainstalowany.")

        generator_device = "cpu" if self.device == "mps" else to_torch_device(self.device)
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
            image_path = str(
                self._save_image(result.images[0], model_cfg, test_cfg, seed, output_dir)
            )
        return duration, image_path

    def load(self, model_cfg: ModelConfig) -> Any:
        """Return the pipeline for `model_cfg`, loading it (and setting `last_load_time_sec`)."""
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

        model_source = model_cfg.source
        if not model_source:
            raise RuntimeError("Model nie ma zdefiniowanego 'model_id' ani 'model_path'.")
        LOGGER.info("Ładowanie modelu '%s' (%s)", model_cfg.name, model_source)
        load_start = time.perf_counter()
        try:
            pipe = DiffusionPipeline.from_pretrained(model_source, **kwargs)
        except Exception as ex:
            raise self._wrap_load_error(model_cfg, model_source, ex) from ex
        pipe.to(to_torch_device(self.device))
        if model_cfg.enable_attention_slicing and hasattr(pipe, "enable_attention_slicing"):
            pipe.enable_attention_slicing()
        if model_cfg.enable_vae_tiling and hasattr(pipe, "enable_vae_tiling"):
            pipe.enable_vae_tiling()
        self.last_load_time_sec = time.perf_counter() - load_start
        LOGGER.info("Model '%s' załadowany w %.1f s", model_cfg.name, self.last_load_time_sec)
        self._loaded[model_cfg.name] = pipe
        return pipe

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

    @staticmethod
    def _set_scheduler(pipeline: Any, scheduler_name: str) -> None:
        scheduler_class_name = SCHEDULERS.get(scheduler_name.lower())
        if not scheduler_class_name:
            return
        try:
            scheduler_class = getattr(importlib.import_module("diffusers"), scheduler_class_name)
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

    @staticmethod
    def _wrap_load_error(model_cfg: ModelConfig, model_source: str, ex: Exception) -> Exception:
        # Modele bramkowane (gated) na Hugging Face zwracają 401/403 lub
        # GatedRepoError, dopóki użytkownik nie zaakceptuje licencji i nie ustawi
        # tokenu. Zamieniamy techniczny wyjątek na czytelną wskazówkę.
        text = f"{type(ex).__name__}: {ex}"
        looks_gated = model_cfg.gated or any(m in text.lower() for m in _GATED_ERROR_MARKERS)
        if not looks_gated:
            return ex
        return RuntimeError(
            f"Nie udało się pobrać modelu bramkowanego '{model_source}'. "
            "Zaakceptuj licencję na stronie modelu (https://huggingface.co/"
            f"{model_source}) i ustaw token dostępu w zmiennej środowiskowej "
            "HF_TOKEN (albo wykonaj `hf auth login`). Szczegóły: "
            f"{text}"
        )

    def _filter_supported_kwargs(
        self, pipe: Any, model_name: str, call_kwargs: dict[str, Any]
    ) -> dict[str, Any]:
        """Drop call arguments the pipeline does not accept, warning once per model/argument."""
        try:
            parameters = inspect.signature(pipe.__call__).parameters
        except (TypeError, ValueError):
            return call_kwargs
        if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in parameters.values()):
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

    @staticmethod
    def _save_image(
        image: Any, model_cfg: ModelConfig, test_cfg: TestConfig, seed: int, output_dir: Path
    ) -> Path:
        filename = f"{model_cfg.name}_{test_cfg.id}_seed{seed}.png".replace(" ", "_")
        path = output_dir / "images" / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        image.save(path)
        return path
