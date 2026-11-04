"""Experiment configuration: data model, loading, validation and CLI overrides."""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

SUPPORTED_DEVICES = ("auto", "mps", "cuda", "rocm", "cpu")


@dataclass
class ModelConfig:
    name: str
    model_id: str | None = None
    model_path: str | None = None
    dtype: str = "float16"
    variant: str | None = None
    use_safetensors: bool = True
    # Nadpisania per model: modele destylowane (turbo) wymagają innej liczby
    # kroków i guidance_scale niż modele pełne, więc wspólne wartości z testu
    # nie zawsze są metodologicznie poprawne.
    steps: int | None = None
    guidance_scale: float | None = None
    enable_attention_slicing: bool = False
    enable_vae_tiling: bool = False
    # Model bramkowany na Hugging Face (wymaga akceptacji licencji i tokenu HF).
    # Pole informacyjne — wpływa na komunikat błędu przy nieudanym pobraniu.
    gated: bool = False

    @property
    def source(self) -> str:
        """Hugging Face id or local path the pipeline is loaded from ("" if neither is set)."""
        return self.model_id or self.model_path or ""


@dataclass
class TestConfig:
    __test__ = False  # not a pytest test class

    id: str
    prompt: str
    negative_prompt: str = ""
    width: int = 512
    height: int = 512
    steps: int = 30
    guidance_scale: float = 7.5
    scheduler: str = "euler_a"
    seeds: list[int] = field(default_factory=lambda: [42])
    num_images: int = 1


@dataclass
class ExperimentConfig:
    output_dir: str = "results"
    device: str = "auto"
    save_images: bool = True
    models: list[ModelConfig] = field(default_factory=list)
    tests: list[TestConfig] = field(default_factory=list)


class ConfigError(ValueError):
    """The configuration failed validation; `errors` holds one message per problem."""

    def __init__(self, errors: list[str]) -> None:
        super().__init__("\n".join(errors))
        self.errors = errors


def effective_steps(model: ModelConfig, test: TestConfig) -> int:
    return model.steps if model.steps is not None else test.steps


def effective_guidance_scale(model: ModelConfig, test: TestConfig) -> float:
    return model.guidance_scale if model.guidance_scale is not None else test.guidance_scale


def load_config(path: Path) -> ExperimentConfig:
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    return ExperimentConfig(
        output_dir=payload.get("output_dir", "results"),
        device=payload.get("device", "auto"),
        save_images=payload.get("save_images", True),
        models=[ModelConfig(**model) for model in payload.get("models", [])],
        tests=[TestConfig(**test) for test in payload.get("tests", [])],
    )


def validate_config(cfg: ExperimentConfig) -> list[str]:
    errors: list[str] = []
    if not cfg.models:
        errors.append("Brak modeli w konfiguracji.")
    if not cfg.tests:
        errors.append("Brak testów w konfiguracji.")
    for model in cfg.models:
        errors.extend(_validate_model(model))
    for test in cfg.tests:
        errors.extend(_validate_test(test))
    if cfg.device.lower() not in SUPPORTED_DEVICES:
        errors.append("Pole 'device' musi mieć wartość: auto, mps, cuda, rocm lub cpu.")
    return errors


def _validate_model(model: ModelConfig) -> list[str]:
    errors: list[str] = []
    name = model.name.strip() if model.name else ""
    model_id = model.model_id.strip() if model.model_id else ""
    model_path = model.model_path.strip() if model.model_path else ""
    if not name:
        errors.append("Każdy model musi mieć 'name'.")
    if model_id and model_path:
        errors.append("Model nie może mieć jednocześnie 'model_id' i 'model_path'.")
    if not model_id and not model_path:
        errors.append("Każdy model musi mieć 'model_id' lub 'model_path'.")
    if model.steps is not None and model.steps <= 0:
        errors.append(f"Model '{model.name}' ma niepoprawną liczbę kroków.")
    if model.guidance_scale is not None and model.guidance_scale < 0:
        errors.append(f"Model '{model.name}' ma niepoprawny guidance_scale.")
    return errors


def _validate_test(test: TestConfig) -> list[str]:
    errors: list[str] = []
    if not test.prompt.strip():
        errors.append(f"Test '{test.id}' ma pusty prompt.")
    if test.steps <= 0:
        errors.append(f"Test '{test.id}' ma niepoprawną liczbę kroków.")
    if not test.seeds:
        errors.append(f"Test '{test.id}' nie ma seedów.")
    return errors


def apply_overrides(
    cfg: ExperimentConfig,
    *,
    device: str | None = None,
    model: str | None = None,
    prompt: str | None = None,
) -> ExperimentConfig:
    """Return a copy of `cfg` with the `run --device/--model/--prompt` overrides applied."""
    cfg = copy.deepcopy(cfg)
    if device:
        cfg.device = device
    if prompt is not None:
        for test in cfg.tests:
            test.prompt = prompt
    if model:
        cfg.models = [_model_override(cfg.models, model)]
    return cfg


def _model_override(models: list[ModelConfig], value: str) -> ModelConfig:
    """Pick a configured model by name, or build one from a local path or a Hugging Face id.

    A new model inherits dtype/variant/use_safetensors from the first configured model.
    """
    candidate = value.strip()
    for model in models:
        if model.name == candidate:
            return model

    base = models[0] if models else None
    defaults: dict[str, Any] = {
        "dtype": base.dtype if base else "float16",
        "variant": base.variant if base else None,
        "use_safetensors": base.use_safetensors if base else True,
    }

    path = Path(candidate).expanduser()
    if path.exists():
        return ModelConfig(name=path.name or "local-model", model_path=str(path), **defaults)

    name = candidate.split("/")[-1] or "model"
    return ModelConfig(name=name, model_id=candidate, **defaults)
