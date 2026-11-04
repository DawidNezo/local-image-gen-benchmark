from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


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


@dataclass
class TestConfig:
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


@dataclass
class ResourceSample:
    timestamp: float
    cpu_percent: float
    ram_percent: float
    gpu_util: float | None
    gpu_memory_percent: float | None


def effective_steps(model: ModelConfig, test: TestConfig) -> int:
    return model.steps if model.steps is not None else test.steps


def effective_guidance_scale(model: ModelConfig, test: TestConfig) -> float:
    return (
        model.guidance_scale if model.guidance_scale is not None else test.guidance_scale
    )


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


def load_config(path: Path) -> ExperimentConfig:
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    models = [ModelConfig(**model) for model in payload.get("models", [])]
    tests = [TestConfig(**test) for test in payload.get("tests", [])]

    return ExperimentConfig(
        output_dir=payload.get("output_dir", "results"),
        device=payload.get("device", "auto"),
        save_images=payload.get("save_images", True),
        models=models,
        tests=tests,
    )


def validate_config(cfg: ExperimentConfig) -> list[str]:
    errors: list[str] = []
    if not cfg.models:
        errors.append("Brak modeli w konfiguracji.")
    if not cfg.tests:
        errors.append("Brak testów w konfiguracji.")

    for model in cfg.models:
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

    for test in cfg.tests:
        if not test.prompt.strip():
            errors.append(f"Test '{test.id}' ma pusty prompt.")
        if test.steps <= 0:
            errors.append(f"Test '{test.id}' ma niepoprawną liczbę kroków.")
        if not test.seeds:
            errors.append(f"Test '{test.id}' nie ma seedów.")

    allowed_devices = {"auto", "mps", "cuda", "rocm", "cpu"}
    if cfg.device.lower() not in allowed_devices:
        errors.append("Pole 'device' musi mieć wartość: auto, mps, cuda, rocm lub cpu.")

    return errors
