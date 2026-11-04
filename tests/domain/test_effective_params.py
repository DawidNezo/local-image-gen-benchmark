from __future__ import annotations

from domain import (
    ExperimentConfig,
    ModelConfig,
    TestConfig,
    effective_guidance_scale,
    effective_steps,
    validate_config,
)


def _test_cfg(steps: int = 30, guidance_scale: float = 7.5) -> TestConfig:
    return TestConfig(
        id="t1", prompt="test prompt", seeds=[1], steps=steps, guidance_scale=guidance_scale
    )


def test_effective_params_default_to_test_values() -> None:
    model = ModelConfig(name="full", model_id="test/model")
    test = _test_cfg(steps=28, guidance_scale=4.5)

    assert effective_steps(model, test) == 28
    assert effective_guidance_scale(model, test) == 4.5


def test_effective_params_prefer_model_overrides() -> None:
    model = ModelConfig(name="turbo", model_id="test/turbo", steps=4, guidance_scale=0.0)
    test = _test_cfg(steps=28, guidance_scale=4.5)

    assert effective_steps(model, test) == 4
    assert effective_guidance_scale(model, test) == 0.0


def test_validate_rejects_invalid_model_overrides() -> None:
    cfg = ExperimentConfig(
        device="cpu",
        models=[ModelConfig(name="bad", model_id="test/model", steps=0, guidance_scale=-1.0)],
        tests=[_test_cfg()],
    )
    errors = validate_config(cfg)
    assert any("liczbę kroków" in error for error in errors)
    assert any("guidance_scale" in error for error in errors)


def test_validate_accepts_model_overrides() -> None:
    cfg = ExperimentConfig(
        device="cpu",
        models=[ModelConfig(name="turbo", model_id="test/turbo", steps=4, guidance_scale=0.0)],
        tests=[_test_cfg()],
    )
    assert validate_config(cfg) == []
