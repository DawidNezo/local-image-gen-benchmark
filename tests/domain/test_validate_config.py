from __future__ import annotations

from domain import ExperimentConfig, ModelConfig, TestConfig, validate_config


def _minimal_test() -> TestConfig:
    return TestConfig(id="t1", prompt="test prompt", seeds=[1])


def test_validate_accepts_model_path() -> None:
    cfg = ExperimentConfig(
        device="cpu",
        models=[ModelConfig(name="local", model_path="/tmp/local-model")],
        tests=[_minimal_test()],
    )
    errors = validate_config(cfg)
    assert errors == []


def test_validate_rejects_missing_model_source() -> None:
    cfg = ExperimentConfig(
        device="cpu",
        models=[ModelConfig(name="bad")],
        tests=[_minimal_test()],
    )
    errors = validate_config(cfg)
    assert any("model_id" in error or "model_path" in error for error in errors)


def test_validate_rejects_both_model_id_and_path() -> None:
    cfg = ExperimentConfig(
        device="cpu",
        models=[ModelConfig(name="bad", model_id="a", model_path="/tmp/a")],
        tests=[_minimal_test()],
    )
    errors = validate_config(cfg)
    assert any("model_id" in error and "model_path" in error for error in errors)


def test_validate_accepts_rocm_device() -> None:
    cfg = ExperimentConfig(
        device="rocm",
        models=[ModelConfig(name="ok", model_id="test/model")],
        tests=[_minimal_test()],
    )
    errors = validate_config(cfg)
    assert errors == []
