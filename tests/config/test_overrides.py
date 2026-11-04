from __future__ import annotations

import pytest

from imagebench.config import ExperimentConfig, ModelConfig, TestConfig, apply_overrides

pytestmark = pytest.mark.unit


def _config_with_models() -> ExperimentConfig:
    return ExperimentConfig(
        device="cpu",
        models=[
            ModelConfig(name="model-a", model_id="org/model-a"),
            ModelConfig(name="model-b", model_id="org/model-b", dtype="bfloat16"),
        ],
        tests=[TestConfig(id="t1", prompt="prompt", seeds=[1])],
    )


def test_apply_overrides_replaces_prompt_and_device() -> None:
    cfg = _config_with_models()

    result = apply_overrides(cfg, device="mps", prompt="nowy prompt")

    assert result.device == "mps"
    assert all(test.prompt == "nowy prompt" for test in result.tests)


def test_apply_overrides_selects_existing_model_by_name() -> None:
    cfg = _config_with_models()

    result = apply_overrides(cfg, model="model-b")

    assert len(result.models) == 1
    assert result.models[0].name == "model-b"
    assert result.models[0].dtype == "bfloat16"


def test_apply_overrides_builds_model_from_hub_id() -> None:
    cfg = _config_with_models()

    result = apply_overrides(cfg, model="org/new-model")

    assert len(result.models) == 1
    assert result.models[0].name == "new-model"
    assert result.models[0].model_id == "org/new-model"


def test_apply_overrides_builds_model_from_local_path(tmp_path) -> None:
    cfg = _config_with_models()
    model_dir = tmp_path / "my-model"
    model_dir.mkdir()

    result = apply_overrides(cfg, model=str(model_dir))

    assert result.models[0].name == "my-model"
    assert result.models[0].model_path == str(model_dir)
    assert result.models[0].model_id is None


def test_apply_overrides_does_not_modify_input() -> None:
    cfg = _config_with_models()

    apply_overrides(cfg, device="mps", model="model-b", prompt="nowy prompt")

    assert cfg.device == "cpu"
    assert [model.name for model in cfg.models] == ["model-a", "model-b"]
    assert cfg.tests[0].prompt == "prompt"
