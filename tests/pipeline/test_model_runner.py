from __future__ import annotations

import pytest

from imagebench.config import ModelConfig
from imagebench.pipeline import ModelRunner

pytestmark = pytest.mark.unit


class FakePipelineWithoutNegativePrompt:
    def __call__(self, prompt, num_inference_steps, guidance_scale, width, height, generator):
        return None


class FakePipelineWithKwargs:
    def __call__(self, prompt, **kwargs):
        return None


def test_filter_drops_unsupported_kwargs() -> None:
    runner = ModelRunner(device="cpu")
    call_kwargs = {
        "prompt": "p",
        "negative_prompt": "n",
        "num_inference_steps": 4,
        "guidance_scale": 0.0,
        "width": 512,
        "height": 512,
        "num_images_per_prompt": 1,
        "generator": None,
    }

    filtered = runner._filter_supported_kwargs(
        FakePipelineWithoutNegativePrompt(), "turbo", call_kwargs
    )

    assert "negative_prompt" not in filtered
    assert "num_images_per_prompt" not in filtered
    assert filtered["prompt"] == "p"
    assert filtered["num_inference_steps"] == 4


def test_filter_keeps_kwargs_for_var_keyword_pipelines() -> None:
    runner = ModelRunner(device="cpu")
    call_kwargs = {"prompt": "p", "negative_prompt": "n"}

    filtered = runner._filter_supported_kwargs(FakePipelineWithKwargs(), "m", call_kwargs)

    assert filtered == call_kwargs


def test_release_loaded_models_clears_cache() -> None:
    runner = ModelRunner(device="cpu")
    runner._loaded["old-model"] = object()

    runner._release_loaded_models()

    assert runner._loaded == {}


def test_wrap_load_error_gives_hint_for_gated_model() -> None:
    model_cfg = ModelConfig(name="sd35", model_id="stabilityai/sd35", gated=True)

    wrapped = ModelRunner._wrap_load_error(model_cfg, "stabilityai/sd35", OSError("boom"))

    assert isinstance(wrapped, RuntimeError)
    assert "HF_TOKEN" in str(wrapped)
    assert "stabilityai/sd35" in str(wrapped)


def test_wrap_load_error_detects_401_from_message() -> None:
    model_cfg = ModelConfig(name="m", model_id="org/m")

    wrapped = ModelRunner._wrap_load_error(
        model_cfg, "org/m", OSError("401 Client Error: Unauthorized")
    )

    assert isinstance(wrapped, RuntimeError)
    assert "HF_TOKEN" in str(wrapped)


def test_wrap_load_error_passes_through_unrelated_error() -> None:
    model_cfg = ModelConfig(name="m", model_id="org/m")
    original = ValueError("dysk pełny")

    wrapped = ModelRunner._wrap_load_error(model_cfg, "org/m", original)

    assert wrapped is original
