# Local Image Generation Benchmark

*[Wersja polska](README.pl.md)*

A Python CLI for benchmarking text-to-image diffusion models locally with Hugging Face Diffusers.
For every model × prompt × seed it measures load time, generation time, throughput and CPU/RAM/GPU
usage, and saves the results as JSONL, CSV and a Markdown report together with environment
metadata. Everything runs locally, with no cloud services.

> CLI messages, logs and reports are in Polish.

## Features

- declarative experiments in JSON: models, prompts, seeds, scheduler, steps, resolution,
- per-model `steps` / `guidance_scale` overrides, so distilled few-step models and full models
  can be compared in one run,
- models from the Hugging Face Hub or a local path, loaded one at a time with memory released
  in between,
- CPU/RAM/GPU sampled every 0.5 s (psutil, `nvidia-smi`, `rocm-smi`, MPS),
- device auto-detection (MPS → CUDA/ROCm → CPU) with an optional CPU fallback,
- runs natively (Apple Silicon with MPS recommended) or in Docker with NVIDIA/ROCm overlays.

## Quick start

Requires Python 3.10+. For GPU acceleration, use a PyTorch build with MPS, CUDA or ROCm.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python -m imagebench devices                                # list available devices
python -m imagebench validate --config config.example.json  # check the config
python -m imagebench run --config config.example.json       # run the benchmark
python -m imagebench report --input results/run_<timestamp>/results.jsonl --output results/report.md
```

The first run of a model downloads its weights (3–11 GB per default model) into `.hf_cache/`.

Useful `run` flags:

| Flag | Effect |
| --- | --- |
| `--model <name\|hub-id\|path>` | run one model: from the config, or an ad-hoc one (inherits `dtype`/`variant` from the first configured model) |
| `--device auto\|mps\|cuda\|rocm\|cpu` | override the configured device |
| `--prompt "..."` | use one prompt for all tests |
| `--limit-models N`, `--limit-tests N` | run only the first N models / tests |
| `--assume-yes` / `--non-interactive` | if the device is unavailable: switch to CPU / fail with exit code 2 (by default it asks) |

`--verbose` before the command enables debug logging.

## Default models

`config.example.json` contains four open models (no Hugging Face token needed) and three prompts
(photorealistic portrait, prompt adherence, isometric illustration) at 1024×1024, 16 runs in
total. Times are averages from the sample run in [`results/report.md`](results/report.md)
(MacBook, Apple Silicon, 32 GB, MPS):

| Model | Type | Steps | License | Time / image |
| --- | --- | ---: | --- | ---: |
| [SANA-Sprint 0.6B](https://huggingface.co/Efficient-Large-Model/Sana_Sprint_0.6B_1024px_diffusers) | DiT, distilled | 2 | NVIDIA NSCL v2 (non-commercial) | 7 s |
| [SDXL-Turbo](https://huggingface.co/stabilityai/sdxl-turbo) | UNet, distilled | 4 | SAI NC Community (non-commercial) | 14 s |
| [SANA 1.6B](https://huggingface.co/Efficient-Large-Model/Sana_1600M_1024px_BF16_diffusers) | DiT | 20 | NVIDIA NSCL v2 (non-commercial) | 68 s |
| [Lumina Image 2.0](https://huggingface.co/Alpha-VLLM/Lumina-Image-2.0) | DiT, 2.6B | 24 | Apache 2.0 | 748 s |

All 16 runs ended with status `ok`, yet three of the four SDXL-Turbo images were completely
black: `ok` means "no exception", not "usable image".

Heavier models also work through `--model`. They are not in the default set because they run
close to the 32 GB limit: `Tongyi-MAI/Z-Image-Turbo`, `stabilityai/stable-diffusion-3.5-medium`
and `black-forest-labs/FLUX.2-klein-9B` (needs `diffusers>=0.37.1`). The last two are gated:
accept the license on the model page and set `HF_TOKEN` (or run `hf auth login`). Without
access, that model fails with a hint and the remaining models still run.

## Configuration

Top-level fields: `output_dir` (default `results`), `device` (`auto`), `save_images` (`true`),
plus the `models` and `tests` lists. See [`config.example.json`](config.example.json).

| `models[]` | Default | Description |
| --- | --- | --- |
| `name` | required | name used in the results |
| `model_id` / `model_path` | – | Hub id or local Diffusers directory (exactly one) |
| `dtype` | `float16` | `float16`, `bfloat16` or `float32` |
| `variant` | `null` | weights variant, e.g. `bf16` |
| `use_safetensors` | `true` | load safetensors weights |
| `steps`, `guidance_scale` | `null` | override the test values for this model |
| `enable_attention_slicing`, `enable_vae_tiling` | `false` | lower peak memory |
| `gated` | `false` | gated model: clearer error when access is missing |

| `tests[]` | Default | Description |
| --- | --- | --- |
| `id`, `prompt` | required | |
| `negative_prompt` | `""` | skipped for pipelines that do not accept it |
| `width`, `height` | `512` | resolution |
| `steps`, `guidance_scale` | `30`, `7.5` | unless the model overrides them |
| `scheduler` | `euler_a` | `default` (the model's own; needed for flow-matching models), `ddim`, `euler`, `euler_a`, `dpmpp`, `pndm` |
| `seeds` | `[42]` | one run per seed |
| `num_images` | `1` | images per pipeline call |

## Output

Each run creates `results/run_<timestamp>/` with:

- `results.jsonl`, `results.csv`: one row per run with the effective parameters,
  `load_time_sec`, `duration_sec`, `throughput_img_s`, average/max CPU, RAM and GPU usage,
  `status` and `error`,
- `summary_by_model.csv`: per-model averages over successful runs,
- `run_meta.json`: device, Python version, platform and package versions,
- `images/`: generated images (if `save_images` is on).

## Docker

```bash
docker compose build
docker compose run --rm model-benchmark run --config config.example.json

# GPU on Linux: add the NVIDIA (or ROCm) overlay
docker compose -f docker-compose.yml -f docker-compose.nvidia.yml run --rm model-benchmark run --config config.example.json
```

Weights are cached in the `hf-cache` volume and results land in the local `results/`. On macOS,
Docker has no access to MPS and runs on the CPU, so measure on the host instead.

## Limitations

- measured on one machine only (MacBook, Apple Silicon, 32 GB, MPS); the NVIDIA/ROCm paths are
  covered by tests with fakes, not by real hardware,
- MPS reports no GPU utilisation; GPU memory there is the PyTorch allocation as a share of unified
  memory,
- in the first run of each model, resource usage also covers loading the weights,
- status `ok` does not check the image (see the black SDXL-Turbo images above).

## Development

```bash
pip install -r requirements-dev.txt
pytest                                # 53 tests, no GPU or downloads; -m unit|integration|contract
pytest --cov                          # with coverage
ruff check . && ruff format --check .
```

GitHub Actions (`.github/workflows/ci.yml`) runs the same checks on Python 3.10 and 3.12, plus
`validate` on the example config and a `report` smoke test.

The code lives in `imagebench/`. `cli.py` handles the commands, `benchmark.py` runs the
benchmark, and `orchestrator.py` loops over model × test × seed. The loop uses `pipeline.py`
(Diffusers), `resource_monitor.py`, `config.py` and `devices.py`, and writes its output through
`results.py`, `report.py` and `metadata.py`.

## License

Copyright (c) 2026 Dawid Mazur. All rights reserved.

The source code is published for viewing only. No license is granted to use, copy, modify or
distribute it without written permission.

---

*Built as the practical part of my MSc thesis in Applied Computer Science (AI), 2026.*
