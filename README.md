# Local Image Generation Benchmark

*[Wersja polska](README.pl.md)*

A Python CLI (Hugging Face Diffusers) for running, testing and comparing image generation models
locally. For every model × scenario × seed combination it measures model load time, generation
time, throughput and resource usage (CPU/RAM/GPU), and writes the results to JSONL/CSV/Markdown
together with environment metadata. Everything runs locally, with no cloud services.

I designed and implemented it as the subject of my MSc thesis in Applied Computer Science
(Artificial Intelligence), *Analysis, design and implementation of an application for comparing
local image generation models* (2026).

> CLI messages, log output and generated reports are currently in Polish.

## Features

- automated comparison runs across many models and scenarios (model × test × seed loop),
- parameterised experiments: `prompt`, `negative_prompt`, `seed`, `scheduler`, step count,
  `guidance_scale`, resolution, number of images,
- per-model parameter overrides (`steps`, `guidance_scale`): distilled (few-step) models need
  different sampling settings than full models; the model override takes priority over the test
  value, and results always store the effective values,
- per-model memory optimisations: `enable_attention_slicing`, `enable_vae_tiling` (lower peak
  memory, which matters for heavier models such as Lumina Image 2.0),
- `"default"` scheduler keeps the model's native scheduler (required for flow-matching models);
  also available: `ddim`, `euler`, `euler_a`, `dpmpp`, `pndm`,
- pipeline call arguments filtered by the `__call__` signature: a pipeline that does not support
  e.g. `negative_prompt` only receives supported arguments (with a warning in the log),
- memory released between models (gc + CUDA/MPS `empty_cache`); models are loaded sequentially,
- model load time (`load_time_sec`) measured as an ease-of-deployment metric,
- background resource monitoring (sampled every 0.5 s): CPU/RAM via psutil, GPU via
  `nvidia-smi` / `rocm-smi` / MPS adapters (memory allocated by PyTorch),
- compute device resolution: `auto` → MPS → CUDA/ROCm → CPU, with a controlled CPU fallback
  (interactive prompt, `--assume-yes` or `--non-interactive`),
- models from Hugging Face (`model_id`) or local Diffusers paths (`model_path`),
- result export: `results.jsonl`, `results.csv`, `summary_by_model.csv`, Markdown report,
- per-run environment metadata in `run_meta.json` (device, package versions, platform),
- runs locally (recommended on Apple Silicon with MPS) or in Docker (NVIDIA/ROCm overlays on Linux).

## Default model set

`config.example.json` contains four recent models chosen to give a **spread of sizes**
(0.6B to 2.6B parameters) while each of them loads and generates an image in tens of seconds to
a few minutes on a MacBook with 32 GB of unified memory, without memory pressure. All of them are
open (no HF token needed), loaded the same way through `DiffusionPipeline.from_pretrained`, at
1024×1024 with the `"default"` scheduler (flow matching). Listed from lightest to heaviest:

| Model | `model_id` | License | Parameters (overrides) | Size / speed |
|---|---|---|---|---|
| SANA-Sprint 0.6B (NVIDIA) | `Efficient-Large-Model/Sana_Sprint_0.6B_1024px_diffusers` | NVIDIA NSCL v2 (non-commercial) | steps=2, guidance_scale=4.5; few-step distillation, 0.6B DiT + Gemma encoder + DC-AE; no negative prompt | smallest; seconds per image |
| SDXL-Turbo (Stability AI) | `stabilityai/sdxl-turbo` | SAI NC Community (non-commercial) | steps=4, guidance_scale=0.0; few-step distillation, ~3.5B UNet; different architecture (baseline) | ~7 GB; seconds per image |
| SANA 1.6B (NVIDIA) | `Efficient-Large-Model/Sana_1600M_1024px_BF16_diffusers` | NVIDIA NSCL v2 (non-commercial) | steps=20, guidance_scale=4.5, `variant="bf16"` (required); full 1.6B DiT + Gemma-2-2B + DC-AE | ~9 GB; tens of seconds |
| Lumina Image 2.0 (Alpha-VLLM) | `Alpha-VLLM/Lumina-Image-2.0` | Apache 2.0 (commercial OK) | steps=24, guidance_scale=4.0, no `variant` (fp32 weights cast to bf16); full 2.6B DiT + Gemma-2 | ~10.6 GB resident; heaviest, a few minutes |

The set covers two architectures (SANA-family DiT and the SDXL-Turbo UNet) and two sampling
regimes (few-step distilled: SANA-Sprint, SDXL-Turbo; full: SANA 1.6B, Lumina), which makes it
possible to compare the quality–time–memory trade-off. None of the default models comes close to
the 32 GB limit.

**Optional, heavier models** (for a bigger time/memory budget) are available through `--model`
and deliberately left out of the default set, because on 32 GB MPS they load slowly or run close
to the memory limit: Z-Image-Turbo (`Tongyi-MAI/Z-Image-Turbo`, ~10B params, ~20 GB),
Stable Diffusion 3.5 Medium (`stabilityai/stable-diffusion-3.5-medium`, gated, 40 steps + T5-XXL)
and FLUX.2-klein-9B (`black-forest-labs/FLUX.2-klein-9B`, gated, ~29 GB, needs
`diffusers>=0.37.1`). Gated models need `HF_TOKEN`, see
[Gated models and the Hugging Face token](#gated-models-and-the-hugging-face-token).
Full FLUX.1-dev was skipped (~34 GB in bf16, does not fit in 32 GB; FP8/GGUF quantisation does
not work on MPS).

```bash
# running an optional, heavier model:
python app.py run --config config.example.json --model Tongyi-MAI/Z-Image-Turbo
```

Test scenarios in `config.example.json`: `s1-fotorealizm-portret` (photorealistic portrait,
2 seeds, negative prompt), `s2-zgodnosc-z-promptem` (prompt adherence: attribute binding and
spatial relations; no negative prompt, so it is fully comparable for distilled models too),
`s3-styl-ilustracyjny` (isometric illustration, negative prompt).

### Gated models and the Hugging Face token

The default set is fully open and **does not need a token**. A token is only needed for the
optional gated models (`"gated": true`): `stabilityai/stable-diffusion-3.5-medium` (Stability
Community license, commercial use OK below USD 1M) or `black-forest-labs/FLUX.2-klein-9B` (FLUX
Non-Commercial license). Each requires accepting the license once on the model page and providing
an HF access token. Create the token in your
[HF account settings](https://huggingface.co/settings/tokens) (*read* scope is enough) and expose
it as an environment variable:

```bash
export HF_TOKEN=hf_...            # or: hf auth login
python app.py run --config config.example.json --model stabilityai/stable-diffusion-3.5-medium
```

`DiffusionPipeline.from_pretrained` reads `HF_TOKEN` automatically. Treat the token as a secret
and **never commit it**. If the token is missing or the license was not accepted, that gated model
gets status `error` with a readable hint, and the remaining models run normally (one model's
failure does not stop the run).

## Project structure

- `app.py`: thin CLI entry point (argparse) with `validate`, `devices`, `run`, `report`
  commands and device fallback handling,
- `domain.py`: domain model (`ModelConfig`, `TestConfig`, `ExperimentConfig`, `RunResult`,
  `ResourceSample`), config loading and validation, `effective_steps` /
  `effective_guidance_scale`,
- `benchmark_runner.py`: `ModelRunner` (pipeline loading, `last_load_time_sec`, schedulers,
  argument filtering, memory release), `resolve_device`, `list_available_devices`, CLI overrides,
- `orchestrator.py`: benchmark loop and aggregation of monitor samples into `RunResult` metrics,
- `resource_monitor.py`: `ResourceMonitor` (sampling thread) and GPU adapters
  `NvidiaSmiGpuAdapter`, `RocmSmiGpuAdapter`, `MpsGpuAdapter`, `NullGpuAdapter`,
- `outputs.py`: `results.jsonl` / `results.csv` / `summary_by_model.csv`, Markdown report
  (`build_markdown_report`), environment metadata (`write_run_metadata` → `run_meta.json`),
- `config.example.json`: sample experiment plan (default model set),
- `tests/`: pytest suite (unit/integration/contract); `pytest.ini` defines the markers,
- `Dockerfile`, `docker-compose.yml`, `docker-compose.nvidia.yml`, `docker-compose.rocm.yml`:
  container setup,
- `.github/workflows/ci.yml`: CI pipeline (lint, tests, config validation, report smoke test),
- `results/report.md`: sample report from a run on Apple Silicon (32 GB).

## Requirements

- Python 3.10+ (developed and tested on 3.11/3.13),
- dependencies from `requirements.txt`, including `diffusers>=0.36` (needed for
  `ZImagePipeline`), `torch>=2.4`, `transformers>=4.51`, `accelerate`, `safetensors`, `psutil`,
- hardware acceleration (optional, recommended):
  - Apple Silicon (M1/M2/M3) + macOS + PyTorch with MPS,
  - Linux/Windows + NVIDIA: PyTorch with CUDA, drivers and `nvidia-smi`,
  - Linux + AMD: PyTorch with ROCm, drivers and `rocm-smi`,
- for the container setup: Docker + Docker Compose.

> On macOS, Metal (MPS) acceleration only works natively on the host, not inside a Docker
> container (Docker on macOS runs in a Linux VM).

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## CLI commands

Global flag: `--verbose` (detailed logging), e.g. `python app.py --verbose run ...`.

### validate: check the configuration

Validates the config file (model and test fields) and whether the declared device is available;
exits with code 2 on errors.

```bash
python app.py validate --config config.example.json
```

### devices: list available devices

```bash
python app.py devices
```

### run: run a comparison

```bash
python app.py run --config config.example.json
```

Flags:

- `--device {auto,mps,cuda,rocm,cpu}`: overrides the device from the config,
- `--model <name|model_id|model_path>`: limits the run to one model; a name from the config
  selects an existing entry, a local path or HF id creates an ad-hoc entry
  (dtype/variant/use_safetensors inherited from the first model in the config),
- `--prompt "<text>"`: overrides the prompt in all tests,
- `--limit-models N`, `--limit-tests N`: limit the scope (debugging),
- `--assume-yes`: automatic CPU fallback when the requested device is unavailable,
- `--non-interactive`: no fallback prompt, hard error instead (exit code 2); `--assume-yes` and
  `--non-interactive` are mutually exclusive.

Examples:

```bash
# quick diagnostic run: first model, first test
python app.py run --config config.example.json --limit-models 1 --limit-tests 1

# a single model from the config, forced MPS
python app.py run --config config.example.json --model z-image-turbo --device mps

# a model outside the config (Hugging Face id)
python app.py run --config config.example.json --model black-forest-labs/FLUX.2-klein-4B

# one prompt for all scenarios
python app.py run --config config.example.json --prompt "studio portrait, 85mm lens"
```

> The first run of a model downloads its weights from Hugging Face (around 10–20 GB per model)
> into the `.hf_cache` directory (`HF_HOME`, set to `./.hf_cache` automatically if undefined).
> Later runs use the cache and start much faster.

### report: Markdown report from results

```bash
python app.py report --input results/run_YYYYMMDD_HHMMSS/results.jsonl --output results/report.md
```

## Running in Docker

Build the image:

```bash
docker compose build
```

Validate the configuration (the service's default command):

```bash
docker compose run --rm model-benchmark validate --config config.example.json
```

Run a comparison:

```bash
docker compose run --rm model-benchmark run --config config.example.json
```

Model weights are cached in the named volume `hf-cache` (`HF_HOME=/app/.hf_cache`), so later
container runs do not download them again; the project directory is mounted at `/app`, so results
end up in the local `results/`.

GPU overlays (Linux):

```bash
# NVIDIA (requires NVIDIA Container Toolkit)
docker compose -f docker-compose.yml -f docker-compose.nvidia.yml run --rm model-benchmark run --config config.example.json

# AMD ROCm (/dev/kfd and /dev/dri devices, video group)
docker compose -f docker-compose.yml -f docker-compose.rocm.yml run --rm model-benchmark run --config config.example.json
```

Note: on macOS the container has no access to MPS, so Docker runs are CPU-only; use a local run on
the host for Metal-accelerated measurements.

## Configuration file

A JSON file (template: `config.example.json`) with global fields and `models` and `tests` lists.

Global fields:

| Field | Type | Default | Description |
|---|---|---|---|
| `output_dir` | string | `"results"` | parent directory for run directories |
| `device` | string | `"auto"` | `auto`, `mps`, `cuda`, `rocm` or `cpu` |
| `save_images` | bool | `true` | save generated images to `images/` |

Model fields (`models[]`):

| Field | Type | Default | Description |
|---|---|---|---|
| `name` | string | (required) | unique model name in the results |
| `model_id` | string \| null | `null` | Hugging Face repository id; mutually exclusive with `model_path` (exactly one must be set) |
| `model_path` | string \| null | `null` | local path to a model in Diffusers format |
| `dtype` | string | `"float16"` | `float16`, `float32` or `bfloat16` |
| `variant` | string \| null | `null` | weights variant (e.g. `"bf16"`, required for SANA) |
| `use_safetensors` | bool | `true` | force the safetensors format |
| `steps` | int \| null | `null` | step count override for this model (takes priority over the test) |
| `guidance_scale` | float \| null | `null` | guidance scale override for this model (takes priority over the test) |
| `enable_attention_slicing` | bool | `false` | enables attention slicing (less memory, more time) |
| `enable_vae_tiling` | bool | `false` | enables tiled VAE decoding (less memory at 1024×1024) |
| `gated` | bool | `false` | marks a gated model (license acceptance and `HF_TOKEN` required); changes the error message on a failed download |

Override semantics: if a model defines `steps` or `guidance_scale`, those values replace the
scenario settings for all of its runs (`effective_steps` / `effective_guidance_scale` in
`domain.py`). This makes it possible to compare distilled models (e.g. SANA-Sprint: 2 steps,
SDXL-Turbo: 4 steps with CFG off) with full models (SANA 1.6B: 20 steps, Lumina 2: 24 steps) in
one run without duplicating scenarios. The `steps` and `guidance_scale` columns in the results always hold the effective values.

Test fields (`tests[]`):

| Field | Type | Default | Description |
|---|---|---|---|
| `id` | string | (required) | scenario id in the results |
| `prompt` | string | (required, non-empty) | description of the image to generate |
| `negative_prompt` | string | `""` | negative prompt; skipped automatically for pipelines that do not support it |
| `width`, `height` | int | `512` | image resolution |
| `steps` | int | `30` | sampling steps (unless the model overrides it) |
| `guidance_scale` | float | `7.5` | classifier-free guidance strength (unless the model overrides it) |
| `scheduler` | string | `"euler_a"` | `ddim`, `euler`, `euler_a`, `dpmpp`, `pndm` or `"default"` (the model's native scheduler, required for flow-matching models) |
| `seeds` | list[int] | `[42]` | list of seeds; each seed is a separate run (reproducibility) |
| `num_images` | int | `1` | images per pipeline call |

## Run artefacts and result columns

Each run creates a `results/run_<timestamp>/` directory with:

- `run_meta.json`: environment metadata (date, device, config path, Python version, platform and
  architecture, versions of torch, diffusers, transformers, accelerate, safetensors, psutil),
  written at the start of every run to document the measurement conditions,
- `results.jsonl`: full run records (one JSON line per run),
- `results.csv`: the same data as a flat table,
- `summary_by_model.csv`: per-model aggregates,
- `images/`: images `<model>_<test>_seed<seed>.png` (if `save_images=true`).

`results.jsonl` / `results.csv` columns: `timestamp`, `model_name`, `model_id`, `test_id`,
`prompt`, `seed`, `scheduler`, `steps`, `guidance_scale` (effective values after overrides),
`width`, `height`, `load_time_sec` (model load time; 0 on an in-process cache hit),
`duration_sec`, `throughput_img_s`, `avg/max_cpu_percent`, `avg/max_ram_percent`,
`avg/max_gpu_util`, `avg/max_gpu_memory_percent`, `status`, `error`, `image_path`.

`summary_by_model.csv` columns: `model_name`, `runs`, `success_runs`, `avg_duration_sec`,
`avg_throughput_img_s`, `avg_gpu_util`, `avg_gpu_memory_percent`.

## Tests and CI

48 pytest tests (markers `unit`, `integration`, `contract`) in `tests/cli`, `tests/domain`,
`tests/runner`, `tests/monitor`, `tests/output`, with data in `tests/fixtures`. The tests need
neither a GPU nor model downloads; they use test doubles (DummyRunner/DummyMonitor, FakeTorch for
`resolve_device`, fake `nvidia-smi`/`rocm-smi` adapters).

```bash
# everything
pytest

# with coverage (as in CI)
pytest --cov=app --cov=domain --cov=benchmark_runner --cov=orchestrator \
  --cov=resource_monitor --cov=outputs --cov-report=term-missing --cov-fail-under=60

# one group
pytest -m unit
```

CI pipeline (`.github/workflows/ci.yml`, GitHub Actions, on push to main/master and pull requests):

1. `ruff check` on all application modules,
2. `python -m py_compile` (syntax check),
3. `pytest` with coverage, 60% threshold (actual coverage ~72%),
4. `python app.py validate --config config.example.json`,
5. report smoke test: `python app.py report --input tests/fixtures/sample_results.jsonl --output results/report_ci.md`.

## Project goals

- **Automated comparison tests**: the `run` command and declarative experiment configuration,
- **Reproducibility**: explicit seeds, scheduler, steps and resolution, plus `run_meta.json` with
  package versions and environment details,
- **Resource usage analysis**: CPU/RAM/GPU monitoring during generation, avg/max aggregates,
- **Ease of deployment**: the `load_time_sec` metric, uniform loading through
  `DiffusionPipeline`, weight caching,
- **Containerisation**: `Dockerfile` + Compose overlays (reproducible runtime; no MPS on macOS),
- **Export and reporting**: JSONL/CSV/Markdown, per-model aggregation.

## Limitations

- on macOS the Docker container has no access to MPS, so Metal-accelerated measurements only run
  on the host,
- on MPS, GPU utilisation is not available (`avg_gpu_util = null`); the reported GPU memory
  percentage is the PyTorch allocation relative to unified memory,
- all measurements so far were taken on one machine (MacBook, Apple Silicon, 32 GB, MPS, natively
  on the host); the CUDA/ROCm Compose overlays and the `nvidia-smi`/`rocm-smi` adapters are
  covered by tests with fakes but have not been verified on physical NVIDIA/AMD hardware,
- status `ok` only means the run raised no exception; the tool does not inspect pixels, so an
  all-black image still counts as a successful run (see [Sample results](#sample-results)),
- NVIDIA GPU metrics depend on `nvidia-smi` and AMD metrics on `rocm-smi` (both must be on PATH),
- the first run of a model is slow (one-time weight download, around 3–11 GB per model for the
  default set); later runs read the weights from the `.hf_cache` cache,
- the default set was chosen so that each model loads and generates in tens of seconds to a few
  minutes on 32 GB MPS without memory pressure; the optional models (Z-Image-Turbo, SD 3.5 Medium,
  FLUX.2-klein-9B) are much heavier, can take many minutes and run close to the 32 GB limit,
- the optional gated models (SD 3.5 Medium, FLUX.2-klein-9B) need license acceptance and
  `HF_TOKEN`; the default set does not need a token,
- some model licenses are restrictive: SANA (NVIDIA NSCL v2) and SDXL-Turbo (SAI NC Community)
  are non-commercial, Lumina Image 2.0 (Apache 2.0) allows commercial use.

## Sample results

`results/report.md` contains a sample run of the default set (1024×1024, Apple Silicon, 32 GB):
average time per image ranged from 7.3 s (SANA-Sprint 0.6B) to 748 s (Lumina Image 2.0).
All 16 runs finished with status `ok`, but manual inspection showed that three of the four
SDXL-Turbo images were completely black, which is why `ok` should not be read as "usable image".

## License

Copyright (c) 2026 Dawid Mazur. All rights reserved.

The source code is published for viewing only. No license is granted to use, copy, modify or
distribute it without written permission.
