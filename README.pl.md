# Aplikacja do porównania lokalnych modeli generatywnych obrazów

*[English version](README.md)*

Narzędzie CLI w Pythonie do lokalnego benchmarku modeli dyfuzyjnych text-to-image (Hugging Face
Diffusers). Dla każdej kombinacji model × prompt × seed mierzy czas ładowania, czas generacji,
przepustowość i zużycie CPU/RAM/GPU, a wyniki zapisuje w JSONL, CSV i raporcie Markdown razem
z metadanymi środowiska. Całość działa lokalnie, bez usług chmurowych.

## Funkcje

- deklaratywne eksperymenty w JSON: modele, prompty, seedy, scheduler, liczba kroków,
  rozdzielczość,
- nadpisania `steps` / `guidance_scale` per model, dzięki czemu modele destylowane (kilka kroków)
  i pełne można porównać w jednym przebiegu,
- modele z Hugging Face Hub albo z lokalnej ścieżki, ładowane pojedynczo, ze zwalnianiem pamięci
  między nimi,
- próbkowanie CPU/RAM/GPU co 0,5 s (psutil, `nvidia-smi`, `rocm-smi`, MPS),
- automatyczny wybór urządzenia (MPS → CUDA/ROCm → CPU) z opcjonalnym przejściem na CPU,
- uruchamianie natywne (zalecane Apple Silicon z MPS) albo w Dockerze z nakładkami NVIDIA/ROCm.

## Szybki start

Wymaga Pythona 3.10+. Do akceleracji GPU potrzebny jest PyTorch z obsługą MPS, CUDA albo ROCm.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python -m imagebench devices                                # lista dostępnych urządzeń
python -m imagebench validate --config config.example.json  # sprawdzenie konfiguracji
python -m imagebench run --config config.example.json       # uruchomienie benchmarku
python -m imagebench report --input results/run_<timestamp>/results.jsonl --output results/report.md
```

Pierwsze uruchomienie modelu pobiera jego wagi (3–11 GB na model z domyślnego zestawu)
do `.hf_cache/`.

Przydatne flagi `run`:

| Flaga | Działanie |
| --- | --- |
| `--model <nazwa\|hub-id\|ścieżka>` | uruchamia jeden model: z konfiguracji albo doraźny (dziedziczy `dtype`/`variant` z pierwszego modelu w konfiguracji) |
| `--device auto\|mps\|cuda\|rocm\|cpu` | nadpisuje urządzenie z konfiguracji |
| `--prompt "..."` | jeden prompt dla wszystkich testów |
| `--limit-models N`, `--limit-tests N` | tylko pierwsze N modeli / testów |
| `--assume-yes` / `--non-interactive` | gdy urządzenie jest niedostępne: przejście na CPU / błąd z kodem 2 (domyślnie program pyta) |

`--verbose` przed komendą włącza szczegółowe logowanie.

## Domyślne modele

`config.example.json` zawiera cztery otwarte modele (bez tokenu Hugging Face) i trzy prompty
(fotorealistyczny portret, zgodność z promptem, ilustracja izometryczna) w 1024×1024, łącznie
16 przebiegów. Czasy to średnie z przykładowego przebiegu w [`results/report.md`](results/report.md)
(MacBook, Apple Silicon, 32 GB, MPS):

| Model | Typ | Kroki | Licencja | Czas / obraz |
| --- | --- | ---: | --- | ---: |
| [SANA-Sprint 0.6B](https://huggingface.co/Efficient-Large-Model/Sana_Sprint_0.6B_1024px_diffusers) | DiT, destylowany | 2 | NVIDIA NSCL v2 (niekomercyjna) | 7 s |
| [SDXL-Turbo](https://huggingface.co/stabilityai/sdxl-turbo) | UNet, destylowany | 4 | SAI NC Community (niekomercyjna) | 14 s |
| [SANA 1.6B](https://huggingface.co/Efficient-Large-Model/Sana_1600M_1024px_BF16_diffusers) | DiT | 20 | NVIDIA NSCL v2 (niekomercyjna) | 68 s |
| [Lumina Image 2.0](https://huggingface.co/Alpha-VLLM/Lumina-Image-2.0) | DiT, 2,6B | 24 | Apache 2.0 | 748 s |

Wszystkie 16 przebiegów zakończyło się statusem `ok`, ale trzy z czterech obrazów SDXL-Turbo
były całkowicie czarne: `ok` oznacza „brak wyjątku”, a nie „użyteczny obraz”.

Cięższe modele też działają przez `--model`. Nie ma ich w domyślnym zestawie, bo pracują blisko
limitu 32 GB: `Tongyi-MAI/Z-Image-Turbo`, `stabilityai/stable-diffusion-3.5-medium`
i `black-forest-labs/FLUX.2-klein-9B` (wymaga `diffusers>=0.37.1`). Dwa ostatnie są bramkowane:
trzeba zaakceptować licencję na stronie modelu i ustawić `HF_TOKEN` (albo wykonać
`hf auth login`). Bez dostępu dany model kończy się błędem z podpowiedzią, a pozostałe działają
dalej.

## Konfiguracja

Pola główne: `output_dir` (domyślnie `results`), `device` (`auto`), `save_images` (`true`)
oraz listy `models` i `tests`. Przykład: [`config.example.json`](config.example.json).

| `models[]` | Domyślnie | Opis |
| --- | --- | --- |
| `name` | wymagane | nazwa w wynikach |
| `model_id` / `model_path` | – | id z Hubu albo lokalny katalog w formacie Diffusers (dokładnie jedno) |
| `dtype` | `float16` | `float16`, `bfloat16` lub `float32` |
| `variant` | `null` | wariant wag, np. `bf16` |
| `use_safetensors` | `true` | wczytywanie wag safetensors |
| `steps`, `guidance_scale` | `null` | nadpisują wartości z testu dla tego modelu |
| `enable_attention_slicing`, `enable_vae_tiling` | `false` | niższe szczytowe zużycie pamięci |
| `gated` | `false` | model bramkowany: czytelniejszy błąd przy braku dostępu |

| `tests[]` | Domyślnie | Opis |
| --- | --- | --- |
| `id`, `prompt` | wymagane | |
| `negative_prompt` | `""` | pomijany dla pipeline'ów, które go nie obsługują |
| `width`, `height` | `512` | rozdzielczość |
| `steps`, `guidance_scale` | `30`, `7.5` | o ile model ich nie nadpisuje |
| `scheduler` | `euler_a` | `default` (natywny scheduler modelu; wymagany dla modeli flow-matching), `ddim`, `euler`, `euler_a`, `dpmpp`, `pndm` |
| `seeds` | `[42]` | jeden przebieg na seed |
| `num_images` | `1` | liczba obrazów na wywołanie pipeline'u |

## Wyniki

Każdy przebieg tworzy katalog `results/run_<timestamp>/`, a w nim:

- `results.jsonl`, `results.csv`: wiersz na przebieg z efektywnymi parametrami,
  `load_time_sec`, `duration_sec`, `throughput_img_s`, średnim i maksymalnym zużyciem CPU, RAM
  i GPU, `status` i `error`,
- `summary_by_model.csv`: średnie per model z udanych przebiegów,
- `run_meta.json`: urządzenie, wersja Pythona, platforma i wersje pakietów,
- `images/`: wygenerowane obrazy (gdy `save_images` jest włączone).

## Docker

```bash
docker compose build
docker compose run --rm model-benchmark run --config config.example.json

# GPU na Linuksie: dodaj nakładkę NVIDIA (albo ROCm)
docker compose -f docker-compose.yml -f docker-compose.nvidia.yml run --rm model-benchmark run --config config.example.json
```

Wagi są cache'owane w wolumenie `hf-cache`, a wyniki trafiają do lokalnego `results/`. Na macOS
Docker nie ma dostępu do MPS i liczy na CPU, więc pomiary trzeba robić na hoście.

## Ograniczenia

- pomiary tylko na jednej maszynie (MacBook, Apple Silicon, 32 GB, MPS); ścieżki NVIDIA/ROCm są
  pokryte testami z atrapami, ale nie sprawdzone na prawdziwym sprzęcie,
- MPS nie raportuje wykorzystania GPU; pamięć GPU to tam alokacja PyTorch jako odsetek pamięci
  zunifikowanej,
- w pierwszym przebiegu każdego modelu zużycie zasobów obejmuje też ładowanie wag,
- status `ok` nie sprawdza obrazu (patrz czarne obrazy SDXL-Turbo wyżej).

## Rozwój

```bash
pip install -r requirements-dev.txt
pytest                                # 53 testy, bez GPU i pobierania; -m unit|integration|contract
pytest --cov                          # z pokryciem
ruff check . && ruff format --check .
```

GitHub Actions (`.github/workflows/ci.yml`) uruchamia te same kroki na Pythonie 3.10 i 3.12,
a do tego `validate` na przykładowej konfiguracji i test dymny `report`.

Kod jest w `imagebench/`. `cli.py` obsługuje komendy, `benchmark.py` uruchamia benchmark,
a `orchestrator.py` przechodzi pętlą po model × test × seed. Pętla korzysta z `pipeline.py`
(Diffusers), `resource_monitor.py`, `config.py` i `devices.py`, a wyniki zapisuje przez
`results.py`, `report.py` i `metadata.py`.

## Licencja

Copyright (c) 2026 Dawid Mazur. Wszelkie prawa zastrzeżone.

Kod jest udostępniony wyłącznie do wglądu. Korzystanie, kopiowanie, modyfikowanie
i rozpowszechnianie wymaga pisemnej zgody autora.

---

*Projekt powstał jako część praktyczna mojej pracy magisterskiej z informatyki stosowanej (AI), 2026.*
