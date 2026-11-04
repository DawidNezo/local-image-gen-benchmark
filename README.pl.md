# Aplikacja do porównania lokalnych modeli generatywnych obrazów

*[English version](README.md)*

Narzędzie CLI (Python + Diffusers) do lokalnego uruchamiania, testowania i porównywania modeli
generatywnych obrazów. Dla każdej kombinacji model × scenariusz × seed mierzy czas ładowania
modelu, czas generacji, przepustowość oraz zużycie zasobów (CPU/RAM/GPU), a wyniki zapisuje
w formatach JSONL/CSV/Markdown wraz z metadanymi środowiska. Projekt powstał jako część
praktyczna pracy magisterskiej i działa w całości lokalnie — bez usług chmurowych.

## Zakres funkcjonalny

- automatyczne przebiegi porównawcze wielu modeli i scenariuszy (pętla model × test × seed),
- parametryzacja eksperymentów: `prompt`, `negative_prompt`, `seed`, `scheduler`, liczba kroków,
  `guidance_scale`, rozdzielczość, liczba obrazów,
- nadpisania parametrów per model (`steps`, `guidance_scale`) — modele destylowane (few-step)
  wymagają innych ustawień próbkowania niż modele pełne; nadpisanie modelu ma priorytet nad
  wartością z testu, a w wynikach zapisywane są wartości efektywne,
- optymalizacje pamięci per model: `enable_attention_slicing`, `enable_vae_tiling`
  (ograniczają szczytowe zużycie pamięci, istotne przy cięższych modelach jak Lumina Image 2.0),
- scheduler `"default"` — pozostawia natywny scheduler modelu (wymagane dla modeli
  flow-matching); dostępne też: `ddim`, `euler`, `euler_a`, `dpmpp`, `pndm`,
- filtrowanie argumentów wywołania pipeline'u według sygnatury `__call__` — pipeline, który nie
  obsługuje np. `negative_prompt`, dostaje tylko wspierane argumenty (z ostrzeżeniem w logu),
- zwalnianie pamięci między modelami (gc + `empty_cache` CUDA/MPS) — modele ładowane sekwencyjnie,
- pomiar czasu ładowania modelu (`load_time_sec`) jako metryka łatwości wdrożenia,
- monitoring zasobów w tle (próbkowanie co 0,5 s): CPU/RAM przez psutil, GPU przez adaptery
  `nvidia-smi` / `rocm-smi` / MPS (pamięć alokowana przez PyTorch),
- rozwiązywanie urządzenia obliczeniowego: `auto` → MPS → CUDA/ROCm → CPU, z kontrolowanym
  fallbackiem do CPU (pytanie interaktywne, `--assume-yes` lub `--non-interactive`),
- obsługa modeli z Hugging Face (`model_id`) oraz lokalnych ścieżek Diffusers (`model_path`),
- eksport wyników: `results.jsonl`, `results.csv`, `summary_by_model.csv`, raport Markdown,
- metadane środowiska każdego przebiegu w `run_meta.json` (urządzenie, wersje pakietów, platforma),
- uruchamianie lokalne (zalecane na Apple Silicon z MPS) lub w kontenerze Docker
  (nakładki NVIDIA/ROCm na Linuksie).

## Domyślny zestaw modeli

Konfiguracja `config.example.json` zawiera cztery modele nowej generacji dobrane tak, aby dawały
**rozpiętość rozmiarów** (od 0,6 mld do 2,6 mld parametrów), a jednocześnie każdy z nich ładował się
i generował obraz w kilkadziesiąt sekund do kilku minut na MacBooku z 32 GB pamięci zunifikowanej —
bez dławienia się i wielominutowego oczekiwania. Wszystkie są otwarte (bez tokenu HF), ładowane
jednolicie przez `DiffusionPipeline.from_pretrained`, w rozdzielczości 1024×1024 ze schedulerem
`"default"` (flow matching). Modele wymieniono w kolejności od najlżejszego:

| Model | `model_id` | Licencja | Parametry (nadpisania) | Rozmiar / szybkość |
|---|---|---|---|---|
| SANA-Sprint 0.6B (NVIDIA) | `Efficient-Large-Model/Sana_Sprint_0.6B_1024px_diffusers` | NVIDIA NSCL v2 (niekomercyjna) | steps=2, guidance_scale=4.5; destylat few-step, DiT 0,6B + enkoder Gemma + DC-AE; negative prompt nieobsługiwany | najmniejszy; generacja ~sekundy |
| SDXL-Turbo (Stability AI) | `stabilityai/sdxl-turbo` | SAI NC Community (niekomercyjna) | steps=4, guidance_scale=0.0; destylat few-step, UNet ~3,5B; inna architektura (punkt odniesienia) | ~7 GB; generacja ~sekundy |
| SANA 1.6B (NVIDIA) | `Efficient-Large-Model/Sana_1600M_1024px_BF16_diffusers` | NVIDIA NSCL v2 (niekomercyjna) | steps=20, guidance_scale=4.5, `variant="bf16"` (obowiązkowy); pełny model DiT 1,6B + Gemma-2-2B + DC-AE | ~9 GB; generacja ~kilkadziesiąt s |
| Lumina Image 2.0 (Alpha-VLLM) | `Alpha-VLLM/Lumina-Image-2.0` | Apache 2.0 (komercja OK) | steps=24, guidance_scale=4.0, bez `variant` (repo ma wagi fp32, rzutowane do bf16); pełny model DiT 2,6B + Gemma-2 | ~10,6 GB rezydentnie; najcięższy — kilka minut |

Zestaw obejmuje dwie architektury (DiT z rodziny SANA oraz UNet w SDXL-Turbo) i dwa reżimy próbkowania
(destylowane few-step: SANA-Sprint, SDXL-Turbo; pełne: SANA 1.6B, Lumina), co pozwala porównać kompromis
jakość–czas–pamięć. Żaden model domyślnego zestawu nie zbliża się do granicy 32 GB, więc nie występuje
dławienie pamięci.

**Modele opcjonalne (cięższe, dla większego budżetu czasu/pamięci)** — dostępne przez flagę `--model`,
świadomie poza domyślnym zestawem, bo na 32 GB MPS ładują się długo lub pracują blisko granicy pamięci:
Z-Image-Turbo (`Tongyi-MAI/Z-Image-Turbo`, ~10 mld param., ~20 GB), Stable Diffusion 3.5 Medium
(`stabilityai/stable-diffusion-3.5-medium`, gated, 40 kroków + T5-XXL) oraz FLUX.2-klein-9B
(`black-forest-labs/FLUX.2-klein-9B`, gated, ~29 GB, wymaga `diffusers>=0.37.1`). Modele gated wymagają
`HF_TOKEN` — patrz [Modele bramkowane i token Hugging Face](#modele-bramkowane-i-token-hugging-face).
Pełny FLUX.1-dev pominięto (w bf16 ~34 GB, nie mieści się w 32 GB; kwantyzacja FP8/GGUF nie działa na MPS).

```bash
# przykład uruchomienia opcjonalnego, cięższego modelu:
python app.py run --config config.example.json --model Tongyi-MAI/Z-Image-Turbo
```

Scenariusze testowe w `config.example.json`: `s1-fotorealizm-portret` (2 seedy, negative prompt),
`s2-zgodnosc-z-promptem` (wiązanie atrybutów i relacje przestrzenne; bez negative promptu — w pełni
porównywalny także dla destylatów), `s3-styl-ilustracyjny` (izometryczna ilustracja, negative prompt).

### Modele bramkowane i token Hugging Face

Domyślny zestaw jest w całości otwarty i **nie wymaga tokenu**. Token jest potrzebny dopiero, gdy
uruchamiasz opcjonalne modele bramkowane (`"gated": true`): `stabilityai/stable-diffusion-3.5-medium`
(licencja Stability Community — komercja OK < 1 mln USD) lub `black-forest-labs/FLUX.2-klein-9B`
(licencja FLUX Non-Commercial — tylko niekomercyjnie). Pobranie każdego z nich wymaga jednorazowego
zaakceptowania licencji na stronie modelu oraz przekazania tokenu dostępu HF. Token generuje się w
[ustawieniach konta HF](https://huggingface.co/settings/tokens) (wystarczy uprawnienie *read*)
i udostępnia przez zmienną środowiskową:

```bash
export HF_TOKEN=hf_...            # albo: hf auth login
python app.py run --config config.example.json --model stabilityai/stable-diffusion-3.5-medium
```

`DiffusionPipeline.from_pretrained` odczytuje `HF_TOKEN` automatycznie. Token traktuj jak sekret —
**nie commituj go** i nie wpisuj do plików repozytorium. Jeśli tokenu brak lub licencja nie została
zaakceptowana, dany model bramkowany otrzyma status `error` z czytelnym komunikatem naprowadzającym,
a pozostałe modele zostaną przetworzone normalnie (błąd jednego modelu nie przerywa przebiegu).

## Struktura projektu

- `app.py` — cienki entrypoint CLI (argparse): komendy `validate`, `devices`, `run`, `report`,
  obsługa fallbacku urządzenia,
- `domain.py` — model domenowy (`ModelConfig`, `TestConfig`, `ExperimentConfig`, `RunResult`,
  `ResourceSample`), wczytywanie i walidacja konfiguracji, funkcje `effective_steps` /
  `effective_guidance_scale`,
- `benchmark_runner.py` — `ModelRunner` (ładowanie pipeline'ów, pomiar `last_load_time_sec`,
  schedulery, filtrowanie argumentów, zwalnianie pamięci), `resolve_device`,
  `list_available_devices`, nadpisania z CLI,
- `orchestrator.py` — pętla benchmarku i agregacja próbek monitora w metryki `RunResult`,
- `resource_monitor.py` — `ResourceMonitor` (wątek próbkujący) + adaptery GPU:
  `NvidiaSmiGpuAdapter`, `RocmSmiGpuAdapter`, `MpsGpuAdapter`, `NullGpuAdapter`,
- `outputs.py` — zapis `results.jsonl` / `results.csv` / `summary_by_model.csv`, raport Markdown
  (`build_markdown_report`), metadane środowiska (`write_run_metadata` → `run_meta.json`),
- `config.example.json` — przykładowy plan eksperymentów (domyślny zestaw modeli),
- `tests/` — testy pytest (unit/integration/contract), `pytest.ini` — konfiguracja markerów,
- `Dockerfile`, `docker-compose.yml`, `docker-compose.nvidia.yml`, `docker-compose.rocm.yml` —
  uruchomienie kontenerowe,
- `.github/workflows/ci.yml` — potok CI (lint, testy, walidacja konfiguracji, smoke test raportu),
- `results/report.md` — przykładowy raport z przebiegu na Apple Silicon (32 GB).

## Wymagania

- Python 3.10+ (rozwijane i testowane na 3.11/3.13),
- zależności z `requirements.txt`, w tym `diffusers>=0.36` (wymagane przez `ZImagePipeline`),
  `torch>=2.4`, `transformers>=4.51`, `accelerate`, `safetensors`, `psutil`,
- akceleracja sprzętowa (opcjonalna, zalecana):
  - Apple Silicon (M1/M2/M3) + macOS + PyTorch z obsługą MPS,
  - Linux/Windows + NVIDIA: PyTorch z CUDA + sterowniki i `nvidia-smi`,
  - Linux + AMD: PyTorch z ROCm + sterowniki i `rocm-smi`,
- dla wariantu kontenerowego: Docker + Docker Compose.

> Na macOS akceleracja Metal (MPS) działa wyłącznie natywnie na hoście — nie wewnątrz kontenera
> Docker (Docker na macOS działa w maszynie wirtualnej z Linuksem).

## Instalacja

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Komendy CLI

Wspólna flaga globalna: `--verbose` (szczegółowe logowanie), np. `python app.py --verbose run ...`.

### validate — walidacja konfiguracji

Sprawdza poprawność pliku konfiguracji (pola modeli i testów) oraz dostępność zadeklarowanego
urządzenia; kod wyjścia 2 przy błędach.

```bash
python app.py validate --config config.example.json
```

### devices — lista dostępnych urządzeń

```bash
python app.py devices
```

### run — uruchomienie przebiegu porównawczego

```bash
python app.py run --config config.example.json
```

Flagi:

- `--device {auto,mps,cuda,rocm,cpu}` — nadpisuje urządzenie z konfiguracji,
- `--model <name|model_id|model_path>` — zawęża przebieg do jednego modelu: nazwa z konfiguracji
  wybiera istniejący wpis, ścieżka lokalna lub identyfikator HF tworzy wpis ad hoc
  (dtype/variant/use_safetensors dziedziczone z pierwszego modelu konfiguracji),
- `--prompt "<tekst>"` — nadpisuje prompt we wszystkich testach,
- `--limit-models N`, `--limit-tests N` — ograniczenie zakresu (debug),
- `--assume-yes` — automatyczny fallback do CPU, gdy żądane urządzenie jest niedostępne,
- `--non-interactive` — bez pytania o fallback: twardy błąd (kod wyjścia 2); flagi
  `--assume-yes` i `--non-interactive` wykluczają się wzajemnie.

Przykłady:

```bash
# szybki przebieg diagnostyczny: pierwszy model, pierwszy test
python app.py run --config config.example.json --limit-models 1 --limit-tests 1

# tylko jeden model z konfiguracji, wymuszony MPS
python app.py run --config config.example.json --model z-image-turbo --device mps

# model spoza konfiguracji (identyfikator Hugging Face)
python app.py run --config config.example.json --model black-forest-labs/FLUX.2-klein-4B

# wspólny prompt dla wszystkich scenariuszy
python app.py run --config config.example.json --prompt "studio portrait, 85mm lens"
```

> Pierwsze uruchomienie danego modelu pobiera jego wagi z Hugging Face (rzędu 10–20 GB na model)
> do katalogu cache `.hf_cache` (zmienna `HF_HOME`, ustawiana automatycznie na `./.hf_cache`,
> jeśli nie jest zdefiniowana). Kolejne przebiegi korzystają z cache i startują znacznie szybciej.

### report — raport Markdown z wyników

```bash
python app.py report --input results/run_YYYYMMDD_HHMMSS/results.jsonl --output results/report.md
python app.py report --input results/run_20260709_123201/results.jsonl --output results/report.md
```

## Uruchomienie w Docker

Budowa obrazu:

```bash
docker compose build
```

Walidacja konfiguracji (domyślna komenda usługi):

```bash
docker compose run --rm model-benchmark validate --config config.example.json
```

Uruchomienie przebiegu:

```bash
docker compose run --rm model-benchmark run --config config.example.json
```

Wagi modeli są cache'owane w nazwanym wolumenie `hf-cache` (`HF_HOME=/app/.hf_cache`),
więc kolejne uruchomienia kontenera nie pobierają ich ponownie; katalog projektu jest montowany
do `/app`, dzięki czemu wyniki trafiają do lokalnego `results/`.

Nakładki GPU (Linux):

```bash
# NVIDIA (wymaga NVIDIA Container Toolkit)
docker compose -f docker-compose.yml -f docker-compose.nvidia.yml run --rm model-benchmark run --config config.example.json

# AMD ROCm (urządzenia /dev/kfd i /dev/dri, grupa video)
docker compose -f docker-compose.yml -f docker-compose.rocm.yml run --rm model-benchmark run --config config.example.json
```

Uwaga: na macOS kontener nie ma dostępu do MPS — przebieg w Dockerze jest CPU-only; do pomiarów
z akceleracją Metal należy używać uruchomienia lokalnego na hoście.

## Struktura pliku konfiguracji

Plik JSON (wzorzec: `config.example.json`) z polami globalnymi oraz listami `models` i `tests`.

Pola globalne:

| Pole | Typ | Domyślnie | Opis |
|---|---|---|---|
| `output_dir` | string | `"results"` | katalog nadrzędny na katalogi przebiegów |
| `device` | string | `"auto"` | `auto`, `mps`, `cuda`, `rocm` lub `cpu` |
| `save_images` | bool | `true` | zapis wygenerowanych obrazów do `images/` |

Pola modelu (`models[]`):

| Pole | Typ | Domyślnie | Opis |
|---|---|---|---|
| `name` | string | — (wymagane) | unikatowa nazwa modelu w wynikach |
| `model_id` | string \| null | `null` | identyfikator repozytorium Hugging Face; wyklucza się z `model_path` (dokładnie jedno z dwóch musi być ustawione) |
| `model_path` | string \| null | `null` | lokalna ścieżka do modelu w formacie Diffusers |
| `dtype` | string | `"float16"` | `float16`, `float32` lub `bfloat16` |
| `variant` | string \| null | `null` | wariant wag (np. `"bf16"` — obowiązkowy dla SANA) |
| `use_safetensors` | bool | `true` | wymuszenie formatu safetensors |
| `steps` | int \| null | `null` | nadpisanie liczby kroków dla tego modelu (priorytet nad wartością z testu) |
| `guidance_scale` | float \| null | `null` | nadpisanie guidance scale dla tego modelu (priorytet nad wartością z testu) |
| `enable_attention_slicing` | bool | `false` | włącza attention slicing (mniejsze zużycie pamięci kosztem czasu) |
| `enable_vae_tiling` | bool | `false` | włącza kafelkowe dekodowanie VAE (mniejsze zużycie pamięci przy 1024×1024) |
| `gated` | bool | `false` | oznacza model bramkowany (wymaga akceptacji licencji i `HF_TOKEN`); wpływa na komunikat błędu przy nieudanym pobraniu |

Semantyka nadpisań: jeżeli model definiuje `steps` lub `guidance_scale`, wartości te zastępują
ustawienia scenariusza dla wszystkich jego przebiegów (`effective_steps` /
`effective_guidance_scale` w `domain.py`). Pozwala to porównywać w jednym przebiegu modele
destylowane (np. SANA-Sprint: 2 kroki, SDXL-Turbo: 4 kroki bez CFG) z modelami pełnymi (SANA 1.6B:
20 kroków, Lumina 2: 24 kroki) bez duplikowania scenariuszy. W plikach wyników kolumny `steps`
i `guidance_scale` zawierają zawsze wartości efektywne.

Pola testu (`tests[]`):

| Pole | Typ | Domyślnie | Opis |
|---|---|---|---|
| `id` | string | — (wymagane) | identyfikator scenariusza w wynikach |
| `prompt` | string | — (wymagane, niepusty) | opis generowanego obrazu |
| `negative_prompt` | string | `""` | prompt negatywny; pomijany automatycznie dla pipeline'ów, które go nie wspierają |
| `width`, `height` | int | `512` | rozdzielczość obrazu |
| `steps` | int | `30` | liczba kroków próbkowania (o ile model nie nadpisuje) |
| `guidance_scale` | float | `7.5` | siła klasyfikatorowego prowadzenia (o ile model nie nadpisuje) |
| `scheduler` | string | `"euler_a"` | `ddim`, `euler`, `euler_a`, `dpmpp`, `pndm` lub `"default"` (natywny scheduler modelu — wymagany dla modeli flow-matching) |
| `seeds` | list[int] | `[42]` | lista seedów; każdy seed to osobny przebieg (powtarzalność) |
| `num_images` | int | `1` | liczba obrazów na wywołanie pipeline'u |

## Artefakty przebiegu i kolumny wyników

Każdy przebieg tworzy katalog `results/run_<timestamp>/` zawierający:

- `run_meta.json` — metadane środowiska: data, urządzenie, ścieżka konfiguracji, wersja Pythona,
  platforma i architektura, wersje pakietów (torch, diffusers, transformers, accelerate,
  safetensors, psutil) — zapisywane na starcie każdego przebiegu, co dokumentuje warunki pomiaru,
- `results.jsonl` — pełne rekordy uruchomień (jeden wiersz JSON na przebieg),
- `results.csv` — te same dane w tabeli płaskiej,
- `summary_by_model.csv` — agregacja per model,
- `images/` — obrazy `<model>_<test>_seed<seed>.png` (jeśli `save_images=true`).

Kolumny `results.jsonl` / `results.csv`: `timestamp`, `model_name`, `model_id`, `test_id`,
`prompt`, `seed`, `scheduler`, `steps`, `guidance_scale` (wartości efektywne po nadpisaniach),
`width`, `height`, `load_time_sec` (czas ładowania modelu; 0 przy trafieniu w cache procesu),
`duration_sec`, `throughput_img_s`, `avg/max_cpu_percent`, `avg/max_ram_percent`,
`avg/max_gpu_util`, `avg/max_gpu_memory_percent`, `status`, `error`, `image_path`.

Kolumny `summary_by_model.csv`: `model_name`, `runs`, `success_runs`, `avg_duration_sec`,
`avg_throughput_img_s`, `avg_gpu_util`, `avg_gpu_memory_percent`.

## Testy i CI

Zestaw 48 testów pytest (markery `unit`, `integration`, `contract`) w katalogach `tests/cli`,
`tests/domain`, `tests/runner`, `tests/monitor`, `tests/output`, z danymi w `tests/fixtures`.
Testy nie wymagają GPU ani pobierania modeli — używają atrap (DummyRunner/DummyMonitor,
FakeTorch dla `resolve_device`, sztuczne adaptery `nvidia-smi`/`rocm-smi`).

```bash
# całość
pytest

# z pomiarem pokrycia (jak w CI)
pytest --cov=app --cov=domain --cov=benchmark_runner --cov=orchestrator \
  --cov=resource_monitor --cov=outputs --cov-report=term-missing --cov-fail-under=60

# wybrana grupa
pytest -m unit
```

Potok CI (`.github/workflows/ci.yml`, GitHub Actions, push na main/master i pull requesty):

1. `ruff check` na wszystkich modułach aplikacji,
2. `python -m py_compile` (kontrola składni),
3. `pytest` z pokryciem — próg 60% (faktyczne pokrycie ~72%),
4. `python app.py validate --config config.example.json`,
5. smoke test raportu: `python app.py report --input tests/fixtures/sample_results.jsonl --output results/report_ci.md`.

## Lista kontrolna PR

- Testy: pytest (unit + integration + contract) oraz smoke testy CLI muszą przechodzić.
- Coverage: ≥ 60% globalnie i ≥ 70% dla zmienianych modułów.
- Wyniki: przy zmianie formatu JSONL/CSV/raportu dodaj lub zaktualizuj testy regresji.
- Dokumentacja: zaktualizuj README przy zmianie zachowania CLI lub konfiguracji.

## Cele projektu

- **Automatyzacja testów porównawczych** — komenda `run` i deklaratywna konfiguracja eksperymentów,
- **Powtarzalność (reprodukowalność)** — jawne seedy, scheduler, kroki, rozdzielczość oraz
  `run_meta.json` z wersjami pakietów i opisem środowiska,
- **Analiza zużycia zasobów** — monitoring CPU/RAM/GPU w trakcie generacji, agregaty avg/max,
- **Ocena łatwości wdrożenia** — metryka `load_time_sec`, jednolite ładowanie przez
  `DiffusionPipeline`, cache wag,
- **Konteneryzacja** — `Dockerfile` + nakładki Compose (powtarzalne środowisko uruchomieniowe;
  na macOS bez MPS),
- **Eksport i raportowanie** — JSONL/CSV/Markdown, agregacja per model.

## Ograniczenia

- na macOS kontener Docker nie ma dostępu do MPS — pomiary z akceleracją Metal wyłącznie na hoście,
- na MPS utylizacja GPU jest niedostępna (`avg_gpu_util = null`); raportowany odsetek pamięci GPU
  to alokacja PyTorch względem pamięci zunifikowanej,
- wszystkie pomiary wykonano na jednej maszynie (MacBook, Apple Silicon, 32 GB, MPS, natywnie na hoście);
  nakładki Compose dla CUDA/ROCm i adaptery `nvidia-smi`/`rocm-smi` są pokryte testami z atrapami,
  ale nie były sprawdzone na fizycznym sprzęcie NVIDIA/AMD,
- status `ok` oznacza tylko brak wyjątku; narzędzie nie analizuje pikseli, więc całkowicie czarny obraz
  też liczy się jako udany przebieg (w przebiegu z `results/report.md` dotyczyło to 3 z 4 obrazów SDXL-Turbo),
- metryki GPU NVIDIA zależą od `nvidia-smi`, a AMD od `rocm-smi` (narzędzia muszą być w PATH),
- pierwsze uruchomienie modelu jest długie (jednorazowe pobieranie wag; dla domyślnego zestawu
  rzędu 3–11 GB na model), kolejne uruchomienia czytają wagi z cache `.hf_cache`,
- domyślny zestaw dobrany tak, by każdy model ładował się i generował w kilkadziesiąt sekund do
  kilku minut na 32 GB MPS, bez dławienia pamięci; modele opcjonalne (Z-Image-Turbo, SD 3.5 Medium,
  FLUX.2-klein-9B) są znacznie cięższe i mogą wymagać wielu minut oraz pracować blisko granicy 32 GB,
- opcjonalne modele bramkowane (SD 3.5 Medium, FLUX.2-klein-9B) wymagają akceptacji licencji i `HF_TOKEN`;
  domyślny zestaw tokenu nie potrzebuje,
- licencje ograniczają część modeli: SANA (NVIDIA NSCL v2) i SDXL-Turbo (SAI NC Community) są
  niekomercyjne, Lumina Image 2.0 (Apache 2.0) dopuszcza komercję.

## Licencja

Copyright (c) 2026 Dawid Mazur. Wszelkie prawa zastrzeżone.

Kod jest udostępniony wyłącznie do wglądu. Korzystanie, kopiowanie, modyfikowanie i rozpowszechnianie
wymaga pisemnej zgody autora.
