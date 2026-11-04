# Raport porównawczy modeli generatywnych

Data: 2026-07-10T11:57:29.886227

## Podsumowanie

| Model | Uruchomienia | Sukcesy | Śr. czas [s] | Śr. throughput [img/s] | Śr. GPU util [%] |
|---|---:|---:|---:|---:|---:|
| sana-sprint-0.6b | 4 | 4 | 7.3391 | 0.1368 | 0.00 |
| sdxl-turbo | 4 | 4 | 13.8124 | 0.0725 | 0.00 |
| sana-1.6b | 4 | 4 | 68.3736 | 0.0147 | 0.00 |
| lumina-2 | 4 | 4 | 748.0474 | 0.0013 | 0.00 |

## Szczegóły

- model=sana-sprint-0.6b test=s1-fotorealizm-portret seed=42 status=ok time=8.125s gpu=None%
- model=sana-sprint-0.6b test=s1-fotorealizm-portret seed=43 status=ok time=7.0256s gpu=None%
- model=sana-sprint-0.6b test=s2-zgodnosc-z-promptem seed=42 status=ok time=7.0342s gpu=None%
- model=sana-sprint-0.6b test=s3-styl-ilustracyjny seed=42 status=ok time=7.1716s gpu=None%
- model=sdxl-turbo test=s1-fotorealizm-portret seed=42 status=ok time=13.9135s gpu=None%
- model=sdxl-turbo test=s1-fotorealizm-portret seed=43 status=ok time=13.9458s gpu=None%
- model=sdxl-turbo test=s2-zgodnosc-z-promptem seed=42 status=ok time=13.3776s gpu=None%
- model=sdxl-turbo test=s3-styl-ilustracyjny seed=42 status=ok time=14.0125s gpu=None%
- model=sana-1.6b test=s1-fotorealizm-portret seed=42 status=ok time=65.5568s gpu=None%
- model=sana-1.6b test=s1-fotorealizm-portret seed=43 status=ok time=67.5398s gpu=None%
- model=sana-1.6b test=s2-zgodnosc-z-promptem seed=42 status=ok time=73.0596s gpu=None%
- model=sana-1.6b test=s3-styl-ilustracyjny seed=42 status=ok time=67.3381s gpu=None%
- model=lumina-2 test=s1-fotorealizm-portret seed=42 status=ok time=731.1403s gpu=None%
- model=lumina-2 test=s1-fotorealizm-portret seed=43 status=ok time=750.1194s gpu=None%
- model=lumina-2 test=s2-zgodnosc-z-promptem seed=42 status=ok time=752.4639s gpu=None%
- model=lumina-2 test=s3-styl-ilustracyjny seed=42 status=ok time=758.4658s gpu=None%