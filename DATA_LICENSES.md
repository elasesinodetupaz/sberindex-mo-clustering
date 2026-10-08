# Лицензии данных и кода

Формулировки лицензий и авторства взяты из `README.md` (разделы «Данные», «Лицензия и цитирование», «Границы МО: источник и лицензия») и `data/geo/ATTRIBUTION.md`. Сам `ATTRIBUTION.md` не менялся.

| Каталог или файл | Источник | Лицензия | Требование |
|---|---|---|---|
| исходные наборы данных (`data/raw/`) | СберИндекс; Росстат, обработка tochno.st | в репозиторий не входят | источники, адреса и даты скачивания указаны в `README.md` |
| `data/processed/` (расчёты по данным СберИндекса, кроме файлов ниже) | расчёт проекта по данным СберИндекса | CC BY-SA 4.0 (лицензия производных данных) | авторство СберИндекса (цитирование ниже), указание изменений (данные пересчитаны проектом), ShareAlike (те же условия) |
| `data/processed/connection_valid.parquet` | очищенная выборка расстояний из `connection.parquet` СберИндекса | не публикуется (`.gitignore`); шаг 07 создаёт файл заново из `data/raw` | CC BY-SA 4.0, если файл будет распространяться |
| `data/processed/transport_network_highway_knn8.parquet` | `connection.parquet` (СберИндекс) | CC BY-SA 4.0 | производные: содержат значения исходного набора для части пар или МО; авторство по `data/geo/ATTRIBUTION.md` и цитированию ниже, ShareAlike |
| `data/processed/transport_network_railway_knn8.parquet` | `connection.parquet` (СберИндекс) | CC BY-SA 4.0 | производные: содержат значения исходного набора для части пар или МО; авторство по `data/geo/ATTRIBUTION.md` и цитированию ниже, ShareAlike |
| `data/processed/transport_network_railway_knn8_before_impute.parquet` | `connection.parquet` (СберИндекс) | CC BY-SA 4.0 | производные: содержат значения исходного набора для части пар или МО; авторство по `data/geo/ATTRIBUTION.md` и цитированию ниже, ShareAlike |
| `data/processed/layer_data_24f.parquet` | `market_access.parquet` (СберИндекс) | CC BY-SA 4.0 | производные: содержат значения исходного набора для части пар или МО; авторство по `data/geo/ATTRIBUTION.md` и цитированию ниже, ShareAlike |
| `data/processed/hypothesis_thresholds_g6g9.parquet` (+ `.sha256`), `hypothesis_series_g9.parquet` (шаг 18b; только метки типов и доли СберИндекса) | расчёт проекта по данным СберИндекса | CC BY-SA 4.0 | авторство СберИндекса (цитирование ниже), указание изменений (данные пересчитаны проектом), ShareAlike (те же условия) |
| `data/processed/hypothesis_thresholds.parquet`, `hypothesis_null_dec.parquet`, `hypothesis_thresholds_supply.parquet`, `hypothesis_results_19a.parquet`, `hypothesis_results_19b.parquet`, `hypothesis_results_supply.parquet`, `hypothesis_verdicts.parquet`, `hypothesis_summary.parquet`, `type_portraits.parquet`, `composition_robustness.parquet` (шаги 18–26; файлы `.sha256` рядом) | расчёт проекта по данным СберИндекса и Росстата (БД ПМО, обработка tochno.st) | CC BY-SA 4.0 и CC BY 4.0 | содержит данные обоих источников, действуют условия обеих лицензий (CC BY-SA 4.0 и CC BY 4.0); авторство обоих источников (цитирование ниже), указание изменений, ShareAlike |
| `data/geo/` | СберИндекс, «Данные о границах и преобразованиях муниципальных образований» (скачано 04.10.2026); построен по данным Росстата и OpenStreetMap | CC BY-SA 4.0 (по официальной странице набора) | авторство, указание изменений (упрощение геометрии, округление, замена атрибутов), ShareAlike; атрибуция Contains data © OpenStreetMap contributors; подробности: `data/geo/ATTRIBUTION.md` |
| `data/external/rosstat_mo_2023_2025.parquet`, `data/external/rosstat_retail_2023_2024.parquet` | Росстат, база данных «Показатели муниципальных образований» (БД ПМО), обработка «Если быть точным» (tochno.st) | Creative Commons BY 4.0 | авторство (цитирование ниже) |
| `data/external/federal_districts.csv`, `data/external/portrait_sources.csv` | составлено вручную; внешние публикации, используемые только как контекст | см. `data/external/README.md` и `README.md` | — |
| `site/index.html` | автор проекта; встроенные данные и границы МО — расчёт проекта по данным СберИндекса, Росстата и справочнику границ | код страницы MIT (`LICENSE`); встроенные данные и границы CC BY-SA 4.0 | сохранение уведомления об авторских правах; для встроенных данных: авторство, указание изменений, ShareAlike |
| `notebooks/figures/*` | рисунки, полученные из данных СберИндекса | CC BY-SA 4.0 | авторство СберИндекса (цитирование ниже), указание изменений (данные пересчитаны проектом), ShareAlike (те же условия) |
| код проекта (`src/`, `run_all.sh`, `config.yaml`, `site/tests/`) | автор проекта | MIT (`LICENSE`) | сохранение уведомления об авторских правах |

## Цитирование

Данные СберИндекса (CC BY-SA 4.0):

- Потребительские безналичные расходы на уровне муниципальных образований по категориям трат. СберИндекс. Данные доступны по адресу https://sberindex.ru/ru/research/data-sense-opisanie-nabora-dannikh-khakatona-sberindeksa-po-munitsipalnim-dannim (данные скачаны 27.09.2026).
- Индекс доступности рынков на уровне муниципальных образований. СберИндекс. Данные доступны по тому же адресу (данные скачаны 27.09.2026).
- Автодорожные и железнодорожные связи между муниципальными образованиями. СберИндекс. Данные доступны по тому же адресу (данные скачаны 27.09.2026).
- Данные о границах и преобразованиях муниципальных образований. СберИндекс. Данные доступны по адресу https://sberindex.ru/ru/research/dataset-borders-and-changes-of-municipalities (данные скачаны 04.10.2026).

Данные Росстата (CC BY 4.0):

- Муниципальная статистика России с 2005 года // Росстат; обработка «Если быть точным», 2024. Условия использования: Creative Commons BY 4.0. URL: https://tochno.st/datasets/bdmo (данные скачаны 29.09.2026).

Исходные наборы данных (`data/raw`) в репозиторий не входят.
