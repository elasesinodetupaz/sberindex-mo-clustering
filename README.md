# Экономическая структура муниципалитетов России: кластеризация динамической атрибутированной сети

Исследовательский проект для конкурса СберИндекса, трек «Кластеризация». Цель — выявить
экономические типы муниципальных образований (МО) по структуре потребительских расходов,
сравнить экономическую близость МО с транспортной и проследить изменение типов во времени
(январь 2023 – декабрь 2024).

Методология вкратце:

- **Экономическая сеть:** доли 5 категорий расходов (продовольствие, здоровье, общепит,
  транспорт, маркетплейсы) → z-score внутри месяца → косинусное сходство → kNN-граф (k = 8).
- **Транспортная сеть:** kNN-графы (k = 8) по автодорожным и железнодорожным расстояниям.
- **market_access** — атрибут узла, а не часть сети.
- **Кластеризация:** KMeans (k = 6) в пространстве признаков и Louvain на экономической сети;
  метрики MQ, AVI, AVU, SW, CH, S_Dbw, prediction strength; анализ устойчивости и отслеживание типов во времени.
- **Выбор k = 6:** k = 6 выбран не статистическим критерием. Внутренние показатели (SW, CH/N, S_Dbw после нормализации, gap statistic, prediction strength, устойчивость по времени) монотонно ухудшаются или смещены к малым k, внешние (связь с регионами, market_access, рёбрами highway) растут с k. k = 6 — компромисс между устойчивостью и связью с внешними данными; на нём выделяется удалённая группа (111 МО; семь регионов дают 87 МО: Республика Саха (Якутия) — 26, Сахалинская область — 16, Амурская область — 14, Хабаровский край — 10, Забайкальский край — 8, Приморский край — 7, Камчатский край — 6; всего 18 регионов); в графе highway kNN8 она не образует единой связной области — 14 компонент, крупнейшая 63 МО; устойчива при k = 7–8 (Жаккар 0.83 и 0.86). Транспортный тип k = 7 слабо воспроизводится и собран в основном из двух кластеров k = 6 (84 + 78 из 187 МО). Слабое место k = 6: удалённая группа — наименее воспроизводимый тип (PS 0.62 (точечно; разница с типом 5 — 0.64 — незначима, p = 0.24), во времени 15 из 23 месяцев).
  До 2026-09-28 каноническим было k = 7; его результаты сохранены с суффиксом `_k7_superseded`
  (подробно: `notebooks/10c_final_clustering_k6.md`, разделы «Обоснование» и «История решения»).
- **Вариант признаков:** Выбран вариант A — доли пяти категорий от «Все категории». На внешних данных A и C равноценны (значимых различий нет), B хуже (значимо ниже A по AMI регион (p = 0.0014), доля рёбер highway внутри (p = 0.00043), z рёбер highway (p = 0.015; не проходит поправку Бонферрони); критерий Уилкоксона по 24 месяцам, p < 0.05, без поправки на множественные сравнения (6 показателей × 2 пары)). Медианный ARI соседних месяцев A / B / C = 0.569 / 0.541 / 0.525; по парному критерию Уилкоксона различие не значимо ни против B (p = 0.21), ни против C (p = 0.065); число проблемных месяцев между вариантами не сопоставляется: критерий зависит от геометрии признаков. «Прочее» (траты вне пяти категорий) входит в A неявно — сумма пяти долей меньше 1, — а «Все категории» охватывает только безналичные траты. Подробно: `notebooks/15_robustness_checks_2.md`, `notebooks/15e_external_validity.md`.

## Быстрый старт

1. **Python.** Версия Python в окружении автора: 3.13.5 (файл `requirements.txt`); проверено на 3.13.5; другие версии не проверялись.
2. **Окружение.**

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

   `requirements.txt` — прямые зависимости; `requirements-lock.txt` — полный список пакетов окружения автора (`pip freeze`).
3. **Данные.** Адреса и даты скачивания наборов указаны в разделе «Данные» ниже. Исходные наборы в репозиторий не входят: положите их в `data/raw/` (`consumption.parquet`, `market_access.parquet`, `connection.parquet`, `Данные_СберИндекс_лицензия.pdf`, `rosstat/`). Файл `data/raw/territories.parquet` создаёт шаг 09 (в таблице шагов ниже — шаг 7). Справочник МО лежит вне репозитория, в каталоге `~/Documents/municipal_dict/` (константа `MUNICIPAL_DICT_PATH` в `src/09_territory_names.py` и `src/17_rosstat_extract.py`; для шага 24б каталог задан в `config.yaml`, `MUNICIPAL_DICT_DIR`). Нужны два файла: `t_dict_municipal_districts.xlsx` и `t_dict_municipal_districts_poly.gpkg`. Оба лежат в одном архиве `https://www.sberbank.com/common/files/t_dict_municipal.rar` (sha256 архива и gpkg заданы в `config.yaml`, раздел `step24b_geometry`; состав архива проверен 08.10.2026: в нём оба файла, их sha256 совпали с файлами автора). Для скачивания нужны сертификаты Минцифры из `certs/` (см. `certs/README.md`) и `bsdtar` для распаковки. Если файла gpkg в каталоге нет, шаг 24б (`src/24b_geometry.py`) сам скачивает архив с проверкой sha256 и распаковывает его в этот каталог. В `run_all.sh` шаг 24б стоит после шагов 09 и 17, которые читают xlsx и сами его не скачивают (при отсутствии файла они завершаются ошибкой чтения), поэтому при чистой установке распакуйте архив в этот каталог до `bash run_all.sh`.
4. **Запуск.**

```bash
bash run_all.sh
```

   Время чистого прогона будет указано после замера.
5. **Лендинг.** Сборка и проверка страницы `site/index.html` (для проверки нужен Chrome):

```bash
.venv/bin/python src/24e_landing.py
.venv/bin/python src/24e_landing_test.py
```

Зависимости и список пакетов: `requirements.txt` — pandas, pyarrow, numpy, scikit-learn, networkx, scipy, matplotlib, shapely, tabulate (таблицы в отчётах), openpyxl (чтение справочника xlsx), s-dbw (индекс S_Dbw), PyYAML (чтение `config.yaml`), pillow, joblib, threadpoolctl.

Каждый шаг — отдельный скрипт `src/NN_*.py`, его можно запускать и по отдельности: `.venv/bin/python src/03_category_shares.py`. Пути — константы в начале каждого скрипта; промежуточные результаты пишутся в `data/processed/`, отчёты — в `notebooks/`.

## Порядок запуска и таблица шагов

**Порядок запуска — не строго по номерам**: шаг 09 (названия МО) нужен шагам 07–08; шаги 10d и
12a/12 (для k = 6 и для прежнего k = 7 — обоснование выбора) — шагу 10c; 10c (канонические
метки KMeans) — шагам 11–16; в конце 10c запускается повторно, чтобы итоговый документ включил
результаты шагов 14–15. `run_all.sh` учитывает эти зависимости. Каноническое k задано в
`config.yaml` (`FINAL_K`; читается через `src/config.py`); шаги 12a и 12 принимают `--k`.
`FINAL_K` одним значением менять нельзя: связанные параметры (`ROBUST2_K`, `ROBUST2_TARGET_K`,
`ROBUST_K_COMPARE`, номера кластеров `ROBUST2_*_CLUSTER`) привязаны к k = 6; при
несогласованности `config.py` остановит запуск.
Число 6 (и соседние 5, 7) также записано напрямую в тексте и логике шагов 10c, 15, 15b и 16; при
реальной смене канона эти места нужно править вручную.

`run_all.sh` задаёт исполняемую последовательность, таблица группирует вызовы одного скрипта.

| № | Скрипт | Что делает | Основные результаты |
|---|---|---|---|
| 1 | `01_data_overview.py` | обзор исходных таблиц, полнота, пересечение id | `notebooks/01_data_overview.md` |
| 2 | `02_missing_data.py` | рабочая выборка МО (24 мес. + market_access) | `data/processed/valid_territories.parquet` |
| 3 | `03_category_shares.py` | доли 5 категорий от «Все категории» | `category_shares.parquet` |
| 4 | `04_economic_network.py` | сеть на сырых долях (отклонённый вариант, для сравнения) | `economic_network_2024_12.parquet` |
| 5 | `05_economic_network_std_knn.py` | экономическая сеть дек. 2024: z-score + cosine + kNN8 | `economic_network_2024_12_std_knn8.*` |
| 6 | `06_economic_networks_monthly.py` | экономические сети за все 24 месяца | `economic_networks/{YYYY-MM}_std_knn8.parquet` |
| 7 | `09_territory_names.py` | справочник названий МО (последняя версия) | `data/raw/territories.parquet` |
| 8 | `07_transport_networks.py` | очистка пар, достройка ж/д-нулей, kNN8 highway/railway | `connection_valid.parquet`, `transport_network_*_knn8.*`, `railway_degree.parquet` |
| 9 | `08_economic_vs_transport.py` | экономическая vs транспортная близость, центральность | `notebooks/08_economic_vs_transport.md` |
| 10 | `10_kmeans_k_selection.py` | метрики KMeans для k = 2..15 (inertia, SW, CH, S_Dbw) | `notebooks/10_kmeans_k_selection.md`, `notebooks/figures/*.png` |
| 11 | `10b_kmeans_cluster_profiles.py` | профили кластеров для k = 4–8 | `kmeans_labels_k*_2024_12.parquet` |
| 12 | `10d_prediction_strength.py` | prediction strength (Tibshirani & Walther, 2005), k = 2..15 | `prediction_strength_*.parquet`, `notebooks/figures/prediction_strength_by_k.png` |
| 13 | `12a_kmeans_runs.py --k 4…8` | 100 запусков KMeans на месяц, официальное разбиение, уверенность МО | `kmeans_labels_k{k}/`, `kmeans_k{k}_runs_*.parquet` |
| 14 | `12_kmeans_temporal_tracking.py --k 4…8` | сопоставление кластеров во времени, переходы, пограничные МО | `kmeans_k{k}_trajectories.parquet`, `kmeans_k{k}_membership_confidence.parquet`, `notebooks/12_kmeans_temporal_tracking_k{k}.md` |
| 15 | `10c_final_clustering_doc.py` | фиксация k = 6, обоснование, проверка плоского минимума | `kmeans_labels_final.parquet`, `kmeans_labels_final_k7_superseded.parquet`, `notebooks/10c_final_clustering_k6.md` |
| 16 | `11_louvain.py` | метрики MQ/AVI/AVU, предварительный Louvain | `notebooks/11_louvain_metrics.md` |
| 17 | `11b_louvain_stability.py` | Louvain: 100 запусков, лучший по MQ, сравнение с KMeans | `louvain_labels_2024_12.parquet`, `notebooks/11b_*.md` |
| 18 | `11c_louvain_core_stability.py` | устойчивость сообществ по матрице совместной встречаемости | `notebooks/11c_*.md` |
| 19 | `13_louvain_monthly_check.py` | Louvain по всем месяцам и сравнение с KMeans | `notebooks/13_louvain_monthly_check.md` |
| 20 | `14a_feature_spec_check.py` | чувствительность канона к определению долей (A/B/C) | `robust_feature_spec*.parquet` |
| 21 | `14b_k_over_time.py` | PS, SW, S_Dbw norm для k = 2..10 по всем 24 месяцам | `robust_k_by_month.parquet`, `robust_ps_clusters_by_month.parquet` |
| 22 | `14_robustness_report.py` | сводка проверок устойчивости выбора k | `notebooks/14_robustness_checks.md` |
| 23 | `16_k_stability_curve.py` | кривая стабильности по k = 4–8: 12 показателей в 8 группах (а)–(з) — смена доминирующего кластера во времени, число МО с долей основного кластера < 60%, проблемные месяцы, ARI соседних месяцев, уверенность сопоставления с декабрём, согласие с Louvain, «подвижные» МО у минимума inertia, prediction strength; изменения показателей при переходах k → k+1 | `k_stability_curve.parquet`, `notebooks/16_k_stability_curve.md`, `notebooks/figures/k_stability_curve.png` |
| 24 | `15a_feature_spec_full.py` | варианты признаков A/B/C при k = 6 по 24 месяцам: лучшие из 100 разбиений, SW, CH/N и PS при k = 4–8, проблемные месяцы | `robust2_spec_labels.parquet`, `robust2_spec_k.parquet`, `robust2_spec_months.parquet` |
| 25 | `15b_k_contingency.py` | таблицы сопряжённости по k (4→6, 5→6, 6→7), лучший Жаккар кластеров k = 6 | `robust2_contingency.parquet`, `robust2_best_jaccard.parquet` |
| 26 | `15c_louvain_resolution.py` | Louvain при разных resolution (в т. ч. K = 6) и kNN = 5/8/12, сравнение с каноном | `robust2_louvain_resolution.parquet`, `robust2_louvain_knn*.parquet` |
| 27 | `15d_gap_statistic.py` | gap statistic для k = 2–10 (декабрь 2024) | `robust2_gap.parquet` |
| 28 | `15_robustness_report_2.py` | сводный отчёт батча 2 (сверка с ручными расчётами, пп. 1–4) | `notebooks/15_robustness_checks_2.md` |
| 29 | `15e_external_validity.py` | внешняя валидность вариантов A/B/C и k = 4–8: log(market_access), регионы, рёбра highway | `robust2_external_validity.parquet`, `notebooks/15e_external_validity.md` |
| 30 | `15f_k_tradeoff.py` | иллюстрация компромисса по k (устойчивость против внешней связи), раздел в отчёте 15e | `notebooks/figures/k_tradeoff.png` |
| 31 | `17_rosstat_extract.py` | показатели Росстата (население, возраст, зарплата, работники по ОКВЭД-2) для 2004 МО, склейка по ОКТМО; только для проверки гипотез | `data/external/rosstat_mo_2023_2025.parquet` |
| 32 | `18_hypothesis_thresholds.py` | предрегистрация порогов для гипотез Г1–Г4, Г7, Г8: нулевые распределения η²_H и ρ (перестановки, уровни «общий» и «внутри регионов», 25 разбиений), контроли; функции — `src/hypothesis_tools.py` | `hypothesis_thresholds.parquet` (+ `.sha256`), `hypothesis_null_dec.parquet`, `notebooks/18_hypothesis_thresholds.md` |
| 33 | `18b_hypothesis_thresholds_g6_g9.py` | предрегистрация порогов для Г6 (доля переходов в ближайшие типы; нуль — случайные конечные типы) и Г9 (ρ между долей сменивших тип и общим сдвигом долей; перестановки и циклические сдвиги) | `hypothesis_thresholds_g6g9.parquet` (+ `.sha256`), `hypothesis_series_g9.parquet`, `notebooks/18b_hypothesis_thresholds_g6_g9.md` |
| 34 | `19a_hypothesis_tests_kw.py` | проверка Г1, Г4, Г7, Г8 (Краскел–Уоллис по типам) по замороженным порогам 18a: η²_H, p, месяцы выше порога, эталон — федеральные округа, порядок типов (Г1), Δ с бутстрепом (Г7); без вердиктов (шаг 19c) | `hypothesis_results_19a.parquet` (+ `.sha256`), `notebooks/19a_hypothesis_results.md`, `data/external/federal_districts.csv` (вход) |
| 35 | `19b_hypothesis_tests_corr_trans.py` | проверка Г2, Г3 (ρ Спирмена), Г6 (доля переходов в ближайшие типы: точный p, бутстреп по МО) и Г9 (ρ между долей сменивших тип и сдвигом долей) по замороженным порогам 18a/18b; без вердиктов (шаг 19c) | `hypothesis_results_19b.parquet` (+ `.sha256`), `notebooks/19b_hypothesis_results.md` |
| 36 | `19c_hypothesis_verdicts.py` | вердикты по гипотезам: поправка Холма по семействам и уровням, четыре ступени «общий / внутри регионов», Г4 по отраслям, размер и эталон «федеральные округа»; без пересчёта | `hypothesis_verdicts.parquet`, `notebooks/19_hypothesis_verdicts.md` |
| 37 | `20_rosstat_retail_extract.py` | показатели розницы и общепита Росстата (площадь торгового зала, магазины, места и объекты общепита) для 2004 МО, IV квартал 2023–2024, склейка по ОКТМО как в шаге 17; только извлечение | `data/external/rosstat_retail_2023_2024.parquet` |
| 38 | `21_supply_thresholds.py` | предрегистрация порогов для дополнительных гипотез Г10–Г13 (предложение розницы и общепита на 1000 жителей): нулевые распределения ρ (24 месяца) и η²_H доли и её остатка (25 разбиений), отрицательные контроли | `hypothesis_thresholds_supply.parquet` (+ `.sha256`), `notebooks/21_supply_thresholds.md` |
| 39 | `22_supply_tests.py` | проверка дополнительных гипотез Г10–Г13 по замороженным порогам 21: ρ Спирмена (Г10, Г11), η²_H доли и остатка с Δ и бутстрепом (Г12, Г13), Холм по {Г10, Г12, Г13}, Г11 — контроль; вердикты; разведочно — частная корреляция с поправкой на размер МО | `hypothesis_results_supply.parquet` (+ `.sha256`), `notebooks/22_supply_results.md` |
| 40 | `23_type_portraits.py` | генератор портретов типов: состав, доли расходов, признаки профиля (σ по долям и по log(доля/«прочее»)) с пометками «держится» / «зависит от записи», имена типов по устойчивым признакам со вторым ярлыком по населению и market_access, пометка пограничных признаков имени, показатели Росстата, надёжность, сравнение с МО того же размера; описательный шаг без p-значений, тестов и вердиктов, все числа считаются из данных проекта или цитируются из замороженных результатов; сверка с контрольными значениями | `type_portraits.parquet` (long: type, block, metric, value, n, source_files), `notebooks/23_type_portraits.md`, `data/external/portrait_sources.csv` (вход) |
| 41 | `25_composition_robustness.py` | устойчивость типологии и связей к композиционной природе долей (разведочно, без p-значений, порогов и вердиктов): воспроизведение канона и ARI между seed, KMeans на clr-кодировках (ARI с каноном, SW), PC1 z-score долей и «прочее» (rest), rest по типам, Г2/Г3 и η²_H по долям и по alr = ln(доля / rest); сверка с контрольными значениями | `composition_robustness.parquet` (long: block, metric, variant, level, value, n, n_regions), `notebooks/25_composition_robustness.md` |
| 42 | `26_hypothesis_summary.py` | сводка проверок внешней валидности: только чтение замороженных результатов шагов 18–22, шага 25 и `type_portraits.parquet` шага 23; результат записан как «отличимо / не отличимо от случайного» с условной подписью размера (η²_H, \|ρ\|), отдельный блок Г10–Г13, устойчивость к кодированию долей, профили типов с пометкой границы порога; без пересчёта, тестов, порогов и вердиктов | `hypothesis_summary.parquet` (long: block, item, level, metric, value, n, q025, q975, result, size, source), `notebooks/26_hypothesis_summary.md` |
| 43 | `27_mirkin_rule.py` | описание типов по правилу интерпретации кластеров (Alvandyan, T.A., Shalileh, S., Dokl. Math. 110 (Suppl 1), S236–S250 (2024), правило Миркина): относительное отклонение среднего доли по типу от общего среднего, то же по медианам, сравнение с σ по долям шага 23; разведочно, вне предрегистрации, без тестов и вердиктов | `mirkin_rule_27.parquet` (long: тип, категория, показатель, значение, n, источник_файл, источник_блок), `notebooks/27_mirkin_rule.md` |
| 44 | `28_interpretation_checks.py` | проверка интерпретаций (разведочно, вне предрегистрации): цитаты описания набора о привязке расходов и отнесении покупок к категориям, независимый пересчёт чисел прежнего отчёта с ожиданиями и допусками из config (STOP при расхождении), состав типов по виду МО, ρ Спирмена на уровне МО с бутстрепом по МО, медианы доли транспорта по типам в группах размера | `interp_checks_28.parquet`, `notebooks/28_interp_checks.md` |
| 45 | `24b_geometry.py` | границы МО для карты: справочник границ (при отсутствии файла скачивание по https с сертификатами Минцифры из `certs/`), упрощение покрытия `shapely.coverage_simplify`, отчёт, подпись источника | `data/geo/mo_national.geojson`, `data/geo/mo_cities.geojson`, `data/geo/geometry_report.md`, `data/geo/geometry_report.json`, `data/geo/ATTRIBUTION.md` |
| 46 | `24c_assign_outside.py` | типы МО вне рабочей выборки: ближайший центроид канонического разбиения для МО из `consumption.parquet`, не вошедших в выборку, при полных пяти долях за декабрь 2024 и расстоянии не больше максимума по выборке; присвоенные типы не входят в статистики шагов 10–28 | `types_outside_sample.parquet`, `notebooks/24c_types_outside_sample.md` |
| 47 | `24g_last_month_types.py` | тип МО без данных за декабрь 2024 по последнему месяцу с полными пятью категориями: ближайший центроид канонического разбиения по параметрам месяца, месяц допускается по согласию с месячными метками на выборке; отдельный класс, не входит в статистики шагов 10–28 и в карту 24d | `types_last_month.parquet`, `month_agreement_24g.parquet`, `notebooks/24g_last_month_types.md` |
| 48 | `24d_map.py` | статическая карта типов МО (фигура 2): тип из выборки цветом, тип вне выборки цветом со штриховкой, МО без типа светлой штриховкой; врезки Москвы и Санкт-Петербурга; технические имена типов; подпись источника границ из `data/geo/ATTRIBUTION.md` | `notebooks/figures/24_fig2_map.png`, `notebooks/figures/24_fig2_map.svg`, `data/processed/figure_data_24d.parquet`, `notebooks/24d_map.md` |
| 49 | `24f_layer_data.py` | данные слоёв лендинга по МО и регионам: класс МО (выборка, тип вне выборки, без типа, тип по последнему месяцу), доступность рынков, число месяцев в каноническом типе, покрытие регионов; только чтение готовых файлов, без моделей | `layer_data_24f.parquet`, `region_coverage_24f.parquet`, `notebooks/24f_layer_data.md` |
| 50 | `24_figures.py` | фигуры 1, 3, 4, 5 (профили типов, маркетплейсы между уровнями, размер МО, надёжность типов): числа только из parquet шагов 19–28 со сверкой с исходными файлами; технические имена типов из шага 23 | `notebooks/figures/24_fig1_profiles.{png,svg}`, `notebooks/figures/24_fig3_marketplaces.{png,svg}`, `notebooks/figures/24_fig4_size.{png,svg}`, `notebooks/figures/24_fig5_reliability.{png,svg}`, `data/processed/figure_data_24.parquet`, `notebooks/24_figures.md` |
| 51 | `24e_landing.py` | каркас лендинга: один автономный HTML-файл с данными, интерактивной картой типов МО и карточками типов; проза читается из `site/content/texts.json`; таблицы проверок разбираются из отчёта шага 26 и сверяются с `hypothesis_summary.parquet`; только чтение готовых файлов | `site/index.html`, `site/README.md`, `notebooks/24e_landing_check.md` |
| 52 | `31_method_comparison.py` | матрица «метод × шесть индексов качества» (SW, CH, S_Dbw, MQ, AVI, AVU) для восьми методов на декабре 2024, ранги, суммарный ранг, попарное ARI; описательно | `notebooks/31_method_comparison.md`, `notebooks/figures/31_method_comparison.png`, `data/processed/method_comparison_31.parquet` |
| 53 | `32_edge_rules.py` | сравнение правил построения рёбер (kNN по косинусу, корреляции рядов, лаговой корреляции, расстояниям highway и railway): структура сетей, Louvain, согласие с каноническими типами, индексы качества, чувствительность к весам; описательно | `notebooks/32_edge_rules.md`, `notebooks/figures/32_edge_rules.png`, `data/processed/edge_rules_32.parquet` |
| 54 | `10c_final_clustering_doc.py` (повторно) | итоговый документ 10c с данными шагов 14–15 | `notebooks/10c_final_clustering_k6.md` |

Вспомогательные модули: `src/config.py` (читает гиперпараметры из `config.yaml` в корне: канонический месяц, k и др.), `src/network_utils.py` (стандартизация, kNN-графы),
`src/partition_metrics.py` (MQ, AVI, AVU), `src/territory_names.py` (подгрузка названий МО).

## Критерий конкурса → где в репозитории

| Критерий (вес) | Где в репозитории |
|---|---|
| Простота объяснения методологии (15%) | `README.md`, `notebooks/10c_final_clustering_k6.md`, `notebooks/23_type_portraits.md` |
| Построение сетевой и атрибутивной структуры (15%) | `src/03_category_shares.py`, `src/05_economic_network_std_knn.py`, `src/06_economic_networks_monthly.py`, `src/07_transport_networks.py`, `notebooks/05_economic_network_2024_12_std_knn8.md`, `notebooks/07_transport_networks.md`, `notebooks/08_economic_vs_transport.md`, `data/processed/economic_networks`, `notebooks/32_edge_rules.md` |
| Сравнение методов (15%) | `src/31_method_comparison.py`, `notebooks/31_method_comparison.md`, `notebooks/figures/31_method_comparison.png`, `data/processed/method_comparison_31.parquet`, `notebooks/11b_louvain_stability.md`, `notebooks/13_louvain_monthly_check.md`, `notebooks/32_edge_rules.md` |
| ICVI: SW, CH, S_Dbw, AVI, AVU, MQ (15%) | `src/partition_metrics.py`, `src/10_kmeans_k_selection.py`, `src/11_louvain.py`, `notebooks/10_kmeans_k_selection.md`, `notebooks/11_louvain_metrics.md`, `notebooks/31_method_comparison.md` |
| Интерпретация, ясность, воспроизводимость, обоснованность (30%) | `notebooks/10c_final_clustering_k6.md`, `notebooks/14_robustness_checks.md`, `notebooks/15_robustness_checks_2.md`, `notebooks/15e_external_validity.md`, `notebooks/19_hypothesis_verdicts.md`, `notebooks/23_type_portraits.md`, `notebooks/26_hypothesis_summary.md`, `docs/interpretation.md`, `docs/type_names.md`, `docs/decisions.md`, `run_all.sh`, `config.yaml`, `requirements.txt`, `requirements-lock.txt` |
| Визуализация (10%) | `notebooks/figures/24_fig1_profiles.png`, `notebooks/figures/24_fig2_map.png`, `notebooks/figures/24_fig3_marketplaces.png`, `notebooks/figures/24_fig4_size.png`, `notebooks/figures/24_fig5_reliability.png`, `notebooks/figures/31_method_comparison.png`, `notebooks/24d_map.md`, `notebooks/24_figures.md`, `site/index.html`, `site/README.md` |

## Данные

Рабочая выборка — 2004 МО с полной историей за 24 месяца и известным индексом доступности
рынков (шаг 02).

| Файл | Содержание | Источник |
|---|---|---|
| `data/raw/consumption.parquet` | средние безналичные расходы жителей МО по категориям, руб., помесячно | СберИндекс |
| `data/raw/market_access.parquet` | индекс доступности рынков 2024, 0–1000 | СберИндекс |
| `data/raw/connection.parquet` | автодорожные и ж/д расстояния между МО, км | СберИндекс |
| `data/raw/Данные_СберИндекс_лицензия.pdf` | описание наборов данных и лицензия | СберИндекс |
| `data/raw/territories.parquet` | названия, типы, регионы и координаты центров МО (создаётся шагом 09) | версионный справочник МО СберИндекса |
| `data/raw/rosstat/population.zip`, `data/raw/rosstat/employment_wages.zip` | разделы 31 «Население» и 32 «Занятость и заработная плата» БД ПМО (в репозиторий не входят) | Росстат, обработка tochno.st |
| `data/external/rosstat_mo_2023_2025.parquet` | население, возрастная структура, зарплата и численность работников организаций по 2004 МО за 2023–2025 (создаётся шагом 17) | Росстат, обработка tochno.st |
| `data/raw/rosstat/retail/data_section2_112_v20250918.parquet` | раздел 2 «Розничная торговля и общественное питание» БД ПМО (в репозиторий не входит; рядом тот же раздел в csv) | Росстат, обработка tochno.st |
| `data/external/federal_districts.csv` | регион → федеральный округ по составу 8 округов на 2024 год (вход шагов 19a и 23) | составлено вручную |
| `data/external/portrait_sources.csv` | внешние публикации, использованные только как контекст в портретах типов (шаг 23): id, что утверждается, источник, адрес, что прочитано, уровень, fact | публикации, см. таблицу |
| `data/external/rosstat_retail_2023_2024.parquet` | площадь торгового зала и число магазинов, места и объекты общепита по 2004 МО за 2023 (основной год) и 2024 (чувствительность) (создаётся шагом 20); Санкт-Петербурга в этих показателях нет, в 2024 году выпадают ещё 15 регионов | Росстат, обработка tochno.st |

Файлы в `data/raw/` не изменяются скриптами (кроме `territories.parquet`, который создаёт шаг 09).
Данные Росстата используются только для проверки гипотез о готовых типах: метки кластеров и
отчёты шагов 10–16 от них не зависят. Показатели, склейка по ОКТМО и ограничения описаны в
`data/external/README.md`.

Справочник МО (`t_dict_municipal_districts.xlsx`) и его полигональный слой (`t_dict_municipal_districts_poly.gpkg`)
в репозиторий не входят: каталог задан константой `MUNICIPAL_DICT_PATH` в `src/09_territory_names.py` (сейчас
`~/Documents/municipal_dict/`). Полигоны использует шаг 24б (`src/24b_geometry.py`; если gpkg в каталоге нет, он скачивает
архив, см. `certs/README.md`). По метаданным справочника координаты центров заданы только для МО вне
городов федерального значения.

### Лицензия и цитирование

Данные СберИндекса распространяются по лицензии **CC BY-SA 4.0**. Цитирование (по описанию
набора данных):

- Потребительские безналичные расходы на уровне муниципальных образований по категориям трат.
  СберИндекс. Данные доступны по адресу
  https://sberindex.ru/ru/research/data-sense-opisanie-nabora-dannikh-khakatona-sberindeksa-po-munitsipalnim-dannim
  (данные скачаны 27.09.2026).
- Индекс доступности рынков на уровне муниципальных образований. СберИндекс. Данные доступны по
  тому же адресу (данные скачаны 27.09.2026).
- Автодорожные и железнодорожные связи между муниципальными образованиями. СберИндекс. Данные
  доступны по тому же адресу (данные скачаны 27.09.2026).

Версионный справочник муниципальных образований России (СберИндекс) построен по данным ОКТМО и
«Численности населения РФ по муниципальным образованиям» (Росстат) и OpenStreetMap (по странице набора);
лицензия справочника границ CC BY-SA 4.0 (по официальной странице набора, см. `data/geo/ATTRIBUTION.md`).

Данные Росстата (БД ПМО в обработке «Если быть точным»; разделы 2, 31 и 32) распространяются по лицензии
**CC BY 4.0**. Цитирование:

- Муниципальная статистика России с 2005 года // Росстат; обработка «Если быть точным», 2024.
  Условия использования: Creative Commons BY 4.0. URL: https://tochno.st/datasets/bdmo
  (данные скачаны 29.09.2026).

### Внешние публикации в портретах типов

Портреты типов (шаг 23, `notebooks/23_type_portraits.md`) ссылаются на внешние публикации S0–S2, S4–S12 только как на
контекст: расшифровка (источник, адрес, что именно прочитано, уровень надёжности, цитируемый факт) — в
`data/external/portrait_sources.csv`. Числа из этих публикаций в расчётах проекта не используются.

### Методическая литература

- Shalileh S., Antonov E. A., Tsyplakova D. A. Cluster Validity across Attribute and Network
  Spaces: Empirical Benchmarks for Attributed Networks Clustering // Doklady Mathematics. 2025.
  Vol. 112, No. 3. P. 553–564. DOI 10.1134/S1064562425700589 — формулы AVI/AVU (18)–(21),
  рекомендации CH/N и baseline-нормализации S_Dbw.
- Tibshirani R., Walther G. Cluster Validation by Prediction Strength // Journal of Computational
  and Graphical Statistics. 2005. Vol. 14, No. 3. P. 511–528 — prediction strength (шаг 10d).
- Akhanli S. E., Hennig C. Comparing clusterings and numbers of clusters by aggregation of
  calibrated clustering validity indexes. arXiv:2002.01822 (2020) — смещение prediction strength к
  меньшему числу кластеров.
- Halkidi M., Vazirgiannis M. Clustering Validity Assessment: Finding the Optimal Partitioning of a
  Data Set // Proc. IEEE ICDM. 2001 — индекс S_Dbw.

## Границы МО: источник и лицензия

Геометрия МО для карты (`data/geo/mo_national.geojson`, `data/geo/mo_cities.geojson`) получена из справочника границ
СберИндекса шагом 24б (`src/24b_geometry.py`).

- Источник: Данные о границах и преобразованиях муниципальных образований. СберИндекс. Данные доступны по адресу
  https://sberindex.ru/ru/research/dataset-borders-and-changes-of-municipalities (данные скачаны 04.10.2026).
- Лицензия: CC BY-SA 4.0 (по официальной странице набора), текст https://creativecommons.org/licenses/by-sa/4.0/legalcode.
- Файлы в `data/geo/` являются адаптацией: выбраны полигоны, действующие в декабре 2024, геометрия упрощена
  (`shapely.coverage_simplify`, допуск 0.005° для карты страны и 0.0005° для врезок Москвы и Санкт-Петербурга),
  координаты округлены. Файлы распространяются на тех же условиях (CC BY-SA 4.0).
- Набор построен СберИндексом по данным Росстата и OpenStreetMap; условия лицензии OpenStreetMap (ODbL) в проекте
  не проверялись. Атрибуция: Contains data © OpenStreetMap contributors.
- Известные ошибки исходного справочника (перекрытия соседних полигонов, неверные рёбра) не исправлялись; числа и
  перечни: `data/geo/ATTRIBUTION.md` и `data/geo/geometry_report.md`.
- Скачивание архива справочника по https требует сертификатов Минцифры: `certs/README.md` (отпечатки, происхождение,
  правило подстановки только в запрос).

## Лицензии и авторство

- Код проекта: MIT, файл `LICENSE`.
- Производные данные проекта: CC BY-SA 4.0; данные Росстата (БД ПМО в обработке tochno.st): CC BY 4.0. Сводная таблица: `DATA_LICENSES.md`.
- Исходные наборы данных в репозиторий не входят; их источники и лицензии указаны в таблице данных и в разделе «Лицензия и цитирование».
- Границы МО и справочник: `data/geo/ATTRIBUTION.md`.

## Что не реализовано

- Пороговое агрегирование Алескерова не реализовано (`notebooks/31_method_comparison.md`, строка 42).
- Правило R2 (порог по косинусу сырых долей) в шаге 32 не вычислялось: нет функции; DTW не реализовано (`notebooks/32_edge_rules.md`, строка 195).

## Воспроизводимость

- Все случайные процедуры имеют фиксированные seed (KMeans, Louvain, выбор пар для ARI).
- При равной inertia KMeans выбирается запуск с наименьшим seed (многопоточное суммирование
  во float иначе делает выбор среди равных решений невоспроизводимым).
- Проверено полным прогоном `run_all.sh` с нуля в чистой копии проекта 29.09.2026 (39/39 шагов без
  ошибок, 21 мин 8 с; предыдущий прогон — 19 мин 17 с): все отчёты `notebooks/`, которые создаёт
  прогон, и таблицы `data/processed/` совпадают с рабочим каталогом побайтно, кроме семи файлов, где
  отличаются последние разряды чисел с плавающей точкой (inertia и связанные величины — до ~2e-12;
  разбиения, метки и все текстовые выводы — точно). Архивные отчёты `*_k7_superseded.md` (прежний
  канон k = 7) прогоном не создаются — это сохранённый снимок состояния на момент смены канона, а не
  воспроизводимый результат; метки прежнего канона `kmeans_labels_final_k7_superseded.parquet`
  пересчитываются шагом 10c и совпали.
- При неожиданностях в данных (дубли, несовпадение id, пропуски) скрипты останавливаются с
  сообщением `STOP: ...`, а не исправляют данные автоматически.

## Документация интерпретации

- `docs/interpretation.md` — метод и правила интерпретации типов МО.
- `docs/type_names.md` — технические и интерпретационные имена типов.
- `docs/decisions.md` — журнал решений по интерпретации.
- `docs/interpretation_sources.csv` — черновой реестр источников интерпретации со статусом чтения.

Интерпретационные имена и черновой реестр источников не входят в расчётные шаги: ни один скрипт `src/` их не читает.

## Известные расхождения в документации

Отчёты принятых шагов в `notebooks/` и `docs/` не менялись; расхождения описаны здесь (источник фактических значений указан в последней колонке).

| № | Файл и строка | Фрагмент | Фактическое значение и источник |
|---|---|---|---|
| 1 | `notebooks/08_economic_vs_transport.md`, строка 6 | railway — с достройкой 95 пар | 100 достроенных пар: `notebooks/07_transport_networks.md`, строки 20–21; `data/processed/connection_valid.parquet`, колонка `imputed` (100 строк True) |
| 2 | `notebooks/12_kmeans_temporal_tracking_k6.md`, строка 6 (тот же фрагмент в `_k7.md` и в docstring `src/12_kmeans_temporal_tracking.py`, строки 5–6) | 7 центроидов каждого месяца сопоставляются напрямую с 7 каноническими | при k = 6 кластеров 6: `data/processed/kmeans_labels_final.parquet` (6 значений `cluster`), `data/processed/kmeans_k6_matching.parquet` (6 значений `cluster`); в колонках последнего файла остались подписи «7×7» и «49 пар» |
| 3 | `README.md`, строка 247 (раздел «Воспроизводимость») | 39/39 шагов без ошибок | `run_all.sh` содержит 62 записей запуска (в таблице шагов выше 54 строк, повторные вызовы сгруппированы); число будет обновлено после замера (часть Б шага 33) |
| 4 | `config.yaml` (строка 469), `notebooks/09_territory_names.md` (строка 4), `notebooks/28_interp_checks.md` (строка 68) | локальные пути автора (`/Users/a1/...`, журнал сессии) | файлы содержат локальные пути автора, не используемые при воспроизведении: `log_path` в `src/28_interpretation_checks.py` только подставляется в текст отчёта, файл не открывается; путь к справочнику в шагах 09 и 17 строится от домашнего каталога (`Path.home()`), в отчёт 09 попадает как текст |
