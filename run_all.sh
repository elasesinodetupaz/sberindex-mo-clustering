#!/usr/bin/env bash
# Полный прогон пайплайна в порядке зависимостей (не строго по номерам шагов — см. README).
# Запуск из корня проекта:  bash run_all.sh      (интерпретатор можно задать: PYTHON=python3 bash run_all.sh)
set -euo pipefail
cd "$(dirname "$0")"
PY="${PYTHON:-.venv/bin/python}"

# Шаги с аргументами: "скрипт аргументы"
STEPS=(
  "01_data_overview"
  "02_missing_data"
  "03_category_shares"
  "04_economic_network"
  "05_economic_network_std_knn"
  "06_economic_networks_monthly"
  "09_territory_names"                # справочник названий нужен шагам 07 и 08
  "07_transport_networks"
  "08_economic_vs_transport"
  "10_kmeans_k_selection"
  "10b_kmeans_cluster_profiles"
  "10d_prediction_strength"           # нужен шагу 10c
  "12a_kmeans_runs --k 7"             # прежний канон k=7: доказательная база для 10c
  "12_kmeans_temporal_tracking --k 7"
  "12a_kmeans_runs --k 6"             # канон k=6 (config.FINAL_K)
  "12_kmeans_temporal_tracking --k 6"
  "12a_kmeans_runs --k 4"             # кривая стабильности (шаг 16): k = 4, 5, 8
  "12_kmeans_temporal_tracking --k 4"
  "12a_kmeans_runs --k 5"
  "12_kmeans_temporal_tracking --k 5"
  "12a_kmeans_runs --k 8"
  "12_kmeans_temporal_tracking --k 8"
  "10c_final_clustering_doc"          # пишет kmeans_labels_final (k=6) — нужен шагам 11 и 13
  "11_louvain"
  "11b_louvain_stability"
  "11c_louvain_core_stability"
  "13_louvain_monthly_check"
  "14a_feature_spec_check"            # проверки устойчивости выбора k
  "14b_k_over_time"
  "14_robustness_report"
  "16_k_stability_curve"
  "15a_feature_spec_full"             # проверки устойчивости, батч 2
  "15b_k_contingency"
  "15c_louvain_resolution"
  "15d_gap_statistic"
  "15_robustness_report_2"
  "15e_external_validity"
  "15f_k_tradeoff"                    # раздел «компромисс по k» в отчёте 15e
  "17_rosstat_extract"                # данные Росстата для проверки гипотез (метки и отчёты не меняет)
  "18_hypothesis_thresholds"          # предрегистрация порогов для гипотез (нулевые распределения, контроли)
  "18b_hypothesis_thresholds_g6_g9"   # предрегистрация порогов для Г6 (переходы) и Г9 (переходы и сдвиги долей)
  "19a_hypothesis_tests_kw"           # проверка Г1, Г4, Г7, Г8 по замороженным порогам 18a (без вердиктов)
  "19b_hypothesis_tests_corr_trans"   # проверка Г2, Г3, Г6, Г9 по замороженным порогам 18a/18b (без вердиктов)
  "19c_hypothesis_verdicts"           # вердикты: поправка Холма и правила по результатам 19a/19b (без пересчёта)
  "20_rosstat_retail_extract"         # данные Росстата о рознице и общепите (только извлечение)
  "21_supply_thresholds"              # предрегистрация порогов для дополнительных гипотез Г10–Г13 (предложение)
  "22_supply_tests"                   # проверка Г10–Г13 и вердикты по замороженным порогам 21
  "23_type_portraits"                 # портреты типов (описательно: без p-значений, тестов и вердиктов)
  "25_composition_robustness"         # устойчивость к композиционной природе долей (разведочно: clr, alr, rest)
  "26_hypothesis_summary"             # сводка проверок внешней валидности (только чтение результатов 18–25 и 23)
  "27_mirkin_rule"                    # описание типов по правилу интерпретации кластеров (разведочно)
  "28_interpretation_checks"          # проверка интерпретаций: описание набора, сверка чисел, разведочные проверки
  "24b_geometry"                      # границы МО для карты: справочник (при необходимости скачивание), coverage_simplify, отчёт
  "24c_assign_outside"                # типы МО вне выборки: ближайший центроид (вне статистик шагов 10–28)
  "24g_last_month_types"              # тип МО без данных за декабрь 2024 по последнему доступному месяцу (отдельный класс, вне статистик)
  "24d_map"                          # статическая карта типов МО (фигура 2): PNG и SVG, данные фигуры, описание
  "24f_layer_data"                   # данные слоёв лендинга: доступность рынков, устойчивость типа, покрытие по регионам, тип по последнему месяцу
  "24_figures"                        # фигуры 1, 3, 4, 5: профили типов, маркетплейсы между уровнями, размер МО, надёжность (данные из parquet шагов 19–28)
  "24e_landing"                       # каркас лендинга: один автономный HTML-файл site/index.html (данные, интерактивная карта, карточки типов; тексты из site/content/texts.json)
  "31_method_comparison"              # матрица «метод × шесть индексов качества» и сравнение методов на декабре 2024 (описательно)
  "32_edge_rules"                     # правила построения рёбер (R1, R3–R6; R2 — нет функции): структура сетей, Louvain, ICVI, чувствительность к весам (описательно)
  "10c_final_clustering_doc"          # повторно: итоговый документ 10c с данными шагов 14–15
)

for entry in "${STEPS[@]}"; do
  read -r step args <<< "$entry"
  echo "=== ${entry} ==="
  start=$(date +%s)
  # shellcheck disable=SC2086
  "$PY" "src/${step}.py" $args > /dev/null
  echo "    ok ($(( $(date +%s) - start )) с)"
done
echo "Готово."
