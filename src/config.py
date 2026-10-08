"""Общие параметры пайплайна, от которых зависят несколько шагов.

Значения читаются из config.yaml в корне проекта (единый источник). Модуль отдаёт те же имена,
значения и типы, что и раньше, поэтому шаги пайплайна импортируют их без изменений.
Диапазоны в YAML заданы как {min, max} включительно и собираются здесь в list(range(...));
пары ROBUST2_CONTINGENCY превращаются в tuple.
"""
from pathlib import Path as _Path

import yaml as _yaml

_CONFIG_PATH = _Path(__file__).resolve().parents[1] / "config.yaml"
with open(_CONFIG_PATH, encoding="utf-8") as _f:
    _CFG = _yaml.safe_load(_f)
_USED: set = set()


def _get(group: str, key: str):
    _USED.add((group, key))
    return _CFG[group][key]


def _k_range(group: str, key: str) -> list:
    r = _get(group, key)
    return list(range(r["min"], r["max"] + 1))


def _range(group: str, key: str) -> range:
    r = _get(group, key)
    return range(r["min"], r["max"] + 1)


# Данные и период
MONTH = _get("data", "MONTH")

# Кластеризация (k = 7 было каноническим до решения от 2026-09-28; см. config.yaml)
FINAL_K = _get("clustering", "FINAL_K")
SUPERSEDED_K = _get("clustering", "SUPERSEDED_K")

# Сети
ECON_KNN_K = _get("networks", "ECON_KNN_K")
ROBUST_KNN_K = ECON_KNN_K   # шаг 14: тот же граф kNN, что у канонической экономической сети
ROBUST_KNN_VARIANTS = _get("networks", "ROBUST_KNN_VARIANTS")

# Шаг 14: проверки устойчивости выбора k
ROBUST_N_RUNS = _get("robustness", "ROBUST_N_RUNS")
ROBUST_K_COMPARE = _get("robustness", "ROBUST_K_COMPARE")
ROBUST_K_RANGE = _k_range("robustness", "ROBUST_K_RANGE")
ROBUST_PS_SPLITS = _get("robustness", "ROBUST_PS_SPLITS")
ROBUST_N_RANDOM = _get("robustness", "ROBUST_N_RANDOM")
ROBUST_SE_MULT = _get("robustness", "ROBUST_SE_MULT")
ROBUST_DECOMP_K = _get("robustness", "ROBUST_DECOMP_K")
ROBUST_N_JOBS = _get("misc", "ROBUST_N_JOBS")
ROBUST_LOUVAIN_RESOLUTIONS = _get("robustness", "ROBUST_LOUVAIN_RESOLUTIONS")
ROBUST_LOUVAIN_SEEDS = _get("robustness", "ROBUST_LOUVAIN_SEEDS")
ROBUST_GAP_REFS = _get("robustness", "ROBUST_GAP_REFS")
ROBUST_GAP_K_RANGE = _k_range("robustness", "ROBUST_GAP_K_RANGE")

# Шаг 16: кривая стабильности по k
STABILITY_KS = _get("temporal", "STABILITY_KS")
KINK_RATIO = _get("temporal", "KINK_RATIO")

# Шаг 15: проверки устойчивости, батч 2
ROBUST2_VARIANTS = _get("robustness", "ROBUST2_VARIANTS")
ROBUST2_K = _get("robustness", "ROBUST2_K")
ROBUST2_K_RANGE = _get("robustness", "ROBUST2_K_RANGE")
ROBUST2_N_RUNS = _get("robustness", "ROBUST2_N_RUNS")
ROBUST2_GROUP_JACCARD = _get("robustness", "ROBUST2_GROUP_JACCARD")
ROBUST2_REMOTE_CLUSTER = _get("robustness", "ROBUST2_REMOTE_CLUSTER")
ROBUST2_CAPITAL_CLUSTER = _get("robustness", "ROBUST2_CAPITAL_CLUSTER")
ROBUST2_COMPARE_TOL = _get("robustness", "ROBUST2_COMPARE_TOL")
ROBUST2_CONTINGENCY = [tuple(p) for p in _get("robustness", "ROBUST2_CONTINGENCY")]
ROBUST2_TARGET_K = _get("robustness", "ROBUST2_TARGET_K")
ROBUST2_EXTRA_RESOLUTIONS = _get("robustness", "ROBUST2_EXTRA_RESOLUTIONS")

# Шаг 15e: внешняя валидность вариантов признаков и выбора k
EXTVAL_KS = _get("robustness", "EXTVAL_KS")
EXTVAL_PERMUTATIONS = _get("robustness", "EXTVAL_PERMUTATIONS")
EXTVAL_SEED = _get("robustness", "EXTVAL_SEED")
EXTVAL_ALPHA = _get("robustness", "EXTVAL_ALPHA")

# Шаг 04: сеть на сырых долях
RAW_NET_EDGE_THRESHOLD = _get("step04_raw_network", "RAW_NET_EDGE_THRESHOLD")
RAW_NET_CANDIDATE_THRESHOLDS = _get("step04_raw_network", "RAW_NET_CANDIDATE_THRESHOLDS")

# Шаг 07: транспортные сети
TRANSPORT_KNN_K = _get("step07_transport", "TRANSPORT_KNN_K")
TRANSPORT_SYMMETRY_TOL_KM = _get("step07_transport", "TRANSPORT_SYMMETRY_TOL_KM")
TRANSPORT_IMPUTE_MISSING_RAILWAY_ZERO = _get("step07_transport", "TRANSPORT_IMPUTE_MISSING_RAILWAY_ZERO")
TRANSPORT_NAMES_CHECK_KM = _get("step07_transport", "TRANSPORT_NAMES_CHECK_KM")
TRANSPORT_DETOUR_RATIO = _get("step07_transport", "TRANSPORT_DETOUR_RATIO")

# Шаг 08: экономическая vs транспортная близость
ECON_TRANSPORT_MONTHS = _get("step08_econ_vs_transport", "ECON_TRANSPORT_MONTHS")
ECON_TRANSPORT_TOP_SIMILARITY_SHARE = _get("step08_econ_vs_transport", "ECON_TRANSPORT_TOP_SIMILARITY_SHARE")
ECON_TRANSPORT_FAR_MIN_KM = _get("step08_econ_vs_transport", "ECON_TRANSPORT_FAR_MIN_KM")
ECON_TRANSPORT_CLUSTER_KM = tuple(_get("step08_econ_vs_transport", "ECON_TRANSPORT_CLUSTER_KM"))
ECON_TRANSPORT_CLUSTER_MA_LOW = tuple(_get("step08_econ_vs_transport", "ECON_TRANSPORT_CLUSTER_MA_LOW"))
ECON_TRANSPORT_CLUSTER_MA_HIGH = tuple(_get("step08_econ_vs_transport", "ECON_TRANSPORT_CLUSTER_MA_HIGH"))

# Шаг 10: выбор k для KMeans
KMEANS_K_RANGE = _range("step10_kmeans", "KMEANS_K_RANGE")
KMEANS_RANDOM_STATE = _get("step10_kmeans", "KMEANS_RANDOM_STATE")
KMEANS_N_INIT = _get("step10_kmeans", "KMEANS_N_INIT")
SDBW_KW = dict(_get("step10_kmeans", "SDBW_KW"))
SDBW_N_RANDOM = _get("step10_kmeans", "SDBW_N_RANDOM")
SDBW_BASELINE_SEED = _get("step10_kmeans", "SDBW_BASELINE_SEED")
SDBW_N_BOOT = _get("step10_kmeans", "SDBW_N_BOOT")

# Шаг 10d: prediction strength
PS_M_SPLITS = _get("step10d_prediction_strength", "PS_M_SPLITS")
PS_SPLIT_SEED = _get("step10d_prediction_strength", "PS_SPLIT_SEED")
PS_THRESHOLDS = _get("step10d_prediction_strength", "PS_THRESHOLDS")

# Шаг 10c: фиксация k, проверка плоского минимума
NEAR_MIN_INERTIA = _get("step10c_final", "NEAR_MIN_INERTIA")
SAME_PARTITION_ARI = _get("step10c_final", "SAME_PARTITION_ARI")
CANONICAL_SEED = KMEANS_RANDOM_STATE   # seed канонического разбиения совпадает с random_state шага 10

# Шаг 12a: запуски KMeans
KMEANS_RUN_SEEDS = _range("step12a_runs", "KMEANS_RUN_SEEDS")

# Шаг 12: сопоставление кластеров во времени
CONFIDENT_MARGIN = _get("step12_temporal", "CONFIDENT_MARGIN")
AMBIGUOUS_MARGIN = _get("step12_temporal", "AMBIGUOUS_MARGIN")
CONFIDENT_SHARE_MIN = _get("step12_temporal", "CONFIDENT_SHARE_MIN")
PROBLEM_MONTH_MIN_AMBIGUOUS = _get("step12_temporal", "PROBLEM_MONTH_MIN_AMBIGUOUS")
PROBLEM_SWITCH_IQR = _get("step12_temporal", "PROBLEM_SWITCH_IQR")
PERIOD_WINDOW = _get("step12_temporal", "PERIOD_WINDOW")
STABLE_MIN_MONTHS = _get("step12_temporal", "STABLE_MIN_MONTHS")
MIN_CONF_MONTHS_IN_WINDOW = _get("step12_temporal", "MIN_CONF_MONTHS_IN_WINDOW")
CONF_THRESHOLD = _get("step12_temporal", "CONF_THRESHOLD")
CONF_THRESHOLDS = _get("step12_temporal", "CONF_THRESHOLDS")
BORDERLINE_CONF = _get("step12_temporal", "BORDERLINE_CONF")
BORDERLINE_MIN_SHARE = _get("step12_temporal", "BORDERLINE_MIN_SHARE")
TEMPORAL_THRESHOLD = _get("step12_temporal", "TEMPORAL_THRESHOLD")
TEMPORAL_THRESHOLDS = _get("step12_temporal", "TEMPORAL_THRESHOLDS")
SWING_TOP2_MIN = _get("step12_temporal", "SWING_TOP2_MIN")
SWING_TOP2_ALT = _get("step12_temporal", "SWING_TOP2_ALT")
CENTROID_SPACE = _get("step12_temporal", "CENTROID_SPACE")
TREND_CONFOUND_ABS_CORR = _get("step12_temporal", "TREND_CONFOUND_ABS_CORR")

# Шаги 11, 11b, 11c: Louvain
LOUVAIN_WEIGHT = _get("step11_louvain", "LOUVAIN_WEIGHT")
LOUVAIN_RESOLUTION = _get("step11_louvain", "LOUVAIN_RESOLUTION")
LOUVAIN_SEED = _get("step11_louvain", "LOUVAIN_SEED")
LOUVAIN_STABILITY_SEEDS = _range("step11_louvain", "LOUVAIN_STABILITY_SEEDS")
LOUVAIN_SEEDS = _range("step11b_louvain_stability", "LOUVAIN_SEEDS")
LOUVAIN_N_RANDOM_PAIRS = _get("step11b_louvain_stability", "LOUVAIN_N_RANDOM_PAIRS")
LOUVAIN_PAIRS_RNG_SEED = _get("step11b_louvain_stability", "LOUVAIN_PAIRS_RNG_SEED")
LOUVAIN_CORE_THRESHOLD = _get("step11c_louvain_core", "LOUVAIN_CORE_THRESHOLD")

# Шаг 13: Louvain по месяцам
LOUVAIN_MONTHLY_N_SEEDS = _get("step13_louvain_monthly", "LOUVAIN_MONTHLY_N_SEEDS")
NEST_PURITY = _get("step13_louvain_monthly", "NEST_PURITY")
TYPICAL_Z = _get("step13_louvain_monthly", "TYPICAL_Z")

# Шаг 15d: gap statistic
GAP_SEED = _get("step15d_gap", "GAP_SEED")

# Шаг 17: извлечение данных Росстата
ROSSTAT_POP_YEARS = _get("step17_rosstat", "ROSSTAT_POP_YEARS")
ROSSTAT_LABOUR_YEARS = _get("step17_rosstat", "ROSSTAT_LABOUR_YEARS")
ROSSTAT_CHUNKSIZE = _get("step17_rosstat", "ROSSTAT_CHUNKSIZE")
ROSSTAT_ANOMALY_SHARE_TOL = _get("step17_rosstat", "ROSSTAT_ANOMALY_SHARE_TOL")

# Шаг 18a: пороги для проверки гипотез (шаг 19 использует те же значения)
ALPHA = _get("step18_thresholds", "ALPHA")
MONTHS_RULE = _get("step18_thresholds", "MONTHS_RULE")
N_PERM = _get("step18_thresholds", "N_PERM")
MIN_GROUP = _get("step18_thresholds", "MIN_GROUP")
HYP_MIN_TYPES = _get("step18_thresholds", "HYP_MIN_TYPES")
HYP_MIN_REGION_N = _get("step18_thresholds", "HYP_MIN_REGION_N")
HYP_SEED = _get("step18_thresholds", "HYP_SEED")
HYP_PERM_BATCH = _get("step18_thresholds", "HYP_PERM_BATCH")
HYP_CONTROL_SEED = _get("step18_thresholds", "HYP_CONTROL_SEED")
HYP_NEG1_N_VARS = _get("step18_thresholds", "HYP_NEG1_N_VARS")
HYP_NEG2_N_VARS = _get("step18_thresholds", "HYP_NEG2_N_VARS")
HYP_NEG_MAX_SHARE = _get("step18_thresholds", "HYP_NEG_MAX_SHARE")
HYP_POSITIVE_CONTROLS = _get("step18_thresholds", "HYP_POSITIVE_CONTROLS")
HYP_KW_VARIABLES = _get("step18_thresholds", "HYP_KW_VARIABLES")
HYP_CORR_PAIRS = _get("step18_thresholds", "HYP_CORR_PAIRS")

# Шаг 18b: пороги для гипотез Г6 и Г9
N_NEAREST = _get("step18b_thresholds", "N_NEAREST")
N_NEAREST_SENS = _get("step18b_thresholds", "N_NEAREST_SENS")
SEED_G6 = _get("step18b_thresholds", "SEED_G6")
SEED_G9 = _get("step18b_thresholds", "SEED_G9")

# Шаг 19a: проверка гипотез по порогам шага 18a
BOOT_B = _get("step19_tests", "BOOT_B")
SEED_G7 = _get("step19_tests", "SEED_G7")
# Шаг 19b: имена с префиксом TEST_, так как SEED_G6 / SEED_G9 уже заняты шагом 18b (step18b_thresholds)
TEST_SEED_G6 = _get("step19_tests", "SEED_G6")
TEST_SEED_G9 = _get("step19_tests", "SEED_G9")

# Шаг 20: извлечение данных о рознице и общепите
RETAIL_YEARS = _get("step20_retail", "RETAIL_YEARS")
RETAIL_PERIOD = _get("step20_retail", "RETAIL_PERIOD")
RETAIL_BATCH_ROWS = _get("step20_retail", "RETAIL_BATCH_ROWS")
RETAIL_CHECK_BAND = _get("step20_retail", "RETAIL_CHECK_BAND")
RETAIL_CHECK_MAX_FLOOR_PER_1000 = _get("step20_retail", "RETAIL_CHECK_MAX_FLOOR_PER_1000")

# Шаг 21: пороги для дополнительных гипотез Г10–Г13
SEED_21 = _get("step21_supply", "SEED_21")
SUPPLY_VARIABLES = _get("step21_supply", "SUPPLY_VARIABLES")
SUPPLY_CORR_PAIRS = _get("step21_supply", "SUPPLY_CORR_PAIRS")
SUPPLY_KW = _get("step21_supply", "SUPPLY_KW")

# Шаг 22: проверка Г10–Г13
SEED_22 = _get("step22_supply", "SEED_22")

# Шаг 23: портреты типов
SIZE_BINS_TH = _get("step23_portraits", "SIZE_BINS_TH")
MIN_MO_REGION = _get("step23_portraits", "MIN_MO_REGION")
TOP_REGIONS = _get("step23_portraits", "TOP_REGIONS")
TOP_REGIONS_TYPE4 = _get("step23_portraits", "TOP_REGIONS_TYPE4")
SMALL_TYPES = _get("step23_portraits", "SMALL_TYPES")
CAPITAL_REGIONS = _get("step23_portraits", "CAPITAL_REGIONS")
MOSCOW_OBLAST = _get("step23_portraits", "MOSCOW_OBLAST")
YAKUTIA = _get("step23_portraits", "YAKUTIA")
YAKUTIA_SIZE_BINS_TH = _get("step23_portraits", "YAKUTIA_SIZE_BINS_TH")
FAR_EAST_DISTRICT = _get("step23_portraits", "FAR_EAST_DISTRICT")
SEED_23 = _get("step23_portraits", "SEED_23")
SIZE_LABEL_RATIO_LARGE = _get("step23_portraits", "SIZE_LABEL_RATIO_LARGE")
SIZE_LABEL_RATIO_CITY = _get("step23_portraits", "SIZE_LABEL_RATIO_CITY")
ACCESS_LABEL_RATIO_LOW = _get("step23_portraits", "ACCESS_LABEL_RATIO_LOW")
PROFILE_BORDER_MARGIN = _get("step23_portraits", "profile_border_margin")
PROFILE_BORDER_LABEL = _get("step23_portraits", "profile_border_label")
ZERO_TOL = _get("step23_portraits", "ZERO_TOL")
PORTRAIT_CONTROLS = _get("step23_portraits", "CONTROLS")

# Шаг 25: устойчивость к композиционной природе долей
KMEANS_SEEDS_CHECK = _get("step25_composition", "KMEANS_SEEDS_CHECK")
COMP_MIN_MO_REGION = _get("step25_composition", "MIN_MO_REGION")

# Шаг 26: сводка проверок внешней валидности
SUMMARY_ETA2_BINS = _get("step26_summary", "ETA2_SIZE_BINS")
SUMMARY_ETA2_LABELS = _get("step26_summary", "ETA2_SIZE_LABELS")
SUMMARY_RHO_BINS = _get("step26_summary", "RHO_SIZE_BINS")
SUMMARY_RHO_LABELS = _get("step26_summary", "RHO_SIZE_LABELS")
SUMMARY_NEG_SOURCES = _get("step26_summary", "NEG_CONTROL_SOURCES")
SUMMARY_NEG_EXPECT = _get("step26_summary", "NEG_CONTROL_EXPECT")

# Шаг 24: фигуры 1, 3, 4, 5 (все ключи группы читаются словарём)
FIGS = {_k: _get("step24_figures", _k) for _k in _CFG["step24_figures"]}

# Шаг 27: правило интерпретации кластеров
MIRKIN_MIN_BASE_PP = _get("step27_mirkin", "MIN_BASE_PP")

# Шаг 28: проверка интерпретаций
SEED_28 = _get("step28_checks", "SEED_28")
PREV_DEC_MONTH = _get("step28_checks", "PREV_DEC_MONTH")
STABLE_SHARE = _get("step28_checks", "STABLE_SHARE")
SUMMER_MONTHS = _get("step28_checks", "SUMMER_MONTHS")
REMOTE_TYPE = _get("step28_checks", "REMOTE_TYPE")
MIDDLE_TYPE = _get("step28_checks", "MIDDLE_TYPE")
TOP_TRANSITIONS = _get("step28_checks", "TOP_TRANSITIONS")
HEALTH_TYPES = _get("step28_checks", "HEALTH_TYPES")
MIN_TYPE_N_GROUP = _get("step28_checks", "MIN_TYPE_N_GROUP")
TOL_SHARE = _get("step28_checks", "TOL_SHARE")
TOL_RHO = _get("step28_checks", "TOL_RHO")
TOL_ARI = _get("step28_checks", "TOL_ARI")
CHECKS_EXPECT = _get("step28_checks", "EXPECT")
PREV_SOURCES = _get("step28_checks", "PREV_SOURCES")
DATA_QUOTES = _get("step28_checks", "DATA_QUOTES")
DATA_ANSWERS = _get("step28_checks", "DATA_ANSWERS")

# Шаг 24б: геометрия МО для карты
GEO_DICT_DIR = _get("step24b_geometry", "MUNICIPAL_DICT_DIR")
GEO_ARCHIVE_URL = _get("step24b_geometry", "ARCHIVE_URL")
GEO_ARCHIVE_SHA256 = _get("step24b_geometry", "ARCHIVE_SHA256")
GEO_GPKG_SHA256 = _get("step24b_geometry", "GPKG_SHA256")
GEO_CA_BUNDLE = _get("step24b_geometry", "CA_BUNDLE")
GEO_SNAPSHOT_MONTH = _get("step24b_geometry", "SNAPSHOT_MONTH")
GEO_ACTIVE_YEAR_TO = _get("step24b_geometry", "ACTIVE_YEAR_TO")
GEO_DOWNLOAD_DATE = _get("step24b_geometry", "DATA_DOWNLOAD_DATE")
GEO_NATIONAL_TOL = _get("step24b_geometry", "NATIONAL_TOLERANCE_DEG")
GEO_NATIONAL_DECIMALS = _get("step24b_geometry", "NATIONAL_DECIMALS")
GEO_CITIES_TOL = _get("step24b_geometry", "CITIES_TOLERANCE_DEG")
GEO_CITIES_DECIMALS = _get("step24b_geometry", "CITIES_DECIMALS")
GEO_CITY_REGION_CODES = _get("step24b_geometry", "CITY_REGION_CODES")
GEO_AREA_CHANGE_THRESHOLD = _get("step24b_geometry", "AREA_CHANGE_THRESHOLD")
GEO_GAP_THRESHOLD = _get("step24b_geometry", "GAP_THRESHOLD_DEG")
GEO_GAP_SAMPLE = _get("step24b_geometry", "GAP_SAMPLE_PER_PAIR")
GEO_GAP_SEED = _get("step24b_geometry", "GAP_SAMPLE_SEED")
GEO_OVERLAP_MIN_AREA = _get("step24b_geometry", "OVERLAP_MIN_AREA_DEG2")
GEO_EXPECT_ACTIVE = _get("step24b_geometry", "EXPECT_ACTIVE_POLYGONS")
GEO_EXPECT_SAMPLE = _get("step24b_geometry", "EXPECT_SAMPLE_POLYGONS")
GEO_EXPECT_OVERLAP_PAIRS = _get("step24b_geometry", "EXPECT_OVERLAP_PAIRS")
GEO_MAX_NATIONAL_AREA_CHANGE_MO = _get("step24b_geometry", "MAX_NATIONAL_AREA_CHANGE_MO")
GEO_MAX_NATIONAL_BYTES = _get("step24b_geometry", "MAX_NATIONAL_BYTES")
GEO_MAX_LOST_PART_AREA = _get("step24b_geometry", "MAX_LOST_PART_AREA_DEG2")

# Шаг 24c: типы МО вне выборки
ASSIGN_GEO_PATH = _get("step24c_assign", "GEO_NATIONAL_PATH")
ASSIGN_SKLEARN_VERSION = _get("step24c_assign", "EXPECT_SKLEARN_VERSION")
ASSIGN_SCALER_MEAN = _get("step24c_assign", "EXPECT_SCALER_MEAN")
ASSIGN_SCALER_SCALE = _get("step24c_assign", "EXPECT_SCALER_SCALE")
ASSIGN_SCALER_TOL = _get("step24c_assign", "SCALER_TOL")
ASSIGN_INERTIA = _get("step24c_assign", "EXPECT_INERTIA")
ASSIGN_INERTIA_TOL = _get("step24c_assign", "INERTIA_TOL")
ASSIGN_TYPE_SIZES = _get("step24c_assign", "EXPECT_TYPE_SIZES")
ASSIGN_EXPECT_ACTIVE = _get("step24c_assign", "EXPECT_ACTIVE")
ASSIGN_EXPECT_A = _get("step24c_assign", "EXPECT_GROUP_A")
ASSIGN_EXPECT_B = _get("step24c_assign", "EXPECT_GROUP_B")
ASSIGN_EXPECT_B_COMPLETE = _get("step24c_assign", "EXPECT_B_COMPLETE")
ASSIGN_EXPECT_B_INCOMPLETE = _get("step24c_assign", "EXPECT_B_INCOMPLETE")
ASSIGN_DIST_TOL = _get("step24c_assign", "DIST_TOL")
ASSIGN_DIST_DECIMALS = _get("step24c_assign", "DIST_DECIMALS")
ASSIGN_SAMPLE_DIST = _get("step24c_assign", "EXPECT_SAMPLE_DIST")
ASSIGN_B_DIST = _get("step24c_assign", "EXPECT_B_DIST")
ASSIGN_BEYOND_IDS = _get("step24c_assign", "EXPECT_BEYOND_IDS")
ASSIGN_EXPECT_ASSIGNED = _get("step24c_assign", "EXPECT_ASSIGNED")
ASSIGN_ASSIGNED_BY_TYPE = _get("step24c_assign", "EXPECT_ASSIGNED_BY_TYPE")

# Шаг 24g: тип МО по последнему доступному месяцу
LASTM_AGREEMENT_MIN = _get("step24g_lastmonth", "AGREEMENT_MIN")
LASTM_TRAJECTORIES_PATH = _get("step24g_lastmonth", "TRAJECTORIES_PATH")
LASTM_EXPECT_MONTHS = _get("step24g_lastmonth", "EXPECT_MONTHS")
LASTM_EXPECT_CANDIDATES = _get("step24g_lastmonth", "EXPECT_CANDIDATES")
LASTM_EXPECT_LAST_MONTH_COUNTS = _get("step24g_lastmonth", "EXPECT_LAST_MONTH_COUNTS")
LASTM_EXPECT_REGION_COUNTS = _get("step24g_lastmonth", "EXPECT_REGION_COUNTS")
LASTM_EXPECT_OTHER_REGIONS_N = _get("step24g_lastmonth", "EXPECT_OTHER_REGIONS_N")
LASTM_EXPECT_DEC_AGREEMENT = _get("step24g_lastmonth", "EXPECT_DEC_AGREEMENT")
LASTM_NOTICE_MIN_ELIGIBLE = _get("step24g_lastmonth", "NOTICE_MIN_ELIGIBLE")

# Шаг 24d: статическая карта типов МО
MAP_GEO_NATIONAL_PATH = _get("step24d_map", "GEO_NATIONAL_PATH")
MAP_GEO_CITIES_PATH = _get("step24d_map", "GEO_CITIES_PATH")
MAP_GEO_REPORT_PATH = _get("step24d_map", "GEO_REPORT_PATH")
MAP_TYPES_OUTSIDE_PATH = _get("step24d_map", "TYPES_OUTSIDE_PATH")
MAP_ATTRIBUTION_PATH = _get("step24d_map", "ATTRIBUTION_PATH")
MAP_ATTRIBUTION_CAPTION_HEADING = _get("step24d_map", "ATTRIBUTION_CAPTION_HEADING")
MAP_PNG_PATH = _get("step24d_map", "PNG_PATH")
MAP_SVG_PATH = _get("step24d_map", "SVG_PATH")
MAP_FIGURE_DATA_PATH = _get("step24d_map", "FIGURE_DATA_PATH")
MAP_REPORT_PATH = _get("step24d_map", "REPORT_PATH")
MAP_FIG_SIZE_IN = _get("step24d_map", "FIG_SIZE_IN")
MAP_DPI = _get("step24d_map", "DPI")
MAP_FONT = _get("step24d_map", "FONT")
MAP_SVG_HASHSALT = _get("step24d_map", "SVG_HASHSALT")
MAP_MAX_SVG_BYTES = _get("step24d_map", "MAX_SVG_BYTES")
MAP_TYPE_COLORS = _get("step24d_map", "TYPE_COLORS")
MAP_BORDER_LINEWIDTH_PT = _get("step24d_map", "BORDER_LINEWIDTH_PT")
MAP_ASSIGNED_HATCH = _get("step24d_map", "ASSIGNED_HATCH")
MAP_ASSIGNED_HATCH_COLOR = _get("step24d_map", "ASSIGNED_HATCH_COLOR")
MAP_NO_TYPE_FACE = _get("step24d_map", "NO_TYPE_FACE")
MAP_NO_TYPE_HATCH = _get("step24d_map", "NO_TYPE_HATCH")
MAP_NO_TYPE_HATCH_COLOR = _get("step24d_map", "NO_TYPE_HATCH_COLOR")
MAP_HATCH_LINEWIDTH_PT = _get("step24d_map", "HATCH_LINEWIDTH_PT")
MAP_PROJECTION_LAT_DEG = _get("step24d_map", "PROJECTION_LAT_DEG")
MAP_TITLE = _get("step24d_map", "TITLE")
MAP_TITLE_FONTSIZE = _get("step24d_map", "TITLE_FONTSIZE")
MAP_TITLE_TOP_IN = _get("step24d_map", "TITLE_TOP_IN")
MAP_LEGEND_FONTSIZE = _get("step24d_map", "LEGEND_FONTSIZE")
MAP_LEGEND_NCOL = _get("step24d_map", "LEGEND_NCOL")
MAP_LEGEND_GAP_IN = _get("step24d_map", "LEGEND_GAP_IN")
MAP_INSET_FONTSIZE = _get("step24d_map", "INSET_FONTSIZE")
MAP_CAPTION_FONTSIZE = _get("step24d_map", "CAPTION_FONTSIZE")
MAP_CAPTION_WRAP_CHARS = _get("step24d_map", "CAPTION_WRAP_CHARS")
MAP_CAPTION_LEFT_IN = _get("step24d_map", "CAPTION_LEFT_IN")
MAP_CAPTION_GAP_IN = _get("step24d_map", "CAPTION_GAP_IN")
MAP_MAP_RECT_IN = _get("step24d_map", "MAP_RECT_IN")
MAP_MOSCOW_INSET_IN = _get("step24d_map", "MOSCOW_INSET_IN")
MAP_SPB_INSET_IN = _get("step24d_map", "SPB_INSET_IN")
MAP_INSET_PAD_FRACTION = _get("step24d_map", "INSET_PAD_FRACTION")
MAP_INSET_FRAME_COLOR = _get("step24d_map", "INSET_FRAME_COLOR")
MAP_INSET_FRAME_LW = _get("step24d_map", "INSET_FRAME_LW")
MAP_LEADER_LW = _get("step24d_map", "LEADER_LW")
MAP_LEADER_HALO_LW = _get("step24d_map", "LEADER_HALO_LW")
MAP_LEADER_RECT_CORNER = _get("step24d_map", "LEADER_RECT_CORNER")
MAP_MIN_MARGIN_IN = _get("step24d_map", "MIN_MARGIN_IN")
MAP_PNG_MARGIN_MAX_IN = _get("step24d_map", "PNG_MARGIN_MAX_IN")
MAP_MAX_GAP_IN = _get("step24d_map", "MAX_GAP_IN")
MAP_TITLE_CENTER_TOL_IN = _get("step24d_map", "TITLE_CENTER_TOL_IN")
MAP_PIXEL_CHECK_MIN_RADIUS_PX = _get("step24d_map", "PIXEL_CHECK_MIN_RADIUS_PX")
MAP_PIXEL_MIC_TOLERANCE_DEG = _get("step24d_map", "PIXEL_MIC_TOLERANCE_DEG")
MAP_PIXEL_TOL_CLASS1 = _get("step24d_map", "PIXEL_TOL_CLASS1")
MAP_PIXEL_TOL_SEGMENT = _get("step24d_map", "PIXEL_TOL_SEGMENT")
MAP_EXPECT_PIXEL_CHECKED_MIN = _get("step24d_map", "EXPECT_PIXEL_CHECKED_MIN")
MAP_EXPECT_PIXEL_TYPE_MIN = _get("step24d_map", "EXPECT_PIXEL_TYPE_MIN")
MAP_EXPECT_PIXEL_CLASS_MIN = _get("step24d_map", "EXPECT_PIXEL_CLASS_MIN")
MAP_EXPECT_PIXEL_AREA_SHARE_MIN = _get("step24d_map", "EXPECT_PIXEL_AREA_SHARE_MIN")
MAP_MAX_PIXEL_MISMATCH = _get("step24d_map", "MAX_PIXEL_MISMATCH")
MAP_LON_AFTER_MAX = _get("step24d_map", "LON_AFTER_MAX")
MAP_LON_AFTER_MIN = _get("step24d_map", "LON_AFTER_MIN")
MAP_LON_BEFORE_RANGE = _get("step24d_map", "LON_BEFORE_RANGE")
MAP_SHIFT_REGION_SUBSTRING = _get("step24d_map", "SHIFT_REGION_SUBSTRING")
MAP_CHECK_409 = _get("step24d_map", "CHECK_409")
MAP_EXPECT_REGIONS_ALL = _get("step24d_map", "EXPECT_REGIONS_ALL")
MAP_EXPECT_REGIONS_SAMPLE = _get("step24d_map", "EXPECT_REGIONS_SAMPLE")
MAP_FORBIDDEN_WORDS = _get("step24d_map", "FORBIDDEN_WORDS")
MAP_EXPECT_ACTIVE = _get("step24d_map", "EXPECT_ACTIVE")
MAP_EXPECT_SAMPLE_BY_TYPE = _get("step24d_map", "EXPECT_SAMPLE_BY_TYPE")
MAP_EXPECT_ASSIGNED_BY_TYPE = _get("step24d_map", "EXPECT_ASSIGNED_BY_TYPE")
MAP_EXPECT_NO_TYPE = _get("step24d_map", "EXPECT_NO_TYPE")
MAP_EXPECT_BEYOND_P95 = _get("step24d_map", "EXPECT_BEYOND_P95")
MAP_EXPECT_INSET_POLYGONS = _get("step24d_map", "EXPECT_INSET_POLYGONS")
MAP_EXPECT_LEGEND_ITEMS = _get("step24d_map", "EXPECT_LEGEND_ITEMS")

# Шаг 24f: данные слоёв лендинга (все ключи группы читаются словарём)
LAYERS = {_k: _get("step24f_layers", _k) for _k in _CFG["step24f_layers"]}

# Шаг 24e: каркас лендинга (все ключи группы читаются словарём)
LANDING = {_k: _get("step24e_landing", _k) for _k in _CFG["step24e_landing"]}
STEP31 = {_k: _get("step31_methods", _k) for _k in _CFG["step31_methods"]}   # шаг 31: матрица «метод × индексы»
STEP32 = {_k: _get("step32_edgerules", _k) for _k in _CFG["step32_edgerules"]}   # шаг 32: правила построения рёбер

# Согласованность связанных параметров: FINAL_K нельзя менять одним значением
if CANONICAL_SEED not in KMEANS_RUN_SEEDS:
    raise SystemExit(f"STOP: CANONICAL_SEED = {CANONICAL_SEED} не входит в KMEANS_RUN_SEEDS {KMEANS_RUN_SEEDS}")
if not FINAL_K == ROBUST2_K == ROBUST2_TARGET_K:
    raise SystemExit(f"STOP: FINAL_K = {FINAL_K}, ROBUST2_K = {ROBUST2_K}, ROBUST2_TARGET_K = {ROBUST2_TARGET_K} "
                     "должны совпадать (при смене k перевыберите и ROBUST2_*_CLUSTER)")
if ROBUST_K_COMPARE != [FINAL_K, SUPERSEDED_K]:
    raise SystemExit(f"STOP: ROBUST_K_COMPARE = {ROBUST_K_COMPARE}, ожидается [FINAL_K, SUPERSEDED_K] = "
                     f"{[FINAL_K, SUPERSEDED_K]}")

# В config.yaml не должно быть ключей, которые модуль не читает (защита от опечаток)
_UNUSED = {(g, k) for g, d in _CFG.items() for k in d} - _USED
if _UNUSED:
    raise SystemExit(f"STOP: в config.yaml есть неизвестные параметры: {sorted(_UNUSED)}")
