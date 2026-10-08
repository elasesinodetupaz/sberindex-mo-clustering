"""Шаг 27. Описание типов по правилу интерпретации кластеров (правило Миркина в изложении Alvandyan, Shalileh 2024).

Разведочно, вне предрегистрации; сформулировано после просмотра профилей. Новых данных нет: новая агрегация готовых
долей (декабрь 2024, 2004 МО) по готовым меткам kmeans_labels_final. Выборка и таблица долей — build_frame шага 23.

Формулы (Alvandyan, T.A., Shalileh, S., Dokl. Math. 110 (Suppl 1), S236–S250 (2024), раздел 3):
  (12) g_v = Σ_i x_iv / N — общее среднее признака v по всем N объектам;
  (13) d_kv = (c_kv − g_v) / g_v — относительное отклонение центроида кластера k от общего среднего.
Здесь c_kv — среднее доли категории v по МО типа k (центроид в исходных, нестандартизованных долях).
Дополнительно: d по медианам = (медиана типа − общая медиана) / общая медиана; σ по долям — z_profile шага 23 (сравнение).

Выход: data/processed/mirkin_rule_27.parquet (long), notebooks/27_mirkin_rule.md
Запуск из корня проекта:  .venv/bin/python src/27_mirkin_rule.py
"""
import hashlib
import importlib
import re
from pathlib import Path

import numpy as np
import pandas as pd

from config import FINAL_K, MIRKIN_MIN_BASE_PP

step23 = importlib.import_module("23_type_portraits")

PROJECT_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
PORT = PROCESSED_DIR / "type_portraits.parquet"
OUT_PATH = PROCESSED_DIR / "mirkin_rule_27.parquet"
REPORT_PATH = PROJECT_DIR / "notebooks" / "27_mirkin_rule.md"

FROZEN = {
    PROCESSED_DIR / "hypothesis_thresholds.parquet": "11081afbfc7e88dbd301004e4c739e3f25e00bf8a47704cf7c74e15e8e6665f7",
    PROCESSED_DIR / "hypothesis_results_19a.parquet": "58e1516ccbf03b2a1fca82971d5170f581815fe18fd8d22bbf261862fdad9d8e",
    PROCESSED_DIR / "hypothesis_results_19b.parquet": "d66d64d68a82867a5add988d4441fb0df2d35297b8c94c1514bc9e513c615d9d",
    PROCESSED_DIR / "hypothesis_results_supply.parquet": "fbfd86ec33e27ad61378987ccd53e727678017cdb1f091808db8f7545bed97db",
    PROCESSED_DIR / "hypothesis_verdicts.parquet": "01177c53c539fbabded34f03e85d6a8838726df29f93ecb9d47739f96323659f",
    PROCESSED_DIR / "composition_robustness.parquet": "41bfe1d9ba8a0925e8bf3c8c17d6b8232cf5f33db2f749c796c77a32834e37e1",
    PORT: "cb3ccd6e72d6a703e78a9f4e93c4b13029292fd28131e39f79f68ba76d267f7c",
    PROCESSED_DIR / "kmeans_labels_final.parquet": "46cf888d54ec9c6c8ca148f001f867b9408727257d9c98f6240fbe36aa60b22e",
    PROCESSED_DIR / "category_shares.parquet": "b6041519211396f5bf9159e0e1521ad5302f06dfc9be0ead2ad5580c5953b280",
}
CATS = ["Продовольствие", "Здоровье", "Общепит", "Транспорт", "Маркетплейсы"]
EXPL = "разведочно, вне предрегистрации; сформулировано после просмотра профилей"
FORBIDDEN = ["подтверждено", "доказано", "причина", "вызывает", "объясняется", "из-за", "следует"]
CITATION = ("Alvandyan, T.A., Shalileh, S. An Empirical Scrutinization of Four Crisp Clustering Methods with Four Distance "
            "Metrics and One Straightforward Interpretation Rule. Dokl. Math. 110 (Suppl 1), S236–S250 (2024). "
            "https://doi.org/10.1134/S1064562424602002. Русская версия: Nalbandian T. A., Shalileh S. A. An empirical "
            "scrutinization of four crisp clustering methods with four distance metrics and one straightforward interpretation "
            "rule. Докл. РАН. Матем., информ., проц. упр. 520:2, 267–283 (2024). DOI 10.31857/S2686954324700632")

# Контрольные значения пользователя (задание шага 27)
TOL_MEDIAN_PORT = 1e-9                    # медианы типа против блока «расходы» type_portraits.parquet
OVERALL_MEDIAN_CTRL = {"Транспорт": 4.9, "Общепит": 2.1}
TOL_OVERALL = 0.05                        # округлённые значения из md шага 23
D_MEDIAN_CTRL = {(3, "Общепит"): 2.10, (3, "Транспорт"): 0.33, (0, "Общепит"): -0.33}
TOL_D_MEDIAN = 0.06                       # ориентиры округлены

ROWS: list = []


def rec(t, cat, metric, value, n, src_file, src_block) -> float:
    ROWS.append({"тип": str(t), "категория": cat, "показатель": metric, "значение": float(value), "n": int(n),
                 "источник_файл": src_file, "источник_блок": src_block})
    return float(value)


def check_frozen(when: str, log: list) -> None:
    for path, expected in FROZEN.items():
        got = hashlib.sha256(path.read_bytes()).hexdigest()
        rel = path.relative_to(PROJECT_DIR)
        print(f"{when}: {got}  {rel} {'OK' if got == expected else 'MISMATCH'}")
        log.append(f"- {when}: `{rel}` — {'совпадает' if got == expected else 'НЕ СОВПАДАЕТ'}")
        if got != expected:
            raise SystemExit(f"STOP: sha256 {rel} = {got}, ожидается {expected}")


def pct(v: float) -> str:
    return f"{v * 100:+.1f}%"


def main() -> None:
    frozen_log: list = []
    check_frozen("до", frozen_log)
    d = step23.step18.load()
    step23.step18.clean_rosstat(d, [])
    sup, _ = step23.step21.supply_values(d["ids"])
    D = step23.build_frame(d, sup)                       # выборка и доли (×100) — как в шаге 23
    z = step23.z_profile(d, D)                           # σ по долям — как в шаге 23 (сверка с 10b внутри)
    if len(D) != 2004 or D[CATS].isna().any().any():
        raise SystemExit(f"STOP: ожидалось 2004 МО без пропусков долей, получено {len(D)}")
    port = pd.read_parquet(PORT)
    src_d, src_p = "category_shares.parquet; kmeans_labels_final.parquet", "type_portraits.parquet"

    g_mean = {c: float(D[c].mean()) for c in CATS}        # формула (12)
    g_med = {c: float(D[c].median()) for c in CATS}
    print("общие средние, %:", {c: round(v, 3) for c, v in g_mean.items()})
    print("общие медианы, %:", {c: round(v, 3) for c, v in g_med.items()})
    small = {c: v for c, v in g_mean.items() if v < MIRKIN_MIN_BASE_PP}
    if small:
        raise SystemExit(f"STOP: общее среднее меньше {MIRKIN_MIN_BASE_PP} п.п.: {small}")
    for c in CATS:
        rec("все", c, "общее среднее, %", g_mean[c], len(D), src_d, "build_frame шага 23")
        rec("все", c, "общая медиана, %", g_med[c], len(D), src_d, "build_frame шага 23")

    ctrl_rows, bad = [], []

    def ctrl(name, mine, exp, tol):
        ok = abs(mine - exp) <= tol + 1e-12
        ctrl_rows.append({"контроль": name, "шаг 27": f"{mine:.6g}", "контрольное": f"{exp:.6g}", "допуск": f"{tol:g}",
                          "статус": "совпало" if ok else "РАСХОЖДЕНИЕ"})
        if not ok:
            bad.append(ctrl_rows[-1])

    res = {}
    sign_mean = sign_med = 0
    for t in range(FINAL_K):
        sub = D[D["type"] == t]
        n = len(sub)
        for c in CATS:
            m_t, md_t = float(sub[c].mean()), float(sub[c].median())
            dm = (m_t - g_mean[c]) / g_mean[c]               # формула (13)
            dmed = (md_t - g_med[c]) / g_med[c]
            sg = float(z.loc[t, c])
            rec(t, c, "среднее по типу, %", m_t, n, src_d, "build_frame шага 23")
            rec(t, c, "медиана по типу, %", md_t, n, src_d, "build_frame шага 23")
            rec(t, c, "d по средним (формула 13)", dm, n, src_d, "build_frame шага 23")
            rec(t, c, "d по медианам", dmed, n, src_d, "build_frame шага 23")
            rec(t, c, "σ по долям (шаг 23)", sg, n, src_p, "z_profile шага 23")
            res[(t, c)] = (m_t, dm, dmed, sg, n)
            sign_mean += int(np.sign(dm) == np.sign(sg))
            sign_med += int(np.sign(dmed) == np.sign(sg))
            pr = port[(port["type"] == str(t)) & (port["block"] == "расходы") & (port["metric"] == f"медиана {c}, %")]
            if len(pr) != 1:
                raise SystemExit(f"STOP: в type_portraits нет ровно одной строки медианы {c} для типа {t}")
            ctrl(f"медиана типа {t}, {c}: шаг 27 против type_portraits", md_t, float(pr["value"].iloc[0]), TOL_MEDIAN_PORT)
            if int(pr["n"].iloc[0]) != n:
                bad.append({"контроль": f"n типа {t}, {c}", "шаг 27": n, "контрольное": int(pr["n"].iloc[0])})
    for c, v in OVERALL_MEDIAN_CTRL.items():
        ctrl(f"общая медиана {c}, % (2004 МО)", g_med[c], v, TOL_OVERALL)
    for (t, c), v in D_MEDIAN_CTRL.items():
        ctrl(f"d по медианам, тип {t}, {c}", res[(t, c)][2], v, TOL_D_MEDIAN)
    n_cases = FINAL_K * len(CATS)
    print(f"знак d по средним совпадает со знаком σ: {sign_mean} из {n_cases}; знак d по медианам: {sign_med} из {n_cases}")
    ctrl_tab = pd.DataFrame(ctrl_rows)
    print(ctrl_tab.to_string(index=False))
    if bad:
        raise SystemExit("STOP: расхождения с контролем:\n" + pd.DataFrame(bad).to_string(index=False))
    if sign_mean != n_cases:
        raise SystemExit(f"STOP: знак d по средним совпадает со знаком σ только в {sign_mean} из {n_cases}")
    rec("все", "все", "совпадений знака d по средним и σ", sign_mean, n_cases, src_d, "расчёт шага 27")
    rec("все", "все", "совпадений знака d по медианам и σ", sign_med, n_cases, src_d, "расчёт шага 27")

    lines = [f"# 27. Описание типов по правилу интерпретации кластеров ({EXPL})", "",
             f"Сгенерировано `src/27_mirkin_rule.py`. Новая агрегация готовых долей (декабрь 2024, {len(D)} МО) по готовым меткам "
             "`kmeans_labels_final.parquet`; выборка и доли — `build_frame` шага 23. Без тестов, порогов и вердиктов.", "",
             "## Заморозка", "", *frozen_log, "",
             "## Правило", "",
             f"Источник: {CITATION}. Раздел 3 (Methodology): правило, описанное в Alvandyan T. A., Shalileh S. (2024) по Миркину:", "",
             "- формула (12): g_v = (Σ_i x_iv) / N — общее среднее признака v по всем N объектам;",
             "- формула (13): d_kv = (c_kv − g_v) / g_v — относительное отклонение центроида кластера k по признаку v.", "",
             "Здесь c_kv — среднее доли категории по МО типа (в процентах от «Все категории», без стандартизации, как рекомендуют "
             "авторы); g_v — среднее по всем МО. Для сравнения: d по медианам — (медиана типа − общая медиана) / общая медиана; "
             "σ по долям — среднее z-score типа из шага 23.", ""]
    for t in range(FINAL_K):
        rows = [{"категория": c, "среднее по типу, %": f"{res[(t, c)][0]:.2f}", "общее среднее, %": f"{g_mean[c]:.2f}",
                 "d по средним, %": pct(res[(t, c)][1]), "d по медианам, %": pct(res[(t, c)][2]),
                 "σ по долям": f"{res[(t, c)][3]:+.2f}"} for c in CATS]
        lines += [f"### Тип {t} (n = {res[(t, CATS[0])][4]})", "", f"Тип {t}: d по средним и по медианам, σ по долям — {EXPL}", "",
                  pd.DataFrame(rows).to_markdown(index=False, disable_numparse=True), ""]
    lines += ["Знак d по средним совпадает со знаком σ по долям по построению (общий числитель), поэтому совпадение не является "
              f"независимой проверкой: совпадений {sign_mean} из {n_cases}; для d по медианам — {sign_med} из {n_cases}.", "",
              "Относительное отклонение завышает категории с малой базой (общепит); величины разных категорий напрямую не сравнивать.", "",
              "## Контрольные сверки", "", ctrl_tab.to_markdown(index=False, disable_numparse=True), ""]
    text = "\n".join(lines)
    hits = {w: len(re.findall(w, text, flags=re.I)) for w in FORBIDDEN}
    nobr = re.sub(r"\[[+-]?\d+\.\d+; [+-]?\d+\.\d+\]", "", text)
    if any(hits.values()) or "[" in nobr or "]" in nobr:
        raise SystemExit(f"STOP: запрещённые слова {hits} или квадратные скобки вне интервала")
    print("запрещённые слова:", hits, "| квадратные скобки вне интервала: 0")
    check_frozen("после", frozen_log)
    i = lines.index("## Заморозка") + 2 + len(FROZEN)
    lines[i:i] = frozen_log[len(FROZEN):]
    out = pd.DataFrame(ROWS)
    if out.duplicated(["тип", "категория", "показатель"]).any():
        raise SystemExit("STOP: дубликаты ключа (тип, категория, показатель)")
    out.to_parquet(OUT_PATH, engine="pyarrow", index=False)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"записано: {OUT_PATH.name} ({len(out)} строк), {REPORT_PATH.name}")


if __name__ == "__main__":
    main()
