"""Шаг 31. Матрица «метод × шесть индексов качества» и сравнение методов на декабре 2024.

Описательный шаг: готовые методы sklearn и networkx, готовые индексы проекта, новых теорий и порогов нет.
Восемь методов (M1–M8) на одних и тех же 2004 МО выборки; индексы SW, CH, S_Dbw считаются в атрибутивном
пространстве варианта признаков метода (z-score внутри месяца), индексы MQ, AVI, AVU — на канонической
экономической сети декабря 2024 (kNN8 по косинусу z-оценок). Функции индексов — те же, что в шагах 10 и 11:
SW и CH — sklearn.metrics, S_Dbw — пакет s-dbw с параметрами config.SDBW_KW (шаг 10), MQ, AVI, AVU —
src/partition_metrics.py. Пороговое агрегирование Алескерова не реализовано.
Все контроли выполняются до записи; при нарушении код выхода 1 и ничего не записывается.

Вход:  data/processed/{category_shares, kmeans_labels_final, robust2_spec_labels, louvain_labels_2024_12_seed42,
       louvain_labels_2024_12, louvain_final_seed_2024_12.txt}, economic_networks/2024-12_std_knn8.parquet,
       notebooks/{10, 11, 11b, 13, 15}*.md (напечатанные значения для контролей)
Выход: data/processed/method_comparison_31.parquet, method_partitions_31.parquet,
       notebooks/31_method_comparison.md, notebooks/figures/31_method_comparison.{png,svg}
Запуск из корня проекта:  .venv/bin/python src/31_method_comparison.py
"""
import importlib
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from s_dbw import S_Dbw
from scipy.stats import rankdata
from sklearn.cluster import AgglomerativeClustering, KMeans, SpectralClustering
from sklearn.metrics import adjusted_rand_score, calinski_harabasz_score, silhouette_score
from sklearn.mixture import GaussianMixture

import config as C
from network_utils import SHARE_COLUMNS, month_shares, standardize_shares
from partition_metrics import partition_metrics

PROJECT_DIR = Path(__file__).resolve().parents[1]
S = C.STEP31
P = lambda rel: PROJECT_DIR / rel  # noqa: E731
CONTROLS: list[tuple[str, str, str, bool]] = []


def stop(msg: str) -> None:
    print(f"STOP: {msg}")
    sys.exit(1)


def ctl(name: str, got, expected, ok: bool | None = None) -> None:
    """Запись контроля; при расхождении STOP (до записи любых файлов)."""
    ok = (got == expected) if ok is None else ok
    sh = lambda v: str(v) if len(str(v)) <= 150 else str(v)[:150] + "…"  # noqa: E731
    CONTROLS.append((name, sh(got), sh(expected), bool(ok)))
    print(f"[{'ok' if ok else 'НАРУШЕН'}] {name}: {sh(got)} (ожидалось {sh(expected)})")
    if not ok:
        stop(f"{name}: получено {sh(got)}, ожидалось {sh(expected)}")


def need(rel: str) -> Path:
    if not P(rel).exists():
        stop(f"нет файла {rel}: предыдущий шаг не выполнен")
    return P(rel)


# ============================================================================ напечатанные значения notebooks
def num_tol(s: str) -> float:
    """Допуск: половина последней напечатанной цифры."""
    s = s.replace(",", "")
    m = re.search(r"\.(\d+)$", s)
    return 0.5 * 10 ** -(len(m.group(1)) if m else 0)


def fnum(s: str) -> float:
    return float(s.replace(",", ""))


def cells(line: str) -> list[str]:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def find_line(rel: str, pattern: str, start: int = 0) -> tuple[int, str]:
    lines = need(rel).read_text(encoding="utf-8").split("\n")
    for i in range(start, len(lines)):
        if re.search(pattern, lines[i]):
            return i + 1, lines[i]
    stop(f"в {rel} нет строки по шаблону {pattern}")


def printed() -> dict:
    """Напечатанные значения: (файл, строка, значение, допуск)."""
    out: dict = {}
    ln, t = find_line(S["NB10_PATH"], r"^\|\s+6\.0000 \|")
    c = cells(t)
    for name, i in (("SW", 2), ("CH", 3), ("S_Dbw", 4)):
        out[("10", "M1", name)] = (S["NB10_PATH"], ln, fnum(c[i]), num_tol(c[i]))
    ln, t = find_line(S["NB11_PATH"], r"\| Louvain \(resolution=1\.0, seed=42\)")
    c = cells(t)
    out[("11", "K", "K")] = (S["NB11_PATH"], ln, int(c[1]), 0.0)
    for name, i in (("MQ", 2), ("AVI", 3), ("AVU", 4)):
        out[("11", "M8", name)] = (S["NB11_PATH"], ln, fnum(c[i]), num_tol(c[i]))
    ln, t = find_line(S["NB11_PATH"], r"\| KMeans k=6 \(kmeans_labels_final\)")
    c = cells(t)
    for name, i in (("MQ", 2), ("AVI", 3), ("AVU", 4)):
        out[("11", "M1", name)] = (S["NB11_PATH"], ln, fnum(c[i]), num_tol(c[i]))
    ln, t = find_line(S["NB11_PATH"], r"2004 МО, [\d,]+ рёбер")
    out[("11", "net", "edges")] = (S["NB11_PATH"], ln, int(re.search(r"2004 МО, ([\d,]+) рёбер", t).group(1).replace(",", "")), 0.0)
    ln, t = find_line(S["NB11B_PATH"], r"\| Louvain, лучший по MQ \(seed=\d+\)")
    c = cells(t)
    out[("11b", "final", "K")] = (S["NB11B_PATH"], ln, int(c[1]), 0.0)
    for name, i in (("MQ", 2), ("AVI", 3), ("AVU", 4)):
        out[("11b", "final", name)] = (S["NB11B_PATH"], ln, fnum(c[i]), num_tol(c[i]))
    ln, t = find_line(S["NB11B_PATH"], r"\| KMeans k=6 \(kmeans_labels_final\)")
    c = cells(t)
    for name, i in (("MQ", 2), ("AVI", 3), ("AVU", 4)):
        out[("11b", "M1", name)] = (S["NB11B_PATH"], ln, fnum(c[i]), num_tol(c[i]))
    ln, t = find_line(S["NB13_PATH"], r"^\| 2024-12 \|\s+\d+ \|\s+\d+ \|\s+[\d.]+ \|")
    c = cells(t)
    out[("13", "best20", "seed")] = (S["NB13_PATH"], ln, int(c[1]), 0.0)
    out[("13", "best20", "K")] = (S["NB13_PATH"], ln, int(c[2]), 0.0)
    for name, i in (("MQ", 3), ("AVI", 4), ("AVU", 5)):
        out[("13", "best20", name)] = (S["NB13_PATH"], ln, fnum(c[i]), num_tol(c[i]))
    ln0, _ = find_line(S["NB15_PATH"], r"^Декабрь 2024:")
    ln, t = find_line(S["NB15_PATH"], r"^\|\s+6\.000 \|", ln0)
    c = cells(t)
    for j, v in enumerate("ABC"):
        out[("15", v, "SW")] = (S["NB15_PATH"], ln, fnum(c[1 + j]), num_tol(c[1 + j]))
        out[("15", v, "CH/N")] = (S["NB15_PATH"], ln, fnum(c[4 + j]), num_tol(c[4 + j]))
    return out


# ============================================================================ данные
def load_data() -> dict:
    shares = pd.read_parquet(need(S["SHARES_PATH"]))
    month = month_shares(shares, C.MONTH)
    ids = month["territory_id"].to_numpy()
    km = pd.read_parquet(need(S["KMEANS_LABELS_PATH"])).set_index("territory_id")["cluster"]
    ctl("выборка: 2004 МО, совпадают category_shares (декабрь) и kmeans_labels_final", (len(ids), sorted(ids.tolist()) == sorted(km.index.tolist())), (2004, True))
    f15 = importlib.import_module("15a_feature_spec_full")
    raw = month[SHARE_COLUMNS].to_numpy()
    X, desc = {}, {}
    for v in "ABC":
        F = f15.features(raw, v)
        X[v], _ = standardize_shares(F)
        desc[v] = F.shape[1]
    ctl("пространства признаков: размерности A, B, C", [desc["A"], desc["B"], desc["C"]], [5, 5, 6])
    net = pd.read_parquet(need(f"data/processed/economic_networks/{C.MONTH}_std_knn{C.ECON_KNN_K}.parquet"))
    G = nx.Graph()
    G.add_weighted_edges_from(((int(x), int(y), float(w)) for x, y, w in net[["territory_id_x", "territory_id_y", S["NETWORK_WEIGHT"]]].itertuples(index=False)),
                              weight=S["NETWORK_WEIGHT"])
    ctl("сеть: узлы = МО выборки", sorted(G.nodes) == sorted(ids.tolist()), True)
    ctl("сеть: веса рёбер положительны", int((net[S["NETWORK_WEIGHT"]] <= 0).sum()), 0)
    comps = nx.number_connected_components(G)
    return {"ids": ids, "X": X, "G": G, "km": km.loc[ids].to_numpy(), "raw": raw, "f15": f15, "components": comps,
            "spec": pd.read_parquet(need(S["SPEC_LABELS_PATH"]))}


def lab_dict(ids, labels) -> dict:
    return {int(i): int(l) for i, l in zip(ids, labels)}


def index_values(X: np.ndarray, G: nx.Graph, ids, labels) -> tuple[dict, dict]:
    """Шесть индексов; недоопределённые — NaN и примечание."""
    vals, notes = {}, {}
    for name, fn in (("SW", lambda: silhouette_score(X, labels)), ("CH", lambda: calinski_harabasz_score(X, labels)),
                     ("S_Dbw", lambda: S_Dbw(X, labels, **C.SDBW_KW))):
        try:
            vals[name] = float(fn())
        except Exception as e:  # noqa: BLE001
            vals[name], notes[name] = np.nan, f"не определён: {type(e).__name__}: {e}"
    try:
        m, _ = partition_metrics(G, lab_dict(ids, labels), weight=S["NETWORK_WEIGHT"])
        vals.update({k: float(m[k]) for k in ("MQ", "AVI", "AVU")})
    except Exception as e:  # noqa: BLE001
        for k in ("MQ", "AVI", "AVU"):
            vals[k], notes[k] = np.nan, f"не определён: {type(e).__name__}: {e}"
    return vals, notes


def main() -> None:
    D = load_data()
    ids, X, G = D["ids"], D["X"], D["G"]
    step11 = importlib.import_module("11_louvain")
    N, E = G.number_of_nodes(), G.number_of_edges()
    dens = 2 * E / (N * (N - 1))
    pr = printed()
    f, ln, v, tol = pr[("11", "net", "edges")]
    ctl(f"сеть: число рёбер = напечатанному ({f}, строка {ln})", E, int(v))

    # ---- метки методов
    parts: dict[str, np.ndarray] = {}
    meth = {m["code"]: m for m in S["METHODS"]}
    seed, ninit = S["SEED"], S["KMEANS_N_INIT"]
    # Louvain: итоговый запуск шага 11b (M8 и k метода M2)
    seed_f = int(need(S["LOUVAIN_FINAL_SEED_PATH"]).read_text().strip().split()[-1])
    lv_f = pd.read_parquet(need(S["LOUVAIN_FINAL_LABELS_PATH"])).set_index("territory_id")["cluster"].loc[ids].to_numpy()
    lv_f_new = np.array([step11.louvain_labels(G, seed_f)[int(i)] for i in ids])
    ctl(f"M8: итоговый Louvain шага 11b (seed {seed_f}, resolution и вес шага 11) = louvain_labels_2024_12.parquet, расхождений", int((lv_f_new != lv_f).sum()), 0)
    runs = pd.read_parquet(need(S["LOUVAIN_RUNS_PATH"]))
    ctl("шаг 11b: запусков 100, seed итогового запуска = лучший по MQ", (len(runs), int(runs.loc[runs["MQ"].idxmax(), "seed"])), (100, seed_f))
    K_L = int(len(set(lv_f.tolist())))
    f, ln, v, tol = pr[("11b", "final", "K")]
    ctl(f"K итогового Louvain = напечатанному ({f}, строка {ln})", K_L, int(v))
    parts["M8"] = lv_f
    # справочно: запуск шага 11 (seed 42)
    lv_saved = pd.read_parquet(need(S["LOUVAIN_STEP11_LABELS_PATH"])).set_index("territory_id")["cluster"].loc[ids].to_numpy()
    lv_new = np.array([step11.louvain_labels(G, C.LOUVAIN_SEED)[int(i)] for i in ids])
    ctl("справочно: метки Louvain шага 11 (seed 42) = сохранённым меткам, расхождений", int((lv_new != lv_saved).sum()), 0)
    K_REF = int(len(set(lv_saved.tolist())))
    f, ln, v, tol = pr[("11", "K", "K")]
    ctl(f"справочно: K Louvain шага 11 = напечатанному ({f}, строка {ln})", K_REF, int(v))
    # M1, M2
    parts["M1"] = KMeans(n_clusters=C.FINAL_K, random_state=seed, n_init=ninit).fit(X["A"]).labels_
    ctl("M1: метки KMeans (A, k = 6, random_state 42, n_init 10) = kmeans_labels_final, расхождений", int((parts["M1"] != D["km"]).sum()), 0)
    parts["M2"] = KMeans(n_clusters=K_L, random_state=seed, n_init=ninit).fit(X["A"]).labels_
    # M3, M4: сохранённые разбиения шага 15a и пересчёт тем же способом
    for code in ("M3", "M4"):
        v_ = meth[code]["variant"]
        sp = D["spec"]
        sp = sp[(sp["вариант"] == v_) & (sp["месяц"] == C.MONTH)].set_index("territory_id")["cluster"]
        ctl(f"{code}: сохранённые метки шага 15a (вариант {v_}, {C.MONTH}) покрывают 2004 МО", (len(sp), sorted(sp.index.tolist()) == sorted(ids.tolist())), (2004, True))
        saved = sp.loc[ids].to_numpy()
        re_ = D["f15"].best_of_runs(X[v_], C.FINAL_K)
        ctl(f"{code}: пересчёт лучшего по inertia из ROBUST2_N_RUNS запусков (функция 15a) = сохранённым меткам, расхождений", int((re_ != saved).sum()), 0)
        parts[code] = saved
    parts["M5"] = AgglomerativeClustering(n_clusters=C.FINAL_K, linkage=S["WARD_LINKAGE"]).fit(X["A"]).labels_
    gm = GaussianMixture(n_components=C.FINAL_K, covariance_type=S["GMM_COVARIANCE_TYPE"], n_init=S["GMM_N_INIT"], reg_covar=S["GMM_REG_COVAR"],
                         max_iter=S["GMM_MAX_ITER"], random_state=seed).fit(X["A"])
    ctl("M6: сходимость EM лучшего запуска", bool(gm.converged_), True)
    parts["M6"] = gm.predict(X["A"])
    A = nx.to_numpy_array(G, nodelist=[int(i) for i in ids], weight=S["SPECTRAL_WEIGHT"])
    ctl("M7: матрица смежности симметрична (макс. |A − Aᵀ|)", float(np.abs(A - A.T).max()), 0.0)
    A = (A + A.T) / 2
    parts["M7"] = SpectralClustering(n_clusters=C.FINAL_K, affinity=S["SPECTRAL_AFFINITY"], assign_labels=S["SPECTRAL_ASSIGN_LABELS"],
                                     n_init=S["SPECTRAL_N_INIT"], random_state=seed).fit(A).labels_
    order = [m["code"] for m in S["METHODS"]]
    ks = {c: int(len(set(parts[c].tolist()))) for c in order}
    ctl("число кластеров у методов: M1, M3, M4, M5, M6, M7 = 6; M2 и M8 = K итогового Louvain",
        [ks[c] for c in order], [6, K_L, 6, 6, 6, 6, 6, K_L])
    ctl("метки: целые, без пропусков, 2004 значений", all(len(parts[c]) == 2004 and np.issubdtype(np.asarray(parts[c]).dtype, np.integer) for c in order), True)

    # ---- индексы: таблица 1 — SW, CH, S_Dbw в собственном пространстве варианта метода; таблица 2 — в общем пространстве A
    vals, notes, valsA = {}, {}, {}
    for c in order:
        vals[c], notes[c] = index_values(X[meth[c]["variant"]], G, ids, parts[c])
        valsA[c], notesA = index_values(X["A"], G, ids, parts[c])
        notes[c].update({k: v for k, v in notesA.items() if k not in notes[c]})
    vref, _ = index_values(X["A"], G, ids, lv_saved)
    net_same = max(abs(vals[c][n] - valsA[c][n]) for c in order for n in ("MQ", "AVI", "AVU"))
    ctl("сетевые индексы в двух таблицах совпадают (макс. |Δ|)", net_same, 0.0)
    same_A = max(abs(vals[c][n] - valsA[c][n]) for c in order if meth[c]["variant"] == "A" for n in ("SW", "CH", "S_Dbw"))
    ctl("SW, CH, S_Dbw методов на варианте A в двух таблицах совпадают (макс. |Δ|)", same_A, 0.0)
    # ---- контроли по напечатанным значениям
    cmp_rows = []

    def compare(src, key, name, got, label):
        f_, ln_, v_, tol_ = pr[(src, key, name)]
        ok = abs(got - v_) <= tol_ + 1e-12
        cmp_rows.append([f"{f_}, строка {ln_}", label, f"{v_:g}", f"{got:.6g}", "да" if ok else "НЕТ"])
        ctl(f"{label}: {f_} строка {ln_} (допуск {tol_:g})", round(got, 6), v_, ok)

    for name in ("SW", "CH", "S_Dbw"):
        compare("10", "M1", name, vals["M1"][name], f"M1 {name}")
    for name in ("MQ", "AVI", "AVU"):
        compare("11", "M1", name, vals["M1"][name], f"M1 {name} (шаг 11)")
        compare("11b", "M1", name, vals["M1"][name], f"M1 {name} (шаг 11b)")
        compare("11", "M8", name, vref[name], f"справочно: Louvain шага 11 (K = {K_REF}) {name}")
        compare("11b", "final", name, vals["M8"][name], f"M8 {name} (итоговый Louvain шага 11b)")
    # шаг 15: KMeans (random_state 42) по вариантам A, B, C; SW и CH/N
    for v_ in "ABC":
        lab = KMeans(n_clusters=C.FINAL_K, random_state=seed, n_init=ninit).fit(X[v_]).labels_
        compare("15", v_, "SW", float(silhouette_score(X[v_], lab)), f"вариант {v_}: SW KMeans k = 6")
        compare("15", v_, "CH/N", float(calinski_harabasz_score(X[v_], lab)) / len(ids), f"вариант {v_}: CH/N KMeans k = 6")
    best = None
    for sd in range(C.LOUVAIN_MONTHLY_N_SEEDS):
        lb = step11.louvain_labels(G, sd)
        mq = partition_metrics(G, lb, weight=S["NETWORK_WEIGHT"])[0]
        if best is None or mq["MQ"] > best[1]["MQ"]:
            best = (sd, mq)
    f_, ln_, sd13, _ = pr[("13", "best20", "seed")]
    ctl(f"справочно: лучший по MQ из {C.LOUVAIN_MONTHLY_N_SEEDS} seed, номер seed = напечатанному ({f_}, строка {ln_})", best[0], int(sd13))
    for name in ("MQ", "AVI", "AVU"):
        compare("13", "best20", name, float(best[1][name]), f"справочно: шаг 13 (лучший из 20 seed) {name}")

    # ---- таблицы рангов
    idx = S["INDICES"]
    T = {}
    for key, V in (("own", vals), ("A", valsA)):
        rank, borda, mean_rank = rank_table(V, order, idx)
        ctl(f"ранги ({key}): сумма рангов по каждому индексу = n(n+1)/2", [round(float(np.nansum([rank[c][n] for c in order])), 9) for n in idx], [36.0] * 6)
        T[key] = {"V": V, "rank": rank, "borda": borda, "mean": mean_rank}
    undefined = [(c, n, notes[c][n]) for c in order for n in notes[c]]
    ctl("неопределённых индексов", len(undefined), 0)
    ari = pd.DataFrame([[adjusted_rand_score(parts[a], parts[b]) for b in order] for a in order], index=order, columns=order)
    ctl("ARI: диагональ = 1, матрица симметрична", (bool(np.allclose(np.diag(ari), 1.0)), bool(np.allclose(ari, ari.T))), (True, True))

    # ---- выходные данные в памяти
    rows = []
    for key, space_name in (("own", "собственное пространство варианта метода"), ("A", "общее пространство A")):
        t = T[key]
        for c in order:
            m = meth[c]
            kk = ks[c]
            var = m["variant"] if key == "own" else "A"
            base = {"таблица": space_name, "метод": f"{c} {m['name']}", "вариант признаков": var, "k": kk}
            for name in idx:
                rows.append({**base, "индекс": name, "значение": t["V"][c][name], "направление": S["DIRECTIONS"][name], "ранг": t["rank"][c][name],
                             "примечание": notes[c].get(name, "")})
            rows.append({**base, "индекс": "сумма рангов (Борда)", "значение": t["borda"][c], "направление": "меньше лучше", "ранг": np.nan, "примечание": ""})
            rows.append({**base, "индекс": "средний ранг (Борда)", "значение": t["mean"][c], "направление": "меньше лучше", "ранг": np.nan, "примечание": ""})
    tab = pd.DataFrame(rows)
    for sp_ in tab["таблица"].unique():
        for nm in ("сумма рангов (Борда)", "средний ранг (Борда)"):
            msk = (tab["индекс"] == nm) & (tab["таблица"] == sp_)
            tab.loc[msk, "ранг"] = rankdata(np.round(tab.loc[msk, "значение"].to_numpy(), 12), method="average")
    tab["k"] = tab["k"].astype(int)
    part_df = pd.DataFrame({"territory_id": ids.astype(int), **{f"{c} {meth[c]['name']}": parts[c].astype(int) for c in order}})
    ctl("method_partitions: 2004 строк, territory_id уникален", (len(part_df), bool(part_df["territory_id"].is_unique)), (2004, True))

    fig1 = draw_figure(T["own"], order, meth, idx, "Метод × индексы качества, декабрь 2024 (2004 МО)\nSW, CH, S_Dbw в собственном пространстве метода",
                       "SW, CH, S_Dbw — в собственном пространстве варианта признаков метода (M3 — B, M4 — C, остальные — A), MQ, AVI, AVU — на канонической сети.")
    fig2 = draw_figure(T["A"], order, meth, idx, "Метод × индексы качества, декабрь 2024 (2004 МО)\nSW, CH, S_Dbw в общем пространстве A",
                       "SW, CH, S_Dbw — в общем пространстве варианта A для всех методов, MQ, AVI, AVU — на канонической сети.")
    report = build_report(meth, order, idx, T, ks, parts, ari, cmp_rows, undefined, N, E, dens, D, K_L, K_REF, vref, [fig1, fig2], seed_f)
    stems = {w: len(re.findall(re.escape(w), report, flags=re.I)) for w in S["FORBIDDEN_STEMS"]}
    ctl("запрещённые слова в md и на рисунках (основы)", {k: v for k, v in stems.items() if v}, {})
    nobr = re.sub(r"\[[+-]?\d+\.\d+; [+-]?\d+\.\d+\]", "", report)
    ctl("md: квадратные скобки только в формате интервала", ("[" in nobr or "]" in nobr), False)
    report = report.replace("@@CONTROLS@@", controls_table())
    report = report.replace("@@NCTL@@", str(len(CONTROLS)))
    if not all(c[3] for c in CONTROLS):
        stop("контроли")
    # ---- запись
    for rel in (S["OUT_TABLE_PATH"], S["OUT_PARTITIONS_PATH"], S["REPORT_PATH"], S["FIG_PNG_PATH"]):
        P(rel).parent.mkdir(parents=True, exist_ok=True)
    tab.to_parquet(P(S["OUT_TABLE_PATH"]), engine="pyarrow", index=False)
    part_df.to_parquet(P(S["OUT_PARTITIONS_PATH"]), engine="pyarrow", index=False)
    for fg, pp, sv in ((fig1, S["FIG_PNG_PATH"], S["FIG_SVG_PATH"]), (fig2, S["FIG2_PNG_PATH"], S["FIG2_SVG_PATH"])):
        fg["fig"].savefig(P(pp), dpi=S["FIG_DPI"], metadata={"Software": None})
        fg["fig"].savefig(P(sv), format="svg", metadata={"Date": None})
    P(S["REPORT_PATH"]).write_text(report, encoding="utf-8")
    print("записано:", S["OUT_TABLE_PATH"], S["OUT_PARTITIONS_PATH"], S["REPORT_PATH"], S["FIG_PNG_PATH"], S["FIG_SVG_PATH"], S["FIG2_PNG_PATH"], S["FIG2_SVG_PATH"])
    print(tab.pivot_table(index=["таблица", "метод"], columns="индекс", values="значение", sort=False).round(4).to_string())


def rank_table(V: dict, order: list, idx: list) -> tuple[dict, dict, dict]:
    rank = {c: {} for c in order}
    for name in idx:
        v = np.array([V[c][name] for c in order], float)
        sign = -1.0 if S["DIRECTIONS"][name] == "больше лучше" else 1.0
        ok = ~np.isnan(v)
        r = np.full(len(order), np.nan)
        r[ok] = rankdata(np.round(sign * v[ok], 12), method="average")
        for c, rr in zip(order, r):
            rank[c][name] = rr
    borda = {c: float(np.nansum([rank[c][n] for n in idx])) for c in order}
    return rank, borda, {c: borda[c] / len(idx) for c in order}

def controls_table() -> str:
    head = "| контроль | получено | ожидалось | статус |\n|---|---|---|---|"
    return head + "\n" + "\n".join(f"| {a.replace('|', '/')} | {b.replace('|', '/')} | {c.replace('|', '/')} | {'пройден' if d else 'НАРУШЕН'} |" for a, b, c, d in CONTROLS)


# ============================================================================ рисунок
def draw_figure(t, order, meth, idx, title, note) -> dict:
    vals, rank, mean_rank = t["V"], t["rank"], t["mean"]
    plt.rcParams.update({"font.family": "DejaVu Sans", "svg.hashsalt": "step31", "svg.fonttype": "path"})
    w, h = S["FIG_SIZE_IN"]
    fig = plt.figure(figsize=(w, h))
    ax = fig.add_axes([0.255, 0.20, 0.655, 0.60])
    cols = idx + ["средний ранг"]
    M = np.array([[rank[c][n] for n in idx] + [mean_rank[c]] for c in order])
    im = ax.imshow(M, cmap="RdYlGn_r", vmin=1, vmax=len(order), aspect="auto")
    ax.set_xticks(range(len(cols)))
    arrows = {"больше лучше": "↑", "меньше лучше": "↓"}
    ax.set_xticklabels([f"{n} {arrows[S['DIRECTIONS'][n]]}" for n in idx] + ["средний ранг ↓"], fontsize=11)
    ax.xaxis.tick_top()
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([f"{c} {meth[c]['name']}" for c in order], fontsize=10.5)
    for i, c in enumerate(order):
        for j, n in enumerate(idx):
            ax.text(j, i, f"{vals[c][n]:.3f}" if abs(vals[c][n]) < 100 else f"{vals[c][n]:.1f}", ha="center", va="center", fontsize=10.5,
                    color="#FFFFFF" if (rank[c][n] <= 1.5 or rank[c][n] >= len(order) - 0.5) else "#111111")
        ax.text(len(idx), i, f"{mean_rank[c]:.2f}", ha="center", va="center", fontsize=10.5, fontweight="bold", color="#111111")
    ax.set_xticks(np.arange(-.5, len(cols), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(order), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.5)
    ax.tick_params(which="both", length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cb.set_label("ранг (1 — лучший)", fontsize=10.5)
    cb.set_ticks(range(1, len(order) + 1))
    fig.text(0.03, 0.09, "Цвет — ранг метода по индексу (1 — лучший), число в ячейке — значение индекса.\n" + note,
             fontsize=9.5, ha="left", va="center", color="#333333", linespacing=1.5)
    fig.suptitle(title, x=0.03, ha="left", y=0.955, fontsize=12.5, fontweight="bold", linespacing=1.4)
    fig.canvas.draw()
    rd = fig.canvas.get_renderer()
    bbs = [t.get_window_extent(rd) for t in fig.findobj(matplotlib.text.Text) if t.get_text().strip() and t.get_visible()]
    bb = matplotlib.transforms.Bbox.union(bbs + [ax.get_window_extent(rd), cb.ax.get_window_extent(rd)])
    dpi = fig.dpi
    margins = [float(x) for x in (bb.x0 / dpi, (fig.get_figwidth() * dpi - bb.x1) / dpi, bb.y0 / dpi, (fig.get_figheight() * dpi - bb.y1) / dpi)]
    inside = []
    for t_ in fig.findobj(matplotlib.text.Text):
        if t_.get_text().strip() and t_.get_visible():
            e = t_.get_window_extent(rd)
            inside.append((e.x0 / dpi, (fig.get_figwidth() * dpi - e.x1) / dpi, e.y0 / dpi, (fig.get_figheight() * dpi - e.y1) / dpi))
    min_in = float(min(min(r) for r in inside))
    ctl(f"рисунок: границы всех текстовых элементов ({len(inside)} шт.: заголовок, подписи осей, подписи значений, сноска) внутри холста с запасом не менее 0.05 дюйма (get_window_extent отрисованного рисунка); минимальный запас, дюймы",
        round(min_in, 3), 0.05, min_in >= 0.05)
    ctl("рисунок: размер холста 11.7×6 дюйма", [round(float(x), 2) for x in fig.get_size_inches()], [11.7, 6.0])
    ctl("рисунок: поля не менее 0.25 дюйма (лево, право, низ, верх)", [round(float(m), 3) for m in margins], [S["FIG_MARGIN_MIN_IN"]] * 4,
        all(m >= S["FIG_MARGIN_MIN_IN"] for m in margins))
    return {"fig": fig, "margins": margins}


# ============================================================================ отчёт
def md_table(header, rows) -> str:
    cell = lambda x: str(x).replace("|", "/").replace("\n", " ")  # noqa: E731
    return "\n".join(["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"] + ["| " + " | ".join(cell(c) for c in r) + " |" for r in rows])


def build_report(meth, order, idx, T, ks, parts, ari, cmp_rows, undefined, N, E, dens, D, K_L, K_REF, vref, figs, seed_f) -> str:
    fmt = lambda V, c, n: (f"{V[c][n]:.4f}" if abs(V[c][n]) < 100 else f"{V[c][n]:.2f}")  # noqa: E731
    rk = lambda r: f"{r:g}"  # noqa: E731
    head = ["код", "метод", "k"] + [f"{n} ({'↑' if S['DIRECTIONS'][n] == 'больше лучше' else '↓'})" for n in idx]
    mrows = {key: [[c, meth[c]["name"], ks[c]] + [f"{fmt(T[key]['V'], c, n)} (ранг {rk(T[key]['rank'][c][n])})" for n in idx] for c in order] for key in ("own", "A")}
    own, com = T["own"], T["A"]
    brows = sorted(([c, meth[c]["name"], f"{own['borda'][c]:g}", f"{own['mean'][c]:.3f}", f"{com['borda'][c]:g}", f"{com['mean'][c]:.3f}"] for c in order), key=lambda r: (float(r[2]), r[0]))
    sizes = [[c, meth[c]["name"], ks[c], ", ".join(str(int(x)) for x in sorted(np.bincount(parts[c]).tolist(), reverse=True))] for c in order]
    aris = [[c] + [f"{ari.loc[c, d]:.3f}" for d in order] for c in order]
    space_rows = [[c, meth[c]["name"], f"вариант {meth[c]['variant']}" + (" (собственное)" if meth[c]["variant"] in "BC" else ""), "вариант A (общее)", "каноническая сеть декабря 2024 (kNN8 по косинусу z-оценок варианта A)"] for c in order]
    diff_rows = [[c, meth[c]["name"]] + [f"{own['V'][c][n]:.4f} / {com['V'][c][n]:.4f}" for n in ("SW", "CH", "S_Dbw")] for c in order if meth[c]["variant"] != "A"]
    L = [
        "# 31. Матрица «метод × шесть индексов качества» и сравнение методов (декабрь 2024)", "",
        "Сгенерировано `src/31_method_comparison.py`. Шаг описательный: готовые методы sklearn и networkx, готовые индексы проекта, новых порогов и предрегистрации нет. "
        "Один месяц (декабрь 2024), 2004 МО выборки.", "",
        "## 1. Данные, сеть и пространства индексов", "",
        "- Атрибутивное пространство: z-оценки внутри месяца (`network_utils.standardize_shares`). Вариант A — 5 долей от «Все категории» (канон); вариант B — 5 долей, нормированных к сумме пяти; "
        "вариант C — 5 долей и доля «прочее» (определения: `features` в `src/15a_feature_spec_full.py`).",
        f"- Сеть для индексов MQ, AVI, AVU: каноническая экономическая сеть декабря 2024 (kNN{C.ECON_KNN_K} по косинусу z-оценок варианта A, шаг 05), вес ребра — {S['NETWORK_WEIGHT']}; одна и та же для всех восьми методов. "
        f"Узлов {N}, рёбер {E}, плотность {dens:.5f}, компонент связности {D['components']}.",
        "- **Пространство признаков для SW, CH, S_Dbw.** Таблица 1 («собственное пространство»): M1, M2, M5, M6, M7, M8 — в варианте A; M3 — в собственном пространстве B; M4 — в собственном пространстве C. "
        "Пространства разные, поэтому добавлена таблица 2: SW, CH, S_Dbw всех восьми методов в общем пространстве варианта A (метки M3 и M4 оцениваются в A). "
        "Сетевые индексы MQ, AVI, AVU в двух таблицах одинаковы.", "",
        md_table(["код", "метод", "SW, CH, S_Dbw: таблица 1", "SW, CH, S_Dbw: таблица 2", "MQ, AVI, AVU (обе таблицы)"], space_rows), "",
        "## 2. Методы и параметры", "",
        md_table(["код", "метод", "вариант признаков метода", "k", "параметры"], [
            ["M1", "KMeans", "A", 6, f"random_state {S['SEED']}, n_init {S['KMEANS_N_INIT']}; метки совпадают с `kmeans_labels_final.parquet`"],
            ["M2", "KMeans", "A", K_L, f"k = число сообществ итогового Louvain шага 11b ({K_L}); random_state {S['SEED']}, n_init {S['KMEANS_N_INIT']}"],
            ["M3", "KMeans", "B", 6, "разбиение шага 15a (лучший по inertia из ROBUST2_N_RUNS запусков); сохранённые метки и пересчёт совпали"],
            ["M4", "KMeans", "C", 6, "то же для варианта C"],
            ["M5", "Ward (AgglomerativeClustering)", "A", 6, f"linkage = {S['WARD_LINKAGE']}"],
            ["M6", "Гауссова смесь (GaussianMixture)", "A", 6, f"covariance_type = {S['GMM_COVARIANCE_TYPE']}, n_init {S['GMM_N_INIT']}, reg_covar {S['GMM_REG_COVAR']:g}, max_iter {S['GMM_MAX_ITER']}, random_state {S['SEED']}"],
            ["M7", "Спектральная кластеризация (SpectralClustering)", "A", 6, f"affinity = {S['SPECTRAL_AFFINITY']} на симметризованной матрице смежности сети (вес {S['SPECTRAL_WEIGHT']}), assign_labels = {S['SPECTRAL_ASSIGN_LABELS']}, n_init {S['SPECTRAL_N_INIT']}, random_state {S['SEED']}"],
            ["M8", "Louvain (networkx)", "A", K_L, f"итоговый запуск шага 11b: лучший по MQ из 100 запусков (seed {seed_f}), resolution = {C.LOUVAIN_RESOLUTION}, вес {C.LOUVAIN_WEIGHT}; метки = `louvain_labels_2024_12.parquet`"]]), "",
        f"Запуск Louvain шага 11 (seed {C.LOUVAIN_SEED}, K = {K_REF}) оставлен только справочным контролем: его метки воспроизведены, "
        f"значения в атрибутивном пространстве A и на сети: SW {vref['SW']:.4f}, CH {vref['CH']:.2f}, S_Dbw {vref['S_Dbw']:.4f}, MQ {vref['MQ']:.4f}, AVI {vref['AVI']:.4f}, AVU {vref['AVU']:.4f}; в рейтинги он не входит. "
        "Число кластеров Louvain задаёт алгоритм, а не исследователь.", "",
        "## 3. Определение индексов", "",
        "- **SW** (Silhouette Width, `sklearn.metrics.silhouette_score`), **CH** (Calinski–Harabasz, `calinski_harabasz_score`): больше лучше. **S_Dbw** (пакет `s-dbw`, метод Halkidi, центр кластера — ближайшая к среднему точка, параметры `SDBW_KW` из config): меньше лучше. Эти три индекса считаются теми же вызовами, что в шаге 10 (`src/10_kmeans_k_selection.py`); в `src/partition_metrics.py` их нет.",
        "- **MQ** (модулярность Ньюмана, `networkx.community.modularity`), **AVI** (среднее Isolability), **AVU** (среднее Unifiability) — `src/partition_metrics.py`, формулы (18)–(21) Shalileh, Antonov, Tsyplakova (2025); подробности и свойства — `notebooks/11_louvain_metrics.md` и docstring модуля. Направления: MQ больше лучше, AVI больше лучше, AVU меньше лучше (как в шаге 11).",
        "- Ранг 1 — лучший метод по индексу с учётом направления; одинаковые значения получают средний ранг (сравнение после округления до 12 знаков). Сумма рангов и средний ранг (Борда) считаются по шести индексам; меньше лучше.",
        "- Пороговое агрегирование Алескерова не реализовано.", "",
        "## 4. Таблица 1: SW, CH, S_Dbw в собственном пространстве варианта метода (значения и ранги)", "",
        md_table(head, mrows["own"]), "",
        "Стрелка в заголовке — направление: ↑ больше лучше, ↓ меньше лучше. MQ, AVI, AVU — на канонической сети.", "",
        "## 5. Таблица 2: SW, CH, S_Dbw в общем пространстве варианта A (значения и ранги)", "",
        md_table(head, mrows["A"]), "",
        "Значения SW, CH, S_Dbw методов M3 и M4 в двух таблицах (собственное пространство / общее A):", "",
        md_table(["код", "метод", "SW", "CH", "S_Dbw"], diff_rows), "",
        "## 6. Суммарный ранг (Борда): обе таблицы рядом", "",
        md_table(["код", "метод", "сумма рангов, таблица 1", "средний ранг, таблица 1", "сумма рангов, таблица 2", "средний ранг, таблица 2"], brows), "",
        "Таблица упорядочена по сумме рангов таблицы 1. Рейтинг Борда не заменяет содержательный выбор метода: он складывает ранги индексов, измеренных в разных пространствах и при разном числе кластеров.", "",
        "## 7. Размеры кластеров", "",
        md_table(["код", "метод", "k", "размеры (по убыванию)"], sizes), "",
        "## 8. Попарное ARI между методами (матрица 8 × 8)", "",
        md_table(["метод"] + order, aris), "",
        "## 9. Рисунки", "",
        f"- `{S['FIG_PNG_PATH']}` и `{S['FIG_SVG_PATH']}`: таблица 1 (собственное пространство). Поля (лево, право, низ, верх), дюймы: " + ", ".join(f"{m:.2f}" for m in figs[0]["margins"]) + ".",
        f"- `{S['FIG2_PNG_PATH']}` и `{S['FIG2_SVG_PATH']}`: таблица 2 (общее пространство A). Поля, дюймы: " + ", ".join(f"{m:.2f}" for m in figs[1]["margins"]) + ".",
        f"- Тепловая карта рангов (метод × шесть индексов) с подписанными значениями, справа средний ранг. Холст {S['FIG_SIZE_IN'][0]}×{S['FIG_SIZE_IN'][1]} дюйма, PNG {S['FIG_DPI']} dpi, шрифт DejaVu Sans.", "",
        "## 10. Неопределённые индексы", "",
        ("Неопределённых индексов нет." if not undefined else md_table(["метод", "индекс", "примечание"], undefined)), "",
        "## 11. Сверка с напечатанными значениями шагов 10, 11, 11b, 13, 15 (допуск — половина последней напечатанной цифры)", "",
        md_table(["файл и строка", "значение", "в notebooks", "в новом расчёте", "совпало"], cmp_rows), "",
        "Новые числа, которых нет в notebooks: SW, CH, S_Dbw для Louvain, ward, гауссовой смеси, спектральной кластеризации и KMeans с k = K Louvain; "
        "MQ, AVI, AVU для вариантов B и C, ward, гауссовой смеси, спектральной кластеризации и KMeans с k = K Louvain; значения таблицы 2; матрица ARI; ранги и суммы рангов. "
        "Значения SW и CH/N вариантов B и C из шага 15 сверены на KMeans с random_state 42 (как в шаге 15a, таблица по k); метки M3 и M4 — разбиения 15a (лучший по inertia из нескольких запусков).", "",
        "## 12. Ограничения", "",
        "- Индексы считаются в разных пространствах: SW, CH, S_Dbw — в признаковом пространстве (таблица 1: собственное пространство варианта метода, таблица 2: общее A), MQ, AVI, AVU — на сети; значения индексов разных вариантов признаков (A, B, C) несравнимы напрямую.",
        f"- Число кластеров Louvain задаёт алгоритм (итоговый запуск 11b: K = {K_L}); M2 взят с тем же k для сопоставимости, но все шесть индексов зависят от k, поэтому сравнение методов с k = 6 и k = {K_L} неравноценно.",
        "- M7 и M8 строятся по сети, M1–M6 — по признакам; MQ для M8 прямо оптимизируется, что даёт преимущество по сетевым индексам.",
        "- Рейтинги Борда не заменяют содержательный выбор метода.",
        "- Пороговое агрегирование Алескерова не реализовано.",
        "- Один месяц (декабрь 2024); устойчивость выводов по времени не проверялась.",
        "- Итоговый Louvain 11b — лучший по MQ из 100 запусков; в 11b показано, что запуски заметно различаются по K и по составу сообществ.", "",
        "## 13. Контроли (выполнены до записи файлов; всего @@NCTL@@)", "",
        "@@CONTROLS@@", "",
    ]
    return "\n".join(L)

if __name__ == "__main__":
    main()
