"""ANCC-формация как cross-well spatial предиктор (ВХОДНОЙ канал, не aux-таргет — не путать со
STRAT-CAL exp_16x, который использует data.TOPS[0] как GT для aux-loss на декодерных фичах).
Идея команды nocturnebflat123 (NNLS/стекинг): LOWO-предсказание ANCC(X,Y) соседних скважин ->
после вычитания h_z даёт TVT-сопоставимый сигнал, которого сеть сейчас не видит (нулевой доступ
к geometрии СОСЕДНИХ скважин — видит только свою). exp_085 PF-ANCC (путь-прайор из GR-мисфита той
же скважины) отклонён по другой причине (избыточная подсказка на тех же данных) — здесь источник
информации принципиально другой (чужие X,Y,ANCC), риск shortcut ниже, но не нулевой.

densequad_all_rows/gpuk_all_rows — ВЕРБАТИМ порт (не переизобретать матем.) из
artifacts/external/rogii_form_ancc_dump_source.py (nocturnebflat123 production kernel, v114
lineage; _DenseQuadImputer99 / _gpk_np_gp+_gpk_gpuk_from_pool). Корректность порта проверяется
построчным сравнением с artifacts/external/form_ancc_lowo.parquet (LOWO train dump) — см.
scratchpad/verify_form_ancc.py. Не трогать эту математику без повторной верификации против
parquet.

API:
  get_form_pred(wid, split="train") -> np.ndarray[n]   f_gp по всем строкам wid__horizontal_well.csv
    train: читает прекомпьюченный LOWO parquet (лик-safe для ЛЮБОГО фолда, self исключён при дампе).
    иначе (test/произвольный wid): свежий расчёт против пула всех 773 train-скважин (без LOWO —
    train-пул не содержит настоящих test id; ИСКЛЮЧЕНИЕ: 3 "видимых" test-примера data.TEST_IDS
    физически совпадают с одноимённой train-скважиной — self не исключается, см. PortVerify notes).
  compute_gp(wid, xy_all, lowo) -> np.ndarray[n]        свежий расчёт с явным контролем LOWO (для
    верификации и для build_chunk_sw-style путей, где нужен полный контроль).
"""
import glob
import os

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

import data

SPW = 60
K_NEI = 10

_PARQUET_PATH = f"{data.ROOT}/artifacts/external/form_ancc_lowo.parquet"

_NP_LS = np.array([0.05, 0.07, 0.1, 0.14, 0.2, 0.28, 0.4, 0.55, 0.8, 1.1, 1.5, 2.1, 3.0])
_NP_S = np.array([0.03, 0.1, 0.3, 0.6, 1.0, 1.5, 2.5, 4.0, 7.0, 12.0])
_NP_GMIN, _NP_GMAX = 1.5, 40.0

def _pdist(A, B):
    return np.sqrt(np.maximum((A * A).sum(1)[:, None] + (B * B).sum(1)[None, :] - 2.0 * A @ B.T, 0.0))

def _np_gp(Xtr, y, Xev):
    ym = y.mean(); ys = y.std() + 1e-9; yn = (y - ym) / ys
    Dtt = _pdist(Xtr, Xtr)
    G = np.geomspace(_NP_GMIN, _NP_GMAX, 8)
    best = None
    for ls in _NP_LS:
        R = np.exp(-Dtt / ls)
        w, V = np.linalg.eigh(R)
        w = np.maximum(w, 0.0)
        Vty = V.T @ yn
        Vty2 = Vty * Vty
        for s in _NP_S:
            for g in G:
                d = s * w + g
                nlml = 0.5 * float((Vty2 / d).sum()) + 0.5 * float(np.log(d).sum())
                if best is None or nlml < best[0]:
                    best = (nlml, ls, float(s), float(g), w, V, Vty)
    _, ls, s, g, w, V, Vty = best
    d = s * w + g
    alpha = V @ (Vty / d)
    Kte = s * np.exp(-_pdist(Xev, Xtr) / ls)
    m_ev = (Kte @ alpha) * ys + ym
    VtKe = V.T @ Kte.T
    var = np.maximum(s - (VtKe * VtKe / d[:, None]).sum(0), 0.0)
    return m_ev, np.sqrt(var) * ys

def gpuk_all_rows(P, cx, cy, ev_xy):
    """P: [m,3] (X,Y,ANCC) пул соседей. ev_xy: [n,2] траектория запрашиваемой скважины.
    Возвращает (f_gp[n], sig_gp[n])."""
    xg, yg, a = P[:, 0], P[:, 1], P[:, 2]
    M = ev_xy - ev_xy.mean(0)
    _wv, V = np.linalg.eigh(M.T @ M)
    Rm = V[:, ::-1].T
    ctr = np.array([cx, cy])
    def rot(xy):
        return (xy - ctr) @ Rm.T
    rt = rot(np.column_stack([xg, yg]))
    scl = np.array([rt[:, 0].std() + 1e-6, rt[:, 1].std() + 1e-6])
    Xtr = rt / scl
    Xev = rot(ev_xy) / scl
    xs, ys_ = Xtr[:, 0], Xtr[:, 1]
    wr = np.sqrt(1.0 / (np.hypot(xs, ys_) + 0.1))
    def Bm(Z):
        x_, y_ = Z[:, 0], Z[:, 1]
        return np.column_stack([np.ones_like(x_), x_, y_, x_ * x_, y_ * y_, x_ * y_])
    B = Bm(Xtr)
    Bw = B * wr[:, None]
    coef, _, _, _ = np.linalg.lstsq(Bw, a * wr, rcond=None)
    y_fit = a - B @ coef
    mev, sev = _np_gp(Xtr, y_fit, Xev)
    return mev + Bm(Xev) @ coef, sev

def densequad_all_rows(P, cx, cy, ev_xy):
    """Дэнс-квад weighted-lstsq вариант (k=10, verbatim _DenseQuadImputer99.impute_well)."""
    xg, yg, a = P[:, 0], P[:, 1], P[:, 2]
    sx = xg.std() + 1e-6; sy = yg.std() + 1e-6
    xs = (xg - cx) / sx; ys_ = (yg - cy) / sy
    wr = np.sqrt(1.0 / (np.hypot(xs, ys_) + 0.1))
    G = np.column_stack([np.ones_like(xs), xs, ys_, xs * xs, ys_ * ys_, xs * ys_])
    Gw = G * wr[:, None]
    coef, _, _, _ = np.linalg.lstsq(Gw, a * wr, rcond=None)
    resid_std = float(np.sqrt(np.mean((G @ coef - a) ** 2)))
    Gi = np.linalg.inv(Gw.T @ Gw + 1e-9 * np.eye(6))
    ex = (ev_xy[:, 0] - cx) / sx; ey = (ev_xy[:, 1] - cy) / sy
    gq = np.column_stack([np.ones_like(ex), ex, ey, ex * ex, ey * ey, ex * ey])
    lev = np.sqrt(np.maximum((gq @ Gi * gq).sum(1), 0.0))
    return gq @ coef, resid_std * np.sqrt(1.0 + lev)

_POOL = None

def _neighbor_pool():
    """773 train-скважины -> {samp[wid]=[spw,3] (X,Y,ANCC), wids, cxy (медианный центр), scale,
    tree (KDTree по cxy/scale)}. Verbatim sampling исходного дамп-скрипта (spw=60 через
    np.linspace по dropna'd рядам, центр = медиана X,Y). Кэшируется — сборка (773 CSV чтения)
    не бесплатна, не пересобирать на каждую скважину."""
    global _POOL
    if _POOL is not None:
        return _POOL
    samp, wids, cxs, cys = {}, [], [], []
    for f in sorted(glob.glob(f"{data.RAW}/train/*__horizontal_well.csv")):
        wid = os.path.basename(f).split("__")[0]
        df = pd.read_csv(f, usecols=["X", "Y", "ANCC"]).dropna()
        if len(df) == 0:
            continue
        ix = np.linspace(0, len(df) - 1, min(SPW, len(df)), dtype=int)
        samp[wid] = df.iloc[ix][["X", "Y", "ANCC"]].to_numpy(np.float64)
        wids.append(wid)
        cxs.append(float(df["X"].median()))
        cys.append(float(df["Y"].median()))
    cxy = np.column_stack([cxs, cys])
    scale = np.where(cxy.std(0) < 1e-3, 1.0, cxy.std(0))
    tree = cKDTree(cxy / scale)
    _POOL = {"samp": samp, "wids": wids, "cxy": cxy, "scale": scale, "tree": tree}
    return _POOL

def _neighbors(wid, well_c, lowo):
    pool = _neighbor_pool()
    _, idx = pool["tree"].query(well_c / pool["scale"], k=min(K_NEI + 6, len(pool["wids"])))
    near = [j for j in np.atleast_1d(idx) if not (lowo and pool["wids"][j] == wid)][:K_NEI]
    return near

def compute_gp(wid: str, xy_all: np.ndarray, lowo: bool) -> np.ndarray:
    """Свежий f_gp по произвольной (X,Y)-траектории xy_all[n,2]. lowo=True исключает wid из
    пула соседей (нужно только когда wid физически есть в train-пуле — верификация/CV)."""
    pool = _neighbor_pool()
    well_c = np.array([float(np.nanmedian(xy_all[:, 0])), float(np.nanmedian(xy_all[:, 1]))])
    near = _neighbors(wid, well_c, lowo)
    if not near:
        return np.full(len(xy_all), np.nan, np.float32)
    P = np.concatenate([pool["samp"][pool["wids"][j]] for j in near], 0)
    f_gp, _ = gpuk_all_rows(P, well_c[0], well_c[1], xy_all.astype(np.float64))
    return np.asarray(f_gp, np.float32)

_TRAIN_CACHE = None

def _train_parquet_cache():
    global _TRAIN_CACHE
    if _TRAIN_CACHE is not None:
        return _TRAIN_CACHE
    df = pd.read_parquet(_PARQUET_PATH, columns=["row_id", "f_gp"])
    parts = df["row_id"].str.rsplit("_", n=1, expand=True)
    wid_col, i_col = parts[0].to_numpy(), parts[1].to_numpy(np.int64)
    cache = {}
    order = np.argsort(wid_col, kind="stable")
    wid_sorted, i_sorted, f_sorted = wid_col[order], i_col[order], df["f_gp"].to_numpy()[order]
    boundaries = np.flatnonzero(np.r_[True, wid_sorted[1:] != wid_sorted[:-1], True])
    for b0, b1 in zip(boundaries[:-1], boundaries[1:]):
        w = wid_sorted[b0]
        n = int(i_sorted[b0:b1].max()) + 1
        arr = np.full(n, np.nan, np.float32)
        arr[i_sorted[b0:b1]] = f_sorted[b0:b1]
        cache[w] = arr
    _TRAIN_CACHE = cache
    return cache

def get_form_pred(wid: str, split: str = "train") -> np.ndarray:
    """f_gp по всем строкам wid__horizontal_well.csv (та же длина, что h["GR"] и т.п. в
    render.load_well_data). train -> прекомпьюченный LOWO parquet (0 стоимости, лик-safe для
    ЛЮБОГО CV-фолда — self уже исключён при дампе). иначе -> свежий расчёт (см. compute_gp)."""
    if split == "train":
        cache = _train_parquet_cache()
        if wid in cache:
            return cache[wid]
    xy_all = pd.read_csv(f"{data.RAW}/{split}/{wid}__horizontal_well.csv",
                          usecols=["X", "Y"]).to_numpy(np.float64)
    return compute_gp(wid, xy_all, lowo=(split == "train"))
