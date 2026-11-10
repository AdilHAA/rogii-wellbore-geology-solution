"""Beam search по GR-мисфиту (порт ядра из паблик SCA-U2net кернела, LB~8.8).

Глобально-когерентная траектория-гипотеза: DP-пучок по строкам typewell-сетки
(шаг ±2 строки/фут MD), цена = (GR−twGR)²/es + mc·|шаг|. Ключевое свойство —
КОММИТИТСЯ в длинные подъёмы (наш главный residual в SDF-UNet).
Используется как приор-канал входа (render_sw: ROGII_SW_BEAM).
"""
import numpy as np
import pandas as pd
from numba import njit

BEAM_CONS = (10, 10.0, 144.0, 2)
BEAM_SM5  = (10,  7.0,  90.0, 5)

@njit(cache=True)
def _beam_jit(sgr, tw_gr, si, BS, mc, es):
    n = len(sgr); nt = len(tw_gr); MAX = BS * 6
    bidx = np.zeros(BS, np.int64); bidx[0] = si
    bcost = np.full(BS, 1e30); bcost[0] = 0.0; bn = np.int64(1)
    hI = np.zeros((n, BS), np.int64); hP = np.zeros((n, BS), np.int64)
    cI = np.zeros(MAX, np.int64); cC = np.full(MAX, 1e30); cP = np.zeros(MAX, np.int64)
    for step in range(n):
        gv = sgr[step]; nc = np.int64(0)
        for bi in range(bn):
            idx = bidx[bi]; cost = bcost[bi]
            for d in range(-2, 3):
                ni = idx + d
                if ni < 0 or ni >= nt: continue
                tot = cost + (gv - tw_gr[ni]) ** 2 / es + mc * (d if d >= 0 else -d)
                fnd = np.int64(-1)
                for ci in range(nc):
                    if cI[ci] == ni: fnd = ci; break
                if fnd >= 0:
                    if tot < cC[fnd]: cC[fnd] = tot; cP[fnd] = bi
                else:
                    if nc < MAX: cI[nc] = ni; cC[nc] = tot; cP[nc] = bi; nc += 1
        kept = min(BS, nc)
        for i in range(kept):
            mi = i
            for j in range(i + 1, nc):
                if cC[j] < cC[mi]: mi = j
            if mi != i:
                cI[i], cI[mi] = cI[mi], cI[i]
                cC[i], cC[mi] = cC[mi], cC[i]
                cP[i], cP[mi] = cP[mi], cP[i]
        hI[step, :kept] = cI[:kept]; hP[step, :kept] = cP[:kept]
        bidx[:kept] = cI[:kept]; bcost[:kept] = cC[:kept]; bn = kept
    best = np.int64(0)
    for b in range(1, bn):
        if bcost[b] < bcost[best]: best = b
    path = np.zeros(n, np.int64); b = best
    for s in range(n - 1, -1, -1): path[s] = hI[s, b]; b = hP[s, b]
    return path

@njit(cache=True)
def _beam_jit_all(sgr, tw_gr, si, BS, mc, es):
    """Как _beam_jit, но возвращает ВСЕ выжившие пути пучка [kept, n] + их косты."""
    n = len(sgr); nt = len(tw_gr); MAX = BS * 6
    bidx = np.zeros(BS, np.int64); bidx[0] = si
    bcost = np.full(BS, 1e30); bcost[0] = 0.0; bn = np.int64(1)
    hI = np.zeros((n, BS), np.int64); hP = np.zeros((n, BS), np.int64)
    cI = np.zeros(MAX, np.int64); cC = np.full(MAX, 1e30); cP = np.zeros(MAX, np.int64)
    for step in range(n):
        gv = sgr[step]; nc = np.int64(0)
        for bi in range(bn):
            idx = bidx[bi]; cost = bcost[bi]
            for d in range(-2, 3):
                ni = idx + d
                if ni < 0 or ni >= nt: continue
                tot = cost + (gv - tw_gr[ni]) ** 2 / es + mc * (d if d >= 0 else -d)
                fnd = np.int64(-1)
                for ci in range(nc):
                    if cI[ci] == ni: fnd = ci; break
                if fnd >= 0:
                    if tot < cC[fnd]: cC[fnd] = tot; cP[fnd] = bi
                else:
                    if nc < MAX: cI[nc] = ni; cC[nc] = tot; cP[nc] = bi; nc += 1
        kept = min(BS, nc)
        for i in range(kept):
            mi = i
            for j in range(i + 1, nc):
                if cC[j] < cC[mi]: mi = j
            if mi != i:
                cI[i], cI[mi] = cI[mi], cI[i]
                cC[i], cC[mi] = cC[mi], cC[i]
                cP[i], cP[mi] = cP[mi], cP[i]
        hI[step, :kept] = cI[:kept]; hP[step, :kept] = cP[:kept]
        bidx[:kept] = cI[:kept]; bcost[:kept] = cC[:kept]; bn = kept
    paths = np.zeros((bn, n), np.int64)
    for b0 in range(bn):
        b = b0
        for s in range(n - 1, -1, -1): paths[b0, s] = hI[s, b]; b = hP[s, b]
    return paths, bcost[:bn]

def _smooth(vals, fb, r):
    s = pd.Series(vals, dtype="float32").interpolate(limit_direction="both").fillna(fb)
    return (s.rolling(r * 2 + 1, center=True, min_periods=1).mean() if r > 0 else s).to_numpy(np.float64)

def beam_tvt(gr_h, t_grid, t_gr_r, start_tvt, cfg):
    """Путь бима в TVT (1 значение на 1-ft точку gr_h). t_grid/t_gr_r — наша 1-ft ресемпл-сетка."""
    bs, mc, es, r = cfg
    si = int(np.abs(t_grid - start_tvt).argmin())
    sgr = _smooth(gr_h, float(np.nanmean(t_gr_r)), r)
    path = _beam_jit(sgr, t_gr_r.astype(np.float64), si, bs, float(mc), float(es))
    return t_grid[path].astype(np.float32)

def beam_paths_all(gr_h, t_grid, t_gr_r, start_tvt, cfg):
    """Все выжившие пути пучка в TVT [K, n] + softmax-веса по финальным костам."""
    bs, mc, es, r = cfg
    si = int(np.abs(t_grid - start_tvt).argmin())
    sgr = _smooth(gr_h, float(np.nanmean(t_gr_r)), r)
    paths, costs = _beam_jit_all(sgr, t_gr_r.astype(np.float64), si, bs, float(mc), float(es))
    w = np.exp(-(costs - costs.min()) / max(costs.std(), 1e-6))
    return t_grid[paths].astype(np.float32), (w / w.sum()).astype(np.float32)
