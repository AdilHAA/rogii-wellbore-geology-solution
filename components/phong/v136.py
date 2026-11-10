# ROGII v136 source — Phong / nocturnebflat123.
# CNN-SDF lineage credited in the original: hengck cnn-sdf + medali1992.
# The section markers are used by final_mix.py to extract Oubank definitions.

import os
import gc
import glob
import json
import time
import warnings
import joblib
from pathlib import Path as _Path
import numpy as np
import pandas as pd
import torch
import torch.nn as _tnn
import torch.nn.functional as _F
from scipy.optimize import nnls as _nnls
from sklearn.linear_model import Ridge as _Ridge
from sklearn.model_selection import GroupKFold
from sklearn.metrics import root_mean_squared_error
from scipy.signal import savgol_filter as _sg
from scipy.spatial import cKDTree
from joblib import Parallel, delayed

SMOKE = False
warnings.filterwarnings("ignore")

class CFG:
    dataset_path = _Path("/kaggle/input/competitions/rogii-wellbore-geology-prediction")
    artifacts_path = _Path("/kaggle/input/datasets/ravaghi/wellbore-geology-prediction-artifacts")
    seed = 42
    n_splits = 5
    cv = GroupKFold(n_splits=n_splits)
    metric = root_mean_squared_error

_CAND_DATASET_PATHS = [
    _Path("/kaggle/input/rogii-wellbore-geology-prediction"),
    _Path("/kaggle/input/competitions/rogii-wellbore-geology-prediction"),
    _Path("data/rogii-wellbore-geology-prediction"),
]
for _p in _CAND_DATASET_PATHS:
    if _p.exists() and (_p / "train").exists():
        CFG.dataset_path = _p
        break

print(f"[v86] CFG.dataset_path = {CFG.dataset_path}", flush=True)

SEED = 42
np.random.seed(SEED)
FORMATIONS = ["ANCC", "ASTNU", "ASTNL", "EGFDU", "EGFDL", "BUDA"]
PLANE_K = 10
DENSE_SPW = 60
DENSE_K = 20

BEAMS = [
    (10, 20.0, 144.0, 2, "cons"),
    (10,  8.0,  64.0, 2, "loose"),
    ( 8, 35.0, 220.0, 1, "vcons"),
    (10, 14.0,  90.0, 5, "sm5"),
    (20,  4.0,  36.0, 3, "vloose"),
    (12, 12.0, 100.0, 3, "mid"),
    (15, 25.0, 180.0, 2, "stiff"),
]

PF_N = 600
ANCC_N = 600
PF_MOM = 0.993; PF_VN = 0.005; PF_PN = 0.01
PF_GR_SIG_MIN = 10.; PF_GR_SIG_MAX = 60.; PF_GR_SIG_DEF = 30.
PF_INIT_V_STD = 0.02; PF_INIT_SPR = 0.5; PF_RESAMP = 0.5
PF_ROUGH_P = 0.2; PF_ROUGH_V = 0.003; PF_GR_WIN = 5; PF_GR_WT = 0.3
ANCC_ALPHA = 0.998; ANCC_RN = 0.002; ANCC_PN = 0.005
ANCC_IR = 0.01; ANCC_IS = 0.3; ANCC_RP = 0.1; ANCC_RR = 0.001

DTW_RADII = (20, 50, 100, 200)
DTW_STOCH_K = 12
DTW_STOCH_TEMP = 3.0

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[v86] Running on device: {DEVICE}", flush=True)

from numba import njit

@njit(cache=True)
def _interp1(grid, v, vmin, step):
    i = int((v - vmin) / step)
    if i < 0: return grid[0]
    n = len(grid) - 1
    if i >= n: return grid[n]
    t = (v - vmin) / step - i
    return grid[i] * (1. - t) + grid[i + 1] * t

@njit(cache=True)
def _resamp(pos, aux, w, N, rp, rv):
    cum = np.zeros(N + 1)
    for j in range(N): cum[j + 1] = cum[j] + w[j]
    u0 = np.random.uniform(0., 1. / N)
    np2 = np.empty(N); na = np.empty(N); ci = 0
    for j in range(N):
        u = u0 + j / N
        while ci < N - 1 and cum[ci + 1] < u: ci += 1
        np2[j] = pos[ci] + rp * np.random.randn()
        na[j] = aux[ci] + rv * np.random.randn()
    return np2, na

@njit(cache=True)
def _beam_jit(sgr, tw_gr, si, BS, mc, es):
    n = len(sgr); nt = len(tw_gr); MAX = BS * 6
    bidx = np.zeros(BS, np.int64); bidx[0] = si
    bcost = np.full(BS, 1e30); bcost[0] = 0.; bn = np.int64(1)
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
def _dtw_sakoe_chiba(query, ref, radius):
    N = len(query); M = len(ref)
    INF = 1e18
    D = np.full((N, M), INF)
    slope = (M - 1.0) / max(N - 1.0, 1.0)
    for i in range(N):
        j_center = int(round(i * slope))
        j_lo = max(0, j_center - radius)
        j_hi = min(M - 1, j_center + radius)
        for j in range(j_lo, j_hi + 1):
            cost = (query[i] - ref[j]) ** 2
            if i == 0 and j == 0:
                D[i, j] = cost
            elif i == 0:
                prev = D[i, j - 1]
                D[i, j] = cost + (prev if prev < INF else INF)
            elif j == 0:
                prev = D[i - 1, j]
                D[i, j] = cost + (prev if prev < INF else INF)
            else:
                a = D[i - 1, j - 1]
                b = D[i - 1, j]
                c = D[i, j - 1]
                mn = a if a < b else b
                mn = mn if mn < c else c
                D[i, j] = cost + (mn if mn < INF else INF)
    i = N - 1; j = M - 1
    pi = np.zeros(N + M, np.int64)
    pj = np.zeros(N + M, np.int64)
    k = 0
    while i > 0 or j > 0:
        pi[k] = i; pj[k] = j; k += 1
        if i == 0:
            j -= 1
        elif j == 0:
            i -= 1
        else:
            a = D[i - 1, j - 1]; b = D[i - 1, j]; c = D[i, j - 1]
            if a <= b and a <= c:
                i -= 1; j -= 1
            elif b <= c:
                i -= 1
            else:
                j -= 1
    pi[k] = 0; pj[k] = 0; k += 1
    return D, pi[:k], pj[:k]

@njit(cache=True)
def _dtw_path_to_tvt(pi, pj, tw_tvt, N):
    j_for_i = np.zeros(N, np.int64)
    for k in range(len(pi)):
        j_for_i[pi[k]] = pj[k]
    result = np.empty(N, np.float32)
    for i in range(N):
        result[i] = tw_tvt[j_for_i[i]]
    return result

@njit(cache=True)
def _dtw_path_slope(pi, pj, N, smooth_win=5):
    j_for_i = np.zeros(N, np.float64)
    for k in range(len(pi)):
        j_for_i[pi[k]] = float(pj[k])
    slope = np.zeros(N, np.float32)
    hw = smooth_win // 2
    for i in range(N):
        i0 = max(0, i - hw); i1 = min(N - 1, i + hw)
        if i1 > i0:
            slope[i] = float((j_for_i[i1] - j_for_i[i0]) / (i1 - i0))
        else:
            slope[i] = 1.0
    return slope

@njit(cache=True)
def _dtw_stochastic_realizations(query, ref, radius, K, temperature):
    N = len(query); M = len(ref)
    INF = 1e18
    slope = (M - 1.0) / max(N - 1.0, 1.0)
    D_base = np.full((N, M), INF)
    for i in range(N):
        j_center = int(round(i * slope))
        j_lo = max(0, j_center - radius)
        j_hi = min(M - 1, j_center + radius)
        for j in range(j_lo, j_hi + 1):
            D_base[i, j] = (query[i] - ref[j]) ** 2
    paths = np.zeros((K, N), np.int64)
    for k in range(K):
        D = np.full((N, M), INF)
        for i in range(N):
            j_center = int(round(i * slope))
            j_lo = max(0, j_center - radius)
            j_hi = min(M - 1, j_center + radius)
            for j in range(j_lo, j_hi + 1):
                noise = -temperature * np.log(-np.log(np.random.uniform(1e-10, 1.0)))
                cost = D_base[i, j] + noise
                if i == 0 and j == 0:
                    D[i, j] = cost
                elif i == 0:
                    prev = D[i, j - 1]
                    D[i, j] = cost + (prev if prev < INF else INF)
                elif j == 0:
                    prev = D[i - 1, j]
                    D[i, j] = cost + (prev if prev < INF else INF)
                else:
                    a = D[i - 1, j - 1]; b = D[i - 1, j]; c = D[i, j - 1]
                    mn = a if a < b else b
                    mn = mn if mn < c else c
                    D[i, j] = cost + (mn if mn < INF else INF)
        i = N - 1; j = M - 1
        j_for_i = np.zeros(N, np.int64)
        while i > 0 or j > 0:
            j_for_i[i] = j
            if i == 0:
                j -= 1
            elif j == 0:
                i -= 1
            else:
                a = D[i - 1, j - 1]; b = D[i - 1, j]; c = D[i, j - 1]
                if a <= b and a <= c:
                    i -= 1; j -= 1
                elif b <= c:
                    i -= 1
                else:
                    j -= 1
        j_for_i[0] = j
        paths[k] = j_for_i
    return paths

@njit(cache=True)
def _pf_ancc(md_v, z_v, gr_v, gg, vmin, step, gs, ls, ir, N,
             ALPHA, RN, PN, IS, RP, RR, RESAMP):
    pos = np.empty(N); rate = np.empty(N); w = np.ones(N) / N
    for j in range(N):
        pos[j] = ls + IS * np.random.randn()
        rate[j] = ir + 0.01 * np.random.randn()
    pts = np.empty(len(md_v)); std_ = np.empty(len(md_v)); pm = md_v[0] - 1.
    for i in range(len(md_v)):
        dm = md_v[i] - pm; dm = max(dm, 1.)
        for j in range(N):
            rate[j] = ALPHA * rate[j] + RN * np.random.randn()
            pos[j] += rate[j] * dm + PN * np.random.randn()
            tvt_j = pos[j] - z_v[i]
            tvt_j = max(tvt_j, vmin - 50.); tvt_j = min(tvt_j, vmin + len(gg) * step + 50.)
            pos[j] = tvt_j + z_v[i]
        if not np.isnan(gr_v[i]):
            ws = 0.
            for j in range(N):
                eg = _interp1(gg, pos[j] - z_v[i], vmin, step)
                d = (gr_v[i] - eg) / gs
                lk = max(np.exp(-0.5 * d * d) if d * d < 600. else 0., 1e-300)
                w[j] *= lk; ws += w[j]
            if ws > 0.:
                for j in range(N): w[j] /= ws
            else:
                for j in range(N): w[j] = 1. / N
        ne = 0.
        for j in range(N): ne += w[j] * w[j]
        if 1. / ne < RESAMP * N:
            pos, rate = _resamp(pos, rate, w, N, RP, RR)
            for j in range(N): w[j] = 1. / N
        tv = 0.
        for j in range(N): tv += w[j] * (pos[j] - z_v[i])
        pts[i] = tv; va = 0.
        for j in range(N): va += w[j] * (pos[j] - z_v[i] - tv) ** 2
        std_[i] = va ** 0.5; pm = md_v[i]
    return pts, std_

@njit(cache=True)
def _pf_z(md_v, z_v, gr_v, gr_sm_v, gg_p, gg_s, vmin, step,
          gs, ip, iv, beta, icpt, zsig, N,
          MOM, VN, PN, GR_WT, RP, RV, RESAMP):
    pos = np.empty(N); vel = np.empty(N); w = np.ones(N) / N
    for j in range(N):
        pos[j] = ip + 0.5 * np.random.randn()
        vel[j] = iv + 0.02 * np.random.randn()
    pts = np.empty(len(md_v)); std_ = np.empty(len(md_v)); pm = md_v[0] - 1.; pz = z_v[0] - 1.
    for i in range(len(md_v)):
        dm = md_v[i] - pm; dm = max(dm, 1.)
        dzd = (z_v[i] - pz) / dm; ve = beta * dzd + icpt
        for j in range(N):
            vel[j] = MOM * vel[j] + VN * np.random.randn()
            pos[j] += vel[j] * dm + PN * np.random.randn()
            pos[j] = max(pos[j], vmin - 50.); pos[j] = min(pos[j], vmin + len(gg_p) * step + 50.)
        if not np.isnan(gr_v[i]):
            ws = 0.
            for j in range(N):
                ep = _interp1(gg_p, pos[j], vmin, step)
                dp = (gr_v[i] - ep) / gs
                lp = max(np.exp(-0.5 * dp * dp) if dp * dp < 600. else 0., 1e-300)
                if not np.isnan(gr_sm_v[i]):
                    es = _interp1(gg_s, pos[j], vmin, step)
                    ds = (gr_sm_v[i] - es) / (gs * 1.5)
                    ls = max(np.exp(-0.5 * ds * ds) if ds * ds < 600. else 0., 1e-300)
                    lk = (1. - GR_WT) * lp + GR_WT * ls
                else:
                    lk = lp
                lk = max(lk, 1e-300); w[j] *= lk; ws += w[j]
            if ws > 0.:
                for j in range(N): w[j] /= ws
            else:
                for j in range(N): w[j] = 1. / N
        ws2 = 0.
        for j in range(N):
            dv = (vel[j] - ve) / max(zsig * 2., 0.005)
            lz = max(np.exp(-0.5 * dv * dv) if dv * dv < 600. else 0., 1e-300)
            w[j] *= lz; ws2 += w[j]
        if ws2 > 0.:
            for j in range(N): w[j] /= ws2
        else:
            for j in range(N): w[j] = 1. / N
        ne = 0.
        for j in range(N): ne += w[j] * w[j]
        if 1. / ne < RESAMP * N:
            pos, vel = _resamp(pos, vel, w, N, RP, RV)
            for j in range(N): w[j] = 1. / N
        wm = 0.
        for j in range(N): wm += w[j] * pos[j]
        pts[i] = wm; va = 0.
        for j in range(N): va += w[j] * (pos[j] - wm) ** 2
        std_[i] = va ** 0.5; pm = md_v[i]; pz = z_v[i]
    return pts, std_

def _grid(tw_tvt, tw_gr, step=0.2):
    tmin = float(tw_tvt.min()); tmax = float(tw_tvt.max())
    tvt_g = np.arange(tmin, tmax + step, step)
    return np.interp(tvt_g, tw_tvt, tw_gr).astype(np.float64), float(tmin), float(step)

def _gr_sig(hw, tw_tvt, tw_gr):
    kn = hw[hw['TVT_input'].notna() & hw['GR'].notna()]
    if len(kn) < 20: return float(PF_GR_SIG_DEF)
    return float(np.clip(np.std(kn['GR'].values - np.interp(kn['TVT_input'].values, tw_tvt, tw_gr)),
                         PF_GR_SIG_MIN, PF_GR_SIG_MAX))

def _nn(arr, v):
    i = int(np.searchsorted(arr, v, 'left'))
    if i >= len(arr): return len(arr) - 1
    if i > 0 and abs(arr[i - 1] - v) <= abs(arr[i] - v): return i - 1
    return i

def _smooth(vals, fb, r):
    s = pd.Series(vals, dtype='float32').interpolate(limit_direction='both').fillna(fb)
    return (s.rolling(r * 2 + 1, center=True, min_periods=1).mean() if r > 0 else s).to_numpy(np.float32)

def beam_search(gr_h, tw_tvt, tw_gr, start_tvt, bs, mc, es, r):
    si = _nn(tw_tvt, start_tvt)
    sgr = _smooth(gr_h, float(np.nanmean(tw_gr)), r).astype(np.float64)
    path = _beam_jit(sgr, tw_gr.astype(np.float64), si, bs, float(mc), float(es))
    return tw_tvt[path].astype(np.float32)

def run_pf_ancc(hw, tw_tvt, tw_gr, N=ANCC_N):
    gs = _gr_sig(hw, tw_tvt, tw_gr)
    kn = hw[hw['TVT_input'].notna()]; ev = hw[hw['TVT_input'].isna()]
    if len(ev) == 0: return np.array([]), np.array([])
    ls = float(kn['TVT_input'].iloc[-1] + kn['Z'].iloc[-1])
    tail = kn.tail(30); dt = np.diff(tail['TVT_input'].values)
    dz = np.diff(tail['Z'].values); dm = np.diff(tail['MD'].values); m = dm > 0
    ir = float(np.median((dt + dz)[m] / dm[m])) if m.sum() >= 3 else 0.
    gg, gmin, gst = _grid(tw_tvt, tw_gr)
    pts, std = _pf_ancc(ev['MD'].values.astype(np.float64), ev['Z'].values.astype(np.float64),
                        ev['GR'].values.astype(np.float64), gg, gmin, gst,
                        gs, ls, ir, N, ANCC_ALPHA, ANCC_RN, ANCC_PN, ANCC_IS, ANCC_RP, ANCC_RR, PF_RESAMP)
    return pts.astype(np.float32), std.astype(np.float32)

def run_pf_z(hw, tw_tvt, tw_gr, N=PF_N):
    gs = _gr_sig(hw, tw_tvt, tw_gr)
    tw_s = pd.Series(tw_gr).rolling(PF_GR_WIN, center=True, min_periods=1).mean().values.astype(np.float32)
    kna = hw[hw['TVT_input'].notna()]; ev = hw[hw['TVT_input'].isna()]
    if len(ev) == 0: return np.array([]), np.array([])
    dz_k = np.diff(kna['Z'].values); dvt = np.diff(kna['TVT_input'].values)
    dmd_k = np.diff(kna['MD'].values); m2 = dmd_k > 0
    if m2.sum() >= 10:
        vz = dz_k[m2] / dmd_k[m2]; vt = dvt[m2] / dmd_k[m2]
        A = np.column_stack([vz, np.ones_like(vz)]); c, _, _, _ = np.linalg.lstsq(A, vt, rcond=None)
        beta, icpt, zsig = float(c[0]), float(c[1]), max(float(np.std(vt - (c[0] * vz + c[1]))), 0.001)
    else:
        beta, icpt, zsig = -1., 0., 0.1
    t2 = kna.tail(20); dvt2 = np.diff(t2['TVT_input'].values); dmd2 = np.diff(t2['MD'].values); m3 = dmd2 > 0
    iv = float(np.median(dvt2[m3] / dmd2[m3])) if m3.sum() >= 3 else 0.
    gg, gmin, gst = _grid(tw_tvt, tw_gr)
    gs2, _, _ = _grid(tw_tvt, tw_s)
    gr_sm = hw['GR'].rolling(PF_GR_WIN, center=True, min_periods=1).mean()
    pts, std = _pf_z(ev['MD'].values.astype(np.float64), ev['Z'].values.astype(np.float64),
                     ev['GR'].values.astype(np.float64),
                     gr_sm.loc[ev.index].values.astype(np.float64),
                     gg, gs2, gmin, gst, gs, float(kna['TVT_input'].iloc[-1]), iv,
                     beta, icpt, zsig, N,
                     PF_MOM, PF_VN, PF_PN, PF_GR_WT, PF_ROUGH_P, PF_ROUGH_V, PF_RESAMP)
    return pts.astype(np.float32), std.astype(np.float32)

def run_dtw_multiscale(query_gr, tw_tvt, tw_gr, last_tvt, radii=DTW_RADII):
    N = len(query_gr)
    qn = (query_gr - query_gr.mean()) / (query_gr.std() + 1e-6)
    rn = (tw_gr - tw_gr.mean()) / (tw_gr.std() + 1e-6)
    si = _nn(tw_tvt, last_tvt)
    qn_f = qn.astype(np.float64)
    rn_f = rn.astype(np.float64)
    dtw_tvts = {}; dtw_slopes = {}; dtw_costs = {}
    inv_cost_sum = 0.0
    tvt_stack = []
    for r in radii:
        D, pi, pj = _dtw_sakoe_chiba(qn_f, rn_f, r)
        cost = float(D[len(qn_f) - 1, len(rn_f) - 1]) / max(len(qn_f) + len(rn_f), 1)
        tvt_pred = _dtw_path_to_tvt(pi[::-1], pj[::-1], tw_tvt.astype(np.float32), N)
        slope = _dtw_path_slope(pi[::-1], pj[::-1], N)
        dtw_tvts[r] = tvt_pred
        dtw_slopes[r] = slope
        dtw_costs[r] = cost
        ic = 1.0 / (cost + 1e-6)
        inv_cost_sum += ic
        tvt_stack.append((tvt_pred, ic))
    weights = np.array([ic / inv_cost_sum for _, ic in tvt_stack], dtype=np.float32)
    tvts_mat = np.stack([t for t, _ in tvt_stack], axis=1)
    dtw_ens = (tvts_mat * weights[None, :]).sum(axis=1).astype(np.float32)
    return dtw_tvts, dtw_slopes, dtw_costs, dtw_ens

def run_dtw_stochastic(query_gr, tw_tvt, tw_gr, last_tvt,
                       radius=50, K=DTW_STOCH_K, temperature=DTW_STOCH_TEMP):
    N = len(query_gr)
    qn = ((query_gr - query_gr.mean()) / (query_gr.std() + 1e-6)).astype(np.float64)
    rn = ((tw_gr - tw_gr.mean()) / (tw_gr.std() + 1e-6)).astype(np.float64)
    paths = _dtw_stochastic_realizations(qn, rn, radius, K, temperature)
    tvt_realiz = np.empty((K, N), dtype=np.float32)
    for k in range(K):
        for i in range(N):
            tvt_realiz[k, i] = tw_tvt[paths[k, i]]
    mean_tvt = tvt_realiz.mean(axis=0).astype(np.float32)
    std_tvt = tvt_realiz.std(axis=0).astype(np.float32)
    cv_tvt = (std_tvt / (np.abs(mean_tvt) + 1e-6)).astype(np.float32)
    return mean_tvt, std_tvt, cv_tvt

_md = np.linspace(1, 50, 20, np.float64); _z = np.zeros(20, np.float64); _gr = np.full(20, 50., np.float64)
_gg = np.linspace(45, 55, 100, np.float64)
_pf_ancc(_md, _z, _gr, _gg, 45., 0.1, 20., 50., 0., 8, 0.998, 0.002, 0.005, 0.3, 0.1, 0.001, 0.5)
_pf_z(_md, _z, _gr, _gr, _gg, _gg, 45., 0.1, 20., 50., 0., -1., 0., 0.1, 8, 0.993, 0.005, 0.01, 0.3, 0.2, 0.003, 0.5)
_beam_jit(np.random.randn(30), np.random.randn(50), 25, 8, 15., 100.)
_q = np.random.randn(40); _r = np.random.randn(50)
_dtw_sakoe_chiba(_q, _r, 10)
_dtw_stochastic_realizations(_q, _r, 10, 3, 2.0)

def robust_slope(x, y, w=None):
    x = np.asarray(x, float); y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 2 or np.std(x[m]) < 1e-6: return 0.
    return float(np.polyfit(x[m], y[m], 1)[0])

def affine_cal(kgr, tw_at_k, min_pts=20):
    v = np.isfinite(kgr) & np.isfinite(tw_at_k)
    if v.sum() < min_pts or np.std(tw_at_k[v]) < 1e-6:
        return 1., float(np.nanmean(kgr) - np.nanmean(tw_at_k)) if v.any() else 0.
    a, b = np.polyfit(tw_at_k[v], kgr[v], 1); return float(a), float(b)

def seg_b_well(ktvt, kz, form_col):
    bv = ktvt + kz - form_col; n = len(bv)
    b_full = float(np.median(bv))
    b_late = float(np.median(bv[max(0, n - 50):])) if n >= 5 else b_full
    t1, t2 = n // 3, 2 * n // 3
    b_early = float(np.median(bv[:max(1, t1)])) if t1 > 0 else b_full
    b_mid = float(np.median(bv[t1:max(t1 + 1, t2)])) if t2 > t1 else b_full
    w = np.exp(0.02 * np.arange(n)); w /= w.sum()
    b_wls = float(np.dot(w, bv))
    return b_full, b_early, b_mid, b_late, b_wls

def multi_scale_ncc(kgr, ktvt, hgr, hws=(8, 15, 25), stride=3):
    out = []
    for hw in hws:
        win = 2 * hw + 1; nk = len(kgr); nh = len(hgr)
        if nk < win + 1 or nh == 0:
            out.append((np.full(nh, ktvt[-1], np.float32), np.zeros(nh, np.float32))); continue
        kg = pd.Series(kgr).rolling(5, center=True, min_periods=1).mean().values.astype(np.float32)
        hg = pd.Series(hgr).rolling(5, center=True, min_periods=1).mean().values.astype(np.float32)
        sts = np.arange(0, nk - win + 1, stride, dtype=np.int32); M = len(sts)
        if M == 0:
            out.append((np.full(nh, ktvt[-1], np.float32), np.zeros(nh, np.float32))); continue
        C = kg[sts[:, None] + np.arange(win, dtype=np.int32)[None, :]].astype(np.float32)
        Cn = (C - C.mean(1, keepdims=True)) / (C.std(1, keepdims=True) + 1e-6)
        hp = np.pad(hg, hw, mode="edge")
        H = hp[np.arange(nh)[:, None] + np.arange(win)[None, :]].astype(np.float32)
        Hn = (H - H.mean(1, keepdims=True)) / (H.std(1, keepdims=True) + 1e-6)
        ncc = Hn @ Cn.T / win; best = ncc.argmax(1); score = ncc.max(1).astype(np.float32)
        out.append((ktvt[np.clip(sts[best] + hw, 0, nk - 1)].astype(np.float32), score))
    tvts = np.stack([o[0] for o in out], 1); scores = np.stack([o[1] for o in out], 1)
    sw = np.exp(3. * scores); sw /= sw.sum(1, keepdims=True) + 1e-9
    sc_ens = (tvts * sw).sum(1).astype(np.float32)
    return out, sc_ens

class FormationPlaneKNN:
    def __init__(self, well_ids, data_dir):
        rows = []
        for wid in well_ids:
            p = data_dir / f"{wid}__horizontal_well.csv"
            try: df = pd.read_csv(p, usecols=["X", "Y"] + FORMATIONS).dropna()
            except: continue
            if len(df) == 0: continue
            row = {"wid": wid, "x": float(df["X"].median()), "y": float(df["Y"].median())}
            for c in FORMATIONS: row[f"{c}_m"] = float(df[c].median())
            rows.append(row)
        self.df = pd.DataFrame(rows); self.wmap = {w: i for i, w in enumerate(self.df["wid"])}
        xy = self.df[["x", "y"]].to_numpy(); self.scale = np.where(xy.std(0) < 1e-3, 1., xy.std(0))
        self.tree = cKDTree(xy / self.scale)
        self.xa = self.df["x"].to_numpy(); self.ya = self.df["y"].to_numpy()
        self.fa = self.df[[f"{c}_m" for c in FORMATIONS]].to_numpy(np.float64)

    def impute(self, xy_q, self_wid=None, k=PLANE_K):
        q = xy_q / self.scale; nf = min(k + 5, len(self.df))
        dist, idx = self.tree.query(q, k=nf, workers=-1)
        if self_wid in self.wmap: dist = np.where(idx == self.wmap[self_wid], np.inf, dist)
        ord = np.argpartition(dist, min(k - 1, nf - 1), 1)[:, :k]
        dk = np.take_along_axis(dist, ord, 1); ik = np.take_along_axis(idx, ord, 1)
        vk = np.isfinite(dk); w = np.where(vk, 1. / (dk + 1e-3), 0.).astype(np.float64)
        xn = self.xa[ik]; yn = self.ya[ik]; fn = self.fa[ik]; wx = w * xn; wy = w * yn
        A = np.zeros((len(q), 3, 3))
        A[:, 0, 0] = (wx * xn).sum(1); A[:, 0, 1] = (wx * yn).sum(1); A[:, 0, 2] = wx.sum(1)
        A[:, 1, 0] = A[:, 0, 1]; A[:, 1, 1] = (wy * yn).sum(1); A[:, 1, 2] = wy.sum(1)
        A[:, 2, 0] = A[:, 0, 2]; A[:, 2, 1] = A[:, 1, 2]; A[:, 2, 2] = w.sum(1)
        A[:, 0, 0] += 1e-9; A[:, 1, 1] += 1e-9; A[:, 2, 2] += 1e-9
        rhs = np.stack([(wx[:, :, None] * fn).sum(1), (wy[:, :, None] * fn).sum(1),
                        (w[:, :, None] * fn).sum(1)], 1)
        try:
            coef = np.linalg.solve(A, rhs)
        except:
            coef = np.zeros((len(q), 3, 6))
            for r in range(len(q)):
                try: coef[r] = np.linalg.pinv(A[r]) @ rhs[r]
                except: pass
        Xq = xy_q[:, 0]; Yq = xy_q[:, 1]
        pred = (Xq[:, None] * coef[:, 0, :] + Yq[:, None] * coef[:, 1, :] + coef[:, 2, :]).astype(np.float32)
        pred[~vk.any(1)] = self.fa.mean(0)
        return pred, np.where(vk, dk, np.inf).min(1).astype(np.float32)

class DenseANCCImputer:
    def __init__(self, well_ids, data_dir, spw=DENSE_SPW):
        xs, ys, anccs, wids = [], [], [], []
        for wid in well_ids:
            p = data_dir / f"{wid}__horizontal_well.csv"
            try: df = pd.read_csv(p, usecols=["X", "Y", "ANCC"]).dropna()
            except: continue
            if len(df) == 0: continue
            ix = np.linspace(0, len(df) - 1, min(spw, len(df)), dtype=int); s = df.iloc[ix]
            xs.append(s["X"].values); ys.append(s["Y"].values)
            anccs.append(s["ANCC"].values); wids.extend([wid] * len(s))
        self.xy = np.column_stack([np.concatenate(xs), np.concatenate(ys)])
        self.ancc = np.concatenate(anccs).astype(np.float32); self.wids = np.array(wids)
        self.scale = np.where(self.xy.std(0) < 1e-3, 1., self.xy.std(0))
        self.tree = cKDTree(self.xy / self.scale)

    def impute(self, xy_q, self_wid=None, k=DENSE_K, nfetch=5000):
        xy_q = np.atleast_2d(xy_q); q = xy_q / self.scale; nf = min(nfetch, len(self.ancc))
        dist, idx = self.tree.query(q, k=nf, workers=-1)
        if self_wid: dist = np.where(self.wids[idx] == self_wid, np.inf, dist)
        ord = np.argpartition(dist, min(k - 1, nf - 1), 1)[:, :k]
        dk = np.take_along_axis(dist, ord, 1); ik = np.take_along_axis(idx, ord, 1)
        vk = np.isfinite(dk); w = np.where(vk, 1. / (dk + 1e-3), 0.)
        sw = w.sum(1); safe = np.where(sw < 1e-9, 1., sw); an = self.ancc[ik]
        ap = (an * w).sum(1) / safe; ap = np.where(sw < 1e-9, float(self.ancc.mean()), ap)
        var = ((an - ap[:, None]) ** 2 * w).sum(1) / safe
        return ap.astype(np.float32), np.sqrt(np.maximum(var, 0.)).astype(np.float32), \
               np.where(vk, dk, np.inf).min(1).astype(np.float32)

# Locate all files using path searcher helper
import glob
from pathlib import Path

def loc(marker, prefer=None, optional=False):
    hits = sorted(glob.glob(f"/kaggle/input/**/{marker}", recursive=True))
    if not hits:
        hits = sorted(glob.glob(f"**/{marker}", recursive=True))
    if prefer:
        preferred = [h for h in hits if prefer in h.replace("\\", "/")]
        if preferred:
            return Path(preferred[0])
    if hits:
        return Path(hits[0])
    if optional:
        return None
    raise FileNotFoundError(marker)

# Load priors for FE
def _load_basin_gr_priors():
    p = loc("basin_gr_priors_v1.parquet")
    df = pd.read_parquet(p).set_index("GR")
    return list(df.columns), df.to_numpy(np.float32)

def _load_basin_gr_d1_priors():
    p = loc("basin_gr_d1_priors_v1.parquet")
    df = pd.read_parquet(p)
    log_cols = [c for c in df.columns if c.startswith("log_p_")]
    forms = [c[len("log_p_"):] for c in log_cols]
    df = df.sort_values(["GR", "gr_d1"]).reset_index(drop=True)
    n_gr = df["GR"].nunique()
    n_d1 = df["gr_d1"].nunique()
    table = df[log_cols].to_numpy(np.float32).reshape(n_gr, n_d1, len(forms))
    return forms, table

PRIOR_FORMATIONS, PRIOR_LOG_P_TABLE = _load_basin_gr_priors()
PRIOR_GR_MIN, PRIOR_GR_STEP = 0.0, 0.5
PRIOR_GR_NBINS = PRIOR_LOG_P_TABLE.shape[0]
PRIOR2D_FORMATIONS, PRIOR2D_LOG_P_TABLE = _load_basin_gr_d1_priors()
PRIOR2D_GR_MIN, PRIOR2D_GR_STEP = 0.0, 0.5
PRIOR2D_D1_MIN, PRIOR2D_D1_MAX, PRIOR2D_D1_STEP = -30.0, 30.0, 1.0
PRIOR2D_GR_NBINS = PRIOR2D_LOG_P_TABLE.shape[0]
PRIOR2D_D1_NBINS = PRIOR2D_LOG_P_TABLE.shape[1]

def build_well(hw_path, tw_path, is_train):
    wid = Path(hw_path).stem.replace("__horizontal_well", "")
    try:
        hw = pd.read_csv(hw_path); tw = pd.read_csv(tw_path).sort_values("TVT")
    except:
        return None
    if is_train and "TVT" not in hw.columns: return None
    kn = hw[hw["TVT_input"].notna()]; ev = hw[hw["TVT_input"].isna()]
    if len(ev) == 0 or len(kn) < 10: return None
    if is_train and hw["TVT"].isna().all(): return None
    tw_tvt = tw["TVT"].to_numpy(np.float32); tw_gr = tw["GR"].to_numpy(np.float32)
    if len(tw_tvt) < 3: return None

    pf_a, std_a = run_pf_ancc(hw, tw_tvt, tw_gr)
    if len(pf_a) == 0: return None
    pf_z, std_z = run_pf_z(hw, tw_tvt, tw_gr)
    pf_use = pf_a.astype(np.float32); std_use = std_a.astype(np.float32)
    has_z = len(pf_z) == len(pf_a) and not np.any(np.isnan(pf_z))

    lk = kn.iloc[-1]; last_tvt = float(lk["TVT_input"])
    gr_full = hw["GR"].astype(float).interpolate(limit_direction="both").fillna(float(np.nanmean(tw_gr)))
    hgr = gr_full.iloc[ev.index[0]:].to_numpy(np.float32)
    kgr = gr_full.iloc[:len(kn)].to_numpy(np.float32)

    bpaths = {}
    for (bs, mc, es, r, tag) in BEAMS:
        bpaths[tag] = beam_search(hgr, tw_tvt, tw_gr, last_tvt, bs, mc, es, r)
    beam_ref = (bpaths["cons"] + bpaths["sm5"]) / 2.

    ktvt = kn["TVT_input"].to_numpy(np.float32)
    sc_res, sc_ens = multi_scale_ncc(kgr, ktvt, hgr, hws=(8, 15, 25), stride=3)
    sc8, sc8s = sc_res[0]; sc15, sc15s = sc_res[1]; sc25, sc25s = sc_res[2]
    sc_cons = (sc8 + sc15 + sc25) / 3.
    sc_trust = float(np.clip(len(kn) / 200., 0., 0.6))
    hyb_ref = (1 - sc_trust) * beam_ref + sc_trust * sc_ens

    full_gr = gr_full.values.astype(np.float32)
    dtw_tvts_ms, dtw_slopes_ms, dtw_costs_ms, dtw_ens_ms = run_dtw_multiscale(
        full_gr, tw_tvt, tw_gr, last_tvt, radii=DTW_RADII
    )

    dtw_mean_stoch, dtw_std_stoch, dtw_cv_stoch = run_dtw_stochastic(
        full_gr, tw_tvt, tw_gr, last_tvt, radius=50, K=DTW_STOCH_K, temperature=DTW_STOCH_TEMP
    )

    nh = len(ev)
    ev_start = ev.index[0]

    def _ev_slice(arr):
        return arr[ev_start:ev_start + nh].astype(np.float32)

    dtw_ens_ev = _ev_slice(dtw_ens_ms)
    dtw_mean_ev = _ev_slice(dtw_mean_stoch)
    dtw_std_ev = _ev_slice(dtw_std_stoch)
    dtw_cv_ev = _ev_slice(dtw_cv_stoch)

    dtw_per_radius_ev = {}
    dtw_slope_ev = {}
    for r in DTW_RADII:
        dtw_per_radius_ev[r] = _ev_slice(dtw_tvts_ms[r])
        dtw_slope_ev[r] = _ev_slice(dtw_slopes_ms[r])

    dtw_slope_mean_ev = np.stack([dtw_slope_ev[r] for r in DTW_RADII], 1).mean(1).astype(np.float32)
    dtw_ens_slope_ev = np.stack([dtw_slope_ev[r] for r in DTW_RADII], 1).mean(1).astype(np.float32)

    dtw_cost_arr = np.array([dtw_costs_ms[r] for r in DTW_RADII], dtype=np.float32)
    dtw_cost_min = float(dtw_cost_arr.min())
    dtw_cost_range = float(dtw_cost_arr.max() - dtw_cost_arr.min())

    tw_at_k = np.interp(ktvt, tw_tvt, tw_gr).astype(np.float32)
    a_cal, b_cal = affine_cal(kgr, tw_at_k)
    kmd = kn["MD"].to_numpy(np.float32); kz = kn["Z"].to_numpy(np.float32)
    pfx_rmse = float(np.sqrt(np.mean((kgr - tw_at_k) ** 2)))
    slp_all = robust_slope(kmd, ktvt); slp_50 = robust_slope(kmd[-50:], ktvt[-50:])
    slp_z = robust_slope(kz, ktvt)

    swid = wid if is_train else None
    xy_ev = ev[["X", "Y"]].to_numpy(np.float64); xy_kn = kn[["X", "Y"]].to_numpy(np.float64)
    form_ev, knn_d = _FI.impute(xy_ev, self_wid=swid)
    form_kn, _ = _FI.impute(xy_kn, self_wid=swid)
    z_kn = kn["Z"].to_numpy(np.float32); z_ev = ev["Z"].to_numpy(np.float32)

    tvt_fs = {}; form_rmse = {}; form_list = []
    for fi2, fn in enumerate(FORMATIONS):
        b_full, b_early, b_mid, b_late, b_wls = seg_b_well(ktvt, z_kn, form_kn[:, fi2])
        tvt_f = (-z_ev + form_ev[:, fi2] + b_full).astype(np.float32)
        tvt_fw = (-z_ev + form_ev[:, fi2] + b_wls).astype(np.float32)
        tvt_f50 = (-z_ev + form_ev[:, fi2] + b_late).astype(np.float32)
        tvt_fs[f"tvtF_{fn}"] = tvt_f; tvt_fs[f"tvtFw_{fn}"] = tvt_fw
        tvt_fs[f"tvtF50_{fn}"] = tvt_f50
        tvt_fs[f"bw_{fn}"] = np.float32(b_full); tvt_fs[f"bww_{fn}"] = np.float32(b_wls)
        tvt_fs[f"bw50_{fn}"] = np.float32(b_late)
        tvt_fs[f"bw_early_{fn}"] = np.float32(b_early)
        tvt_fs[f"bw_mid_{fn}"] = np.float32(b_mid)
        form_rmse[fn] = float(np.sqrt(np.mean((ktvt - (-z_kn + form_kn[:, fi2] + b_full)) ** 2)))
        form_list.append(tvt_f)

    fs = np.stack(form_list, 1)
    form_mean_d = (fs.mean(1) - last_tvt).astype(np.float32)
    form_std_d = fs.std(1).astype(np.float32)
    form_rng_d = (fs.max(1) - fs.min(1)).astype(np.float32)

    d_ancc, d_std, d_dist = _DI.impute(xy_ev, self_wid=swid)
    d_kn, d_std_kn, _ = _DI.impute(xy_kn, self_wid=swid)
    b_vd = ktvt + z_kn - d_kn
    _, b_de, b_dm, b_dl, b_dw = seg_b_well(ktvt, z_kn, d_kn)
    b_d = float(np.median(b_vd))
    tvt_dense = (-z_ev + d_ancc + b_d).astype(np.float32)
    tvt_densew = (-z_ev + d_ancc + b_dw).astype(np.float32)
    tvt_dense50 = (-z_ev + d_ancc + b_dl).astype(np.float32)
    res_kn = ktvt + z_kn - d_kn
    d_rmse = float(np.sqrt(np.mean(res_kn ** 2)))
    d_bias = float(np.mean(res_kn)); d_nb_std = float(np.mean(d_std_kn))

    all_sigs = [pf_use] + [p for p in bpaths.values()] + \
               [sc8, sc15, sc25, sc_ens, tvt_fs["tvtF_ANCC"], tvt_dense, dtw_ens_ev]
    sig_mat = np.stack(all_sigs, 1)
    sig_std = sig_mat.std(1).astype(np.float32)
    sig_mean = (sig_mat.mean(1) - last_tvt).astype(np.float32)

    gr_s = pd.Series(gr_full.values); rolls = {}
    for w in [5, 21, 51, 101]:
        r = gr_s.rolling(w, center=True, min_periods=1)
        rolls[f"grm{w}"] = r.mean().iloc[ev.index].values.astype(np.float32)
        rolls[f"grs{w}"] = r.std().fillna(0).iloc[ev.index].values.astype(np.float32)
    for lag in [1, 5, 15, 30]:
        rolls[f"glag{lag}"] = gr_s.shift(lag).bfill().iloc[ev.index].values.astype(np.float32)
        rolls[f"glead{lag}"] = gr_s.shift(-lag).ffill().iloc[ev.index].values.astype(np.float32)
    gr_d1 = gr_s.diff().fillna(0.).iloc[ev.index].values.astype(np.float32)
    gr_d2 = gr_s.diff().diff().fillna(0.).iloc[ev.index].values.astype(np.float32)
    gr_env = gr_s.rolling(21, center=True, min_periods=1).max().iloc[ev.index].values.astype(np.float32)
    gr_nrg = np.sqrt(np.maximum((gr_s ** 2).rolling(21, center=True, min_periods=1).mean(), 0.)
                     ).iloc[ev.index].values.astype(np.float32)

    hmd = ev["MD"].to_numpy(np.float32); md_since = hmd - float(lk["MD"])
    slp_b_all = (last_tvt + slp_all * md_since).astype(np.float32)
    slp_b_50 = (last_tvt + slp_50 * md_since).astype(np.float32)

    mdd = hw["MD"].diff().replace(0, np.nan)
    dzdmd = (hw["Z"].diff() / mdd).iloc[ev.index].values.astype(np.float32)
    dxdmd = (hw["X"].diff() / mdd).iloc[ev.index].values.astype(np.float32)
    dydmd = (hw["Y"].diff() / mdd).iloc[ev.index].values.astype(np.float32)

    frac = (np.arange(nh) / max(nh - 1, 1)).astype(np.float32)

    def sc(v): return np.full(nh, np.float32(v), np.float32)

    feats = {
        "well": wid, "id": [f"{wid}_{i}" for i in ev.index],
        "last_known_tvt": sc(last_tvt),
        "pf_ancc": pf_use, "pf_ancc_std": std_use,
        "pf_ancc_delta": (pf_use - last_tvt).astype(np.float32),
        "pf_z": (pf_z.astype(np.float32) if has_z else sc(last_tvt)),
        "pf_z_delta": ((pf_z - last_tvt).astype(np.float32) if has_z else sc(0.)),
        "pf_vs_z": ((pf_use - pf_z.astype(np.float32)) if has_z else sc(0.)),
        **{f"beam_{t}_d": (p - np.float32(last_tvt)).astype(np.float32) for t, p in bpaths.items()},
        "beam_mean_d": np.stack([(p - last_tvt) for p in bpaths.values()], 1).mean(1).astype(np.float32),
        "beam_std_d": np.stack([(p - last_tvt) for p in bpaths.values()], 1).std(1).astype(np.float32),
        "beam_med_d": np.median(np.stack([(p - last_tvt) for p in bpaths.values()], 1), 1).astype(np.float32),
        "sc8_d": (sc8 - np.float32(last_tvt)).astype(np.float32), "sc8_sc": sc8s,
        "sc15_d": (sc15 - np.float32(last_tvt)).astype(np.float32), "sc15_sc": sc15s,
        "sc25_d": (sc25 - np.float32(last_tvt)).astype(np.float32), "sc25_sc": sc25s,
        "sc_cons_d": (sc_cons - np.float32(last_tvt)).astype(np.float32),
        "sc_ens_d": (sc_ens - np.float32(last_tvt)).astype(np.float32),
        "sc_trust": sc(sc_trust), "hyb_d": (hyb_ref - np.float32(last_tvt)).astype(np.float32),
        "sig_std": sig_std, "sig_mean_d": sig_mean,
        **tvt_fs,
        **{f"frm_rmse_{fn}": sc(form_rmse[fn]) for fn in FORMATIONS},
        "form_mean_d": form_mean_d, "form_std_d": form_std_d, "form_rng_d": form_rng_d,
        "spatial_ancc_d": (form_ev[:, 0] - np.float32(np.interp(last_tvt, tw_tvt, tw_gr))),
        "spatial_knn_dist": knn_d,
        "dense_ancc": d_ancc, "dense_std": d_std, "dense_dist": d_dist,
        "tvt_dense_d": (tvt_dense - last_tvt).astype(np.float32),
        "tvt_densew_d": (tvt_densew - last_tvt).astype(np.float32),
        "tvt_dense50_d": (tvt_dense50 - last_tvt).astype(np.float32),
        "dense_rmse": sc(d_rmse), "dense_bias": sc(d_bias), "dense_nb_std": sc(d_nb_std),
        "pf_vs_spatial": (pf_use - tvt_fs["tvtF_ANCC"]).astype(np.float32),
        "pf_vs_dense": (pf_use - tvt_dense).astype(np.float32),
        "spatial_vs_dense": (tvt_fs["tvtF_ANCC"] - tvt_dense).astype(np.float32),
        "beam_vs_spatial": (bpaths["cons"] - tvt_fs["tvtF_ANCC"]).astype(np.float32),
        "sc_vs_beam": (sc_ens - bpaths["cons"]).astype(np.float32),
        "dtw_ens_d": (dtw_ens_ev - last_tvt).astype(np.float32),
        "dtw_stoch_mean_d": (dtw_mean_ev - last_tvt).astype(np.float32),
        "dtw_stoch_std": dtw_std_ev,
        "dtw_stoch_cv": dtw_cv_ev,
        "dtw_slope_mean": dtw_slope_mean_ev,
        **{f"dtw_r{r}_d": (dtw_per_radius_ev[r] - last_tvt).astype(np.float32) for r in DTW_RADII},
        **{f"dtw_slope_r{r}": dtw_slope_ev[r] for r in DTW_RADII},
        "dtw_cost_min": sc(dtw_cost_min),
        "dtw_cost_range": sc(dtw_cost_range),
        "dtw_vs_beam": (dtw_ens_ev - bpaths["cons"]).astype(np.float32),
        "dtw_vs_pf": (dtw_ens_ev - pf_use).astype(np.float32),
        "dtw_vs_sc": (dtw_ens_ev - sc_ens).astype(np.float32),
        **{f"tddtw{int(o)}": hgr - np.interp(dtw_ens_ev + o, tw_tvt, tw_gr).astype(np.float32)
           for o in DTW_OFFS},
        "cal_a": sc(a_cal), "cal_b": sc(b_cal),
        "pfx_rmse": sc(pfx_rmse), "known_len": sc(len(kn)), "eval_len": sc(nh),
        "slp_all": sc(slp_all), "slp_50": sc(slp_50), "slp_z": sc(slp_z),
        "slp_b_d_all": (slp_b_all - last_tvt).astype(np.float32),
        "slp_b_d_50": (slp_b_50 - last_tvt).astype(np.float32),
        "ktvt_range": sc(float(np.ptp(ktvt))), "ktvt_std": sc(float(ktvt.std())),
        "md_since": md_since, "frac": frac, "frac2": frac ** 2, "sqrt_frac": np.sqrt(frac),
        "z": z_ev,
        "dx": (ev["X"] - float(lk["X"])).to_numpy(np.float32),
        "dy": (ev["Y"] - float(lk["Y"])).to_numpy(np.float32),
        "dz": (z_ev - float(lk["Z"])).astype(np.float32),
        "dxy": np.sqrt((ev["X"] - float(lk["X"])) ** 2 + (ev["Y"] - float(lk["Y"])) ** 2).to_numpy(np.float32),
        "dzdmd": dzdmd, "dxdmd": dxdmd, "dydmd": dydmd,
        "gr": hgr, "gr_d1": gr_d1, "gr_d2": gr_d2, "gr_env": gr_env, "gr_nrg": gr_nrg,
        "gr_vs_tw_anc": hgr - np.float32(np.interp(last_tvt, tw_tvt, tw_gr)),
        "gr_vs_slp_all": hgr - np.interp(slp_b_all, tw_tvt, tw_gr).astype(np.float32),
        **{f"tda{int(o)}": hgr - np.float32(np.interp(last_tvt + o, tw_tvt, tw_gr)) for o in ANCH_OFFS},
        **{f"tdbc{int(o)}": hgr - np.interp(beam_ref + o, tw_tvt, tw_gr).astype(np.float32) for o in BEAM_OFFS},
        **{f"tdsc{int(o)}": hgr - np.interp(sc_ens + o, tw_tvt, tw_gr).astype(np.float32) for o in SC_OFFS},
        **{f"tdpf{int(o)}": hgr - np.interp(pf_use + o, tw_tvt, tw_gr).astype(np.float32) for o in PF_OFFS},
        "tw_range": sc(float(np.ptp(tw_tvt))), "tw_gr_mean": sc(float(tw_gr.mean())),
    }
    for k, v in rolls.items(): feats[k] = v

    if PRIOR_LOG_P_TABLE.shape[1] > 0:
        _gr_clip = np.clip(hgr.astype(np.float32), 0.0, 250.0)
        _gr_idx = np.clip(
            np.round((_gr_clip - PRIOR_GR_MIN) / PRIOR_GR_STEP).astype(np.int64),
            0, PRIOR_GR_NBINS - 1,
        )
        _ll = PRIOR_LOG_P_TABLE[_gr_idx]
        for _i, _fm in enumerate(PRIOR_FORMATIONS):
            feats[f"gr_form_loglik_{_fm}"] = _ll[:, _i].astype(np.float32)

    if PRIOR2D_LOG_P_TABLE.shape[2] > 0:
        _gr_clip2 = np.clip(hgr.astype(np.float32), 0.0, 250.0)
        _gr_idx2 = np.clip(
            np.round((_gr_clip2 - PRIOR2D_GR_MIN) / PRIOR2D_GR_STEP).astype(np.int64),
            0, PRIOR2D_GR_NBINS - 1,
        )
        _d1_clip = np.clip(gr_d1.astype(np.float32), PRIOR2D_D1_MIN, PRIOR2D_D1_MAX)
        _d1_idx2 = np.clip(
            np.round((_d1_clip - PRIOR2D_D1_MIN) / PRIOR2D_D1_STEP).astype(np.int64),
            0, PRIOR2D_D1_NBINS - 1,
        )
        _ll2 = PRIOR2D_LOG_P_TABLE[_gr_idx2, _d1_idx2]
        for _i, _fm in enumerate(PRIOR2D_FORMATIONS):
            feats[f"gr2d_form_loglik_{_fm}"] = _ll2[:, _i].astype(np.float32)

    result = pd.DataFrame(feats)
    if is_train:
        if "TVT" not in ev.columns or ev["TVT"].isna().all(): return None
        result["target"] = (ev["TVT"].to_numpy(np.float32) - np.float32(last_tvt))
    return result

ANCH_OFFS = np.array([-80, -40, -20, -10, -5, 0, 5, 10, 20, 40, 80], np.float32)
BEAM_OFFS = np.array([-40, -20, -10, -5, -3, 0, 3, 5, 10, 20, 40], np.float32)
SC_OFFS   = np.array([-30, -15, -8, -4, -2, 0, 2, 4, 8, 15, 30], np.float32)
PF_OFFS   = np.array([-30, -15, -8, -4, -2, 0, 2, 4, 8, 15, 30], np.float32)
DTW_OFFS  = np.array([-20, -10, -5, -2, 0, 2, 5, 10, 20], np.float32)

hw_paths = sorted((CFG.dataset_path / "train").glob("*__horizontal_well.csv"))
train_wids = [p.stem.replace("__horizontal_well", "") for p in hw_paths]
FI = FormationPlaneKNN(train_wids, CFG.dataset_path / "train")
DI = DenseANCCImputer(train_wids, CFG.dataset_path / "train")
_FI = FI; _DI = DI

TRAIN_ORIG = loc("train_original.parquet", prefer="augment-data-template-v38/")
train_cols = ["id", "well", "last_known_tvt", "pf_ancc", "md_since"]
train_df = pd.read_parquet(TRAIN_ORIG, columns=train_cols)
ids = train_df["id"].astype(str)
ids_np = ids.to_numpy()
base = train_df["last_known_tvt"].to_numpy(np.float64)
pf_delta = train_df["pf_ancc"].to_numpy(np.float64) - base
md_since = np.maximum(train_df["md_since"].to_numpy(np.float64), 0.0)
N = len(train_df)

HC_DIR = loc("models/lightgbm-1/oof_preds.pkl", prefer="hill-climb").parents[2]
hcoof = pd.read_parquet(HC_DIR / "oof.parquet", columns=["row_id", "fold", "true"])
rid = hcoof["row_id"].astype(str)
fold = pd.Series(hcoof["fold"].to_numpy(), index=rid).reindex(ids).to_numpy(np.int8)
true = pd.Series(hcoof["true"].to_numpy(), index=rid).reindex(ids).to_numpy(np.float64)
y_d = true - base

oof_preds = {}

def safe_array(x):
    a = np.asarray(x, np.float64)
    return np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)

for name in ["lightgbm-1", "lightgbm-2", "lightgbm-3", "catboost-1", "catboost-2", "catboost-3"]:
    oof_preds[name] = safe_array(pd.Series(joblib.load(HC_DIR / "models" / name / "oof_preds.pkl"), index=rid).reindex(ids).to_numpy())

AUG = loc("models/catboost-1/fold_0.cbm", prefer="augment-cats").parents[2]
cba = pd.read_parquet(AUG / "oof.parquet", columns=["row_id", "pred"])
oof_preds["cb_aug"] = safe_array(pd.Series(cba["pred"].to_numpy(np.float64), index=cba["row_id"].astype(str)).reindex(ids).to_numpy() - base)

NN_ROOT = loc("oof_nn_anchor_full.parquet", prefer="nn-anchor-full").parent
nn_oof = pd.read_parquet(NN_ROOT / "oof_nn_anchor_full.parquet").set_index("row_id").reindex(ids_np)
seed_cols = [c for c in nn_oof.columns if c.startswith("pred_seed_")]
oof_preds["nn"] = safe_array(nn_oof[seed_cols].to_numpy(np.float64).mean(1) - base)

V22_OOF = loc("oof_tw_attn_v22.parquet", prefer="v22-nn")
V35_OOF = loc("oof_tw_attn_v35.parquet", prefer="rogii-v35-nn")
V35V2_OOF = loc("oof_tw_attn_v35_pfv2.parquet", prefer="rogii-v71-v35-pfv2")

for nm, path in [("v22", V22_OOF), ("v35", V35_OOF), ("v35pfv2", V35V2_OOF)]:
    tmp = pd.read_parquet(path, columns=["row_id", "pred"]).set_index("row_id").reindex(ids_np)
    oof_preds[nm] = safe_array(tmp["pred"].to_numpy(np.float64) - base)

PF_OOF = loc("oof_pf_sets123.parquet", prefer="rogii-pf-sets123-oof")
pf = pd.read_parquet(PF_OOF)
pfx = pf["row_id"].astype(str)
for pid in ["P06", "P23", "P51", "P54"]:
    s = pd.Series(pf[f"pf_{pid}"].to_numpy(np.float64), index=pfx).reindex(ids).to_numpy()
    bad = ~np.isfinite(s)
    if bad.any():
        s[bad] = base[bad]
    oof_preds[f"pf_{pid}"] = safe_array(s - base)

PFB_SET1 = loc("oof_v35_pfbank_set1.parquet", prefer="rogii-pfbank-nn-oof")
PFB_SET2 = loc("oof_v35_pfbank_set2.parquet", prefer="rogii-v35-pfbank-set2")
PFB_SET3 = loc("oof_v35_pfbank_set3.parquet", prefer="rogii-v35-pfbank-set3")

def load_pfb(path, tag):
    o = pd.read_parquet(path)
    ox = o["row_id"].astype(str)
    for sd in [42, 1234, 7]:
        col = f"pred_s{sd}"
        if col not in o.columns:
            continue
        s = pd.Series(o[col].to_numpy(np.float64), index=ox).reindex(ids).to_numpy()
        bad = ~np.isfinite(s)
        if bad.any():
            s[bad] = base[bad]
        oof_preds[f"{tag}_s{sd}"] = safe_array(s - base)

load_pfb(PFB_SET1, "nn_pfb_set1")
load_pfb(PFB_SET2, "nn_pfb_set2")
load_pfb(PFB_SET3, "nn_pfb_set3")

PFB_FAKEAUG = loc("oof_set1_fakeaug_3seed.parquet", prefer="rogii-pfbank-fakeaug-3seed")
load_pfb(PFB_FAKEAUG, "nn_pfb_fakeaug")

GRIDB_OOF = loc("oof_grid_b.parquet", prefer="rogii-fakeaug-grid-b")
load_pfb(GRIDB_OOF, "nn_grid_b")

LGBPF_OOF = loc("oof_per_model.parquet", prefer="rogii-v82-lgbpf-sets")
lgbpf = pd.read_parquet(LGBPF_OOF)
lx = lgbpf["row_id"].astype(str)
for col in ["lgbpf_set1", "lgbpf_set2", "lgbpf_set3"]:
    s = pd.Series(lgbpf[col].to_numpy(np.float64), index=lx).reindex(ids).to_numpy()
    bad = ~np.isfinite(s)
    if bad.any():
        s[bad] = base[bad]
    oof_preds[col] = safe_array(s - base)

print(f"[v86] Finished loading OOF predictions for all members.", flush=True)

test_paths = sorted((CFG.dataset_path / "test").glob("*__horizontal_well.csv"))
typewell_paths = sorted((CFG.dataset_path / "test").glob("*__typewell.csv"))
assert len(test_paths) == len(typewell_paths), "Mismatch test horizontal vs typewell"

if SMOKE:
    test_paths = test_paths[:2]
    typewell_paths = typewell_paths[:2]
    print(f"[v86] SMOKE MODE: Subsampled test set to {len(test_paths)} wells.", flush=True)

test_wids = [p.stem.replace("__horizontal_well", "") for p in test_paths]
Nte_wells = len(test_wids)
print(f"[v86] Test set contains {Nte_wells} wells.", flush=True)

test_parts = []
for hw_p, tw_p in zip(test_paths, typewell_paths):
    part = build_well(hw_p, tw_p, is_train=False)
    if part is not None:
        test_parts.append(part)
test_df = pd.concat(test_parts, ignore_index=True) if test_parts else pd.DataFrame(columns=["id", "last_known_tvt", "well", "md_since"])
Nte = len(test_df)
print(f"[v86] Engineered features test size: {test_df.shape}", flush=True)

ids_te = test_df["id"].astype(str).values
base_te = test_df["last_known_tvt"].to_numpy(np.float64)
md_te = np.maximum(test_df["md_since"].to_numpy(np.float64), 0.0)

test_preds = {}

FEAT = list(joblib.load(HC_DIR / "models" / "lightgbm-1" / "models.pkl")[0].feature_name())
X_test = test_df[FEAT].to_numpy(np.float32)

for name in ["lightgbm-1", "lightgbm-2", "lightgbm-3", "catboost-1", "catboost-2", "catboost-3"]:
    models = joblib.load(HC_DIR / "models" / name / "models.pkl")
    t = np.zeros(Nte, dtype=np.float64)
    is_lgb = name.startswith("light")
    for f in range(5):
        t += (models[f].predict(X_test, num_iteration=models[f].best_iteration) if is_lgb
              else models[f].predict(X_test)) / 5
    test_preds[name] = t

from catboost import CatBoostRegressor
cbt = np.zeros(Nte, dtype=np.float64)
ns = 0
for s in [1, 2, 3]:
    folds = [CatBoostRegressor().load_model(str(AUG / f"models/catboost-{s}/fold_{g}.cbm")) for g in range(5)]
    cbt += np.mean([folds[g].predict(X_test) for g in range(5)], axis=0)
    ns += 1
test_preds["cb_aug"] = cbt / ns
del X_test; gc.collect()

ck = torch.load(NN_ROOT / "nn_checkpoints.pt", map_location="cpu", weights_only=False)
checkpoints = ck["checkpoints"]
NN_FEATURES = ck["nn_features"]
fmean = ck["fmean_global"].astype(np.float32)
fstd = ck["fstd_global"].astype(np.float32)
arch = ck["arch"]
norms = ck["norms"]
SEEDS_USED = ck["seeds_used"]
N_SEEDS = ck["n_seeds"]

D_CTX = arch["d_ctx"]
TCN_CH = arch["tcn_channels"]
TCN_DIL = tuple(arch["tcn_dilations"])
TCN_DO = arch["tcn_dropout"]
TW_MAX = arch["tw_max_len"]
TW_F = arch["tw_features"]
TARGET_NORM = norms["TARGET_NORM"]
TVT_NORM = norms["TVT_NORM"]
GR_NORM = norms["GR_NORM"]

class _CC(_tnn.Module):
    def __init__(s, ic, oc, k, d):
        super().__init__()
        s.pad = (k - 1) * d
        s.conv = _tnn.Conv1d(ic, oc, k, dilation=d)
    def forward(s, x):
        return s.conv(_F.pad(x, (s.pad, 0)))

class _Ctx(_tnn.Module):
    def __init__(s, n_in=2, d_ctx=D_CTX, ch=32, dils=(1, 2, 4)):
        super().__init__()
        L = []
        ic = n_in
        for d in dils:
            L += [_tnn.Conv1d(ic, ch, 3, padding=d, dilation=d), _tnn.GELU(), _tnn.Dropout(0.15)]
            ic = ch
        s.body = _tnn.Sequential(*L)
        s.proj = _tnn.Conv1d(ch, d_ctx, 1)
    def forward(s, tx, tm):
        h = s.proj(s.body(tx))
        m = tm.unsqueeze(1)
        return (h * m).sum(-1) / m.sum(-1).clamp(min=1.0)

class _TCN(_tnn.Module):
    def __init__(s, n_row, d_ctx, ch=TCN_CH, k=3, dils=TCN_DIL, p=TCN_DO):
        super().__init__()
        ic = n_row + d_ctx
        L = []
        for d in dils:
            L += [_CC(ic, ch, k, d), _tnn.GELU(), _tnn.Dropout(p)]
            ic = ch
        s.encoder = _tnn.Sequential(*L)
        s.head = _tnn.Conv1d(ch, 1, 1)
    def forward(s, xr, c):
        cb = c.unsqueeze(-1).expand(-1, -1, xr.shape[-1])
        return s.head(s.encoder(torch.cat([xr, cb], 1))).squeeze(1)

ft = np.nan_to_num(test_df[NN_FEATURES].to_numpy(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
ftn = ((ft - fmean) / fstd).astype(np.float32)
del ft

wells_te = test_df["well"].to_numpy()
md_te_nn = test_df["md_since"].to_numpy(np.float32)
TW_TEST = CFG.dataset_path / "test"

def _load_tw(uw):
    T, M = {}, {}
    for w in uw:
        tw = pd.read_csv(TW_TEST / f"{w}__typewell.csv").dropna(subset=["TVT", "GR"]).sort_values("TVT").reset_index(drop=True)
        if len(tw) > TW_MAX:
            idx = np.linspace(0, len(tw) - 1, TW_MAX).astype(int)
            tw = tw.iloc[idx].reset_index(drop=True)
        a = np.zeros((TW_F, TW_MAX), np.float32)
        m = np.zeros((TW_MAX,), np.float32)
        n = len(tw)
        a[0, :n] = tw["TVT"].to_numpy(np.float32) / TVT_NORM
        a[1, :n] = tw["GR"].to_numpy(np.float32) / GR_NORM
        if n < TW_MAX:
            a[0, n:] = a[0, n - 1]
            a[1, n:] = a[1, n - 1]
        m[:n] = 1.0
        T[w] = a
        M[w] = m
    return T, M

twT, twM = _load_tw(np.unique(wells_te))

def _seqs(ws, ms, gi, mx=12000):
    o = []
    for w in np.unique(ws):
        loc = np.where(ws == w)[0]
        loc = loc[np.argsort(ms[loc])]
        if len(loc) > mx:
            loc = loc[:mx]
        o.append({"well": w, "gi": gi[loc], "n": len(loc)})
    return o

def _pad(seqs):
    B = len(seqs)
    Tm = max(s["n"] for s in seqs)
    Fd = ftn.shape[1]
    x = np.zeros((B, Fd, Tm), np.float32)
    mk = np.zeros((B, Tm), np.float32)
    tx = np.zeros((B, TW_F, TW_MAX), np.float32)
    tm = np.zeros((B, TW_MAX), np.float32)
    for b, s in enumerate(seqs):
        gi = s["gi"]
        n = len(gi)
        x[b, :, :n] = ftn[gi].T
        mk[b, :n] = 1.0
        tx[b] = twT[s["well"]]
        tm[b] = twM[s["well"]]
    return {
        "x": torch.from_numpy(x).to(DEVICE),
        "m": torch.from_numpy(mk).to(DEVICE),
        "tx": torch.from_numpy(tx).to(DEVICE),
        "tm": torch.from_numpy(tm).to(DEVICE)
    }

seqs_te = _seqs(wells_te, md_te_nn, np.arange(Nte))
acc = np.zeros((N_SEEDS, Nte), np.float32)
cnt = np.zeros(N_SEEDS, np.int32)
s2i = {sd: i for i, sd in enumerate(SEEDS_USED)}

for ci, c in enumerate(checkpoints):
    si = s2i[c["seed"]]
    cx = _Ctx(d_ctx=D_CTX).to(DEVICE)
    tc = _TCN(n_row=len(NN_FEATURES), d_ctx=D_CTX).to(DEVICE)
    cx.load_state_dict(c["ctxnet_sd"])
    tc.load_state_dict(c["tcn_sd"])
    cx.eval()
    tc.eval()
    st = np.zeros(Nte, np.float32)
    with torch.no_grad():
        for s in seqs_te:
            bb = _pad([s])
            ctx = cx(bb["tx"], bb["tm"])
            dn = tc(bb["x"], ctx).squeeze(0).cpu().numpy()
            for j, gi in enumerate(s["gi"]):
                st[int(gi)] = float(dn[j] * TARGET_NORM)
    acc[si] += st
    cnt[si] += 1
    del cx, tc

test_preds["nn"] = (acc / np.maximum(cnt, 1)[:, None]).mean(0).astype(np.float32)
print(f"[v83/86] old-nn test recomputed on {DEVICE}", flush=True)
del ftn, seqs_te; gc.collect()

V35V2_CK = loc("nn_checkpoints_v35_pfv2.pt")

def _load_v35pfv2_raw(p):
    hw = pd.read_csv(p).reset_index(drop=True)
    for col in ("X", "Y", "Z", "MD", "GR"):
        if col not in hw.columns: continue
        if hw[col].isna().any():
            hw[col] = hw[col].interpolate(limit_direction="both")
            if hw[col].isna().any():
                mv = hw[col].mean(); hw[col] = hw[col].fillna(0.0 if pd.isna(mv) else mv)
    return hw

def _pfv2_one_well(wid):
    p_hw = CFG.dataset_path / "test" / f"{wid}__horizontal_well.csv"
    p_tw = CFG.dataset_path / "test" / f"{wid}__typewell.csv"
    try:
        hw = _load_v35pfv2_raw(p_hw)
        tw = pd.read_csv(p_tw).sort_values("TVT")
    except:
        return None
    tw_tvt = tw["TVT"].to_numpy(np.float32)
    tw_gr = tw["GR"].to_numpy(np.float32)
    pf_a, _ = run_pf_ancc(hw, tw_tvt, tw_gr)
    if len(pf_a) == 0: return None
    ev = hw[hw["TVT_input"].isna()]
    return pd.DataFrame({"id": [f"{wid}_{i}" for i in ev.index], "pf_ancc": pf_a})

# Construct PF35V2 map
_pfv2_parts = [_pfv2_one_well(w) for w in test_wids]
_pfv2_parts = [p for p in _pfv2_parts if p is not None]
_pfv2_df = pd.concat(_pfv2_parts, ignore_index=True) if _pfv2_parts else pd.DataFrame(columns=["id", "pf_ancc"])
PF35V2_MAP = dict(zip(_pfv2_df["id"].astype(str).values, _pfv2_df["pf_ancc"].astype(np.float64).values))

ck35v2 = torch.load(V35V2_CK, map_location="cpu", weights_only=False)
C35v2 = ck35v2["config"]
NF35v2 = C35v2["N_FEATS"]
HID35v2 = C35v2["HID"]
NB35v2 = C35v2["N_BLOCKS"]
K35v2 = C35v2["KERNEL"]
DO35v2 = C35v2["DROPOUT"]
DTVT_STD35v2 = float(ck35v2["norm"]["dtvt_std"])
MEAN35v2 = np.asarray(ck35v2["norm"]["mean"], np.float32)
STD35v2 = np.asarray(ck35v2["norm"]["std"], np.float32)

class _DTCN35v2(_tnn.Module):
    def __init__(s, in_feats=NF35v2, hid=HID35v2, n_blocks=NB35v2, k=K35v2):
        super().__init__()
        s.inp = _tnn.Conv1d(in_feats, hid, 1)
        s.b1 = _tnn.ModuleList()
        s.b2 = _tnn.ModuleList()
        for i in range(n_blocks):
            d = 2**i
            pad = d*(k-1)//2
            s.b1.append(_tnn.Conv1d(hid, hid, k, padding=pad, dilation=d))
            s.b2.append(_tnn.Conv1d(hid, hid, k, padding=pad, dilation=d))
        s.out = _tnn.Conv1d(hid, 1, 1)
        s.drop = _tnn.Dropout1d(DO35v2)
    def forward(s, x):
        h = _F.gelu(s.inp(x))
        for c1, c2 in zip(s.b1, s.b2):
            r = h; h = s.drop(_F.gelu(c1(h))); h = _F.gelu(c2(h)); h = h + r
        return s.out(h).squeeze(1).squeeze(0)

def _smstd22(a, w):
    m = _sm22(a, w); m2 = _sm22(a*a, w); return np.sqrt(np.clip(m2-m*m, 1e-8, None))

def _sm22(a, w):
    pad = w//2; ap = np.pad(a, pad, mode="edge"); return np.convolve(ap, np.ones(w)/w, mode="valid")[:len(a)]

def _load_v35pfv2(p):
    wid = p.stem.replace("__horizontal_well", "")
    hw = _load_v35pfv2_raw(p)
    kn = hw["TVT_input"].notna().values; kni = np.where(kn)[0]; evi = np.where(~kn)[0]
    if len(kni) < 200: return None
    md = hw["MD"].values.astype(np.float64); x = hw["X"].values.astype(np.float64); yv = hw["Y"].values.astype(np.float64)
    z = hw["Z"].values.astype(np.float64); gr = hw["GR"].values.astype(np.float64)
    ai = int(kni[-1]); n = len(md)
    mds = md - md[ai]; dmd = np.diff(md, prepend=md[0]); dx = np.diff(x, prepend=x[0]); dy = np.diff(yv, prepend=yv[0]); dz = np.diff(z, prepend=z[0])
    grd1 = np.diff(gr, prepend=gr[0]); grenv = _smstd22(gr, 30); atvt = float(hw["TVT_input"].values[ai])
    tir = np.zeros(n); tir[kni] = hw["TVT_input"].values[kni] - atvt; tim = kn.astype(np.float64)
    pf_rel = np.zeros(n)
    for i in evi:
        v = PF35V2_MAP.get(f"{wid}_{int(i)}")
        if v is not None: pf_rel[i] = v - atvt
    feats = np.stack([mds, dmd, x, yv, z, dx, dy, dz, gr, grd1, grenv, tir, tim, pf_rel], axis=1).astype(np.float32)
    pm = (np.arange(n) > ai).astype(np.float32)
    rids = np.array([f"{wid}_{int(i)}" for i in evi])
    return dict(feats=feats, atvt=atvt, pm=pm, evi=evi, rids=rids)

# Load model weights and run
v35v2_models = []
for d in ck35v2["seed_fold_state_dicts"]:
    m = _DTCN35v2().to(DEVICE)
    m.load_state_dict(d["sd"])
    m.eval()
    v35v2_models.append(m)

_m35v2 = torch.tensor(MEAN35v2, device=DEVICE)
_s35v2 = torch.tensor(STD35v2, device=DEVICE)
v35v2_rows = {}
_v35v2_skip = 0

with torch.no_grad():
    for p in test_paths:
        w = _load_v35pfv2(p)
        if w is None:
            _v35v2_skip += 1
            continue
        xx = ((torch.tensor(w["feats"], device=DEVICE) - _m35v2) / _s35v2).t().unsqueeze(0).contiguous()
        pm = torch.tensor(w["pm"], device=DEVICE)
        acc = np.zeros(len(w["evi"]), np.float64)
        for m in v35v2_models:
            dn = m(xx)
            rec = w["atvt"] + torch.cumsum(dn * DTVT_STD35v2 * pm, dim=0)
            acc += rec.cpu().numpy()[w["evi"]].astype(np.float64)
        acc /= len(v35v2_models)
        for r_, v_ in zip(w["rids"], acc):
            v35v2_rows[r_] = v_

v35v2_abs_te = test_df["id"].astype(str).map(v35v2_rows).to_numpy(np.float64)
_miss35v2 = ~np.isfinite(v35v2_abs_te)
if _miss35v2.any():
    v35v2_abs_te[_miss35v2] = base_te[_miss35v2]

test_preds["v35pfv2"] = v35v2_abs_te - base_te
print(f"[v86] Finished v35pfv2 test inference.", flush=True)

V22_CK = loc("nn_checkpoints_v22.pt")
V35_CK = loc("nn_checkpoints_v35.pt")

class _DTCN22(_tnn.Module):
    def __init__(s, in_feats=13, hid=64, n_blocks=10, k=3):
        super().__init__()
        s.inp = _tnn.Conv1d(in_feats, hid, 1)
        s.b1 = _tnn.ModuleList()
        s.b2 = _tnn.ModuleList()
        for i in range(n_blocks):
            d = 2**i
            pad = d*(k-1)//2
            s.b1.append(_tnn.Conv1d(hid, hid, k, padding=pad, dilation=d))
            s.b2.append(_tnn.Conv1d(hid, hid, k, padding=pad, dilation=d))
        s.out = _tnn.Conv1d(hid, 1, 1)
        s.drop = _tnn.Dropout1d(0.1)
    def forward(s, x):
        h = _F.gelu(s.inp(x))
        for c1, c2 in zip(s.b1, s.b2):
            r = h; h = s.drop(_F.gelu(c1(h))); h = _F.gelu(c2(h)); h = h + r
        return s.out(h).squeeze(1).squeeze(0)

def _load_v22(p):
    wid = p.stem.replace("__horizontal_well", "")
    hw = _load_v35pfv2_raw(p)
    kn = hw["TVT_input"].notna().values; kni = np.where(kn)[0]; evi = np.where(~kn)[0]
    if len(kni) < 200: return None
    md = hw["MD"].values.astype(np.float64); x = hw["X"].values.astype(np.float64); yv = hw["Y"].values.astype(np.float64)
    z = hw["Z"].values.astype(np.float64); gr = hw["GR"].values.astype(np.float64)
    ai = int(kni[-1]); n = len(md)
    mds = md - md[ai]; dmd = np.diff(md, prepend=md[0]); dx = np.diff(x, prepend=x[0]); dy = np.diff(yv, prepend=yv[0]); dz = np.diff(z, prepend=z[0])
    grd1 = np.diff(gr, prepend=gr[0]); grenv = _smstd22(gr, 30); atvt = float(hw["TVT_input"].values[ai])
    tir = np.zeros(n); tir[kni] = hw["TVT_input"].values[kni] - atvt; tim = kn.astype(np.float64)
    feats = np.stack([mds, dmd, x, yv, z, dx, dy, dz, gr, grd1, grenv, tir, tim], axis=1).astype(np.float32)
    pm = (np.arange(n) > ai).astype(np.float32)
    rids = np.array([f"{wid}_{int(i)}" for i in evi])
    return dict(feats=feats, atvt=atvt, pm=pm, evi=evi, rids=rids)

ck22 = torch.load(V22_CK, map_location="cpu", weights_only=False)
v22_models = []
for d in ck22["seed_fold_state_dicts"]:
    C = ck22["config"]
    m = _DTCN22(in_feats=C["N_FEATS"], hid=C["HID"], n_blocks=C["N_BLOCKS"], k=C["KERNEL"]).to(DEVICE)
    m.load_state_dict(d["sd"])
    m.eval()
    v22_models.append(m)

_m22 = torch.tensor(np.asarray(ck22["norm"]["mean"], np.float32), device=DEVICE)
_s22 = torch.tensor(np.asarray(ck22["norm"]["std"], np.float32), device=DEVICE)
v22_rows = {}

with torch.no_grad():
    for p in test_paths:
        w = _load_v22(p)
        if w is None: continue
        xx = ((torch.tensor(w["feats"], device=DEVICE) - _m22) / _s22).t().unsqueeze(0).contiguous()
        pm = torch.tensor(w["pm"], device=DEVICE)
        acc = np.zeros(len(w["evi"]), np.float64)
        for m in v22_models:
            dn = m(xx)
            rec = w["atvt"] + torch.cumsum(dn * float(ck22["norm"]["dtvt_std"]) * pm, dim=0)
            acc += rec.cpu().numpy()[w["evi"]].astype(np.float64)
        acc /= len(v22_models)
        for r_, v_ in zip(w["rids"], acc):
            v22_rows[r_] = v_

v22_abs_te = test_df["id"].astype(str).map(v22_rows).to_numpy(np.float64)
_miss22 = ~np.isfinite(v22_abs_te)
if _miss22.any():
    v22_abs_te[_miss22] = base_te[_miss22]
test_preds["v22"] = v22_abs_te - base_te

PF35_MAP = dict(zip(test_df["id"].astype(str).values, test_df["pf_ancc"].astype(np.float64).values))
def _load_v35(p):
    wid = p.stem.replace("__horizontal_well", "")
    hw = _load_v35pfv2_raw(p)
    kn = hw["TVT_input"].notna().values; kni = np.where(kn)[0]; evi = np.where(~kn)[0]
    if len(kni) < 200: return None
    md = hw["MD"].values.astype(np.float64); x = hw["X"].values.astype(np.float64); yv = hw["Y"].values.astype(np.float64)
    z = hw["Z"].values.astype(np.float64); gr = hw["GR"].values.astype(np.float64)
    ai = int(kni[-1]); n = len(md)
    mds = md - md[ai]; dmd = np.diff(md, prepend=md[0]); dx = np.diff(x, prepend=x[0]); dy = np.diff(yv, prepend=yv[0]); dz = np.diff(z, prepend=z[0])
    grd1 = np.diff(gr, prepend=gr[0]); grenv = _smstd22(gr, 30); atvt = float(hw["TVT_input"].values[ai])
    tir = np.zeros(n); tir[kni] = hw["TVT_input"].values[kni] - atvt; tim = kn.astype(np.float64)
    pf_rel = np.zeros(n)
    for i in evi:
        v = PF35_MAP.get(f"{wid}_{int(i)}")
        if v is not None: pf_rel[i] = v - atvt
    feats = np.stack([mds, dmd, x, yv, z, dx, dy, dz, gr, grd1, grenv, tir, tim, pf_rel], axis=1).astype(np.float32)
    pm = (np.arange(n) > ai).astype(np.float32)
    rids = np.array([f"{wid}_{int(i)}" for i in evi])
    return dict(feats=feats, atvt=atvt, pm=pm, evi=evi, rids=rids)

ck35 = torch.load(V35_CK, map_location="cpu", weights_only=False)
v35_models = []
for d in ck35["seed_fold_state_dicts"]:
    C = ck35["config"]
    m = _DTCN22(in_feats=C["N_FEATS"], hid=C["HID"], n_blocks=C["N_BLOCKS"], k=C["KERNEL"]).to(DEVICE)
    m.load_state_dict(d["sd"])
    m.eval()
    v35_models.append(m)

_m35 = torch.tensor(np.asarray(ck35["norm"]["mean"], np.float32), device=DEVICE)
_s35 = torch.tensor(np.asarray(ck35["norm"]["std"], np.float32), device=DEVICE)
v35_rows = {}

with torch.no_grad():
    for p in test_paths:
        w = _load_v35(p)
        if w is None: continue
        xx = ((torch.tensor(w["feats"], device=DEVICE) - _m35) / _s35).t().unsqueeze(0).contiguous()
        pm = torch.tensor(w["pm"], device=DEVICE)
        acc = np.zeros(len(w["evi"]), np.float64)
        for m in v35_models:
            dn = m(xx)
            rec = w["atvt"] + torch.cumsum(dn * float(ck35["norm"]["dtvt_std"]) * pm, dim=0)
            acc += rec.cpu().numpy()[w["evi"]].astype(np.float64)
        acc /= len(v35_models)
        for r_, v_ in zip(w["rids"], acc):
            v35_rows[r_] = v_

v35_abs_te = test_df["id"].astype(str).map(v35_rows).to_numpy(np.float64)
_miss35 = ~np.isfinite(v35_abs_te)
if _miss35.any():
    v35_abs_te[_miss35] = base_te[_miss35]
test_preds["v35"] = v35_abs_te - base_te

print(f"[v86] Finished v22 & v35 test inference.", flush=True)

# Run the 14-profile particle filters
_ALL14 = [
    ("P06",0.000961,0.002405,0.9994996,0.8832),("P23",0.001679,0.008291,0.9986679,1.4888),
    ("P51",0.000627,0.012862,0.9994678,1.3597),("P34",0.002591,0.001572,0.9986164,0.9167),
    ("P00",0.002,0.005,0.998,1.0),("P47",0.002202,0.023204,0.9991609,1.1602),
    ("P26",0.001427,0.003534,0.9991874,0.712),("P54",0.000782,0.001116,0.9989704,2.0338),
    ("P37",0.003533,0.019424,0.9994264,0.6399),("P15",0.001967,0.014268,0.9970956,0.6591),
    ("P42",0.001742,0.004797,0.9993618,1.9468),("P39",0.001329,0.0115,0.998187,0.6062),
    ("P29",0.004637,0.006808,0.9980513,0.8121),("P14",0.000673,0.03138,0.9989144,1.5863)
]
_PIDS = [p[0] for p in _ALL14]
_PROFS = [{"VN":v,"PN":pn,"MOM":m,"gs_mult":g} for (_,v,pn,m,g) in _ALL14]
_PFMEM = ["P06","P23","P51","P54"]

_PIDS = _PIDS + ["BIOU00", "BIOUP06", "BIOUP42", "BIOUP51", "BIASP06",
                 "OU_P00", "OU_P06", "OU_P23", "OU_P26"]
_PROFS = _PROFS + [
    {"VN":0.002,       "PN":0.005,      "MOM":0.998,     "gs_mult":1.0},
    {"VN":0.000961117, "PN":0.0024054,  "MOM":0.9994996, "gs_mult":0.883208},
    {"VN":0.00174216,  "PN":0.00479679, "MOM":0.9993618, "gs_mult":1.94683},
    {"VN":0.000626873, "PN":0.0128618,  "MOM":0.9994678, "gs_mult":1.35974},
    {"VN":0.000961117, "PN":0.0024054,  "MOM":0.9994996, "gs_mult":0.883208},

    {"VN":0.002,       "PN":0.005,      "MOM":0.998,     "gs_mult":1.0},
    {"VN":0.000961117, "PN":0.0024054,  "MOM":0.9994996, "gs_mult":0.883208},
    {"VN":0.0016791,   "PN":0.00829073, "MOM":0.9986679, "gs_mult":1.48881},
    {"VN":0.00142661,  "PN":0.00353359, "MOM":0.9991874, "gs_mult":0.711957},
]
_SHARPM = np.array([0]*14 + [1, 1, 1, 1, 1] + [0, 0, 0, 0], np.bool_)
_OUM = np.array([0.0]*14 + [1.0, 1.0, 1.0, 1.0, 0.0] + [1.0, 1.0, 1.0, 1.0], np.float32)
_KSTAR99 = 0.006; _DQK99 = 10; _DQSPW99 = 60
_BWT99 = 2.0; _SDFLOOR99 = 6.0; _NMIN99 = 25; _SIBSMIN99 = 5
_NPF = len(_PROFS); _PFS = 128; _PFN = 500; _PFCW = 30; _PFINIT = 4.5; _PFSC = (3.,5.,8.,12.)
_PFDEV = "cuda" if torch.cuda.is_available() else "cpu"

@torch.no_grad()
def _pf_mp(ws, profiles, seed=0):
    f32 = torch.float32; W = len(ws); g = torch.Generator(device=_PFDEV).manual_seed(seed)
    Tm = max(w["ne"] for w in ws); Ttw = max(len(w["tw_tvt"]) for w in ws)
    md = np.zeros((W,Tm),np.float32); z = np.zeros((W,Tm),np.float32); gr = np.zeros((W,Tm),np.float32)
    twt = np.full((W,Ttw),np.inf,np.float32); twg = np.zeros((W,Ttw),np.float32)
    lsA = np.zeros(W,np.float32); irA = np.zeros(W,np.float32); gsA = np.zeros(W,np.float32)
    loA = np.zeros(W,np.float32); hiA = np.zeros(W,np.float32); lmd = np.zeros(W,np.float32)
    NBmax = max(max(int(w.get("tb_nb",0)),1) for w in ws)
    TBb = np.zeros((W,NBmax),np.float32); TBs = np.full((W,NBmax),1e9,np.float32); TBv = np.zeros((W,NBmax),np.bool_)
    T0A = np.zeros(W,np.float32); NBA = np.zeros(W,np.float32); HASA = np.zeros(W,np.bool_)
    surfA = np.zeros((W,Tm),np.float32); sigfA = np.full((W,Tm),80.0,np.float32); HSFA = np.zeros(W,np.float32)
    for j,w in enumerate(ws):
        n = w["ne"]; md[j,:n] = w["md"]; z[j,:n] = w["z"]; gr[j,:n] = w["gr"]
        if n < Tm: md[j,n:] = w["md"][-1] + np.arange(1,Tm-n+1); z[j,n:] = w["z"][-1]; gr[j,n:] = w["gr"][-1]
        surfA[j,:n] = w["surf"]; sigfA[j,:n] = w["sigf"]
        if n < Tm: surfA[j,n:] = w["surf"][-1]
        HSFA[j] = 1.0 if w.get("has_surf", False) else 0.0
        Tw = len(w["tw_tvt"]); twt[j,:Tw] = w["tw_tvt"]; twg[j,:Tw] = w["tw_gr"]
        lsA[j] = w["ls"]; irA[j] = w["ir"]; gsA[j] = w["gs"]
        loA[j] = w["tw_tvt"][0] - 100; hiA[j] = w["tw_tvt"][-1] + 100; lmd[j] = w["last_md"]
        nb99 = int(w.get("tb_nb",0))
        if nb99 > 0:
            TBb[j,:nb99] = w["tb_bias"]; TBs[j,:nb99] = w["tb_sd"]; TBv[j,:nb99] = w["tb_valid"]
        T0A[j] = float(w.get("tb_t0",0.0)); NBA[j] = float(nb99); HASA[j] = bool(w.get("has_tb",False))
    T = lambda a: torch.as_tensor(a,device=_PFDEV)
    md,z,gr,twt,twg = T(md),T(z),T(gr),T(twt),T(twg)
    TBb,TBs,TBv = T(TBb),T(TBs),T(TBv)
    surfT99, sigfT99 = T(surfA), T(sigfA); HSF99 = T(HSFA).view(W,1,1,1)
    ouF99 = T(_OUM[:len(profiles)]).view(1,len(profiles),1,1)
    T099 = T(T0A).view(W,1); NB99 = T(NBA).view(W,1); HAS99 = T(HASA).view(W,1)
    shp99 = T(_SHARPM[:len(profiles)]).view(1,len(profiles),1,1)
    one99 = torch.ones((),device=_PFDEV,dtype=f32)
    ls = T(lsA).view(W,1,1,1); ir = T(irA).view(W,1,1,1); gsW = T(gsA).view(W,1,1,1); loV = T(loA).view(W,1,1,1); hiV = T(hiA).view(W,1,1,1)
    MOM = T(np.array([p["MOM"] for p in profiles],np.float32)).view(1,_NPF,1,1)
    VN = T(np.array([p["VN"] for p in profiles],np.float32)).view(1,_NPF,1,1)
    PN = T(np.array([p["PN"] for p in profiles],np.float32)).view(1,_NPF,1,1)
    gsm = T(np.array([p["gs_mult"] for p in profiles],np.float32)).view(1,_NPF,1,1)
    gs = gsW * gsm; RP,RR,RESAMP = 0.1,0.001,0.5; S = _PFS; N = _PFN
    pos = ls + _PFINIT * torch.randn(W,_NPF,S,N,generator=g,device=_PFDEV,dtype=f32)
    rate = ir + 0.01 * torch.randn(W,_NPF,S,N,generator=g,device=_PFDEV,dtype=f32)
    wt = torch.full((W,_NPF,S,N),1.0/N,device=_PFDEV,dtype=f32)
    res = torch.empty(W,_NPF,S,Tm,device=_PFDEV,dtype=f32)
    res2 = torch.empty(W,_NPF,S,Tm,device=_PFDEV,dtype=f32)
    log_lik = torch.zeros(W,_NPF,S,device=_PFDEV,dtype=f32)
    ar = (torch.arange(N,device=_PFDEV,dtype=f32)/N).view(1,1,1,N)
    prev = T(lmd).clone()
    for i in range(Tm):
        zi = z[:,i].view(W,1,1,1); gri = gr[:,i].view(W,1,1,1); dm = (md[:,i]-prev).clamp(min=1.0).view(W,1,1,1); prev = md[:,i]
        rate = MOM * rate + VN * torch.randn(W,_NPF,S,N,generator=g,device=_PFDEV,dtype=f32)
        pos = pos + rate * dm + PN * torch.randn(W,_NPF,S,N,generator=g,device=_PFDEV,dtype=f32)

        sfi99 = sigfT99[:,i].view(W,1,1,1); sui99 = surfT99[:,i].view(W,1,1,1)
        g_ou99 = torch.clamp(_KSTAR99*dm, max=0.05) * torch.clamp((10.0/sfi99)**2, max=1.0) * HSF99 * ouF99
        pos = pos + g_ou99 * (sui99 - pos)
        tvt_p = (pos-zi).clamp(loV,hiV); pos = tvt_p + zi
        x = tvt_p.reshape(W,_NPF*S*N); idx = torch.searchsorted(twt,x,right=True).clamp(1,Ttw-1)
        x0 = torch.gather(twt,1,idx-1); x1 = torch.gather(twt,1,idx); y0 = torch.gather(twg,1,idx-1); y1 = torch.gather(twg,1,idx)
        eg = (y0+(y1-y0)*(x-x0)/(x1-x0).clamp(min=1e-12)).reshape(W,_NPF,S,N)

        bi99 = torch.floor((x-T099)/_BWT99).long(); inr99 = (bi99>=0)&(bi99<NB99.long()); bic99 = bi99.clamp(0,TBv.shape[1]-1)
        vld99 = (inr99 & torch.gather(TBv,1,bic99) & HAS99).reshape(W,_NPF,S,N) & shp99
        mu = eg + torch.where(vld99, torch.gather(TBb,1,bic99).reshape(W,_NPF,S,N), torch.zeros((),device=_PFDEV,dtype=f32))
        d = (gri-mu)/gs
        lk = torch.exp(-0.5*torch.clamp(d*d,max=600.)).clamp_min(1e-30)
        log_lik = log_lik + torch.log((wt*lk).sum(3).clamp_min(1e-30))
        wt = wt * lk; wt = wt / wt.sum(3,keepdim=True).clamp_min(1e-30)
        n_eff = 1.0/(wt*wt).sum(3); mask = (n_eff < RESAMP*N).unsqueeze(3)
        cum = torch.cumsum(wt,dim=3); u0 = torch.rand(W,_NPF,S,1,generator=g,device=_PFDEV,dtype=f32)/N
        idx2 = torch.searchsorted(cum,(u0+ar).clamp(max=1.0)).clamp(0,N-1)
        pos_r = torch.gather(pos,3,idx2) + RP * torch.randn(W,_NPF,S,N,generator=g,device=_PFDEV,dtype=f32)
        rate_r = torch.gather(rate,3,idx2) + RR * torch.randn(W,_NPF,S,N,generator=g,device=_PFDEV,dtype=f32)
        pos = torch.where(mask,pos_r,pos); rate = torch.where(mask,rate_r,rate); wt = torch.where(mask,torch.full_like(wt,1.0/N),wt)
        tvt = pos - zi; res[:,:,:,i] = (wt*tvt).sum(3); res2[:,:,:,i] = (wt*tvt*tvt).sum(3)
    ln = log_lik - log_lik.max(dim=2,keepdim=True).values; mo = {}; so = {}
    for sc in _PFSC:
        ew = torch.exp(ln/float(sc)); ew = ew/ew.sum(2,keepdim=True); mm = (ew.unsqueeze(3)*res).sum(2); e2 = (ew.unsqueeze(3)*res2).sum(2)
        mo[sc] = mm.cpu().numpy(); so[sc] = torch.sqrt((e2-mm*mm).clamp_min(0.)).cpu().numpy()
    return {w["wid"]:{"mean":{f"pf_scale_{sc:g}":mo[sc][j] for sc in _PFSC},"std":{f"pf_scale_{sc:g}":so[sc][j] for sc in _PFSC}} for j,w in enumerate(ws)}

# Beam search parameters
_BEAMC = [
    (10,20.,144.,2),(10,8.,64.,2),(8,35.,220.,1),(10,14.,90.,5),(20,4.,36.,3),(12,12.,100.,3),
    (15,25.,180.,2),(20,30.,200.,2),(15,10.,80.,4),(25,6.,50.,3),(10,40.,300.,1),(12,18.,120.,5),
    (30,8.,70.,2),(10,50.,400.,0)
]
_SELT = 4840.0; _SELZ = (136.73000000000016, 185.5133333333342)
_SELV = {
    0:"pf_scale_5_hold_0.2", 1:"pf_scale_3_hold_0.15", 2:"pf_scale_12_beam_0.2_hold_0.15",
    3:"pf_scale_5_hold_0.15", 4:"pf_scale_5_beam_0.05_hold_0.05", 5:"pf_scale_12_beam_0.2_hold_0.05"
}
_SELG = "pf_scale_8_hold_0.2"
_PFTEST = str(CFG.dataset_path / "test")

def _v99_tops_sig(tw_df):
    g_ = tw_df.dropna(subset=["Geology"]).sort_values("TVT")
    seq, prev = [], None
    for geo, tvt in zip(g_["Geology"].values, g_["TVT"].values):
        if geo != prev:
            seq.append(f"{geo}:{round(float(tvt),2):.2f}"); prev = geo
    return "|".join(seq)

def _v99_grid1ft(tw_tvt, tw_gr):
    lo = int(np.ceil(tw_tvt[0])); hi = int(np.floor(tw_tvt[-1]))
    if hi <= lo: return None, None
    gx = np.arange(lo, hi + 1, dtype=np.float64)
    return gx, np.interp(gx, tw_tvt, tw_gr)

_t99 = time.time()
_TRAIN99 = str(CFG.dataset_path / "train")
_w99 = []
for _p in sorted(glob.glob(f"{_TRAIN99}/*__horizontal_well.csv")):
    _wid = os.path.basename(_p).split("__")[0]
    _hw = pd.read_csv(_p); _tw = pd.read_csv(f"{_TRAIN99}/{_wid}__typewell.csv")
    if len(_tw) < 3 or _tw["TVT"].notna().sum() < 3: continue
    _tws = _tw.sort_values("TVT")
    _ttvt = _tws["TVT"].values.astype(np.float64)
    _tgr = _tws["GR"].fillna(_tws["GR"].mean()).values.astype(np.float64)
    _gr = _hw["GR"].interpolate(limit_direction="both").fillna(np.nanmean(_tgr)).values.astype(np.float64)
    _tvt = _hw["TVT"].values.astype(np.float64) if "TVT" in _hw else np.full(len(_hw), np.nan)
    _ok = np.isfinite(_tvt) & np.isfinite(_gr)
    _gx, _gy = _v99_grid1ft(_ttvt, _tgr)
    _w99.append(dict(wid=_wid, sig=_v99_tops_sig(_tw), span=_ttvt[-1]-_ttvt[0],
                     t=_tvt[_ok], g=_gr[_ok], gx0=(_gx[0] if _gx is not None else None), gy=_gy))
from collections import defaultdict as _dd99

_NW99 = len(_w99); _par99 = list(range(_NW99))
def _find99(a):
    while _par99[a] != a: _par99[a] = _par99[_par99[a]]; a = _par99[a]
    return a
for _i in range(_NW99):
    _wi = _w99[_i]
    if _wi["gy"] is None: continue
    _x0i = int(_wi["gx0"]); _ni = len(_wi["gy"]); _gyi = _wi["gy"]
    for _j in range(_i + 1, _NW99):
        _wj = _w99[_j]
        if _wj["gy"] is None: continue
        _o0 = int(_wj["gx0"]); _mm = len(_wj["gy"])
        _lo = max(_x0i, _o0); _h = min(_x0i + _ni, _o0 + _mm); _ov = _h - _lo
        if _ov <= 0 or _ov < 0.9 * min(_ni, _mm): continue
        _dd = _gyi[_lo - _x0i:_h - _x0i] - _wj["gy"][_lo - _o0:_h - _o0]
        if float(np.sqrt(np.mean(_dd * _dd))) < 0.01: _par99[_find99(_i)] = _find99(_j)
for _i in range(_NW99): _w99[_i]["cl"] = _find99(_i)
_bysig99 = _dd99(list)
for _w in _w99: _bysig99[_w["cl"]].append(_w)
TBL99 = {}
for _sig, _ws99 in _bysig99.items():
    if len(_ws99) < _SIBSMIN99: continue
    _mo = max(_ws99, key=lambda w: w["span"])
    _twm = pd.read_csv(f"{_TRAIN99}/{_mo['wid']}__typewell.csv").sort_values("TVT")
    _mgr = _twm["GR"].fillna(_twm["GR"].mean()).values.astype(np.float64)
    _mtvt = _twm["TVT"].values.astype(np.float64)
    _m0, _m1 = float(_mtvt[0]), float(_mtvt[-1])
    _edges = np.arange(np.floor(_m0), np.ceil(_m1) + _BWT99, _BWT99); _nb = len(_edges) - 1
    _cnt = np.zeros(_nb); _s1 = np.zeros(_nb); _s2 = np.zeros(_nb)
    for _w in _ws99:
        _t, _g = _w["t"], _w["g"]; _in = (_t >= _m0) & (_t <= _m1); _t, _g = _t[_in], _g[_in]
        _b = np.clip(np.digitize(_t, _edges) - 1, 0, _nb - 1)
        _r = _g - np.interp(_t, _mtvt, _mgr)
        np.add.at(_cnt, _b, 1.0); np.add.at(_s1, _b, _r); np.add.at(_s2, _b, _r * _r)
    with np.errstate(invalid="ignore", divide="ignore"):
        _mu = _s1 / _cnt
        _va = _s2 / _cnt - _mu * _mu
    _valid = _cnt >= _NMIN99
    TBL99[_sig] = dict(t0=float(_edges[0]), nb=int(_nb),
                       bias=np.where(_valid, _mu, 0.0).astype(np.float32),
                       sd=np.maximum(np.where(_valid, np.sqrt(np.maximum(_va, 0.0)), 1e9), _SDFLOOR99).astype(np.float32),
                       valid=_valid)
print(f"[V99] sharp tables: {len(TBL99)} templates ({time.time()-_t99:.0f}s)", flush=True)

def _v99_match_test(tw_path):
    _tw = pd.read_csv(tw_path)
    if len(_tw) < 3 or _tw["TVT"].notna().sum() < 3: return None
    _tws = _tw.sort_values("TVT")
    _gx, _gy = _v99_grid1ft(_tws["TVT"].values.astype(np.float64),
                            _tws["GR"].fillna(_tws["GR"].mean()).values.astype(np.float64))
    if _gx is None or len(_gx) < 50: return None
    _x0 = int(_gx[0]); _n = len(_gx)
    _best = (1e18, None)
    for _w in _w99:
        if _w["gy"] is None: continue
        _o0 = int(_w["gx0"]); _m = len(_w["gy"])
        _lo = max(_x0, _o0); _hi = min(_x0 + _n, _o0 + _m)
        _ov = _hi - _lo
        if _ov < 300: continue
        _dd = _gy[_lo - _x0:_hi - _x0] - _w["gy"][_lo - _o0:_hi - _o0]
        _rm = float(np.sqrt(np.mean(_dd * _dd)))
        if _rm >= 0.001 * _ov: continue
        if _rm < _best[0]: _best = (_rm, _w["cl"])
    return _best[1]

TEST_SIG = {}
for _p in sorted(glob.glob(f"{_PFTEST}/*__typewell.csv")):
    _wid = os.path.basename(_p).split("__")[0]
    _sig = _v99_match_test(_p)
    TEST_SIG[_wid] = _sig if (_sig in TBL99) else None
_nmap = sum(1 for v in TEST_SIG.values() if v is not None)
print(f"[V109] GR-cluster test map (gate ov>=300 & rmse<0.001*ov): {_nmap}/{len(TEST_SIG)} wells matched to a table "
      f"({time.time()-_t99:.0f}s total)", flush=True)

class _DenseQuadImputer99:
    def __init__(s, well_ids, data_dir, spw=_DQSPW99):
        s.samp = {}; wids = []; cx = []; cy = []
        for wid in well_ids:
            try:
                df = pd.read_csv(f"{data_dir}/{wid}__horizontal_well.csv", usecols=["X", "Y", "ANCC"]).dropna()
            except Exception:
                continue
            if len(df) == 0:
                continue
            ix = np.linspace(0, len(df) - 1, min(spw, len(df)), dtype=int)
            s.samp[wid] = df.iloc[ix][["X", "Y", "ANCC"]].to_numpy(np.float64)
            wids.append(wid); cx.append(float(df["X"].median())); cy.append(float(df["Y"].median()))
        s.wids = wids; s.cxy = np.column_stack([cx, cy])
        s.scale = np.where(s.cxy.std(0) < 1e-3, 1., s.cxy.std(0)); s.tree = cKDTree(s.cxy / s.scale)
    def impute_well(s, well_c, ev_xy, kn_xy, self_wid=None, k=_DQK99):
        _, idx = s.tree.query(well_c / s.scale, k=min(k + 6, len(s.wids)))
        near = [j for j in np.atleast_1d(idx) if s.wids[j] != self_wid][:k]
        if not near:
            return None, None
        Pm = np.concatenate([s.samp[s.wids[j]] for j in near], 0)
        cx, cy = float(well_c[0]), float(well_c[1]); xg = Pm[:, 0]; yg = Pm[:, 1]; a = Pm[:, 2]
        sx = xg.std() + 1e-6; sy = yg.std() + 1e-6
        xs = (xg - cx) / sx; ys = (yg - cy) / sy
        wr = np.sqrt(1.0 / (np.hypot(xs, ys) + 0.1))
        G = np.column_stack([np.ones_like(xs), xs, ys, xs * xs, ys * ys, xs * ys])
        Gw = G * wr[:, None]
        coef, _, _, _ = np.linalg.lstsq(Gw, a * wr, rcond=None)
        resid_std = float(np.sqrt(np.mean((G @ coef - a) ** 2)))
        Gi = np.linalg.inv(Gw.T @ Gw + 1e-9 * np.eye(6))
        def _ev(xy):
            ex = (xy[:, 0] - cx) / sx; ey = (xy[:, 1] - cy) / sy
            gq = np.column_stack([np.ones_like(ex), ex, ey, ex * ex, ey * ey, ex * ey])
            lev = np.sqrt(np.maximum((gq @ Gi * gq).sum(1), 0.0))
            return gq @ coef, resid_std * np.sqrt(1.0 + lev)
        return _ev(ev_xy), _ev(kn_xy)

_t99b = time.time()
_DQ99 = _DenseQuadImputer99([os.path.basename(p).split("__")[0]
                             for p in sorted(glob.glob(f"{_TRAIN99}/*__horizontal_well.csv"))], _TRAIN99)
print(f"[V101] dense-quad imputer: {len(_DQ99.wids)} train wells sampled ({time.time()-_t99b:.0f}s)", flush=True)

def _v101_surf_for(wid, hw, kn_upto, ev_idx):
    """world-frame surface obs + per-row sigma for eval rows; kn_upto = anchor row index.
    Deployable at test: imputer is TRAIN-only; the well itself only contributes X/Y + known-zone datum."""
    try:
        X = hw["X"].values.astype(np.float64); Y = hw["Y"].values.astype(np.float64)
        well_c = np.array([float(np.nanmedian(X)), float(np.nanmedian(Y))])
        kn_xy = np.column_stack([X[:kn_upto + 1], Y[:kn_upto + 1]])
        ev_xy = np.column_stack([X[ev_idx], Y[ev_idx]])
        r_ev, r_kn = _DQ99.impute_well(well_c, ev_xy, kn_xy, self_wid=wid)
        if r_ev is None:
            raise ValueError("no neighbors")
        F_ev, U_ev = r_ev; F_kn, _ = r_kn
        tvt_in = hw["TVT_input"].values.astype(np.float64)[:kn_upto + 1]
        Zk = hw["Z"].values.astype(np.float64)[:kn_upto + 1]
        okk = np.isfinite(tvt_in)
        r = (tvt_in + Zk)[okk] - F_kn[okk]
        c_well = float(np.median(r))
        sig_kn = float(np.clip(np.sqrt(np.mean((r - c_well) ** 2)), 4.0, 40.0))
        sigF = np.clip(np.sqrt(sig_kn ** 2 + U_ev ** 2), 6.0, 80.0).astype(np.float32)
        return (F_ev + c_well).astype(np.float32), sigF, True
    except Exception:
        n = len(ev_idx)
        return np.zeros(n, np.float32), np.full(n, 80.0, np.float32), False

def _bsel_one(wid):
    hw = pd.read_csv(f"{_PFTEST}/{wid}__horizontal_well.csv"); tw = pd.read_csv(f"{_PFTEST}/{wid}__typewell.csv")
    kn = hw[hw["TVT_input"].notna()]; ev = hw[hw["TVT_input"].isna()]
    if len(ev) == 0: return wid, np.zeros(0,np.float32), _SELG, 0.0
    last_tvt = float(kn.iloc[-1]["TVT_input"]); tw_s = tw.sort_values("TVT")
    tw_tvt = tw_s["TVT"].values.astype(np.float32); tw_gr = tw_s["GR"].fillna(tw_s["GR"].mean()).values.astype(np.float32)
    gr_all = hw["GR"].interpolate(limit_direction="both").fillna(tw_gr.mean()).values.astype(np.float32); hgr = gr_all[ev.index]
    bm = np.stack([beam_search(hgr,tw_tvt,tw_gr,last_tvt,bs,mc,es,r) for (bs,mc,es,r) in _BEAMC],0).mean(0)
    em = hw["TVT_input"].isna().to_numpy(); n_eval = float(em.sum())
    z = hw.loc[em,"Z"].values.astype(float); z_span = float(np.nanmax(z)-np.nanmin(z)) if len(z) else 0.0
    nb_ = int(n_eval > _SELT); zb = int(np.searchsorted(_SELZ,z_span,side="right"))
    return wid, bm, _SELV.get(nb_+2*zb,_SELG), last_tvt

def _pv(name):
    parts = name.split("_"); sc = float(parts[2]); bw = 0.0; hd = 0.0
    if "beam" in parts: bw = float(parts[parts.index("beam")+1])
    if "hold" in parts: hd = float(parts[parts.index("hold")+1])
    return sc,bw,hd

def _av(name,pbs,beam,lkt):
    sc,bw,hd = _pv(name); base_ = pbs.get(f"pf_scale_{sc:g}", pbs["pf_scale_8"])
    pred = (1.0-bw)*base_ + bw*beam; return (1.0-hd)*pred + hd*lkt

def _prep(wid):
    hw = pd.read_csv(f"{_PFTEST}/{wid}__horizontal_well.csv"); tw = pd.read_csv(f"{_PFTEST}/{wid}__typewell.csv")
    if hw["TVT_input"].isna().sum() == 0 or hw["TVT_input"].notna().sum() < 10: return None
    if len(tw) < 3 or tw["TVT"].notna().sum() < 3: return None
    tw_s = tw.sort_values("TVT"); tw_tvt = tw_s["TVT"].values.astype(np.float32); tw_gr = tw_s["GR"].fillna(tw_s["GR"].mean()).values.astype(np.float32)
    kn = hw[hw["TVT_input"].notna()]; ev = hw[hw["TVT_input"].isna()]; last = kn.iloc[-1]
    tw_at_k = np.interp(kn["TVT_input"].values,tw_tvt,tw_gr); gs = float(np.clip(np.nanstd(kn["GR"].fillna(0).values-tw_at_k),10.,60.))
    tail = kn.tail(30); dt = np.diff(tail["TVT_input"].values); dz = np.diff(tail["Z"].values); dm = np.diff(tail["MD"].values); m = dm > 0
    ir = float(np.median((dt+dz)[m]/dm[m])) if m.sum() >= 3 else 0.0
    gr_v = hw["GR"].interpolate(limit_direction="both").fillna(tw_gr.mean()).values.astype(np.float32)[ev.index]
    ev_ix = ev.index.values.astype(np.int64)
    surf, sigf, has_s = _v101_surf_for(wid, hw, int(kn.index[-1]), ev_ix)
    d = dict(wid=wid, md=ev["MD"].values.astype(np.float32), z=ev["Z"].values.astype(np.float32), gr=gr_v,
             tw_tvt=tw_tvt, tw_gr=tw_gr, ls=float(last["TVT_input"])+float(last["Z"]), ir=ir, gs=gs,
             last_md=float(last["MD"]), ne=int(len(ev)), ev_idx=ev_ix,
             surf=surf, sigf=sigf, has_surf=has_s)
    _tb = TBL99.get(TEST_SIG.get(wid))
    if _tb is not None:
        d.update(tb_t0=_tb["t0"], tb_nb=_tb["nb"], tb_bias=_tb["bias"], tb_sd=_tb["sd"],
                 tb_valid=_tb["valid"], has_tb=True)
    return d

# Parallel run of beam search selector on test wells
print("[v86] running beam search selector...", flush=True)
_tw = [p.stem.replace("__horizontal_well", "") for p in test_paths]
_bsel = {wid:(bm,var,lt) for wid,bm,var,lt in Parallel(n_jobs=4,prefer="processes")(delayed(_bsel_one)(w) for w in _tw)}
_ws = [d for d in (_prep(w) for w in _tw) if d is not None]; _ws.sort(key=lambda d:d["ne"])

_pf = {}
for _c in range(0, len(_ws), _PFCW):
    _pf.update(_pf_mp(_ws[_c:_c+_PFCW], _PROFS))
    if _PFDEV == "cuda": torch.cuda.synchronize()

PFV = {pid:{} for pid in _PIDS}; PFSTD = {pid:{} for pid in _PIDS}
for w in _ws:
    bm,var,lt = _bsel[w["wid"]]; cell = _pf[w["wid"]]; pbs = cell["mean"]; pstd = cell["std"]; ne = w["ne"]
    sc,_,_ = _pv(var); skey = f"pf_scale_{sc:g}"; skey = skey if skey in pstd else "pf_scale_8"
    rids = [f"{w['wid']}_{int(i)}" for i in w["ev_idx"]]
    for pi,pid in enumerate(_PIDS):
        pbs_p = {k:v[pi,:ne] for k,v in pbs.items()}
        fin = _av(var,pbs_p,bm,lt)
        sdp = pstd[skey][pi,:ne]
        for r_,fv,sv in zip(rids,fin,sdp):
            PFV[pid][r_] = float(fv)
            PFSTD[pid][r_] = float(sv)
del _pf; gc.collect()

for pid in _PFMEM:
    _abs = test_df["id"].astype(str).map(PFV[pid]).to_numpy(np.float64)
    _m = ~np.isfinite(_abs)
    if _m.any():
        _abs[_m] = base_te[_m]
    test_preds[f"pf_{pid}"] = _abs - base_te

_RAWPIDS = ["BIOU00", "BIOUP06", "BIOUP42", "BIOUP51", "OU_P00", "OU_P06", "OU_P23", "OU_P26"]
for pid in _RAWPIDS:
    _abs = test_df["id"].astype(str).map(PFV[pid]).to_numpy(np.float64)
    _m = ~np.isfinite(_abs)
    if _m.any():
        _abs[_m] = base_te[_m]
    test_preds[f"raw_{pid}"] = _abs - base_te

class _DTCNpfb(_tnn.Module):

    def __init__(s, in_feats, hid=64, n_blocks=10, k=3, act="gelu", n_form_pw=0,
                 use_tcn=True, use_rnn=False, bidir=False, is_lstm=False, rnn_layers=1):
        super().__init__()
        s.n_form_pw = int(n_form_pw); s.n_seq = in_feats - s.n_form_pw
        s.act = _F.silu if act == "silu" else _F.gelu
        s.use_tcn = use_tcn; s.use_rnn = use_rnn
        s.inp = _tnn.Conv1d(s.n_seq,hid,1)
        s.b1 = _tnn.ModuleList()
        s.b2 = _tnn.ModuleList()
        for i in range(n_blocks):
            d = 2**i; pad = d*(k-1)//2
            s.b1.append(_tnn.Conv1d(hid,hid,k,padding=pad,dilation=d))
            s.b2.append(_tnn.Conv1d(hid,hid,k,padding=pad,dilation=d))
        s.rnn = None; s.rnn_proj = None
        if use_rnn:
            cls = _tnn.LSTM if is_lstm else _tnn.GRU
            s.rnn = cls(hid, hid, num_layers=rnn_layers, batch_first=True, bidirectional=bidir)
            if bidir: s.rnn_proj = _tnn.Conv1d(2 * hid, hid, 1)
        if s.n_form_pw > 0:
            s.out1 = _tnn.Conv1d(hid + s.n_form_pw, hid, 1); s.out2 = _tnn.Conv1d(hid, 1, 1)
        else:
            s.out = _tnn.Conv1d(hid,1,1)
        s.drop = _tnn.Dropout1d(0.1)
    def forward(s,x):
        a = s.act; xf = None
        if s.n_form_pw > 0:
            xf = x[:, s.n_seq:]; x = x[:, :s.n_seq]
        h = a(s.inp(x))
        if s.use_tcn:
            for c1,c2 in zip(s.b1,s.b2):
                r = h; h = s.drop(a(c1(h))); h = a(c2(h)); h = h + r
        if s.use_rnn:
            hs = h.transpose(1, 2); hs, _ = s.rnn(hs); h = hs.transpose(1, 2)
            if s.rnn_proj is not None: h = s.rnn_proj(h)
        if s.n_form_pw > 0:
            return s.out2(a(s.out1(torch.cat([h, xf], dim=1)))).squeeze(1).squeeze(0)
        return s.out(h).squeeze(1).squeeze(0)

def _load_pfbank(p, profs):
    w = _load_v35pfv2(p)
    if w is None: return None
    n = w["feats"].shape[0]; atvt = w["atvt"]; rids_full = [f"{p.stem.replace('__horizontal_well','')}_{int(i)}" for i in range(n)]
    extra = []
    for pid in profs:
        cr = np.zeros(n, np.float32); cs = np.zeros(n, np.float32)
        for i in w["evi"]:
            rid = f"{p.stem.replace('__horizontal_well','')}_{int(i)}"
            v = PFV[pid].get(rid)
            sv = PFSTD[pid].get(rid)
            if v is not None: cr[i] = v - atvt
            if sv is not None: cs[i] = sv
        extra.append(cr); extra.append(cs)
    feats = np.concatenate([w["feats"]] + [e[:,None] for e in extra], axis=1).astype(np.float32)
    return dict(feats=feats, atvt=atvt, pm=w["pm"], evi=w["evi"], rids=w["rids"])

# Skip nn_pfb_all14
for _nm,_ck,_per_seed in [
    ("nn_pfb_set1","nn_ckpt_pfbank_set1.pt",True),
    ("nn_pfb_set2","nn_ckpt_pfbank_set2.pt",True),
    ("nn_pfb_set3","nn_ckpt_pfbank_set3.pt",True),
    ("nn_pfb_fakeaug","nn_ckpt_set1_fakeaug_3seed.pt",True),
    ("nn_grid_b","nn_ckpt_grid_b.pt",True)
]:
    _c = torch.load(loc(_ck), map_location="cpu", weights_only=False)
    _profs = _c["profiles"]
    _nf = _c["config"]["N_FEATS"]
    _dstd = float(_c["norm"]["dtvt_std"])
    _mn = torch.tensor(np.asarray(_c["norm"]["mean"],np.float32),device=DEVICE)
    _sd = torch.tensor(np.asarray(_c["norm"]["std"],np.float32),device=DEVICE)
    _by_seed = {}
    for d in _c["seed_fold_state_dicts"]:
        mm = _DTCNpfb(_nf).to(DEVICE)
        mm.load_state_dict(d["sd"])
        mm.eval()
        _by_seed.setdefault(int(d["seed"]), []).append(mm)

    _seeds = sorted(_by_seed)
    _rows = {sd:{} for sd in _seeds}
    _sk = 0
    with torch.no_grad():
        for p in test_paths:
            w = _load_pfbank(p, _profs)
            if w is None:
                _sk += 1
                continue
            xx = ((torch.tensor(w["feats"], device=DEVICE) - _mn) / _sd).t().unsqueeze(0).contiguous()
            pm = torch.tensor(w["pm"], device=DEVICE)
            for sd in _seeds:
                acc = np.zeros(len(w["evi"]), np.float64)
                for mm in _by_seed[sd]:
                    dn = mm(xx)
                    rec = w["atvt"] + torch.cumsum(dn * _dstd * pm, dim=0)
                    acc += rec.cpu().numpy()[w["evi"]].astype(np.float64)
                acc /= len(_by_seed[sd])
                for r_, v_ in zip(w["rids"], acc):
                    _rows[sd][r_] = v_

    for sd in _seeds:
        _mem = f"{_nm}_s{sd}"
        _abs = test_df["id"].astype(str).map(_rows[sd]).to_numpy(np.float64)
        _m = ~np.isfinite(_abs)
        if _m.any():
            _abs[_m] = base_te[_m]
        test_preds[_mem] = _abs - base_te

print(f"[v86] Finished NN pfbank inference.", flush=True)

LGBPF_ROOT = loc("oof_per_model.parquet", prefer="rogii-v82-lgbpf-sets").parent
SETS = {
    "set1": ["P06", "P23", "P51", "P34"],
    "set2": ["P00", "P47", "P26", "P54", "P37"],
    "set3": ["P15", "P42", "P39", "P29", "P14"],
}

for pid in _PIDS:
    test_df[f"pf_{pid}"] = test_df["id"].astype(str).map(PFV[pid]).to_numpy(np.float64)
    test_df[f"pfstd_{pid}"] = test_df["id"].astype(str).map(PFSTD[pid]).to_numpy(np.float64)

for pid in _PIDS:
    bad = ~np.isfinite(test_df[f"pf_{pid}"])
    if bad.any():
        test_df.loc[bad, f"pf_{pid}"] = base_te[bad]
    bad = ~np.isfinite(test_df[f"pfstd_{pid}"])
    if bad.any():
        test_df.loc[bad, f"pfstd_{pid}"] = 0.0

for sn, pids in SETS.items():
    pc = [f"pf_{p}" for p in pids]
    test_df[f"{sn}_pfmean"] = test_df[pc].mean(axis=1).astype(np.float32)
    test_df[f"{sn}_pfstd"]  = test_df[pc].std(axis=1).astype(np.float32)
    test_df[f"{sn}_pfrng"]  = (test_df[pc].max(axis=1) - test_df[pc].min(axis=1)).astype(np.float32)

for sn, pids in SETS.items():
    member = f"lgbpf_{sn}"
    mdir = LGBPF_ROOT / "models" / member
    fc = json.load(open(mdir / "feature_cols.json", "r"))
    X_te = test_df[fc].to_numpy(np.float32)
    preds = np.zeros(len(X_te), dtype=np.float64)
    for fi in range(5):
        m = joblib.load(mdir / f"fold_{fi}.pkl")
        preds += m.predict(X_te, num_iteration=getattr(m, "best_iteration", -1))
    preds /= 5.0
    test_preds[member] = preds

print(f"[v86] Finished LGBPF test inference.", flush=True)

POOL8 = ["lightgbm-1", "lightgbm-2", "lightgbm-3", "catboost-1", "catboost-2", "catboost-3", "cb_aug", "nn"]
POOL9 = POOL8 + ["v22"]
POOL10 = POOL9 + ["v35"]
POOL11 = POOL10 + ["v35pfv2"]
POOL21 = POOL11 + ["pf_P06", "pf_P23", "pf_P51", "pf_P54", "nn_pfb_set1_s42", "nn_pfb_set1_s1234", "nn_pfb_set1_s7", "nn_pfb_all14"]
SET2 = ["nn_pfb_set2_s42", "nn_pfb_set2_s1234", "nn_pfb_set2_s7"]
SET3 = ["nn_pfb_set3_s42", "nn_pfb_set3_s1234", "nn_pfb_set3_s7"]
POOL27 = POOL21 + SET2 + SET3

BASE_MEMBERS = [m for m in POOL27 if m != "nn_pfb_all14"] + ["lgbpf_set1", "lgbpf_set2", "lgbpf_set3", "nn_pfb_fakeaug_s42", "nn_pfb_fakeaug_s1234", "nn_pfb_fakeaug_s7", "nn_grid_b_s42"]
print(f"[v86] Stacking pool contains {len(BASE_MEMBERS)} members.", flush=True)

FORMPW3_OOF = loc("oof_form_pw_3seed.parquet", prefer="rogii-pfbank-formpw-3seed")
load_pfb(FORMPW3_OOF, "nn_pfb_formpw3")
GRUHEAD_OOF = loc("oof_e5_form_gruhead.parquet", prefer="rogii-gruhead-oof")
load_pfb(GRUHEAD_OOF, "nn_pfb_gruhead")
_NEW = ["nn_pfb_formpw3_s42", "nn_pfb_formpw3_s1234", "nn_pfb_formpw3_s7", "nn_pfb_gruhead_s42"]
_true_res = true - base
for _m in _NEW:
    assert _m in oof_preds, f"missing OOF member {_m}"
    _err = oof_preds[_m] - _true_res
    _cc = sorted(((q, float(np.corrcoef(_err, oof_preds[q] - _true_res)[0,1])) for q in BASE_MEMBERS), key=lambda x:-x[1])
    print(f"[V96] {_m} standalone OOF={np.sqrt(np.mean((oof_preds[_m]+base-true)**2)):.4f} | err-corr max {_cc[0]} min {_cc[-1]}", flush=True)
BASE_MEMBERS = list(BASE_MEMBERS) + _NEW
print(f"[V96] BASE_MEMBERS={len(BASE_MEMBERS)} (31 v94 + 3 formpw3 + 1 gruhead)", flush=True)

load_pfb(loc("oof_m1.parquet", prefer="rogii-oubank-models-oof"), "oubank_model1")
load_pfb(loc("oof_m2.parquet", prefer="rogii-oubank-models-oof"), "oubank_model2")

_rawtr = pd.read_parquet(loc("raw_pf_members.parquet", prefer="rogii-raw-pf-members")).set_index("row_id")
_rawtr.index = _rawtr.index.astype(str)
_rawtr = _rawtr.reindex(ids)
for _pid in ["BIOU00", "BIOUP06", "BIOUP42", "BIOUP51", "OU_P00", "OU_P06", "OU_P23", "OU_P26"]:
    _s = _rawtr[f"pf_{_pid}"].to_numpy(np.float64)
    _bad = ~np.isfinite(_s)
    if _bad.any():
        _s[_bad] = base[_bad]
    oof_preds[f"raw_{_pid}"] = safe_array(_s - base)
del _rawtr; gc.collect()
_V99NEW = ["oubank_model1_s42", "oubank_model2_s42"]
for _m in _V99NEW:
    assert _m in oof_preds, f"missing OOF member {_m}"
    _err = oof_preds[_m] - _true_res
    _cc = sorted(((q, float(np.corrcoef(_err, oof_preds[q] - _true_res)[0,1])) for q in BASE_MEMBERS), key=lambda x:-x[1])
    print(f"[V114] {_m} standalone OOF={np.sqrt(np.mean((oof_preds[_m]+base-true)**2)):.4f} | err-corr max {_cc[0]} min {_cc[-1]}", flush=True)
BASE_MEMBERS = list(BASE_MEMBERS) + _V99NEW
print(f"[V114] BASE_MEMBERS={len(BASE_MEMBERS)} (35 core + model1[B] + model2[C]; raw8 dropped = clean v111, LB 5.897) — P/Q gate at end", flush=True)

_ADIL_OOF = pd.read_parquet(loc("oof_adil151.parquet", prefer="rogii-adil151-oof-probe"))
_as = pd.Series(_ADIL_OOF["pred"].to_numpy(np.float64),
                index=_ADIL_OOF["row_id"].astype(str)).reindex(ids).to_numpy(np.float64)
_abad = ~np.isfinite(_as)
if _abad.any(): _as[_abad] = base[_abad]
oof_preds["adil_exp151"] = safe_array(_as - base)
_aerr = oof_preds["adil_exp151"] - _true_res
_acc = sorted(((q, float(np.corrcoef(_aerr, oof_preds[q] - _true_res)[0,1])) for q in BASE_MEMBERS), key=lambda x:-x[1])
print(f"[V123] adil_exp151 TRAIN OOF loaded, coverage {(~_abad).mean()*100:.1f}%; standalone "
      f"OOF={np.sqrt(np.mean((oof_preds['adil_exp151']+base-true)**2)):.4f} (probe 7.2642) | "
      f"err-corr max {_acc[0]} min {_acc[-1]}", flush=True)
BASE_MEMBERS = list(BASE_MEMBERS) + ["adil_exp151"]
assert len(BASE_MEMBERS) == 38, len(BASE_MEMBERS)
print(f"[V123] BASE_MEMBERS={len(BASE_MEMBERS)} (37 v114 + adil_exp151) -> P=38, Q=37 (both inherit; C-type allowed in Q)", flush=True)

for _fnm, _ffn, _fpref in [("model1_adil4fb", "oof_oubank_model1_adil4fb.parquet", "rogii-adil-oof-all"),
                           ("model2_adil4fb", "oof_oubank_model2_adil4fb.parquet", "rogii-adil-oof-all"),
                           ("model1_adil3fb", "oof_oubank_model1_adil3fb.parquet", "rogii-adil3fb-oof"),
                           ("model2_adil3fb", "oof_oubank_model2_adil3fb.parquet", "rogii-adil3fb-oof")]:
    _fo = pd.read_parquet(loc(_ffn, prefer=_fpref))
    _fs = pd.Series(_fo["pred"].to_numpy(np.float64), index=_fo["row_id"].astype(str)).reindex(ids).to_numpy(np.float64)
    if (~np.isfinite(_fs)).any():
        _fs[~np.isfinite(_fs)] = base[~np.isfinite(_fs)]
    oof_preds[_fnm] = safe_array(_fs - base)
    print(f"[V136] {_fnm} TRAIN OOF standalone={np.sqrt(np.mean((_fs - true) ** 2)):.4f}", flush=True)
BASE_MEMBERS = list(BASE_MEMBERS) + ["model1_adil4fb", "model2_adil4fb", "model1_adil3fb", "model2_adil3fb"]
print(f"[V136] BASE_MEMBERS={len(BASE_MEMBERS)} (38 + 4 fusion = 42)", flush=True)

def group_stats(pm_dict, members):
    cols = [m for m in members if m in pm_dict]
    if not cols:
        return None, None
    M = np.column_stack([pm_dict[m] for m in cols]).astype(np.float64, copy=False)
    return M.mean(1), M.std(1)

def add_col(cols, names, name, values):
    cols.append(np.asarray(values, np.float64))
    names.append(name)

def make_features(pm_dict, blend, md_arr, pfd_arr, members):
    cols, names = [], []
    strong_list = ["nn_pfb_set1_s42", "nn_pfb_set1_s1234", "nn_pfb_set1_s7", "v35", "v35pfv2", "pf_P06"]
    strong = [m for m in strong_list if m in pm_dict]

    add_col(cols, names, "md_since", md_arr)
    add_col(cols, names, "blend", blend)

    all_mean, all_std = group_stats(pm_dict, members)
    _, strong_std = group_stats(pm_dict, strong)
    add_col(cols, names, "spread_all", all_std)
    add_col(cols, names, "spread_strong", strong_std)

    for m in ["pf_P06", "v35", "v35pfv2"]:
        add_col(cols, names, f"d_{m}", pm_dict[m] - blend)
    add_col(cols, names, "d_pfancc", pfd_arr - blend)

    groups = {
        "set1": ["nn_pfb_set1_s42", "nn_pfb_set1_s1234", "nn_pfb_set1_s7"],
        "set2": ["nn_pfb_set2_s42", "nn_pfb_set2_s1234", "nn_pfb_set2_s7"],
        "set3": ["nn_pfb_set3_s42", "nn_pfb_set3_s1234", "nn_pfb_set3_s7"],
        "rawpf": ["pf_P06", "pf_P23", "pf_P51", "pf_P54"],
        "lgbpf": ["lgbpf_set1", "lgbpf_set2", "lgbpf_set3"],
    }
    for gname, gm in groups.items():
        gm = [m for m in gm if m in pm_dict]
        if len(gm) >= 2:
            mn, sd = group_stats(pm_dict, gm)
            add_col(cols, names, f"{gname}_mean_delta", mn - blend)
            add_col(cols, names, f"{gname}_spread", sd)
        elif len(gm) == 1:
            add_col(cols, names, f"{gname}_delta", pm_dict[gm[0]] - blend)

    for m in members:
        if m in {"lightgbm-2", "lightgbm-3", "catboost-1"}:
            continue
        add_col(cols, names, f"mem_{m}", pm_dict[m] - blend)

    F = np.column_stack(cols).astype(np.float64, copy=False)
    return F, names

A_tr = np.column_stack([oof_preds[m] for m in BASE_MEMBERS]).astype(np.float64, copy=False)
K = A_tr.shape[1]
EQ = np.ones(K, dtype=np.float64) / K
S = 0.1

oof_blend = np.zeros(N, dtype=np.float64)
weights = []
for f in range(5):
    tr = fold != f
    va = fold == f
    w = (1.0 - S) * _nnls(A_tr[tr], y_d[tr])[0] + S * EQ
    weights.append(w)
    oof_blend[va] = A_tr[va] @ w

wf = (1.0 - S) * _nnls(A_tr, y_d)[0] + S * EQ
cv_blend = root_mean_squared_error(y_d, oof_blend)
print(f"[v86] Train OOF shrink blend CV refit = {cv_blend:.6f}", flush=True)
del A_tr; gc.collect()

_BM27 = [m for m in BASE_MEMBERS if not m.startswith("nn_pfb_fakeaug")]
_A27 = np.column_stack([oof_preds[m] for m in _BM27]).astype(np.float64, copy=False)
_EQ27 = np.ones(_A27.shape[1], dtype=np.float64) / _A27.shape[1]
_ob27 = np.zeros(N, dtype=np.float64)
for _f in range(5):
    _tr = fold != _f; _va = fold == _f
    _w27 = (1.0 - S) * _nnls(_A27[_tr], y_d[_tr])[0] + S * _EQ27
    _ob27[_va] = _A27[_va] @ _w27
_cv27 = root_mean_squared_error(y_d, _ob27)
print(f"[v93] GATE shrink CV: {len(_BM27)}-member(no fakeaug)={_cv27:.6f} | {len(BASE_MEMBERS)}-member(+3seed fakeaug)={cv_blend:.6f} | delta={cv_blend - _cv27:+.6f}", flush=True)
del _A27, _ob27, _EQ27; gc.collect()

_BM30 = [m for m in BASE_MEMBERS if m != "nn_grid_b_s42"]
_A30 = np.column_stack([oof_preds[m] for m in _BM30]).astype(np.float64, copy=False)
_EQ30 = np.ones(_A30.shape[1], dtype=np.float64) / _A30.shape[1]
_ob30 = np.zeros(N, dtype=np.float64)
for _f in range(5):
    _tr = fold != _f; _va = fold == _f
    _ob30[_va] = _A30[_va] @ ((1.0 - S) * _nnls(_A30[_tr], y_d[_tr])[0] + S * _EQ30)
_cv30 = root_mean_squared_error(y_d, _ob30)
print(f"[v94] GATE shrink CV: 30(no gridB)={_cv30:.6f} | 31(+gridB)={cv_blend:.6f} | delta={cv_blend - _cv30:+.6f}", flush=True)
del _A30, _ob30, _EQ30; gc.collect()

_NEW96 = ["nn_pfb_formpw3_s42", "nn_pfb_formpw3_s1234", "nn_pfb_formpw3_s7", "nn_pfb_gruhead_s42"]
_BM31 = [m for m in BASE_MEMBERS if m not in _NEW96]
_A31 = np.column_stack([oof_preds[m] for m in _BM31]).astype(np.float64, copy=False)
_EQ31 = np.ones(_A31.shape[1], dtype=np.float64) / _A31.shape[1]
_ob31 = np.zeros(N, dtype=np.float64)
for _f in range(5):
    _tr = fold != _f; _va = fold == _f
    _ob31[_va] = _A31[_va] @ ((1.0 - S) * _nnls(_A31[_tr], y_d[_tr])[0] + S * _EQ31)
_cv31 = root_mean_squared_error(y_d, _ob31)
print(f"[V96] GATE shrink CV: {len(_BM31)}(v94 base)={_cv31:.6f} | {len(BASE_MEMBERS)}(+formpw3+gru)={cv_blend:.6f} | delta={cv_blend - _cv31:+.6f}  (<0 = new members help)", flush=True)
del _A31, _ob31, _EQ31; gc.collect()

_BM35 = [m for m in BASE_MEMBERS if m not in _V99NEW]
_A35 = np.column_stack([oof_preds[m] for m in _BM35]).astype(np.float64, copy=False)
_EQ35 = np.ones(_A35.shape[1], dtype=np.float64) / _A35.shape[1]
_ob35 = np.zeros(N, dtype=np.float64)
for _f in range(5):
    _tr = fold != _f; _va = fold == _f
    _ob35[_va] = _A35[_va] @ ((1.0 - S) * _nnls(_A35[_tr], y_d[_tr])[0] + S * _EQ35)
_cv35 = root_mean_squared_error(y_d, _ob35)
print(f"[V103] GATE shrink CV: {len(_BM35)}(v96 base)={_cv35:.6f} | {len(BASE_MEMBERS)}(v103 members)={cv_blend:.6f} | delta={cv_blend - _cv35:+.6f}  (probe v8 winner NNLS ref 6.561054)", flush=True)
del _A35, _ob35, _EQ35; gc.collect()

F_oof, feature_names = make_features(oof_preds, oof_blend, md_since, pf_delta, BASE_MEMBERS)
spread = F_oof[:, feature_names.index("spread_all")].astype(np.float32)
target_resid = y_d - oof_blend
alpha = 30.0

def get_bin_edges(x, q):
    x = np.asarray(x, np.float64)
    edges = np.nanquantile(x, np.linspace(0.0, 1.0, q + 1))
    edges = np.unique(edges[1:-1])
    return edges

def apply_bin_edges(x, edges):
    x = np.asarray(x, np.float64)
    if len(edges) == 0:
        return np.zeros(len(x), dtype=np.int64), 1
    return np.searchsorted(edges, x, side="right").astype(np.int64), len(edges) + 1

print("[v86] Sequential residual calibration in 5-fold CV...", flush=True)

ridge_corr = np.zeros(N, dtype=np.float64)
for f in range(5):
    tr = fold != f
    va = fold == f
    mu = F_oof[tr].mean(0)
    sd = F_oof[tr].std(0) + 1e-9
    model = _Ridge(alpha=alpha).fit((F_oof[tr] - mu) / sd, target_resid[tr])
    ridge_corr[va] = model.predict((F_oof[va] - mu) / sd)

pred = oof_blend.copy()

lam1 = 10000.0
gg1 = 0.85
clip_lo1 = -0.25
clip_hi1 = 1.75

pred_s1 = np.zeros(N, dtype=np.float64)
for f in range(5):
    tr = fold != f
    va = fold == f
    edges = get_bin_edges(spread[tr], 128)
    ktr, n_key = apply_bin_edges(spread[tr], edges)
    kva, _ = apply_bin_edges(spread[va], edges)

    ctr = ridge_corr[tr]
    rtr = target_resid[tr]

    cnt = np.bincount(ktr, minlength=n_key).astype(np.float64)
    num = np.bincount(ktr, weights=ctr * rtr, minlength=n_key).astype(np.float64)
    den = np.bincount(ktr, weights=ctr * ctr, minlength=n_key).astype(np.float64)

    raw = np.full(n_key, gg1, dtype=np.float64)
    ok = den > 1e-9
    raw[ok] = num[ok] / den[ok]

    gam = (cnt * raw + lam1 * gg1) / (cnt + lam1)
    gam = np.clip(gam, clip_lo1, clip_hi1)
    pred_s1[va] = oof_blend[va] + gam[kva] * ridge_corr[va]

pred = pred_s1

cv_final = root_mean_squared_error(y_d, pred)
print(f"[v86] Final honest CV refit after 1 calibration (gamma128; steps 2-4 DROPPED) = {cv_final:.6f}", flush=True)
print(f"[V135][G1] cv_final={cv_final:.6f} (n_members={len(BASE_MEMBERS)}) -> v136 42-member (base37+adil151+4 fusion); probe v40 target 5.804394 (informational, NOT fatal)", flush=True)

# ===== V96 form-aware NN TEST INFERENCE

import time as _time
_t0v = _time.time()
def _form_channels(p):
    hw = _load_v35pfv2_raw(p)
    kn = hw["TVT_input"].notna().values; kni = np.where(kn)[0]; evi = np.where(~kn)[0]
    n = len(hw); x = hw["X"].values.astype(np.float64); y = hw["Y"].values.astype(np.float64); z = hw["Z"].values.astype(np.float64)
    ai = int(kni[-1]); atvt = float(hw["TVT_input"].values[ai]); ktvt = hw["TVT_input"].values[kni].astype(np.float64); kz = z[kni]
    f_ev, knn_d = FI.impute(np.column_stack([x[evi], y[evi]]), self_wid=None)
    f_kn, _ = FI.impute(np.column_stack([x[kni], y[kni]]), self_wid=None)
    tvtfs = []; cols = []
    for fi in range(len(FORMATIONS)):
        bb = seg_b_well(ktvt, kz, f_kn[:, fi])[0]
        tf = -z[evi] + f_ev[:, fi] + bb
        ch = np.zeros(n, np.float32); ch[evi] = np.nan_to_num(tf - atvt, nan=0.0); cols.append(ch); tvtfs.append(tf)
    tvm = np.stack(tvtfs, 1)
    fm = np.zeros(n, np.float32); fm[evi] = np.nan_to_num(tvm.mean(1) - atvt, nan=0.0); cols.append(fm)
    fs = np.zeros(n, np.float32); fs[evi] = np.nan_to_num(tvm.std(1), nan=0.0); cols.append(fs)
    kd = np.zeros(n, np.float32); kd[evi] = np.nan_to_num(knn_d, nan=0.0); cols.append(kd)
    return np.stack(cols, 1).astype(np.float32)
def _detect_arch(sd0, hid=64):
    npw = (int(sd0["out1.weight"].shape[1]) - int(sd0["out1.weight"].shape[0])) if "out1.weight" in sd0 else 0
    rk = [k for k in sd0 if k.startswith("rnn.")]
    use_rnn = len(rk) > 0
    layers = (1 + max([int(k.split("_l")[1].split("_")[0]) for k in rk if "weight_ih_l" in k], default=0)) if use_rnn else 1
    return dict(n_form_pw=npw, use_tcn=("b1.0.weight" in sd0), use_rnn=use_rnn,
                bidir=("rnn.weight_ih_l0_reverse" in sd0),
                is_lstm=(use_rnn and int(sd0["rnn.weight_ih_l0"].shape[0]) == 4 * hid), rnn_layers=layers)
def _v99_sharp_channels(p, w, pids=("BIOU00",)):

    n = w["feats"].shape[0]; atvt = w["atvt"]
    stem = p.stem.replace("__horizontal_well", "")
    cols = []
    for pid in pids:
        cr = np.zeros(n, np.float32); cs = np.zeros(n, np.float32)
        for i in w["evi"]:
            rid = f"{stem}_{int(i)}"
            v = PFV[pid].get(rid); sv = PFSTD[pid].get(rid)
            if v is not None: cr[i] = v - atvt
            if sv is not None: cs[i] = sv
        cols.append(cr); cols.append(cs)
    return np.stack(cols, 1).astype(np.float32)

_BANK4 = ("BIOU00", "BIOUP06", "BIOUP42", "BIOUP51")
_OU4 = ("OU_P00", "OU_P06", "OU_P23", "OU_P26")

V96_NN = [("nn_pfb_formpw3", loc("nn_ckpt_form_pw_3seed.pt"), None),
          ("nn_pfb_gruhead", loc("nn_ckpt_e5_form_gruhead.pt"), None),
          ("oubank_model1", loc("nn_ckpt_m1.pt", prefer="rogii-oubank-ckpts"), "bank_biou4_ou4"),
          ("oubank_model2", loc("nn_ckpt_m2.pt", prefer="rogii-oubank-ckpts"), "ou4")]
for _nm, _ckp, _wsharp in V96_NN:
    _c = torch.load(_ckp, map_location="cpu", weights_only=False)
    _profs = _c["profiles"]; _nf = _c["config"]["N_FEATS"]; _act = _c["config"].get("ACTIVATION", "gelu")
    _dstd = float(_c["norm"]["dtvt_std"])
    _mn = torch.tensor(np.asarray(_c["norm"]["mean"], np.float32), device=DEVICE)
    _sd = torch.tensor(np.asarray(_c["norm"]["std"], np.float32), device=DEVICE)
    _arch = _detect_arch(_c["seed_fold_state_dicts"][0]["sd"])
    _by_seed = {}
    for d in _c["seed_fold_state_dicts"]:
        mm = _DTCNpfb(_nf, act=_act, **_arch).to(DEVICE); mm.load_state_dict(d["sd"]); mm.eval()
        _by_seed.setdefault(int(d["seed"]), []).append(mm)
    print(f"[V96] {_nm}: N_FEATS={_nf} act={_act} arch={_arch} seeds={sorted(_by_seed)}", flush=True)
    _rows = {sd: {} for sd in _by_seed}; _skip = 0
    with torch.no_grad():
        for p in test_paths:
            w = _load_pfbank(p, _profs)
            if w is None: _skip += 1; continue
            if _wsharp == "bank_biou4":
                _bse = w["feats"]; _mid = [_v99_sharp_channels(p, w, _BANK4)]
            elif _wsharp == "bank_biou4_ou4":
                _bse = w["feats"]; _mid = [_v99_sharp_channels(p, w, _BANK4 + _OU4)]
            elif _wsharp == "ou4":
                _bse = w["feats"]; _mid = [_v99_sharp_channels(p, w, _OU4)]
            else:
                _bse = w["feats"]; _mid = []
            feats = np.concatenate([_bse] + _mid + [_form_channels(p)], axis=1).astype(np.float32)
            assert feats.shape[1] == _nf, (feats.shape, _nf)
            xx = ((torch.tensor(feats, device=DEVICE) - _mn) / _sd).t().unsqueeze(0).contiguous()
            pm = torch.tensor(w["pm"], device=DEVICE)
            for sd in _by_seed:
                acc = np.zeros(len(w["evi"]), np.float64)
                for mm in _by_seed[sd]:
                    dn = mm(xx); rec = w["atvt"] + torch.cumsum(dn * _dstd * pm, dim=0)
                    acc += rec.cpu().numpy()[w["evi"]].astype(np.float64)
                acc /= len(_by_seed[sd])
                for r_, v_ in zip(w["rids"], acc): _rows[sd][r_] = v_
    for sd in _by_seed:
        _mem = f"{_nm}_s{sd}"
        _abs = test_df["id"].astype(str).map(_rows[sd]).to_numpy(np.float64)
        _m = ~np.isfinite(_abs)
        if _m.any(): _abs[_m] = base_te[_m]
        test_preds[_mem] = safe_array(_abs - base_te)
        print(f"[V96] {_mem} test coverage {(~_m).mean()*100:.1f}% (skip {_skip})", flush=True)
print(f"[V96] form-aware NN test inference done {_time.time()-_t0v:.0f}s", flush=True)

for _ak, _av in [("ROGII_SAVGOL", "31"), ("ROGII_SW_TSTEP", "1.0"), ("ROGII_SW_THALF", "128"),
                 ("ROGII_SW_SCALE", "50"), ("ROGII_SW_LOC", "1"), ("ROGII_SW_GEO", "1"),
                 ("ROGII_SW_MISFIT", "1"), ("ROGII_CHUNK_W", "48"), ("ROGII_CHUNK_HIST", "8")]:
    os.environ.setdefault(_ak, _av)
import sys as _sys
_a_src_hits = (glob.glob("/kaggle/input/rogii-artifacts/src/data.py")
               + glob.glob("/kaggle/input/datasets/*/*/src/data.py")
               + glob.glob("/kaggle/input/*/src/data.py")
               + glob.glob("/kaggle/input/*/*/src/data.py"))
assert _a_src_hits, "adil rogii-artifacts not found"
_A_ART = os.path.dirname(os.path.dirname(_a_src_hits[0]))
_sys.path.insert(0, _A_ART)
_sys.path.insert(0, f"{_A_ART}/src")
import data as _a_data
_a_data.RAW = str(CFG.dataset_path)
_a_data.CACHE = "/tmp/derived"
import render as _a_render
import render_sw as _a_render_sw
from rollout2 import rollout_well as _a_rollout
from unet import GeoSteerNet as _AGeoSteerNet
assert _a_render_sw.N_CH == 9, _a_render_sw.N_CH
_A_MODELS = []
for _f in range(5):
    _am = _AGeoSteerNet(base_ch=32, in_ch=_a_render_sw.N_CH, out_clamp=_a_render_sw.SDF_CLAMP,
                        prox_tau=0.5, n_modes=4)
    _am.load_state_dict(torch.load(f"{_A_ART}/models/exp_151_fold{_f}.pt", map_location=DEVICE))
    _am.to(DEVICE).eval()
    _A_MODELS.append(_am)
print(f"[V123] adil exp_151: 5 fold-models loaded (N_CH={_a_render_sw.N_CH})", flush=True)
_t0a = _time.time()
_a_rows = {}
_a_fb = 0
with torch.no_grad():
    for p in test_paths:
        _wid = p.stem.replace("__horizontal_well", "")
        _hw = pd.read_csv(p)
        _ev = _hw.index[_hw["TVT_input"].isna()].to_numpy()
        if len(_ev) == 0:
            continue
        _anch = float(_hw["TVT_input"].dropna().iloc[-1]) if _hw["TVT_input"].notna().any() else float(_hw["Z"].median())
        try:
            _wd = _a_render.load_well_data(_wid, split="test")
            if not np.isfinite(_wd["h_gr_smooth"]).all():
                _fill = float(np.nanmedian(_wd["t_gr"])) if np.isfinite(_wd["t_gr"]).any() else 0.0
                _wd["h_gr_smooth"] = np.nan_to_num(_wd["h_gr_smooth"], nan=_fill)
            _full = _a_rollout(_wd, _A_MODELS, DEVICE, 48, 8, stride_frac=0.5, rebase_frac=0.75,
                               tri=True, tau=0.2, lam=1.1, K=4, logits=False)
            _ap = np.asarray(_full[_ev], dtype=float)
            if not np.isfinite(_ap).all():
                _ap = np.where(np.isfinite(_ap), _ap, _anch)
        except Exception as _ae:
            print(f"[V123] WARN {_wid}: {type(_ae).__name__}: {_ae} -> anchor fallback", flush=True)
            _ap = np.full(len(_ev), _anch)
            _a_fb += 1
        for _i, _v in zip(_ev, _ap):
            _a_rows[f"{_wid}_{int(_i)}"] = float(_v)
_a_abs = test_df["id"].astype(str).map(_a_rows).to_numpy(np.float64)
_a_miss = ~np.isfinite(_a_abs)
if _a_miss.any():
    _a_abs[_a_miss] = base_te[_a_miss]
test_preds["adil_exp151"] = safe_array(_a_abs - base_te)
print(f"[V123] adil_exp151 test coverage {(~_a_miss).mean()*100:.1f}% ({_a_fb} fallback wells, "
      f"{_time.time()-_t0a:.0f}s)", flush=True)
del _A_MODELS
gc.collect()
if DEVICE == "cuda": torch.cuda.empty_cache()

_adil_abs = {"exp_151": _a_rows}
def _adil_roll(exp):
    _ms = []
    for _f in range(5):
        _m = _AGeoSteerNet(base_ch=32, in_ch=_a_render_sw.N_CH, out_clamp=_a_render_sw.SDF_CLAMP, prox_tau=0.5, n_modes=4)
        _m.load_state_dict(torch.load(f"{_A_ART}/models/{exp}_fold{_f}.pt", map_location=DEVICE)); _m.to(DEVICE).eval(); _ms.append(_m)
    _rr = {}; _fb = 0; _t = _time.time()
    with torch.no_grad():
        for p in test_paths:
            _wid = p.stem.replace("__horizontal_well", ""); _hw = pd.read_csv(p); _ev = _hw.index[_hw["TVT_input"].isna()].to_numpy()
            if len(_ev) == 0:
                continue
            _anch = float(_hw["TVT_input"].dropna().iloc[-1]) if _hw["TVT_input"].notna().any() else float(_hw["Z"].median())
            try:
                _wd = _a_render.load_well_data(_wid, split="test")
                if not np.isfinite(_wd["h_gr_smooth"]).all():
                    _fill = float(np.nanmedian(_wd["t_gr"])) if np.isfinite(_wd["t_gr"]).any() else 0.0
                    _wd["h_gr_smooth"] = np.nan_to_num(_wd["h_gr_smooth"], nan=_fill)
                _full = _a_rollout(_wd, _ms, DEVICE, 48, 8, stride_frac=0.5, rebase_frac=0.75, tri=True, tau=0.2, lam=1.1, K=4, logits=False)
                _ap = np.asarray(_full[_ev], dtype=float)
                if not np.isfinite(_ap).all():
                    _ap = np.where(np.isfinite(_ap), _ap, _anch)
            except Exception as _e:
                print(f"[V136] WARN adil {exp} {_wid}: {type(_e).__name__}: {_e}", flush=True); _ap = np.full(len(_ev), _anch); _fb += 1
            for _i, _v in zip(_ev, _ap):
                _rr[f"{_wid}_{int(_i)}"] = float(_v)
    del _ms; gc.collect()
    if DEVICE == "cuda": torch.cuda.empty_cache()
    print(f"[V136] adil {exp} rolled ({_fb} fb, {_time.time()-_t:.0f}s)", flush=True)
    return _rr
for _ex in ["exp_158_apc", "exp_165_burst", "exp_135", "exp_143"]:
    _adil_abs[_ex] = _adil_roll(_ex)

def _fusion_infer(member, ckpt_file, sharp_cols, adil_exps):
    _ckpt = loc(ckpt_file, prefer="rogii-adil3fb-ckpts")
    _c = torch.load(_ckpt, map_location="cpu", weights_only=False)
    _profs = _c["profiles"]; _nf = _c["config"]["N_FEATS"]; _act = _c["config"].get("ACTIVATION", "gelu")
    _dstd = float(_c["norm"]["dtvt_std"])
    _mn = torch.tensor(np.asarray(_c["norm"]["mean"], np.float32), device=DEVICE)
    _sd = torch.tensor(np.asarray(_c["norm"]["std"], np.float32), device=DEVICE)
    _arch = _detect_arch(_c["seed_fold_state_dicts"][0]["sd"])
    _byseed = {}
    for d in _c["seed_fold_state_dicts"]:
        mm = _DTCNpfb(_nf, act=_act, **_arch).to(DEVICE); mm.load_state_dict(d["sd"]); mm.eval(); _byseed.setdefault(int(d["seed"]), []).append(mm)
    _na = len(adil_exps); _rows = {sd: {} for sd in _byseed}; _sk = 0
    with torch.no_grad():
        for p in test_paths:
            w = _load_pfbank(p, _profs)
            if w is None:
                _sk += 1; continue
            _base_sharp = np.concatenate([w["feats"], _v99_sharp_channels(p, w, sharp_cols)], axis=1).astype(np.float32)
            _formc = _form_channels(p).astype(np.float32)
            _ca = np.zeros((_base_sharp.shape[0], _na), np.float32); _atvt = w["atvt"]
            for _j, _i in enumerate(w["evi"]):
                _rid = w["rids"][_j]
                for _k, _ex in enumerate(adil_exps):
                    _vv = _adil_abs[_ex].get(_rid)
                    if _vv is not None and np.isfinite(_vv):
                        _ca[_i, _k] = _vv - _atvt

            feats = np.concatenate([_base_sharp, _ca, _formc], axis=1).astype(np.float32)
            if feats.shape[1] != _nf:
                print(f"[V136] WARN {member} feats {feats.shape[1]} != {_nf} skip", flush=True); _sk += 1; continue
            xx = ((torch.tensor(feats, device=DEVICE) - _mn) / _sd).t().unsqueeze(0).contiguous()
            pm = torch.tensor(w["pm"], device=DEVICE)
            for sd in _byseed:
                acc = np.zeros(len(w["evi"]), np.float64)
                for mm in _byseed[sd]:
                    dn = mm(xx); rec = w["atvt"] + torch.cumsum(dn * _dstd * pm, dim=0); acc += rec.cpu().numpy()[w["evi"]].astype(np.float64)
                acc /= len(_byseed[sd])
                for r_, v_ in zip(w["rids"], acc):
                    _rows[sd][r_] = v_
    _sd0 = sorted(_byseed)[0]
    _abs = test_df["id"].astype(str).map(_rows[_sd0]).to_numpy(np.float64)
    _miss = ~np.isfinite(_abs)
    if _miss.any():
        _abs[_miss] = base_te[_miss]
    test_preds[member] = safe_array(_abs - base_te)
    print(f"[V136] {member} test coverage {(~_miss).mean()*100:.1f}% (N_FEATS={_nf}, skip {_sk})", flush=True)
    del _byseed; gc.collect()
    if DEVICE == "cuda": torch.cuda.empty_cache()

_A4 = ["exp_158_apc", "exp_151", "exp_135", "exp_143"]
_A3 = ["exp_151", "exp_158_apc", "exp_165_burst"]
_fusion_infer("model1_adil4fb", "nn_ckpt_oubank_model1_adil4fb.pt", list(_BANK4) + list(_OU4), _A4)
_fusion_infer("model2_adil4fb", "nn_ckpt_oubank_model2_adil4fb.pt", list(_OU4), _A4)
_fusion_infer("model1_adil3fb", "nn_ckpt_oubank_model1_adil3fb.pt", list(_BANK4) + list(_OU4), _A3)
_fusion_infer("model2_adil3fb", "nn_ckpt_oubank_model2_adil3fb.pt", list(_OU4), _A3)
del _adil_abs; gc.collect()
if DEVICE == "cuda": torch.cuda.empty_cache()

_miss_bm = [m for m in BASE_MEMBERS if m not in test_preds]
if _miss_bm:
    print(f"[V136] WARN test_preds missing {_miss_bm} (blend fill/fail per member)", flush=True)

mu = F_oof.mean(0)
sd = F_oof.std(0) + 1e-9
model_full = _Ridge(alpha=alpha).fit((F_oof - mu) / sd, target_resid)

A_te = np.column_stack([test_preds[m] for m in BASE_MEMBERS]).astype(np.float64, copy=False)
test_blend = A_te @ wf

pf_te = test_df["pf_ancc"].to_numpy(np.float64) - base_te
F_te, _ = make_features(test_preds, test_blend, md_te, pf_te, BASE_MEMBERS)
spread_te = F_te[:, feature_names.index("spread_all")].astype(np.float32)

F_te_norm = (F_te - mu) / sd
ridge_corr_te = model_full.predict(F_te_norm)

pred_tr = oof_blend.copy()
pred_te = test_blend.copy()

edges1 = get_bin_edges(spread, 128)
ktr, n_key = apply_bin_edges(spread, edges1)
kte, _ = apply_bin_edges(spread_te, edges1)

cnt = np.bincount(ktr, minlength=n_key).astype(np.float64)
num = np.bincount(ktr, weights=ridge_corr * target_resid, minlength=n_key).astype(np.float64)
den = np.bincount(ktr, weights=ridge_corr * ridge_corr, minlength=n_key).astype(np.float64)

raw = np.full(n_key, gg1, dtype=np.float64)
ok = den > 1e-9
raw[ok] = num[ok] / den[ok]

gam = (cnt * raw + lam1 * gg1) / (cnt + lam1)
gam = np.clip(gam, clip_lo1, clip_hi1)

pred_tr = oof_blend + gam[ktr] * ridge_corr
pred_te = test_blend + gam[kte] * ridge_corr_te

test_df = test_df.copy()
test_df["pred"] = base_te + pred_te

for _, gg in test_df.copy().groupby("well", sort=False):
    v = gg["pred"].values
    n = len(v)
    wl = min(17, n)
    if wl % 2 == 0:
        wl -= 1
    if wl >= 5:
        test_df.loc[gg.index, "pred"] = _sg(v, wl, 3)

import hedge as _a_hedge
_HIDX = _a_hedge.build_train_index(str(CFG.dataset_path / "train"))
_n_hedged = 0
_ids_hg = test_df["id"].astype(str)
for _whg in test_wids:
    try:
        _hwg = pd.read_csv(CFG.dataset_path / "test" / f"{_whg}__horizontal_well.csv")
        _dup, _trtvt = _a_hedge.find_duplicate(_hwg, str(CFG.dataset_path / "train"), _HIDX)
    except Exception as _hge:
        print(f"[V135][HEDGE] WARN {_whg}: {type(_hge).__name__}: {_hge} (ignored)", flush=True)
        continue
    if _dup is None:
        continue
    _evg = _hwg.index[_hwg["TVT_input"].isna()].to_numpy()
    _map_hg = {f"{_whg}_{int(_i)}": float(_v) for _i, _v in zip(_evg, _trtvt)}
    _mrows = _ids_hg.map(_map_hg)
    _mask_hg = _mrows.notna().to_numpy()
    if not _mask_hg.any():
        continue
    test_df.loc[_mask_hg, "pred"] = 0.45 * _mrows[_mask_hg].to_numpy(np.float64) \
        + 0.55 * test_df.loc[_mask_hg, "pred"].to_numpy(np.float64)
    _n_hedged += 1
    print(f"[V135][HEDGE] {_whg} == train {_dup} -> 0.45/0.55 blend ({int(_mask_hg.sum())} rows)", flush=True)
print(f"[V135][HEDGE] hedged {_n_hedged}/{len(test_wids)} test wells", flush=True)

sample = pd.read_csv(CFG.dataset_path / "sample_submission.csv")
sub = sample[["id"]].merge(test_df[["id", "pred"]].rename(columns={"pred": "tvt"}), on="id", how="left")
_fill = float(base.mean() + y_d.mean())
sub["tvt"] = sub["tvt"].fillna(_fill)
sub[["id", "tvt"]].to_csv("submission.csv", index=False)
print(f"[v86] Written submission.csv: {len(sub)} rows, NaNs filled: {sub['tvt'].isna().sum()}", flush=True)
