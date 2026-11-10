"""PF-ANCC прайор (порт из паблик-решения 7.159, fork aevionlabs).

Частичный фильтр в geo_z-пространстве (pos = tvt + z — рамка hengck 702474):
динамика = random-walk наклона (ALPHA-затухание) + позиц. шум; лайклихуд = гаусс
GR-мисфита против typewell-грида; систематический ресэмплинг по ESS.
Выход per-ft: постериорные mean(TVT) и std(TVT). Отличие от бимов: калиброванная
НЕОПРЕДЕЛЁННОСТЬ (std растёт в ambiguity-зонах) — рендерится как ширина полосы.

Для скорости в трейне: N=200 частиц + MD-подвыборка x4 (~10мс/скважина).
"""
import numpy as np
from numba import njit

import os as _os
ALPHA = float(_os.environ.get("PF_ALPHA", "0.998"))
RN = float(_os.environ.get("PF_RN", "0.002"))
PN = float(_os.environ.get("PF_PN", "0.005"))
IS = float(_os.environ.get("PF_IS", "0.3"))
RP = 0.1
RR = 0.001
RESAMP = 0.5
GR_SIG_MIN = float(_os.environ.get("PF_SIGMIN", "10.0"))
GR_SIG_MAX = float(_os.environ.get("PF_SIGMAX", "60.0"))
GR_SIG_DEF = 30.0

@njit(cache=True)
def _interp1(grid, v, vmin, step):
    i = int((v - vmin) / step)
    if i < 0:
        return grid[0]
    n = len(grid) - 1
    if i >= n:
        return grid[n]
    t = (v - vmin) / step - i
    return grid[i] * (1.0 - t) + grid[i + 1] * t

@njit(cache=True)
def _resamp(pos, aux, w, N, rp, rv):
    cum = np.zeros(N + 1)
    for j in range(N):
        cum[j + 1] = cum[j] + w[j]
    u0 = np.random.uniform(0.0, 1.0 / N)
    np2 = np.empty(N)
    na = np.empty(N)
    ci = 0
    for j in range(N):
        u = u0 + j / N
        while ci < N - 1 and cum[ci + 1] < u:
            ci += 1
        np2[j] = pos[ci] + rp * np.random.randn()
        na[j] = aux[ci] + rv * np.random.randn()
    return np2, na

@njit(cache=True)
def _pf_ancc(md_v, z_v, gr_v, gg, vmin, step, gs, ls, ir, N, seed, is_):

    np.random.seed(seed)
    pos = np.empty(N)
    rate = np.empty(N)
    w = np.ones(N) / N
    for j in range(N):
        pos[j] = ls + is_ * np.random.randn()
        rate[j] = ir + 0.01 * np.random.randn()
    pts = np.empty(len(md_v))
    std_ = np.empty(len(md_v))
    log_lik = 0.0
    pm = md_v[0] - 1.0
    for i in range(len(md_v)):
        dm = md_v[i] - pm
        dm = max(dm, 1.0)
        for j in range(N):
            rate[j] = ALPHA * rate[j] + RN * np.random.randn()
            pos[j] += rate[j] * dm + PN * np.random.randn()
            tvt_j = pos[j] - z_v[i]
            tvt_j = max(tvt_j, vmin - 100.0)
            tvt_j = min(tvt_j, vmin + len(gg) * step + 100.0)
            pos[j] = tvt_j + z_v[i]
        if not np.isnan(gr_v[i]):
            ws = 0.0
            for j in range(N):
                eg = _interp1(gg, pos[j] - z_v[i], vmin, step)
                d = (gr_v[i] - eg) / gs
                lk = max(np.exp(-0.5 * d * d) if d * d < 600.0 else 0.0, 1e-300)
                w[j] *= lk
                ws += w[j]

            log_lik += np.log(max(ws, 1e-300))
            if ws > 0.0:
                for j in range(N):
                    w[j] /= ws
            else:
                for j in range(N):
                    w[j] = 1.0 / N
        ne = 0.0
        for j in range(N):
            ne += w[j] * w[j]
        if 1.0 / ne < RESAMP * N:
            pos, rate = _resamp(pos, rate, w, N, RP, RR)
            for j in range(N):
                w[j] = 1.0 / N
        tv = 0.0
        for j in range(N):
            tv += w[j] * (pos[j] - z_v[i])
        pts[i] = tv
        va = 0.0
        for j in range(N):
            va += w[j] * (pos[j] - z_v[i] - tv) ** 2
        std_[i] = va ** 0.5
        pm = md_v[i]
    return pts, std_, log_lik

@njit(cache=True)
def _pf_shape(md_v, z_v, grd_v, ggd, vmin, step, gs, ls, ir, N):
    """PF с likelihood по ПРОИЗВОДНОЙ GR (форма, не уровень) — иная структура ошибки."""
    pos = np.empty(N); rate = np.empty(N); w = np.ones(N) / N
    for j in range(N):
        pos[j] = ls + IS * np.random.randn(); rate[j] = ir + 0.01 * np.random.randn()
    pts = np.empty(len(md_v)); std_ = np.empty(len(md_v)); pm = md_v[0] - 1.0
    for i in range(len(md_v)):
        dm = max(md_v[i] - pm, 1.0)
        for j in range(N):
            rate[j] = ALPHA * rate[j] + RN * np.random.randn()
            pos[j] += rate[j] * dm + PN * np.random.randn()
            tvt_j = pos[j] - z_v[i]
            tvt_j = max(tvt_j, vmin - 50.0); tvt_j = min(tvt_j, vmin + len(ggd) * step + 50.0)
            pos[j] = tvt_j + z_v[i]
        if not np.isnan(grd_v[i]):
            ws = 0.0
            for j in range(N):
                eg = _interp1(ggd, pos[j] - z_v[i], vmin, step)
                d = (grd_v[i] - eg) / gs
                lk = max(np.exp(-0.5 * d * d) if d * d < 600.0 else 0.0, 1e-300)
                w[j] *= lk; ws += w[j]
            if ws > 0.0:
                for j in range(N): w[j] /= ws
            else:
                for j in range(N): w[j] = 1.0 / N
        ne = 0.0
        for j in range(N): ne += w[j] * w[j]
        if 1.0 / ne < RESAMP * N:
            pos, rate = _resamp(pos, rate, w, N, RP, RR)
            for j in range(N): w[j] = 1.0 / N
        tv = 0.0
        for j in range(N): tv += w[j] * (pos[j] - z_v[i])
        pts[i] = tv
        va = 0.0
        for j in range(N): va += w[j] * (pos[j] - z_v[i] - tv) ** 2
        std_[i] = va ** 0.5; pm = md_v[i]
    return pts, std_

def _prep_ancc(wd, start_ft, end_ft, sub):
    """Общая подготовка PF: якорь ls, наклон ir, gr_sig, typewell-грид, подвыборка MD."""
    tvt = wd["h_tvt_true"]
    z = wd["h_z"]
    md = wd["h_md"]
    gr = wd["h_gr_rawi"]
    ls = float(tvt[start_ft] + z[start_ft])
    a0 = max(0, start_ft - 30)
    dt = np.diff(tvt[a0:start_ft + 1])
    dzz = np.diff(z[a0:start_ft + 1])
    dmm = np.diff(md[a0:start_ft + 1])
    m = dmm > 0
    ir = float(np.median((dt + dzz)[m] / dmm[m])) if m.sum() >= 3 else 0.0

    order = np.argsort(wd["t_tvt"])
    tt, tg = wd["t_tvt"][order].astype(np.float64), wd["t_gr"][order].astype(np.float64)
    sim = np.interp(tvt[a0:start_ft], tt, tg)
    if start_ft - a0 >= 20:
        gs = float(np.clip(np.std(gr[a0:start_ft] - sim), GR_SIG_MIN, GR_SIG_MAX))
    else:
        gs = GR_SIG_DEF

    step = 0.5
    tvt_g = np.arange(tt.min(), tt.max() + step, step)
    gg = np.interp(tvt_g, tt, tg).astype(np.float64)
    vmin = float(tt.min())
    idx = np.arange(start_ft, end_ft, sub)
    return (md[idx].astype(np.float64), z[idx].astype(np.float64), gr[idx].astype(np.float64),
            gg, vmin, step, gs, ls, ir, idx, a0)

def pf_prior_tvt(wd, start_ft, end_ft, n_particles=200, sub=4, seed=0):
    """PF-ANCC от start_ft до end_ft скважины wd. Возвращает (mean_tvt, std_tvt) длиной end-start (per-ft).
    start_ft — якорь (last known): инициализация ls=tvt+z в этой точке, наклон ir по последним 30 точкам."""
    md_s, z_s, gr_s, gg, vmin, step, gs, ls, ir, idx, a0 = _prep_ancc(wd, start_ft, end_ft, sub)
    _mode = _os.environ.get("PF_MODE", "ancc")
    if _mode in ("ir0", "wide", "shape"):
        ir = 0.0
    if _mode == "shape":
        gr = wd["h_gr_rawi"]
        grd = np.gradient(gr).astype(np.float64)
        ggd = np.gradient(gg).astype(np.float64)
        gsd = float(np.clip(np.std(np.gradient(gr[a0:start_ft])) if start_ft - a0 >= 20 else 3.0, 1.0, 15.0))
        pts, std_ = _pf_shape(md_s, z_s, grd[idx], ggd, vmin, step, gsd, ls, ir, n_particles)
    else:
        pts, std_, _ = _pf_ancc(md_s, z_s, gr_s, gg, vmin, step, gs, ls, ir, n_particles,
                                int(seed) & 0x7FFFFFFF, IS)

    full = np.arange(start_ft, end_ft)
    mean_f = np.interp(full, idx, pts).astype(np.float32)
    std_f = np.interp(full, idx, std_).astype(np.float32)
    return mean_f, std_f

def pf_prior_tvt_seeds(wd, start_ft, end_ft, n_particles=200, sub=4, n_seeds=32,
                       init_spread=4.5, seed=0):
    """Мульти-сидовый прогон PF: возвращает сырые per-seed траектории + маргинальные лог-лайклихуды.
    (pts [S,n], std [S,n], ll [S], idx) — комбинирование (softmax по ll) делается снаружи,
    так что свип температуры бесплатен."""
    md_s, z_s, gr_s, gg, vmin, step, gs, ls, ir, idx, _ = _prep_ancc(wd, start_ft, end_ft, sub)
    S = n_seeds
    pts_all = np.empty((S, len(idx)))
    std_all = np.empty((S, len(idx)))
    ll = np.empty(S)
    for s_ in range(S):
        sd = (int(seed) + 7919 * s_) & 0x7FFFFFFF
        pts_all[s_], std_all[s_], ll[s_] = _pf_ancc(
            md_s, z_s, gr_s, gg, vmin, step, gs, ls, ir, n_particles, sd, init_spread)
    return pts_all, std_all, ll, idx

def combine_seeds(pts_all, std_all, ll, scale):
    """Softmax-взвешивание сидов по лог-лайклихуду (механизм паблик-PF 7.159).
    scale=None => plain mean. Возвращает (mean [n], std [n], w [S])."""
    S = len(ll)
    if scale is None:
        w = np.full(S, 1.0 / S)
    else:
        a = (ll - ll.max()) / max(float(scale), 1e-9)
        w = np.exp(a)
        w /= w.sum()
    mean = (w[:, None] * pts_all).sum(0)
    var = (w[:, None] * (std_all ** 2 + pts_all ** 2)).sum(0) - mean ** 2
    return mean, np.sqrt(np.clip(var, 0.0, None)), w

def pf_prior_tvt_ens(wd, start_ft, end_ft, n_particles=200, sub=4, n_seeds=32,
                     scale=5.0, init_spread=4.5, seed=0):
    """Likelihood-weighted ансамбль сидов (апгрейд P1). Интерфейс = pf_prior_tvt."""
    pts_all, std_all, ll, idx = pf_prior_tvt_seeds(
        wd, start_ft, end_ft, n_particles, sub, n_seeds, init_spread, seed)
    mean, std_, _ = combine_seeds(pts_all, std_all, ll, scale)
    full = np.arange(start_ft, end_ft)
    mean_f = np.interp(full, idx, mean).astype(np.float32)
    std_f = np.interp(full, idx, std_).astype(np.float32)
    return mean_f, std_f
