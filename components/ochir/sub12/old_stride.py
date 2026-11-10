# Ochir Dorzhiev
from __future__ import annotations

import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numba import njit
from scipy.ndimage import gaussian_filter1d
from scipy.signal import fftconvolve

DATA = Path('/mnt/data/rogii_archive')

RATE_GRID_DEFAULT = np.array([
    -0.065, -0.055, -0.0475, -0.040, -0.033, -0.027, -0.022,
    -0.017, -0.012, -0.008, -0.004, 0.0, 0.004, 0.008, 0.012,
    0.017, 0.022, 0.027, 0.033, 0.040, 0.0475, 0.055, 0.065,
], dtype=np.float64)

@dataclass(frozen=True)
class VariableCfg:
    name: str
    endpoint_step: float = 50.0
    lengths_blocks: tuple[int, ...] = (1, 2, 3, 4, 6, 8, 12, 16)
    obs_step: float = 25.0
    beam: int = 80
    alpha: float = -0.5
    trend: str = 'regression'
    gamma: float = 12.0
    likelihood_weight: float = 1.0
    derivative_weight: float = 0.0
    derivative_gamma: float = 12.0
    motif_corr_weight: float = 0.0
    reverse_margin_weight: float = 0.0
    min_corr_points: int = 5
    local_bias_strength: float = 0.0
    likelihood: str = 'cauchy'
    student_dof: float = 3.0
    calibration: str = 'offset'
    tw_smooth_ft: float = 1.0
    tw_response: str = 'gaussian'
    self_weight: float = 0.0
    self_mode: str = 'mixture'
    self_radius_ft: float = 0.5
    self_neighbors: int = 5
    self_gamma: float = 12.0
    self_adaptive: bool = False
    self_validation_ft: float = 750.0
    self_min_validation: int = 40
    persist_sigma: float = 0.014
    persist_rho: float = 1.0
    persist_weight: float = 8.0
    rate_sigma: float = 0.030
    rate_weight: float = 2.0
    length_median_ft: float = 171.0
    length_log_sigma: float = 0.624
    length_weight: float = 1.0
    length_centered: bool = True
    segment_penalty: float = 0.0
    posterior_temp: float = 2.0
    tvt_bin: float = 0.5
    max_same: int = 1
    diversity_mode: str = 'tvt_rate'
    missing_cost: float = 8.0
    min_final_fraction: float = 0.35
    rate_grid: tuple[float, ...] = tuple(RATE_GRID_DEFAULT.tolist())


def robust_affine(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]
    y = y[m]
    if len(x) < 20:
        return 1.0, float(np.nanmedian(y - x)) if len(x) else 0.0
    A = np.c_[x, np.ones(len(x))]
    coef = np.linalg.lstsq(A, y, rcond=None)[0]
    for _ in range(6):
        r = y - A @ coef
        sc = 1.4826 * np.median(np.abs(r - np.median(r))) + 1e-6
        u = r / (1.5 * sc)
        w = np.ones(len(r))
        bad = np.abs(u) > 1
        w[bad] = 1 / np.abs(u[bad])
        coef = np.linalg.lstsq(A * np.sqrt(w)[:, None], y * np.sqrt(w), rcond=None)[0]
    return float(np.clip(coef[0], 0.5, 2.0)), float(coef[1])


def calibrate_typewell(
    tt: np.ndarray,
    tg: np.ndarray,
    tin: np.ndarray,
    gr: np.ndarray,
    mode: str,
) -> tuple[np.ndarray, float, float]:
    px = np.interp(tin, tt, tg, left=np.nan, right=np.nan)
    m = np.isfinite(px) & np.isfinite(gr)
    if mode == 'none' or not m.any():
        return tg.copy(), 1.0, 0.0
    if mode.startswith('offset'):
        b = float(np.nanmedian(gr[m] - px[m]))
        strength = 1.0
        if mode != 'offset':
            try:
                strength = float(mode.replace('offset', ''))
            except ValueError:
                strength = 1.0
        b *= strength
        return tg + b, 1.0, b
    a, b = robust_affine(px[m], gr[m])
    return a * tg + b, a, b


def apply_typewell_response(
    values: np.ndarray,
    step_ft: float,
    family: str,
    width_ft: float,
) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if family == 'point' or width_ft <= 0:
        return values.copy()
    if family == 'gaussian':
        return gaussian_filter1d(values, width_ft / max(step_ft, 1e-6), mode='nearest')
    if family == 'cusp':
        radius = max(2, int(math.ceil(8.0 * width_ft / max(step_ft, 1e-6))))
        x = np.arange(-radius, radius + 1, dtype=np.float64) * step_ft
        kernel = np.exp(-np.abs(x) / max(width_ft, 1e-6))
        kernel /= kernel.sum()
        padded = np.pad(values, radius, mode='edge')
        return fftconvolve(padded, kernel, mode='same')[radius:-radius]
    raise ValueError(f'unknown typewell response: {family}')


def build_self_reference(
    grid_tvt: np.ndarray,
    prefix_tvt: np.ndarray,
    prefix_gr: np.ndarray,
    radius_ft: float,
    neighbors: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Map known horizontal GR onto the typewell TVT grid without bridging gaps."""
    valid = np.isfinite(prefix_tvt) & np.isfinite(prefix_gr)
    x = np.asarray(prefix_tvt, dtype=np.float64)[valid]
    y = np.asarray(prefix_gr, dtype=np.float64)[valid]
    profile = np.zeros(len(grid_tvt), dtype=np.float64)
    available = np.zeros(len(grid_tvt), dtype=np.uint8)
    if len(x) == 0:
        return profile, available
    order = np.argsort(x)
    x, y = x[order], y[order]
    neighbors = max(1, int(neighbors))
    for i, tvt in enumerate(grid_tvt):
        pos = int(np.searchsorted(x, tvt))
        lo = max(0, pos - 3 * neighbors)
        hi = min(len(x), pos + 3 * neighbors + 1)
        idx = np.arange(lo, hi)
        if len(idx) == 0:
            continue
        distance = np.abs(x[idx] - tvt)
        idx = idx[np.argsort(distance)[:neighbors]]
        idx = idx[np.abs(x[idx] - tvt) <= radius_ft]
        if len(idx):
            profile[i] = float(np.median(y[idx]))
            available[i] = 1
    return profile, available


def lookup_self_profile(
    query_tvt: np.ndarray,
    grid_tvt: np.ndarray,
    profile: np.ndarray,
    available: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    index = np.rint(
        (np.asarray(query_tvt, dtype=np.float64) - grid_tvt[0])
        / max(float(np.median(np.diff(grid_tvt))), 1e-6)
    ).astype(np.int64)
    valid = (index >= 0) & (index < len(profile))
    clipped = np.clip(index, 0, len(profile) - 1)
    valid &= available[clipped].astype(bool)
    result = np.full(len(index), np.nan, dtype=np.float64)
    result[valid] = profile[clipped[valid]]
    return result, valid


def choose_self_weight(
    md: np.ndarray,
    prefix_tvt: np.ndarray,
    prefix_gr: np.ndarray,
    ps: int,
    tt: np.ndarray,
    typewell_response: np.ndarray,
    cfg: VariableCfg,
) -> tuple[float, int, float]:
    """Select self shrinkage on a pseudo-hidden tail of the known prefix."""
    split_md = float(md[ps] - cfg.self_validation_ft)
    train = np.flatnonzero(
        (np.arange(ps + 1) <= ps)
        & np.isfinite(prefix_tvt)
        & np.isfinite(prefix_gr)
        & (md[:ps + 1] < split_md)
    )
    valid = np.flatnonzero(
        np.isfinite(prefix_tvt)
        & np.isfinite(prefix_gr)
        & (md[:ps + 1] >= split_md)
    )
    if len(train) < 50 or len(valid) < cfg.self_min_validation:
        return float(cfg.self_weight), 0, np.nan

    calibrated, _, _ = calibrate_typewell(
        tt,
        typewell_response,
        prefix_tvt[train],
        prefix_gr[train],
        cfg.calibration,
    )
    profile, available = build_self_reference(
        tt,
        prefix_tvt[train],
        prefix_gr[train],
        cfg.self_radius_ft,
        cfg.self_neighbors,
    )
    self_pred, covered = lookup_self_profile(
        prefix_tvt[valid], tt, profile, available
    )
    type_pred = np.interp(
        prefix_tvt[valid], tt, calibrated, left=np.nan, right=np.nan
    )
    observed = prefix_gr[valid]
    covered &= np.isfinite(type_pred) & np.isfinite(observed)
    n = int(covered.sum())
    if n < cfg.self_min_validation:
        return float(cfg.self_weight), n, float(n / max(len(valid), 1))

    candidates = np.asarray([0.0, 0.25, 0.5, 0.75, 1.0], dtype=np.float64)
    costs = []
    for weight in candidates:
        prediction = (
            (1.0 - weight) * type_pred[covered]
            + weight * self_pred[covered]
        )
        error = (prediction - observed[covered]) / max(cfg.gamma, 1e-6)
        costs.append(float(np.mean(np.log1p(error * error))))
    selected = float(candidates[int(np.argmin(costs))])
    return selected, n, float(n / max(len(valid), 1))


def baseline_path(
    md: np.ndarray,
    z: np.ndarray,
    ps: int,
    t0: float,
    alpha: float,
    trend: str,
) -> np.ndarray:
    x = md[ps:] - md[ps]
    zz = z[ps:]
    if trend == 'endpoint':
        fit = zz[0] + (zz[-1] - zz[0]) * x / max(x[-1], 1e-9)
    elif trend == 'poly2':
        A = np.c_[x * x, x, np.ones(len(x))]
        fit = A @ np.linalg.lstsq(A, zz, rcond=None)[0]
    else:
        A = np.c_[x, np.ones(len(x))]
        fit = A @ np.linalg.lstsq(A, zz, rcond=None)[0]
    dev = zz - fit
    dev -= dev[0]
    out = np.full(len(md), np.nan)
    out[ps:] = t0 + alpha * dev
    return out


def bin_observations(
    md: np.ndarray,
    gr: np.ndarray,
    start: float,
    end: float,
    step: float,
) -> tuple[np.ndarray, np.ndarray]:
    edges = np.arange(start, end + step, step)
    if edges[-1] < end:
        edges = np.r_[edges, end]
    centers = (edges[:-1] + edges[1:]) / 2
    idx = np.searchsorted(md, edges)
    vals = np.full(len(centers), np.nan)
    for i in range(len(centers)):
        x = gr[idx[i]:idx[i + 1]]
        if np.isfinite(x).any():
            vals[i] = np.nanmedian(x)
    return centers, vals


@njit(cache=False, nogil=True)
def _interp_regular(t: float, tmin: float, tstep: float, tg: np.ndarray) -> tuple[float, bool]:
    u = (t - tmin) / tstep
    i = int(math.floor(u))
    if i < 0 or i >= len(tg) - 1:
        return 0.0, False
    f = u - i
    return tg[i] * (1.0 - f) + tg[i + 1] * f, True


@njit(cache=False, nogil=True)
def _lookup_self(t: float, tmin: float, tstep: float, sg: np.ndarray, sv: np.ndarray) -> tuple[float, bool]:
    i = int(round((t - tmin) / tstep))
    if i < 0 or i >= len(sg) or sv[i] == 0:
        return 0.0, False
    return sg[i], True


@njit(cache=False, nogil=True)
def _observation_cost(error: float, gamma: float, likelihood_kind: int, student_dof: float) -> float:
    normalized = error / gamma
    if likelihood_kind == 0:
        return math.log1p(normalized * normalized)
    if likelihood_kind == 1:
        return normalized * normalized
    return 0.5 * (student_dof + 1.0) * math.log1p(
        normalized * normalized / student_dof
    )


@njit(cache=False, nogil=True)
def _decode_core(
    bounds: np.ndarray,
    base_bounds: np.ndarray,
    obs_md: np.ndarray,
    obs_gr: np.ndarray,
    base_obs: np.ndarray,
    obs_left: np.ndarray,
    obs_right: np.ndarray,
    tg: np.ndarray,
    sg: np.ndarray,
    sv: np.ndarray,
    tmin: float,
    tstep: float,
    rate_grid: np.ndarray,
    lengths: np.ndarray,
    beam: int,
    gamma: float,
    likelihood_kind: int,
    student_dof: float,
    likelihood_weight: float,
    self_weight: float,
    self_mode: int,
    self_gamma: float,
    derivative_weight: float,
    derivative_gamma: float,
    motif_corr_weight: float,
    reverse_margin_weight: float,
    min_corr_points: int,
    local_bias_strength: float,
    persist_sigma: float,
    persist_rho: float,
    persist_weight: float,
    rate_sigma: float,
    rate_weight: float,
    length_median: float,
    length_log_sigma: float,
    length_weight: float,
    length_centered: int,
    segment_penalty: float,
    tvt_bin: float,
    max_same: int,
    diversity_mode: int,
    missing_cost: float,
    min_final_fraction: float,
):
    N = len(bounds)
    V = len(rate_grid)
    inf = 1e300
    state_res = np.full((N, beam), np.nan)
    state_rate = np.full((N, beam), -1, np.int32)
    state_cost = np.full((N, beam), inf)
    parent_ep = np.full((N, beam), -1, np.int32)
    parent_idx = np.full((N, beam), -1, np.int32)
    seg_rate = np.full((N, beam), -1, np.int32)
    counts = np.zeros(N, np.int32)

    zero_idx = int(np.argmin(np.abs(rate_grid)))
    state_res[0, 0] = 0.0
    state_rate[0, 0] = zero_idx
    state_cost[0, 0] = 0.0
    counts[0] = 1


    mode_len = length_median * math.exp(-length_log_sigma * length_log_sigma)
    zmode = (math.log(max(mode_len, 1.0)) - math.log(length_median)) / length_log_sigma
    mode_nll = math.log(max(mode_len, 1.0)) + math.log(length_log_sigma) + 0.5 * zmode * zmode

    maxcand = beam + beam * V
    cand_res = np.empty(maxcand)
    cand_rate = np.empty(maxcand, np.int32)
    cand_cost = np.empty(maxcand)
    cand_pep = np.empty(maxcand, np.int32)
    cand_pidx = np.empty(maxcand, np.int32)
    cand_seg_rate = np.empty(maxcand, np.int32)

    for j in range(N - 1):
        nj = counts[j]
        if nj == 0:
            continue
        for il in range(len(lengths)):
            lb = int(lengths[il])
            k = j + lb
            if k >= N:
                k = N - 1
            if k <= j:
                continue
            dur = bounds[k] - bounds[j]
            nominal = lb * (bounds[1] - bounds[0])
            if k == N - 1 and dur < min_final_fraction * nominal:
                continue

            nold = counts[k]
            nc = 0
            for q in range(nold):
                cand_res[nc] = state_res[k, q]
                cand_rate[nc] = state_rate[k, q]
                cand_cost[nc] = state_cost[k, q]
                cand_pep[nc] = parent_ep[k, q]
                cand_pidx[nc] = parent_idx[k, q]
                cand_seg_rate[nc] = seg_rate[k, q]
                nc += 1

            oi0 = obs_left[j]
            oi1 = obs_right[k]

            zlen = (math.log(max(dur, 1.0)) - math.log(length_median)) / length_log_sigma
            lnll = math.log(max(dur, 1.0)) + math.log(length_log_sigma) + 0.5 * zlen * zlen
            if length_centered == 1:
                lnll -= mode_nll
            lcost = length_weight * lnll + segment_penalty

            for q in range(nj):
                start_res = state_res[j, q]
                prev_r = rate_grid[state_rate[j, q]]
                prev_cost = state_cost[j, q]
                for ri in range(V):
                    rr = rate_grid[ri]
                    ll = 0.0


                    ebias = 0.0
                    ne = 0
                    if local_bias_strength > 0.0:
                        for oi in range(oi0, oi1):
                            if not math.isfinite(obs_gr[oi]):
                                continue
                            tvt = base_obs[oi] + start_res + rr * (obs_md[oi] - bounds[j])
                            pred, ok = _interp_regular(tvt, tmin, tstep, tg)
                            if ok:
                                ebias += pred - obs_gr[oi]
                                ne += 1
                        if ne > 0:
                            ebias = local_bias_strength * ebias / ne
                    prev_pred = 0.0
                    prev_obs = 0.0
                    prev_ok = False
                    dll = 0.0


                    cn = 0
                    so = 0.0
                    sp = 0.0
                    soo = 0.0
                    spp = 0.0
                    sop = 0.0
                    srp = 0.0
                    srr = 0.0
                    sor = 0.0
                    for oi in range(oi0, oi1):
                        if not math.isfinite(obs_gr[oi]):
                            prev_ok = False
                            continue
                        tvt = base_obs[oi] + start_res + rr * (obs_md[oi] - bounds[j])
                        pred, ok = _interp_regular(tvt, tmin, tstep, tg)
                        if not ok:
                            ll += missing_cost
                            prev_ok = False
                        else:
                            type_error = pred - obs_gr[oi] - ebias
                            self_pred, self_ok = _lookup_self(tvt, tmin, tstep, sg, sv)
                            effective_pred = pred
                            if self_weight > 0.0 and self_ok:
                                if self_mode == 0:
                                    effective_pred = (1.0 - self_weight) * pred + self_weight * self_pred
                                    ll += _observation_cost(
                                        effective_pred - obs_gr[oi] - ebias,
                                        gamma,
                                        likelihood_kind,
                                        student_dof,
                                    )
                                else:
                                    ll += (1.0 - self_weight) * _observation_cost(
                                        type_error, gamma, likelihood_kind, student_dof
                                    )
                                    ll += self_weight * _observation_cost(
                                        self_pred - obs_gr[oi],
                                        self_gamma,
                                        likelihood_kind,
                                        student_dof,
                                    )
                                    effective_pred = (1.0 - self_weight) * pred + self_weight * self_pred
                            else:
                                ll += _observation_cost(
                                    type_error, gamma, likelihood_kind, student_dof
                                )
                            if prev_ok and derivative_weight > 0.0:
                                de = (effective_pred - prev_pred) - (obs_gr[oi] - prev_obs)
                                dll += math.log1p((de / derivative_gamma) * (de / derivative_gamma))

                            oo = obs_gr[oi]
                            cn += 1
                            so += oo
                            sp += effective_pred
                            soo += oo * oo
                            spp += effective_pred * effective_pred
                            sop += oo * effective_pred


                            oj = oi1 - 1 - (oi - oi0)
                            tvtr = base_obs[oj] + start_res + rr * (obs_md[oj] - bounds[j])
                            predr, okr = _interp_regular(tvtr, tmin, tstep, tg)
                            if okr:
                                srp += predr
                                srr += predr * predr
                                sor += oo * predr
                            prev_pred = effective_pred
                            prev_obs = oo
                            prev_ok = True
                    ll += derivative_weight * dll
                    if cn >= min_corr_points and (motif_corr_weight > 0.0 or reverse_margin_weight > 0.0):
                        vo = soo - so * so / cn
                        vp = spp - sp * sp / cn
                        if vo > 1e-8 and vp > 1e-8:
                            cf = (sop - so * sp / cn) / math.sqrt(vo * vp)
                            if cf > 1.0:
                                cf = 1.0
                            elif cf < -1.0:
                                cf = -1.0
                            ll += motif_corr_weight * cn * (1.0 - cf)
                            vr = srr - srp * srp / cn
                            if reverse_margin_weight > 0.0 and vr > 1e-8:
                                cr = (sor - so * srp / cn) / math.sqrt(vo * vr)
                                if cr > 1.0:
                                    cr = 1.0
                                elif cr < -1.0:
                                    cr = -1.0
                                ll += reverse_margin_weight * cn * (cr - cf)
                    dp = (rr - persist_rho * prev_r) / persist_sigma
                    rp = rr / rate_sigma
                    total = prev_cost + likelihood_weight * ll + 0.5 * persist_weight * dp * dp + 0.5 * rate_weight * rp * rp + lcost
                    cand_res[nc] = start_res + rr * dur
                    cand_rate[nc] = ri
                    cand_cost[nc] = total
                    cand_pep[nc] = j
                    cand_pidx[nc] = q
                    cand_seg_rate[nc] = ri
                    nc += 1

            order = np.argsort(cand_cost[:nc])
            ns = 0

            sel_bin = np.empty(beam, np.int64)
            sel_rate = np.empty(beam, np.int32)
            sel_count = np.zeros(beam, np.int32)
            for oo in range(nc):
                ci = order[oo]
                abin = int(round((base_bounds[k] + cand_res[ci]) / tvt_bin))
                ri = cand_rate[ci] if diversity_mode == 0 else -1
                found = -1
                for s in range(ns):
                    if sel_bin[s] == abin and sel_rate[s] == ri:
                        found = s
                        break
                if found >= 0:
                    if sel_count[found] >= max_same:
                        continue
                    sel_count[found] += 1
                else:
                    sel_bin[ns] = abin
                    sel_rate[ns] = ri
                    sel_count[ns] = 1
                state_res[k, ns] = cand_res[ci]
                state_rate[k, ns] = cand_rate[ci]
                state_cost[k, ns] = cand_cost[ci]
                parent_ep[k, ns] = cand_pep[ci]
                parent_idx[k, ns] = cand_pidx[ci]
                seg_rate[k, ns] = cand_seg_rate[ci]
                ns += 1
                if ns >= beam:
                    break

            if ns < beam:
                for oo in range(nc):
                    ci = order[oo]
                    duplicate = False
                    for s in range(ns):
                        if (state_res[k, s] == cand_res[ci] and
                            state_rate[k, s] == cand_rate[ci] and
                            state_cost[k, s] == cand_cost[ci]):
                            duplicate = True
                            break
                    if duplicate:
                        continue
                    state_res[k, ns] = cand_res[ci]
                    state_rate[k, ns] = cand_rate[ci]
                    state_cost[k, ns] = cand_cost[ci]
                    parent_ep[k, ns] = cand_pep[ci]
                    parent_idx[k, ns] = cand_pidx[ci]
                    seg_rate[k, ns] = cand_seg_rate[ci]
                    ns += 1
                    if ns >= beam:
                        break
            counts[k] = ns

    return state_res, state_rate, state_cost, parent_ep, parent_idx, seg_rate, counts


def decode_well(
    hp: str | Path,
    cfg: VariableCfg,
    return_ensemble: bool = False,
) -> dict[str, Any]:
    hp = Path(hp)
    stem = hp.name.split('__')[0]
    d = pd.read_csv(hp, usecols=['MD', 'Z', 'TVT', 'TVT_input', 'GR']).sort_values('MD').reset_index(drop=True)
    md = d.MD.to_numpy(float)
    z = d.Z.to_numpy(float)
    truth = d.TVT.to_numpy(float)
    tin = d.TVT_input.to_numpy(float)
    gr = d.GR.to_numpy(float)
    known = np.flatnonzero(np.isfinite(tin))
    if len(known) < 20 or known[-1] >= len(d) - 2:
        return {'stem': stem, 'config': cfg.name, 'error': 'invalid'}
    ps = int(known[-1])
    t0 = float(tin[ps])
    base = baseline_path(md, z, ps, t0, cfg.alpha, cfg.trend)

    twp = hp.with_name(stem + '__typewell.csv')
    tw = pd.read_csv(twp, usecols=['TVT', 'GR']).dropna().sort_values('TVT')
    tt = tw.TVT.to_numpy(float)
    tg = tw.GR.to_numpy(float)
    tstep = float(np.median(np.diff(tt)))
    tg = apply_typewell_response(tg, tstep, cfg.tw_response, cfg.tw_smooth_ft)
    selected_self_weight = float(cfg.self_weight)
    self_validation_points = 0
    self_validation_coverage = np.nan
    if cfg.self_adaptive:
        selected_self_weight, self_validation_points, self_validation_coverage = choose_self_weight(
            md,
            tin[:ps + 1],
            gr[:ps + 1],
            ps,
            tt,
            tg,
            cfg,
        )
    tg, cal_a, cal_b = calibrate_typewell(tt, tg, tin[:ps + 1], gr[:ps + 1], cfg.calibration)
    self_gr, self_valid = build_self_reference(
        tt,
        tin[:ps + 1],
        gr[:ps + 1],
        cfg.self_radius_ft,
        cfg.self_neighbors,
    )

    omd, ogr = bin_observations(md, gr, md[ps], md[-1] + 1e-6, cfg.obs_step)
    use = omd > md[ps] + 1e-6
    omd = omd[use]
    ogr = ogr[use]
    base_obs = np.interp(omd, md[ps:], base[ps:])

    bounds = list(np.arange(md[ps], md[-1] + cfg.endpoint_step * 0.5, cfg.endpoint_step))
    if bounds[-1] < md[-1] - 1e-6:
        bounds.append(float(md[-1]))
    else:
        bounds[-1] = float(md[-1])
    bounds = np.asarray(bounds, dtype=np.float64)
    base_bounds = np.interp(bounds, md[ps:], base[ps:])
    obs_left = np.searchsorted(omd, bounds, side='right').astype(np.int32)
    obs_right = np.searchsorted(omd, bounds, side='right').astype(np.int32)

    rate_grid = np.asarray(cfg.rate_grid, dtype=np.float64)
    lengths = np.asarray(cfg.lengths_blocks, dtype=np.int32)
    likelihood_kind = {'cauchy': 0, 'mse': 1, 'student': 2}[cfg.likelihood]

    states = _decode_core(
        bounds, base_bounds, omd, ogr, base_obs, obs_left, obs_right,
        tg.astype(np.float64), self_gr, self_valid, float(tt[0]), tstep, rate_grid, lengths,
        int(cfg.beam), float(cfg.gamma), likelihood_kind, float(cfg.student_dof),
        float(cfg.likelihood_weight),
        selected_self_weight, 0 if cfg.self_mode == 'blend' else 1, float(cfg.self_gamma),
        float(cfg.derivative_weight), float(cfg.derivative_gamma),
        float(cfg.motif_corr_weight), float(cfg.reverse_margin_weight), int(cfg.min_corr_points),
        float(cfg.local_bias_strength),
        float(cfg.persist_sigma), float(cfg.persist_rho), float(cfg.persist_weight),
        float(cfg.rate_sigma), float(cfg.rate_weight),
        float(cfg.length_median_ft), float(cfg.length_log_sigma),
        float(cfg.length_weight), int(cfg.length_centered),
        float(cfg.segment_penalty), float(cfg.tvt_bin), int(cfg.max_same),
        0 if cfg.diversity_mode == 'tvt_rate' else 1,
        float(cfg.missing_cost), float(cfg.min_final_fraction),
    )
    state_res, state_rate, state_cost, parent_ep, parent_idx, seg_rate, counts = states
    N = len(bounds)
    nf = int(counts[-1])
    if nf == 0:
        return {'stem': stem, 'config': cfg.name, 'error': 'no_final'}

    residual_paths = np.zeros((nf, N), dtype=float)
    segment_counts = np.zeros(nf, dtype=int)
    for qf in range(nf):
        segs = []
        ep = N - 1
        qi = qf
        while ep > 0:
            pep = int(parent_ep[ep, qi])
            if pep < 0:
                break
            ri = int(seg_rate[ep, qi])
            segs.append((pep, ep, rate_grid[ri]))
            qi = int(parent_idx[ep, qi])
            ep = pep
        segs.reverse()
        cur = 0.0
        for aep, bep, rr in segs:
            residual_paths[qf, aep:bep + 1] = cur + rr * (bounds[aep:bep + 1] - bounds[aep])
            cur += rr * (bounds[bep] - bounds[aep])
        segment_counts[qf] = len(segs)

    absolute = residual_paths + base_bounds[None, :]
    costs = state_cost[-1, :nf]
    smd = md[ps + 1:]
    yy = truth[ps + 1:]


    dense_take = np.arange(0, len(smd), 5, dtype=int)
    dense_md = smd[dense_take]
    dense_gr = gr[ps + 1:][dense_take]
    dense_c8 = np.zeros(nf, dtype=float)
    dense_c12 = np.zeros(nf, dtype=float)
    dense_c20 = np.zeros(nf, dtype=float)
    dense_mse20 = np.zeros(nf, dtype=float)
    for qi in range(nf):
        pt = np.interp(dense_md, bounds, absolute[qi])
        pg = np.interp(pt, tt, tg, left=np.nan, right=np.nan)
        mm = np.isfinite(pg) & np.isfinite(dense_gr)
        if not np.any(mm):
            dense_c8[qi] = dense_c12[qi] = dense_c20[qi] = dense_mse20[qi] = 1e9
        else:
            er = pg[mm] - dense_gr[mm]
            missing = float(np.sum(~mm)) * cfg.missing_cost
            dense_c8[qi] = float(np.sum(np.log1p((er / 8.0) ** 2)) + missing)
            dense_c12[qi] = float(np.sum(np.log1p((er / 12.0) ** 2)) + missing)
            dense_c20[qi] = float(np.sum(np.log1p((er / 20.0) ** 2)) + missing)
            dense_mse20[qi] = float(np.sum((er / 20.0) ** 2) + missing)
    best = int(np.argmin(costs))
    map_pred = np.interp(smd, bounds, absolute[best])
    dc = costs - np.min(costs)
    w = np.exp(-dc / max(cfg.posterior_temp, 1e-6))
    w /= np.sum(w)
    mean_bounds = np.sum(absolute * w[:, None], axis=0)
    mean_pred = np.interp(smd, bounds, mean_bounds)
    top = np.argsort(costs)[:min(25, nf)]
    wt = np.exp(-(costs[top] - costs[top].min()) / max(cfg.posterior_temp, 1e-6))
    wt /= wt.sum()
    cons_bounds = np.sum(absolute[top] * wt[:, None], axis=0)
    cons_pred = np.interp(smd, bounds, cons_bounds)
    base_pred = np.interp(smd, md[ps:], base[ps:])

    def stat(p: np.ndarray) -> tuple[float, float, float]:
        e = p - yy
        return float(np.sqrt(np.mean(e * e))), float(np.sum(e * e)), float(e[-1])

    map_rmse, map_sse, map_end = stat(map_pred)
    mean_rmse, mean_sse, mean_end = stat(mean_pred)
    cons_rmse, cons_sse, cons_end = stat(cons_pred)
    base_rmse, base_sse, base_end = stat(base_pred)

    beam_rmses = np.empty(nf, dtype=float)
    beam_sses = np.empty(nf, dtype=float)
    for qi in range(nf):
        pp = np.interp(smd, bounds, absolute[qi])
        ee = pp - yy
        beam_sses[qi] = float(np.sum(ee * ee))
        beam_rmses[qi] = float(np.sqrt(np.mean(ee * ee)))
    oracle_q = int(np.argmin(beam_rmses))

    result = {
        'stem': stem,
        'config': cfg.name,
        'nrows': len(yy),
        'map_rmse': map_rmse,
        'map_sse': map_sse,
        'map_endpoint_error': map_end,
        'mean_rmse': mean_rmse,
        'mean_sse': mean_sse,
        'mean_endpoint_error': mean_end,
        'consensus_rmse': cons_rmse,
        'consensus_sse': cons_sse,
        'consensus_endpoint_error': cons_end,
        'baseline_rmse': base_rmse,
        'baseline_sse': base_sse,
        'baseline_endpoint_error': base_end,
        'cal_a': cal_a,
        'cal_b': cal_b,
        'selected_self_weight': selected_self_weight,
        'self_validation_points': self_validation_points,
        'self_validation_coverage': self_validation_coverage,
        'final_cost': float(costs.min()),
        'beam_spread': float(np.std(absolute[:, -1])),
        'map_segments': int(segment_counts[best]),
        'posterior_mean_segments': float(np.sum(segment_counts * w)),
        'beam_oracle_rmse': float(beam_rmses[oracle_q]),
        'beam_oracle_sse': float(beam_sses[oracle_q]),
        'beam_oracle_cost_rank': int(np.where(np.argsort(costs) == oracle_q)[0][0] + 1),
        'beam_costs': json.dumps(costs.tolist()),
        'beam_dense_c8': json.dumps(dense_c8.tolist()),
        'beam_dense_c12': json.dumps(dense_c12.tolist()),
        'beam_dense_c20': json.dumps(dense_c20.tolist()),
        'beam_dense_mse20': json.dumps(dense_mse20.tolist()),
        'beam_true_rmses': json.dumps(beam_rmses.tolist()),
        'boundaries': json.dumps(bounds.tolist()),
        'mean_tvt_boundaries': json.dumps(mean_bounds.tolist()),
        'map_tvt_boundaries': json.dumps(absolute[best].tolist()),
    }
    if return_ensemble:
        result['_ensemble_bounds'] = bounds
        result['_ensemble_paths'] = absolute
        result['_ensemble_costs'] = costs
        result['_ensemble_temperature'] = float(cfg.posterior_temp)
    return result


def pooled_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for config, g in df.dropna(subset=['mean_sse']).groupby('config'):
        rows.append({
            'config': config,
            'wells': len(g),
            'pooled_map': float(np.sqrt(g.map_sse.sum() / g.nrows.sum())),
            'pooled_mean': float(np.sqrt(g.mean_sse.sum() / g.nrows.sum())),
            'pooled_consensus': float(np.sqrt(g.consensus_sse.sum() / g.nrows.sum())),
            'pooled_baseline': float(np.sqrt(g.baseline_sse.sum() / g.nrows.sum())),
            'mean_per_well': float(g.mean_rmse.mean()),
            'median_per_well': float(g.mean_rmse.median()),
            'mean_segments': float(g.posterior_mean_segments.mean()),
        })
    return pd.DataFrame(rows).sort_values('pooled_mean')
