# Alexey Zakharov (drzoidberg), ROGII team solution.
"""GR normalization and tail constraints."""
from __future__ import annotations

import numpy as np


def well_beta(d: dict) -> float:
    """Full-well L2 coefficient, used only to estimate fold-train beta."""
    hw = d["hw"]
    g = hw["GR"].to_numpy(float)
    tvt = hw["TVT"].to_numpy(float)
    tt = np.asarray(d["tt"], dtype=np.float64)
    tg = np.asarray(d["tg"], dtype=np.float64)
    use = np.isfinite(g) & np.isfinite(tvt) & (tvt >= tt[0]) & (tvt <= tt[-1])
    if use.sum() < 2:
        return np.nan
    y = np.interp(tvt[use], tt, tg)
    den = float(np.dot(g[use], g[use]))
    return float(np.dot(g[use], y) / den) if den > 0 else np.nan


def fold_train_betas(wd: dict) -> tuple[dict[int, float], float, dict[str, float]]:
    """Mean per-well beta excluding the validation fold, plus full-train beta."""
    by_well = {w: well_beta(d) for w, d in wd.items()}
    finite = {w: b for w, b in by_well.items() if np.isfinite(b)}
    folds = sorted({int(wd[w]["f"]) for w in finite})
    beta_by_fold = {
        fold: float(np.mean([b for w, b in finite.items() if int(wd[w]["f"]) != fold]))
        for fold in folds
    }
    return beta_by_fold, float(np.mean(list(finite.values()))), by_well


def tail_l2_spec(d: dict, beta0: float) -> dict:
    """Observable tail scale and weighted-tail target for one well."""
    hw = d["hw"]
    g = hw["GR"].to_numpy(float)
    tvt_input = hw["TVT_input"].to_numpy(float)
    tt = np.asarray(d["tt"], dtype=np.float64)
    tg = np.asarray(d["tg"], dtype=np.float64)
    observed = np.isfinite(g)
    prefix = observed & np.isfinite(tvt_input)
    prefix &= (tvt_input >= tt[0]) & (tvt_input <= tt[-1])
    tail = observed & ~np.isfinite(tvt_input)
    if tail.sum() == 0:
        raise ValueError("no observed tail GR")
    gp = g[prefix]
    gt = g[tail]
    yp = np.interp(tvt_input[prefix], tt, tg)
    prefix_g2 = float(np.dot(gp, gp))
    tail_g2 = float(np.dot(gt, gt))
    prefix_gy = float(np.dot(gp, yp))
    target_cross = float(beta0 * (prefix_g2 + tail_g2) - prefix_gy)
    scale = float(target_cross / tail_g2)
    sum_tail_g = float(np.sum(gt))
    if not np.isfinite(scale) or tail_g2 <= 0 or sum_tail_g <= 0:
        raise ValueError("invalid tail L2 moments")
    return {
        "scale": scale,
        "target_cross": target_cross,
        "target_weighted_mean": target_cross / sum_tail_g,
        "sum_tail_g": sum_tail_g,
        "prefix_g2": prefix_g2,
        "tail_g2": tail_g2,
        "prefix_gy": prefix_gy,
        "n_prefix": int(prefix.sum()),
        "n_tail": int(tail.sum()),
    }


def emission_gr(d: dict, tail_scale: float) -> np.ndarray:
    """Match the stock interpolation first, then scale only masked-tail rows."""
    hw = d["hw"]
    tw_mean = float(np.nanmean(np.asarray(d["tg"], dtype=np.float64)))
    gr = (
        hw["GR"]
        .interpolate(limit_direction="both")
        .fillna(tw_mean)
        .to_numpy(float)
    )
    tail = hw["TVT_input"].isna().to_numpy()
    gr[tail] *= float(tail_scale)
    return gr


def constraint_arrays(d: dict, prep: dict, spec: dict, sigma_api: float) -> dict:
    """Build H=(sum g*TW - C)/sum(g), in API-normalized units."""
    hw = d["hw"]
    raw_g = hw["GR"].to_numpy(float)
    tail_idx = np.asarray(prep["ev_idx"], dtype=np.int64)
    gt = raw_g[tail_idx]
    observed = np.isfinite(gt)
    if observed.sum() == 0:
        raise ValueError("no observed tail GR")
    weights = np.zeros(len(tail_idx), dtype=np.float64)
    weights[observed] = gt[observed] / float(spec["sum_tail_g"])

    tt = np.asarray(d["tt"], dtype=np.float64)
    tg = np.asarray(d["tg"], dtype=np.float64)
    tvt_state = (
        np.asarray(prep["tvt_ref"], dtype=np.float64)[:, None]
        + np.asarray(prep["grid"], dtype=np.float64)[None, :]
    )
    f_state = np.interp(tvt_state.ravel(), tt, tg).reshape(tvt_state.shape)
    f_scale = float(np.std(tg))
    if not np.isfinite(f_scale) or f_scale < 1e-8:
        f_scale = 1.0
    target = float(spec["target_weighted_mean"])
    return {
        "tilt_h": (weights[:, None] * (f_state - target) / f_scale).astype(np.float32),
        "sigma_h2": float((float(sigma_api) / f_scale) ** 2),
        "target": target,
        "f_scale": f_scale,
        "weights": weights,
    }


def weighted_residual(d: dict, tvt_tail: np.ndarray, spec: dict) -> float:
    """Weighted-tail residual in API units for a proposed tail trajectory."""
    hw = d["hw"]
    tail_idx = np.flatnonzero(hw["TVT_input"].isna().to_numpy())
    if len(tail_idx) != len(tvt_tail):
        raise ValueError("tail prediction length mismatch")
    g = hw["GR"].to_numpy(float)[tail_idx]
    observed = np.isfinite(g) & np.isfinite(tvt_tail)
    tt = np.asarray(d["tt"], dtype=np.float64)
    tg = np.asarray(d["tg"], dtype=np.float64)
    y = np.interp(np.asarray(tvt_tail, dtype=float)[observed], tt, tg)
    cross = float(np.dot(g[observed], y))
    return float((cross - spec["target_cross"]) / spec["sum_tail_g"])
