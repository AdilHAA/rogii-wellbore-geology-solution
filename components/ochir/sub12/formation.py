#!/usr/bin/env python3
# Ochir Dorzhiev
"""Rank real formation-trace analog paths with inference-visible evidence.

The outer split is grouped by exact typewell hash.  Candidate paths for every
query are transferred from other training wells and aligned only on the known
TVT prefix.  Hidden TVT is used as a candidate-quality label for outer-train
queries and for final reporting; it is never used to build or rank outer-val
candidates.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from tqdm import tqdm


HORIZONTAL_SUFFIX = "__horizontal_well.csv"
TYPEWELL_SUFFIX = "__typewell.csv"


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    keep = np.isfinite(a) & np.isfinite(b)
    return float(np.sqrt(np.mean((a[keep] - b[keep]) ** 2))) if keep.any() else np.nan


def corr(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    keep = np.isfinite(a) & np.isfinite(b)
    if keep.sum() < 12:
        return np.nan
    aa, bb = a[keep] - np.mean(a[keep]), b[keep] - np.mean(b[keep])
    den = float(np.sqrt(np.dot(aa, aa) * np.dot(bb, bb)))
    return float(np.dot(aa, bb) / den) if den > 1e-9 else np.nan


def slope(x: np.ndarray, y: np.ndarray) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    keep = np.isfinite(x) & np.isfinite(y)
    if keep.sum() < 8:
        return np.nan
    xx = x[keep] - np.mean(x[keep])
    yy = y[keep] - np.mean(y[keep])
    den = float(np.dot(xx, xx))
    return float(np.dot(xx, yy) / den) if den > 1e-9 else np.nan


def robust_scale(x: np.ndarray) -> float:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 8:
        return 20.0
    value = 1.4826 * float(np.median(np.abs(x - np.median(x))))
    return max(value, 5.0)


def robust_affine(reference: np.ndarray, observed: np.ndarray) -> tuple[float, float]:
    reference, observed = np.asarray(reference, float), np.asarray(observed, float)
    keep = np.isfinite(reference) & np.isfinite(observed)
    if keep.sum() < 20:
        return 1.0, 0.0
    x, y = reference[keep], observed[keep]
    for _ in range(3):
        if len(x) < 20 or np.std(x) < 1e-6:
            break
        a, b = np.polyfit(x, y, 1)
        residual = y - (a * x + b)
        threshold = 3.0 * robust_scale(residual)
        good = np.abs(residual - np.median(residual)) <= threshold
        x, y = x[good], y[good]
    if len(x) < 8 or np.std(x) < 1e-6:
        return 1.0, float(np.nanmedian(y) - np.nanmedian(x))
    a, b = np.polyfit(x, y, 1)
    return float(np.clip(a, 0.25, 4.0)), float(np.clip(b, -200.0, 200.0))


def fill_short(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, float).copy()
    idx = np.arange(len(x))
    finite = np.isfinite(x)
    if finite.sum() >= 2:
        x[~finite] = np.interp(idx[~finite], idx[finite], x[finite])
    elif finite.sum() == 1:
        x[:] = x[finite][0]
    else:
        x[:] = 85.0
    return x


def smooth(x: np.ndarray, window: int = 11) -> np.ndarray:
    return pd.Series(fill_short(x)).rolling(window, center=True, min_periods=1).median().to_numpy(float)


def principal_direction(xy: np.ndarray) -> np.ndarray:
    xy = np.asarray(xy, float)
    sample = xy[np.linspace(0, len(xy) - 1, min(len(xy), 1000)).astype(int)]
    centered = sample - np.mean(sample, axis=0, keepdims=True)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    direction = vh[0]
    if np.dot(direction, sample[-1] - sample[0]) < 0:
        direction = -direction
    return direction / max(float(np.linalg.norm(direction)), 1e-12)


def typewell_hash(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def deduplicate_curve(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    frame = pd.DataFrame({"x": x, "y": y}).replace([np.inf, -np.inf], np.nan).dropna()
    frame = frame.groupby("x", as_index=False)["y"].median().sort_values("x")
    return frame.x.to_numpy(float), frame.y.to_numpy(float)


def load_well(train_dir: Path, wid: str) -> dict | None:
    hp = train_dir / f"{wid}{HORIZONTAL_SUFFIX}"
    tp = train_dir / f"{wid}{TYPEWELL_SUFFIX}"
    try:
        h = pd.read_csv(hp, usecols=["MD", "X", "Y", "Z", "TVT", "TVT_input", "GR", "ANCC"])
        t = pd.read_csv(tp, usecols=["TVT", "GR"])
    except Exception:
        return None
    known = np.flatnonzero(np.isfinite(pd.to_numeric(h.TVT_input, errors="coerce")))
    if len(known) < 32 or known[-1] >= len(h) - 16:
        return None
    ps = int(known[-1])
    arrays = {c.lower(): pd.to_numeric(h[c], errors="coerce").to_numpy(float) for c in h.columns}
    xy = np.c_[arrays["x"], arrays["y"]]
    direction = principal_direction(xy[ps:])
    type_tvt, type_gr = deduplicate_curve(
        pd.to_numeric(t.TVT, errors="coerce").to_numpy(float),
        pd.to_numeric(t.GR, errors="coerce").to_numpy(float),
    )
    prefix_valid = np.isfinite(arrays["gr"][known])
    expected = np.interp(arrays["tvt_input"][known][prefix_valid], type_tvt, type_gr)
    gain, bias = robust_affine(expected, arrays["gr"][known][prefix_valid])
    gr_scale = robust_scale(arrays["gr"][known])

    pseudo_valid = np.isfinite(arrays["gr"][known]) & np.isfinite(arrays["tvt_input"][known])
    pseudo_tvt, pseudo_gr = deduplicate_curve(
        arrays["tvt_input"][known][pseudo_valid], arrays["gr"][known][pseudo_valid]
    )
    return {
        "wid": wid,
        "hash": typewell_hash(tp),
        **arrays,
        "xy": xy,
        "known": known,
        "ps": ps,
        "direction": direction,
        "centroid": np.nanmedian(xy[ps:], axis=0),
        "type_tvt": type_tvt,
        "type_gr": type_gr,
        "gr_gain": gain,
        "gr_bias": bias,
        "gr_scale": gr_scale,
        "gr_smooth": smooth(arrays["gr"]),
        "pseudo_tvt": pseudo_tvt,
        "pseudo_gr": pseudo_gr,
    }


def angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip(abs(float(np.dot(a, b))), 0.0, 1.0))))


def interp_linear_extrap(x: np.ndarray, xp: np.ndarray, fp: np.ndarray) -> np.ndarray:
    order = np.argsort(xp)
    xp, fp = np.asarray(xp, float)[order], np.asarray(fp, float)[order]
    keep = np.isfinite(xp) & np.isfinite(fp)
    xp, fp = xp[keep], fp[keep]
    if len(xp) < 8:
        return np.full(len(x), np.nan)
    unique = np.r_[True, np.diff(xp) > 1e-6]
    xp, fp = xp[unique], fp[unique]
    if len(xp) < 8 or np.ptp(xp) < 1e-6:
        return np.full(len(x), np.nan)
    out = np.interp(x, xp, fp)
    edge = min(64, max(8, len(xp) // 8))
    lc, rc = np.polyfit(xp[:edge], fp[:edge], 1), np.polyfit(xp[-edge:], fp[-edge:], 1)
    out[x < xp[0]] = np.polyval(lc, x[x < xp[0]])
    out[x > xp[-1]] = np.polyval(rc, x[x > xp[-1]])
    return out


def trend_sign_agreement(observed: np.ndarray, expected: np.ndarray,
                         smooth_win: int = 401, step: int = 200) -> float:
    """Compare slow GR trends (heavy smoothing first to kill Milankovitch oscillations).
    Computes slope of smooth_win-rolling-mean over step-sized strides.
    smooth_win ~400ft >> Milankovitch period ~15ft in MD space.
    """
    obs_f = fill_short(observed)
    exp_f = fill_short(expected)
    obs_s = pd.Series(obs_f).rolling(smooth_win, center=True, min_periods=smooth_win // 3).mean().to_numpy(float)
    exp_s = pd.Series(exp_f).rolling(smooth_win, center=True, min_periods=smooth_win // 3).mean().to_numpy(float)
    agrees, total = 0, 0
    n = len(obs_s)
    for i in range(0, n - step, step):
        j = min(i + step, n - 1)
        d_obs = obs_s[j] - obs_s[i]
        d_exp = exp_s[j] - exp_s[i]
        if not (np.isfinite(d_obs) and np.isfinite(d_exp)):
            continue
        if abs(d_obs) < 0.05 or abs(d_exp) < 0.05:
            continue
        total += 1
        if np.sign(d_obs) == np.sign(d_exp):
            agrees += 1
    return float(agrees / total) if total > 0 else np.nan


def inflection_alignment(observed: np.ndarray, expected: np.ndarray, smooth_win: int = 31, tol: int = 50) -> float:
    """Fraction of expected GR inflection points (peaks/troughs) that have a matching
    observed inflection within ±tol samples. Robust to calibration drift."""
    def find_inflections(x: np.ndarray, win: int) -> np.ndarray:
        s = pd.Series(x).rolling(win, center=True, min_periods=win // 2).mean().to_numpy(float)
        d = np.diff(s)
        sign_changes = np.where(np.diff(np.sign(d)))[0] + 1
        return sign_changes

    m = np.isfinite(observed) & np.isfinite(expected)
    if m.sum() < smooth_win * 4:
        return np.nan
    obs_filled = observed.copy()
    obs_filled[~np.isfinite(obs_filled)] = np.nanmean(observed)
    exp_filled = expected.copy()
    exp_filled[~np.isfinite(exp_filled)] = np.nanmean(expected)
    exp_inflections = find_inflections(exp_filled, smooth_win)
    obs_inflections = find_inflections(obs_filled, smooth_win)
    if len(exp_inflections) == 0:
        return np.nan
    hits = 0
    for ei in exp_inflections:
        if len(obs_inflections) > 0 and np.min(np.abs(obs_inflections - ei)) <= tol:
            hits += 1
    return float(hits / len(exp_inflections))


def sequence_features(prefix: str, observed: np.ndarray, expected: np.ndarray, scale: float) -> dict:
    keep = np.isfinite(observed) & np.isfinite(expected)
    out = {f"{prefix}_coverage": float(np.mean(keep))}
    if keep.sum() < 12:
        for name in ("mae", "rmse", "corr", "dcorr", "bias", "seg_corr_mean", "seg_corr_max",
                     "trend_agree", "trend_agree_200"):
            out[f"{prefix}_{name}"] = np.nan
        return out
    o, e = observed[keep], expected[keep]
    residual = (o - e) / max(scale, 1e-6)
    out.update(
        {
            f"{prefix}_mae": float(np.mean(np.abs(residual))),
            f"{prefix}_rmse": float(np.sqrt(np.mean(residual**2))),
            f"{prefix}_corr": corr(o, e),
            f"{prefix}_dcorr": corr(np.diff(o), np.diff(e)),
            f"{prefix}_bias": float(np.mean(residual)),
        }
    )
    segment_corrs = []
    edges = np.linspace(0, len(observed), 5).astype(int)
    for start, stop in zip(edges[:-1], edges[1:]):
        value = corr(observed[start:stop], expected[start:stop])
        if np.isfinite(value):
            segment_corrs.append(value)
    out[f"{prefix}_seg_corr_mean"] = float(np.mean(segment_corrs)) if segment_corrs else np.nan
    out[f"{prefix}_seg_corr_max"] = float(np.max(segment_corrs)) if segment_corrs else np.nan

    obs_slow = pd.Series(fill_short(observed)).rolling(401, center=True, min_periods=134).mean().to_numpy(float)
    exp_slow = pd.Series(fill_short(expected)).rolling(401, center=True, min_periods=134).mean().to_numpy(float)
    out[f"{prefix}_slow_corr"] = corr(obs_slow, exp_slow)
    out[f"{prefix}_slow_bias"] = float(np.nanmean(obs_slow - exp_slow)) / max(scale, 1e-6)

    out[f"{prefix}_trend_agree"] = trend_sign_agreement(observed, expected, smooth_win=401, step=200)
    out[f"{prefix}_trend_agree_200"] = trend_sign_agreement(observed, expected, smooth_win=201, step=100)
    return out


def build_candidate(
    query: dict, neighbor: dict, prefix_window: int, score_tail: bool = True,
) -> dict | None:
    direction = query["direction"]
    perpendicular = np.array([-direction[1], direction[0]])
    qs, ns = query["xy"] @ direction, neighbor["xy"] @ direction
    stride = max(1, len(ns) // 1800)
    surface = interp_linear_extrap(qs, ns[::stride], neighbor["ancc"][::stride])
    known = query["known"][-min(prefix_window, len(query["known"])) :]
    keep = np.isfinite(surface[known])
    if keep.sum() < 16:
        return None
    known = known[keep]
    observed_u = query["tvt_input"][known] + query["z"][known]
    offset = float(np.nanmedian(observed_u - surface[known]))
    future = np.arange(query["ps"] + 1, len(query["tvt"]))
    prediction = surface[future] + offset - query["z"][future]
    if not np.isfinite(prediction).all():
        return None
    truth = query["tvt"][future] if score_tail else None
    flat_value = float(query["tvt_input"][query["ps"]])
    prefix_residual = observed_u - (surface[known] + offset)

    delta = neighbor["centroid"] - query["centroid"]
    perp_distance = float(abs(np.dot(delta, perpendicular)))
    along_distance = float(abs(np.dot(delta, direction)))
    qmin, qmax = float(np.min(qs[future])), float(np.max(qs[future]))
    nmin, nmax = float(np.min(ns)), float(np.max(ns))
    overlap = max(0.0, min(qmax, nmax) - max(qmin, nmin)) / max(qmax - qmin, 1e-6)
    extrap = float(np.mean((qs[future] < nmin) | (qs[future] > nmax)))

    features = {
        "angle_deg": angle_deg(query["direction"], neighbor["direction"]),
        "perp_distance_kft": perp_distance / 1000.0,
        "along_distance_kft": along_distance / 1000.0,
        "centroid_distance_kft": float(np.linalg.norm(delta)) / 1000.0,
        "overlap_fraction": overlap,
        "extrap_fraction": extrap,
        "prefix_rmse": float(np.sqrt(np.mean(prefix_residual**2))),
        "prefix_mae": float(np.mean(np.abs(prefix_residual))),
        "prefix_maxabs": float(np.max(np.abs(prefix_residual))),
        "prefix_residual_slope": slope(query["md"][known], prefix_residual),
        "tail_length_kft": float(query["md"][future][-1] - query["md"][future][0]) / 1000.0,
        "tail_rows_k": len(future) / 1000.0,
        "query_z_slope": slope(query["md"][future], query["z"][future]),
        "query_z_range": float(np.ptp(query["z"][future])) / 100.0,
        "known_u_slope": slope(query["md"][known], observed_u),
        "candidate_tvt_slope": slope(query["md"][future], prediction),
        "candidate_tvt_drift": float(prediction[-1] - prediction[0]) / 50.0,
        "candidate_tvt_range": float(np.ptp(prediction)) / 50.0,
        "candidate_flat_rmse": rmse(prediction, np.full(len(prediction), flat_value)) / 20.0,
        "candidate_u_slope": slope(query["md"][future], surface[future]),
        "candidate_u_range": float(np.ptp(surface[future])) / 50.0,
        "candidate_u_curvature": float(np.mean(np.abs(np.diff(surface[future], n=2)))) * 1000.0,
        "source_rows_k": len(neighbor["md"]) / 1000.0,
    }
    for window in (64, 128, 256, 512):
        idx = known[-min(window, len(known)) :]
        residual = query["tvt_input"][idx] + query["z"][idx] - (surface[idx] + offset)
        features[f"prefix_rmse_{window}"] = float(np.sqrt(np.mean(residual**2)))
        features[f"prefix_slope_{window}"] = slope(query["md"][idx], residual)

    type_expected = query["gr_gain"] * np.interp(
        prediction, query["type_tvt"], query["type_gr"]
    ) + query["gr_bias"]
    type_in_range = (prediction >= query["type_tvt"][0]) & (prediction <= query["type_tvt"][-1])
    observed_gr = query["gr"][future].copy()
    observed_gr[~type_in_range] = np.nan
    type_expected[~type_in_range] = np.nan
    features.update(sequence_features("type_gr", observed_gr, type_expected, query["gr_scale"]))
    observed_smooth = query["gr_smooth"][future].copy()
    observed_smooth[~np.isfinite(query["gr"][future])] = np.nan
    features.update(sequence_features("type_gr_smooth", observed_smooth, smooth(type_expected), query["gr_scale"]))

    if len(query["pseudo_tvt"]) >= 8:
        pseudo_expected = np.interp(prediction, query["pseudo_tvt"], query["pseudo_gr"])
        pseudo_range = (prediction >= query["pseudo_tvt"][0]) & (prediction <= query["pseudo_tvt"][-1])
        pseudo_expected[~pseudo_range] = np.nan
        pseudo_observed = query["gr"][future].copy()
        pseudo_observed[~pseudo_range] = np.nan
        features.update(sequence_features("pseudo_gr", pseudo_observed, pseudo_expected, query["gr_scale"]))

    return {
        "neighbor": neighbor["wid"],
        "features": features,
        "prediction": prediction.astype(np.float32),
        "tail_rmse": rmse(prediction, truth) if truth is not None else np.nan,
    }


def candidates_for(query: dict, sources: list[dict], args) -> list[dict]:
    direction = query["direction"]
    perpendicular = np.array([-direction[1], direction[0]])
    coarse = []
    for neighbor in sources:
        if neighbor["wid"] == query["wid"] or neighbor["hash"] == query["hash"]:
            continue
        angle = angle_deg(direction, neighbor["direction"])
        if angle > args.max_angle:
            continue
        delta = neighbor["centroid"] - query["centroid"]
        perp = abs(float(np.dot(delta, perpendicular)))
        along = abs(float(np.dot(delta, direction)))
        coarse.append((perp + args.along_weight * along + 20.0 * angle, neighbor))
    coarse.sort(key=lambda item: item[0])
    result = []
    for _, neighbor in coarse[: args.max_candidates]:
        candidate = build_candidate(
            query, neighbor, args.prefix_window,
            score_tail=getattr(args, "score_tail", True),
        )
        if candidate is not None:
            result.append(candidate)
    return result


def candidate_frame(query_ids: list[str], records: dict, sources: list[dict], args, retain: bool):
    rows, retained, groups = [], {}, []
    for wid in tqdm(query_ids, desc="build candidate groups"):
        query = records.get(wid)
        if query is None:
            continue
        candidates = candidates_for(query, sources, args)
        if len(candidates) < 4:
            continue
        candidates.sort(key=lambda item: item["tail_rmse"])
        n = len(candidates)
        for rank, item in enumerate(candidates):
            relevance = int(np.clip(10 - np.floor(10 * rank / max(n, 1)), 0, 10))
            rows.append(
                {
                    "well": wid,
                    "neighbor": item["neighbor"],
                    "tail_rmse": item["tail_rmse"],
                    "relevance": relevance,
                    **item["features"],
                }
            )
        groups.append(n)
        if retain:
            retained[wid] = candidates
    return pd.DataFrame(rows), retained, groups


def pooled(rows: list[dict], column: str) -> float:
    numerator = sum(row["n_eval"] * row[column] ** 2 for row in rows if np.isfinite(row[column]))
    denominator = sum(row["n_eval"] for row in rows if np.isfinite(row[column]))
    return float(np.sqrt(numerator / max(denominator, 1)))


def run(args) -> None:
    root = Path(args.train_dir)
    ids = sorted(path.name.split("__")[0] for path in root.glob(f"*{HORIZONTAL_SUFFIX}"))
    hashes = np.asarray([typewell_hash(root / f"{wid}{TYPEWELL_SUFFIX}") for wid in ids])
    tr_idx, va_idx = list(GroupKFold(args.folds).split(ids, groups=hashes))[args.fold]
    train_ids, val_ids = [ids[i] for i in tr_idx], [ids[i] for i in va_idx]
    if args.max_train_queries:
        train_ids = train_ids[: args.max_train_queries]
    if args.max_val_queries:
        val_ids = val_ids[: args.max_val_queries]

    records = {}
    for wid in tqdm(ids, desc="load wells"):
        record = load_well(root, wid)
        if record is not None:
            records[wid] = record
    sources = [records[wid] for wid in [ids[i] for i in tr_idx] if wid in records]
    train_frame, _, train_groups = candidate_frame(train_ids, records, sources, args, retain=False)
    val_frame, retained, _ = candidate_frame(val_ids, records, sources, args, retain=True)

    blocked = {"well", "neighbor", "tail_rmse", "relevance"}
    features = [c for c in train_frame.columns if c not in blocked]
    x_train = train_frame[features].replace([np.inf, -np.inf], np.nan).astype(np.float32)
    x_val = val_frame[features].replace([np.inf, -np.inf], np.nan).astype(np.float32)

    reg = lgb.LGBMRegressor(
        objective="regression_l1", n_estimators=args.trees, learning_rate=0.03,
        num_leaves=31, min_child_samples=80, reg_lambda=5.0, reg_alpha=0.1,
        colsample_bytree=0.85, verbosity=-1, random_state=2026,
    )
    reg.fit(x_train, np.log1p(train_frame.tail_rmse.to_numpy(float)))
    ranker = lgb.LGBMRanker(
        objective="lambdarank", metric="ndcg", n_estimators=args.trees,
        learning_rate=0.03, num_leaves=31, min_child_samples=60,
        reg_lambda=5.0, reg_alpha=0.1, colsample_bytree=0.85,
        verbosity=-1, random_state=2026,
    )
    ranker.fit(x_train, train_frame.relevance.to_numpy(int), group=train_groups)
    val_frame["reg_score"] = reg.predict(x_val)
    val_frame["rank_score"] = ranker.predict(x_val)

    rows, prediction_rows = [], []
    for wid, group in val_frame.groupby("well", sort=False):
        query = records[wid]
        candidates = retained[wid]
        by_neighbor = {item["neighbor"]: item for item in candidates}
        future = np.arange(query["ps"] + 1, len(query["tvt"]))
        truth = query["tvt"][future]
        flat = np.full(len(future), query["tvt_input"][query["ps"]])
        reg_row = group.loc[group.reg_score.idxmin()]
        rank_row = group.loc[group.rank_score.idxmax()]
        heuristic_row = group.sort_values(["prefix_rmse", "perp_distance_kft"]).iloc[0]
        picks = {
            "reg": by_neighbor[reg_row.neighbor]["prediction"],
            "rank": by_neighbor[rank_row.neighbor]["prediction"],
            "heuristic": by_neighbor[heuristic_row.neighbor]["prediction"],
        }
        oracle_item = min(candidates, key=lambda item: item["tail_rmse"])
        metrics = {
            "well": wid, "n_eval": len(future), "flat_rmse": rmse(flat, truth),
            "oracle_rmse": oracle_item["tail_rmse"],
        }
        for name, prediction in picks.items():
            metrics[f"{name}_rmse"] = rmse(prediction, truth)
            for clip in (10, 20, 30, 40):
                clipped = np.clip(prediction, flat - clip, flat + clip)
                metrics[f"{name}_clip{clip}_rmse"] = rmse(clipped, truth)
        rows.append(metrics)
        for j, idx in enumerate(future):
            prediction_rows.append(
                {
                    "well": wid, "row": int(idx), "truth": float(truth[j]),
                    "flat": float(flat[j]), "reg": float(picks["reg"][j]),
                    "rank": float(picks["rank"][j]),
                    "heuristic": float(picks["heuristic"][j]),
                    "oracle": float(oracle_item["prediction"][j]),
                    "fold": args.fold,
                }
            )

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    train_frame.to_parquet(out / f"fold{args.fold}_train_candidates.parquet", index=False)
    val_frame.to_parquet(out / f"fold{args.fold}_val_candidates.parquet", index=False)
    per_well = pd.DataFrame(rows)
    per_well.to_csv(out / f"fold{args.fold}_per_well.csv", index=False)
    pd.DataFrame(prediction_rows).to_parquet(out / f"fold{args.fold}_predictions.parquet", index=False)
    pd.DataFrame({"feature": features, "importance": reg.feature_importances_}).sort_values(
        "importance", ascending=False
    ).to_csv(out / f"fold{args.fold}_feature_importance.csv", index=False)
    metric_cols = [c for c in per_well.columns if c.endswith("_rmse")]
    summary = pd.DataFrame({"method": metric_cols, "pooled_rmse": [pooled(rows, c) for c in metric_cols]})
    summary = summary.sort_values("pooled_rmse")
    summary.to_csv(out / f"fold{args.fold}_summary.csv", index=False)
    print(f"train candidates={len(train_frame):,} val candidates={len(val_frame):,} features={len(features)}")
    print(summary.to_string(index=False))
    print("\nTop features")
    print(pd.read_csv(out / f"fold{args.fold}_feature_importance.csv").head(20).to_string(index=False))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train-dir", default="rogii-wellbore-geology-prediction/train")
    p.add_argument("--folds", type=int, default=4)
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--max-angle", type=float, default=25.0)
    p.add_argument("--max-candidates", type=int, default=80)
    p.add_argument("--prefix-window", type=int, default=512)
    p.add_argument("--along-weight", type=float, default=0.15)
    p.add_argument("--trees", type=int, default=350)
    p.add_argument("--max-train-queries", type=int, default=0)
    p.add_argument("--max-val-queries", type=int, default=0)
    p.add_argument("--out-dir", default="formation_analog_ranker")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
