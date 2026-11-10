# Alexey Zakharov (drzoidberg), ROGII team solution.
"""Static PF/Q/HMM deployment."""
from __future__ import annotations

import gc
import os
import sys
import time
import types
from pathlib import Path

import numpy as np
import pandas as pd
import torch

SUPPORT = Path(__file__).resolve().parent
sys.path.insert(0, str(SUPPORT))

import surface_models


_shim = types.ModuleType("direct_model")
_shim.DirectionalGradientModel = surface_models.DirectionalGradientModel
sys.modules["direct_model"] = _shim

import segment_confidence
from deployment_gate import load_gate, q_weight
from gate_features import summarize_gate_features
from h6_decoder_grid_fast import fixed_prep, plan_batches
from hmm_fixed_fast import run_batch_fixed_decoders
from kappa_mapping import kappa_from_kvar, load_sigma_cdf
from l2_utils import emission_gr, tail_l2_spec
from phase4_pf import _smoothed_dydx
from phase4_pf_cuda_logz import run_pf_wells_cuda_batch_perseed as cudab


BETA0 = 0.9990418109874853
Q_W_STATIC = 0.5810017584414995
TAU_F, TAU_B = 3.5, 6.0
NS_F, NS_B = 64, 32
NP = 500
GW = 0.30
VN = PN = 0.002
FLL = 1000
NANCH = 30
DENSITY_H = 8.0
BETA_Q = 0.10
REG_A, REG_B = 5.0, 1e-3
HMM_GS, HMM_DHALF, HMM_MIX = 45.0, 120.0, 0.60
MEMBERS = {
    "g30": ("rk", "kvk", "tg", 30.0),
    "g45": ("rk", "kvk", "tg", 45.0),
    "g60": ("rk", "kvk", "tg", 60.0),
    "mom": ("rkp", "kmk", "tg", 60.0),
    "adjh": ("rk", "kvk", "tg_cal", 60.0),


    "g43": ("rk", "kvk", "tg", 43.0),
}
MEMBER_ORDER = ("g30", "g45", "g60", "mom", "adjh")
ANCHORS = ((0.25, 25), (0.50, 50), (0.75, 75), (1.00, 100))


def pick_device(requested: str | None = None) -> str:
    if requested:
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _empty_cache(device: str) -> None:
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    elif device == "mps":
        torch.mps.empty_cache()


def _fit_linear(p_gr: np.ndarray, tw_at: np.ndarray) -> tuple[float, float]:
    """Fit affine typewell calibration on the known prefix."""
    lo, hi = np.percentile(p_gr, [1, 99])
    pg = np.clip(p_gr, lo, hi)
    x = tw_at
    sxx = x @ x + REG_A
    sx = x.sum()
    s1 = len(x) + REG_B
    sxy = x @ pg + REG_A
    sy = pg.sum()
    det = sxx * s1 - sx * sx
    if abs(det) < 1e-9:
        return 1.0, 0.0
    a = (sxy * s1 - sx * sy) / det
    b = (sxx * sy - sx * sxy) / det
    return float(np.clip(a, 0.3, 3.0)), float(np.clip(b, -100, 100))


def load_assets(support_dir: str | Path | None = None, surface_device: str = "cpu") -> dict:
    sup = Path(support_dir) if support_dir else SUPPORT
    surface = surface_models.load_surface(
        sup / "krige_shipped_robust_direct_alltrain.pkl", device=surface_device)
    return {
        "surface": surface,
        "sigma_cdf": load_sigma_cdf(sup / "sigma_cdf.npy"),
        "segments": segment_confidence.load_segments(sup / "conf_segments_alltrain.pkl"),
        "gate": load_gate(sup / "deployment_gate.json"),
    }


def prep_well(wid: str, test_dir: str | Path, assets: dict) -> dict:
    """Build the per-well context dict (test-side WD entry)."""
    test_dir = Path(test_dir)
    raw = pd.read_csv(test_dir / f"{wid}__horizontal_well.csv")
    order = raw["MD"].to_numpy(float).argsort(kind="stable")
    hw = raw.iloc[order].reset_index(drop=True)
    orig_row = raw.index.to_numpy()[order]
    tw = pd.read_csv(test_dir / f"{wid}__typewell.csv").sort_values("TVT")

    d: dict = {"well": wid, "hw": hw, "orig_row": orig_row}
    d["md"] = hw["MD"].to_numpy(float)
    d["Z"] = hw["Z"].to_numpy(float)
    d["tt"] = tw["TVT"].to_numpy(float)
    d["tg"] = tw["GR"].fillna(tw["GR"].mean()).to_numpy(float)

    be, kv = assets["surface"].predict_well(hw)
    d["be"] = np.asarray(be, float)
    d["kv"] = np.asarray(kv, float)

    d["rk"] = _smoothed_dydx(d["be"], d["md"], win_frac=GW)
    d["rkp"] = d["rk"]
    d["kvk"] = kappa_from_kvar(d["kv"], assets["sigma_cdf"], member="g60")
    d["kmk"] = kappa_from_kvar(d["kv"], assets["sigma_cdf"], member="mom")
    d["kc"] = segment_confidence.kappa(hw, assets["segments"])

    pref = hw["TVT_input"].notna().to_numpy() & hw["GR"].notna().to_numpy()
    a, b = 1.0, 0.0
    if pref.sum() >= 100:
        p_tvt = hw["TVT_input"].to_numpy(float)[pref]
        p_gr = hw["GR"].to_numpy(float)[pref]
        a, b = _fit_linear(p_gr, np.interp(p_tvt, d["tt"], d["tg"]))
    d["tg_cal"] = a * d["tg"] + b

    try:
        spec = tail_l2_spec(d, BETA0)
        d["gr_scale"] = float(spec["scale"])
    except (ValueError, ZeroDivisionError):
        d["gr_scale"] = 1.0
    d["gr_scaled"] = emission_gr(d, d["gr_scale"])
    d["ev"] = hw["TVT_input"].isna().to_numpy()
    return d


def _run_raw(jobs: list, ns: int, batch: int, gs: float, device: str) -> list:
    out = []
    start = 0
    while start < len(jobs):
        chunk = jobs[start:start + batch]
        try:
            out.extend(cudab(
                [x[1] for x in chunk], ns, NP, 0, gs,
                FLL=FLL, vn=VN, pn=PN, device=device, dtype=torch.float32,
            ))
            start += len(chunk)
        except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
            if "out of memory" not in str(exc).lower() or batch <= 1:
                raise
            batch = max(1, batch // 2)
            _empty_cache(device)
            print(f"PF batch OOM; retrying at batch={batch}", flush=True)
    return out


def _density(sp: np.ndarray) -> np.ndarray:
    return np.maximum((np.abs(sp[:, None, :] - sp[None, :, :]) <= DENSITY_H).sum(1), 1.0)


def run_forward_legs(WD: dict, wells: list, device: str, batch: int,
                     members: tuple = MEMBER_ORDER) -> dict:
    """Anti-density likelihood-weighted forward members; ev-length TVT arrays."""
    fwd = {tag: {} for tag in members}
    t0 = time.time()
    for tag in members:
        rk_key, kap_key, tw_key, gs = MEMBERS[tag]
        jobs = []
        for w in wells:
            d = WD[w]
            hw_scaled = d["hw"].copy()
            hw_scaled["GR"] = d["gr_scaled"]
            jobs.append((w, dict(
                well=w, hw=hw_scaled, tw_tvt=d["tt"], tw_gr=d[tw_key],
                rk_full=d[rk_key], kap_v_ev=d[kap_key],
            )))
        raw = _run_raw(jobs, NS_F, batch, gs, device)
        for (w, _), r in zip(jobs, raw):
            sp = r["seed_preds"].astype(np.float32, copy=False)
            evidence = np.exp((r["logz"] - r["logz"].max()) / TAU_F)
            weights = evidence[:, None] / np.sqrt(_density(sp))
            weights /= weights.sum(0)
            fwd[tag][w] = (weights * sp).sum(0)
        del raw
        _empty_cache(device)
        print(f"FWD {tag:5s} done {time.time()-t0:.0f}s", flush=True)
    return fwd


def _backward_prep(d: dict, frac: float, tag: str):
    hw = d["hw"]
    md = d["md"]
    z = d["Z"]
    rk_key, kap_key, tw_key, _ = MEMBERS[tag]
    base = d["be"]
    rk = d[rk_key]
    kap = d[kap_key]
    pref = hw["TVT_input"].notna().to_numpy()
    if not pref.any():
        return None
    hi = int(np.flatnonzero(pref)[-1])
    surf = base - z - (base[hi] - (hw["TVT_input"].to_numpy(float)[hi] + z[hi]))
    ti = np.flatnonzero(~pref)
    kf = int(round(frac * (len(ti) - 1)))
    if kf < NANCH + 10:
        return None
    anchor = ti[kf]
    cov = ti[:kf + 1]
    rev = np.arange(anchor, ti[0] - 1, -1)
    revmap = {int(row): j for j, row in enumerate(rev)}
    md_rev = np.concatenate([[0.0], np.cumsum(np.abs(np.diff(md[rev])))])
    tin = np.full(len(rev), np.nan)
    tin[:NANCH] = surf[rev[:NANCH]]
    hwr = pd.DataFrame({
        "MD": md_rev, "Z": z[rev], "GR": d["gr_scaled"][rev], "TVT_input": tin,
    })
    job = dict(
        well=d["well"], hw=hwr, tw_tvt=d["tt"], tw_gr=d[tw_key],
        rk_full=-rk[rev], kap_v_ev=(kap[rev])[np.isnan(tin)],
    )
    positions = np.array([revmap[int(row)] for row in cov])
    return job, positions, kf


def run_backward_legs(WD: dict, wells: list, device: str, batch: int,
                      members: tuple = MEMBER_ORDER) -> dict:
    """Evidence-weighted quarter-backward legs; ev-length arrays, NaN past anchor."""
    bwd = {tag: {a: {} for _, a in ANCHORS} for tag in members}
    t0 = time.time()
    for tag in members:
        gs = MEMBERS[tag][3]
        for frac, anchor_tag in ANCHORS:
            preps = []
            for w in wells:
                p = _backward_prep(WD[w], frac, tag)
                if p is not None:
                    preps.append((w, *p))
            jobs = [(w, job) for w, job, _, _ in preps]
            raw = _run_raw(jobs, NS_B, batch, gs, device)
            for (w, _, positions, kf), r in zip(preps, raw):
                sp = r["seed_preds"]
                full = np.tile(r["out"], (sp.shape[0], 1))
                full[:, r["ev_pos"]] = sp
                evidence = np.exp((r["logz"] - r["logz"].max()) / TAU_B)
                pred = ((evidence / evidence.sum())[:, None] * full).sum(0)[positions]
                arr = np.full(int(WD[w]["ev"].sum()), np.nan)
                arr[:kf + 1] = pred
                bwd[tag][anchor_tag][w] = arr
            del raw
            _empty_cache(device)
            print(f"BWD {tag:5s} a{anchor_tag} done {time.time()-t0:.0f}s", flush=True)
    return bwd


def build_q(WD: dict, wells: list, fwd: dict, bwd: dict,
            members: tuple = MEMBER_ORDER) -> tuple[dict, dict]:
    """Per-well consistency composite -> (Q prediction, member weights)."""
    nm = len(members)
    q_pred, q_weights = {}, {}
    for w in wells:
        n = int(WD[w]["ev"].sum())
        fw = np.stack([fwd[tag][w] for tag in members])
        frac = np.arange(n) / max(n - 1, 1)
        composites = []
        for tag in members:
            out = np.full(n, np.nan)
            lo = 0.0
            for hi, anchor in ((0.25, 25), (0.50, 50), (0.75, 75), (1.01, 100)):
                mask = (frac >= lo) & (frac < hi)
                values = bwd[tag][anchor].get(w)
                fallback = bwd[tag][100].get(w)
                if values is None:
                    values = fallback
                if values is not None:
                    seg = values[mask].copy()
                    if fallback is not None:
                        bad = ~np.isfinite(seg)
                        seg[bad] = fallback[mask][bad]
                    out[mask] = seg
                lo = hi
            composites.append(out)
        bw = np.stack(composites)
        if np.isfinite(bw).any():
            with np.errstate(invalid="ignore"):
                disagreement = np.array([
                    [np.nanmean(np.abs(fw[i] - bw[j])) for j in range(nm)]
                    for i in range(nm)
                ])
            consistency = np.nanmin(disagreement, 1)
            if np.isfinite(consistency).all():
                weights = np.exp(-BETA_Q * (consistency - consistency.min()))
                weights /= weights.sum()
            else:
                weights = np.full(nm, 1.0 / nm)
        else:
            weights = np.full(nm, 1.0 / nm)
        q_pred[w] = (weights[:, None] * fw).sum(0)
        q_weights[w] = {tag: float(weights[i]) for i, tag in enumerate(members)}
    return q_pred, q_weights


def _hmm_prepare_one(WD: dict, w: str) -> dict:
    """Prepare per-well HMM inputs with fixed GR scale and tail shrinkage."""
    d = WD[w]
    p = fixed_prep(
        WD, w, "fixed45", "endpoint", False, 0.002, 1.0,
        "interp", 40.0, 0.05, HMM_DHALF, 0.20, 41, 0.0, 0.0, 5,
    )
    scale = d["gr_scale"]
    tt = np.asarray(d["tt"], dtype=np.float64)
    tg = np.asarray(d["tg"], dtype=np.float64)
    tvt_state = (
        np.asarray(p["tvt_ref"], dtype=np.float64)[:, None]
        + np.asarray(p["grid"], dtype=np.float64)[None, :]
    )
    f_state = np.interp(tvt_state.ravel(), tt, tg).reshape(tvt_state.shape)
    gr = d["gr_scaled"][np.asarray(p["ev_idx"], dtype=np.int64)]
    res2 = (gr[:, None] - f_state) ** 2
    row_scale = np.asarray(p.get("row_scale", np.ones(p["T"])), dtype=np.float64)
    z2 = np.minimum(row_scale[:, None] * res2 / (HMM_GS ** 2), 600.0)
    p["em"] = np.where(p["valid"], np.exp(-0.5 * z2), 1.0).astype(np.float32)
    return p

def _run_hmm_batch(preps: list, device: str) -> list:
    """One decoder batch with OOM halving; per-well CPU fallback (MPS only)."""
    try:
        return run_batch_fixed_decoders(
            preps, dev=device, pos_mode="linear", rate_mode="tri",
            tilt_eta=np.zeros(len(preps), dtype=np.float32),
        )
    except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
        if "out of memory" not in str(exc).lower():
            raise
        _empty_cache(device)
        if len(preps) > 1:
            half = len(preps) // 2
            print(f"HMM batch OOM; splitting {len(preps)} -> {half}", flush=True)
            return _run_hmm_batch(preps[:half], device) + _run_hmm_batch(preps[half:], device)
        if device == "mps":


            print(f"HMM well OOM on mps; decoding {preps[0]['well']} on cpu", flush=True)
            return _run_hmm_batch(preps, "cpu")
        raise


def run_hmm(WD: dict, wells: list, device: str, budget: float) -> dict:
    """Fixed-gs45 segment-confidence HMM; returns smooth/filter/mode/hmm arrays."""
    todo = [w for w in wells if int(WD[w]["ev"].sum()) >= 20]
    out = {}
    t0 = time.time()
    done = 0
    for batch_wells in plan_batches(WD, todo, budget, 0.05, HMM_DHALF, 41):
        preps = [_hmm_prepare_one(WD, w) for w in batch_wells]
        results = _run_hmm_batch(preps, device)
        for r in results:
            smooth = r["smooth"].astype(np.float64)
            filt = r["filter"].astype(np.float64)
            out[r["well"]] = {
                "smooth": smooth, "filter": filt,
                "mode": r["mode"].astype(np.float64),
                "hmm": HMM_MIX * smooth + (1.0 - HMM_MIX) * filt,
            }
        done += len(preps)
        _empty_cache(device)
        print(f"HMM {done}/{len(todo)} {time.time()-t0:.0f}s", flush=True)
    return out


def combine_well(mode: str, d: dict, q: np.ndarray, q_w: dict,
                 fwd: dict, hmm: dict | None, gate_model: dict) -> tuple[np.ndarray, float]:
    if mode == "q_only" or hmm is None:
        return q, 1.0
    if mode == "static":
        w = Q_W_STATIC
        return w * q + (1.0 - w) * hmm["hmm"], w
    if mode == "gate":
        try:
            ev = d["ev"]
            features = summarize_gate_features(
                pf_predictions={tag: fwd[tag][d["well"]] for tag in MEMBER_ORDER},
                q_prediction=q,
                hmm_prediction=hmm["hmm"],
                hmm_smooth=hmm["smooth"],
                hmm_filter=hmm["filter"],
                hmm_mode=hmm["mode"],
                kvar=d["kv"][ev],
                q_member_weights=q_w,
                raw_tail_gr=d["hw"]["GR"].to_numpy(float)[ev],
            )
            w = q_weight(features, gate_model)
        except (ValueError, KeyError) as exc:
            print(f"gate fallback ({d['well']}): {exc}", flush=True)
            w = Q_W_STATIC
        return w * q + (1.0 - w) * hmm["hmm"], w
    raise ValueError(f"unknown mode {mode!r}")


def run_deployment(
    test_dir: str | Path,
    mode: str,
    out_csv: str | Path = "submission.csv",
    support_dir: str | Path | None = None,
    device: str | None = None,
    batch_fwd: int = 64,
    batch_bwd: int = 256,
    hmm_budget: float = 1.2e10,
    wells: list | None = None,
    sample_submission: str | Path | None = None,
    members: tuple = MEMBER_ORDER,
) -> pd.DataFrame:
    assert mode in ("q_only", "static", "gate")
    unknown = set(members).difference(MEMBERS)
    if unknown:
        raise ValueError(f"unknown PF members: {sorted(unknown)}")
    if mode == "gate" and tuple(members) != MEMBER_ORDER:
        raise ValueError("the SOTA5971 gate is fitted on the release 5-member set only")
    device = pick_device(device)
    test_dir = Path(test_dir)
    assets = load_assets(support_dir, surface_device="cuda" if device == "cuda" else "cpu")
    if wells is None:
        wells = sorted(p.name.split("__")[0] for p in test_dir.glob("*__horizontal_well.csv"))
    print(f"DEPLOY mode={mode} members={list(members)} wells={len(wells)} device={device} "
          f"NS_F={NS_F} NS_B={NS_B} NP={NP}", flush=True)

    t0 = time.time()
    WD = {}
    for i, w in enumerate(wells):
        WD[w] = prep_well(w, test_dir, assets)
        if (i + 1) % 25 == 0 or i + 1 == len(wells):
            print(f"prep {i+1}/{len(wells)} {time.time()-t0:.0f}s", flush=True)

    fwd = run_forward_legs(WD, wells, device, batch_fwd, members)
    bwd = run_backward_legs(WD, wells, device, batch_bwd, members)
    q_pred, q_weights = build_q(WD, wells, fwd, bwd, members)

    hmm = {}
    if mode != "q_only":
        hmm = run_hmm(WD, wells, device, hmm_budget)

    rows = []
    gate_log = []
    for w in wells:
        d = WD[w]
        pred, weight = combine_well(
            mode, d, q_pred[w], q_weights[w], fwd, hmm.get(w), assets["gate"])
        gate_log.append((w, weight))
        ev_rows = d["orig_row"][d["ev"]]
        for ridx, value in zip(ev_rows, pred):
            rows.append((f"{w}_{int(ridx)}", float(value)))
    if mode != "q_only":
        ws = np.array([x[1] for x in gate_log])
        print(f"Q weights: min={ws.min():.3f} med={np.median(ws):.3f} max={ws.max():.3f}",
              flush=True)

    sub = pd.DataFrame(rows, columns=["id", "tvt"])
    if sample_submission and Path(sample_submission).exists():
        samp = pd.read_csv(sample_submission)[["id"]]
        merged = samp.merge(sub, on="id", how="left")
        print(f"sample align: {merged.tvt.notna().sum()}/{len(samp)} matched, "
              f"NaN={merged.tvt.isna().sum()}", flush=True)
        sub = merged
    if sub["tvt"].isna().any():
        raise RuntimeError(f"{int(sub.tvt.isna().sum())} unfilled submission rows")
    sub.to_csv(out_csv, index=False)
    print(f"wrote {out_csv}: {sub.shape} elapsed={time.time()-t0:.0f}s", flush=True)
    return sub
