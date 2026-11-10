# Alexey Zakharov (drzoidberg), ROGII team solution.
"""Residual-grid HMM input preparation."""
from __future__ import annotations

import argparse
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hmm_latent_gpu import run_batch_latent


DSTEP = 0.05
DHALF = 45.0
R = 41
RHO = 0.20


VARIANTS = {
    "fixed30": dict(gs=(30.0,), lz=0.0),
    "fixed35": dict(gs=(35.0,), lz=0.0),
    "fixed43": dict(gs=(43.0,), lz=0.0),
    "fixed45": dict(gs=(45.0,), lz=0.0),
    "fixed50": dict(gs=(50.0,), lz=0.0),
    "fixed55": dict(gs=(55.0,), lz=0.0),
    "fixed60": dict(gs=(60.0,), lz=0.0),
    "fixed90": dict(gs=(90.0,), lz=0.0),
    "latent250": dict(gs=(43.0, 90.0), lz=250.0),
    "latent500": dict(gs=(43.0, 90.0), lz=500.0),
    "latent1000": dict(gs=(43.0, 90.0), lz=1000.0),
    "latent2000": dict(gs=(43.0, 90.0), lz=2000.0),
    "latent45_90_500": dict(gs=(45.0, 90.0), lz=500.0),
    "latent45_90_1000": dict(gs=(45.0, 90.0), lz=1000.0),
}


def load_fold_pure_wd():
    src = open("/tmp/lw_gen.py").read()
    glob = {}
    exec(src.split("# member configs:")[0], glob)
    wd = glob["WD"]
    wells = [w for w in sorted(wd) if wd[w]["hw"].TVT_input.isna().to_numpy().sum() >= 20]
    return wd, wells


def _gr_row_scale(md_all, raw_gr, tail, policy="interp", gap_ft=40.0):
    if policy == "interp":
        return np.ones(len(tail), dtype=np.float32)
    obs = np.isfinite(raw_gr)
    if policy == "observed":
        return obs[tail].astype(np.float32)
    if policy != "shortgap":
        raise ValueError("gr_policy must be interp, observed, or shortgap")
    scale = obs.astype(np.float32)
    n = len(raw_gr)
    i = 0
    while i < n:
        if obs[i]:
            i += 1
            continue
        j = i
        while j < n and not obs[j]:
            j += 1
        left = i - 1
        right = j
        if left >= 0 and right < n:
            span = float(md_all[right] - md_all[left])
            if np.isfinite(span) and span <= gap_ft:
                scale[i:j] = 1.0
            else:
                scale[i:j] = 0.0
        else:
            scale[i:j] = 0.0
        i = j
    return scale[tail].astype(np.float32)


def prep(
    wd,
    w,
    support_policy="endpoint",
    actual_dm=False,
    sig_r=0.002,
    kap_scale=1.0,
    gr_policy="interp",
    gr_gap_ft=40.0,
    dstep=DSTEP,
    dhalf=DHALF,
    rho=RHO,
    n_rates=R,
):
    d = wd[w]
    hw = d["hw"]
    ev = hw.TVT_input.isna().to_numpy()
    ti = hw["TVT_input"].to_numpy(float)
    md_all = hw["MD"].to_numpy(float)
    z_all = hw["Z"].to_numpy(float)
    tw_tvt = d["tt"]
    tw_gr = d["tg"]
    raw_gr = hw["GR"].to_numpy(float)
    gr = hw["GR"].interpolate(limit_direction="both").fillna(np.nanmean(tw_gr)).to_numpy(float)

    known = np.flatnonzero(np.isfinite(ti))
    if len(known) == 0:
        raise ValueError(f"{w}: no prefix TVT")
    last = known[-1]
    tail = np.flatnonzero(ev)
    md = md_all[tail]
    z = z_all[tail]
    gr_tail = gr[tail]
    row_scale = _gr_row_scale(md_all, raw_gr, tail, policy=gr_policy, gap_ft=gr_gap_ft)

    raw_dm = np.diff(np.concatenate([[md_all[last]], md]))
    if actual_dm:
        if not np.all(np.isfinite(raw_dm)) or np.any(raw_dm <= 0):
            raise ValueError(f"{w}: invalid MD increment")
        dm = raw_dm
    else:
        dm = np.maximum(raw_dm, 1.0)

    tl = known[-30:]
    dt = np.diff(ti[tl])
    dzt = np.diff(z_all[tl])
    dmt = np.diff(md_all[tl])
    ok = dmt > 0
    ir_abs = float(np.median((dt + dzt)[ok] / dmt[ok])) if ok.sum() >= 3 else 0.0

    rk = d["rk"][tail].astype(float)
    kap = d["kc"][tail].astype(float) * float(kap_scale)
    u_ref = (ti[last] + z_all[last]) + np.cumsum(rk * dm)
    tvt_ref = u_ref - z

    p_count = int(round(2 * float(dhalf) / float(dstep))) + 1
    dgrid = np.linspace(-float(dhalf), float(dhalf), p_count)
    rates = np.linspace(-float(rho), float(rho), int(n_rates))
    tvt_all = tvt_ref[:, None] + dgrid[None, :]
    valid = np.ones_like(tvt_all, dtype=bool)
    if support_policy == "neutral":
        valid &= (tvt_all >= tw_tvt[0]) & (tvt_all <= tw_tvt[-1])
    elif support_policy != "endpoint":
        raise ValueError("support_policy must be endpoint or neutral")

    gr_tw = np.interp(tvt_all.ravel(), tw_tvt, tw_gr).reshape(tvt_all.shape)
    res2 = (gr_tail[:, None] - gr_tw) ** 2
    return dict(
        res2=res2.astype(np.float32),
        valid=valid,
        row_scale=row_scale,
        dm=dm.astype(np.float64),
        md=md.astype(np.float64),
        dz=np.zeros(len(md), dtype=np.float64),
        rk=np.zeros(len(md), dtype=np.float64),
        kap=kap.astype(np.float64),
        rates=rates.astype(np.float64),
        step=float(dstep),
        grid=dgrid.astype(np.float32),
        start_p=float(float(dhalf) / float(dstep)),
        ir=float(ir_abs - rk[0]),
        ev_idx=tail,
        nrow=len(ti),
        T=len(md),
        P=p_count,
        sig_r=float(sig_r),
        sig_p=0.002,
        start_sig=0.75,
        r0_sig=0.01,
        tvt_ref=tvt_ref.astype(np.float64),
        well=w,
    )


def truth(wd, w):
    d = wd[w]
    ev = d["hw"].TVT_input.isna().to_numpy()
    tv = d["hw"].TVT.to_numpy(float)[ev]
    return tv, np.isfinite(tv), int(d["f"])


def pooled_score(wd, wells, pred):
    sse = 0.0
    n = 0
    fold = {i: [0.0, 0] for i in range(5)}
    for w in wells:
        tv, ok, f = truth(wd, w)
        pr = pred[w]
        e = pr[ok] - tv[ok]
        sse += float(np.dot(e, e))
        n += int(ok.sum())
        fold[f][0] += float(np.dot(e, e))
        fold[f][1] += int(ok.sum())
    folds = [
        float(np.sqrt(fold[i][0] / fold[i][1])) if fold[i][1] else float("nan")
        for i in range(5)
    ]
    return float(np.sqrt(sse / n)), folds


def select_wells(wells, mode, n):
    if mode == "full":
        return list(wells)
    return list(wells[::5][:n])


def decode_prediction(mean, feats, decoder):
    if decoder == "smooth":
        return mean
    if decoder == "filter":
        return feats["filter"]
    if decoder == "mode":
        return feats["mode"]
    if decoder.startswith("mix"):
        try:
            a = float(decoder[3:])
        except ValueError as exc:
            raise ValueError("mix decoder must look like mix0.75") from exc
        return a * mean + (1.0 - a) * feats["filter"]
    raise ValueError(f"unknown decoder {decoder}")


def run_variant(wd, wells, variant, budget, support_policy, actual_dm, decoder, pos_mode, save_feats=False):
    cfg = VARIANTS[variant]
    gs = tuple(cfg["gs"])
    zdim = len(gs)
    pred = {}
    feats = {}
    order = sorted(wells, key=lambda w: wd[w]["hw"].TVT_input.isna().sum())
    i = 0
    t0 = time.time()
    while i < len(order):
        batch = []
        tmax = 0
        pmax = 0
        while i < len(order):
            p = prep(wd, order[i], support_policy=support_policy, actual_dm=actual_dm)
            nt = max(tmax, p["T"])
            np_ = max(pmax, p["P"])
            nb = len(batch) + 1
            if batch and (nb * nt * np_ * R * zdim * 2 > budget or nb > 48):
                break
            batch.append(p)
            tmax = nt
            pmax = np_
            i += 1
        out = run_batch_latent(
            batch,
            pos_mode=pos_mode,
            gs_values=gs,
            regime_length_ft=float(cfg["lz"]),
            dev="cuda",
            return_feats=save_feats or decoder != "smooth",
        )
        for p, res in zip(batch, out):
            w = p["well"]
            if save_feats or decoder != "smooth":
                mean, std, ev_idx, nrow, ff = res
                feats[w] = ff
            else:
                mean, std, ev_idx, nrow = res
                ff = {}
            pred[w] = decode_prediction(mean, ff, decoder)
        torch.cuda.empty_cache()
    return pred, feats, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["screen", "full"])
    ap.add_argument("variants", nargs="*", default=["fixed45", "latent500", "latent1000"])
    ap.add_argument("--screen-n", type=int, default=150)
    ap.add_argument("--budget", type=float, default=4.8e9)
    ap.add_argument("--support-policy", choices=["endpoint", "neutral"], default="endpoint")
    ap.add_argument("--actual-dm", action="store_true")
    ap.add_argument("--decoder", default="smooth")
    ap.add_argument("--pos-mode", choices=["floor", "linear"], default="floor")
    ap.add_argument("--save", default="")
    ap.add_argument("--save-feats", action="store_true")
    args = ap.parse_args()

    wd, all_wells = load_fold_pure_wd()
    wells = select_wells(all_wells, args.mode, args.screen_n)
    print(
        "wells=%d mode=%s grid P=%d R=%d support=%s actual_dm=%s pos=%s"
        % (
            len(wells),
            args.mode,
            int(round(2 * DHALF / DSTEP)) + 1,
            R,
            args.support_policy,
            args.actual_dm,
            args.pos_mode,
        ),
        flush=True,
    )
    all_preds = {}
    all_feats = {}
    for variant in args.variants:
        if variant not in VARIANTS:
            raise SystemExit(f"unknown variant {variant}; known={sorted(VARIANTS)}")
        pred, feats, dt = run_variant(
            wd,
            wells,
            variant,
            args.budget,
            args.support_policy,
            args.actual_dm,
            args.decoder,
            args.pos_mode,
            save_feats=args.save_feats,
        )
        pooled, folds = pooled_score(wd, wells, pred)
        all_preds[variant] = pred
        all_feats[variant] = feats
        print(
            "%-18s decoder=%-8s pooled %.4f folds [%s] %.0fs"
            % (variant, args.decoder, pooled, " ".join("%.3f" % x for x in folds), dt),
            flush=True,
        )
        if args.save and args.mode == "full":
            with open(args.save, "wb") as f:
                pickle.dump(dict(preds=all_preds, feats=all_feats), f)
            print("saved", args.save, flush=True)


if __name__ == "__main__":
    main()
