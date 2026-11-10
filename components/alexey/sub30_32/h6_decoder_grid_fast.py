# Alexey Zakharov (drzoidberg), ROGII team solution.
"""Batched fixed-GR-scale HMM decoding."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import os
import pickle
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from h6_latent_gs import R, VARIANTS, load_fold_pure_wd, pooled_score, prep, select_wells
from hmm_fixed_fast import run_batch_fixed_decoders


DECODERS = (
    "smooth",
    "filter",
    "mode",
    "mix0.10",
    "mix0.20",
    "mix0.30",
    "mix0.40",
    "mix0.50",
    "mix0.60",
    "mix0.70",
    "mix0.80",
    "mix0.90",
)


def decoder_names(include_viterbi=False):
    names = list(DECODERS)
    if include_viterbi:
        names.append("viterbi")
    return names


def fixed_prep(
    wd,
    w,
    variant,
    support_policy,
    actual_dm,
    sig_r,
    kap_scale,
    gr_policy,
    gr_gap_ft,
    dstep,
    dhalf,
    rho,
    n_rates,
    shape_block_ft,
    shape_alpha,
    shape_min_points,
):
    cfg = VARIANTS[variant]
    if len(cfg["gs"]) != 1:
        raise ValueError(f"{variant} is not a fixed-gs variant")
    gs = float(cfg["gs"][0])
    p = prep(
        wd,
        w,
        support_policy=support_policy,
        actual_dm=actual_dm,
        sig_r=sig_r,
        kap_scale=kap_scale,
        gr_policy=gr_policy,
        gr_gap_ft=gr_gap_ft,
        dstep=dstep,
        dhalf=dhalf,
        rho=rho,
        n_rates=n_rates,
    )
    res2 = np.asarray(p["res2"], dtype=np.float64)
    row_scale = np.asarray(p.get("row_scale", np.ones(p["T"])), dtype=np.float32)
    z2_endpoint = row_scale[:, None].astype(np.float64) * res2 / (gs * gs)
    if shape_block_ft > 0.0 and shape_alpha > 0.0:
        md = np.asarray(p["md"], dtype=np.float64)
        valid = np.asarray(p["valid"], dtype=bool)
        shape_res2 = np.empty_like(res2)
        weights = row_scale[:, None].astype(np.float64) * valid.astype(np.float64)
        for t in range(p["T"]):
            lo = int(np.searchsorted(md, md[t] - float(shape_block_ft), side="left"))
            ww = weights[lo : t + 1]
            cnt = (ww > 0.0).sum(axis=0)
            den = ww.sum(axis=0)
            num = (res2[lo : t + 1] * ww).sum(axis=0)
            local = np.divide(num, den, out=res2[t].copy(), where=den > 0.0)
            shape_res2[t] = np.where(cnt >= int(shape_min_points), local, res2[t])
        z2_shape = shape_res2 / (gs * gs)
        z2 = (1.0 - float(shape_alpha)) * z2_endpoint + float(shape_alpha) * z2_shape
    else:
        z2 = z2_endpoint
    z2 = np.minimum(z2, 600.0)
    p["em"] = np.where(p["valid"], np.exp(-0.5 * z2), 1.0).astype(np.float32)
    return p


def decode_arrays(res, include_viterbi=False):
    names = decoder_names(include_viterbi)
    out = {name: None for name in names}
    smooth = res["smooth"]
    filt = res["filter"]
    out["smooth"] = smooth
    out["filter"] = filt
    out["mode"] = res["mode"]
    for name in DECODERS:
        if name.startswith("mix"):
            a = float(name[3:])
            out[name] = a * smooth + (1.0 - a) * filt
    if include_viterbi:
        out["viterbi"] = res["viterbi"]
    return out


def print_diag_summary(diag_by_well):
    if not diag_by_well:
        print("diag none", flush=True)
        return
    keys = ("pos_edge1", "pos_edge5", "rate_edge1", "rate_edge2")
    print("diagnostics: posterior edge mass across tail rows", flush=True)
    for key in keys:
        vals = []
        well_max = []
        for diag in diag_by_well.values():
            if key not in diag:
                continue
            x = np.asarray(diag[key], dtype=np.float64)
            x = x[np.isfinite(x)]
            if x.size == 0:
                continue
            vals.append(x)
            well_max.append(float(np.max(x)))
        if not vals:
            print(f"diag {key} missing", flush=True)
            continue
        v = np.concatenate(vals)
        wm = np.asarray(well_max, dtype=np.float64)
        print(
            "diag %-10s mean %.6g p95 %.6g p99 %.6g max %.6g row>1e-3 %.4f row>1e-2 %.4f row>5e-2 %.4f wellmax_p95 %.6g wellmax_max %.6g"
            % (
                key,
                float(np.mean(v)),
                float(np.quantile(v, 0.95)),
                float(np.quantile(v, 0.99)),
                float(np.max(v)),
                float(np.mean(v > 1e-3)),
                float(np.mean(v > 1e-2)),
                float(np.mean(v > 5e-2)),
                float(np.quantile(wm, 0.95)),
                float(np.max(wm)),
            ),
            flush=True,
        )


def plan_batches(wd, wells, budget, dstep, dhalf, n_rates):
    """Plan batches within the decoder memory budget."""
    order = sorted(wells, key=lambda w: wd[w]["hw"].TVT_input.isna().sum())
    p_count = int(round(2 * float(dhalf) / float(dstep))) + 1
    batches = []
    batch = []
    tmax = 0
    for w in order:
        t = int(wd[w]["hw"].TVT_input.isna().sum())
        nt = max(tmax, t)
        nb = len(batch) + 1
        if batch and (nb * nt * p_count * int(n_rates) * 2 > budget or nb > 48):
            batches.append(batch)
            batch = []
            tmax = 0
            nt = t
            nb = 1
        batch.append(w)
        tmax = nt
    if batch:
        batches.append(batch)
    return batches


def run_variant(
    wd,
    wells,
    variant,
    budget,
    support_policy,
    actual_dm,
    sig_r,
    kap_scale,
    gr_policy,
    gr_gap_ft,
    pos_mode,
    rate_mode,
    block_lag_ft,
    include_viterbi,
    dstep,
    dhalf,
    rho,
    n_rates,
    shape_block_ft,
    shape_alpha,
    shape_min_points,
    report_diag,
    prep_workers,
):
    decoders = decoder_names(include_viterbi)
    preds = {name: {} for name in decoders}
    diag_by_well = {}
    batches = plan_batches(wd, wells, budget, dstep, dhalf, n_rates)
    prep_kw = dict(
        variant=variant,
        support_policy=support_policy,
        actual_dm=actual_dm,
        sig_r=sig_r,
        kap_scale=kap_scale,
        gr_policy=gr_policy,
        gr_gap_ft=gr_gap_ft,
        dstep=dstep,
        dhalf=dhalf,
        rho=rho,
        n_rates=n_rates,
        shape_block_ft=shape_block_ft,
        shape_alpha=shape_alpha,
        shape_min_points=shape_min_points,
    )

    def prep_one(w):
        return fixed_prep(wd, w, **prep_kw)

    t0 = time.time()
    executor = None
    if int(prep_workers) > 1:
        executor = ThreadPoolExecutor(max_workers=int(prep_workers))
    try:
        for batch_wells in batches:
            if executor is None:
                batch = [prep_one(w) for w in batch_wells]
            else:
                batch = list(executor.map(prep_one, batch_wells))
            for res in run_batch_fixed_decoders(
                batch,
                dev="cuda",
                pos_mode=pos_mode,
                rate_mode=rate_mode,
                block_lag_ft=block_lag_ft,
                include_viterbi=include_viterbi,
            ):
                w = res["well"]
                if report_diag:
                    diag_by_well[w] = res.get("diag", {})
                dec = decode_arrays(res, include_viterbi=include_viterbi)
                for name, arr in dec.items():
                    preds[name][w] = arr
            torch.cuda.empty_cache()
    finally:
        if executor is not None:
            executor.shutdown(wait=True)
    return preds, time.time() - t0, diag_by_well


def add_force_wells(wells, force_text):
    out = list(wells)
    if not force_text:
        return out
    seen = set(out)
    for w in force_text.split(","):
        w = w.strip()
        if w and w not in seen:
            out.append(w)
            seen.add(w)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["screen", "full"])
    ap.add_argument("variants", nargs="*", default=["fixed45"])
    ap.add_argument("--screen-n", type=int, default=50)
    ap.add_argument("--force-wells", default="")
    ap.add_argument("--budget", type=float, default=4.8e9)
    ap.add_argument("--support-policy", choices=["endpoint", "neutral"], default="endpoint")
    ap.add_argument("--actual-dm", action="store_true")
    ap.add_argument("--sig-r", type=float, default=0.002)
    ap.add_argument("--kap-scale", type=float, default=1.0)
    ap.add_argument("--gr-policy", choices=["interp", "observed", "shortgap"], default="interp")
    ap.add_argument("--gr-gap-ft", type=float, default=40.0)
    ap.add_argument("--pos-mode", choices=["floor", "linear"], default="floor")
    ap.add_argument("--rate-mode", choices=["tri", "gaussian"], default="tri")
    ap.add_argument("--block-lag-ft", type=float, default=0.0)
    ap.add_argument("--include-viterbi", action="store_true")
    ap.add_argument("--dstep", type=float, default=0.05)
    ap.add_argument("--dhalf", type=float, default=45.0)
    ap.add_argument("--rho", type=float, default=0.20)
    ap.add_argument("--n-rates", type=int, default=41)
    ap.add_argument("--shape-block-ft", type=float, default=0.0)
    ap.add_argument("--shape-alpha", type=float, default=0.0)
    ap.add_argument("--shape-min-points", type=int, default=5)
    ap.add_argument("--prep-workers", type=int, default=1)
    ap.add_argument("--report-diag", action="store_true")
    ap.add_argument("--save", default="")
    args = ap.parse_args()

    wd, all_wells = load_fold_pure_wd()
    wells = add_force_wells(select_wells(all_wells, args.mode, args.screen_n), args.force_wells)
    print(
        "FAST wells=%d mode=%s support=%s actual_dm=%s sig_r=%g kap_scale=%g gr=%s gap=%g pos=%s rate=%s block_lag=%g viterbi=%s dstep=%g dhalf=%g rho=%g R=%d shape_block=%g shape_alpha=%g prep_workers=%d"
        % (
            len(wells),
            args.mode,
            args.support_policy,
            args.actual_dm,
            args.sig_r,
            args.kap_scale,
            args.gr_policy,
            args.gr_gap_ft,
            args.pos_mode,
            args.rate_mode,
            args.block_lag_ft,
            args.include_viterbi,
            args.dstep,
            args.dhalf,
            args.rho,
            args.n_rates,
            args.shape_block_ft,
            args.shape_alpha,
            args.prep_workers,
        ),
        flush=True,
    )
    saved = {}
    for variant in args.variants:
        preds, dt, diag = run_variant(
            wd,
            wells,
            variant,
            args.budget,
            args.support_policy,
            args.actual_dm,
            args.sig_r,
            args.kap_scale,
            args.gr_policy,
            args.gr_gap_ft,
            args.pos_mode,
            args.rate_mode,
            args.block_lag_ft,
            args.include_viterbi,
            args.dstep,
            args.dhalf,
            args.rho,
            args.n_rates,
            args.shape_block_ft,
            args.shape_alpha,
            args.shape_min_points,
            args.report_diag,
            args.prep_workers,
        )
        saved[variant] = preds
        print("\n%s %.0fs" % (variant, dt), flush=True)
        for name in decoder_names(args.include_viterbi):
            pooled, folds = pooled_score(wd, wells, preds[name])
            print(
                "%-8s pooled %.4f folds [%s]"
                % (name, pooled, " ".join("%.3f" % x for x in folds)),
                flush=True,
            )
        if args.report_diag:
            print_diag_summary(diag)
        if args.save:
            with open(args.save, "wb") as f:
                pickle.dump(saved, f)
            print("saved", args.save, flush=True)


if __name__ == "__main__":
    main()
