# Alexey Zakharov (drzoidberg), ROGII team solution.
"""Particle filtering with a kriging dip prior."""
import os, glob, argparse, warnings, time
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np, pandas as pd
from scipy.signal import savgol_filter

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "rogii-wellbore-geology-prediction")
TRAIN_DIR = os.path.join(DATA, "train")
ART = os.path.join(DATA, "model_artifacts")


_RKRIGE = None
_KAPADAPT = None
_KVAR = None
_ALPHA = 1.0
_VN = 0.002
_PN = 0.005
_CONFIGS = ()
import pandas as _pd
_GSMAP=None
_gm=os.environ.get('ROGII_GS_MAP')
if _gm:
    _t=_pd.read_parquet(_gm); _GSMAP=dict(zip(_t.well,_t.gs))
_GSROWMAP=None
_grm=os.environ.get('ROGII_GS_ROWMAP')
if _grm:
    _tr=_pd.read_parquet(_grm); _GSROWMAP={w:g.set_index('MD')['gs'] for w,g in _tr.groupby('well')}


def _smoothed_dydx(y, x, win_frac=0.04, polyorder=2):
    n = len(y)
    if n < 7:
        return np.gradient(y, x)
    win = max(7, int(win_frac * n) | 1)
    win = min(win, n if n % 2 == 1 else n - 1)
    ys = savgol_filter(y, win, min(polyorder, win - 1))
    return np.gradient(ys, x)


def run_pf(hw, n_particles, seed, target_full, kappa_full, mode="ar1", kvar_full=None, alpha=1.0, vn=0.002, pn=0.005,
           gr_off=False, gs_scale=1.0, emission="gauss", gr_smooth=1, studentt_nu=4.0,
           seg_W=40, seg_c=0.0, return_logZ=False, return_spread=False):
    """Run one particle-filter pass with a per-row dip-reversion target."""
    kn = hw[hw["TVT_input"].notna()]
    ev = hw[hw["TVT_input"].isna()]
    out_vals = hw["TVT_input"].values.astype(float).copy()
    if len(ev) == 0:
        return out_vals
    last = kn.iloc[-1]
    last_tvt = float(last["TVT_input"]); last_Z = float(last["Z"]); last_MD = float(last["MD"])


    tail = kn.tail(30)
    dt = np.diff(tail["TVT_input"].values); dz = np.diff(tail["Z"].values); dm = np.diff(tail["MD"].values)
    m = dm > 0
    ir = float(np.median((dt + dz)[m] / dm[m])) if m.sum() >= 3 else 0.0


    tw = run_pf._tw_cache
    tw_tvt = tw[0]; tw_gr = tw[1]
    tw_at_k = np.interp(kn["TVT_input"].values, tw_tvt, tw_gr)
    gs = float(np.clip(np.nanstd(kn["GR"].fillna(0).values - tw_at_k), 10., 60.)) * gs_scale
    _gsf=os.environ.get("ROGII_GS_FIXED")
    if _gsf not in (None,""): gs=float(_gsf)

    _em=os.environ.get("ROGII_EMISSION")
    if _em: emission=_em
    _snu=os.environ.get("ROGII_STUDENTT_NU")
    if _snu: studentt_nu=float(_snu)
    _sc=os.environ.get("ROGII_SEG_C")
    if _sc not in (None,""): seg_c=float(_sc)
    _sw=os.environ.get("ROGII_SEG_W")
    if _sw not in (None,""): seg_W=int(float(_sw))

    _seg_phase=int(os.environ.get("ROGII_SEG_PHASE","0"))
    _kramp=os.environ.get("ROGII_KAPPA_RAMP")
    _bias_rho=os.environ.get("ROGII_BIAS_RHO")
    _bias_n=float(os.environ.get("ROGII_BIAS_N","0.0005"))
    _gw=getattr(run_pf,"_gs_well",None)
    if _gw is not None: gs=float(_gw)*gs_scale
    _GSROWS=getattr(run_pf,"_gs_rows",None)

    N = n_particles
    rng = np.random.default_rng(seed)
    ls = last_tvt + last_Z
    pos = ls + 2.0 * rng.standard_normal(N)
    rate = ir + 0.01 * rng.standard_normal(N)
    w = np.ones(N) / N
    VN = vn; PN = pn; RP = 0.1; RR = 0.001; RESAMP = 0.5

    md_v = ev["MD"].values.astype(float); z_v = ev["Z"].values.astype(float)
    gr_interp = hw["GR"].interpolate(limit_direction="both").fillna(np.mean(tw_gr))
    gr_v = gr_interp.values.astype(float)[ev.index]
    if gr_smooth > 1:
        k = np.ones(int(gr_smooth)) / int(gr_smooth)
        gr_v = np.convolve(gr_v, k, mode="same")
    ev_idx = ev.index.to_numpy()
    rk_v = target_full[ev_idx]
    kap_v = (np.full(len(ev), float(kappa_full)) if np.ndim(kappa_full) == 0
             else np.asarray(kappa_full)[ev_idx])
    kvar_v = (np.asarray(kvar_full)[ev_idx] if kvar_full is not None else np.zeros(len(ev)))

    if len(ev) > 1 and _kramp == "up":
        kap_v = kap_v * np.linspace(0.0, 1.0, len(ev))
    elif len(ev) > 1 and _kramp == "down":
        kap_v = kap_v * np.linspace(1.0, 0.0, len(ev))

    bias = None; _brho = 0.0
    if _bias_rho is not None:
        _brho = float(_bias_rho)
        bias = (ir - rk_v[0]) + 0.005 * rng.standard_normal(N)
    gsr_v = (np.asarray(_GSROWS, float) if _GSROWS is not None and len(_GSROWS)==len(ev) else np.full(len(ev), gs))

    _skip = np.zeros(len(ev), bool)
    _GAPG = os.environ.get("ROGII_GAP_G")
    if _GAPG not in (None, ""):
        _G = float(_GAPG)
        _grnan = ~np.isfinite(hw["GR"].to_numpy(float))[ev_idx]
        i0 = 0
        while i0 < len(ev):
            if _grnan[i0]:
                j = i0
                while j + 1 < len(ev) and _grnan[j + 1]: j += 1
                if (md_v[j] - md_v[i0]) > _G: _skip[i0:j + 1] = True
                i0 = j + 1
            else:
                i0 += 1

    _JSJ = float(os.environ.get("ROGII_JUMP_SJ", "0") or 0)
    _JLAM = float(os.environ.get("ROGII_JUMP_LAMBDA", "0") or 0)
    _JP = (_JLAM / 4840.0) if _JLAM > 0 else 0.0

    _FLL = int(os.environ.get("ROGII_FLSMOOTH_L", "0") or 0)
    if _FLL > 0:
        _buf = np.zeros((_FLL, N)); _smoothed = np.full(len(ev), np.nan)

    res = np.empty(len(ev)); prev_MD = last_MD; logZ = 0.0


    wsd = np.zeros(len(ev))
    if emission in ("segcorr", "segderiv", "segrank"):
        seg_tvt = np.zeros((int(seg_W), N)); seg_obs = np.zeros(int(seg_W)); bufn = 0; _first_seg = True
    if emission == "hybrid":
        hyb_md = np.zeros(int(seg_W)); hyb_obs = np.zeros(int(seg_W)); hbufn = 0
    for i in range(len(ev)):
        dm_step = max(md_v[i] - prev_MD, 1.0)
        if mode == "kalman":

            rate = rate + VN * rng.standard_normal(N)
            P = float(rate.var())
            R = alpha * float(kvar_v[i])
            K = P / (P + R + 1e-12)
            rate = rate + K * (rk_v[i] - rate)
        elif bias is not None:


            bias = _brho * bias + _bias_n * rng.standard_normal(N)
            rate = rk_v[i] + bias
        else:

            rate = rate + kap_v[i] * (rk_v[i] - rate) + VN * rng.standard_normal(N)
            if _JSJ > 0.0 and not _skip[i]:
                _jm = rng.random(N) < _JP * dm_step
                if _jm.any(): rate[_jm] += _JSJ * rng.standard_normal(int(_jm.sum()))
        pos = pos + rate * dm_step + PN * rng.standard_normal(N)
        tvt_p = pos - z_v[i]
        tvt_p = np.clip(tvt_p, tw_tvt[0] - 100, tw_tvt[-1] + 100)
        pos = tvt_p + z_v[i]
        if _skip[i]:
            tp = pos - z_v[i]; res[i] = float(np.dot(w, tp))
            wsd[i] = float(np.sqrt(max(np.dot(w, (tp - res[i]) ** 2), 0.0)))
            prev_MD = md_v[i]; continue
        if gr_off:

            res[i] = float(np.mean(pos - z_v[i]))
        elif emission in ("segcorr", "segderiv", "segrank"):


            seg_tvt[bufn] = tvt_p; seg_obs[bufn] = gr_v[i]; bufn += 1
            _trig = (int(seg_W) - _seg_phase) if _first_seg else int(seg_W)
            if bufn >= _trig or i == len(ev) - 1:
                Wn = bufn; _first_seg = False
                exp = np.interp(seg_tvt[:Wn].ravel(), tw_tvt, tw_gr).reshape(Wn, N)
                obs = seg_obs[:Wn]
                if emission == "segderiv" and Wn >= 3:
                    o = np.diff(obs); e = np.diff(exp, axis=0)
                    o = o - o.mean(); e = e - e.mean(axis=0, keepdims=True)
                elif emission == "segrank" and Wn >= 3:
                    o = obs.argsort().argsort().astype(float); o = o - o.mean()
                    e = np.argsort(np.argsort(exp, axis=0), axis=0).astype(float); e = e - e.mean(axis=0, keepdims=True)
                else:
                    o = obs - obs.mean(); e = exp - exp.mean(axis=0, keepdims=True)
                denom = np.sqrt((o @ o) * (e * e).sum(axis=0)) + 1e-9
                corr = (o[:, None] * e).sum(axis=0) / denom
                lk = np.maximum(np.exp(seg_c * corr), 1e-300)
                w = w * lk; ws = w.sum(); w = w / ws if ws > 0 else np.ones(N) / N
                if 1.0 / (w ** 2).sum() < RESAMP * N:
                    cum = np.cumsum(w); u0 = rng.uniform(0, 1.0 / N)
                    idx = np.clip(np.searchsorted(cum, u0 + np.arange(N) / N), 0, N - 1)
                    pos = pos[idx] + RP * rng.standard_normal(N)
                    rate = rate[idx] + RR * rng.standard_normal(N)
                    if bias is not None: bias = bias[idx]
                    w = np.ones(N) / N
                    if _FLL > 0: _buf[:] = _buf[:, idx]
                bufn = 0
            if _FLL > 0:
                if i >= _FLL: _smoothed[i - _FLL] = float(np.dot(w, _buf[i % _FLL])) - z_v[i - _FLL]
                _buf[i % _FLL] = pos
            res[i] = float(np.dot(w, pos - z_v[i]))
        else:
            eg = np.interp(tvt_p, tw_tvt, tw_gr)
            d = (gr_v[i] - eg) / gsr_v[i]
            if emission == "studentt":
                lk = (1.0 + d ** 2 / studentt_nu) ** (-(studentt_nu + 1.0) / 2.0)
            elif emission == "l1":
                lk = np.exp(-np.minimum(np.abs(d), 25.0))
            else:
                lk = np.exp(-0.5 * np.minimum(d ** 2, 600.))
            lk = np.maximum(lk, 1e-300)
            ws = float((w * lk).sum())
            logZ += np.log(max(ws, 1e-300))
            w = w * lk
            ws = w.sum(); w = w / ws if ws > 0 else np.ones(N) / N
            if _FLL > 0:
                if i >= _FLL: _smoothed[i - _FLL] = float(np.dot(w, _buf[i % _FLL])) - z_v[i - _FLL]
                _buf[i % _FLL] = pos
            if 1.0 / (w ** 2).sum() < RESAMP * N:
                cum = np.cumsum(w); u0 = rng.uniform(0, 1.0 / N)
                idx = np.clip(np.searchsorted(cum, u0 + np.arange(N) / N), 0, N - 1)
                pos = pos[idx] + RP * rng.standard_normal(N)
                rate = rate[idx] + RR * rng.standard_normal(N)
                if bias is not None: bias = bias[idx]
                w = np.ones(N) / N
                if _FLL > 0: _buf[:] = _buf[:, idx]
            if emission == "hybrid":


                hyb_md[hbufn] = md_v[i]; hyb_obs[hbufn] = gr_v[i]; hbufn += 1
                if hbufn == int(seg_W) or i == len(ev) - 1:
                    Wn = hbufn
                    dmd = md_v[i] - hyb_md[:Wn]
                    tvt_recon = tvt_p[None, :] - rate[None, :] * dmd[:, None]
                    exp = np.interp(tvt_recon.ravel(), tw_tvt, tw_gr).reshape(Wn, N)
                    o = hyb_obs[:Wn] - hyb_obs[:Wn].mean()
                    e = exp - exp.mean(axis=0, keepdims=True)
                    denom = np.sqrt((o @ o) * (e * e).sum(axis=0)) + 1e-9
                    corr = (o[:, None] * e).sum(axis=0) / denom
                    w = w * np.maximum(np.exp(seg_c * corr), 1e-300)
                    ws = w.sum(); w = w / ws if ws > 0 else np.ones(N) / N
                    if 1.0 / (w ** 2).sum() < RESAMP * N:
                        cum = np.cumsum(w); u0 = rng.uniform(0, 1.0 / N)
                        idx = np.clip(np.searchsorted(cum, u0 + np.arange(N) / N), 0, N - 1)
                        pos = pos[idx] + RP * rng.standard_normal(N)
                        rate = rate[idx] + RR * rng.standard_normal(N)
                        w = np.ones(N) / N
                    hbufn = 0
            res[i] = float(np.dot(w, pos - z_v[i]))
        tp_sp = pos - z_v[i]
        wsd[i] = float(np.sqrt(max(np.dot(w, (tp_sp - res[i]) ** 2), 0.0)))
        prev_MD = md_v[i]
    if _FLL > 0:
        for k in range(max(0, len(ev) - _FLL), len(ev)):
            _smoothed[k] = float(np.dot(w, _buf[k % _FLL])) - z_v[k]
        out_vals[list(ev.index)] = _smoothed
    else:
        out_vals[list(ev.index)] = res
    if return_spread:
        wsd_full = np.full(len(out_vals), np.nan); wsd_full[list(ev.index)] = wsd
        return out_vals, wsd_full
    return (out_vals, logZ) if return_logZ else out_vals


def process_well(args):
    wid, n_seeds, n_particles = args
    hw = pd.read_csv(os.path.join(TRAIN_DIR, f"{wid}__horizontal_well.csv"))
    hw = hw.sort_values("MD").reset_index(drop=True)
    tw = pd.read_csv(os.path.join(TRAIN_DIR, f"{wid}__typewell.csv")).sort_values("TVT")
    run_pf._gs_well = (_GSMAP.get(wid) if _GSMAP is not None else None)
    run_pf._gs_rows = None
    if _GSROWMAP is not None and wid in _GSROWMAP:
        _evmd = hw.loc[hw['TVT_input'].isna(), 'MD'].to_numpy()
        _ser = _GSROWMAP[wid].reindex(_evmd)
        run_pf._gs_rows = _ser.fillna(_ser.median()).to_numpy()
    run_pf._tw_cache = (tw["TVT"].values.astype(float),
                        tw["GR"].fillna(tw["GR"].mean()).values.astype(float))
    ev_mask = hw["TVT_input"].isna().to_numpy()
    truth = hw["TVT"].to_numpy().astype(float)
    if ev_mask.sum() == 0:
        return wid, None

    rk = _RKRIGE.get(wid)
    if rk is None:
        return wid, None
    nrow = len(hw)
    targets = {"flat": np.zeros(nrow), "fused": rk}
    if any(m == "oracle" for _, m, _, _ in _CONFIGS) and "BUDA" in hw.columns:
        targets["oracle"] = _smoothed_dydx(hw["BUDA"].to_numpy(float), hw["MD"].to_numpy(float))
    kap_adapt = _KAPADAPT.get(wid) if _KAPADAPT is not None else None


    pref = hw["TVT_input"].notna().to_numpy()
    bias_w = 0.0
    if pref.sum() >= 8:
        md = hw["MD"].to_numpy(float)
        p_obs = hw["TVT_input"].to_numpy(float) + hw["Z"].to_numpy(float)
        g_obs = _smoothed_dydx(p_obs[pref], md[pref])
        g_kr_pref = rk[pref]
        bias_w = float(np.median(g_obs - g_kr_pref))

    kvar_w = _KVAR.get(wid) if _KVAR is not None else None
    out = {"wid": wid, "n_eval": int(ev_mask.sum()), "bias_w": bias_w}
    for name, mode, kappa_spec, debias in _CONFIGS:
        tgt = rk if mode in ("fused", "kalman") else targets.get(mode, np.zeros(nrow))
        if debias:
            tgt = tgt + debias * bias_w
        kappa_full = kap_adapt if kappa_spec == "adaptive" else kappa_spec
        if kappa_spec == "adaptive" and kap_adapt is None:
            continue
        if mode == "kalman":
            pf_mode = "kalman"; alpha_eff = _ALPHA * float(kappa_spec)
        else:
            pf_mode = "ar1"; alpha_eff = _ALPHA
        if _DUMP_PER_ROW:

            pairs = [run_pf(hw, n_particles, s, tgt, kappa_full, mode=pf_mode, kvar_full=kvar_w,
                            alpha=alpha_eff, vn=_VN, pn=_PN, return_spread=True)
                     for s in range(n_seeds)]
            preds = np.stack([p[0] for p in pairs], 0)
            wsd_mean = np.nanmean(np.stack([p[1] for p in pairs], 0), 0)
        else:
            preds = np.stack([run_pf(hw, n_particles, s, tgt, kappa_full,
                                     mode=pf_mode, kvar_full=kvar_w, alpha=alpha_eff, vn=_VN, pn=_PN)
                              for s in range(n_seeds)], 0)
        pred = preds.mean(0)
        disag = preds.std(0, ddof=1) if preds.shape[0] >= 2 else np.zeros(nrow)
        err = pred[ev_mask] - truth[ev_mask]
        out[f"sse__{name}"] = float(np.sum(err ** 2))
        out[f"rmse__{name}"] = float(np.sqrt(np.mean(err ** 2)))
        if _DUMP_PER_ROW:
            out.setdefault("_per_row", {})[name] = (pred.astype(np.float32),
                                                    disag.astype(np.float32),
                                                    wsd_mean.astype(np.float32))
    if _DUMP_PER_ROW:
        out["_md"] = hw["MD"].to_numpy(np.float32)
    return wid, out


def _init_worker(rk_dict, kap_dict, kvar_dict, alpha, vn, pn, configs, dump_per_row=False):
    global _RKRIGE, _KAPADAPT, _KVAR, _ALPHA, _VN, _PN, _CONFIGS, _DUMP_PER_ROW
    _RKRIGE = rk_dict; _KAPADAPT = kap_dict; _KVAR = kvar_dict
    _ALPHA = alpha; _VN = vn; _PN = pn; _CONFIGS = tuple(configs)
    _DUMP_PER_ROW = bool(dump_per_row)


_DUMP_PER_ROW = False


def load_folds(regime):
    fn = "cv_folds_gkf.parquet" if regime == "gkf" else "cv_folds.parquet"
    f = pd.read_parquet(os.path.join(ART, fn))[["well", "fold"]]
    return dict(zip(f["well"], f["fold"]))


def build_rkrige(regime, kappa_lo=0.002, kappa_hi=0.012, grad_winfrac=0.04, oof_suffix=""):
    """r_krige per well = smoothed d(buda_est)/dMD along full-res path."""
    oof = pd.read_parquet(os.path.join(ART, f"phase4_krige_oof_{regime}{oof_suffix}.parquet"))
    if os.environ.get("ROGII_KAPPA_SRC") == "gradspread":
        gsdf = pd.read_parquet(os.path.join(ART, f"phase4_gradspread_{regime}.parquet"))
        oof = oof.merge(gsdf, on=["well", "MD"], how="left")
        sig = oof["grad_spread"].fillna(oof["grad_spread"].median()).to_numpy()
        order = np.argsort(np.argsort(sig)); qrank = order / max(len(sig) - 1, 1)
    else:
        sig = np.sqrt(np.clip(oof["kvar"].to_numpy(), 0, None))
        if os.environ.get("ROGII_KAPPA_FROZEN", "1") != "0":
            _cdf = os.path.join(ART, f"sigma_cdf_{regime}{oof_suffix}.npy")
            if not os.path.exists(_cdf):
                _cdf = os.path.join(ART, f"sigma_cdf_{regime}.npy")
            knots = np.load(_cdf)
            qrank = np.clip(np.interp(sig, knots, np.linspace(0.0, 1.0, len(knots))), 0.0, 1.0)
        else:
            order = np.argsort(np.argsort(sig)); qrank = order / max(len(sig) - 1, 1)

    _kmid = os.environ.get("ROGII_KAPPA_MID")
    if _kmid not in (None, ""):
        _ka = np.interp(qrank, [0.0, 0.5, 1.0], [kappa_hi, float(_kmid), kappa_lo])
    else:
        _ka = kappa_lo + (kappa_hi - kappa_lo) * (1.0 - qrank)
    oof = oof.assign(_kappa=_ka)
    rk, kap, kvar = {}, {}, {}
    for wid, g in oof.groupby("well", sort=False):
        g = g.sort_values("MD")
        rk[wid] = _smoothed_dydx(g["buda_est"].to_numpy(), g["MD"].to_numpy(), win_frac=grad_winfrac)
        kap[wid] = g["_kappa"].to_numpy()
        kvar[wid] = np.clip(g["kvar"].to_numpy(), 1e-9, None)
    return rk, kap, kvar


def calibrate_alpha(rk, kvar, grad_winfrac=0.04):
    """Calibrate kriging-gradient uncertainty on held-out training wells."""
    z = []
    for wid in rk:
        csv = os.path.join(TRAIN_DIR, f"{wid}__horizontal_well.csv")
        if not os.path.exists(csv):
            continue
        d = pd.read_csv(csv, usecols=["MD", "BUDA"]).sort_values("MD")
        b = d["BUDA"].to_numpy(float)
        if not np.isfinite(b).all() or len(b) != len(rk[wid]):
            continue
        g_true = _smoothed_dydx(b, d["MD"].to_numpy(float), win_frac=grad_winfrac)
        e = g_true - rk[wid]
        z.append(e / np.sqrt(kvar[wid]))
    z = np.concatenate(z)
    med = np.median(z)
    mad = np.median(np.abs(z - med))
    alpha = float((1.4826 * mad) ** 2)
    return alpha, len(z)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regime", default="gkf")
    ap.add_argument("--flat_kappa", type=float, default=0.002, help="flat baseline ~ MOM=0.998")
    ap.add_argument("--fused_kappas", type=float, nargs="*", default=[0.005, 0.01, 0.02, 0.05])
    ap.add_argument("--skip_flat", action="store_true", help="omit flat (regime-independent; compute once)")
    ap.add_argument("--oracle_kappas", type=float, nargs="*", default=[],
                    help="add oracle-gradient configs (d(BUDA_true)/dMD) at these kappas — headroom probe")
    ap.add_argument("--adaptive", action="store_true",
                    help="add sigma-adaptive-kappa fused config (kappa in [kappa_lo,kappa_hi] by sigma rank)")
    ap.add_argument("--kappa_lo", type=float, default=0.002)
    ap.add_argument("--kappa_hi", type=float, default=0.012)
    ap.add_argument("--debias_lambdas", type=float, nargs="*", default=[],
                    help="add prefix-gradient-debiased adaptive-kappa fused configs at these shrink lambdas")
    ap.add_argument("--grad_winfrac", type=float, default=0.04,
                    help="Savitzky-Golay window (fraction of path) for the kriged-gradient d(BUDA_est)/dMD")
    ap.add_argument("--kalman", action="store_true",
                    help="add Kalman-gain dip-fusion config: K=P/(P+alpha*kvar), alpha calibrated on train OOF")
    ap.add_argument("--alpha", type=float, default=0.0,
                    help="override calibrated alpha for the Kalman R scale (0 = use calibrated)")
    ap.add_argument("--vn", type=float, default=0.002, help="PF dip-rate process noise")
    ap.add_argument("--pn", type=float, default=0.005, help="PF position/level process noise")
    ap.add_argument("--oof_suffix", default="", help="suffix for phase4_krige_oof_<regime><suffix>.parquet")
    ap.add_argument("--alpha_scales", type=float, nargs="*", default=[1.0],
                    help="multipliers on the calibrated alpha (sensitivity check, NOT score-tuning)")
    ap.add_argument("--n_seeds", type=int, default=24)
    ap.add_argument("--n_particles", type=int, default=500)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2)))
    ap.add_argument("--max_wells", type=int, default=0)
    ap.add_argument("--dump_per_row", action="store_true",
                    help="Also dump per-row (well, MD, pf_pred, pf_disag) to phase4_pf_preds_<regime>.parquet")
    args = ap.parse_args()

    fold_of = load_folds(args.regime)
    rk, kap, kvar = build_rkrige(args.regime, kappa_lo=args.kappa_lo, kappa_hi=args.kappa_hi,
                                 grad_winfrac=args.grad_winfrac, oof_suffix=args.oof_suffix)
    alpha = 1.0
    if args.kalman:
        if args.alpha > 0:
            alpha = args.alpha; nz = 0
        else:
            alpha, nz = calibrate_alpha(rk, kvar, grad_winfrac=args.grad_winfrac)
        print(f"[Kalman] calibrated alpha = {alpha:.5g}  (from {nz} train OOF rows; R = alpha*kvar)")
    wids = sorted(rk.keys())
    if args.max_wells > 0:
        per = max(1, args.max_wells // 5); keep, cnt = [], {f: 0 for f in range(5)}
        for w in wids:
            f = fold_of.get(w)
            if f is not None and cnt[f] < per:
                keep.append(w); cnt[f] += 1
        wids = keep

    configs = [] if args.skip_flat else [("flat", "flat", args.flat_kappa, 0.0)]
    configs += [(f"fused_k{k:g}", "fused", k, 0.0) for k in args.fused_kappas]
    configs += [(f"oracle_k{k:g}", "oracle", k, 0.0) for k in args.oracle_kappas]
    arange = f"{args.kappa_lo:g}-{args.kappa_hi:g}"
    if args.adaptive:
        configs += [(f"adaptive_{arange}", "fused", "adaptive", 0.0)]
    configs += [(f"debias{L:g}_{arange}", "fused", "adaptive", L) for L in args.debias_lambdas]
    if args.kalman:

        configs += [(f"kalman_a{s:g}", "kalman", s, 0.0) for s in args.alpha_scales]
    print(f"regime={args.regime} wells={len(wids)} configs={[c[0] for c in configs]} "
          f"grad_winfrac={args.grad_winfrac} vn={args.vn} pn={args.pn} seeds={args.n_seeds} particles={args.n_particles} workers={args.workers}")

    tasks = [(w, args.n_seeds, args.n_particles) for w in wids]
    results, t0, done = [], time.time(), 0
    with ProcessPoolExecutor(max_workers=args.workers,
                             initializer=_init_worker,
                             initargs=(rk, kap, kvar, alpha, args.vn, args.pn, configs,
                                       bool(args.dump_per_row))) as ex:
        futs = {ex.submit(process_well, t): t[0] for t in tasks}
        for fut in as_completed(futs):
            wid, rec = fut.result(); done += 1
            if rec is not None:
                rec["fold"] = fold_of.get(wid, -1); results.append(rec)
            if done % 50 == 0 or done == len(tasks):
                el = time.time() - t0
                print(f"  {done}/{len(tasks)}  {el:.0f}s ({el/done:.2f}s/well)")


    per_row_records = []
    summary_records = []
    for rec in results:
        if "_per_row" in rec:
            mds = rec.pop("_md", None)
            blobs = rec.pop("_per_row", {})
            if mds is not None:
                for cfg_name, (pred, disag, wsd) in blobs.items():
                    per_row_records.append((rec["wid"], cfg_name, mds, pred, disag, wsd))
        summary_records.append(rec)
    df = pd.DataFrame(summary_records)
    df.to_csv(os.path.join(ART, f"phase4_pf_{args.regime}.csv"), index=False)
    if per_row_records:

        cfg_long = {}
        for wid, cfg_name, mds, pred, disag, wsd in per_row_records:
            cfg_long.setdefault(cfg_name, []).append(pd.DataFrame({
                "well": wid, "MD": mds,
                f"pf_pred__{cfg_name}": pred,
                f"pf_disag__{cfg_name}": disag,
                f"pf_wsd__{cfg_name}": wsd,
            }))
        per_cfg = {c: pd.concat(dfs, ignore_index=True) for c, dfs in cfg_long.items()}
        merged = None
        for c, frame in per_cfg.items():
            merged = frame if merged is None else merged.merge(frame, on=["well", "MD"], how="outer")
        out_path = os.path.join(ART, f"phase4_pf_preds_{args.regime}{args.oof_suffix}.parquet")
        merged.to_parquet(out_path, index=False)
        print(f"[per-row] wrote {out_path}  rows={len(merged):,}  cols={merged.columns.tolist()}")
    print(f"\n==== {args.regime} pooled tail RMSE ====")
    print("config".ljust(16) + "".join(f"f{f}".rjust(9) for f in range(5)) +
          "POOLED".rjust(10) + "p50".rjust(8) + "p90".rjust(8))
    for name, _, _, _ in configs:
        sse = df[f"sse__{name}"]; nev = df["n_eval"]
        row = "".join(f"{np.sqrt(sse[df.fold==f].sum()/nev[df.fold==f].sum()):9.3f}" for f in range(5))
        pooled = np.sqrt(sse.sum() / nev.sum())
        p50 = df[f"rmse__{name}"].median(); p90 = df[f"rmse__{name}"].quantile(0.9)
        print(f"{name.ljust(16)}{row}{pooled:10.3f}{p50:8.2f}{p90:8.2f}")


if __name__ == "__main__":
    main()
