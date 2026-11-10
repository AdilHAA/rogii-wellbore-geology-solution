# Alexey Zakharov (drzoidberg), ROGII team solution.
"""Anisotropic ordinary kriging with moving neighborhoods."""
import os
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.optimize import curve_fit

try:
    import cupy as cp
    _HAVE_CUPY = True
    _CUDA_OK = cp.cuda.runtime.getDeviceCount() > 0
except Exception:
    _HAVE_CUPY = False
    _CUDA_OK = False


def median_cell_decluster(x, y, z, cell=400.0):
    """Median per cell -> (cx, cy, cz)."""
    ci = np.round(x / cell).astype(int)
    cj = np.round(y / cell).astype(int)
    df = pd.DataFrame({"ci": ci, "cj": cj, "x": x, "y": y, "z": z})
    g = (df.groupby(["ci", "cj"])
           .agg(x=("x", "median"), y=("y", "median"), z=("z", "median"))
           .reset_index())
    return g["x"].to_numpy(), g["y"].to_numpy(), g["z"].to_numpy()


def journel_cell_weights(x, y, cell):
    """Cell-declustering weights."""
    ci = np.round(x / cell).astype(np.int64)
    cj = np.round(y / cell).astype(np.int64)
    key = ci.astype(np.int64) * 4_000_003 + cj
    _, inv, counts = np.unique(key, return_inverse=True, return_counts=True)
    w = 1.0 / counts[inv]
    return w * (len(x) / w.sum())


def select_journel_cell(x, y, z, candidates):
    """Select cell size using the declustered mean."""
    naive = float(np.mean(z))
    table = []
    for c in candidates:
        w = journel_cell_weights(x, y, c)
        table.append((float(c), float(np.sum(w * z) / np.sum(w))))
    best_cell = min(table, key=lambda t: t[1])[0]
    return best_cell, naive, table


def fit_planar_trend(x, y, z, w=None):
    xm, ym = float(x.mean()), float(y.mean())
    A = np.c_[np.ones_like(x), x - xm, y - ym]
    if w is None:
        beta, *_ = np.linalg.lstsq(A, z, rcond=None)
    else:
        sw = np.sqrt(w)
        beta, *_ = np.linalg.lstsq(A * sw[:, None], z * sw, rcond=None)
    return float(beta[0]), float(beta[1]), float(beta[2]), xm, ym


def _eval_trend(beta_xm_ym, x, y):
    b0, b1, b2, xm, ym = beta_xm_ym
    return b0 + b1 * (x - xm) + b2 * (y - ym)


def _quad_add(quad, xm, ym, x, y):
    """Quadratic trend contribution (0 if quad is None)."""
    if quad is None:
        return 0.0 * x if hasattr(x, "shape") else 0.0
    dx = x - xm; dy = y - ym
    return quad[0] * dx * dx + quad[1] * dy * dy + quad[2] * dx * dy


def gamma_exp(h_norm, nugget, sill):
    return nugget + sill * (1.0 - np.exp(-3.0 * h_norm))


def fit_anisotropic_variogram(u, v, resid, n_lags=24, max_pairs=200_000,
                              a_strike_over_a_dip=3.0, rng_seed=42, weights=None):
    rng = np.random.default_rng(rng_seed)
    n = len(u)
    pw = None
    if n * (n - 1) // 2 > max_pairs:
        m = int(np.sqrt(2 * max_pairs)) + 1
        if weights is None:
            idx = rng.choice(n, m, replace=False)
        else:

            p = weights / weights.sum()
            idx = rng.choice(n, m, replace=False, p=p)
        u = u[idx]; v = v[idx]; resid = resid[idx]
        if weights is not None:
            pw = weights[idx]
        n = len(u)
    elif weights is not None:
        pw = weights
    iu = np.triu_indices(n, k=1)
    du = u[:, None] - u[None, :]; dv = v[:, None] - v[None, :]
    dr = resid[:, None] - resid[None, :]
    du_p = du[iu]; dv_p = dv[iu]; dr_p = dr[iu]
    if pw is not None:
        wpair = (pw[:, None] * pw[None, :])[iu]
    h_iso = np.sqrt(du_p ** 2 + dv_p ** 2)
    max_h = float(np.percentile(h_iso, 80))
    if max_h <= 0:
        return dict(nugget=0.0, sill=float(0.5 * np.var(resid)), a_dip=1.0, a_strike=1.0)
    bins = np.linspace(0, max_h, n_lags + 1)
    centers = 0.5 * (bins[:-1] + bins[1:])
    g_emp = np.zeros(n_lags); n_p = np.zeros(n_lags)
    for i in range(n_lags):
        m = (h_iso >= bins[i]) & (h_iso < bins[i + 1])
        if m.sum() > 0:
            if pw is not None:
                ww = wpair[m]
                g_emp[i] = 0.5 * float(np.sum(ww * dr_p[m] ** 2) / np.sum(ww))
            else:
                g_emp[i] = 0.5 * float(np.mean(dr_p[m] ** 2))
            n_p[i] = int(m.sum())
    valid = n_p > 20
    if valid.sum() < 4:
        return dict(nugget=0.0, sill=float(np.var(resid)),
                    a_dip=max_h * 0.3, a_strike=max_h * 0.3 * a_strike_over_a_dip)
    sill_est = float(np.var(resid))
    try:
        def model(h, c0, c1, a):
            return c0 + c1 * (1.0 - np.exp(-3.0 * h / np.maximum(a, 1.0)))
        popt, _ = curve_fit(
            model, centers[valid], g_emp[valid],
            p0=[0.0, sill_est, max_h * 0.3],
            bounds=([0, 0, 100.0], [sill_est * 2, sill_est * 5, max_h * 5.0]),
            maxfev=2000,
        )
        c0, c1, a_iso = popt
    except Exception:
        c0, c1, a_iso = 0.0, sill_est, max_h * 0.3
    return dict(nugget=float(c0), sill=float(c1),
                a_dip=float(a_iso), a_strike=float(a_iso * a_strike_over_a_dip))


class KrigeSurface:
    """Anisotropic ordinary kriging, moving neighborhood, planar trend."""

    def __init__(self, train_x, train_y, train_z,
                 well_id_per_pick=None, M=80, decluster_cell=400.0,
                 a_strike_over_a_dip=3.0, device=None, rng_seed=42,
                 journel_cell=None, trend_order=None, drift_mode=None):
        train_x = np.asarray(train_x, float)
        train_y = np.asarray(train_y, float)
        train_z = np.asarray(train_z, float)
        if well_id_per_pick is None:
            cx, cy, cz = median_cell_decluster(train_x, train_y, train_z, decluster_cell)
            wid_dec = None
        else:
            well_id_per_pick = np.asarray(well_id_per_pick)
            cxL, cyL, czL, widL = [], [], [], []
            for w in pd.unique(well_id_per_pick):
                m = well_id_per_pick == w
                ax, ay, az = median_cell_decluster(train_x[m], train_y[m], train_z[m], decluster_cell)
                cxL.append(ax); cyL.append(ay); czL.append(az)
                widL.append(np.full(len(ax), w))
            cx = np.concatenate(cxL); cy = np.concatenate(cyL)
            cz = np.concatenate(czL); wid_dec = np.concatenate(widL)


        if journel_cell is None:
            self.journel_cell = None
            w_dec = None
        else:
            if journel_cell == "auto":
                span = max(cx.max() - cx.min(), cy.max() - cy.min())
                cands = np.geomspace(max(decluster_cell, 500.0), span / 4.0, 10)
                jc, naive, _ = select_journel_cell(cx, cy, cz, cands)
                self.journel_cell = float(jc)
            else:
                self.journel_cell = float(journel_cell)
            w_dec = journel_cell_weights(cx, cy, self.journel_cell)
        self.decluster_weights = w_dec
        b0, b1, b2, xm, ym = fit_planar_trend(cx, cy, cz, w=w_dec)
        self.beta = (b0, b1, b2, xm, ym)
        trend_at = _eval_trend(self.beta, cx, cy)
        residuals = cz - trend_at


        if trend_order is None:
            trend_order = int(os.environ.get("ROGII_TREND_ORDER", "1"))
        self.trend_order = int(trend_order)


        if drift_mode is None:
            drift_mode = os.environ.get("ROGII_DRIFT_MODE", "global")
        self.drift_mode = str(drift_mode)
        self.quad = None
        if self.trend_order >= 2:
            dx = cx - xm; dy = cy - ym
            Aq = np.c_[dx * dx, dy * dy, dx * dy]
            sw = np.sqrt(w_dec) if w_dec is not None else np.ones(len(cx))
            cq, *_ = np.linalg.lstsq(Aq * sw[:, None], residuals * sw, rcond=None)
            self.quad = (float(cq[0]), float(cq[1]), float(cq[2]))
            residuals = residuals - Aq @ cq
        gn = float(np.hypot(b1, b2)) + 1e-12
        self.u_hat = (float(b1 / gn), float(b2 / gn))
        self.v_hat = (-self.u_hat[1], self.u_hat[0])
        u = (cx - xm) * self.u_hat[0] + (cy - ym) * self.u_hat[1]
        v = (cx - xm) * self.v_hat[0] + (cy - ym) * self.v_hat[1]
        var = fit_anisotropic_variogram(u, v, residuals, a_strike_over_a_dip=a_strike_over_a_dip,
                                        rng_seed=rng_seed, weights=w_dec)
        self.nugget = var["nugget"]; self.sill = var["sill"]
        self.a_dip = var["a_dip"]; self.a_strike = var["a_strike"]


        import os as _os
        self.ridge_rel = float(_os.environ.get("ROGII_NUGGET_REL", "1e-4"))
        self.nugget_eff = max(self.nugget, self.ridge_rel * self.sill)
        u_resc = u / self.a_dip
        v_resc = v / self.a_strike
        self.tree = cKDTree(np.c_[u_resc, v_resc])
        self.cx = cx; self.cy = cy; self.cz = cz
        self.residuals = residuals
        self.well_id = wid_dec
        self.M = M
        self.n_declustered = len(cx)

        if device is None:
            self.device = "cuda" if (_HAVE_CUPY and _CUDA_OK) else "cpu"
        else:
            self.device = device

        if self.device == "cuda" and _HAVE_CUPY:
            self._cx_t = cp.asarray(self.cx.astype(np.float32))
            self._cy_t = cp.asarray(self.cy.astype(np.float32))
            self._resid_t = cp.asarray(self.residuals.astype(np.float32))

    def _h_norm_pairs(self, du, dv):
        uu = du * self.u_hat[0] + dv * self.u_hat[1]
        vv = du * self.v_hat[0] + dv * self.v_hat[1]
        return np.sqrt((uu / self.a_dip) ** 2 + (vv / self.a_strike) ** 2)

    def predict(self, qx, qy, self_wid=None):
        qx = np.asarray(qx, float); qy = np.asarray(qy, float)
        n_q = len(qx)
        if n_q == 0:
            return np.zeros(0), np.zeros(0)
        if self.device == "cuda" and _HAVE_CUPY:
            try:
                return self._predict_gpu(qx, qy, self_wid)
            except Exception as e:

                pass
        return self._predict_cpu(qx, qy, self_wid)

    def predict_batch(self, qx, qy, self_wids=None, chunk_size=60_000):
        """Batched predict accepting per-query self_wid array."""
        qx = np.asarray(qx, float); qy = np.asarray(qy, float)
        n_q = len(qx)
        if n_q == 0:
            return np.zeros(0), np.zeros(0)
        if self_wids is not None and np.ndim(self_wids) == 0:
            sw_arr = None
            scalar_sw = self_wids
        elif self_wids is not None:
            sw_arr = np.asarray(self_wids)
            scalar_sw = None
        else:
            sw_arr = None
            scalar_sw = None
        est = np.zeros(n_q); var = np.zeros(n_q)
        for s in range(0, n_q, chunk_size):
            e = min(s + chunk_size, n_q)
            ch_sw = scalar_sw if sw_arr is None else sw_arr[s:e]
            if self.device == "cuda" and _HAVE_CUPY:
                try:
                    est[s:e], var[s:e] = self._predict_gpu(qx[s:e], qy[s:e], ch_sw)
                    continue
                except Exception:
                    pass
            est[s:e], var[s:e] = self._predict_cpu(qx[s:e], qy[s:e], ch_sw)
        return est, var


    def _predict_cpu(self, qx, qy, self_wid):
        n_q = len(qx)
        sw_arr = np.asarray(self_wid) if (self_wid is not None and np.ndim(self_wid) > 0) else None
        sw_scalar = self_wid if (self_wid is not None and np.ndim(self_wid) == 0) else None
        u_q = (qx - self.beta[3]) * self.u_hat[0] + (qy - self.beta[4]) * self.u_hat[1]
        v_q = (qx - self.beta[3]) * self.v_hat[0] + (qy - self.beta[4]) * self.v_hat[1]
        uq_r = u_q / self.a_dip; vq_r = v_q / self.a_strike
        tr_all = _eval_trend(self.beta, qx, qy) + _quad_add(self.quad, self.beta[3], self.beta[4], qx, qy)
        K_extra = self.M + (40 if (self_wid is not None and self.well_id is not None) else 0)
        K_extra = min(K_extra, self.n_declustered)
        dists, idx = self.tree.query(np.c_[uq_r, vq_r], k=K_extra, workers=-1)
        dists = np.atleast_2d(dists); idx = np.atleast_2d(idx)
        est = np.zeros(n_q); var = np.zeros(n_q)
        sill_full = self.sill + self.nugget_eff
        for i in range(n_q):
            ii = idx[i]
            if self_wid is not None and self.well_id is not None:
                sw_i = sw_arr[i] if sw_arr is not None else sw_scalar
                keep = self.well_id[ii] != sw_i
                ii = ii[keep][:self.M]
            else:
                ii = ii[:self.M]
            M_eff = len(ii)
            if M_eff < 2:
                est[i] = tr_all[i]
                var[i] = sill_full
                continue
            cxn = self.cx[ii]; cyn = self.cy[ii]; rn = self.residuals[ii]
            du_nn = cxn[:, None] - cxn[None, :]
            dv_nn = cyn[:, None] - cyn[None, :]
            h_nn = self._h_norm_pairs(du_nn, dv_nn)

            C_mat = self.sill * np.exp(-3.0 * h_nn)
            C_mat[np.diag_indices(M_eff)] += self.nugget_eff
            du_qn = qx[i] - cxn; dv_qn = qy[i] - cyn
            h_qn = self._h_norm_pairs(du_qn, dv_qn)
            c_vec = self.sill * np.exp(-3.0 * h_qn)
            trend_q = tr_all[i]
            if self.drift_mode == "uk1":


                zn = self.cz[ii]
                dxn = cxn - qx[i]; dyn = cyn - qy[i]
                sx = dxn.std() + 1e-9; sy = dyn.std() + 1e-9
                F = np.column_stack([np.ones(M_eff), dxn / sx, dyn / sy])
                P = F.shape[1]
                A = np.zeros((M_eff + P, M_eff + P))
                A[:M_eff, :M_eff] = C_mat
                A[:M_eff, M_eff:] = F; A[M_eff:, :M_eff] = F.T
                b = np.zeros(M_eff + P)
                b[:M_eff] = c_vec; b[M_eff] = 1.0
                try:
                    wfull = np.linalg.solve(A, b)
                    w = wfull[:M_eff]
                    if not np.all(np.isfinite(w)) or np.abs(w).sum() > 50.0:
                        raise np.linalg.LinAlgError
                    est[i] = float(np.sum(w * zn))
                    var[i] = max(0.0, sill_full - float(np.sum(w * c_vec)) - float(wfull[M_eff]))
                except np.linalg.LinAlgError:

                    est[i] = trend_q + float(rn.mean()); var[i] = sill_full
                continue
            A = np.zeros((M_eff + 1, M_eff + 1))
            A[:M_eff, :M_eff] = C_mat
            A[:M_eff, M_eff] = 1.0; A[M_eff, :M_eff] = 1.0
            b = np.zeros(M_eff + 1)
            b[:M_eff] = c_vec; b[M_eff] = 1.0
            try:
                wfull = np.linalg.solve(A, b)
            except np.linalg.LinAlgError:
                est[i] = trend_q + float(rn.mean())
                var[i] = sill_full
                continue
            w = wfull[:M_eff]; lam = wfull[M_eff]

            if not np.all(np.isfinite(w)) or np.abs(w).sum() > 50.0:
                est[i] = trend_q + float(rn.mean())
                var[i] = sill_full
                continue
            est[i] = trend_q + float(np.sum(w * rn))

            var[i] = max(0.0, sill_full - float(np.sum(w * c_vec)) - lam)
        return est, var


    def _predict_gpu(self, qx, qy, self_wid):
        n_q = len(qx)
        sw_arr = np.asarray(self_wid) if (self_wid is not None and np.ndim(self_wid) > 0) else None
        sw_scalar = self_wid if (self_wid is not None and np.ndim(self_wid) == 0) else None

        u_q = (qx - self.beta[3]) * self.u_hat[0] + (qy - self.beta[4]) * self.u_hat[1]
        v_q = (qx - self.beta[3]) * self.v_hat[0] + (qy - self.beta[4]) * self.v_hat[1]
        uq_r = u_q / self.a_dip; vq_r = v_q / self.a_strike
        K_extra = self.M + (40 if (self_wid is not None and self.well_id is not None) else 0)
        K_extra = min(K_extra, self.n_declustered)
        dists, idx_np = self.tree.query(np.c_[uq_r, vq_r], k=K_extra, workers=-1)
        idx_np = np.atleast_2d(idx_np)

        if self_wid is not None and self.well_id is not None:
            if sw_arr is not None:
                keep = (self.well_id[idx_np] != sw_arr[:, None])
            else:
                keep = (self.well_id[idx_np] != sw_scalar)

            order = np.argsort(~keep, axis=1, kind="stable")
            take = np.take_along_axis(idx_np, order[:, :self.M], axis=1).astype(np.int64)
            valid_count = np.minimum(keep.sum(axis=1), self.M).astype(np.int32)
        else:
            take = idx_np[:, :self.M].astype(np.int64)
            valid_count = np.full(n_q, self.M, dtype=np.int32)


        dtype = cp.float32
        ti = cp.asarray(take.astype(np.int64))
        cxn = self._cx_t[ti]
        cyn = self._cy_t[ti]
        rn = self._resid_t[ti]
        qxt = cp.asarray(qx.astype(np.float32))
        qyt = cp.asarray(qy.astype(np.float32))

        du_nn = cxn[:, :, None] - cxn[:, None, :]
        dv_nn = cyn[:, :, None] - cyn[:, None, :]
        uu = du_nn * self.u_hat[0] + dv_nn * self.u_hat[1]
        vv = du_nn * self.v_hat[0] + dv_nn * self.v_hat[1]
        h_nn = cp.sqrt((uu / self.a_dip) ** 2 + (vv / self.a_strike) ** 2)


        C_mat = self.sill * cp.exp(-3.0 * h_nn)

        du_qn = qxt[:, None] - cxn
        dv_qn = qyt[:, None] - cyn
        uuq = du_qn * self.u_hat[0] + dv_qn * self.u_hat[1]
        vvq = du_qn * self.v_hat[0] + dv_qn * self.v_hat[1]
        h_qn = cp.sqrt((uuq / self.a_dip) ** 2 + (vvq / self.a_strike) ** 2)
        c_vec = self.sill * cp.exp(-3.0 * h_qn)

        Mp = self.M + 1
        A = cp.zeros((n_q, Mp, Mp), dtype=dtype)
        A[:, :self.M, :self.M] = C_mat
        A[:, :self.M, self.M] = 1.0
        A[:, self.M, :self.M] = 1.0

        A[:, :self.M, :self.M] += float(self.nugget_eff) * cp.eye(self.M, dtype=dtype)
        b = cp.zeros((n_q, Mp, 1), dtype=dtype)
        b[:, :self.M, 0] = c_vec
        b[:, self.M, 0] = 1.0

        w_full = cp.linalg.solve(A, b)[:, :, 0]
        w = w_full[:, :self.M]
        lam = w_full[:, self.M]
        est_resid = (w * rn).sum(axis=-1)
        sill_full = self.sill + self.nugget_eff
        var = cp.maximum(0.0, sill_full - (w * c_vec).sum(axis=-1) - lam)

        dxq = qxt - self.beta[3]; dyq = qyt - self.beta[4]
        trend = self.beta[0] + self.beta[1] * dxq + self.beta[2] * dyq
        if self.quad is not None:
            trend = trend + self.quad[0] * dxq * dxq + self.quad[1] * dyq * dyq + self.quad[2] * dxq * dyq
        est = trend + est_resid


        deg = cp.asarray(valid_count < self.M)
        bad = (~cp.isfinite(est_resid)) | (cp.abs(w).sum(axis=-1) > 50.0) | deg
        if bad.any():
            mean_rn = rn.mean(axis=-1)
            est = cp.where(bad, trend + mean_rn, est)
            var = cp.where(bad, cp.full_like(var, sill_full), var)
        return cp.asnumpy(est).astype(np.float64), cp.asnumpy(var).astype(np.float64)

    def __repr__(self):
        return (f"KrigeSurface(n_declust={self.n_declustered}, M={self.M}, "
                f"nugget={self.nugget:.2f}, sill={self.sill:.2f}, "
                f"a_dip={self.a_dip:.0f}, a_strike={self.a_strike:.0f}, "
                f"dip_dir=({self.u_hat[0]:.3f},{self.u_hat[1]:.3f}), "
                f"journel={self.journel_cell}, device={self.device})")


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    N = 1500
    x = rng.uniform(0, 20000, N); y = rng.uniform(0, 20000, N)
    z_true = 10 + 0.02 * x + 0.03 * y + 5 * np.sin(x / 1000) + 5 * np.cos(y / 1000)
    z_obs = z_true + rng.normal(0, 1.0, N)
    wid = (np.arange(N) // 20)

    print("--- CPU path ---")
    ks_cpu = KrigeSurface(x, y, z_obs, well_id_per_pick=wid, M=40, device="cpu")
    print(ks_cpu)
    qx = rng.uniform(2000, 18000, 5000); qy = rng.uniform(2000, 18000, 5000)
    import time
    t0 = time.time()
    e_cpu, v_cpu = ks_cpu.predict(qx, qy, self_wid=5)
    t_cpu = time.time() - t0
    print(f"  5000 queries on CPU: {t_cpu:.2f}s  (est[0:3]={e_cpu[:3]})")

    if _HAVE_CUPY and _CUDA_OK:
        print("\n--- GPU path ---")
        ks_gpu = KrigeSurface(x, y, z_obs, well_id_per_pick=wid, M=40, device="cuda")
        print(ks_gpu)

        _ = ks_gpu.predict(qx[:10], qy[:10], self_wid=5)
        t0 = time.time()
        e_gpu, v_gpu = ks_gpu.predict(qx, qy, self_wid=5)
        t_gpu = time.time() - t0
        print(f"  5000 queries on GPU: {t_gpu:.2f}s  (est[0:3]={e_gpu[:3]})")
        print(f"  GPU speedup: {t_cpu/t_gpu:.1f}x")

        rel_diff = np.max(np.abs(e_gpu - e_cpu) / (np.abs(e_cpu) + 1e-6))
        print(f"  max rel diff (est): {rel_diff:.2e}")
        rel_diff_v = np.max(np.abs(v_gpu - v_cpu) / (np.abs(v_cpu) + 1e-6))
        print(f"  max rel diff (var): {rel_diff_v:.2e}")
    else:
        print("\nNo CUDA available — skipping GPU compare.")
