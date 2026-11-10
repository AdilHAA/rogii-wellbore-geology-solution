"""Single-window (вся скважина за один проход) — дизайн hengck cnn-sdf, БЕЗ авторегресса.

Изображение [3, T, H]: T=256 строк typewell (±64ft вокруг уровня PS — покрывает 98% дрейфа,
см. замер), H = вся eval-зона (компрессия S ft/столбец). Каналы: [t_gr, h_gr, history] (broadcast).
Таргет: TVT-diff SDF = (h_tvt − t_tvt)/40, clip ±3 (монотонен по T ⇒ ОДИН чистый нуль-кроссинг,
устойчивое восстановление — в отличие от argmin по EDT). Инференс: 1 forward, нуль-кроссинг по столбцу.
"""
import os
import numpy as np

import render

def _ig(n, d):
    return int(os.environ.get(n, d))

S_SW        = _ig("ROGII_SW_S", 32)
T_STEP      = float(os.environ.get("ROGII_SW_TSTEP", 0.5))
T_HALF_SW   = _ig("ROGII_SW_THALF", 128)
T_SW        = T_HALF_SW * 2
H_BEFORE_SW = _ig("ROGII_SW_HBEFORE", 16)
H_MAX_SW    = _ig("ROGII_SW_HMAX", 256)
USE_LOC     = _ig("ROGII_SW_LOC", 1)
USE_GEO     = _ig("ROGII_SW_GEO", 0)
USE_MISFIT  = _ig("ROGII_SW_MISFIT", 0)
USE_DHIST   = _ig("ROGII_SW_DENSEHIST", 0)
USE_DZ      = _ig("ROGII_SW_DZ", 0)
USE_GRMASK  = _ig("ROGII_SW_GRMASK", 0)
USE_PLANAR  = _ig("ROGII_SW_PLANAR", 0)

USE_EXT     = _ig("ROGII_SW_EXT", 0)

USE_QGR     = _ig("ROGII_SW_QGR", 0)

USE_TREND   = _ig("ROGII_SW_TREND", 0)

USE_GR2     = _ig("ROGII_SW_GR2", 0)
USE_DMISFIT = _ig("ROGII_SW_DMISFIT", 0)

USE_MISFIT2 = _ig("ROGII_SW_MISFIT2", 0)

USE_MM      = _ig("ROGII_SW_MM", 0)

USE_NCC     = _ig("ROGII_SW_NCC", 0)

USE_FORM    = _ig("ROGII_SW_FORM", 0)

USE_XVOL    = _ig("ROGII_SW_XVOL", 0)

USE_TWCAL   = _ig("ROGII_SW_TWCAL", 0)

USE_BEAM    = _ig("ROGII_SW_BEAM", 0)
USE_GEOPRIOR= _ig("ROGII_SW_GEOPRIOR", 0)
USE_PF      = _ig("ROGII_SW_PF", 0)

N_CH        = 3 + USE_LOC + 4 * USE_GEO + USE_DZ + USE_MISFIT + USE_DHIST + USE_GRMASK + USE_TREND + 2 * USE_GR2 + 2 * USE_EXT + 3 * (1 if USE_BEAM else 0) + USE_GEOPRIOR + 2 * USE_PF + USE_MISFIT2 + USE_DMISFIT + (1 if USE_NCC else 0) + 2 * USE_MM + USE_FORM
SDF_SCALE   = float(os.environ.get("ROGII_SW_SCALE", 40))
SDF_CLAMP   = 3.0

_ECDF = None
def _qgr(x):
    """GR -> глобальный перцентиль [0,100] (монотонно; artifacts/gr_ecdf.npy)."""
    global _ECDF
    if _ECDF is None:
        import data as _d
        _ECDF = np.load(f"{_d.ROOT}/artifacts/gr_ecdf.npy")
    return np.interp(x, _ECDF, np.linspace(0, 100, len(_ECDF))).astype(np.float32)

def _resample_typewell(wd):
    """Равномерная сетка typewell с шагом T_STEP (кэш в wd)."""
    if "t_grid" not in wd:
        tt, tg = wd["t_tvt"], wd["t_gr"]
        grid = np.arange(float(tt.min()), float(tt.max()), T_STEP, dtype=np.float32)
        tg_r = np.interp(grid, tt, tg).astype(np.float32)
        if USE_TWCAL:
            tg_r = _twcal(wd, grid, tg_r)
        wd["t_grid"] = grid
        wd["t_gr_r"] = tg_r
    return wd["t_grid"], wd["t_gr_r"]

def _twcal(wd, grid, tg_r):
    """idea #6 (STRIDE): аффинный (gain,offset) фит typewell GR -> GR этой скважины на ИЗВЕСТНОМ (pre-PS)
    участке (h_tvt_true==h_tvt_input там), применённый ко ВСЕМУ typewell-логу (включая скрытую часть).
    Fallback identity при слишком коротком известном участке (риск шумного фита на коротких латералях)."""
    ps = wd["ps_idx"]
    if ps < 20:
        return tg_r
    h_gr_known = wd["h_gr_smooth"][:ps + 1].astype(np.float64)
    h_tvt_known = wd["h_tvt_true"][:ps + 1].astype(np.float64)
    t_match = np.interp(h_tvt_known, grid, tg_r)
    A = np.stack([t_match, np.ones_like(t_match)], axis=1)
    (gain, offset), *_ = np.linalg.lstsq(A, h_gr_known, rcond=None)
    if not (np.isfinite(gain) and np.isfinite(offset)):
        return tg_r
    return (gain * tg_r + offset).astype(np.float32)

def build_chunk_sw(wd: dict, c_anchor: int, anchor_tvt: float, w_cols: int,
                   hist_cols: int = 8, hist_tvt=None):
    """Порт A (SegN авторегрессия): anchor-relative ЧАНК. T-сетка центрирована на anchor_tvt (не на PS),
    столбцы = hist_cols истории до якоря + w_cols предсказываемых после. Каналы = чемпион (LOC/GEO/MISFIT).
    hist_tvt[hist_cols] — известный/предсказанный TVT в истории (для линии якоря и loc). c_anchor — MD-индекс якоря.
    Возвращает image[N_CH,T,n], sdf, h_mask, t_mask, t_seg_tvt, n, i0, dz_ft(0), а также c0 (MD-инд. 1-го столбца)."""
    h_gr, h_tvt = wd["h_gr_smooth"], wd["h_tvt_true"]
    L = len(h_gr)
    c0 = c_anchor - hist_cols * S_SW
    if c0 < 0:
        hist_cols = c_anchor // S_SW; c0 = c_anchor - hist_cols * S_SW
    n = hist_cols + min(w_cols, (L - c_anchor) // S_SW)
    if n < hist_cols + 2:
        raise ValueError("chunk too short")
    seg = slice(c0, c0 + n * S_SW)
    h_seg_gr = h_gr[seg].reshape(n, S_SW).mean(1).astype(np.float32)
    if USE_QGR: h_seg_gr = _qgr(h_seg_gr)
    h_seg_tvt = h_tvt[seg].reshape(n, S_SW).mean(1).astype(np.float32)

    t_grid, t_gr_r = _resample_typewell(wd)
    t_a = int(np.abs(t_grid - anchor_tvt).argmin())
    j0, j1 = t_a - T_HALF_SW, t_a + T_HALF_SW
    pl, pr = max(0, -j0), max(0, j1 - len(t_grid))
    sl = slice(max(0, j0), min(len(t_grid), j1))
    t_seg_gr = np.pad(t_gr_r[sl], (pl, pr), "edge")
    if USE_QGR: t_seg_gr = _qgr(t_seg_gr)
    t_seg_tvt = np.pad(t_grid[sl], (pl, pr), "edge").astype(np.float32)
    t_mask = np.ones(T_SW, np.float32)
    if j0 < 0: t_mask[:-j0] = 0.0
    if j1 > len(t_grid): t_mask[T_SW - (j1 - len(t_grid)):] = 0.0

    tgt_seg = h_seg_tvt
    ch0 = np.broadcast_to(t_seg_gr[:, None], (T_SW, n))
    ch1 = np.broadcast_to(h_seg_gr[None, :], (T_SW, n))
    matched = np.abs(t_seg_tvt[:, None] - tgt_seg[None, :]).argmin(0)
    ch2 = np.zeros((T_SW, n), np.float32)
    nh = hist_cols

    hp = hist_tvt if hist_tvt is not None else h_seg_tvt[:nh]
    for c in range(min(nh, len(hp))):
        r = int(np.abs(t_seg_tvt - hp[c]).argmin()); ch2[r, c] = 1.0
    image = np.zeros((N_CH, T_SW, H_MAX_SW), np.float32)
    image[0, :, :n] = ch0; image[1, :, :n] = ch1; image[2, :, :n] = ch2
    c = 3
    if USE_LOC:
        loc = (t_seg_tvt - anchor_tvt) / (T_HALF_SW * T_STEP)
        image[c, :, :n] = np.broadcast_to(loc[:, None], (T_SW, n)); c += 1
    if USE_GEO:
        if "geo_sc" not in wd:
            md, x, y, z = wd["h_md"], wd["h_x"], wd["h_y"], wd["h_z"]
            dmd, dz, dx, dy = np.gradient(md), np.gradient(z), np.gradient(x), np.gradient(y)
            dip = np.arctan2(dmd, dz); az = np.arctan2(dx, dy)
            wd["geo_sc"] = np.stack([np.sin(dip), np.cos(dip), np.sin(az), np.cos(az)]).astype(np.float32)
        g = wd["geo_sc"][:, seg].reshape(4, n, S_SW).mean(2)
        for k in range(4):
            image[3 + USE_LOC + k, :, :n] = np.broadcast_to(g[k][None, :], (T_SW, n))
        c = 3 + USE_LOC + 4
    if USE_MISFIT:
        image[c, :, :n] = ch0 - ch1; c += 1
    if USE_MM:
        raw_mm = wd["h_gr_rawi"][seg].reshape(n, S_SW)
        image[c, :, :n] = np.broadcast_to(raw_mm.max(1).astype(np.float32)[None, :], (T_SW, n))
        image[c + 1, :, :n] = np.broadcast_to(raw_mm.min(1).astype(np.float32)[None, :], (T_SW, n))
        c += 2
    if USE_NCC:

        k = int(USE_NCC)
        z_col = wd["h_z"][seg].reshape(n, S_SW).mean(1).astype(np.float32)
        cols = np.arange(n)
        idx_r = np.arange(T_SW, dtype=np.int64)[:, None]
        m_ = 2 * k + 1
        sh = np.zeros(n, np.float64); sh2 = np.zeros(n, np.float64)
        st = np.zeros((T_SW, n), np.float64); st2 = np.zeros((T_SW, n), np.float64)
        sht = np.zeros((T_SW, n), np.float64)
        for j in range(-k, k + 1):
            cj = np.clip(cols + j, 0, n - 1)
            h_j = h_seg_gr[cj].astype(np.float64)
            drow = np.round((z_col[cj] - z_col) / T_STEP).astype(np.int64)
            rj = np.clip(idx_r - drow[None, :], 0, T_SW - 1)
            t_j = t_seg_gr[rj].astype(np.float64)
            sh += h_j; sh2 += h_j * h_j
            st += t_j; st2 += t_j * t_j
            sht += t_j * h_j[None, :]
        mu_h = sh / m_; var_h = np.maximum(sh2 / m_ - mu_h ** 2, 0.0)
        mu_t = st / m_; var_t = np.maximum(st2 / m_ - mu_t ** 2, 0.0)
        cov = sht / m_ - mu_t * mu_h[None, :]
        ncc = cov / (np.sqrt(var_t + 1.0) * np.sqrt(var_h + 1.0)[None, :])
        image[c, :, :n] = ncc.astype(np.float32)
        c += 1
    if USE_FORM:
        h_form_col = (wd["h_form_pred"][seg] - wd["h_z"][seg]).reshape(n, S_SW).mean(1).astype(np.float32)
        image[c, :, :n] = np.broadcast_to(h_form_col[None, :], (T_SW, n)); c += 1

    sdf = np.zeros((T_SW, H_MAX_SW), np.float32)
    sdf[:, :n] = np.clip((tgt_seg[None, :] - t_seg_tvt[:, None]) / SDF_SCALE, -SDF_CLAMP, SDF_CLAMP)
    h_mask = np.zeros(H_MAX_SW, np.float32); h_mask[nh:n] = 1.0

    hc_mask = np.zeros(H_MAX_SW, np.float32); hc_mask[:n] = 1.0
    struct_valid_flag = wd.get("h_struct_valid", False)
    if struct_valid_flag and np.isfinite(wd["h_struct"][seg]).all():
        s_seg = wd["h_struct"][seg].reshape(n, S_SW).mean(1).astype(np.float32)
        a_idx = min(nh, len(hp)) - 1
        ds = s_seg - s_seg[a_idx] + hp[a_idx]
        sdf_struct = np.zeros((T_SW, H_MAX_SW), np.float32)
        sdf_struct[:, :n] = np.clip((ds[None, :] - t_seg_tvt[:, None]) / SDF_SCALE, -SDF_CLAMP, SDF_CLAMP)
        struct_valid = 1.0
    else:
        sdf_struct = np.zeros((T_SW, H_MAX_SW), np.float32); struct_valid = 0.0

    dz_full = np.zeros(H_MAX_SW, np.float32)
    t_seg_tvt_full = np.zeros(T_SW, np.float32); t_seg_tvt_full[:] = t_seg_tvt
    xvol_extra = {}
    if USE_XVOL:
        h_native = np.zeros((H_MAX_SW, S_SW), np.float32)
        h_native[:n] = wd["h_gr_rawi"][seg].reshape(n, S_SW)
        xvol_extra = {"xvol_h_native": h_native, "xvol_t_native": t_seg_gr.astype(np.float32)}
    return {"image": image, "sdf": sdf, "h_mask": h_mask, "t_mask": t_mask,
            "t_seg_tvt": t_seg_tvt_full, "n": n, "nh": nh, "c0": c0, "dz_ft": dz_full,
            "sdf_struct": sdf_struct, "struct_valid": struct_valid, "hc_mask": hc_mask, **xvol_extra}

def build_sample_sw(wd: dict, offset: int = 0):
    """Один сэмпл на всю скважину. offset сдвигает PS-якорь на offset столбцов."""
    h_gr, h_tvt = wd["h_gr_smooth"], wd["h_tvt_true"]
    ps = wd["ps_idx"] + offset * S_SW
    i0 = ps - H_BEFORE_SW * S_SW
    if i0 < 0 or ps >= len(h_gr):
        raise ValueError("offset out of bounds")
    n = min(H_MAX_SW, (len(h_gr) - i0) // S_SW)
    if n < H_BEFORE_SW + 4:
        raise ValueError("too short")
    seg = slice(i0, i0 + n * S_SW)
    h_seg_gr = h_gr[seg].reshape(n, S_SW).mean(1).astype(np.float32)
    if USE_QGR:
        h_seg_gr = _qgr(h_seg_gr)
    h_seg_tvt = h_tvt[seg].reshape(n, S_SW).mean(1).astype(np.float32)

    t_grid, t_gr_r = _resample_typewell(wd)
    t_ps = int(np.abs(t_grid - h_tvt[ps]).argmin())
    j0, j1 = t_ps - T_HALF_SW, t_ps + T_HALF_SW
    pl, pr = max(0, -j0), max(0, j1 - len(t_grid))
    sl = slice(max(0, j0), min(len(t_grid), j1))
    t_seg_gr = np.pad(t_gr_r[sl], (pl, pr), "edge")
    if USE_QGR:
        t_seg_gr = _qgr(t_seg_gr)
    t_seg_tvt = np.pad(t_grid[sl], (pl, pr), "edge").astype(np.float32)
    t_mask = np.ones(T_SW, np.float32)
    if j0 < 0: t_mask[:-j0] = 0.0
    if j1 > len(t_grid): t_mask[T_SW - (j1 - len(t_grid)):] = 0.0

    dz_ft = np.zeros(n, np.float32)
    if USE_PLANAR:
        dz_ft = (wd["h_z"][seg].reshape(n, S_SW).mean(1) - wd["h_z"][ps]).astype(np.float32)
        tq = (t_seg_tvt[:, None] - dz_ft[None, :]).ravel()
        ch0 = np.interp(tq, t_grid, t_gr_r).astype(np.float32).reshape(T_SW, n)
    else:
        ch0 = np.broadcast_to(t_seg_gr[:, None], (T_SW, n))
    tgt_seg = (h_seg_tvt + dz_ft) if USE_PLANAR else h_seg_tvt
    ch1 = np.broadcast_to(h_seg_gr[None, :], (T_SW, n))
    matched = np.abs(t_seg_tvt[:, None] - tgt_seg[None, :]).argmin(0)
    ch2 = np.zeros((T_SW, n), np.float32)
    nh = min(H_BEFORE_SW, n)
    if USE_DHIST:

        ch2[:, :nh] = np.clip((tgt_seg[None, :nh] - t_seg_tvt[:, None]) / SDF_SCALE, -SDF_CLAMP, SDF_CLAMP)
    else:
        render.draw_path(ch2, matched, nh)
    if USE_BEAM:
        import beam as _beam
        bkey = f"_beams_{offset}"
        if bkey not in wd:
            t_grid_f, t_gr_f = _resample_typewell(wd)
            raw = wd.get("h_gr_rawi", h_gr)[i0 + nh * S_SW: i0 + n * S_SW]
            if USE_BEAM >= 2:
                packs = [_beam.beam_paths_all(raw, t_grid_f, t_gr_f, float(h_tvt[ps]), cfg)
                         for cfg in (_beam.BEAM_CONS, _beam.BEAM_SM5)]
                wd[bkey] = [(pp[:, :(n - nh) * S_SW].reshape(pp.shape[0], n - nh, S_SW).mean(2).astype(np.float32), ww)
                            for pp, ww in packs]
            else:
                paths = [_beam.beam_tvt(raw, t_grid_f, t_gr_f, float(h_tvt[ps]), cfg)
                         for cfg in (_beam.BEAM_CONS, _beam.BEAM_SM5)]
                wd[bkey] = [p_[:(n - nh) * S_SW].reshape(n - nh, S_SW).mean(1).astype(np.float32) for p_ in paths]
        beams_col = wd[bkey]
    image = np.zeros((N_CH, T_SW, H_MAX_SW), np.float32)
    image[0, :, :n] = ch0; image[1, :, :n] = ch1; image[2, :, :n] = ch2
    if USE_LOC:
        loc = (t_seg_tvt - h_tvt[ps]) / (T_HALF_SW * T_STEP)
        image[3, :, :n] = np.broadcast_to(loc[:, None], (T_SW, n))
    if USE_GEO:
        if "geo_sc" not in wd:
            md, x, y, z = wd["h_md"], wd["h_x"], wd["h_y"], wd["h_z"]
            dmd, dz, dx, dy = np.gradient(md), np.gradient(z), np.gradient(x), np.gradient(y)
            dip = np.arctan2(dmd, dz)
            az = np.arctan2(dx, dy)
            wd["geo_sc"] = np.stack([np.sin(dip), np.cos(dip), np.sin(az), np.cos(az)]).astype(np.float32)
        g = wd["geo_sc"][:, seg].reshape(4, n, S_SW).mean(2)
        base = 3 + USE_LOC
        for k in range(4):
            image[base + k, :, :n] = np.broadcast_to(g[k][None, :], (T_SW, n))
    c = 3 + USE_LOC + 4 * USE_GEO
    if USE_DZ:
        dz_col = ((wd["h_z"][seg].reshape(n, S_SW).mean(1) - wd["h_z"][ps]) / (T_HALF_SW * T_STEP)).astype(np.float32)
        image[c, :, :n] = np.broadcast_to(dz_col[None, :], (T_SW, n))
        c += 1
    if USE_MISFIT:
        image[c, :, :n] = ch0 - ch1
        c += 1
    if USE_DHIST:
        image[c, :, :nh] = 1.0
        c += 1
    if USE_GRMASK:
        gp = wd["h_gr_present"][seg].reshape(n, S_SW).mean(1).astype(np.float32)
        image[c, :, :n] = np.broadcast_to(gp[None, :], (T_SW, n))
        c += 1
    if USE_TREND:
        htvt_in = wd["h_tvt_true"]
        w0 = max(0, ps - 512)
        xs = np.arange(w0, ps + 1, dtype=np.float32)
        ys = htvt_in[w0:ps + 1]
        slope = float(np.polyfit(xs, ys, 1)[0]) if len(xs) > 32 else 0.0
        cols_ft = i0 + (np.arange(n) + 0.5) * S_SW
        trend = htvt_in[ps] + slope * (cols_ft - ps)
        tr_ch = np.clip((t_seg_tvt[:, None] - trend[None, :]) / (T_HALF_SW * T_STEP), -3, 3)
        image[c, :, :n] = tr_ch.astype(np.float32)
        c += 1
    if USE_EXT:
        global _ECDF
        if _ECDF is None:
            import data as _d
            _ECDF = np.load(f"{_d.ROOT}/artifacts/gr_ecdf.npy")
        p01, p10, p90, p99 = _ECDF[10], _ECDF[100], _ECDF[900], _ECDF[990]
        def _ext(x):
            hi = np.clip((x - p90) / max(p99 - p90, 1e-3), 0, 1.5)
            lo = np.clip((p10 - x) / max(p10 - p01, 1e-3), 0, 1.5)
            return (hi - lo).astype(np.float32)
        image[c, :, :n] = np.broadcast_to(_ext(t_seg_gr)[:, None], (T_SW, n))
        raw = wd["h_gr_rawi"][seg].reshape(n, S_SW)
        hi = np.clip((raw.max(1) - p90) / max(p99 - p90, 1e-3), 0, 1.5)
        lo = np.clip((p10 - raw.min(1)) / max(p10 - p01, 1e-3), 0, 1.5)
        image[c + 1, :, :n] = np.broadcast_to((hi - lo).astype(np.float32)[None, :], (T_SW, n))
        c += 2
    if USE_BEAM:
        rows_grid = np.arange(T_SW, dtype=np.float32)[:, None]
        brows = []
        for b_ in beams_col:
            if USE_BEAM >= 2:
                pp, ww = b_
                r_all = np.abs(t_seg_tvt[:, None, None] - pp.T[None, :, :]).argmin(0).astype(np.float32)
                band = np.zeros((T_SW, r_all.shape[0]), np.float32)
                for k_ in range(pp.shape[0]):
                    band += ww[k_] * np.exp(-0.5 * ((rows_grid - r_all[None, :, k_][0][None, :]) / 2.0) ** 2)
                band /= max(band.max(), 1e-6)
                r0 = (r_all * ww[None, :]).sum(1)
            else:
                r0 = np.abs(t_seg_tvt[:, None] - b_[None, :]).argmin(0).astype(np.float32)
                band = np.exp(-0.5 * ((rows_grid - r0[None, :]) / 2.0) ** 2).astype(np.float32)
            brows.append(r0)
            image[c, :, nh:n] = band
            image[c, :, :nh] = np.exp(-0.5 * ((rows_grid - matched[None, :nh].astype(np.float32)) / 2.0) ** 2)
            c += 1
        dis = np.abs(brows[0] - brows[1]).astype(np.float32) * T_STEP / SDF_SCALE
        image[c, :, nh:n] = np.broadcast_to(np.clip(dis, 0, SDF_CLAMP)[None, :], (T_SW, n - nh))
        c += 1
    if USE_PF:
        import pf_prior as _pfp
        pkey = f"_pf_{offset}"
        if pkey not in wd:
            end_ft = i0 + n * S_SW
            wd[pkey] = _pfp.pf_prior_tvt(wd, ps, end_ft, seed=abs(hash(wd["well_id"])) % 100000 + offset)
        mean_f, std_f = wd[pkey]

        n_post = n - nh
        mcol = mean_f[:n_post * S_SW].reshape(n_post, S_SW).mean(1) if n_post > 0 else np.zeros(0, np.float32)
        scol = std_f[:n_post * S_SW].reshape(n_post, S_SW).mean(1) if n_post > 0 else np.zeros(0, np.float32)
        rows_pf = np.arange(T_SW, dtype=np.float32)[:, None]
        r0 = np.abs(t_seg_tvt[:, None] - mcol[None, :]).argmin(0).astype(np.float32)
        sig_rows = np.clip(scol / T_STEP, 1.5, 25.0)
        image[c, :, nh:n] = np.exp(-0.5 * ((rows_pf - r0[None, :]) / sig_rows[None, :]) ** 2).astype(np.float32)
        image[c, :, :nh] = np.exp(-0.5 * ((rows_pf - matched[None, :nh].astype(np.float32)) / 2.0) ** 2)
        image[c + 1, :, nh:n] = np.broadcast_to(np.clip(scol / SDF_SCALE, 0, SDF_CLAMP)[None, :], (T_SW, n - nh))
        c += 2
    if USE_GEOPRIOR:
        z_col = wd["h_z"][seg].reshape(n, S_SW).mean(1).astype(np.float32)
        geo_col = h_seg_tvt + z_col
        k0 = max(0, nh - 24)
        if nh - k0 >= 4:
            xs = np.arange(k0, nh, dtype=np.float32)
            slope = float(np.polyfit(xs, geo_col[k0:nh], 1)[0])
        else:
            slope = 0.0
        prior_geo = geo_col[nh - 1] + slope * (np.arange(n, dtype=np.float32) - (nh - 1))
        prior_tvt = prior_geo - z_col
        prior_f = prior_tvt + (dz_ft if USE_PLANAR else 0.0)
        r0 = np.abs(t_seg_tvt[:, None] - prior_f[None, :]).argmin(0).astype(np.float32)
        rows_g = np.arange(T_SW, dtype=np.float32)[:, None]
        image[c, :, :n] = np.exp(-0.5 * ((rows_g - r0[None, :]) / 3.0) ** 2).astype(np.float32)
        c += 1
    if USE_GR2:
        if "t_gr_lp" not in wd:
            k = np.exp(-0.5 * (np.arange(-300, 301) / 100.0) ** 2); k = (k / k.sum()).astype(np.float32)
            wd["t_gr_lp"] = np.convolve(np.pad(t_gr_r, 300, "reflect"), k, "same")[300:-300].astype(np.float32)
            hk = np.exp(-0.5 * (np.arange(-600, 601) / 200.0) ** 2); hk = (hk / hk.sum()).astype(np.float32)
            wd["h_gr_lp"] = np.convolve(np.pad(h_gr, 600, "reflect"), hk, "same")[600:-600].astype(np.float32)
        tlp = np.pad(wd["t_gr_lp"][sl], (pl, pr), "edge")
        hlp = wd["h_gr_lp"][seg].reshape(n, S_SW).mean(1).astype(np.float32)
        image[c, :, :n] = np.broadcast_to(tlp[:, None], (T_SW, n))
        image[c + 1, :, :n] = np.broadcast_to(hlp[None, :], (T_SW, n))

    if USE_MISFIT2:
        if "t_gr_lp2" not in wd:
            k = np.exp(-0.5 * (np.arange(-300, 301) / 100.0) ** 2); k = (k / k.sum()).astype(np.float32)
            wd["t_gr_lp2"] = np.convolve(np.pad(t_gr_r, 300, "reflect"), k, "same")[300:-300].astype(np.float32)
        hk = np.exp(-0.5 * (np.arange(-600, 601) / 200.0) ** 2); hk = (hk / hk.sum()).astype(np.float32)
        h_lp = np.convolve(np.pad(h_gr, 600, "reflect"), hk, "same")[600:-600]
        tlp2 = np.pad(wd["t_gr_lp2"][sl], (pl, pr), "edge")
        hlp2 = h_lp[seg].reshape(n, S_SW).mean(1).astype(np.float32)
        image[c, :, :n] = (tlp2[:, None] - hlp2[None, :]).astype(np.float32)
        c += 1

    if USE_DMISFIT:
        dt = np.gradient(t_seg_gr).astype(np.float32)
        dh = np.gradient(h_seg_gr).astype(np.float32)
        image[c, :, :n] = (dt[:, None] - dh[None, :]).astype(np.float32)
        c += 1
    if USE_MM:
        raw_mm = wd["h_gr_rawi"][seg].reshape(n, S_SW)
        image[c, :, :n] = np.broadcast_to(raw_mm.max(1).astype(np.float32)[None, :], (T_SW, n))
        image[c + 1, :, :n] = np.broadcast_to(raw_mm.min(1).astype(np.float32)[None, :], (T_SW, n))
        c += 2
    if USE_NCC:

        k = int(USE_NCC)
        z_col = wd["h_z"][seg].reshape(n, S_SW).mean(1).astype(np.float32)
        cols = np.arange(n)
        idx_r = np.arange(T_SW, dtype=np.int64)[:, None]
        m_ = 2 * k + 1
        sh = np.zeros(n, np.float64); sh2 = np.zeros(n, np.float64)
        st = np.zeros((T_SW, n), np.float64); st2 = np.zeros((T_SW, n), np.float64)
        sht = np.zeros((T_SW, n), np.float64)
        for j in range(-k, k + 1):
            cj = np.clip(cols + j, 0, n - 1)
            h_j = h_seg_gr[cj].astype(np.float64)
            drow = np.round((z_col[cj] - z_col) / T_STEP).astype(np.int64)
            rj = np.clip(idx_r - drow[None, :], 0, T_SW - 1)
            t_j = t_seg_gr[rj].astype(np.float64)
            sh += h_j; sh2 += h_j * h_j
            st += t_j; st2 += t_j * t_j
            sht += t_j * h_j[None, :]
        mu_h = sh / m_; var_h = np.maximum(sh2 / m_ - mu_h ** 2, 0.0)
        mu_t = st / m_; var_t = np.maximum(st2 / m_ - mu_t ** 2, 0.0)
        cov = sht / m_ - mu_t * mu_h[None, :]
        ncc = cov / (np.sqrt(var_t + 1.0) * np.sqrt(var_h + 1.0)[None, :])
        image[c, :, :n] = ncc.astype(np.float32)
        c += 1

    sdf = np.zeros((T_SW, H_MAX_SW), np.float32)
    sdf[:, :n] = np.clip((tgt_seg[None, :] - t_seg_tvt[:, None]) / SDF_SCALE, -SDF_CLAMP, SDF_CLAMP)

    h_mask = np.zeros(H_MAX_SW, np.float32); h_mask[:n] = 1.0
    h_mask[:min(H_BEFORE_SW, n)] = 0.0

    dz_full = np.zeros(H_MAX_SW, np.float32); dz_full[:n] = dz_ft
    t_seg_tvt_full = np.zeros(T_SW, np.float32); t_seg_tvt_full[:] = t_seg_tvt
    return {"image": image, "sdf": sdf, "h_mask": h_mask, "t_mask": t_mask,
            "t_seg_tvt": t_seg_tvt_full, "n": n, "i0": i0, "ps": ps, "dz_ft": dz_full,
            "matched": matched.astype(np.int32)}

def recover_tvt_sw(sdf_pred: np.ndarray, t_seg_tvt: np.ndarray) -> np.ndarray:
    """Нуль-кроссинг TVT-diff SDF по каждому столбцу (+субпиксель). sdf убывает по T."""
    T, H = sdf_pred.shape
    out = np.empty(H, np.float32)
    idx = np.arange(T)
    for h in range(H):
        col = sdf_pred[:, h]
        t0 = int(np.abs(col).argmin())
        if 0 < t0 < T - 1:
            d = col[t0 + 1] - col[t0 - 1]
            frac = float(np.clip(-2 * col[t0] / d, -1, 1)) if abs(d) > 1e-6 else 0.0
        else:
            frac = 0.0
        out[h] = np.interp(t0 + frac, idx, t_seg_tvt)
    return out

DP_LAMBDA = float(os.environ.get("ROGII_DP_LAMBDA", "0"))
SOFT_DECODE = int(os.environ.get("ROGII_SOFT", "0"))
SOFT_TAU = float(os.environ.get("ROGII_SOFT_TAU", "0.7"))
SOFT_LAM = float(os.environ.get("ROGII_SOFT_LAM", "1.1"))
SOFT_K = int(os.environ.get("ROGII_SOFT_K", "4"))
DP_DRIFT_W = float(os.environ.get("ROGII_DP_DRIFT_W", "0"))

def recover_tvt_dp(sdf_pred: np.ndarray, t_seg_tvt: np.ndarray, lam: float = 0.35, K: int = 3,
                   d0=None) -> np.ndarray:
    """Глобально-когерентный путь по |SDF|-полю (DP, переход +-K строк, цена lam*|d|).
    Лечит дребезг веток в конце подъёмов (visual-iterate 2026-07-04): fold-0 10.31->8.71."""
    T, n = sdf_pred.shape
    a = np.abs(sdf_pred)
    cost = a[:, 0].astype(np.float64).copy()
    prev = np.zeros((n, T), np.int16)
    for c in range(1, n):
        newc = np.full(T, 1e18)
        base = 0.0 if d0 is None else float(d0[c])
        for d in range(-K, K + 1):
            sh = np.roll(cost, d) + lam * abs(d - base)
            if d > 0: sh[:d] = 1e18
            if d < 0: sh[d:] = 1e18
            b = sh < newc
            newc = np.where(b, sh, newc)
            prev[c][b] = d
        cost = newc + a[:, c]
    r = int(cost.argmin()); path = np.zeros(n, np.int32)
    for c in range(n - 1, -1, -1):
        path[c] = r; r -= int(prev[c][r])
    return t_seg_tvt[path].astype(np.float32)

def _band_lse(vec, K, lam):
    """logsumexp_{d=-K..K} (vec[t-d] - lam*|d|) по строкам (банд-переход),边ы = -inf."""
    T = len(vec)
    acc = np.full(T, -np.inf)
    for d in range(-K, K + 1):
        sh = np.roll(vec, d) - lam * abs(d)
        if d > 0: sh[:d] = -np.inf
        if d < 0: sh[d:] = -np.inf
        acc = np.logaddexp(acc, sh)
    return acc

def _isotonic_decr(y):
    """PAVA-проекция на невозрастающую последовательность (SDF монотонен по T по построению)."""
    y = y.astype(np.float64).copy(); n = len(y)

    v = -y; w = np.ones(n); lvl = v.copy()
    i = 0; idx = list(range(n + 1)); vals = list(v); wts = [1.0] * n

    stack_v = []; stack_w = []; stack_n = []
    for j in range(n):
        cv = v[j]; cw = 1.0; cn = 1
        while stack_v and stack_v[-1] > cv:
            pv = stack_v.pop(); pw = stack_w.pop(); pn = stack_n.pop()
            cv = (cv * cw + pv * pw) / (cw + pw); cw += pw; cn += pn
        stack_v.append(cv); stack_w.append(cw); stack_n.append(cn)
    out = np.empty(n); p = 0
    for cv, cn in zip(stack_v, stack_n):
        out[p:p + cn] = cv; p += cn
    return -out

def recover_tvt_soft_mtp(sdf_k: np.ndarray, logit: np.ndarray, t_seg_tvt: np.ndarray,
                         tau: float = 0.2, lam: float = 1.1, K: int = 4, rho: float = 4.0):
    """exp_121 MTP#2: FB по joint-состояниям (мода m, строка t). Эмиссия phi[m,t,c]=log w_m(c)−|sdf_m|/tau;
    переход по t — band (lam), смена моды −rho (мода меняется только на разломах). Возврат: posterior-mean
    по t (маргинал по m). sdf_k [M,T,n], logit [M,n]."""
    M, T, n = sdf_k.shape
    lw = logit.astype(np.float64) - logit.max(0, keepdims=True)
    lw = lw - np.log(np.exp(lw).sum(0, keepdims=True))
    phi = lw[:, None, :] - np.abs(sdf_k.astype(np.float64)) / max(tau, 1e-6)

    def _blse_bwd(v):
        acc = np.full(T, -np.inf)
        for d in range(-K, K + 1):
            sh = np.roll(v, -d) - lam * abs(d)
            if d > 0: sh[T - d:] = -np.inf
            elif d < 0: sh[:-d] = -np.inf
            acc = np.logaddexp(acc, sh)
        return acc

    def step(prev, bwd=False):
        bl = np.stack([(_blse_bwd(prev[m]) if bwd else _band_lse(prev[m], K, lam)) for m in range(M)])
        out = np.empty_like(bl)
        for m in range(M):
            if M > 1:
                oth = np.logaddexp.reduce(np.delete(bl, m, axis=0), axis=0) - rho
            else:
                oth = np.full(T, -np.inf)
            out[m] = np.logaddexp(bl[m], oth)
        return out

    la = np.empty((n, M, T)); la[0] = phi[:, :, 0]
    for c in range(1, n):
        la[c] = phi[:, :, c] + step(la[c - 1])
    lb = np.empty((n, M, T)); lb[n - 1] = 0.0
    for c in range(n - 2, -1, -1):
        lb[c] = step(phi[:, :, c + 1] + lb[c + 1], bwd=True)
    logg = la + lb
    logg -= logg.max(axis=(1, 2), keepdims=True)
    w = np.exp(logg); w /= w.sum(axis=(1, 2), keepdims=True)
    rows = np.arange(T)
    mean_row = (w.sum(1) * rows[None, :]).sum(1)
    return np.interp(mean_row, rows, t_seg_tvt).astype(np.float32)

def recover_tvt_soft_logits(logits: np.ndarray, t_seg_tvt: np.ndarray, tau: float = 1.0,
                            lam: float = 0.25, K: int = 3) -> np.ndarray:
    """SegN-v2: postерior-mean декод для CE-обученных ЛОГИТОВ [T, n] (калиброванный постериор).
    phi = log_softmax(logits/tau) по строкам; дальше тот же forward-backward, что recover_tvt_soft."""
    T, n = logits.shape
    z = logits.astype(np.float64) / max(tau, 1e-6)
    z -= z.max(0, keepdims=True)
    phi = (z - np.log(np.exp(z).sum(0, keepdims=True) + 1e-300)).T
    phi = phi.T
    la = np.empty((n, T)); la[0] = phi[:, 0]
    for c in range(1, n):
        la[c] = phi[:, c] + _band_lse(la[c - 1], K, lam)
    lb = np.empty((n, T)); lb[n - 1] = 0.0
    for c in range(n - 2, -1, -1):
        g = phi[:, c + 1] + lb[c + 1]
        acc = np.full(T, -np.inf)
        for d in range(-K, K + 1):
            sh = np.roll(g, -d) - lam * abs(d)
            if d > 0: sh[T - d:] = -np.inf
            if d < 0: sh[:-d] = -np.inf
            acc = np.logaddexp(acc, sh)
        lb[c] = acc
    logg = la + lb
    logg -= logg.max(1, keepdims=True)
    w = np.exp(logg); w /= w.sum(1, keepdims=True)
    rows = np.arange(T)
    mean_row = (w * rows[None, :]).sum(1)
    return np.interp(mean_row, rows, t_seg_tvt).astype(np.float32)

def recover_tvt_soft(sdf_pred: np.ndarray, t_seg_tvt: np.ndarray, tau: float = 1.0,
                     lam: float = 0.25, K: int = 3, mono: bool = False, emit_extra=None) -> np.ndarray:
    """Постериорное СРЕДНЕЕ пути (forward-backward marginals) вместо жёсткого min-пути.
    Метрика = pooled RMSE => байес-оптимум = per-column среднее по постериору путей, НЕ MAP.
    Эмиссия: log-потенциал phi[t,c] = -|sdf[t,c]|/tau; переход: -lam*|d|, банд |d|<=K.
    tau->0 концентрирует постериор к MAP (DP — предельная точка грида)."""
    T, n = sdf_pred.shape
    sd = sdf_pred.astype(np.float64)
    if mono:
        sd = np.stack([_isotonic_decr(sd[:, c]) for c in range(n)], axis=1)
    phi = -np.abs(sd) / max(tau, 1e-6)
    if emit_extra is not None:
        phi = phi + emit_extra[:, :n].astype(np.float64)
    la = np.empty((n, T)); la[0] = phi[:, 0]
    for c in range(1, n):
        la[c] = phi[:, c] + _band_lse(la[c - 1], K, lam)
    lb = np.empty((n, T)); lb[n - 1] = 0.0
    for c in range(n - 2, -1, -1):
        g = phi[:, c + 1] + lb[c + 1]

        acc = np.full(T, -np.inf)
        for d in range(-K, K + 1):
            sh = np.roll(g, -d) - lam * abs(d)
            if d > 0: sh[T - d:] = -np.inf
            if d < 0: sh[:-d] = -np.inf
            acc = np.logaddexp(acc, sh)
        lb[c] = acc
    logg = la + lb
    logg -= logg.max(1, keepdims=True)
    w = np.exp(logg); w /= w.sum(1, keepdims=True)
    rows = np.arange(T)
    mean_row = (w * rows[None, :]).sum(1)
    return np.interp(mean_row, rows, t_seg_tvt).astype(np.float32)

def recover_tvt_mom(sdf_pred, t_seg_tvt, tau=0.7, Kv=3, acc=0.6, slope=0.05):
    """#4 momentum-HMM ПОВЕРХ soft: состояние (строка r, скорость v in -Kv..Kv), r' = r + v'.
    Штраф ускорения acc*|v'-v| (инерция наклона) + слабый slope*|v'| (регуляр к 0). Постериор-среднее
    по r (marginal). Байес-оптимум под RMSE + 2-й порядок вместо плоского L1 (attacks ветвевую инерцию)."""
    T, n = sdf_pred.shape
    V = 2 * Kv + 1; vs = np.arange(-Kv, Kv + 1)
    phi = -np.abs(sdf_pred.astype(np.float64)) / max(tau, 1e-6)
    NEG = -1e18

    la = np.full((n, T, V), NEG)
    la[0] = phi[:, 0, None] - slope * np.abs(vs)[None, :]
    for c in range(1, n):
        prev = la[c - 1]
        newc = np.full((T, V), NEG)
        for vj, vp in enumerate(vs):

            m = np.full(T, NEG)
            for vi, vpp in enumerate(vs):
                m = np.logaddexp(m, prev[:, vi] - acc * abs(vp - vpp))
            sh = np.roll(m, vp)
            if vp > 0: sh[:vp] = NEG
            if vp < 0: sh[vp:] = NEG
            newc[:, vj] = sh - slope * abs(vp)
        la[c] = newc + phi[:, c, None]

    lb = np.full((n, T, V), NEG); lb[n - 1] = 0.0
    for c in range(n - 2, -1, -1):
        nxt = phi[:, c + 1, None] + lb[c + 1]
        cur = np.full((T, V), NEG)
        for vj, vp in enumerate(vs):
            g = nxt[:, vj]
            back = np.roll(g, -vp)
            if vp > 0: back[T - vp:] = NEG
            if vp < 0: back[:-vp] = NEG
            for vi, vpp in enumerate(vs):
                cur[:, vi] = np.logaddexp(cur[:, vi], back - acc * abs(vp - vpp) - slope * abs(vp))
        lb[c] = cur
    logg = la + lb
    lr = np.logaddexp.reduce(logg, axis=2)
    lr -= lr.max(1, keepdims=True)
    w = np.exp(lr); w /= w.sum(1, keepdims=True)
    rows = np.arange(T)
    mean_row = (w * rows[None, :]).sum(1)
    return np.interp(mean_row, rows, t_seg_tvt).astype(np.float32)
