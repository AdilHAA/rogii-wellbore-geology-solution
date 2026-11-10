"""Вход модели (РЕНДЕР — главный рычаг) + SDF-таргет.

Порт валидированного пайплайна medali1992 (CV row-RMSE 7.503, fold-0 7.24) на
геометрию hengck23 cnn-mtp: изображение T_TOTAL×H_TOTAL = 64×24 (S=32 компрессия).
Каналы: [heatmap = t_gr − h_gr, history]. Таргет: знаковый EDT-SDF.
cv2 заменён numpy-растеризатором (cv2 в окружении нет). См. research/hengck/BUILD_SPEC.md.
"""
import os

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from scipy.ndimage import distance_transform_edt

import data

def _ig(name, default):
    return int(os.environ.get(name, default))

S            = _ig("ROGII_S", 32)
H_BEFORE_PS  = _ig("ROGII_HBEFORE", 8)
H_AFTER_PS   = _ig("ROGII_HAFTER", 16)
H_TOTAL      = H_BEFORE_PS + H_AFTER_PS
T_HALF       = _ig("ROGII_THALF", 32)
T_TOTAL      = T_HALF * 2
SAVGOL_WIN   = _ig("ROGII_SAVGOL", 101)
SAVGOL_POLY  = 2
USE_FORM     = _ig("ROGII_SW_FORM", 0)

TRAIN_OFFSETS = list(range(-4, 5, 2))
VALID_OFFSETS = [0]

def draw_path(img: np.ndarray, rows, n_cols: int, val: float = 1.0) -> None:
    """Связный staircase-растер пути rows[0..n_cols-1] в img[row, col] (замена cv2.line)."""
    T = img.shape[0]
    r = np.clip(np.asarray(rows[:n_cols]).astype(int), 0, T - 1)
    for c in range(n_cols - 1):
        lo, hi = (r[c], r[c + 1]) if r[c] <= r[c + 1] else (r[c + 1], r[c])
        img[lo:hi + 1, c] = val
    img[r[n_cols - 1], n_cols - 1] = val

def make_history(matched, n_cols: int = H_BEFORE_PS, noise_rows: float = 0.0, rng=None) -> np.ndarray:
    """История (известный путь) cols 0..n_cols-1. noise_rows>0 -> random-walk шум
    (анти-exposure-bias: на инференсе история = свои дрейфующие предсказания)."""
    rows = np.asarray(matched[:n_cols], dtype=np.float32).copy()
    if noise_rows > 0:
        r = rng or np.random
        sigma = r.uniform(0.0, noise_rows)
        walk = np.cumsum(r.normal(0.0, sigma, n_cols)).astype(np.float32)
        rows = rows + walk
    img = np.zeros((T_TOTAL, H_TOTAL), np.float32)
    draw_path(img, rows, n_cols)
    return img

def load_well_data(wid: str, split: str = "train") -> dict:
    """Загрузка+препроцесс одной скважины (кэшируется в датасете)."""
    base = f"{data.RAW}/{split}/{wid}"
    h = pd.read_csv(f"{base}__horizontal_well.csv")
    t = pd.read_csv(f"{base}__typewell.csv")

    h_gr_raw = h["GR"].interpolate(method="linear").bfill().ffill().values
    h_gr_smooth = savgol_filter(h_gr_raw, SAVGOL_WIN, SAVGOL_POLY).astype(np.float32)
    h_gr_present = (~h["GR"].isna()).values.astype(np.float32)
    h_gr_rawi = h_gr_raw.astype(np.float32)

    h_tvt_input = h["TVT_input"].values.astype(np.float32)
    h_tvt_true = (h["TVT"].values.astype(np.float32)
                  if "TVT" in h.columns else h_tvt_input.copy())

    known = ~np.isnan(h_tvt_input)
    ps_idx = int(np.flatnonzero(known)[-1]) if known.any() else 0

    def _col(name):
        return h[name].values.astype(np.float32) if name in h.columns else np.zeros(len(h), np.float32)

    h_struct = _col(data.TOPS[0])
    h_struct_valid = bool(data.TOPS[0] in h.columns and np.isfinite(h[data.TOPS[0]].values).all())

    if USE_FORM:
        import form_ancc
        h_form_pred = form_ancc.get_form_pred(wid, split)
    else:
        h_form_pred = np.zeros(len(h), np.float32)

    return {
        "well_id": wid,
        "h_gr_smooth": h_gr_smooth,
        "h_tvt_true": h_tvt_true,
        "h_tvt_input": h_tvt_input,
        "ps_idx": ps_idx,
        "h_md": _col("MD"), "h_x": _col("X"), "h_y": _col("Y"), "h_z": _col("Z"),
        "h_gr_present": h_gr_present,
        "h_gr_rawi": h_gr_rawi,
        "h_struct": h_struct, "h_struct_valid": h_struct_valid,
        "h_form_pred": h_form_pred,
        "t_gr": t["GR"].values.astype(np.float32),
        "t_tvt": t["TVT"].values.astype(np.float32),
    }

def _typewell_crop(t_gr, t_tvt, t_ps):
    """Окно typewell [t_ps−T_HALF, t_ps+T_HALF] с edge-паддингом + t_mask."""
    j0, j1 = t_ps - T_HALF, t_ps + T_HALF
    if j0 < 0 or j1 > len(t_gr):
        pl, pr = max(0, -j0), max(0, j1 - len(t_gr))
        sl = slice(max(0, j0), min(len(t_gr), j1))
        t_seg_gr = np.pad(t_gr[sl], (pl, pr), mode="edge")
        t_seg_tvt = np.pad(t_tvt[sl], (pl, pr), mode="edge")
    else:
        t_seg_gr, t_seg_tvt = t_gr[j0:j1], t_tvt[j0:j1]
    t_mask = np.ones(T_TOTAL, np.float32)
    if j0 < 0:
        t_mask[:-j0] = 0.0
    if j1 > len(t_gr):
        t_mask[T_TOTAL - (j1 - len(t_gr)):] = 0.0
    return t_seg_gr.astype(np.float32), t_seg_tvt.astype(np.float32), t_mask

def build_sample(wd: dict, offset: int = 0, step: int = S) -> dict:
    """Обучающий сэмпл (heatmap, history, sdf, masks, ...) с PS-сдвигом offset столбцов."""
    h_gr, h_tvt = wd["h_gr_smooth"], wd["h_tvt_true"]
    t_gr, t_tvt = wd["t_gr"], wd["t_tvt"]
    ps_idx = wd["ps_idx"]

    ps_shifted = ps_idx + offset * step
    mb, ma = H_BEFORE_PS * step, H_AFTER_PS * step
    if ps_shifted < mb or ps_shifted + ma > len(h_gr):
        raise ValueError(f"offset={offset} вне границ (ps={ps_shifted}, len={len(h_gr)})")

    i0, i1 = ps_shifted - mb, ps_shifted + ma
    h_seg_gr = h_gr[i0:i1].reshape(H_TOTAL, step).mean(1).astype(np.float32)
    h_seg_tvt = h_tvt[i0:i1].reshape(H_TOTAL, step).mean(1).astype(np.float32)

    t_ps = int(np.abs(t_tvt - h_tvt[ps_shifted]).argmin())
    t_seg_gr, t_seg_tvt, t_mask = _typewell_crop(t_gr, t_tvt, t_ps)

    heatmap = (t_seg_gr[:, None] - h_seg_gr[None, :]).astype(np.float32)
    matched = np.abs(t_seg_tvt[:, None] - h_seg_tvt[None, :]).argmin(0).astype(np.int32)

    history = np.zeros((T_TOTAL, H_TOTAL), np.float32)
    draw_path(history, matched, H_BEFORE_PS)

    sdf = build_sdf(matched)

    h_mask = np.ones(H_TOTAL, np.float32)
    if i0 < 0:
        h_mask[:int(np.ceil(-i0 / step))] = 0.0
    if i1 > len(h_gr):
        h_mask[H_TOTAL - int(np.ceil((i1 - len(h_gr)) / step)):] = 0.0

    return {
        "heatmap": heatmap, "history": history, "sdf": sdf,
        "matched": matched.astype(np.float32), "target": h_seg_tvt,
        "h_mask": h_mask, "t_mask": t_mask, "t_seg_tvt": t_seg_tvt,
    }

def build_sdf(matched: np.ndarray) -> np.ndarray:
    """Знаковый EDT-SDF (medali): расстояние до границы × знак(row − matched)."""
    boundary = np.zeros((T_TOTAL, H_TOTAL), np.uint8)
    draw_path(boundary, matched, H_TOTAL, val=1)
    unsigned = distance_transform_edt(1 - boundary).astype(np.float32)
    sign = np.sign(np.arange(T_TOTAL)[:, None] - matched[None, :].astype(np.float32))
    return (sign * unsigned).astype(np.float32)

def build_inference_window(wd: dict, ps_shifted: int, pred_tvt: np.ndarray, step: int = S) -> dict:
    """Окно для инференса: history строится из УЖЕ предсказанных TVT (sliding-window)."""
    h_gr, t_gr, t_tvt = wd["h_gr_smooth"], wd["t_gr"], wd["t_tvt"]
    mb, ma = H_BEFORE_PS * step, H_AFTER_PS * step
    if ps_shifted < mb or ps_shifted + ma > len(h_gr):
        raise ValueError(f"ps_shifted={ps_shifted} вне границ")

    i0, i1 = ps_shifted - mb, ps_shifted + ma
    h_seg_gr = h_gr[i0:i1].reshape(H_TOTAL, step).mean(1).astype(np.float32)

    ps_tvt = pred_tvt[ps_shifted]
    if np.isnan(ps_tvt):
        ps_tvt = wd["h_tvt_true"][ps_shifted]
    t_ps = int(np.abs(t_tvt - ps_tvt).argmin())
    t_seg_gr, t_seg_tvt, t_mask = _typewell_crop(t_gr, t_tvt, t_ps)

    heatmap = (t_seg_gr[:, None] - h_seg_gr[None, :]).astype(np.float32)

    history = np.zeros((T_TOTAL, H_TOTAL), np.float32)
    hist_rows = np.full(H_BEFORE_PS, T_HALF, np.float32)
    for col in range(H_BEFORE_PS):
        ft = i0 + col * step + step // 2
        v = pred_tvt[ft] if ft < len(pred_tvt) else np.nan
        if not np.isnan(v):
            hist_rows[col] = int(np.abs(t_seg_tvt - v).argmin())
    draw_path(history, hist_rows, H_BEFORE_PS)

    h_mask = np.ones(H_TOTAL, np.float32)
    if i0 < 0:
        h_mask[:int(np.ceil(-i0 / step))] = 0.0
    if i1 > len(h_gr):
        h_mask[H_TOTAL - int(np.ceil((i1 - len(h_gr)) / step)):] = 0.0

    return {"heatmap": heatmap, "history": history, "h_mask": h_mask,
            "t_mask": t_mask, "t_seg_tvt": t_seg_tvt}
