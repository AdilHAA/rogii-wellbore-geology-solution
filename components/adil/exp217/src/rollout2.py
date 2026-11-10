"""SegN-v2 rollout: авторегрессия с ПЕРЕКРЫТИЕМ чанков + треугольным усреднением + ре-якорь
из усреднённой кривой на позиции rebase_frac*W (не на последнем столбце). Инференс-only —
A/B на готовых chunk-чекпойнтах (exp_104/105) против жёсткого ре-якоря train_chunk.autoreg_predict.
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import render_sw
from render_sw import S_SW

def _theil_sen_reanchor(col_tvt, rb_col, n_cols):
    """Робастный ре-якорь: Theil-Sen прямая (медиана попарных наклонов — устойчива к выбросам,
    в отличие от OLS) по последним n_cols УЖЕ раскодированным столбцам col_tvt ПЕРЕД rb_col
    (сам rb_col в фит не входит — он и есть шумная точка, которую заменяем), экстраполяция
    прямой на позицию rb_col. Денойзит ре-якорь локальным трендом вместо одного шумного readout
    (текущий путь: скользящее среднее pred_sum/pred_w, которое лагает на изломах). Возвращает
    None при <2 валидных точках — вызывающий код должен упасть на старый путь."""
    lo = max(0, rb_col - n_cols)
    xs = np.arange(lo, rb_col, dtype=np.float64)
    if xs.size < 2:
        return None
    ys = col_tvt[lo:rb_col].astype(np.float64)
    dx = xs[:, None] - xs[None, :]; dy = ys[:, None] - ys[None, :]
    iu = np.triu_indices(xs.size, k=1)
    d = dx[iu]
    valid = np.abs(d) > 1e-9
    if not valid.any():
        return float(np.median(ys))
    slope = float(np.median(dy[iu][valid] / d[valid]))
    intercept = float(np.median(ys - slope * xs))
    return intercept + slope * rb_col

@torch.no_grad()
def rollout_well(wd, model, device, W, HIST, stride_frac=0.5, rebase_frac=0.6, tri=True,
                 tau=0.7, lam=1.1, K=4, logits=False, gain=0.0, gain_q=0.5, anchor_ref=None, aw=0.5, record=False,
                 dz_alpha=0.0, dz_beta=0.5, dz_sig0=0.0, conf_gate=0.0, reanchor_linefit_n=0, branch_temp=0.0,
                 branch_rng=None, start_offset=0, force_branch=None, record_conf=False, edit_gain=0.0):
    """dz_alpha>0 (P2 стадия B): mean-reverting ре-якорь atvt -= dz_alpha*EMA(δ̂)*w(σ̂),
    δ̂/log σ̂ из dz-головы ТОГО ЖЕ форварда (модель с dz_w>0); w(σ̂)=min(1, dz_sig0/σ̂)
    при dz_sig0>0, иначе 1; EMA по шагам с β=dz_beta. Дефолт 0 — код-путь не меняется.
    conf_gate>0 (exp_159 CONF-CALIB): доверие-взвешенный откат нового якоря к консервативному
    fallback (среднее последних 2 принятых history-точек) при высокой энтропии K-селектора в
    колонке ре-якоря — trust=1-clip(H/logK,0,1), atvt=(1-conf_gate*(1-trust))*atvt+conf_gate*(1-trust)*fallback.
    Дефолт 0 — код-путь не меняется.
    reanchor_linefit_n>0 (line-fit ре-якорь, patent-motivated: GT TVT почти всюду кусочно-линейна
    с редкими изломами): вместо скользящего среднего pred_sum/pred_w на новом
    якоре берём Theil-Sen прямую по последним reanchor_linefit_n столбцам ЭТОГО чанка (col_tvt,
    ещё до треугольного усреднения между чанками) и экстраполируем её на rb_col — целиком
    заменяет и primary (mean) путь, и fallback (single-column) путь ниже. 0 (дефолт) — код-путь
    не меняется, exp_158 воспроизводится байт-в-байт."""
    y = wd["h_tvt_true"]; L = len(wd["h_gr_smooth"]); ps = wd["ps_idx"]
    pred_sum = np.zeros(L); pred_w = np.zeros(L)
    ent_sum = np.zeros(L); spread_sum = np.zeros(L)
    pred = np.full(L, np.nan, np.float32); pred[:ps + 1] = y[:ps + 1]
    stride = max(1, int(W * stride_frac))
    dz_ema = None

    c_anchor = ps - int(start_offset) * S_SW
    c_anchor = max(HIST * S_SW, min(c_anchor, ps))
    atvt = float(y[c_anchor])
    hist_tvt = np.array([y[max(0, c_anchor - (HIST - k) * S_SW):max(1, c_anchor - (HIST - k - 1) * S_SW)].mean()
                         for k in range(HIST)], np.float32)
    guard = 0; states = []
    while c_anchor + S_SW < L and guard < 800:
        guard += 1
        if record and c_anchor > ps:

            states.append((int(c_anchor), float(atvt), hist_tvt.astype(np.float32).copy(), len(states)))
        try:
            s = render_sw.build_chunk_sw(wd, c_anchor, atvt, W, HIST, hist_tvt=hist_tvt)
        except ValueError:
            break
        img = torch.from_numpy(s["image"])[None].to(device)
        binp = {"image": img}
        if dz_alpha > 0:
            binp["h_mask"] = torch.from_numpy(s["h_mask"])[None].to(device)
            binp["t_mask"] = torch.from_numpy(s["t_mask"])[None].to(device)
        _ms = model if isinstance(model, (list, tuple)) else [model]
        if any(getattr(m_, "param_heads", False) for m_ in _ms):
            binp["h_mask"] = torch.from_numpy(s["h_mask"])[None].to(device)
            binp["t_mask"] = torch.from_numpy(s["t_mask"])[None].to(device)
            binp["t_seg_tvt"] = torch.from_numpy(s["t_seg_tvt"])[None].to(device)
        if any(getattr(m_, "edit_head", False) for m_ in _ms):
            binp["t_mask"] = torch.from_numpy(s["t_mask"])[None].to(device)
        if getattr(model, "xvol", False) or (isinstance(model, (list, tuple)) and getattr(model[0], "xvol", False)):
            binp["xvol_h"] = torch.from_numpy(s["xvol_h_native"])[None].to(device)
            binp["xvol_t"] = torch.from_numpy(s["xvol_t_native"])[None].to(device)
            binp["xvol_n"] = torch.tensor([s["n"]], dtype=torch.long, device=device)
        models = model if isinstance(model, (list, tuple)) else [model]
        acc = None; dz_cur = None; dz_sig = None; logit_acc = None; sk_acc = None; ed_acc = None
        for m_ in models:
            out_ = m_(binp)
            if edit_gain != 0.0 and "edit_delta" in out_:
                ed = out_["edit_delta"][0].cpu().numpy()
                ed_acc = ed if ed_acc is None else ed_acc + ed
            if dz_alpha > 0 and "dz" in out_:
                dz_cur = float(out_["dz"][0, 0].cpu()); dz_sig = float(np.exp(float(out_["dz"][0, 1].cpu())))
            if "logit" in out_:
                sk = out_["sdf"][0].cpu().numpy()
                lgt = out_["logit"][0].cpu().numpy()
                if force_branch is not None:
                    o = sk[int(force_branch), :, :]
                elif branch_rng is not None and branch_temp > 0:
                    z = lgt / branch_temp; z = z - z.max(0, keepdims=True)
                    pbr = np.exp(z); pbr /= pbr.sum(0, keepdims=True)
                    cdf = np.cumsum(pbr, 0)
                    u = branch_rng.random(sk.shape[2])
                    br = (u[None, :] > cdf).sum(0).clip(0, sk.shape[0] - 1)
                    o = sk[br, :, np.arange(sk.shape[2])].T
                elif branch_temp > 0:
                    z = lgt / branch_temp; z = z - z.max(0, keepdims=True)
                    pbr = np.exp(z); pbr /= pbr.sum(0, keepdims=True)
                    o = (sk * pbr[:, None, :]).sum(0)
                else:
                    br = lgt.argmax(0)
                    o = sk[br, :, np.arange(sk.shape[2])].T
                if conf_gate > 0 or record_conf:
                    logit_acc = lgt if logit_acc is None else logit_acc + lgt
                    if record_conf:
                        sk_acc = sk if sk_acc is None else sk_acc + sk
            else:
                o = out_["sdf"][0, 0].cpu().numpy()
            acc = o if acc is None else acc + o
        sdf = (acc / len(models))[:, :s["n"]]
        logit_mean = (logit_acc / len(models)) if logit_acc is not None else None
        dec = render_sw.recover_tvt_soft_logits if logits else render_sw.recover_tvt_soft
        col_tvt = dec(sdf, s["t_seg_tvt"], tau=tau, lam=lam, K=K)
        col_out = col_tvt
        if ed_acc is not None:

            col_out = col_tvt + edit_gain * (ed_acc / len(models))[:s["n"]].astype(col_tvt.dtype)
        if gain > 0:
            z = sdf.astype(np.float64) / max(tau, 1e-6)
            z -= z.max(0, keepdims=True)
            pcol = np.exp(z); pcol /= pcol.sum(0, keepdims=True)
            ent = -(pcol * np.log(pcol + 1e-12)).sum(0)
            thr = np.quantile(ent, gain_q)
            sharp = ent <= thr
            col_tvt = np.where(sharp, atvt + (1.0 + gain) * (col_tvt - atvt), col_tvt).astype(np.float32)
        nh, n, c0 = s["nh"], s["n"], s["c0"]
        npred = n - nh
        col_ent = None; col_spread = None
        if record_conf and logit_acc is not None:
            lm = (logit_acc / len(models))[:, :n]
            z = lm - lm.max(0, keepdims=True); p = np.exp(z); p /= p.sum(0, keepdims=True)
            col_ent = -(p * np.log(p + 1e-12)).sum(0)
            skm = (sk_acc / len(models))
            tvt_k = np.stack([dec(skm[k][:, :n], s["t_seg_tvt"], tau=tau, lam=lam, K=K) for k in range(skm.shape[0])], 0)
            col_spread = tvt_k.std(0)
        for c in range(nh, n):
            f0 = max(c0 + c * S_SW, ps + 1); f1 = min(c0 + (c + 1) * S_SW, L)
            if f0 >= f1:
                continue
            j = c - nh
            w = (1.0 - abs((j + 0.5) / npred - 0.5) * 2.0) + 0.1 if tri else 1.0
            pred_sum[f0:f1] += w * col_out[c]; pred_w[f0:f1] += w
            if col_ent is not None:
                ent_sum[f0:f1] += w * col_ent[c]; spread_sum[f0:f1] += w * col_spread[c]

        rb_col = nh + max(1, int(npred * rebase_frac))
        new_anchor = c0 + rb_col * S_SW
        if new_anchor <= c_anchor or new_anchor + S_SW >= L:

            break
        f_rb = min(new_anchor, L - 1)
        seg0 = max(ps + 1, new_anchor - S_SW)
        lf = _theil_sen_reanchor(col_tvt, rb_col, reanchor_linefit_n) if reanchor_linefit_n > 0 else None
        if lf is not None:

            lf = float(np.clip(lf, s["t_seg_tvt"].min(), s["t_seg_tvt"].max()))
            atvt = lf
        elif pred_w[seg0:new_anchor].sum() > 0:
            atvt = float((pred_sum[seg0:new_anchor] / np.maximum(pred_w[seg0:new_anchor], 1e-9)).mean())
        else:
            atvt = float(col_tvt[rb_col])
        if anchor_ref is not None:
            ref = float(np.nanmean(anchor_ref[seg0:new_anchor])) if new_anchor > seg0 else np.nan
            if np.isfinite(ref):
                atvt = (1.0 - aw) * atvt + aw * ref
        if dz_alpha > 0 and dz_cur is not None:
            dz_ema = dz_cur if dz_ema is None else dz_beta * dz_ema + (1.0 - dz_beta) * dz_cur
            wq = min(1.0, dz_sig0 / max(dz_sig, 1e-6)) if dz_sig0 > 0 else 1.0
            atvt -= dz_alpha * dz_ema * wq
        if conf_gate > 0 and logit_mean is not None:
            col = min(max(rb_col - 1, 0), logit_mean.shape[1] - 1)
            lg_col = logit_mean[:, col].astype(np.float64)
            p_col = np.exp(lg_col - lg_col.max()); p_col /= p_col.sum()
            h_sel = float(-(p_col * np.log(p_col + 1e-12)).sum())
            trust = 1.0 - min(max(h_sel / np.log(K), 0.0), 1.0)
            fallback = float(np.mean(hist_tvt[-2:]))
            atvt = (1.0 - conf_gate * (1.0 - trust)) * atvt + conf_gate * (1.0 - trust) * fallback

        ht = []
        for k in range(HIST):
            a = new_anchor - (HIST - k) * S_SW; b = new_anchor - (HIST - k - 1) * S_SW
            a = max(a, 0); b = max(b, 1)
            if a <= ps:
                ht.append(float(y[a:max(b, a + 1)].mean()))
            elif lf is not None and k == HIST - 1:

                ht.append(atvt)
            elif pred_w[a:b].sum() > 0:
                ht.append(float((pred_sum[a:b] / np.maximum(pred_w[a:b], 1e-9)).mean()))
            else:
                ht.append(atvt)
        hist_tvt = np.array(ht, np.float32)
        c_anchor = new_anchor
    m = pred_w > 0
    pred[m] = pred_sum[m] / pred_w[m]
    lastv = y[ps]
    for i in range(ps + 1, L):
        if np.isnan(pred[i]): pred[i] = lastv
        else: lastv = pred[i]
    if record_conf:
        ent_row = np.full(L, np.nan, np.float32); spread_row = np.full(L, np.nan, np.float32)
        ent_row[m] = ent_sum[m] / pred_w[m]; spread_row[m] = spread_sum[m] / pred_w[m]
        return pred, ent_row, spread_row
    if record:
        return pred, states
    return pred
