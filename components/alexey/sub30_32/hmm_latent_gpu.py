# Alexey Zakharov (drzoidberg), ROGII team solution.
"""Latent-GR-scale residual-grid HMM smoother."""
from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import torch


def _make_emissions(
    res2: np.ndarray,
    valid: np.ndarray,
    gs_values: Iterable[float],
    max_gr_z2: float = 600.0,
    min_relative_log_weight: float = -80.0,
    include_gs_normalizer: bool = True,
) -> np.ndarray:
    """Build relative probability emissions with one rowwise shift per row."""
    gs_arr = np.asarray(tuple(gs_values), dtype=np.float64)
    if gs_arr.ndim != 1 or len(gs_arr) == 0 or np.any(gs_arr <= 0):
        raise ValueError("gs_values must be positive")
    res2 = np.asarray(res2, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    if res2.shape != valid.shape:
        raise ValueError("res2 and valid must have the same shape")
    t, p = res2.shape
    log_em = np.zeros((t, len(gs_arr), p), dtype=np.float64)
    for zi, gs in enumerate(gs_arr):
        z2 = np.minimum(res2 / (gs * gs), max_gr_z2)
        le = -0.5 * z2
        if include_gs_normalizer:
            le = le - math.log(gs)
        log_em[:, zi, :] = np.where(valid, le, 0.0)
    row_max = np.max(log_em, axis=(1, 2), keepdims=True)
    rel = np.maximum(log_em - row_max, min_relative_log_weight)
    return np.exp(rel).astype(np.float32)


def run_batch_latent(
    preps,
    dev: str = "cuda",
    pos_mode: str = "floor",
    gs_values=(43.0, 90.0),
    regime_length_ft: float = 1000.0,
    regime_prior=None,
    max_gr_z2: float = 600.0,
    include_gs_normalizer: bool = True,
    return_feats: bool = False,
):
    """Run full forward-backward smoothing for one fixed or latent-gs config."""
    if not preps:
        return []
    gs_values = tuple(float(x) for x in gs_values)
    zdim = len(gs_values)
    if zdim < 1:
        raise ValueError("at least one gs value is required")
    if pos_mode not in {"floor", "linear"}:
        raise ValueError("pos_mode must be floor or linear")
    if regime_prior is None:
        prior = np.full(zdim, 1.0 / zdim, dtype=np.float32)
    else:
        prior = np.asarray(regime_prior, dtype=np.float32)
        if prior.shape != (zdim,) or np.any(prior < 0) or prior.sum() <= 0:
            raise ValueError("regime_prior shape/content mismatch")
        prior = prior / prior.sum()

    bsz = len(preps)
    rdim = len(preps[0]["rates"])
    tmax = max(p["T"] for p in preps)
    pmax = max(p["P"] for p in preps)
    step = float(preps[0]["step"])
    sig_r = float(preps[0]["sig_r"])
    sig_p = float(preps[0]["sig_p"])
    f32 = torch.float32

    em = torch.zeros(bsz, tmax, zdim, pmax, device=dev, dtype=f32)
    dm = torch.ones(bsz, tmax, device=dev, dtype=f32)
    dz = torch.zeros(bsz, tmax, device=dev, dtype=f32)
    rk = torch.zeros(bsz, tmax, device=dev, dtype=f32)
    kap = torch.zeros(bsz, tmax, device=dev, dtype=f32)
    rates = torch.zeros(bsz, rdim, device=dev, dtype=f32)
    grid = torch.zeros(bsz, pmax, device=dev, dtype=f32)
    tw = torch.zeros(bsz, dtype=torch.long)
    pw = torch.zeros(bsz, dtype=torch.long)
    init = torch.zeros(bsz, zdim, pmax, rdim, device=dev, dtype=f32)
    prior_t = torch.from_numpy(prior).to(dev, f32)

    for bi, prep in enumerate(preps):
        t, p = int(prep["T"]), int(prep["P"])
        tw[bi] = t
        pw[bi] = p
        em_np = _make_emissions(
            prep["res2"],
            prep["valid"],
            gs_values,
            max_gr_z2=max_gr_z2,
            include_gs_normalizer=include_gs_normalizer,
        )
        em[bi, :t, :, :p] = torch.from_numpy(em_np).to(dev)
        dm[bi, :t] = torch.from_numpy(np.asarray(prep["dm"], dtype=np.float32)).to(dev)
        dz[bi, :t] = torch.from_numpy(np.asarray(prep["dz"], dtype=np.float32)).to(dev)
        rk[bi, :t] = torch.from_numpy(np.asarray(prep["rk"], dtype=np.float32)).to(dev)
        kap[bi, :t] = torch.from_numpy(np.asarray(prep["kap"], dtype=np.float32)).to(dev)
        rates[bi] = torch.from_numpy(np.asarray(prep["rates"], dtype=np.float32)).to(dev)
        grid[bi, :p] = torch.from_numpy(np.asarray(prep["grid"], dtype=np.float32)).to(dev)

        pos = torch.arange(p, device=dev, dtype=f32)
        dpos = (pos - float(prep["start_p"])) * step
        lp0 = torch.exp(-0.5 * (dpos / float(prep["start_sig"])) ** 2)
        dr = (rates[bi][None, :] - float(prep["ir"])) / float(prep["r0_sig"])
        base = lp0[:, None] * torch.exp(-0.5 * dr * dr)
        init[bi, :, :p, :] = prior_t[:, None, None] * base[None, :, :]

    init /= init.sum((1, 2, 3), keepdim=True).clamp_min(1e-30)
    sr = (rates[:, 1] - rates[:, 0]).clamp_min(1e-9)
    pidx = torch.arange(pmax, device=dev)
    ridx = torch.arange(rdim, device=dev)
    alpha = torch.zeros(bsz, tmax, zdim, pmax, rdim, device=dev, dtype=torch.float16)
    filter_mean = torch.zeros(bsz, tmax, device=dev, dtype=f32)

    def krate(t):
        dmt = dm[:, t]
        v_r = (sig_r * torch.sqrt(dmt) / sr) ** 2
        m_r = kap[:, t][:, None] * (rk[:, t][:, None] - rates) * dmt[:, None] / sr[:, None]
        pp = 0.5 * (v_r[:, None] + m_r)
        pm = 0.5 * (v_r[:, None] - m_r)
        pp = pp.clamp_min(1e-12)
        pm = pm.clamp_min(1e-12)
        tot = pp + pm
        sc = torch.where(tot > 0.9, 0.9 / tot, torch.ones_like(tot))
        pp = pp * sc
        pm = pm * sc
        mid = (1.0 - pp - pm).clamp_min(1e-12)
        k = torch.zeros(bsz, rdim, rdim, device=dev, dtype=f32)
        k[:, ridx, ridx] = mid
        k[:, ridx[1:], ridx[1:] - 1] = pm[:, 1:]
        k[:, ridx[:-1], ridx[:-1] + 1] = pp[:, :-1]
        return k

    def regime_mix(x, t):
        if zdim == 1:
            return x
        if regime_length_ft <= 0:
            sw = torch.full((bsz,), 1.0 - 1.0 / zdim, device=dev, dtype=f32)
        else:
            sw = (1.0 - torch.exp(-dm[:, t] / float(regime_length_ft))).clamp(0.0, 1.0)
        if zdim == 2:
            stay = 1.0 - sw
            y0 = stay[:, None, None] * x[:, 0] + sw[:, None, None] * x[:, 1]
            y1 = sw[:, None, None] * x[:, 0] + stay[:, None, None] * x[:, 1]
            return torch.stack((y0, y1), dim=1)

        stay = 1.0 - sw
        other = sw / float(zdim - 1)
        total = x.sum(1, keepdim=True)
        return stay[:, None, None, None] * x + other[:, None, None, None] * (total - x)

    def posshift(x, t, transpose=False):
        mu = rates * dm[:, t][:, None] - dz[:, t][:, None]
        if pos_mode == "linear":
            u = mu / step
            j0 = torch.floor(u)
            frac = (u - j0).clamp(0.0, 1.0)
            out = torch.zeros_like(x)
            for off, wt in ((j0, 1.0 - frac), (j0 + 1.0, frac)):
                sign = 1.0 if not transpose else -1.0
                src = (pidx[None, None, :, None] - sign * off[:, None, None, :]).round().long()
                valid = (src >= 0) & (src < pmax)
                srcc = src.clamp(0, pmax - 1).expand(bsz, zdim, pmax, rdim)
                got = torch.gather(x, 2, srcc)
                out = out + torch.where(
                    valid.expand_as(got),
                    got * wt[:, None, None, :],
                    torch.zeros_like(got),
                )
            return out
        b0 = torch.floor(mu / step + 0.5)
        spe = max(sig_p, 0.35 * step)
        out = torch.zeros_like(x)
        for kk in range(5):
            off = b0 + (kk - 2)
            d = off * step - mu
            wt = torch.exp(-0.5 * (d / spe) ** 2)
            sign = 1.0 if not transpose else -1.0
            src = (pidx[None, None, :, None] - sign * off[:, None, None, :]).round().long()
            valid = (src >= 0) & (src < pmax)
            srcc = src.clamp(0, pmax - 1).expand(bsz, zdim, pmax, rdim)
            got = torch.gather(x, 2, srcc)
            out = out + torch.where(valid.expand_as(got), got * wt[:, None, None, :], torch.zeros_like(got))
        offs = torch.arange(5, device=dev, dtype=f32) - 2.0
        wsum = torch.exp(
            -0.5
            * (((offs[None, None, :] + b0[:, :, None]) * step - mu[:, :, None]) / spe) ** 2
        ).sum(-1).clamp_min(1e-12)
        return out / wsum[:, None, None, :]

    prev = init
    twd = tw.to(dev)
    for t in range(tmax):
        premix = regime_mix(prev, t)
        k = krate(t)
        tmp = torch.bmm(
            premix.reshape(bsz * zdim, pmax, rdim),
            k[:, None, :, :].expand(bsz, zdim, rdim, rdim).reshape(bsz * zdim, rdim, rdim),
        ).reshape(bsz, zdim, pmax, rdim)
        cur = posshift(tmp, t, transpose=False)
        cur = cur * em[:, t, :, :, None]
        scale = cur.sum((1, 2, 3), keepdim=True).clamp_min(1e-30)
        cur = cur / scale
        active = (t < twd)[:, None, None, None]
        cur = torch.where(active, cur, prev)
        gp_f = cur.sum((1, 3))
        fm = (gp_f * grid).sum(-1)
        filter_mean[:, t] = torch.where((t < twd), fm, torch.zeros_like(fm))
        alpha[:, t] = cur.half()
        prev = cur

    beta_next = torch.ones(bsz, zdim, pmax, rdim, device=dev, dtype=f32)
    beta_next = beta_next / beta_next.sum((1, 2, 3), keepdim=True).clamp_min(1e-30)
    mean = torch.zeros(bsz, tmax, device=dev, dtype=f32)
    std = torch.zeros(bsz, tmax, device=dev, dtype=f32)
    p_sharp = torch.zeros(bsz, tmax, device=dev, dtype=f32)
    ent_z = torch.zeros(bsz, tmax, device=dev, dtype=f32)
    mode = torch.zeros(bsz, tmax, device=dev, dtype=f32)

    for t in range(tmax - 1, -1, -1):
        real = t < twd
        g = alpha[:, t].float() * beta_next
        norm = g.sum((1, 2, 3), keepdim=True).clamp_min(1e-30)
        g = g / norm
        gp = g.sum((1, 3))
        m = (gp * grid).sum(-1)
        v = (gp * grid * grid).sum(-1) - m * m
        mean[:, t] = torch.where(real, m, torch.zeros_like(m))
        std[:, t] = torch.where(real, torch.sqrt(v.clamp_min(0.0)), torch.zeros_like(m))
        mode_idx = gp.argmax(-1)
        mode[:, t] = torch.where(real, grid[torch.arange(bsz, device=dev), mode_idx], torch.zeros_like(m))
        gz = g.sum((2, 3))
        p_sharp[:, t] = torch.where(real, gz[:, 0], torch.zeros_like(m))
        ez = -(gz.clamp_min(1e-12) * gz.clamp_min(1e-12).log()).sum(-1)
        ent_z[:, t] = torch.where(real, ez, torch.zeros_like(m))

        if t > 0:
            msg = em[:, t, :, :, None] * beta_next
            btmp = posshift(msg, t, transpose=True)
            k = krate(t)
            bcur = torch.bmm(
                btmp.reshape(bsz * zdim, pmax, rdim),
                k.transpose(1, 2)[:, None, :, :].expand(bsz, zdim, rdim, rdim).reshape(
                    bsz * zdim, rdim, rdim
                ),
            ).reshape(bsz, zdim, pmax, rdim)
            bcur = regime_mix(bcur, t)
            bcur = bcur / bcur.sum((1, 2, 3), keepdim=True).clamp_min(1e-30)
            beta_next = torch.where(real[:, None, None, None], bcur, beta_next)

    out = []
    for bi, prep in enumerate(preps):
        t = prep["T"]
        pred = mean[bi, :t].cpu().numpy() + prep["tvt_ref"]
        st = std[bi, :t].cpu().numpy()
        feats = dict(
            p_sharp=p_sharp[bi, :t].cpu().numpy(),
            regime_entropy=ent_z[bi, :t].cpu().numpy(),
            mode=mode[bi, :t].cpu().numpy() + prep["tvt_ref"],
            filter=filter_mean[bi, :t].cpu().numpy() + prep["tvt_ref"],
        )
        if return_feats:
            out.append((pred, st, prep["ev_idx"], prep["nrow"], feats))
        else:
            out.append((pred, st, prep["ev_idx"], prep["nrow"]))
    return out
