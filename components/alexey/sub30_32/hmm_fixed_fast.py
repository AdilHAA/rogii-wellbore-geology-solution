# Alexey Zakharov (drzoidberg), ROGII team solution.
"""Fixed-GR-scale residual-grid HMM."""
from __future__ import annotations

import numpy as np
import torch


def run_batch_fixed_decoders(
    preps,
    dev: str = "cuda",
    pos_mode: str = "floor",
    rate_mode: str = "tri",
    block_lag_ft: float = 0.0,
    include_viterbi: bool = False,
    tilt_eta=None,
    moments_only: bool = False,
):
    """Run fixed-GR-scale HMM decoding."""
    if not preps:
        return []
    bsz = len(preps)
    rdim = len(preps[0]["rates"])
    tmax = max(p["T"] for p in preps)
    pmax = max(p["P"] for p in preps)
    step = float(preps[0]["step"])
    sig_r = float(preps[0]["sig_r"])
    sig_p = float(preps[0]["sig_p"])
    if pos_mode not in {"floor", "linear"}:
        raise ValueError("pos_mode must be floor or linear")
    if rate_mode not in {"tri", "gaussian"}:
        raise ValueError("rate_mode must be tri or gaussian")
    if include_viterbi and (pos_mode != "linear" or rate_mode != "tri"):
        raise ValueError("include_viterbi currently supports pos_mode=linear and rate_mode=tri")
    f32 = torch.float32

    em = torch.zeros(bsz, tmax, pmax, device=dev, dtype=f32)
    dm = torch.ones(bsz, tmax, device=dev, dtype=f32)
    dz = torch.zeros(bsz, tmax, device=dev, dtype=f32)
    rk = torch.zeros(bsz, tmax, device=dev, dtype=f32)
    kap = torch.zeros(bsz, tmax, device=dev, dtype=f32)
    rates = torch.zeros(bsz, rdim, device=dev, dtype=f32)
    grid = torch.zeros(bsz, pmax, device=dev, dtype=f32)
    tw = torch.zeros(bsz, dtype=torch.long)
    reset_end = torch.zeros(bsz, tmax, device=dev, dtype=torch.bool)
    init = torch.zeros(bsz, pmax, rdim, device=dev, dtype=f32)
    tilt_h = torch.zeros(bsz, tmax, pmax, device=dev, dtype=f32)

    for bi, p in enumerate(preps):
        t, pp = int(p["T"]), int(p["P"])
        tw[bi] = t
        if block_lag_ft and block_lag_ft > 0:
            md_np = np.asarray(p["md"], dtype=np.float64)
            block = np.floor((md_np - md_np[0]) / float(block_lag_ft)).astype(np.int64)
            ends = np.zeros(t, dtype=bool)
            ends[-1] = True
            if t > 1:
                ends[:-1] = block[1:] != block[:-1]
            reset_end[bi, :t] = torch.from_numpy(ends).to(dev)
        em[bi, :t, :pp] = torch.from_numpy(np.asarray(p["em"], dtype=np.float32)).to(dev)
        dm[bi, :t] = torch.from_numpy(np.asarray(p["dm"], dtype=np.float32)).to(dev)
        dz[bi, :t] = torch.from_numpy(np.asarray(p["dz"], dtype=np.float32)).to(dev)
        rk[bi, :t] = torch.from_numpy(np.asarray(p["rk"], dtype=np.float32)).to(dev)
        kap[bi, :t] = torch.from_numpy(np.asarray(p["kap"], dtype=np.float32)).to(dev)
        rates[bi] = torch.from_numpy(np.asarray(p["rates"], dtype=np.float32)).to(dev)
        grid[bi, :pp] = torch.from_numpy(np.asarray(p["grid"], dtype=np.float32)).to(dev)
        if "tilt_h" in p:
            h = np.asarray(p["tilt_h"], dtype=np.float32)
            if h.shape != (t, pp):
                raise ValueError(f"{p['well']}: tilt_h must have shape {(t, pp)}")
            tilt_h[bi, :t, :pp] = torch.from_numpy(h).to(dev)

        pos = torch.arange(pp, device=dev, dtype=f32)
        dpos = (pos - float(p["start_p"])) * step
        lp0 = torch.exp(-0.5 * (dpos / float(p["start_sig"])) ** 2)
        dr = (rates[bi][None, :] - float(p["ir"])) / float(p["r0_sig"])
        init[bi, :pp] = lp0[:, None] * torch.exp(-0.5 * dr * dr)

    init /= init.sum((1, 2), keepdim=True).clamp_min(1e-30)
    if tilt_eta is None:
        eta = torch.zeros(bsz, device=dev, dtype=f32)
    else:
        eta = torch.as_tensor(tilt_eta, device=dev, dtype=f32).reshape(-1)
        if eta.numel() == 1:
            eta = eta.expand(bsz)
        elif eta.numel() != bsz:
            raise ValueError("tilt_eta must be scalar or have one value per prep")
    log_tilt = eta[:, None, None] * tilt_h
    log_tilt = log_tilt - log_tilt.amax(dim=2, keepdim=True)
    em = em * torch.exp(log_tilt)


    sr = (rates[:, 1] - rates[:, 0]).clamp_min(1e-9)
    v_r = (sig_r * torch.sqrt(dm)[:, :, None] / sr[:, None, None]) ** 2
    m_r = kap[:, :, None] * (rk[:, :, None] - rates[:, None, :]) * dm[:, :, None] / sr[:, None, None]
    pp = (0.5 * (v_r + m_r)).clamp_min(1e-12)
    pm = (0.5 * (v_r - m_r)).clamp_min(1e-12)
    tot = pp + pm
    sc = torch.where(tot > 0.9, 0.9 / tot, torch.ones_like(tot))
    pp = pp * sc
    pm = pm * sc
    mid = (1.0 - pp - pm).clamp_min(1e-12)

    pidx = torch.arange(pmax, device=dev)
    alpha = None
    if not moments_only:
        alpha = torch.zeros(bsz, tmax, pmax, rdim, device=dev, dtype=torch.float16)
    filter_mean = torch.zeros(bsz, tmax, device=dev, dtype=f32)
    twd = tw.to(dev)
    neg = torch.tensor(-1.0e30, device=dev, dtype=f32)
    if include_viterbi and moments_only:
        raise ValueError("Viterbi is unavailable in moments_only mode")
    if include_viterbi:
        vdelta = torch.where(init > 0, torch.log(init), neg)
        ptr_p = torch.empty(bsz, tmax, pmax, rdim, device=dev, dtype=torch.uint8)
        ptr_r = torch.empty(bsz, tmax, pmax, rdim, device=dev, dtype=torch.uint8)

    def rate_fwd(x, t):
        out = x * mid[:, t, None, :]
        out[:, :, 1:] = out[:, :, 1:] + x[:, :, :-1] * pp[:, t, None, :-1]
        out[:, :, :-1] = out[:, :, :-1] + x[:, :, 1:] * pm[:, t, None, 1:]
        return out

    def rate_bwd(x, t):
        out = x * mid[:, t, None, :]
        out[:, :, :-1] = out[:, :, :-1] + x[:, :, 1:] * pp[:, t, None, :-1]
        out[:, :, 1:] = out[:, :, 1:] + x[:, :, :-1] * pm[:, t, None, 1:]
        return out

    def rate_viterbi(x, t):
        log_mid = torch.log(mid[:, t].clamp_min(1e-30))
        log_pp = torch.log(pp[:, t].clamp_min(1e-30))
        log_pm = torch.log(pm[:, t].clamp_min(1e-30))
        cand0 = x + log_mid[:, None, :]
        cand1 = torch.full_like(x, -1.0e30)
        cand2 = torch.full_like(x, -1.0e30)
        cand1[:, :, 1:] = x[:, :, :-1] + log_pp[:, None, :-1]
        cand2[:, :, :-1] = x[:, :, 1:] + log_pm[:, None, 1:]
        vals = torch.stack((cand0, cand1, cand2), dim=0)
        return vals.max(dim=0)

    def rate_kernel_gaussian(t):


        mu = rates + kap[:, t, None] * (rk[:, t, None] - rates) * dm[:, t, None]
        sigma = (sig_r * torch.sqrt(dm[:, t])).clamp_min(1e-8)
        lo = rates - 0.5 * sr[:, None]
        hi = rates + 0.5 * sr[:, None]
        inv = 1.0 / (sigma[:, None, None] * 2.0**0.5)
        zhi = (hi[:, None, :] - mu[:, :, None]) * inv
        zlo = (lo[:, None, :] - mu[:, :, None]) * inv
        k = 0.5 * (torch.erf(zhi) - torch.erf(zlo))
        return k.clamp_min(0.0)

    def apply_rate_fwd(x, t):
        if rate_mode == "tri":
            return rate_fwd(x, t)
        return torch.bmm(x, rate_kernel_gaussian(t))

    def apply_rate_bwd(x, t):
        if rate_mode == "tri":
            return rate_bwd(x, t)
        return torch.bmm(x, rate_kernel_gaussian(t).transpose(1, 2))

    def posshift(x, t, transpose=False):
        mu = rates * dm[:, t][:, None] - dz[:, t][:, None]
        if pos_mode == "linear":
            u = mu / step
            j0 = torch.floor(u)
            frac = (u - j0).clamp(0.0, 1.0)
            out = torch.zeros_like(x)
            for off, wt in ((j0, 1.0 - frac), (j0 + 1.0, frac)):
                sign = 1.0 if not transpose else -1.0
                src = (pidx[None, :, None] - sign * off[:, None, :]).round().long()
                valid = (src >= 0) & (src < pmax)
                srcc = src.clamp(0, pmax - 1)
                got = torch.gather(x, 1, srcc)
                out = out + torch.where(valid, got * wt[:, None, :], torch.zeros_like(got))
            return out
        b0 = torch.floor(mu / step + 0.5)
        spe = max(sig_p, 0.35 * step)
        out = torch.zeros_like(x)
        for kk in range(5):
            off = b0 + (kk - 2)
            d = off * step - mu
            wt = torch.exp(-0.5 * (d / spe) ** 2)
            sign = 1.0 if not transpose else -1.0
            src = (pidx[None, :, None] - sign * off[:, None, :]).round().long()
            valid = (src >= 0) & (src < pmax)
            srcc = src.clamp(0, pmax - 1)
            got = torch.gather(x, 1, srcc)
            out = out + torch.where(valid, got * wt[:, None, :], torch.zeros_like(got))
        offs = torch.arange(5, device=dev, dtype=f32) - 2.0
        wsum = torch.exp(
            -0.5
            * (((offs[None, None, :] + b0[:, :, None]) * step - mu[:, :, None]) / spe) ** 2
        ).sum(-1).clamp_min(1e-12)
        return out / wsum[:, None, :]

    def pos_viterbi(x, t):
        mu = rates * dm[:, t][:, None] - dz[:, t][:, None]
        u = mu / step
        j0 = torch.floor(u)
        frac = (u - j0).clamp(0.0, 1.0)
        vals = []
        for off, wt in ((j0, 1.0 - frac), (j0 + 1.0, frac)):
            src = (pidx[None, :, None] - off[:, None, :]).round().long()
            valid = (src >= 0) & (src < pmax)
            srcc = src.clamp(0, pmax - 1)
            got = torch.gather(x, 1, srcc) + torch.log(wt[:, None, :].clamp_min(1e-30))
            vals.append(torch.where(valid, got, torch.full_like(got, -1.0e30)))
        st = torch.stack(vals, dim=0)
        return st.max(dim=0)

    prev = init
    moment1 = torch.zeros_like(init) if moments_only else None
    moment2 = torch.zeros_like(init) if moments_only else None
    for t in range(tmax):
        pred = posshift(apply_rate_fwd(prev, t), t, False)
        cur = pred * em[:, t, :, None]
        norm = cur.sum((1, 2), keepdim=True).clamp_min(1e-30)
        cur = cur / norm
        real = (t < twd)[:, None, None]
        cur = torch.where(real, cur, prev)
        if moments_only:
            pred1 = posshift(apply_rate_fwd(moment1, t), t, False)
            pred2 = posshift(apply_rate_fwd(moment2, t), t, False)
            h = tilt_h[:, t, :, None]
            next1 = (pred1 + pred * h) * em[:, t, :, None] / norm
            next2 = (
                pred2 + 2.0 * pred1 * h + pred * h * h
            ) * em[:, t, :, None] / norm
            moment1 = torch.where(real, next1, moment1)
            moment2 = torch.where(real, next2, moment2)
        if include_viterbi:
            rv, rc = rate_viterbi(vdelta, t)
            pv, pc = pos_viterbi(rv, t)
            log_em = torch.where(em[:, t, :, None] > 0, torch.log(em[:, t, :, None]), neg)
            next_delta = pv + log_em
            vdelta = torch.where(real, next_delta, vdelta)
            ptr_r[:, t] = rc.to(torch.uint8)
            ptr_p[:, t] = pc.to(torch.uint8)
        gp = cur.sum(-1)
        fm = (gp * grid).sum(-1)
        filter_mean[:, t] = torch.where((t < twd), fm, torch.zeros_like(fm))
        if alpha is not None:
            alpha[:, t] = cur.half()
        prev = cur

    if moments_only:
        out = []
        for bi, p in enumerate(preps):
            mean_h = float(moment1[bi].sum().item())
            second_h = float(moment2[bi].sum().item())
            out.append(
                {
                    "well": p["well"],
                    "mean_h": mean_h,
                    "var_h": max(0.0, second_h - mean_h * mean_h),
                }
            )
        return out

    beta_next = torch.ones(bsz, pmax, rdim, device=dev, dtype=f32)
    smooth_mean = torch.zeros(bsz, tmax, device=dev, dtype=f32)
    smooth_mode = torch.zeros(bsz, tmax, device=dev, dtype=f32)
    smooth_std = torch.zeros(bsz, tmax, device=dev, dtype=f32)
    pos_edge1 = torch.zeros(bsz, tmax, device=dev, dtype=f32)
    pos_edge5 = torch.zeros(bsz, tmax, device=dev, dtype=f32)
    rate_edge1 = torch.zeros(bsz, tmax, device=dev, dtype=f32)
    rate_edge2 = torch.zeros(bsz, tmax, device=dev, dtype=f32)
    posterior_h = torch.zeros(bsz, device=dev, dtype=f32)
    ar_b = torch.arange(bsz, device=dev)

    for t in range(tmax - 1, -1, -1):
        real = t < twd
        if block_lag_ft and block_lag_ft > 0:
            reset = reset_end[:, t] & real
            beta_next = torch.where(reset[:, None, None], torch.ones_like(beta_next), beta_next)
        g = alpha[:, t].float() * beta_next
        g = g / g.sum((1, 2), keepdim=True).clamp_min(1e-30)
        gp = g.sum(-1)
        gr = g.sum(1)
        m = (gp * grid).sum(-1)
        v = (gp * grid * grid).sum(-1) - m * m
        smooth_mean[:, t] = torch.where(real, m, torch.zeros_like(m))
        smooth_std[:, t] = torch.where(real, torch.sqrt(v.clamp_min(0.0)), torch.zeros_like(m))
        imode = gp.argmax(-1)
        smooth_mode[:, t] = torch.where(real, grid[ar_b, imode], torch.zeros_like(m))
        pos_edge1[:, t] = torch.where(real, gp[:, 0] + gp[:, -1], torch.zeros_like(m))
        pos_edge5[:, t] = torch.where(real, gp[:, :5].sum(-1) + gp[:, -5:].sum(-1), torch.zeros_like(m))
        rate_edge1[:, t] = torch.where(real, gr[:, 0] + gr[:, -1], torch.zeros_like(m))
        rate_edge2[:, t] = torch.where(real, gr[:, :2].sum(-1) + gr[:, -2:].sum(-1), torch.zeros_like(m))
        posterior_h = posterior_h + torch.where(
            real,
            (gp * tilt_h[:, t]).sum(-1),
            torch.zeros_like(m),
        )
        if t > 0:
            msg = em[:, t, :, None] * beta_next
            btmp = posshift(msg, t, True)
            bcur = apply_rate_bwd(btmp, t)
            bcur = bcur / bcur.sum((1, 2), keepdim=True).clamp_min(1e-30)
            beta_next = torch.where(real[:, None, None], bcur, beta_next)

    out = []
    vpaths = [None] * bsz
    if include_viterbi:
        ptr_p_cpu = ptr_p.cpu().numpy()
        ptr_r_cpu = ptr_r.cpu().numpy()
        delta_cpu = vdelta.cpu().numpy()
        for bi, p in enumerate(preps):
            tlen = int(p["T"])
            rates_np = np.asarray(p["rates"], dtype=np.float64)
            dm_np = np.asarray(p["dm"], dtype=np.float64)
            dz_np = np.asarray(p["dz"], dtype=np.float64)
            grid_np = np.asarray(p["grid"], dtype=np.float64)
            flat = int(np.argmax(delta_cpu[bi]))
            pos = flat // rdim
            rr = flat % rdim
            path = np.empty(tlen, dtype=np.float64)
            for tt in range(tlen - 1, -1, -1):
                pos = int(np.clip(pos, 0, len(grid_np) - 1))
                rr = int(np.clip(rr, 0, rdim - 1))
                path[tt] = grid_np[pos]
                pc = int(ptr_p_cpu[bi, tt, pos, rr])
                mu = rates_np[rr] * dm_np[tt] - dz_np[tt]
                j0 = np.floor(mu / step)
                off = j0 if pc == 0 else j0 + 1.0
                src_pos = int(round(pos - off))
                src_pos = int(np.clip(src_pos, 0, len(grid_np) - 1))
                rc = int(ptr_r_cpu[bi, tt, src_pos, rr])
                if rc == 1:
                    src_r = rr - 1
                elif rc == 2:
                    src_r = rr + 1
                else:
                    src_r = rr
                pos, rr = src_pos, int(np.clip(src_r, 0, rdim - 1))
            vpaths[bi] = path
    for bi, p in enumerate(preps):
        t = p["T"]
        ref = p["tvt_ref"]
        d = dict(
            smooth=smooth_mean[bi, :t].cpu().numpy() + ref,
            filter=filter_mean[bi, :t].cpu().numpy() + ref,
            mode=smooth_mode[bi, :t].cpu().numpy() + ref,
            std=smooth_std[bi, :t].cpu().numpy(),
            diag=dict(
                pos_edge1=pos_edge1[bi, :t].cpu().numpy(),
                pos_edge5=pos_edge5[bi, :t].cpu().numpy(),
                rate_edge1=rate_edge1[bi, :t].cpu().numpy(),
                rate_edge2=rate_edge2[bi, :t].cpu().numpy(),
                posterior_h=float(posterior_h[bi].item()),
            ),
            ev_idx=p["ev_idx"],
            nrow=p["nrow"],
            well=p["well"],
        )
        if include_viterbi:
            d["viterbi"] = vpaths[bi] + ref
        out.append(d)
    return out
