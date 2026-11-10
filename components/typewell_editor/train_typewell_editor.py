# Andrey Serzhenko
"""EXP-256: Prefix-conditioned Typewell Editor.

The model edits the typewell itself.  It receives the full original typewell
and only the observable horizontal prefix projected onto the typewell TVT
coordinate.  It outputs a corrected GR curve on a fixed normalized TVT grid.

The true postfix TVT is never used to build model inputs.  During training it
is used only to sample the predicted typewell for the measurement-consistency
loss.  Validation follows exactly the same input contract and uses true TVT
only for metrics.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--raw-root", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--fold", type=int, default=None)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--grid", type=int, default=2048)
    p.add_argument("--val-every", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default=None)
    return p.parse_args()


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def robust_fill(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, np.float32)
    if np.isfinite(x).all():
        return x
    good = np.isfinite(x)
    if not good.any():
        return np.zeros_like(x)
    idx = np.arange(len(x), dtype=np.float32)
    out = np.interp(idx, idx[good], x[good]).astype(np.float32)
    return out


def interp_np(x: np.ndarray, xp: np.ndarray, fp: np.ndarray) -> np.ndarray:
    order = np.argsort(xp)
    xp = np.asarray(xp, np.float32)[order]
    fp = np.asarray(fp, np.float32)[order]
    xp, keep = np.unique(xp, return_index=True)
    fp = fp[keep]
    if len(xp) < 2:
        return np.full_like(np.asarray(x, np.float32), fp[0] if len(fp) else 0.0)
    return np.interp(np.asarray(x, np.float32), xp, fp).astype(np.float32)


def load_well(raw_root: Path, wid: str) -> dict[str, Any]:
    base = raw_root / "competition" / "raw" / "train" / wid
    h = pd.read_csv(str(base) + "__horizontal_well.csv")
    t = pd.read_csv(str(base) + "__typewell.csv")
    h_gr = robust_fill(h["GR"].to_numpy(np.float32))
    tvt_true = h["TVT"].to_numpy(np.float32)
    tvt_input = h["TVT_input"].to_numpy(np.float32)
    known = np.isfinite(tvt_input)
    gr_present = np.isfinite(h["GR"].to_numpy(np.float32))
    if not known.any() or len(t) < 8:
        raise ValueError("missing prefix or short typewell")
    return {
        "wid": wid,
        "h_gr": h_gr,
        "h_gr_present": gr_present.astype(np.float32),
        "tvt_true": tvt_true,
        "tvt_input": tvt_input,
        "t_tvt": t["TVT"].to_numpy(np.float32),
        "t_gr": robust_fill(t["GR"].to_numpy(np.float32)),
    }


def list_wells(raw_root: Path) -> list[str]:
    d = raw_root / "competition" / "raw" / "train"
    return sorted(p.name.split("__")[0] for p in d.glob("*__horizontal_well.csv"))


def make_folds(raw_root: Path, wids: list[str]) -> dict[str, int]:
    fold_path = raw_root.parent.parent.parent / "research" / "folds.csv"


    candidates = [fold_path, Path(__file__).resolve().parents[2] / "research" / "folds.csv"]
    for path in candidates:
        if path.exists():
            df = pd.read_csv(path, dtype={"wid": str})
            got = {str(r.wid): int(r.fold) for r in df.itertuples() if str(r.wid) in set(wids)}
            if len(got) == len(wids):
                return got
    rng = np.random.RandomState(42)
    order = rng.permutation(sorted(wids))
    return {wid: i % 5 for i, wid in enumerate(order)}


def robust_stats(wells: dict[str, dict], ids: list[str]) -> tuple[float, float]:
    vals = np.concatenate([wells[wid]["t_gr"] for wid in ids])
    med = float(np.nanmedian(vals))
    q25, q75 = np.nanpercentile(vals, [25, 75])
    scale = max(10.0, float((q75 - q25) / 1.349))
    return med, scale


def project_prefix(w: dict[str, Any], u: np.ndarray, center: float, scale: float) -> tuple[np.ndarray, ...]:
    """Project only known horizontal observations onto normalized TVT grid."""
    t0, t1 = float(w["t_tvt"][0]), float(w["t_tvt"][-1])
    denom = max(t1 - t0, 1e-3)
    known = np.isfinite(w["tvt_input"]) & np.isfinite(w["h_gr_present"].astype(bool))
    if known.sum() < 2:
        raise ValueError("not enough known prefix points")
    ku = np.clip((w["tvt_input"][known] - t0) / denom, 0.0, 1.0)
    kg = w["h_gr"][known]
    order = np.argsort(ku)
    ku, kg = ku[order], kg[order]
    ku, keep = np.unique(ku, return_index=True)
    kg = kg[keep]
    if len(ku) < 2:
        raise ValueError("not enough unique prefix TVT points")
    prefix = np.zeros_like(u, np.float32)
    mask = ((u >= ku[0]) & (u <= ku[-1])).astype(np.float32)
    prefix[mask > 0] = np.interp(u[mask > 0], ku, kg).astype(np.float32)
    original = interp_np(u, (w["t_tvt"] - t0) / denom, w["t_gr"])
    residual = (prefix - original) * mask
    smooth_resid = residual.copy()


    k = 17
    kernel = np.ones(k, np.float32) / k
    smooth_resid = np.convolve(np.pad(residual, (k // 2, k // 2), mode="edge"), kernel, mode="valid")
    end_u = float(ku[-1])
    dist = np.clip(u - end_u, -1.0, 1.0).astype(np.float32)
    return prefix, residual, smooth_resid.astype(np.float32), mask, dist, original.astype(np.float32)


def build_item(w: dict[str, Any], grid: int, center: float, scale: float) -> dict[str, np.ndarray]:
    t = w["t_tvt"]
    t0, t1 = float(t[0]), float(t[-1])
    u = np.linspace(0.0, 1.0, grid, dtype=np.float32)
    tw = interp_np(u, (t - t0) / max(t1 - t0, 1e-3), w["t_gr"])
    prefix, residual, smooth_resid, pmask, dist, tw_check = project_prefix(w, u, center, scale)
    tw_norm = (tw - center) / scale
    d1 = np.gradient(tw_norm, u).astype(np.float32)
    d2 = np.gradient(d1, u).astype(np.float32)


    d1 = np.clip(d1 / 8.0, -5.0, 5.0).astype(np.float32)
    d2 = np.clip(d2 / 8.0, -5.0, 5.0).astype(np.float32)
    x = np.stack([
        tw_norm,
        d1,
        d2,
        (prefix - center) / scale * pmask,
        residual / scale,
        pmask,
        dist,
        u,
    ], axis=0).astype(np.float32)
    tvt_u = np.clip((w["tvt_true"] - t0) / max(t1 - t0, 1e-3), 0.0, 1.0)
    y = ((w["h_gr"] - center) / scale).astype(np.float32)

    valid = np.isfinite(y) & np.isfinite(tvt_u)
    post = ~np.isfinite(w["tvt_input"])
    weights = np.where(post, 1.0, 0.25).astype(np.float32)
    weights[~valid] = 0.0
    return {
        "x": x,
        "grid_u": u,
        "tvt_u": tvt_u.astype(np.float32),
        "target": np.nan_to_num(y, nan=0.0),
        "weights": weights,
        "valid": valid.astype(np.float32),
        "t0": np.array([t0], np.float32),
        "t1": np.array([t1], np.float32),
        "original_t": w["t_tvt"].astype(np.float32),
        "original_gr": w["t_gr"].astype(np.float32),
        "wid": np.array([w["wid"]]),
    }


class EditorDataset(Dataset):
    def __init__(self, wells: dict[str, dict], ids: list[str], grid: int, center: float, scale: float):
        self.items = []
        self.wids = []
        for wid in ids:
            try:
                item = build_item(wells[wid], grid, center, scale)
            except Exception:
                continue
            self.items.append(item)
            self.wids.append(wid)
        if not self.items:
            raise RuntimeError("empty typewell editor dataset")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int) -> dict[str, Any]:
        return self.items[i]


def collate(items: list[dict[str, Any]]) -> dict[str, Any]:
    max_m = max(len(x["target"]) for x in items)
    b = len(items)
    x = torch.from_numpy(np.stack([z["x"] for z in items]))
    tvt_u = np.zeros((b, max_m), np.float32)
    target = np.zeros((b, max_m), np.float32)
    weights = np.zeros((b, max_m), np.float32)
    valid = np.zeros((b, max_m), np.float32)
    lengths = []
    for i, z in enumerate(items):
        m = len(z["target"])
        lengths.append(m)
        tvt_u[i, :m] = z["tvt_u"]
        target[i, :m] = z["target"]
        weights[i, :m] = z["weights"]
        valid[i, :m] = z["valid"]
    return {
        "x": x,
        "tvt_u": torch.from_numpy(tvt_u),
        "target": torch.from_numpy(target),
        "weights": torch.from_numpy(weights),
        "valid": torch.from_numpy(valid),
        "lengths": lengths,
        "items": items,
    }


class ResidualBlock(nn.Module):
    def __init__(self, channels: int, dilation: int):
        super().__init__()
        pad = 2 * dilation
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, 5, padding=pad, dilation=dilation),
            nn.GroupNorm(8, channels),
            nn.GELU(),
            nn.Conv1d(channels, channels, 5, padding=pad, dilation=dilation),
            nn.GroupNorm(8, channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class TypewellEditor(nn.Module):
    def __init__(self, in_ch: int = 8, hidden: int = 48, delta_limit_norm: float = 0.70):
        super().__init__()
        self.delta_limit = float(delta_limit_norm)
        self.stem = nn.Sequential(nn.Conv1d(in_ch, hidden, 5, padding=2), nn.GroupNorm(8, hidden), nn.GELU())
        self.blocks = nn.ModuleList([ResidualBlock(hidden, d) for d in [1, 2, 4, 8, 16, 32, 64, 128]])
        self.ctx = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, hidden * 2))
        self.head = nn.Sequential(nn.Conv1d(hidden, hidden, 5, padding=2), nn.GELU(), nn.Conv1d(hidden, 1, 1))
        self.affine = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 2))
        self.uncertainty = nn.Sequential(nn.Conv1d(hidden, hidden, 3, padding=1), nn.GELU(), nn.Conv1d(hidden, 1, 1))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)
        nn.init.zeros_(self.affine[-1].weight)
        nn.init.zeros_(self.affine[-1].bias)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.stem(x)
        for block in self.blocks:
            z = block(z)
        pooled = z.mean(dim=-1)
        film = self.ctx(pooled)
        gamma, beta = film.chunk(2, dim=-1)
        z = z * (1.0 + 0.1 * torch.tanh(gamma[:, :, None])) + 0.1 * beta[:, :, None]
        local = self.delta_limit * torch.tanh(self.head(z).squeeze(1))
        aff = self.affine(pooled)
        offset = 0.20 * torch.tanh(aff[:, 0:1])
        gain = 0.20 * torch.tanh(aff[:, 1:2])
        delta = local + offset + gain * x[:, 0]
        log_sigma = self.uncertainty(z).squeeze(1).clamp(-5.0, 3.0)
        return delta, log_sigma


def sample_grid(curve: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    n = curve.shape[-1]
    q = u.clamp(0.0, 1.0) * (n - 1)
    i0 = q.floor().long().clamp(0, n - 1)
    i1 = (i0 + 1).clamp(0, n - 1)
    a = q - i0.float()
    v0 = torch.gather(curve, 1, i0)
    v1 = torch.gather(curve, 1, i1)
    return v0 * (1.0 - a) + v1 * a


def second_diff(x: torch.Tensor) -> torch.Tensor:
    return x[..., 2:] - 2.0 * x[..., 1:-1] + x[..., :-2]


def editor_loss(model: nn.Module, batch: dict[str, Any], device: str) -> tuple[torch.Tensor, dict[str, float]]:
    x = batch["x"].to(device)
    target = batch["target"].to(device)
    tvt_u = batch["tvt_u"].to(device)
    weights = batch["weights"].to(device)
    delta, log_sigma = model(x)
    corrected = x[:, 0] + delta
    pred = sample_grid(corrected, tvt_u)
    err = F.huber_loss(pred, target, reduction="none", delta=0.12)
    data = (err * weights).sum() / weights.sum().clamp_min(1.0)
    smooth = second_diff(delta).pow(2).mean()
    identity = delta.pow(2).mean()


    point_err = (pred.detach() - target).abs()
    sigma = log_sigma.gather(1, (tvt_u.clamp(0, 1) * (log_sigma.shape[1] - 1)).long())
    unc = ((point_err / torch.exp(sigma).clamp_min(1e-3) + sigma) * weights).sum() / weights.sum().clamp_min(1.0)
    loss = data + 0.002 * smooth + 0.0005 * identity + 0.01 * unc
    return loss, {"loss": float(loss.detach()), "data": float(data.detach()), "smooth": float(smooth.detach())}


@torch.no_grad()
def evaluate(model: nn.Module, ds: EditorDataset, device: str, center: float, scale: float,
             out_dir: Path, split_name: str) -> dict[str, Any]:
    model.eval()
    rows = []
    out_dir.mkdir(parents=True, exist_ok=True)
    for item in ds.items:
        x = torch.from_numpy(item["x"])[None].to(device)
        delta, _ = model(x)
        new_norm = (x[:, 0] + delta)[0].cpu().numpy()
        new_grid = new_norm * scale + center
        orig = item["original_gr"]
        true_tvt = item["tvt_u"]
        base_norm = sample_grid(torch.from_numpy(((orig - center) / scale)[None]).to(device),
                                torch.from_numpy(true_tvt[None]).to(device))[0].cpu().numpy()
        pred_norm = sample_grid(torch.from_numpy(new_norm[None]).to(device),
                                torch.from_numpy(true_tvt[None]).to(device))[0].cpu().numpy()
        y = item["target"] * scale + center
        base = base_norm * scale + center
        pred = pred_norm * scale + center
        w = item["weights"]
        post = w > 0.5
        prefix = (w > 0) & ~post
        valid = item["valid"] > 0

        def rm(a, b, m):
            m = m & valid
            return float(np.sqrt(np.mean((a[m] - b[m]) ** 2))) if m.any() else float("nan")

        for part, m in (("prefix", prefix), ("postfix", post), ("full", w > 0)):
            rows.append({"wid": item["wid"][0], "split": split_name, "part": part,
                         "baseline_rmse": rm(y, base, m), "corrected_rmse": rm(y, pred, m),
                         "n": int((m & valid).sum())})

        nt = interp_np(item["original_t"], item["grid_u"], new_grid)
        pd.DataFrame({"TVT": item["original_t"], "GR": nt}).to_csv(
            out_dir / f"{item['wid'][0]}__corrected_typewell.csv", index=False)
    frame = pd.DataFrame(rows)
    frame.to_csv(out_dir / f"{split_name}_per_well.csv", index=False)
    summary = {}
    for part in ["prefix", "postfix", "full"]:
        q = frame[frame.part == part]
        for col in ["baseline_rmse", "corrected_rmse"]:
            summary[f"{part}_{col}_mean_well"] = float(q[col].mean())
            summary[f"{part}_{col}_pooled"] = float(np.sqrt(np.average(q[col] ** 2, weights=q.n)))
        summary[f"{part}_delta_mean_well"] = float((q.corrected_rmse - q.baseline_rmse).mean())
    return summary


def train_fold(fold: int, args: argparse.Namespace, wells: dict[str, dict], folds: dict[str, int], device: str) -> dict:
    out = Path(args.out_dir) / f"fold{fold}"
    out.mkdir(parents=True, exist_ok=True)
    train_ids = [w for w in sorted(wells) if folds[w] != fold]
    val_ids = [w for w in sorted(wells) if folds[w] == fold]
    center, scale = robust_stats(wells, train_ids)
    train_ds = EditorDataset(wells, train_ids, args.grid, center, scale)
    val_ds = EditorDataset(wells, val_ids, args.grid, center, scale)
    loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=args.workers,
                        collate_fn=collate, drop_last=False, pin_memory=(device == "cuda"))
    model = TypewellEditor().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=2e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, args.epochs), eta_min=1e-5)
    best = float("inf")
    best_state = None
    log_path = out / "train.log"
    with log_path.open("w", encoding="utf-8") as log:
        def emit(s: str):
            print(s, flush=True); log.write(s + "\n"); log.flush()
        emit(f"fold={fold} train_wells={len(train_ds)} val_wells={len(val_ds)} grid={args.grid} center={center:.4f} scale={scale:.4f} device={device}")
        for ep in range(args.epochs):
            model.train(); t0 = time.time(); total = 0.0; n_batches = 0
            for batch in loader:
                opt.zero_grad(set_to_none=True)
                loss, parts = editor_loss(model, batch, device)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"non-finite loss fold={fold} epoch={ep}")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                total += float(loss.detach()); n_batches += 1
            sched.step()
            if ep % args.val_every == 0 or ep == args.epochs - 1:
                metrics = evaluate(model, val_ds, device, center, scale, out / "corrected_typewell", "val")
                score = metrics["postfix_corrected_rmse_pooled"]
                mark = " *best*" if score < best else ""
                if score < best:
                    best = score; best_state = copy.deepcopy(model.state_dict())
                    torch.save({"model": best_state, "center": center, "scale": scale,
                                "grid": args.grid, "epoch": ep}, out / "best.pt")
                emit(f"epoch={ep:03d} train_loss={total/max(n_batches,1):.6f} val_postfix_pooled={score:.6f} val_full_pooled={metrics['full_corrected_rmse_pooled']:.6f} sec={time.time()-t0:.1f}{mark}")
        if best_state is not None:
            model.load_state_dict(best_state)
        final = evaluate(model, val_ds, device, center, scale, out / "best_corrected_typewell", "val_best")
        (out / "summary.json").write_text(json.dumps({"fold": fold, "center": center, "scale": scale,
            "best_epoch_metric": best, "metrics": final}, indent=2), encoding="utf-8")
        return {"fold": fold, **final, "best_postfix_corrected": best}


def main() -> None:
    args = parse_args()
    seed_all(args.seed)
    raw_root = Path(args.raw_root)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    wids = list_wells(raw_root)
    folds = make_folds(raw_root, wids)
    wells = {}
    for i, wid in enumerate(wids, 1):
        try:
            wells[wid] = load_well(raw_root, wid)
        except Exception as e:
            print(f"skip {wid}: {e}", flush=True)
        if i % 100 == 0:
            print(f"loaded {i}/{len(wids)}", flush=True)
    if args.fold is not None:
        result = train_fold(args.fold, args, wells, folds, device)
        print(json.dumps(result, indent=2), flush=True)
        return
    results = []
    for fold in range(5):
        results.append(train_fold(fold, args, wells, folds, device))
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()
