# Andrey Serzhenko
"""Leakage-safe inference for the saved EXP-256 V2 typewell editor.

The input contract is deliberately smaller than the training target:
original typewell + finite TVT_input/GR prefix only. True TVT and postfix GR
are not read or required. The default mode averages the five saved fold
models after converting each output back to native TVT/GR coordinates.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import train_typewell_editor as exp


def robust_affine(w: dict) -> tuple[float, float]:
    known = np.isfinite(w["tvt_input"]) & (w["h_gr_present"] > 0)
    if known.sum() < 8:
        return 0.0, 1.0
    x = np.interp(w["tvt_input"][known], w["t_tvt"], w["t_gr"])
    y = w["h_gr"][known]
    keep = np.isfinite(x) & np.isfinite(y)
    x, y = x[keep].astype(np.float64), y[keep].astype(np.float64)
    if len(x) < 8 or np.var(x) < 1e-8:
        return 0.0, 1.0
    a, b = 0.0, 1.0
    weights = np.ones(len(x), np.float64)
    for _ in range(5):
        X = np.column_stack([np.ones(len(x)), x])
        sw = np.sqrt(weights)
        a, b = np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)[0]
        r = y - (a + b * x)
        s = max(1.0, 1.4826 * np.median(np.abs(r - np.median(r))))
        weights = np.minimum(1.0, (2.5 * s) / np.maximum(np.abs(r), 1e-6))
    return float(np.clip(a, -60.0, 60.0)), float(np.clip(b, 0.70, 1.30))


def load_inference_well(raw_root: Path, wid: str) -> dict:
    base = raw_root / "competition" / "raw" / "train" / wid
    h = pd.read_csv(str(base) + "__horizontal_well.csv")
    t = pd.read_csv(str(base) + "__typewell.csv")
    if "TVT_input" not in h or "GR" not in h:
        raise ValueError("horizontal file must contain TVT_input and GR")
    if "TVT" not in t or "GR" not in t:
        raise ValueError("typewell file must contain TVT and GR")
    tvt_input = h["TVT_input"].to_numpy(np.float32)
    h_raw = h["GR"].to_numpy(np.float32)
    h_present = np.isfinite(h_raw)
    t_tvt = t["TVT"].to_numpy(np.float32)
    t_gr = exp.robust_fill(t["GR"].to_numpy(np.float32))


    return {
        "wid": wid,
        "h_gr": exp.robust_fill(h_raw),
        "h_gr_present": h_present.astype(np.float32),
        "tvt_true": np.full(len(h), float(t_tvt[0]), np.float32),
        "tvt_input": tvt_input,
        "t_tvt": t_tvt,
        "t_gr": t_gr,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--raw-root", required=True)
    p.add_argument("--weights-root", required=True)
    p.add_argument("--well-id", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--fold", type=int, default=None)
    p.add_argument("--device", default=None)
    return p.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    raw_root = Path(args.raw_root)
    weights_root = Path(args.weights_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    w = load_inference_well(raw_root, args.well_id)
    fold_ids = [args.fold] if args.fold is not None else list(range(5))
    native_t = w["t_tvt"]
    curves = []
    for fold in fold_ids:
        ckpt_path = weights_root / f"fold{fold}" / "best.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(ckpt_path)
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        center, scale, grid = float(ckpt["center"]), float(ckpt["scale"]), int(ckpt["grid"])
        a, b = robust_affine(w)
        w2 = dict(w)
        w2["t_gr"] = (a + b * w["t_gr"]).astype(np.float32)
        item = exp.build_item(w2, grid, center, scale)
        model = exp.TypewellEditor().to(device)
        model.load_state_dict(ckpt["model"])
        model.eval()
        x = torch.from_numpy(item["x"])[None].to(device)
        delta, _ = model(x)
        new_norm = (x[:, 0] + delta)[0].cpu().numpy()
        t0, t1 = float(native_t[0]), float(native_t[-1])
        u_native = np.clip((native_t - t0) / max(t1 - t0, 1e-3), 0.0, 1.0)
        curve = exp.interp_np(u_native, item["grid_u"], new_norm * scale + center)
        curves.append(curve)
        pd.DataFrame({"TVT": native_t, "GR": curve}).to_csv(
            out_dir / f"{args.well_id}__corrected_typewell_fold{fold}.csv", index=False)
    ensemble = np.mean(np.stack(curves, axis=0), axis=0)
    out_path = out_dir / f"{args.well_id}__corrected_typewell_ensemble.csv"
    pd.DataFrame({"TVT": native_t, "GR": ensemble}).to_csv(out_path, index=False)
    print(f"saved {out_path} folds={fold_ids} device={device}")


if __name__ == "__main__":
    main()
