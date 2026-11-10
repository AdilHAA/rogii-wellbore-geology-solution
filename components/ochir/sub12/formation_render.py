#!/usr/bin/env python3
# Ochir Dorzhiev
"""Materialize candidate-conditioned full-well tensors for an energy CNN.

Every candidate is rendered through the same function.  Candidate source/family
is never exposed to the model; only inference-visible agreement and physics
channels are stored.  Hidden TVT is retained solely as the candidate-quality
training label and for full-tail evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from tqdm import tqdm


HORIZONTAL_SUFFIX = "__horizontal_well.csv"
TYPEWELL_SUFFIX = "__typewell.csv"


def typewell_hash(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def robust_scale(values: np.ndarray) -> float:
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    if len(values) < 8:
        return 20.0
    scale = 1.4826 * np.median(np.abs(values - np.median(values)))
    return float(max(scale, 5.0))


def affine_calibration(reference: np.ndarray, observed: np.ndarray) -> tuple[float, float]:
    valid = np.isfinite(reference) & np.isfinite(observed)
    if valid.sum() < 20 or np.std(reference[valid]) < 1e-6:
        return 1.0, 0.0
    x, y = reference[valid], observed[valid]
    for _ in range(3):
        gain, bias = np.polyfit(x, y, 1)
        residual = y - (gain * x + bias)
        keep = np.abs(residual - np.median(residual)) <= 3.0 * robust_scale(residual)
        x, y = x[keep], y[keep]
        if len(x) < 20:
            break
    if len(x) < 8 or np.std(x) < 1e-6:
        return 1.0, float(np.median(y) - np.median(x))
    gain, bias = np.polyfit(x, y, 1)
    return float(np.clip(gain, 0.25, 4.0)), float(np.clip(bias, -200.0, 200.0))


def deduplicate(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    frame = pd.DataFrame({"x": x, "y": y}).replace([np.inf, -np.inf], np.nan).dropna()
    frame = frame.groupby("x", as_index=False).y.median().sort_values("x")
    return frame.x.to_numpy(float), frame.y.to_numpy(float)


def interp_index(values: np.ndarray, positions: np.ndarray) -> np.ndarray:
    index = np.arange(len(values), dtype=float)
    valid = np.isfinite(values)
    if valid.sum() >= 2:
        return np.interp(positions, index[valid], values[valid]).astype(np.float32)
    fill = float(values[valid][0]) if valid.any() else 0.0
    return np.full(len(positions), fill, dtype=np.float32)


def render_well(
    root: Path,
    well: str,
    rows: np.ndarray,
    candidates: np.ndarray,
    width: int,
    evidence_masks: np.ndarray | None = None,
) -> np.ndarray:
    horizontal = pd.read_csv(root / f"{well}{HORIZONTAL_SUFFIX}")
    typewell = pd.read_csv(root / f"{well}{TYPEWELL_SUFFIX}")
    type_tvt, type_gr = deduplicate(
        pd.to_numeric(typewell.TVT, errors="coerce").to_numpy(float),
        pd.to_numeric(typewell.GR, errors="coerce").to_numpy(float),
    )
    h_gr_all = pd.to_numeric(horizontal.GR, errors="coerce").to_numpy(float)
    h_z_all = pd.to_numeric(horizontal.Z, errors="coerce").to_numpy(float)
    h_md_all = pd.to_numeric(horizontal.MD, errors="coerce").to_numpy(float)
    tvt_input = pd.to_numeric(horizontal.TVT_input, errors="coerce").to_numpy(float)
    known = np.isfinite(tvt_input) & np.isfinite(h_gr_all)
    prefix_expected = np.interp(tvt_input[known], type_tvt, type_gr)
    gain, bias = affine_calibration(prefix_expected, h_gr_all[known])
    center = float(np.nanmedian(h_gr_all[known])) if known.any() else float(np.nanmedian(type_gr))
    scale = robust_scale(h_gr_all[known] if known.any() else type_gr)

    observed = h_gr_all[rows]
    z = h_z_all[rows]
    md = h_md_all[rows]
    positions = np.linspace(0.0, len(rows) - 1.0, width)
    observed_r = interp_index(observed, positions)
    z_r = interp_index(z, positions)
    md_r = interp_index(md, positions)
    observed_valid = np.isfinite(observed).astype(float)
    valid_r = np.interp(positions, np.arange(len(rows)), observed_valid)
    valid_r = (valid_r > 0.75).astype(np.float32)
    observed_z = np.clip((observed_r - center) / scale, -6.0, 6.0)
    dz_dmd = np.gradient(z_r) / np.maximum(np.gradient(md_r), 1e-3)

    rendered = np.empty((len(candidates), 8, width), dtype=np.float32)
    flat = candidates[0]
    for index, candidate in enumerate(candidates):
        candidate_valid_r = valid_r
        if evidence_masks is not None:
            mask = np.asarray(evidence_masks[index], dtype=float)
            if mask.shape != observed.shape:
                raise ValueError(
                    f"evidence mask {index} has shape {mask.shape}, expected {observed.shape}"
                )
            mask_r = np.interp(positions, np.arange(len(rows)), mask)
            candidate_valid_r = valid_r * (mask_r > 0.75).astype(np.float32)
        candidate_r = interp_index(candidate, positions)
        expected = gain * np.interp(candidate_r, type_tvt, type_gr) + bias
        in_range = ((candidate_r >= type_tvt[0]) & (candidate_r <= type_tvt[-1])).astype(np.float32)
        evidence_valid = candidate_valid_r * in_range
        expected_z = np.clip((expected - center) / scale, -6.0, 6.0)
        residual = np.clip(observed_z - expected_z, -6.0, 6.0) * evidence_valid
        dtvt_dmd = np.gradient(candidate_r) / np.maximum(np.gradient(md_r), 1e-3)
        du_dmd = np.gradient(candidate_r + z_r) / np.maximum(np.gradient(md_r), 1e-3)
        rendered[index] = np.stack(
            [
                observed_z * valid_r,
                expected_z * evidence_valid,
                residual,
                evidence_valid,
                np.clip((candidate_r - interp_index(flat, positions)) / 50.0, -4.0, 4.0),
                np.clip(100.0 * dtvt_dmd, -5.0, 5.0),
                np.clip(100.0 * du_dmd, -5.0, 5.0),
                np.clip(100.0 * dz_dmd, -5.0, 5.0),
            ],
            axis=0,
        )
    return rendered.astype(np.float16)


def load_stride(path: Path) -> pd.DataFrame:
    pieces = []
    for fold in range(4):
        file = path / f"fold{fold}_predictions.parquet"
        part = pd.read_parquet(file, columns=["id", "pred_stride"])
        pieces.append(part)
    return pd.concat(pieces, ignore_index=True).drop_duplicates("id")


def run(args: argparse.Namespace) -> None:
    root = Path(args.train_dir)
    output = Path(args.out_dir)
    output.mkdir(parents=True, exist_ok=True)
    print("Loading per-well path candidates", flush=True)
    source = pd.read_pickle(args.feature_cache)
    candidate_columns = [
        "well", "id", "last_known_tvt", "target", "pf_ancc", "pf_z",
        "beam_cons_d", "beam_loose_d", "beam_vcons_d", "beam_sm5_d",
        "beam_vloose_d", "beam_mid_d", "beam_stiff_d", "sc8_d", "sc15_d",
        "sc25_d", "sc_cons_d", "sc_ens_d", "hyb_d", "sig_mean_d",
    ]
    frame = source[candidate_columns].copy()
    del source
    likelihood = pd.read_pickle(args.likpf_cache)
    frame = frame.merge(likelihood, on="id", how="left", validate="one_to_one")
    del likelihood
    frame["row"] = frame.id.str.rsplit("_", n=1).str[-1].astype(np.int32)
    stride = load_stride(Path(args.stride_dir))
    frame = frame.merge(stride, on="id", how="left", validate="one_to_one")
    del stride
    analog = pd.read_parquet(
        args.analog_consensus,
        columns=["well", "row", "mean1", "mean2", "mean3", "soft3"],
    )
    analog["analog1"] = analog.mean1
    analog["analog2"] = 2.0 * analog.mean2 - analog.mean1
    analog["analog3"] = 3.0 * analog.mean3 - 2.0 * analog.mean2
    frame = frame.merge(
        analog[["well", "row", "analog1", "analog2", "analog3", "soft3"]],
        on=["well", "row"], how="left", validate="one_to_one",
    )
    del analog

    ids = sorted(frame.well.unique())
    hashes = np.asarray([typewell_hash(root / f"{well}{TYPEWELL_SUFFIX}") for well in ids])
    fold_by_well = {}
    for fold, (_, valid_index) in enumerate(GroupKFold(args.folds).split(ids, groups=hashes)):
        fold_by_well.update({ids[index]: fold for index in valid_index})

    names = [
        "flat", "pf_ancc", "pf_z", "beam_cons", "beam_loose", "beam_vcons",
        "beam_sm5", "beam_vloose", "beam_mid", "beam_stiff", "sc8", "sc15",
        "sc25", "sc_cons", "sc_ens", "hyb", "sig_mean", "likpf3", "likpf5",
        "likpf8", "likpf12", "likpf_mean", "stride", "analog1", "analog2",
        "analog3", "analog_soft3",
    ]
    delta_columns = [
        "beam_cons_d", "beam_loose_d", "beam_vcons_d", "beam_sm5_d",
        "beam_vloose_d", "beam_mid_d", "beam_stiff_d", "sc8_d", "sc15_d",
        "sc25_d", "sc_cons_d", "sc_ens_d", "hyb_d", "sig_mean_d",
    ]
    absolute_columns = [
        "likpf_scale_3", "likpf_scale_5", "likpf_scale_8", "likpf_scale_12",
        "likpf_mean", "pred_stride", "analog1", "analog2", "analog3", "soft3",
    ]
    quality_rows = []
    for well, group in tqdm(frame.groupby("well", sort=False), total=len(ids), desc="render wells"):
        group = group.sort_values("row").reset_index(drop=True)
        last = group.last_known_tvt.to_numpy(np.float32)
        flat = last.copy()
        paths = [flat, group.pf_ancc.to_numpy(np.float32), group.pf_z.to_numpy(np.float32)]
        paths.extend((last + group[column].to_numpy(np.float32)) for column in delta_columns)
        paths.extend(group[column].to_numpy(np.float32) for column in absolute_columns)
        candidates = np.stack(paths).astype(np.float32)
        candidates[~np.isfinite(candidates)] = np.broadcast_to(flat, candidates.shape)[~np.isfinite(candidates)]
        truth = last + group.target.to_numpy(np.float32)
        quality = np.sqrt(np.mean((candidates - truth[None]) ** 2, axis=1)).astype(np.float32)
        rendered = render_well(root, well, group.row.to_numpy(np.int32), candidates, args.width)
        np.savez_compressed(
            output / f"{well}.npz",
            x=rendered,
            candidates=candidates.astype(np.float32),
            truth=truth.astype(np.float32),
            rows=group.row.to_numpy(np.int32),
            quality=quality,
            fold=np.int8(fold_by_well[well]),
        )
        quality_rows.append(
            {
                "well": well,
                "fold": fold_by_well[well],
                "n_rows": len(group),
                "flat_rmse": quality[0],
                "oracle_rmse": float(np.min(quality)),
                "oracle_name": names[int(np.argmin(quality))],
            }
        )
    quality_table = pd.DataFrame(quality_rows)
    quality_table.to_csv(output / "candidate_quality.csv", index=False)
    with (output / "metadata.json").open("w") as stream:
        json.dump({"candidate_names": names, "channels": [
            "horizontal_gr", "candidate_typewell_gr", "gr_residual", "gr_valid",
            "candidate_minus_flat", "dtvt_dmd", "du_dmd", "dz_dmd",
        ], "width": args.width}, stream, indent=2)
    pooled_flat = np.sqrt(np.average(quality_table.flat_rmse**2, weights=quality_table.n_rows))
    pooled_oracle = np.sqrt(np.average(quality_table.oracle_rmse**2, weights=quality_table.n_rows))
    print(f"Materialized {len(quality_table)} wells | flat={pooled_flat:.4f} oracle={pooled_oracle:.4f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", default="rogii-wellbore-geology-prediction/train")
    parser.add_argument(
        "--feature-cache",
        default="single_lgbm_cache/v1_features_train_n773_2897e0dfb285_ctx2897e0dfb285_p200_s16_sc3.0-5.0-8.0-12.0_istrain1.pkl",
    )
    parser.add_argument(
        "--likpf-cache",
        default="single_lgbm_cache/v1_likpf_train_n773_2897e0dfb285_ctxnoctx_p200_s16_sc3.0-5.0-8.0-12.0.pkl",
    )
    parser.add_argument(
        "--stride-dir",
        default="stride_backbone_v2_audit/full_oof_blend_cons_rawspatial",
    )
    parser.add_argument("--analog-consensus", default="formation_analog_ranker_full/consensus_paths.parquet")
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--out-dir", default="whole_path_energy_data")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
