#!/usr/bin/env python3
# Ochir Dorzhiev
"""Source-blind mixture of frozen physical path experts.

The four input paths are generated out-of-fold elsewhere.  A shared sequence
encoder scores every path from inference-visible evidence only:

* observed post-PS horizontal GR;
* typewell GR rendered through several tool-response kernels;
* calibrated and uncalibrated residuals;
* all-prefix and direction-conditioned self-reference residuals;
* relative path and trajectory geometry.

The scorer is permutation equivariant and receives no candidate-family label.
Its scores make a soft posterior blend.  A second head may add only an
anchored two-parameter low-frequency correction.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import minimize
from scipy.signal import fftconvolve
from torch import nn
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parent
EXPERT_COLUMNS = ("pred_a", "pred_s", "pred_o", "pred_d")
EXPERT_NAMES = ("a", "s", "o", "d")
START_WEIGHTS = np.asarray((0.52, 0.15, 1e-5, 0.33), dtype=np.float32)
RESPONSE_SPECS = (
    ("point", 0.0),
    ("gaussian", 0.5),
    ("cusp", 0.75),
    ("box", 2.0),
)
CACHE_VERSION = "v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--components",
        type=Path,
        default=ROOT / "stable_blend_portfolio" / "v6_components_oof.parquet",
    )
    parser.add_argument(
        "--train-dir",
        type=Path,
        default=ROOT / "rogii-wellbore-geology-prediction" / "train",
    )
    parser.add_argument(
        "--feature-cache",
        type=Path,
        default=ROOT / "source_blind_physical_moe_cache",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "source_blind_physical_moe",
    )
    parser.add_argument("--points", type=int, default=256)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--embedding", type=int, default=32)
    parser.add_argument("--blocks", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument(
        "--weight-mode",
        choices=("global", "local"),
        default="global",
    )
    parser.add_argument(
        "--local-smoothing",
        type=int,
        default=31,
        help="Odd smoothing width for local candidate logits.",
    )
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument(
        "--prior-floor",
        type=float,
        default=0.0,
        help="Minimum probability assigned to every expert before evidence.",
    )
    parser.add_argument("--max-logit-delta", type=float, default=1.5)
    parser.add_argument("--correction-scale-ft", type=float, default=20.0)
    parser.add_argument("--disable-correction", action="store_true")
    parser.add_argument("--rank-weight", type=float, default=0.25)
    parser.add_argument("--rank-temperature", type=float, default=0.25)
    parser.add_argument("--anchor-weight", type=float, default=0.25)
    parser.add_argument("--correction-weight", type=float, default=0.05)
    parser.add_argument("--source-dropout", type=float, default=0.10)
    parser.add_argument(
        "--folds",
        default="all",
        help="Comma-separated heldout folds or 'all'.",
    )
    parser.add_argument(
        "--checkpoint-mode",
        choices=("final", "best"),
        default="final",
    )
    parser.add_argument(
        "--production",
        action="store_true",
        help="Train one final model on every available well.",
    )
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--max-wells", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def robust_scale(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 8:
        return 20.0
    center = float(np.median(values))
    scale = 1.4826 * float(np.median(np.abs(values - center)))
    return max(scale, 8.0)


def robust_affine(reference: np.ndarray, observed: np.ndarray) -> tuple[float, float]:
    valid = np.isfinite(reference) & np.isfinite(observed)
    x = np.asarray(reference, dtype=float)[valid]
    y = np.asarray(observed, dtype=float)[valid]
    if len(x) < 20 or np.std(x) < 1e-6:
        offset = float(np.median(y - x)) if len(x) else 0.0
        return 1.0, float(np.clip(offset, -100.0, 100.0))
    design = np.column_stack((x, np.ones(len(x))))
    beta = np.linalg.lstsq(design, y, rcond=None)[0]
    for _ in range(5):
        residual = y - design @ beta
        center = float(np.median(residual))
        scale = 1.4826 * float(np.median(np.abs(residual - center))) + 1e-6
        weight = np.minimum(1.0, 1.5 * scale / (np.abs(residual - center) + 1e-9))
        root = np.sqrt(weight)
        beta = np.linalg.lstsq(design * root[:, None], y * root, rcond=None)[0]
    gain = float(np.clip(beta[0], 0.5, 2.0))
    offset = float(np.clip(beta[1], -100.0, 100.0))
    return gain, offset


def regular_typewell(path: Path) -> tuple[np.ndarray, np.ndarray]:
    frame = pd.read_csv(path, usecols=["TVT", "GR"]).dropna()
    frame = frame.groupby("TVT", as_index=False)["GR"].median().sort_values("TVT")
    x = frame["TVT"].to_numpy(float)
    y = frame["GR"].to_numpy(float)
    step = float(np.median(np.diff(x)))
    step = float(np.clip(step, 0.1, 1.0))
    grid = np.arange(x[0], x[-1] + 0.5 * step, step)
    return grid, np.interp(grid, x, y)


def response_curve(values: np.ndarray, step: float, family: str, width: float) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if family == "point" or width <= 0.0:
        return values.copy()
    if family == "gaussian":
        return gaussian_filter1d(values, width / max(step, 1e-6), mode="nearest")
    if family == "cusp":
        radius = max(2, int(math.ceil(8.0 * width / max(step, 1e-6))))
        x = np.arange(-radius, radius + 1, dtype=float) * step
        kernel = np.exp(-np.abs(x) / max(width, 1e-6))
    elif family == "box":
        radius = max(1, int(math.ceil(0.5 * width / max(step, 1e-6))))
        kernel = np.ones(2 * radius + 1, dtype=float)
    else:
        raise ValueError(f"Unknown response family: {family}")
    kernel /= kernel.sum()
    padded = np.pad(values, len(kernel) // 2, mode="edge")
    filtered = fftconvolve(padded, kernel, mode="same")
    radius = len(kernel) // 2
    return filtered[radius:-radius]


def bin_observed(md: np.ndarray, gr: np.ndarray, sample_md: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    edges = np.empty(len(sample_md) + 1, dtype=float)
    edges[1:-1] = 0.5 * (sample_md[:-1] + sample_md[1:])
    edges[0] = sample_md[0] - 0.5 * (sample_md[1] - sample_md[0])
    edges[-1] = sample_md[-1] + 0.5 * (sample_md[-1] - sample_md[-2])
    positions = np.searchsorted(md, edges)
    output = np.zeros(len(sample_md), dtype=np.float32)
    valid = np.zeros(len(sample_md), dtype=np.float32)
    for index in range(len(sample_md)):
        chunk = gr[positions[index] : positions[index + 1]]
        finite = chunk[np.isfinite(chunk)]
        if len(finite):
            output[index] = float(np.median(finite))
            valid[index] = 1.0
    return output, valid


def rolling_direction(md: np.ndarray, tvt: np.ndarray) -> np.ndarray:
    rate = np.gradient(tvt, md)
    rate = (
        pd.Series(rate)
        .rolling(21, center=True, min_periods=1)
        .median()
        .to_numpy(float)
    )
    direction = np.ones(len(rate), dtype=np.int8)
    direction[rate < -0.005] = 0
    direction[rate > 0.005] = 2
    return direction


def reference_lookup(
    query: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    radius: float = 1.0,
    neighbors: int = 7,
) -> tuple[np.ndarray, np.ndarray]:
    valid = np.isfinite(x) & np.isfinite(y)
    x = np.asarray(x, dtype=float)[valid]
    y = np.asarray(y, dtype=float)[valid]
    output = np.zeros(len(query), dtype=np.float32)
    available = np.zeros(len(query), dtype=np.float32)
    if len(x) == 0:
        return output, available
    order = np.argsort(x)
    x, y = x[order], y[order]
    for index, value in enumerate(query):
        position = int(np.searchsorted(x, value))
        lo = max(0, position - 3 * neighbors)
        hi = min(len(x), position + 3 * neighbors + 1)
        candidate = np.arange(lo, hi)
        if len(candidate) == 0:
            continue
        distance = np.abs(x[candidate] - value)
        candidate = candidate[np.argsort(distance)[:neighbors]]
        candidate = candidate[np.abs(x[candidate] - value) <= radius]
        if len(candidate):
            output[index] = float(np.median(y[candidate]))
            available[index] = 1.0
    return output, available


def resample(values: np.ndarray, size: int) -> np.ndarray:
    old_x = np.linspace(0.0, 1.0, len(values), dtype=float)
    new_x = np.linspace(0.0, 1.0, size, dtype=float)
    return np.interp(new_x, old_x, values).astype(np.float32)


def finite_feature(values: np.ndarray, valid: np.ndarray | None = None) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if valid is None:
        valid = np.isfinite(values)
    else:
        valid = np.asarray(valid, dtype=bool) & np.isfinite(values)
    return np.where(valid, np.clip(values, -6.0, 6.0), 0.0).astype(np.float32)


def materialize_features(
    well: str,
    group: pd.DataFrame,
    train_dir: Path,
    points: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    horizontal = pd.read_csv(
        train_dir / f"{well}__horizontal_well.csv",
        usecols=["MD", "Z", "GR", "TVT_input"],
    ).sort_values("MD")
    md = horizontal["MD"].to_numpy(float)
    z = horizontal["Z"].to_numpy(float)
    gr = horizontal["GR"].to_numpy(float)
    tvt_input = horizontal["TVT_input"].to_numpy(float)
    known = np.flatnonzero(np.isfinite(tvt_input))
    prefix_end = int(known[-1])
    suffix_md = md[prefix_end + 1 :]
    if len(suffix_md) != len(group):
        raise ValueError(f"{well}: component/raw suffix mismatch")

    sample_md = np.linspace(suffix_md[0], suffix_md[-1], points)
    sample_z = np.interp(sample_md, md, z)
    experts_dense = group[list(EXPERT_COLUMNS)].to_numpy(np.float32).T
    experts = np.stack(
        [np.interp(sample_md, suffix_md, path) for path in experts_dense],
        axis=0,
    ).astype(np.float32)
    truth = np.interp(
        sample_md,
        suffix_md,
        group["true_tvt"].to_numpy(float),
    ).astype(np.float32)

    observed, observed_valid = bin_observed(md, gr, sample_md)
    prefix_gr = gr[: prefix_end + 1]
    prefix_tvt = tvt_input[: prefix_end + 1]
    prefix_md = md[: prefix_end + 1]
    prefix_center_values = prefix_gr[np.isfinite(prefix_gr)]
    observed_center = (
        float(np.median(prefix_center_values))
        if len(prefix_center_values)
        else float(np.nanmedian(gr))
    )
    scale = robust_scale(prefix_center_values)
    observed_norm = finite_feature((observed - observed_center) / scale, observed_valid > 0)

    type_tvt, type_gr = regular_typewell(train_dir / f"{well}__typewell.csv")
    type_step = float(np.median(np.diff(type_tvt)))
    prefix_expected = np.interp(
        prefix_tvt,
        type_tvt,
        type_gr,
        left=np.nan,
        right=np.nan,
    )
    gain, offset = robust_affine(prefix_expected, prefix_gr)
    responses = {
        name: response_curve(type_gr, type_step, name, width)
        for name, width in RESPONSE_SPECS
    }

    prefix_direction = rolling_direction(prefix_md, prefix_tvt)
    median_path = np.median(experts, axis=0)
    expert_spread = np.std(experts, axis=0)
    position = np.linspace(0.0, 1.0, points, dtype=np.float32)
    ps_tvt = float(tvt_input[prefix_end])
    ps_surface = ps_tvt + float(z[prefix_end])
    feature_names: list[str] = []
    candidate_features: list[np.ndarray] = []

    for candidate in experts:
        channels: list[np.ndarray] = []

        def append(name: str, values: np.ndarray) -> None:
            if len(candidate_features) == 0:
                feature_names.append(name)
            channels.append(np.asarray(values, dtype=np.float32))

        type_valid = (candidate >= type_tvt[0]) & (candidate <= type_tvt[-1])
        append("observed_gr", observed_norm)
        append("observed_valid", observed_valid)
        append("typewell_valid", type_valid.astype(np.float32))

        for response_name, _ in RESPONSE_SPECS:
            expected = np.interp(
                candidate,
                type_tvt,
                responses[response_name],
                left=np.nan,
                right=np.nan,
            )
            calibrated = gain * expected + offset
            valid = type_valid & np.isfinite(expected) & (observed_valid > 0)
            append(
                f"{response_name}_expected",
                finite_feature((expected - observed_center) / scale, type_valid),
            )
            append(
                f"{response_name}_expected_calibrated",
                finite_feature((calibrated - observed_center) / scale, type_valid),
            )
            append(
                f"{response_name}_residual",
                finite_feature((expected - observed) / scale, valid),
            )
            append(
                f"{response_name}_residual_calibrated",
                finite_feature((calibrated - observed) / scale, valid),
            )
            derivative_residual = np.gradient(calibrated) - np.gradient(observed)
            append(
                f"{response_name}_derivative_residual",
                finite_feature(derivative_residual / scale, valid),
            )

        self_all, self_all_valid = reference_lookup(
            candidate,
            prefix_tvt,
            prefix_gr,
        )
        candidate_direction = rolling_direction(sample_md, candidate)
        self_directional = np.zeros(points, dtype=np.float32)
        self_directional_valid = np.zeros(points, dtype=np.float32)
        for direction in range(3):
            query_mask = candidate_direction == direction
            if not np.any(query_mask):
                continue
            source_mask = prefix_direction == direction
            values, available = reference_lookup(
                candidate[query_mask],
                prefix_tvt[source_mask],
                prefix_gr[source_mask],
            )
            self_directional[query_mask] = values
            self_directional_valid[query_mask] = available

        all_joint_valid = (self_all_valid > 0) & (observed_valid > 0)
        directional_joint_valid = (
            (self_directional_valid > 0) & (observed_valid > 0)
        )
        append(
            "self_all_expected",
            finite_feature(
                (self_all - observed_center) / scale,
                self_all_valid > 0,
            ),
        )
        append(
            "self_all_residual",
            finite_feature((self_all - observed) / scale, all_joint_valid),
        )
        append("self_all_valid", self_all_valid)
        append(
            "self_directional_expected",
            finite_feature(
                (self_directional - observed_center) / scale,
                self_directional_valid > 0,
            ),
        )
        append(
            "self_directional_residual",
            finite_feature(
                (self_directional - observed) / scale,
                directional_joint_valid,
            ),
        )
        append("self_directional_valid", self_directional_valid)

        candidate_surface = candidate + sample_z
        tvt_rate = np.gradient(candidate, sample_md) * 100.0
        surface_rate = np.gradient(candidate_surface, sample_md) * 100.0
        curvature = np.gradient(surface_rate, sample_md) * 1000.0
        append("path_tvt_relative", np.clip((candidate - ps_tvt) / 50.0, -6.0, 6.0))
        append(
            "path_surface_relative",
            np.clip((candidate_surface - ps_surface) / 50.0, -6.0, 6.0),
        )
        append("path_tvt_rate_100ft", np.clip(tvt_rate / 5.0, -6.0, 6.0))
        append("path_surface_rate_100ft", np.clip(surface_rate / 5.0, -6.0, 6.0))
        append("path_curvature", np.clip(curvature / 5.0, -6.0, 6.0))
        append("deviation_from_median", np.clip((candidate - median_path) / 20.0, -6.0, 6.0))
        append("expert_spread", np.clip(expert_spread / 20.0, 0.0, 6.0))
        append("position", position)
        candidate_features.append(np.stack(channels, axis=0))

    return (
        np.stack(candidate_features, axis=0).astype(np.float32),
        experts,
        truth,
        feature_names,
    )


@dataclass
class WellItem:
    well: str
    fold: int
    nrows: int
    features: np.ndarray
    experts: np.ndarray
    truth: np.ndarray
    original_experts: np.ndarray
    original_truth: np.ndarray


def load_items(args: argparse.Namespace) -> tuple[list[WellItem], list[str]]:
    frame = pd.read_parquet(
        args.components,
        columns=["well", "fold", "position", "true_tvt", *EXPERT_COLUMNS],
    ).sort_values(["well", "position"])
    args.feature_cache.mkdir(parents=True, exist_ok=True)
    items: list[WellItem] = []
    channel_names: list[str] | None = None
    total = frame["well"].nunique()
    for index, (well, group) in enumerate(frame.groupby("well", sort=True), 1):
        if args.max_wells and len(items) >= args.max_wells:
            break
        cache_path = (
            args.feature_cache
            / f"{well}_p{args.points}_{CACHE_VERSION}.npz"
        )
        if cache_path.exists() and not args.rebuild_cache:
            with np.load(cache_path, allow_pickle=False) as packed:
                features = packed["features"].astype(np.float32)
                experts = packed["experts"].astype(np.float32)
                truth = packed["truth"].astype(np.float32)
                names = [str(value) for value in packed["feature_names"]]
        else:
            features, experts, truth, names = materialize_features(
                str(well),
                group,
                args.train_dir,
                args.points,
            )
            np.savez_compressed(
                cache_path,
                features=features.astype(np.float16),
                experts=experts,
                truth=truth,
                feature_names=np.asarray(names),
            )
        if channel_names is None:
            channel_names = names
        elif channel_names != names:
            raise ValueError(f"{well}: feature channel mismatch")
        items.append(
            WellItem(
                well=str(well),
                fold=int(group["fold"].iloc[0]),
                nrows=len(group),
                features=features,
                experts=experts,
                truth=truth,
                original_experts=group[list(EXPERT_COLUMNS)]
                .to_numpy(np.float32)
                .T,
                original_truth=group["true_tvt"].to_numpy(np.float32),
            )
        )
        if index % 100 == 0 or index == total:
            print(f"loaded features {index}/{total}", flush=True)
    if not items or channel_names is None:
        raise RuntimeError("No wells loaded")
    return items, channel_names


class PhysicalExpertDataset(Dataset):
    def __init__(
        self,
        items: list[WellItem],
        augment: bool,
        source_dropout: float,
    ) -> None:
        self.items = items
        self.augment = augment
        self.source_dropout = source_dropout

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        item = self.items[index]
        order = np.arange(len(EXPERT_COLUMNS))
        if self.augment:
            order = np.random.permutation(order)
        active = np.ones(len(EXPERT_COLUMNS), dtype=np.float32)
        if self.augment and np.random.random() < self.source_dropout:
            active[np.random.randint(0, len(active))] = 0.0
        return {
            "features": torch.from_numpy(item.features[order]),
            "experts": torch.from_numpy(item.experts[order]),
            "truth": torch.from_numpy(item.truth),
            "source_indices": torch.from_numpy(order.astype(np.int64)),
            "active": torch.from_numpy(active[order]),
            "nrows": torch.tensor(float(item.nrows), dtype=torch.float32),
        }


class ResidualBlock(nn.Module):
    def __init__(self, hidden: int, dilation: int, dropout: float) -> None:
        super().__init__()
        padding = 2 * dilation
        self.net = nn.Sequential(
            nn.Conv1d(hidden, hidden, 5, padding=padding, dilation=dilation),
            nn.GroupNorm(4 if hidden % 4 == 0 else 1, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden, hidden, 1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs + self.net(inputs)


class SourceBlindPhysicalMoE(nn.Module):
    def __init__(
        self,
        channels: int,
        hidden: int,
        embedding: int,
        blocks: int,
        dropout: float,
        max_logit_delta: float,
        correction_scale_ft: float,
        base_weights: np.ndarray,
        correction_enabled: bool,
        weight_mode: str,
        local_smoothing: int,
    ) -> None:
        super().__init__()
        self.max_logit_delta = float(max_logit_delta)
        self.correction_scale_ft = float(correction_scale_ft)
        self.correction_enabled = bool(correction_enabled)
        self.weight_mode = weight_mode
        self.local_smoothing = int(local_smoothing)
        if self.local_smoothing < 1 or self.local_smoothing % 2 == 0:
            raise ValueError("--local-smoothing must be a positive odd integer")
        self.stem = nn.Sequential(
            nn.Conv1d(channels, hidden, 7, padding=3),
            nn.GroupNorm(4 if hidden % 4 == 0 else 1, hidden),
            nn.GELU(),
        )
        self.blocks = nn.Sequential(
            *[
                ResidualBlock(hidden, 2**index, dropout)
                for index in range(blocks)
            ]
        )
        self.embedding = nn.Sequential(
            nn.Linear(3 * hidden, embedding),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.score_head = nn.Sequential(
            nn.Linear(4 * embedding, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )
        self.local_score_head: nn.Sequential | None = None
        if self.weight_mode == "local":
            self.local_score_head = nn.Sequential(
                nn.Conv1d(4 * hidden, hidden, 1),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Conv1d(hidden, 1, 1),
            )
        self.correction_head = nn.Sequential(
            nn.Linear(3 * embedding, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 2),
        )
        nn.init.zeros_(self.score_head[-1].weight)
        nn.init.zeros_(self.score_head[-1].bias)
        if self.local_score_head is not None:
            nn.init.zeros_(self.local_score_head[-1].weight)
            nn.init.zeros_(self.local_score_head[-1].bias)
        nn.init.zeros_(self.correction_head[-1].weight)
        nn.init.zeros_(self.correction_head[-1].bias)
        self.register_buffer(
            "base_logits",
            torch.log(torch.from_numpy(base_weights.astype(np.float32))),
        )

    def encode(
        self,
        features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, candidates, channels, points = features.shape
        sequence = self.blocks(
            self.stem(features.reshape(batch * candidates, channels, points))
        )
        pooled = torch.cat(
            (
                torch.mean(sequence, dim=2),
                torch.std(sequence, dim=2, unbiased=False),
                torch.amax(sequence, dim=2),
            ),
            dim=1,
        )
        embedding = self.embedding(pooled).reshape(batch, candidates, -1)
        return sequence.reshape(batch, candidates, -1, points), embedding

    def forward(
        self,
        features: torch.Tensor,
        experts: torch.Tensor,
        source_indices: torch.Tensor,
        active: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        sequence, embedding = self.encode(features)
        if self.weight_mode == "global":
            context = torch.mean(embedding, dim=1, keepdim=True)
            context_expand = context.expand_as(embedding)
            score_input = torch.cat(
                (
                    embedding,
                    context_expand,
                    embedding - context_expand,
                    torch.abs(embedding - context_expand),
                ),
                dim=2,
            )
            delta = self.max_logit_delta * torch.tanh(
                self.score_head(score_input).squeeze(-1)
            )
            prior_logits = self.base_logits[source_indices]
            logits = prior_logits + delta
            logits = torch.where(
                active > 0,
                logits,
                torch.full_like(logits, -30.0),
            )
            weights = torch.softmax(logits, dim=1)
            blend = torch.sum(weights[:, :, None] * experts, dim=1)
            summary_weights = weights
            context = torch.mean(embedding, dim=1, keepdim=True)
        else:
            if self.local_score_head is None:
                raise RuntimeError("Local score head is not initialized")
            batch, candidates, hidden, points = sequence.shape
            local_context = torch.mean(sequence, dim=1, keepdim=True)
            local_context = local_context.expand_as(sequence)
            local_input = torch.cat(
                (
                    sequence,
                    local_context,
                    sequence - local_context,
                    torch.abs(sequence - local_context),
                ),
                dim=2,
            )
            delta = self.local_score_head(
                local_input.reshape(batch * candidates, 4 * hidden, points)
            ).reshape(batch, candidates, points)
            if self.local_smoothing > 1:
                delta = F.avg_pool1d(
                    delta.reshape(batch * candidates, 1, points),
                    kernel_size=self.local_smoothing,
                    stride=1,
                    padding=self.local_smoothing // 2,
                ).reshape(batch, candidates, points)
            delta = self.max_logit_delta * torch.tanh(delta)
            prior_logits = self.base_logits[source_indices][:, :, None]
            logits = prior_logits + delta
            logits = torch.where(
                active[:, :, None] > 0,
                logits,
                torch.full_like(logits, -30.0),
            )
            weights = torch.softmax(logits, dim=1)
            blend = torch.sum(weights * experts, dim=1)
            summary_weights = torch.mean(weights, dim=2)
            context = torch.mean(embedding, dim=1, keepdim=True)

        weighted = torch.sum(summary_weights[:, :, None] * embedding, dim=1)
        mean = context[:, 0]
        std = torch.std(embedding, dim=1, unbiased=False)
        correction_input = torch.cat((weighted, mean, std), dim=1)
        if self.correction_enabled:
            coefficients = self.correction_scale_ft * torch.tanh(
                self.correction_head(correction_input)
            )
        else:
            coefficients = torch.zeros(
                len(features), 2, device=features.device, dtype=features.dtype
            )
        x = torch.linspace(
            0.0, 1.0, experts.shape[2], device=experts.device, dtype=experts.dtype
        )
        correction = (
            coefficients[:, :1] * x[None]
            + coefficients[:, 1:] * (4.0 * x[None] * (1.0 - x[None]))
        )
        prediction = blend + correction
        return prediction, blend, weights, coefficients


def fit_static_weights(items: list[WellItem]) -> np.ndarray:
    gram = np.zeros((len(EXPERT_COLUMNS), len(EXPERT_COLUMNS)), dtype=np.float64)
    cross = np.zeros(len(EXPERT_COLUMNS), dtype=np.float64)
    count = 0
    for item in items:
        experts = item.original_experts.astype(np.float64)
        truth = item.original_truth.astype(np.float64)
        gram += experts @ experts.T
        cross += experts @ truth
        count += item.nrows

    result = minimize(
        lambda weights: float(
            (weights @ gram @ weights - 2.0 * weights @ cross) / count
        ),
        START_WEIGHTS.astype(np.float64),
        jac=lambda weights: 2.0 * (gram @ weights - cross) / count,
        method="SLSQP",
        bounds=[(1e-5, 1.0)] * len(EXPERT_COLUMNS),
        constraints={"type": "eq", "fun": lambda weights: weights.sum() - 1.0},
        options={"maxiter": 200, "ftol": 1e-12},
    )
    if not result.success:
        raise RuntimeError(result.message)
    return result.x.astype(np.float32)


def floor_prior(weights: np.ndarray, floor: float) -> np.ndarray:
    weights = np.asarray(weights, dtype=np.float64)
    if floor < 0.0 or floor * len(weights) >= 1.0:
        raise ValueError("--prior-floor must be in [0, 1 / n_experts)")
    if floor == 0.0:
        return weights.astype(np.float32)
    smoothed = floor + (1.0 - floor * len(weights)) * weights
    return (smoothed / smoothed.sum()).astype(np.float32)


def training_loss(
    model: SourceBlindPhysicalMoE,
    batch: dict[str, torch.Tensor],
    device: torch.device,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, dict[str, float]]:
    values = {key: value.to(device) for key, value in batch.items()}
    prediction, _, weights, coefficients = model(
        values["features"],
        values["experts"],
        values["source_indices"],
        values["active"],
    )
    normalized_error = (prediction - values["truth"]) / 20.0
    per_well_mse = torch.mean(torch.square(normalized_error), dim=1)
    data_loss = torch.sum(per_well_mse * values["nrows"]) / torch.sum(
        values["nrows"]
    )

    candidate_error = torch.square(
        (values["experts"] - values["truth"][:, None, :]) / 20.0
    )
    if weights.ndim == 3:
        target_error = F.avg_pool1d(
            candidate_error,
            kernel_size=args.local_smoothing,
            stride=1,
            padding=args.local_smoothing // 2,
        )
        target_probability = torch.softmax(
            -target_error / max(args.rank_temperature, 1e-6),
            dim=1,
        )
        rank_loss = torch.mean(
            torch.sum(
                -target_probability
                * torch.log(torch.clamp(weights, min=1e-8)),
                dim=1,
            )
        )
        prior = torch.softmax(
            model.base_logits[values["source_indices"]],
            dim=1,
        )[:, :, None]
    else:
        candidate_mse = torch.mean(candidate_error, dim=2)
        target_probability = torch.softmax(
            -candidate_mse / max(args.rank_temperature, 1e-6),
            dim=1,
        )
        rank_loss = torch.mean(
            torch.sum(
                -target_probability
                * torch.log(torch.clamp(weights, min=1e-8)),
                dim=1,
            )
        )
        prior = torch.softmax(
            model.base_logits[values["source_indices"]],
            dim=1,
        )
    anchor_loss = torch.mean(torch.square(weights - prior))
    correction_loss = torch.mean(
        torch.square(coefficients / max(args.correction_scale_ft, 1e-6))
    )
    total = (
        data_loss
        + args.rank_weight * rank_loss
        + args.anchor_weight * anchor_loss
        + args.correction_weight * correction_loss
    )
    return total, {
        "data": float(data_loss.detach().cpu()),
        "rank": float(rank_loss.detach().cpu()),
        "anchor": float(anchor_loss.detach().cpu()),
        "correction": float(correction_loss.detach().cpu()),
    }


@torch.no_grad()
def predict_item(
    model: SourceBlindPhysicalMoE,
    item: WellItem,
    device: torch.device,
    evidence_mode: str = "original",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    features = torch.from_numpy(item.features)[None].to(device)
    experts = torch.from_numpy(item.experts)[None].to(device)
    source_indices = torch.arange(len(EXPERT_COLUMNS), device=device)[None]
    active = torch.ones(1, len(EXPERT_COLUMNS), device=device)
    if evidence_mode == "rolled":
        features = torch.roll(features, shifts=1, dims=1)
    elif evidence_mode == "zero_gr":
        features = features.clone()
        features[:, :, :29] = 0.0
    elif evidence_mode != "original":
        raise ValueError(evidence_mode)
    prediction, blend, weights, coefficients = model(
        features, experts, source_indices, active
    )
    compact_weights = weights[0].cpu().numpy()
    coefficients_np = coefficients[0].cpu().numpy()
    compact_x = np.linspace(0.0, 1.0, item.experts.shape[1])
    full_x = np.linspace(0.0, 1.0, item.nrows)
    if compact_weights.ndim == 1:
        full_weights = np.repeat(
            compact_weights[:, None],
            item.nrows,
            axis=1,
        )
    else:
        full_weights = np.stack(
            [
                np.interp(full_x, compact_x, compact_weights[index])
                for index in range(len(EXPERT_COLUMNS))
            ],
            axis=0,
        )
        full_weights /= np.sum(full_weights, axis=0, keepdims=True)
    full_blend = np.sum(full_weights * item.original_experts, axis=0)
    correction = (
        coefficients_np[0] * full_x
        + coefficients_np[1] * (4.0 * full_x * (1.0 - full_x))
    )
    full_prediction = full_blend + correction
    return (
        full_prediction.astype(np.float32),
        full_blend.astype(np.float32),
        full_weights.astype(np.float32),
        coefficients_np.astype(np.float32),
    )


def aggregate_metrics(rows: list[dict], prefix: str) -> dict[str, float]:
    sse = sum(row[f"{prefix}_sse"] for row in rows)
    nrows = sum(row["nrows"] for row in rows)
    rmse_values = np.asarray([row[f"{prefix}_rmse"] for row in rows])
    return {
        "pooled": float(np.sqrt(sse / nrows)),
        "mean_per_well": float(np.mean(rmse_values)),
        "median_per_well": float(np.median(rmse_values)),
        "q90_per_well": float(np.quantile(rmse_values, 0.90)),
    }


@torch.no_grad()
def evaluate(
    model: SourceBlindPhysicalMoE,
    items: list[WellItem],
    device: torch.device,
    evidence_mode: str = "original",
) -> tuple[dict[str, dict[str, float]], list[dict], dict[str, np.ndarray]]:
    model.eval()
    rows: list[dict] = []
    predictions: dict[str, np.ndarray] = {}
    for item in items:
        prediction, blend, weights, coefficients = predict_item(
            model, item, device, evidence_mode
        )
        truth = item.original_truth.astype(float)
        static_weights = torch.softmax(model.base_logits, dim=0).cpu().numpy()
        static = np.sum(static_weights[:, None] * item.original_experts, axis=0)
        expert_rmse = np.sqrt(
            np.mean(
                np.square(item.original_experts.astype(float) - truth[None]),
                axis=1,
            )
        )
        row: dict[str, float | str | int] = {
            "well": item.well,
            "fold": item.fold,
            "nrows": item.nrows,
            "oracle_expert_rmse": float(np.min(expert_rmse)),
            "endpoint_coefficient": float(coefficients[0]),
            "midpoint_coefficient": float(coefficients[1]),
        }
        for name, values in (
            ("static", static),
            ("blend", blend),
            ("prediction", prediction),
        ):
            error = values - truth
            row[f"{name}_sse"] = float(np.dot(error, error))
            row[f"{name}_rmse"] = float(np.sqrt(np.mean(error * error)))
        for index, name in enumerate(EXPERT_NAMES):
            row[f"weight_{name}"] = float(np.mean(weights[index]))
            row[f"rmse_{name}"] = float(expert_rmse[index])
        rows.append(row)
        predictions[item.well] = prediction
    metrics = {
        name: aggregate_metrics(rows, name)
        for name in ("static", "blend", "prediction")
    }
    oracle_sse = sum(
        row["oracle_expert_rmse"] ** 2 * row["nrows"] for row in rows
    )
    metrics["oracle_expert"] = {
        "pooled": float(np.sqrt(oracle_sse / sum(row["nrows"] for row in rows))),
        "mean_per_well": float(
            np.mean([row["oracle_expert_rmse"] for row in rows])
        ),
    }
    return metrics, rows, predictions


@torch.no_grad()
def permutation_invariance_error(
    model: SourceBlindPhysicalMoE,
    item: WellItem,
    device: torch.device,
) -> float:
    features = torch.from_numpy(item.features)[None].to(device)
    experts = torch.from_numpy(item.experts)[None].to(device)
    source = torch.arange(len(EXPERT_COLUMNS), device=device)[None]
    active = torch.ones(1, len(EXPERT_COLUMNS), device=device)
    original = model(features, experts, source, active)[0]
    order = torch.tensor([2, 0, 3, 1], device=device)
    permuted = model(
        features[:, order],
        experts[:, order],
        source[:, order],
        active[:, order],
    )[0]
    return float(torch.max(torch.abs(original - permuted)).cpu())


def train_fold(
    args: argparse.Namespace,
    items: list[WellItem],
    channel_names: list[str],
    heldout_fold: int,
    device: torch.device,
) -> tuple[list[dict], dict[str, np.ndarray], dict]:
    train_items = [item for item in items if item.fold != heldout_fold]
    val_items = [item for item in items if item.fold == heldout_fold]
    fitted_weights = fit_static_weights(train_items)
    base_weights = floor_prior(fitted_weights, args.prior_floor)
    model = SourceBlindPhysicalMoE(
        channels=len(channel_names),
        hidden=args.hidden,
        embedding=args.embedding,
        blocks=args.blocks,
        dropout=args.dropout,
        max_logit_delta=args.max_logit_delta,
        correction_scale_ft=args.correction_scale_ft,
        base_weights=base_weights,
        correction_enabled=not args.disable_correction,
        weight_mode=args.weight_mode,
        local_smoothing=args.local_smoothing,
    ).to(device)
    loader = DataLoader(
        PhysicalExpertDataset(train_items, True, args.source_dropout),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    initial_metrics, _, _ = evaluate(model, val_items, device)
    print(
        f"fold {heldout_fold} ep00 | "
        f"static={initial_metrics['static']['pooled']:.4f} "
        f"pred={initial_metrics['prediction']['pooled']:.4f} "
        f"oracle={initial_metrics['oracle_expert']['pooled']:.4f} "
        f"weights={np.round(base_weights, 4).tolist()} "
        f"fitted={np.round(fitted_weights, 4).tolist()}",
        flush=True,
    )
    best_metric = initial_metrics["prediction"]["pooled"]
    best_epoch = 0
    best_state = {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }
    stale = 0
    last_epoch = 0
    for epoch in range(1, args.epochs + 1):
        last_epoch = epoch
        model.train()
        running = {
            "loss": 0.0,
            "data": 0.0,
            "rank": 0.0,
            "anchor": 0.0,
            "correction": 0.0,
        }
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            loss, parts = training_loss(model, batch, device, args)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running["loss"] += float(loss.detach().cpu())
            for key, value in parts.items():
                running[key] += value
        metrics, _, _ = evaluate(model, val_items, device)
        count = max(len(loader), 1)
        print(
            f"fold {heldout_fold} ep{epoch:02d} | "
            f"loss={running['loss']/count:.4f} "
            f"data={running['data']/count:.4f} "
            f"rank={running['rank']/count:.4f} "
            f"anchor={running['anchor']/count:.5f} "
            f"corr={running['correction']/count:.5f} | "
            f"blend={metrics['blend']['pooled']:.4f} "
            f"pred={metrics['prediction']['pooled']:.4f} "
            f"mean={metrics['prediction']['mean_per_well']:.4f}",
            flush=True,
        )
        value = metrics["prediction"]["pooled"]
        if value < best_metric - 1e-4:
            best_metric = value
            best_epoch = epoch
            best_state = {
                key: tensor.detach().cpu().clone()
                for key, tensor in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
            if args.checkpoint_mode == "best" and stale >= args.patience:
                break

    if args.checkpoint_mode == "best":
        model.load_state_dict(best_state)
    else:
        best_epoch = last_epoch
        best_state = {
            key: tensor.detach().cpu().clone()
            for key, tensor in model.state_dict().items()
        }
    metrics, rows, predictions = evaluate(model, val_items, device)
    rolled, _, _ = evaluate(model, val_items, device, "rolled")
    zero_gr, _, _ = evaluate(model, val_items, device, "zero_gr")
    invariance = permutation_invariance_error(model, val_items[0], device)
    print(
        f"fold {heldout_fold} guardrail | "
        f"original={metrics['prediction']['pooled']:.4f} "
        f"rolled={rolled['prediction']['pooled']:.4f} "
        f"zero_gr={zero_gr['prediction']['pooled']:.4f} "
        f"perm_diff={invariance:.3e}",
        flush=True,
    )
    checkpoint = {
        "model": best_state,
        "heldout_fold": heldout_fold,
        "epoch": best_epoch,
        "metrics": metrics,
        "rolled_evidence_metrics": rolled,
        "zero_gr_metrics": zero_gr,
        "permutation_invariance_error": invariance,
        "base_weights": base_weights.tolist(),
        "fitted_weights": fitted_weights.tolist(),
        "feature_names": channel_names,
        "args": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
    }
    return rows, predictions, checkpoint


def train_production(
    args: argparse.Namespace,
    items: list[WellItem],
    channel_names: list[str],
    device: torch.device,
) -> dict:
    fitted_weights = fit_static_weights(items)
    base_weights = floor_prior(fitted_weights, args.prior_floor)
    model = SourceBlindPhysicalMoE(
        channels=len(channel_names),
        hidden=args.hidden,
        embedding=args.embedding,
        blocks=args.blocks,
        dropout=args.dropout,
        max_logit_delta=args.max_logit_delta,
        correction_scale_ft=args.correction_scale_ft,
        base_weights=base_weights,
        correction_enabled=not args.disable_correction,
        weight_mode=args.weight_mode,
        local_smoothing=args.local_smoothing,
    ).to(device)
    loader = DataLoader(
        PhysicalExpertDataset(items, True, args.source_dropout),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = {
            "loss": 0.0,
            "data": 0.0,
            "rank": 0.0,
            "anchor": 0.0,
            "correction": 0.0,
        }
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            loss, parts = training_loss(model, batch, device, args)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running["loss"] += float(loss.detach().cpu())
            for key, value in parts.items():
                running[key] += value
        count = max(len(loader), 1)
        print(
            f"production ep{epoch:02d} | "
            f"loss={running['loss']/count:.4f} "
            f"data={running['data']/count:.4f} "
            f"rank={running['rank']/count:.4f} "
            f"anchor={running['anchor']/count:.5f}",
            flush=True,
        )
    train_metrics, _, _ = evaluate(model, items, device)
    print(
        f"production diagnostic | "
        f"train_static={train_metrics['static']['pooled']:.4f} "
        f"train_model={train_metrics['prediction']['pooled']:.4f}",
        flush=True,
    )
    return {
        "model": {
            key: tensor.detach().cpu()
            for key, tensor in model.state_dict().items()
        },
        "production": True,
        "epochs": args.epochs,
        "train_metrics": train_metrics,
        "base_weights": base_weights.tolist(),
        "fitted_weights": fitted_weights.tolist(),
        "feature_names": channel_names,
        "args": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
    }


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = choose_device(args.device)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    items, channel_names = load_items(args)
    available_folds = sorted({item.fold for item in items})
    if args.folds == "all":
        folds = available_folds
    else:
        folds = [int(value) for value in args.folds.split(",") if value]
    print(
        f"device={device} wells={len(items)} folds={folds} "
        f"features={len(channel_names)} points={args.points} "
        f"hidden={args.hidden} embedding={args.embedding}",
        flush=True,
    )
    print("Feature channels:", ", ".join(channel_names), flush=True)

    if args.production:
        checkpoint = train_production(
            args,
            items,
            channel_names,
            device,
        )
        output = args.out_dir / "production.pth"
        torch.save(checkpoint, output)
        print(f"Saved {output}", flush=True)
        return

    all_rows: list[dict] = []
    all_predictions: dict[str, np.ndarray] = {}
    for fold in folds:
        seed_everything(args.seed + fold)
        rows, predictions, checkpoint = train_fold(
            args,
            items,
            channel_names,
            fold,
            device,
        )
        all_rows.extend(rows)
        all_predictions.update(predictions)
        torch.save(checkpoint, args.out_dir / f"fold_{fold}.pth")

    per_well = pd.DataFrame(all_rows).sort_values("well")
    per_well.to_csv(args.out_dir / "per_well.csv", index=False)
    summary = {
        name: aggregate_metrics(all_rows, name)
        for name in ("static", "blend", "prediction")
    }
    summary["folds"] = folds
    summary["wells"] = len(all_rows)
    summary["args"] = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    if set(folds) == set(available_folds):
        source = pd.read_parquet(
            args.components,
            columns=["well", "fold", "position", "true_tvt", *EXPERT_COLUMNS],
        ).sort_values(["well", "position"])
        diagnostics = pd.DataFrame(all_rows).set_index("well")
        pieces = []
        for well, group in source.groupby("well", sort=True):
            piece = group.copy()
            row = diagnostics.loc[str(well)]
            if args.disable_correction:
                blend = all_predictions[str(well)]
            else:
                blend = sum(
                    float(row[f"weight_{name}"])
                    * piece[column].to_numpy(float)
                    for name, column in zip(EXPERT_NAMES, EXPERT_COLUMNS)
                )
            piece["pred_source_blind_blend"] = blend.astype(np.float32)
            piece["pred_source_blind_corrected"] = all_predictions[str(well)]
            pieces.append(piece)
        pd.concat(pieces, ignore_index=True).to_parquet(
            args.out_dir / "oof_predictions.parquet",
            index=False,
        )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
