# Alexey Zakharov (drzoidberg), ROGII team solution.
"""Per-well uncertainty features for the Q/HMM gate."""
from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np


PF_MEMBER_ORDER = ("g30", "g45", "g60", "mom", "adjh")


def _vector(name: str, values: Sequence[float], rows: int | None = None) -> np.ndarray:
    array = np.asarray(values, dtype=float).reshape(-1)
    if rows is not None and len(array) != rows:
        raise ValueError(f"{name} has {len(array)} rows; expected {rows}")
    if name != "raw_tail_gr" and not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    return array


def _mean_p90(values: np.ndarray) -> tuple[float, float]:
    return float(np.mean(values)), float(np.quantile(values, 0.90))


def summarize_gate_features(
    *,
    pf_predictions: Mapping[str, Sequence[float]],
    q_prediction: Sequence[float],
    hmm_prediction: Sequence[float],
    hmm_smooth: Sequence[float],
    hmm_filter: Sequence[float],
    hmm_mode: Sequence[float],
    kvar: Sequence[float],
    q_member_weights: Mapping[str, float] | Sequence[float],
    raw_tail_gr: Sequence[float],
) -> dict[str, float]:
    """Build 16 uncertainty features for one completed well."""
    q = _vector("q_prediction", q_prediction)
    rows = len(q)
    if rows == 0:
        raise ValueError("a well must contain at least one predicted row")
    hmm = _vector("hmm_prediction", hmm_prediction, rows)
    smooth = _vector("hmm_smooth", hmm_smooth, rows)
    filt = _vector("hmm_filter", hmm_filter, rows)
    mode = _vector("hmm_mode", hmm_mode, rows)
    variance = _vector("kvar", kvar, rows)
    raw_gr = _vector("raw_tail_gr", raw_tail_gr, rows)

    missing = [name for name in PF_MEMBER_ORDER if name not in pf_predictions]
    if missing:
        raise ValueError(f"missing PF members: {missing}")
    legs = np.column_stack(
        [_vector(f"pf_predictions[{name}]", pf_predictions[name], rows) for name in PF_MEMBER_ORDER]
    )

    if isinstance(q_member_weights, Mapping):
        weights = np.asarray([q_member_weights[name] for name in PF_MEMBER_ORDER], dtype=float)
    else:
        weights = np.asarray(q_member_weights, dtype=float).reshape(-1)
    if weights.shape != (len(PF_MEMBER_ORDER),) or not np.isfinite(weights).all():
        raise ValueError("q_member_weights must contain five finite values in PF_MEMBER_ORDER")
    if np.any(weights < 0.0) or not np.isclose(weights.sum(), 1.0, atol=1e-6):
        raise ValueError("q_member_weights must be non-negative and sum to one")

    pf_std = np.std(legs, axis=1)
    pf_range = np.ptp(legs, axis=1)
    qh_gap = np.abs(q - hmm)
    hmm_sf_gap = np.abs(smooth - filt)
    hmm_mode_gap = np.abs(smooth - mode)
    sigma = np.sqrt(np.clip(variance, 0.0, None))

    features: dict[str, float] = {}
    for prefix, values in (
        ("pf_std", pf_std),
        ("pf_range", pf_range),
        ("qh_gap", qh_gap),
        ("hmm_sf", hmm_sf_gap),
        ("hmm_mode", hmm_mode_gap),
        ("sigma", sigma),
    ):
        features[f"{prefix}_mean"], features[f"{prefix}_p90"] = _mean_p90(values)
    features["q_weight_entropy"] = float(
        -np.sum(weights * np.log(np.clip(weights, 1e-12, None))) / np.log(len(weights))
    )
    features["q_weight_max"] = float(np.max(weights))
    features["gr_observed_frac"] = float(np.isfinite(raw_gr).mean())
    features["log_rows"] = float(np.log(rows))
    return features
