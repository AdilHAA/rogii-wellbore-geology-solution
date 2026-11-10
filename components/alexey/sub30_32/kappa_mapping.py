# Alexey Zakharov (drzoidberg), ROGII team solution.
"""Frozen sigma-rank-to-kappa mapping."""
from __future__ import annotations

from pathlib import Path

import numpy as np


DEFAULT_KAPPA = np.array([0.03, 0.014230249470757707, 0.003], dtype=float)
MOM_KAPPA = np.array([0.005, 0.002, 0.001], dtype=float)
RANK_KNOTS = np.array([0.0, 0.5, 1.0], dtype=float)


def load_sigma_cdf(path: str | Path) -> np.ndarray:
    knots = np.load(path)
    if knots.ndim != 1 or len(knots) != 1001 or np.any(np.diff(knots) < 0):
        raise ValueError("expected the frozen monotone 1001-point sigma CDF")
    return knots.astype(float, copy=False)


def kappa_from_kvar(kvar, sigma_cdf, member: str = "g60") -> np.ndarray:
    sigma = np.sqrt(np.clip(np.asarray(kvar, dtype=float), 0.0, None))
    quantile_grid = np.linspace(0.0, 1.0, len(sigma_cdf))
    rank = np.clip(np.interp(sigma, sigma_cdf, quantile_grid), 0.0, 1.0)
    kappa_knots = MOM_KAPPA if member == "mom" else DEFAULT_KAPPA
    return np.interp(rank, RANK_KNOTS, kappa_knots).astype(np.float32)
