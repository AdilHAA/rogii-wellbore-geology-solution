# Alexey Zakharov (drzoidberg), ROGII team solution.
"""Logistic Q/HMM gate."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

import numpy as np


def load_gate(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())


def q_weight(features: Mapping[str, float], model: dict) -> float:
    log_features = set(model["preprocessing"]["log1p_features"])
    transformed = {
        name: float(np.log1p(float(features[name]))) if name in log_features else float(features[name])
        for name in model["feature_order"]
    }
    score = float(model["coefficients"]["intercept"])
    for name in model["feature_order"]:
        mean = float(model["preprocessing"]["mean_after_transform"][name])
        std = float(model["preprocessing"]["std_after_transform"][name])
        score += float(model["coefficients"][name]) * (transformed[name] - mean) / std
    return float(1.0 / (1.0 + np.exp(-score)))
