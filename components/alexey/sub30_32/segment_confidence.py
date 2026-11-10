# Alexey Zakharov (drzoidberg), ROGII team solution.
"""Frozen segment-confidence kappa model."""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


def load_segments(path: str | Path) -> dict:
    with Path(path).open("rb") as stream:
        model = pickle.load(stream)
    required = {"SC", "SH", "SA", "R", "BW", "PGATE", "CS", "SEG", "KLO", "KHI"}
    missing = required.difference(model)
    if missing:
        raise ValueError(f"segment model is missing {sorted(missing)}")
    model["tree"] = cKDTree(np.asarray(model["SC"], dtype=float))
    return model


def kappa(horizontal_well_frame, model: dict) -> np.ndarray:
    """Return full-row kappa using only trajectory XY and frozen train segments."""
    md = horizontal_well_frame["MD"].to_numpy(float)
    x = horizontal_well_frame["X"].to_numpy(float)
    y = horizontal_well_frame["Y"].to_numpy(float)
    sc = np.asarray(model["SC"], dtype=float)
    sh = np.asarray(model["SH"], dtype=float)
    sa = np.asarray(model["SA"], dtype=float)
    spread = np.full(len(md), np.nan)
    segment_id = np.floor((md - md[0]) / float(model["SEG"])).astype(int)

    for segment in np.unique(segment_id):
        rows = np.flatnonzero(segment_id == segment)
        horizontal_distance = np.hypot(x[rows][-1] - x[rows][0], y[rows][-1] - y[rows][0])
        if horizontal_distance < 50.0:
            continue
        hx = (x[rows][-1] - x[rows][0]) / horizontal_distance
        hy = (y[rows][-1] - y[rows][0]) / horizontal_distance
        neighbors = model["tree"].query_ball_point((x[rows].mean(), y[rows].mean()), float(model["R"]))
        if len(neighbors) < 3:
            continue
        ids = np.asarray(neighbors, dtype=int)
        alignment = sh[ids, 0] * hx + sh[ids, 1] * hy
        parallel = np.abs(alignment) > float(model["PGATE"])
        ids, alignment = ids[parallel], alignment[parallel]
        if len(ids) < 3:
            continue
        distance = np.hypot(sc[ids, 0] - x[rows].mean(), sc[ids, 1] - y[rows].mean())
        weight = np.exp(-(distance / float(model["BW"])) ** 2)
        values = sa[ids] * np.sign(alignment)
        center = np.sum(weight * values) / np.sum(weight)
        spread[rows] = np.sqrt(np.sum(weight * (values - center) ** 2) / np.sum(weight))

    confidence = np.where(np.isnan(spread), 0.0, np.exp(-spread / float(model["CS"])))
    return float(model["KLO"]) + confidence * (float(model["KHI"]) - float(model["KLO"]))
