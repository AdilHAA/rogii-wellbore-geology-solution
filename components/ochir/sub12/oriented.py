# Ochir Dorzhiev
"""Truth-free oriented STRIDE decoder.

The decoder searches a path in structural-surface space

    U(MD) = TVT(MD) + Z(MD)

and evaluates candidate surface slopes with an oriented raw-GR cost.  Hidden
suffix TVT is never read by :func:`decode_well`; scoring is deliberately kept
in separate helper functions.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
from numba import njit
from scipy.ndimage import gaussian_filter1d
from scipy.signal import fftconvolve

DEFAULT_SURFACE_SLOPES = tuple(
    np.arange(-0.10, 0.1101, 0.01, dtype=np.float64).round(10).tolist()
)


@dataclass(frozen=True)
class OrientedConfig:
    """Configuration for the oriented surface decoder."""

    name: str = "cap30_s008"
    endpoint_step: float = 100.0
    obs_step: float = 25.0
    lengths_blocks: tuple[int, ...] = (1,)
    beam: int = 80
    slope_grid: tuple[float, ...] = DEFAULT_SURFACE_SLOPES
    raw_cost_cap: float = 30.0
    raw_cost_scale: float = 10.0
    observation_model: str = "clipped_l1"
    student_dof: float = 3.0
    gr_transform: str = "raw"
    gr_asinh_scale: float = 20.0
    gr_calibration: str = "none"
    gr_calibration_shrink: float = 1.0
    gr_calibration_slope_min: float = 0.25
    gr_calibration_slope_max: float = 4.0
    typewell_response: str = "point"
    typewell_response_ft: float = 0.0
    self_weight: float = 0.0
    self_radius_ft: float = 0.5
    self_neighbors: int = 5
    self_prefix_mode: str = "all"
    self_lateral_dz_limit: float = 0.15
    self_directional: bool = False
    self_direction_threshold: float = 0.005
    emission_weight: float = 1.0
    smooth_sigma: float = 0.008
    turn_sigma: float = 0.050
    turn_penalty: float = 4.0
    use_turn_mixture: bool = False
    absolute_slope_sigma: float = 0.10
    absolute_slope_weight: float = 0.10
    segment_penalty: float = 0.0
    missing_cost: float = 3.0
    tvt_bin: float = 1.0
    max_same: int = 1
    posterior_temp: float = 4.0
    mode_radius_ft: float = 8.0
    min_final_fraction: float = 0.0
    prefix_slope_lookback: float = 800.0
    sequence_derivative_weight: float = 0.0
    sequence_motif_weight: float = 0.0
    sequence_reverse_weight: float = 0.0
    sequence_derivative_gamma: float = 12.0
    sequence_window_points: tuple[int, ...] = (4, 8, 16)
    sequence_tvt_step: float = 2.0

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)


def candidate_configs() -> list[OrientedConfig]:
    """Candidate grid used in the reported nested OOF experiment."""

    return [
        OrientedConfig(name="cap20_s008", raw_cost_cap=20.0, smooth_sigma=0.008),
        OrientedConfig(name="cap30_s005", raw_cost_cap=30.0, smooth_sigma=0.005),
        OrientedConfig(name="cap30_s006", raw_cost_cap=30.0, smooth_sigma=0.006),
        OrientedConfig(name="cap30_s008", raw_cost_cap=30.0, smooth_sigma=0.008),
        OrientedConfig(name="cap30_s010", raw_cost_cap=30.0, smooth_sigma=0.010),
        OrientedConfig(name="cap30_s012", raw_cost_cap=30.0, smooth_sigma=0.012),
        OrientedConfig(name="cap40_s008", raw_cost_cap=40.0, smooth_sigma=0.008),
        OrientedConfig(name="noclip_s008", raw_cost_cap=1e6, smooth_sigma=0.008),
    ]


def parse_json_array(value: str | Iterable[float] | np.ndarray) -> np.ndarray:
    """Parse an array stored in a CSV JSON cell."""

    if isinstance(value, np.ndarray):
        return value.astype(float, copy=False)
    if isinstance(value, str):
        return np.asarray(json.loads(value), dtype=float)
    return np.asarray(list(value), dtype=float)


def _transform_gr(values: np.ndarray, mode: str, asinh_scale: float) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if mode == "raw":
        return values.copy()
    if mode == "log":
        return np.log1p(np.maximum(values, 0.0))
    if mode == "asinh":
        return np.arcsinh(values / max(float(asinh_scale), 1e-6))
    raise ValueError(f"Unknown GR transform: {mode}")


def _apply_typewell_response(
    values: np.ndarray,
    step_ft: float,
    family: str,
    width_ft: float,
) -> np.ndarray:
    """Apply a simple vertical response function before path evaluation."""

    values = np.asarray(values, dtype=np.float64)
    if family == "point" or width_ft <= 0.0:
        return values.copy()
    if family == "gaussian":
        sigma = width_ft / max(step_ft, 1e-6)
        return gaussian_filter1d(values, sigma, mode="nearest")
    if family == "cusp":
        radius = max(2, int(math.ceil(8.0 * width_ft / max(step_ft, 1e-6))))
        x = np.arange(-radius, radius + 1, dtype=np.float64) * step_ft
        kernel = np.exp(-np.abs(x) / max(width_ft, 1e-6))
        kernel /= kernel.sum()
        padded = np.pad(values, radius, mode="edge")
        return fftconvolve(padded, kernel, mode="same")[radius:-radius]
    raise ValueError(f"Unknown typewell response: {family}")


def _build_self_reference(
    grid_tvt: np.ndarray,
    prefix_tvt: np.ndarray,
    prefix_gr: np.ndarray,
    radius_ft: float,
    neighbors: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Map known horizontal GR onto TVT without interpolating uncovered gaps."""

    valid = np.isfinite(prefix_tvt) & np.isfinite(prefix_gr)
    x = np.asarray(prefix_tvt, dtype=np.float64)[valid]
    y = np.asarray(prefix_gr, dtype=np.float64)[valid]
    profile = np.zeros(len(grid_tvt), dtype=np.float64)
    available = np.zeros(len(grid_tvt), dtype=np.uint8)
    if len(x) == 0:
        return profile, available
    order = np.argsort(x)
    x, y = x[order], y[order]
    neighbors = max(1, int(neighbors))
    for index, tvt in enumerate(grid_tvt):
        position = int(np.searchsorted(x, tvt))
        lo = max(0, position - 3 * neighbors)
        hi = min(len(x), position + 3 * neighbors + 1)
        candidates = np.arange(lo, hi)
        if len(candidates) == 0:
            continue
        distance = np.abs(x[candidates] - tvt)
        candidates = candidates[np.argsort(distance)[:neighbors]]
        candidates = candidates[np.abs(x[candidates] - tvt) <= radius_ft]
        if len(candidates):
            profile[index] = float(np.median(y[candidates]))
            available[index] = 1
    return profile, available


def _build_self_bank(
    grid_tvt: np.ndarray,
    prefix_md: np.ndarray,
    prefix_z: np.ndarray,
    prefix_tvt: np.ndarray,
    prefix_gr: np.ndarray,
    cfg: OrientedConfig,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Build all/up/flat/down self templates from the selected prefix section."""

    selected = np.ones(len(prefix_tvt), dtype=bool)
    fallback = np.ones(len(prefix_tvt), dtype=bool)
    lateral_start = 0
    if cfg.self_prefix_mode in {"lateral", "lateral_prefer"}:
        z_rate = np.gradient(prefix_z, prefix_md)
        z_rate = (
            pd.Series(z_rate)
            .rolling(21, center=True, min_periods=1)
            .median()
            .to_numpy(float)
        )
        lateral = np.abs(z_rate) < cfg.self_lateral_dz_limit
        lateral_start = len(prefix_tvt) - 1
        while lateral_start > 0 and lateral[lateral_start]:
            lateral_start -= 1
        if not lateral[lateral_start]:
            lateral_start += 1
        selected[:lateral_start] = False
        if selected.sum() < 20:
            selected[:] = True
            lateral_start = 0
        if cfg.self_prefix_mode == "lateral":
            fallback = selected.copy()
    elif cfg.self_prefix_mode != "all":
        raise ValueError(f"Unknown self prefix mode: {cfg.self_prefix_mode}")

    bank = np.zeros((5, len(grid_tvt)), dtype=np.float64)
    available = np.zeros((5, len(grid_tvt)), dtype=np.uint8)
    bank[0], available[0] = _build_self_reference(
        grid_tvt,
        prefix_tvt[fallback],
        prefix_gr[fallback],
        cfg.self_radius_ft,
        cfg.self_neighbors,
    )
    bank[1], available[1] = _build_self_reference(
        grid_tvt,
        prefix_tvt[selected],
        prefix_gr[selected],
        cfg.self_radius_ft,
        cfg.self_neighbors,
    )

    tvt_rate = np.gradient(prefix_tvt, prefix_md)
    tvt_rate = (
        pd.Series(tvt_rate)
        .rolling(21, center=True, min_periods=1)
        .median()
        .to_numpy(float)
    )
    direction = np.ones(len(prefix_tvt), dtype=np.int8)
    direction[tvt_rate < -cfg.self_direction_threshold] = 0
    direction[tvt_rate > cfg.self_direction_threshold] = 2
    for direction_index in range(3):
        use = selected & (direction == direction_index)
        bank[direction_index + 2], available[direction_index + 2] = (
            _build_self_reference(
                grid_tvt,
                prefix_tvt[use],
                prefix_gr[use],
                cfg.self_radius_ft,
                cfg.self_neighbors,
            )
        )
    return bank, available, lateral_start


def _robust_affine(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    valid = np.isfinite(x) & np.isfinite(y)
    x = np.asarray(x, dtype=np.float64)[valid]
    y = np.asarray(y, dtype=np.float64)[valid]
    if len(x) < 20 or np.std(x) < 1e-9:
        return 1.0, float(np.nanmedian(y - x)) if len(x) else 0.0
    design = np.column_stack([x, np.ones(len(x))])
    beta = np.linalg.lstsq(design, y, rcond=None)[0]
    for _ in range(5):
        residual = y - design @ beta
        center = float(np.median(residual))
        scale = 1.4826 * float(np.median(np.abs(residual - center))) + 1e-6
        weight = np.minimum(1.0, 1.5 * scale / (np.abs(residual - center) + 1e-9))
        root = np.sqrt(weight)
        beta = np.linalg.lstsq(design * root[:, None], y * root, rcond=None)[0]
    return float(beta[0]), float(beta[1])


def _calibrate_typewell_gr(
    md_gr: np.ndarray,
    tvt_input: np.ndarray,
    prefix_end: int,
    typewell_tvt: np.ndarray,
    typewell_gr: np.ndarray,
    cfg: OrientedConfig,
) -> tuple[np.ndarray, float, float]:
    if cfg.gr_calibration == "none":
        return typewell_gr, 1.0, 0.0
    prefix = np.arange(prefix_end + 1)
    valid = np.isfinite(md_gr[prefix]) & np.isfinite(tvt_input[prefix])
    prefix = prefix[valid]
    if len(prefix) < 20:
        return typewell_gr, 1.0, 0.0
    expected = np.interp(
        tvt_input[prefix], typewell_tvt, typewell_gr, left=np.nan, right=np.nan
    )
    observed = md_gr[prefix]
    valid = np.isfinite(expected) & np.isfinite(observed)
    expected = expected[valid]
    observed = observed[valid]
    if len(expected) < 20:
        return typewell_gr, 1.0, 0.0
    if cfg.gr_calibration == "offset":
        gain = 1.0
        offset = float(np.median(observed - expected))
    elif cfg.gr_calibration == "affine":
        gain, offset = _robust_affine(expected, observed)
        gain = float(
            np.clip(
                gain,
                cfg.gr_calibration_slope_min,
                cfg.gr_calibration_slope_max,
            )
        )
    else:
        raise ValueError(f"Unknown GR calibration: {cfg.gr_calibration}")
    shrink = float(np.clip(cfg.gr_calibration_shrink, 0.0, 1.0))
    gain = 1.0 + shrink * (gain - 1.0)
    offset *= shrink
    return gain * typewell_gr + offset, float(gain), float(offset)


def interpolate_boundary_prediction(
    md: np.ndarray,
    row: Mapping[str, Any] | pd.Series,
    value_column: str = "mean_tvt_boundaries",
) -> np.ndarray:
    """Interpolate a boundary representation to dense MD samples."""

    bounds = parse_json_array(row["boundaries"])
    values = parse_json_array(row[value_column])
    if bounds.ndim != 1 or values.ndim != 1 or len(bounds) != len(values):
        raise ValueError("Invalid boundary representation")
    order = np.argsort(bounds)
    return np.interp(np.asarray(md, dtype=float), bounds[order], values[order])


def _bin_observations(
    md: np.ndarray,
    gr: np.ndarray,
    z: np.ndarray,
    start: float,
    end: float,
    step: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    edges = np.arange(start, end + step, step, dtype=float)
    if edges[-1] < end:
        edges = np.r_[edges, end]
    centers = (edges[:-1] + edges[1:]) / 2.0
    index = np.searchsorted(md, edges)
    gr_binned = np.full(len(centers), np.nan, dtype=float)
    z_binned = np.interp(centers, md, z)
    for i in range(len(centers)):
        chunk = gr[index[i] : index[i + 1]]
        if np.isfinite(chunk).any():
            gr_binned[i] = float(np.nanmedian(chunk))
    return centers, gr_binned, z_binned


def _prefix_surface_slope(
    md: np.ndarray,
    z: np.ndarray,
    tvt_input: np.ndarray,
    prefix_end: int,
    lookback: float,
) -> float:
    lower_md = md[prefix_end] - lookback
    row_index = np.arange(len(md))
    mask = (
        (row_index <= prefix_end)
        & (md >= lower_md)
        & np.isfinite(tvt_input)
    )
    if mask.sum() < 10:
        mask = (row_index <= prefix_end) & np.isfinite(tvt_input)
    if mask.sum() < 2:
        return 0.0
    surface = tvt_input[mask] + z[mask]
    x = md[mask] - md[prefix_end]
    return float(np.polyfit(x, surface, 1)[0])


@njit(cache=False, nogil=True)
def _interp_regular(
    t: float,
    t_min: float,
    t_step: float,
    values: np.ndarray,
) -> tuple[float, bool]:
    u = (t - t_min) / t_step
    i = int(math.floor(u))
    if i < 0 or i >= len(values) - 1:
        return 0.0, False
    fraction = u - i
    return values[i] * (1.0 - fraction) + values[i + 1] * fraction, True


@njit(cache=False, nogil=True)
def _lookup_self(
    t: float,
    t_min: float,
    t_step: float,
    values: np.ndarray,
    available: np.ndarray,
    direction_index: int,
    use_directional: int,
) -> tuple[float, bool]:
    index = int(round((t - t_min) / t_step))
    if index < 0 or index >= values.shape[1]:
        return 0.0, False
    if use_directional == 1:
        row = direction_index + 2
        if available[row, index] != 0:
            return values[row, index], True
    if available[1, index] != 0:
        return values[1, index], True
    if available[0, index] != 0:
        return values[0, index], True
    return 0.0, False


@njit(cache=False, nogil=True)
def _build_sequence_cost_volume(
    bounds: np.ndarray,
    z_bounds: np.ndarray,
    obs_md: np.ndarray,
    obs_gr: np.ndarray,
    obs_z: np.ndarray,
    obs_right: np.ndarray,
    typewell_gr: np.ndarray,
    typewell_t_min: float,
    typewell_t_step: float,
    slope_grid: np.ndarray,
    tvt_grid_min: float,
    tvt_grid_step: float,
    n_tvt: int,
    window_points: np.ndarray,
    derivative_gamma: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build directional derivative/NCC maps for every endpoint and TVT state."""

    n_endpoints = len(bounds)
    n_slopes = len(slope_grid)
    derivative = np.zeros((n_endpoints, n_slopes, n_tvt), np.float32)
    motif = np.zeros((n_endpoints, n_slopes, n_tvt), np.float32)
    reverse_margin = np.zeros((n_endpoints, n_slopes, n_tvt), np.float32)

    maximum_window = 0
    for index in range(len(window_points)):
        if window_points[index] > maximum_window:
            maximum_window = int(window_points[index])

    for endpoint in range(1, n_endpoints):
        observation_end = int(obs_right[endpoint])
        if observation_end < 2:
            continue
        derivative_start = max(0, observation_end - maximum_window)
        endpoint_md = bounds[endpoint]
        endpoint_z = z_bounds[endpoint]

        for slope_index in range(n_slopes):
            slope = slope_grid[slope_index]
            for tvt_index in range(n_tvt):
                endpoint_tvt = tvt_grid_min + tvt_grid_step * tvt_index
                endpoint_surface = endpoint_tvt + endpoint_z

                derivative_sum = 0.0
                derivative_count = 0
                for lag_index in range(3):
                    lag = 1 << lag_index
                    lag_weight = 1.0 / (1 << lag_index)
                    for observation_index in range(
                        derivative_start + lag, observation_end
                    ):
                        previous_index = observation_index - lag
                        observed_now = obs_gr[observation_index]
                        observed_previous = obs_gr[previous_index]
                        if not (
                            math.isfinite(observed_now)
                            and math.isfinite(observed_previous)
                        ):
                            continue
                        surface_now = endpoint_surface + slope * (
                            obs_md[observation_index] - endpoint_md
                        )
                        surface_previous = endpoint_surface + slope * (
                            obs_md[previous_index] - endpoint_md
                        )
                        predicted_now, valid_now = _interp_regular(
                            surface_now - obs_z[observation_index],
                            typewell_t_min,
                            typewell_t_step,
                            typewell_gr,
                        )
                        predicted_previous, valid_previous = _interp_regular(
                            surface_previous - obs_z[previous_index],
                            typewell_t_min,
                            typewell_t_step,
                            typewell_gr,
                        )
                        if not (valid_now and valid_previous):
                            continue
                        difference_error = (
                            predicted_now
                            - predicted_previous
                            - observed_now
                            + observed_previous
                        )
                        derivative_sum += lag_weight * math.log1p(
                            (difference_error / derivative_gamma)
                            * (difference_error / derivative_gamma)
                        )
                        derivative_count += 1
                if derivative_count > 0:
                    derivative[endpoint, slope_index, tvt_index] = (
                        derivative_sum / derivative_count
                    )

                motif_sum = 0.0
                reverse_sum = 0.0
                motif_windows = 0
                for window_index in range(len(window_points)):
                    width = int(window_points[window_index])
                    observation_start = max(0, observation_end - width)
                    count = 0
                    observed_total = 0.0
                    predicted_total = 0.0
                    observed_square = 0.0
                    predicted_square = 0.0
                    cross = 0.0
                    reverse_total = 0.0
                    reverse_square = 0.0
                    reverse_cross = 0.0
                    reverse_valid = True

                    for observation_index in range(
                        observation_start, observation_end
                    ):
                        observed_value = obs_gr[observation_index]
                        if not math.isfinite(observed_value):
                            continue
                        surface = endpoint_surface + slope * (
                            obs_md[observation_index] - endpoint_md
                        )
                        predicted_value, valid = _interp_regular(
                            surface - obs_z[observation_index],
                            typewell_t_min,
                            typewell_t_step,
                            typewell_gr,
                        )
                        if not valid:
                            continue

                        mirror_index = (
                            observation_start
                            + observation_end
                            - 1
                            - observation_index
                        )
                        mirror_surface = endpoint_surface + slope * (
                            obs_md[mirror_index] - endpoint_md
                        )
                        reverse_value, reverse_ok = _interp_regular(
                            mirror_surface - obs_z[mirror_index],
                            typewell_t_min,
                            typewell_t_step,
                            typewell_gr,
                        )
                        if not reverse_ok:
                            reverse_valid = False

                        count += 1
                        observed_total += observed_value
                        predicted_total += predicted_value
                        observed_square += observed_value * observed_value
                        predicted_square += predicted_value * predicted_value
                        cross += observed_value * predicted_value
                        if reverse_ok:
                            reverse_total += reverse_value
                            reverse_square += reverse_value * reverse_value
                            reverse_cross += observed_value * reverse_value

                    if count < 4:
                        continue
                    observed_variance = (
                        observed_square - observed_total * observed_total / count
                    )
                    predicted_variance = (
                        predicted_square - predicted_total * predicted_total / count
                    )
                    if observed_variance <= 1e-8 or predicted_variance <= 1e-8:
                        continue
                    forward_correlation = (
                        cross - observed_total * predicted_total / count
                    ) / math.sqrt(observed_variance * predicted_variance)
                    forward_correlation = min(1.0, max(-1.0, forward_correlation))
                    motif_sum += 1.0 - forward_correlation
                    motif_windows += 1

                    reverse_variance = (
                        reverse_square - reverse_total * reverse_total / count
                    )
                    if reverse_valid and reverse_variance > 1e-8:
                        reverse_correlation = (
                            reverse_cross - observed_total * reverse_total / count
                        ) / math.sqrt(observed_variance * reverse_variance)
                        reverse_correlation = min(
                            1.0, max(-1.0, reverse_correlation)
                        )
                        reverse_sum += reverse_correlation - forward_correlation

                if motif_windows > 0:
                    motif[endpoint, slope_index, tvt_index] = (
                        motif_sum / motif_windows
                    )
                    reverse_margin[endpoint, slope_index, tvt_index] = (
                        reverse_sum / motif_windows
                    )

    return derivative, motif, reverse_margin


@njit(cache=False, nogil=True)
def _lookup_cost_volume(
    volume: np.ndarray,
    endpoint: int,
    slope_index: int,
    tvt: float,
    tvt_grid_min: float,
    tvt_grid_step: float,
) -> float:
    coordinate = (tvt - tvt_grid_min) / tvt_grid_step
    left = int(math.floor(coordinate))
    if left < 0 or left >= volume.shape[2] - 1:
        return 0.0
    fraction = coordinate - left
    return (
        volume[endpoint, slope_index, left] * (1.0 - fraction)
        + volume[endpoint, slope_index, left + 1] * fraction
    )


@njit(cache=False, nogil=True)
def _decode_core(
    bounds: np.ndarray,
    z_bounds: np.ndarray,
    obs_md: np.ndarray,
    obs_gr: np.ndarray,
    obs_z: np.ndarray,
    obs_left: np.ndarray,
    obs_right: np.ndarray,
    typewell_gr: np.ndarray,
    self_gr: np.ndarray,
    self_valid: np.ndarray,
    self_weight: float,
    self_directional: int,
    self_direction_threshold: float,
    typewell_t_min: float,
    typewell_t_step: float,
    slope_grid: np.ndarray,
    lengths_blocks: np.ndarray,
    beam: int,
    raw_cost_cap: float,
    raw_cost_scale: float,
    observation_model: int,
    student_dof: float,
    emission_weight: float,
    smooth_sigma: float,
    turn_sigma: float,
    turn_penalty: float,
    use_turn_mixture: int,
    absolute_slope_sigma: float,
    absolute_slope_weight: float,
    segment_penalty: float,
    missing_cost: float,
    sequence_derivative: np.ndarray,
    sequence_motif: np.ndarray,
    sequence_reverse: np.ndarray,
    sequence_tvt_min: float,
    sequence_tvt_step: float,
    sequence_derivative_weight: float,
    sequence_motif_weight: float,
    sequence_reverse_weight: float,
    tvt_bin: float,
    max_same: int,
    min_final_fraction: float,
    initial_surface: float,
    initial_slope_index: int,
) -> tuple[np.ndarray, ...]:
    n_endpoints = len(bounds)
    n_slopes = len(slope_grid)
    infinity = 1e300

    state_surface = np.full((n_endpoints, beam), np.nan)
    state_slope = np.full((n_endpoints, beam), -1, np.int32)
    state_cost = np.full((n_endpoints, beam), infinity)
    parent_endpoint = np.full((n_endpoints, beam), -1, np.int32)
    parent_index = np.full((n_endpoints, beam), -1, np.int32)
    segment_slope = np.full((n_endpoints, beam), -1, np.int32)
    counts = np.zeros(n_endpoints, np.int32)

    state_surface[0, 0] = initial_surface
    state_slope[0, 0] = initial_slope_index
    state_cost[0, 0] = 0.0
    counts[0] = 1

    max_candidates = beam + beam * n_slopes
    candidate_surface = np.empty(max_candidates)
    candidate_slope = np.empty(max_candidates, np.int32)
    candidate_cost = np.empty(max_candidates)
    candidate_parent_endpoint = np.empty(max_candidates, np.int32)
    candidate_parent_index = np.empty(max_candidates, np.int32)
    candidate_segment_slope = np.empty(max_candidates, np.int32)

    for start_endpoint in range(n_endpoints - 1):
        n_start_states = counts[start_endpoint]
        if n_start_states == 0:
            continue

        for length_index in range(len(lengths_blocks)):
            length_blocks = int(lengths_blocks[length_index])
            end_endpoint = start_endpoint + length_blocks
            if end_endpoint >= n_endpoints:
                end_endpoint = n_endpoints - 1
            if end_endpoint <= start_endpoint:
                continue

            duration = bounds[end_endpoint] - bounds[start_endpoint]
            nominal_duration = length_blocks * (bounds[1] - bounds[0])
            if (
                end_endpoint == n_endpoints - 1
                and duration < min_final_fraction * nominal_duration
            ):
                continue

            n_existing = counts[end_endpoint]
            n_candidates = 0
            for q in range(n_existing):
                candidate_surface[n_candidates] = state_surface[end_endpoint, q]
                candidate_slope[n_candidates] = state_slope[end_endpoint, q]
                candidate_cost[n_candidates] = state_cost[end_endpoint, q]
                candidate_parent_endpoint[n_candidates] = parent_endpoint[end_endpoint, q]
                candidate_parent_index[n_candidates] = parent_index[end_endpoint, q]
                candidate_segment_slope[n_candidates] = segment_slope[end_endpoint, q]
                n_candidates += 1

            observation_start = obs_left[start_endpoint]
            observation_end = obs_right[end_endpoint]

            for parent_q in range(n_start_states):
                start_surface = state_surface[start_endpoint, parent_q]
                previous_slope = slope_grid[state_slope[start_endpoint, parent_q]]
                previous_cost = state_cost[start_endpoint, parent_q]

                for slope_index in range(n_slopes):
                    candidate_local_slope = slope_grid[slope_index]
                    emission_cost = 0.0

                    for obs_index in range(observation_start, observation_end):
                        if not math.isfinite(obs_gr[obs_index]):
                            continue
                        surface = start_surface + candidate_local_slope * (
                            obs_md[obs_index] - bounds[start_endpoint]
                        )
                        tvt = surface - obs_z[obs_index]
                        predicted_gr, valid = _interp_regular(
                            tvt,
                            typewell_t_min,
                            typewell_t_step,
                            typewell_gr,
                        )
                        if not valid:
                            emission_cost += missing_cost
                        else:
                            if self_weight > 0.0:
                                if obs_index > 0:
                                    z_delta = obs_z[obs_index] - obs_z[obs_index - 1]
                                    md_delta = obs_md[obs_index] - obs_md[obs_index - 1]
                                elif len(obs_md) > 1:
                                    z_delta = obs_z[1] - obs_z[0]
                                    md_delta = obs_md[1] - obs_md[0]
                                else:
                                    z_delta = 0.0
                                    md_delta = 1.0
                                z_rate = z_delta / max(md_delta, 1e-6)
                                tvt_rate = candidate_local_slope - z_rate
                                direction_index = 1
                                if tvt_rate < -self_direction_threshold:
                                    direction_index = 0
                                elif tvt_rate > self_direction_threshold:
                                    direction_index = 2
                                self_prediction, self_ok = _lookup_self(
                                    tvt,
                                    typewell_t_min,
                                    typewell_t_step,
                                    self_gr,
                                    self_valid,
                                    direction_index,
                                    self_directional,
                                )
                                if self_ok:
                                    predicted_gr = (
                                        (1.0 - self_weight) * predicted_gr
                                        + self_weight * self_prediction
                                    )
                            absolute_error = abs(predicted_gr - obs_gr[obs_index])
                            normalized_error = absolute_error / raw_cost_scale
                            if observation_model == 0:
                                if absolute_error > raw_cost_cap:
                                    absolute_error = raw_cost_cap
                                emission_cost += absolute_error / raw_cost_scale
                            elif observation_model == 1:
                                emission_cost += 0.5 * normalized_error * normalized_error
                            else:
                                if absolute_error > raw_cost_cap:
                                    absolute_error = raw_cost_cap
                                normalized_error = absolute_error / raw_cost_scale
                                emission_cost += (
                                    0.5
                                    * (student_dof + 1.0)
                                    * math.log1p(
                                        normalized_error * normalized_error / student_dof
                                    )
                                )

                    slope_delta = candidate_local_slope - previous_slope
                    smooth_cost = 0.5 * (slope_delta / smooth_sigma) ** 2
                    if use_turn_mixture == 1:
                        turn_cost = turn_penalty + 0.5 * (slope_delta / turn_sigma) ** 2
                        transition_cost = min(smooth_cost, turn_cost)
                    else:
                        transition_cost = smooth_cost

                    absolute_slope_cost = (
                        0.5
                        * absolute_slope_weight
                        * (candidate_local_slope / absolute_slope_sigma) ** 2
                    )
                    candidate_endpoint_surface = (
                        start_surface + candidate_local_slope * duration
                    )
                    candidate_endpoint_tvt = (
                        candidate_endpoint_surface - z_bounds[end_endpoint]
                    )
                    sequence_cost = 0.0
                    if sequence_derivative_weight != 0.0:
                        sequence_cost += sequence_derivative_weight * _lookup_cost_volume(
                            sequence_derivative,
                            end_endpoint,
                            slope_index,
                            candidate_endpoint_tvt,
                            sequence_tvt_min,
                            sequence_tvt_step,
                        )
                    if sequence_motif_weight != 0.0:
                        sequence_cost += sequence_motif_weight * _lookup_cost_volume(
                            sequence_motif,
                            end_endpoint,
                            slope_index,
                            candidate_endpoint_tvt,
                            sequence_tvt_min,
                            sequence_tvt_step,
                        )
                    if sequence_reverse_weight != 0.0:
                        sequence_cost += sequence_reverse_weight * _lookup_cost_volume(
                            sequence_reverse,
                            end_endpoint,
                            slope_index,
                            candidate_endpoint_tvt,
                            sequence_tvt_min,
                            sequence_tvt_step,
                        )
                    total_cost = (
                        previous_cost
                        + emission_weight * emission_cost
                        + transition_cost
                        + absolute_slope_cost
                        + sequence_cost
                        + segment_penalty
                    )

                    candidate_surface[n_candidates] = candidate_endpoint_surface
                    candidate_slope[n_candidates] = slope_index
                    candidate_cost[n_candidates] = total_cost
                    candidate_parent_endpoint[n_candidates] = start_endpoint
                    candidate_parent_index[n_candidates] = parent_q
                    candidate_segment_slope[n_candidates] = slope_index
                    n_candidates += 1

            order = np.argsort(candidate_cost[:n_candidates])
            n_selected = 0
            selected_tvt_bin = np.empty(beam, np.int64)
            selected_slope = np.empty(beam, np.int32)
            selected_count = np.zeros(beam, np.int32)

            for order_index in range(n_candidates):
                candidate_index = order[order_index]
                endpoint_tvt = (
                    candidate_surface[candidate_index] - z_bounds[end_endpoint]
                )
                endpoint_bin = int(round(endpoint_tvt / tvt_bin))
                slope_index = candidate_slope[candidate_index]

                duplicate_index = -1
                for selected_index in range(n_selected):
                    if (
                        selected_tvt_bin[selected_index] == endpoint_bin
                        and selected_slope[selected_index] == slope_index
                    ):
                        duplicate_index = selected_index
                        break

                if duplicate_index >= 0:
                    if selected_count[duplicate_index] >= max_same:
                        continue
                    selected_count[duplicate_index] += 1
                else:
                    selected_tvt_bin[n_selected] = endpoint_bin
                    selected_slope[n_selected] = slope_index
                    selected_count[n_selected] = 1

                state_surface[end_endpoint, n_selected] = candidate_surface[candidate_index]
                state_slope[end_endpoint, n_selected] = candidate_slope[candidate_index]
                state_cost[end_endpoint, n_selected] = candidate_cost[candidate_index]
                parent_endpoint[end_endpoint, n_selected] = candidate_parent_endpoint[candidate_index]
                parent_index[end_endpoint, n_selected] = candidate_parent_index[candidate_index]
                segment_slope[end_endpoint, n_selected] = candidate_segment_slope[candidate_index]
                n_selected += 1
                if n_selected >= beam:
                    break

            counts[end_endpoint] = n_selected

    return (
        state_surface,
        state_slope,
        state_cost,
        parent_endpoint,
        parent_index,
        segment_slope,
        counts,
    )


def _mode_cluster_mean(
    paths: np.ndarray,
    costs: np.ndarray,
    temperature: float,
    radius: float,
) -> tuple[np.ndarray, int, float]:
    order = np.argsort(costs)
    weights = np.exp(-(costs - costs.min()) / max(temperature, 1e-6))
    weights /= weights.sum()
    assigned = np.zeros(len(costs), dtype=bool)
    clusters: list[list[int]] = []

    for seed_index in order:
        if assigned[seed_index]:
            continue
        members: list[int] = []
        for candidate_index in order:
            if assigned[candidate_index]:
                continue
            rms = float(
                np.sqrt(np.mean((paths[candidate_index] - paths[seed_index]) ** 2))
            )
            if rms <= radius:
                assigned[candidate_index] = True
                members.append(int(candidate_index))
        clusters.append(members)

    masses = np.asarray([weights[members].sum() for members in clusters])
    best_cluster = int(np.argmax(masses))
    members = clusters[best_cluster]
    member_weights = weights[members]
    member_weights /= member_weights.sum()
    mean_path = np.sum(paths[members] * member_weights[:, None], axis=0)
    return mean_path, len(clusters), float(masses[best_cluster])


def _load_observables(horizontal_path: Path) -> pd.DataFrame:
    required = ["MD", "Z", "TVT_input", "GR"]
    header = pd.read_csv(horizontal_path, nrows=0).columns
    missing = [column for column in required if column not in header]
    if missing:
        raise ValueError(f"{horizontal_path}: missing columns {missing}")
    return (
        pd.read_csv(horizontal_path, usecols=required)
        .sort_values("MD")
        .reset_index(drop=True)
    )


def decode_well(
    horizontal_path: str | Path,
    typewell_path: str | Path | None = None,
    config: OrientedConfig | None = None,
    return_ensemble: bool = False,
) -> dict[str, Any]:
    """Decode one well using only test-time observables.

    Parameters
    ----------
    horizontal_path:
        CSV containing at least ``MD, Z, TVT_input, GR``.  A hidden ``TVT``
        column, if present, is deliberately ignored.
    typewell_path:
        Paired typewell CSV.  Defaults to ``<stem>__typewell.csv`` next to the
        horizontal file.
    config:
        Decoder configuration.
    """

    cfg = config or OrientedConfig()
    horizontal_path = Path(horizontal_path)
    stem = horizontal_path.name.split("__")[0]
    if typewell_path is None:
        typewell_path = horizontal_path.with_name(f"{stem}__typewell.csv")
    typewell_path = Path(typewell_path)

    well = _load_observables(horizontal_path)
    md = well["MD"].to_numpy(float)
    z = well["Z"].to_numpy(float)
    tvt_input = well["TVT_input"].to_numpy(float)
    gr = well["GR"].to_numpy(float)

    known = np.flatnonzero(np.isfinite(tvt_input))
    if len(known) < 20 or known[-1] >= len(well) - 2:
        return {
            "stem": stem,
            "config": cfg.name,
            "error": "invalid visible-prefix boundary",
        }

    prefix_end = int(known[-1])
    initial_tvt = float(tvt_input[prefix_end])
    initial_surface = initial_tvt + z[prefix_end]

    typewell = (
        pd.read_csv(typewell_path, usecols=["TVT", "GR"])
        .dropna()
        .sort_values("TVT")
    )
    typewell_tvt = typewell["TVT"].to_numpy(float)
    typewell_gr = typewell["GR"].to_numpy(float)
    if len(typewell_tvt) < 2:
        return {"stem": stem, "config": cfg.name, "error": "invalid typewell"}

    typewell_step = float(np.median(np.diff(typewell_tvt)))
    if typewell_step <= 0:
        return {"stem": stem, "config": cfg.name, "error": "invalid typewell axis"}


    typewell_gr = _apply_typewell_response(
        typewell_gr,
        typewell_step,
        cfg.typewell_response,
        cfg.typewell_response_ft,
    )
    gr = _transform_gr(gr, cfg.gr_transform, cfg.gr_asinh_scale)
    typewell_gr = _transform_gr(
        typewell_gr, cfg.gr_transform, cfg.gr_asinh_scale
    )
    typewell_gr, gr_cal_gain, gr_cal_offset = _calibrate_typewell_gr(
        gr,
        tvt_input,
        prefix_end,
        typewell_tvt,
        typewell_gr,
        cfg,
    )
    self_gr, self_valid, self_lateral_start = _build_self_bank(
        typewell_tvt,
        md[: prefix_end + 1],
        z[: prefix_end + 1],
        tvt_input[: prefix_end + 1],
        gr[: prefix_end + 1],
        cfg,
    )

    obs_md, obs_gr, obs_z = _bin_observations(
        md,
        gr,
        z,
        md[prefix_end],
        md[-1] + 1e-6,
        cfg.obs_step,
    )
    suffix_observations = obs_md > md[prefix_end] + 1e-6
    obs_md = obs_md[suffix_observations]
    obs_gr = obs_gr[suffix_observations]
    obs_z = obs_z[suffix_observations]

    bounds_list = list(
        np.arange(
            md[prefix_end],
            md[-1] + cfg.endpoint_step * 0.5,
            cfg.endpoint_step,
            dtype=float,
        )
    )
    if bounds_list[-1] < md[-1] - 1e-6:
        bounds_list.append(float(md[-1]))
    else:
        bounds_list[-1] = float(md[-1])
    bounds = np.asarray(bounds_list, dtype=float)
    z_bounds = np.interp(bounds, md, z)

    obs_left = np.searchsorted(obs_md, bounds, side="right").astype(np.int32)
    obs_right = obs_left.copy()
    slopes = np.asarray(cfg.slope_grid, dtype=float)
    lengths = np.asarray(cfg.lengths_blocks, dtype=np.int32)

    initial_slope = _prefix_surface_slope(
        md,
        z,
        tvt_input,
        prefix_end,
        cfg.prefix_slope_lookback,
    )
    initial_slope_index = int(np.argmin(np.abs(slopes - initial_slope)))

    use_sequence_volume = (
        cfg.sequence_derivative_weight != 0.0
        or cfg.sequence_motif_weight != 0.0
        or cfg.sequence_reverse_weight != 0.0
    )
    sequence_tvt_min = float(typewell_tvt[0])
    sequence_tvt_step = float(cfg.sequence_tvt_step)
    if use_sequence_volume:
        n_sequence_tvt = (
            int(math.ceil((typewell_tvt[-1] - typewell_tvt[0]) / sequence_tvt_step))
            + 1
        )
        sequence_derivative, sequence_motif, sequence_reverse = (
            _build_sequence_cost_volume(
                bounds,
                z_bounds,
                obs_md,
                obs_gr,
                obs_z,
                obs_right,
                typewell_gr.astype(float),
                float(typewell_tvt[0]),
                typewell_step,
                slopes,
                sequence_tvt_min,
                sequence_tvt_step,
                n_sequence_tvt,
                np.asarray(cfg.sequence_window_points, dtype=np.int32),
                float(cfg.sequence_derivative_gamma),
            )
        )
    else:
        sequence_derivative = np.zeros((1, 1, 2), dtype=np.float32)
        sequence_motif = np.zeros((1, 1, 2), dtype=np.float32)
        sequence_reverse = np.zeros((1, 1, 2), dtype=np.float32)

    states = _decode_core(
        bounds,
        z_bounds,
        obs_md,
        obs_gr,
        obs_z,
        obs_left,
        obs_right,
        typewell_gr.astype(float),
        self_gr,
        self_valid,
        float(np.clip(cfg.self_weight, 0.0, 1.0)),
        1 if cfg.self_directional else 0,
        float(cfg.self_direction_threshold),
        float(typewell_tvt[0]),
        typewell_step,
        slopes,
        lengths,
        int(cfg.beam),
        float(cfg.raw_cost_cap),
        float(cfg.raw_cost_scale),
        {"clipped_l1": 0, "gaussian": 1, "student": 2}[cfg.observation_model],
        float(cfg.student_dof),
        float(cfg.emission_weight),
        float(cfg.smooth_sigma),
        float(cfg.turn_sigma),
        float(cfg.turn_penalty),
        1 if cfg.use_turn_mixture else 0,
        float(cfg.absolute_slope_sigma),
        float(cfg.absolute_slope_weight),
        float(cfg.segment_penalty),
        float(cfg.missing_cost),
        sequence_derivative,
        sequence_motif,
        sequence_reverse,
        sequence_tvt_min,
        sequence_tvt_step,
        float(cfg.sequence_derivative_weight),
        float(cfg.sequence_motif_weight),
        float(cfg.sequence_reverse_weight),
        float(cfg.tvt_bin),
        int(cfg.max_same),
        float(cfg.min_final_fraction),
        float(initial_surface),
        int(initial_slope_index),
    )

    (
        _state_surface,
        _state_slope,
        state_cost,
        parent_endpoint,
        parent_index,
        segment_slope,
        counts,
    ) = states

    n_endpoints = len(bounds)
    n_final = int(counts[-1])
    if n_final == 0:
        return {"stem": stem, "config": cfg.name, "error": "no final path"}

    surface_paths = np.zeros((n_final, n_endpoints), dtype=float)
    segment_counts = np.zeros(n_final, dtype=int)
    turn_counts = np.zeros(n_final, dtype=int)

    for final_index in range(n_final):
        segments: list[tuple[int, int, float]] = []
        endpoint = n_endpoints - 1
        beam_index = final_index
        while endpoint > 0:
            previous_endpoint = int(parent_endpoint[endpoint, beam_index])
            if previous_endpoint < 0:
                break
            slope_index = int(segment_slope[endpoint, beam_index])
            segments.append(
                (previous_endpoint, endpoint, float(slopes[slope_index]))
            )
            beam_index = int(parent_index[endpoint, beam_index])
            endpoint = previous_endpoint
        segments.reverse()

        current_surface = initial_surface
        previous_slope: float | None = None
        surface_paths[final_index, 0] = initial_surface
        for start_endpoint, end_endpoint, slope in segments:
            local_md = bounds[start_endpoint : end_endpoint + 1]
            surface_paths[final_index, start_endpoint : end_endpoint + 1] = (
                current_surface + slope * (local_md - bounds[start_endpoint])
            )
            current_surface += slope * (
                bounds[end_endpoint] - bounds[start_endpoint]
            )
            if previous_slope is not None and abs(slope - previous_slope) > 0.015:
                turn_counts[final_index] += 1
            previous_slope = slope
        segment_counts[final_index] = len(segments)

    tvt_paths = surface_paths - z_bounds[None, :]
    costs = state_cost[-1, :n_final]
    map_index = int(np.argmin(costs))
    map_bounds = tvt_paths[map_index]

    delta_cost = costs - costs.min()
    posterior_weights = np.exp(
        -delta_cost / max(cfg.posterior_temp, 1e-6)
    )
    posterior_weights /= posterior_weights.sum()
    mean_bounds = np.sum(tvt_paths * posterior_weights[:, None], axis=0)

    mode_bounds, n_modes, dominant_mode_mass = _mode_cluster_mean(
        tvt_paths,
        costs,
        cfg.posterior_temp,
        cfg.mode_radius_ft,
    )

    result = {
        "stem": stem,
        "config": cfg.name,
        "gr_transform": cfg.gr_transform,
        "gr_calibration": cfg.gr_calibration,
        "typewell_response": cfg.typewell_response,
        "typewell_response_ft": cfg.typewell_response_ft,
        "self_weight": cfg.self_weight,
        "self_prefix_mode": cfg.self_prefix_mode,
        "self_directional": cfg.self_directional,
        "self_coverage": float(np.mean(self_valid[0])),
        "self_primary_coverage": float(np.mean(self_valid[1])),
        "self_directional_coverage": float(np.mean(self_valid[2:])),
        "self_lateral_start_index": int(self_lateral_start),
        "gr_cal_gain": gr_cal_gain,
        "gr_cal_offset": gr_cal_offset,
        "config_json": cfg.to_json(),
        "prefix_end_index": prefix_end,
        "prefix_end_md": float(md[prefix_end]),
        "n_suffix_rows": int(len(md) - prefix_end - 1),
        "prefix_surface_slope": initial_slope,
        "final_cost": float(costs.min()),
        "beam_spread": float(np.std(tvt_paths[:, -1])),
        "n_modes": int(n_modes),
        "dominant_mode_mass": dominant_mode_mass,
        "map_segments": int(segment_counts[map_index]),
        "map_turns": int(turn_counts[map_index]),
        "posterior_mean_turns": float(np.sum(turn_counts * posterior_weights)),
        "boundaries": json.dumps(bounds.tolist()),
        "map_tvt_boundaries": json.dumps(map_bounds.tolist()),
        "mean_tvt_boundaries": json.dumps(mean_bounds.tolist()),
        "mode_tvt_boundaries": json.dumps(mode_bounds.tolist()),
        "beam_costs": json.dumps(costs.tolist()),
    }
    if return_ensemble:
        result["_ensemble_bounds"] = bounds
        result["_ensemble_paths"] = tvt_paths
        result["_ensemble_costs"] = costs
        result["_ensemble_temperature"] = float(cfg.posterior_temp)
    return result


def score_prediction(
    horizontal_path: str | Path,
    decoded_row: Mapping[str, Any] | pd.Series,
    value_column: str = "mean_tvt_boundaries",
) -> dict[str, float | int]:
    """Score a decoded path when truth is available.

    This function is intentionally separate from :func:`decode_well`.
    """

    horizontal_path = Path(horizontal_path)
    header = pd.read_csv(horizontal_path, nrows=0).columns
    if "TVT" not in header:
        raise ValueError(f"{horizontal_path} has no TVT truth column")
    well = (
        pd.read_csv(horizontal_path, usecols=["MD", "TVT", "TVT_input"])
        .sort_values("MD")
        .reset_index(drop=True)
    )
    prefix_end = int(np.flatnonzero(well["TVT_input"].notna().to_numpy())[-1])
    md = well["MD"].to_numpy(float)[prefix_end + 1 :]
    truth = well["TVT"].to_numpy(float)[prefix_end + 1 :]
    prediction = interpolate_boundary_prediction(md, decoded_row, value_column)
    error = prediction - truth
    return {
        "nrows": int(len(truth)),
        "sse": float(np.sum(error**2)),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "endpoint_error": float(error[-1]),
        "prediction_range": float(np.ptp(prediction)),
        "truth_range": float(np.ptp(truth)),
    }


def score_all_outputs(
    horizontal_path: str | Path,
    decoded_row: Mapping[str, Any] | pd.Series,
) -> dict[str, float | int]:
    result: dict[str, float | int] = {}
    for output_name in ("map", "mean", "mode"):
        metrics = score_prediction(
            horizontal_path,
            decoded_row,
            f"{output_name}_tvt_boundaries",
        )
        for key, value in metrics.items():
            result[f"{output_name}_{key}"] = value
    result["nrows"] = result["mean_nrows"]
    return result
