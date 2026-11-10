# Ochir Dorzhiev
"""Inference-only formation branch used by the v6 + formation Kaggle notebook."""

from __future__ import annotations

import base64
import bz2
import hashlib
import types
from pathlib import Path
from types import SimpleNamespace

import lightgbm as lgb
import numpy as np
import pandas as pd
from numba import njit


def decode_booster(encoded: str) -> lgb.Booster:
    text = bz2.decompress(base64.b64decode(encoded)).decode("utf-8")
    return lgb.Booster(model_str=text)


@njit(cache=False)
def _interp_regular(grid, value, minimum, step):
    index = int((value - minimum) / step)
    if index < 0:
        return grid[0]
    last = len(grid) - 1
    if index >= last:
        return grid[last]
    fraction = (value - minimum) / step - index
    return grid[index] * (1.0 - fraction) + grid[index + 1] * fraction


@njit(cache=False, nogil=True)
def _pf_all_seeds(
    md,
    z,
    gr,
    type_gr,
    type_min,
    type_step,
    gr_sigma,
    start_u,
    initial_rate,
    particles,
    seeds,
    seed_base,
):
    momentum, velocity_noise, position_noise = 0.998, 0.002, 0.005
    rough_position, rough_rate, resample_fraction = 0.1, 0.001, 0.5
    count = len(md)
    predictions = np.empty((seeds, count))
    likelihoods = np.empty(seeds)
    type_max = type_min + len(type_gr) * type_step
    for seed_index in range(seeds):
        np.random.seed(seed_base + seed_index)
        position = np.empty(particles)
        rate = np.empty(particles)
        weight = np.ones(particles) / particles
        for particle in range(particles):
            position[particle] = start_u + 4.5 * np.random.randn()
            rate[particle] = initial_rate + 0.01 * np.random.randn()
        log_likelihood = 0.0
        previous_md = md[0] - 1.0
        for row in range(count):
            delta_md = max(md[row] - previous_md, 1.0)
            for particle in range(particles):
                rate[particle] = (
                    momentum * rate[particle] + velocity_noise * np.random.randn()
                )
                position[particle] += (
                    rate[particle] * delta_md + position_noise * np.random.randn()
                )
                tvt = position[particle] - z[row]
                tvt = min(max(tvt, type_min - 100.0), type_max + 100.0)
                position[particle] = tvt + z[row]
            if not np.isnan(gr[row]):
                average_likelihood = 0.0
                for particle in range(particles):
                    expected = _interp_regular(
                        type_gr, position[particle] - z[row], type_min, type_step
                    )
                    residual = (gr[row] - expected) / gr_sigma
                    likelihood = np.exp(-0.5 * min(residual * residual, 600.0))
                    likelihood = max(likelihood, 1e-300)
                    average_likelihood += weight[particle] * likelihood
                    weight[particle] *= likelihood
                average_likelihood = max(average_likelihood, 1e-300)
                log_likelihood += np.log(average_likelihood)
                total = np.sum(weight)
                if total > 0.0:
                    weight /= total
                else:
                    weight[:] = 1.0 / particles
            effective = 1.0 / np.sum(weight * weight)
            if effective < resample_fraction * particles:
                cumulative = np.cumsum(weight)
                offset = np.random.uniform(0.0, 1.0 / particles)
                new_position = np.empty(particles)
                new_rate = np.empty(particles)
                source = 0
                for particle in range(particles):
                    target = offset + particle / particles
                    while source < particles - 1 and cumulative[source] < target:
                        source += 1
                    new_position[particle] = (
                        position[source] + rough_position * np.random.randn()
                    )
                    new_rate[particle] = rate[source] + rough_rate * np.random.randn()
                position, rate = new_position, new_rate
                weight[:] = 1.0 / particles
            estimate = 0.0
            for particle in range(particles):
                estimate += weight[particle] * (position[particle] - z[row])
            predictions[seed_index, row] = estimate
            previous_md = md[row]
        likelihoods[seed_index] = log_likelihood
    return predictions, likelihoods


def _regular_typewell(type_tvt, type_gr, step=0.2):
    order = np.argsort(type_tvt)
    type_tvt, type_gr = type_tvt[order], type_gr[order]
    valid = np.isfinite(type_tvt) & np.isfinite(type_gr)
    type_tvt, type_gr = type_tvt[valid], type_gr[valid]
    grid = np.arange(type_tvt.min(), type_tvt.max() + step, step)
    return np.interp(grid, type_tvt, type_gr).astype(np.float64), float(grid[0]), step


def likelihood_pf_paths(
    horizontal: pd.DataFrame,
    typewell: pd.DataFrame,
    particles: int = 200,
    seeds: int = 16,
    max_points: int = 200,
    seed_base: int = 0,
) -> list[np.ndarray]:
    known = horizontal[horizontal.TVT_input.notna()]
    evaluation = horizontal[horizontal.TVT_input.isna()]
    if len(evaluation) <= max_points:
        positions = np.arange(len(evaluation), dtype=np.int32)
    else:
        positions = np.unique(
            np.rint(np.linspace(0, len(evaluation) - 1, max_points)).astype(np.int32)
        )
    reduced = pd.concat([known, evaluation.iloc[positions]], ignore_index=True)
    reduced_eval = reduced[reduced.TVT_input.isna()]

    typewell = typewell.sort_values("TVT")
    type_tvt = pd.to_numeric(typewell.TVT, errors="coerce").to_numpy(float)
    type_gr = (
        pd.to_numeric(typewell.GR, errors="coerce")
        .interpolate(limit_direction="both")
        .fillna(pd.to_numeric(typewell.GR, errors="coerce").mean())
        .to_numpy(float)
    )
    last = known.iloc[-1]
    start_u = float(last.TVT_input) + float(last.Z)
    expected_known = np.interp(known.TVT_input.to_numpy(float), type_tvt, type_gr)
    gr_sigma = float(
        np.clip(
            np.nanstd(pd.to_numeric(known.GR, errors="coerce") - expected_known),
            10.0,
            60.0,
        )
    )
    tail = known.tail(30)
    delta_tvt = np.diff(tail.TVT_input.to_numpy(float))
    delta_z = np.diff(tail.Z.to_numpy(float))
    delta_md = np.diff(tail.MD.to_numpy(float))
    valid = delta_md > 0
    initial_rate = (
        float(np.median((delta_tvt + delta_z)[valid] / delta_md[valid]))
        if valid.sum() >= 3
        else 0.0
    )
    regular_gr, type_min, type_step = _regular_typewell(type_tvt, type_gr)
    observed = (
        pd.to_numeric(reduced.GR, errors="coerce")
        .interpolate(limit_direction="both")
        .fillna(float(np.nanmean(type_gr)))
        .loc[reduced_eval.index]
        .to_numpy(float)
    )
    predictions, likelihoods = _pf_all_seeds(
        reduced_eval.MD.to_numpy(float),
        reduced_eval.Z.to_numpy(float),
        observed,
        regular_gr,
        type_min,
        type_step,
        gr_sigma,
        start_u,
        initial_rate,
        particles,
        seeds,
        seed_base,
    )
    centered = likelihoods - likelihoods.max()
    reduced_paths = []
    for scale in (3.0, 5.0, 8.0, 12.0):
        weights = np.exp(centered / scale)
        weights /= weights.sum()
        reduced_paths.append(weights @ predictions)
    reduced_paths.append(predictions.mean(axis=0))
    full_md = evaluation.MD.to_numpy(float)
    reduced_md = reduced_eval.MD.to_numpy(float)
    last_tvt = float(known.TVT_input.iloc[-1])
    output = []
    for path in reduced_paths:
        valid = np.isfinite(path) & np.isfinite(reduced_md)
        if valid.sum() >= 2:
            output.append(np.interp(full_md, reduced_md[valid], path[valid]))
        elif valid.any():
            output.append(np.full(len(full_md), float(path[valid][0])))
        else:
            output.append(np.full(len(full_md), last_tvt))
    return output


def load_test_query(root: Path, well: str, formation_module) -> dict:
    horizontal_path = root / f"{well}__horizontal_well.csv"
    typewell_path = root / f"{well}__typewell.csv"
    horizontal = pd.read_csv(horizontal_path)
    typewell = pd.read_csv(typewell_path)
    columns = ["MD", "X", "Y", "Z", "TVT_input", "GR"]
    arrays = {
        column.lower(): pd.to_numeric(horizontal[column], errors="coerce").to_numpy(float)
        for column in columns
    }


    arrays["tvt"] = np.full(len(horizontal), np.nan, dtype=float)
    known = np.flatnonzero(np.isfinite(arrays["tvt_input"]))
    ps = int(known[-1])
    xy = np.c_[arrays["x"], arrays["y"]]
    direction = formation_module.principal_direction(xy[ps:])
    type_tvt, type_gr = formation_module.deduplicate_curve(
        pd.to_numeric(typewell.TVT, errors="coerce").to_numpy(float),
        pd.to_numeric(typewell.GR, errors="coerce").to_numpy(float),
    )
    prefix_valid = np.isfinite(arrays["gr"][known])
    expected = np.interp(
        arrays["tvt_input"][known][prefix_valid], type_tvt, type_gr
    )
    gain, bias = formation_module.robust_affine(
        expected, arrays["gr"][known][prefix_valid]
    )
    pseudo_valid = np.isfinite(arrays["gr"][known])
    pseudo_tvt, pseudo_gr = formation_module.deduplicate_curve(
        arrays["tvt_input"][known][pseudo_valid], arrays["gr"][known][pseudo_valid]
    )
    return {
        "wid": well,
        "hash": formation_module.typewell_hash(typewell_path),
        **arrays,
        "xy": xy,
        "known": known,
        "ps": ps,
        "direction": direction,
        "centroid": np.nanmedian(xy[ps:], axis=0),
        "type_tvt": type_tvt,
        "type_gr": type_gr,
        "gr_gain": gain,
        "gr_bias": bias,
        "gr_scale": formation_module.robust_scale(arrays["gr"][known]),
        "gr_smooth": formation_module.smooth(arrays["gr"]),
        "pseudo_tvt": pseudo_tvt,
        "pseudo_gr": pseudo_gr,
    }


def heuristic_score(features: dict) -> float:
    def value(name, fallback):
        result = float(features.get(name, fallback))
        return result if np.isfinite(result) else fallback

    return (
        value("prefix_rmse", 1e6)
        + 0.5 * value("perp_distance_kft", 1e6)
        + 5.0 * value("extrap_fraction", 1.0)
        + 0.1 * value("angle_deg", 90.0)
    )


def normalize_one_well(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, float)
    return (values - np.mean(values)) / max(float(np.std(values)), 1e-6)


def build_feature_frame(
    well,
    horizontal,
    typewell,
    hidden_rows,
    candidates,
    image,
    analog_metadata,
    artifacts,
    router_module,
):
    flat = candidates[0]
    likpf_indices = list(range(1, 6))
    analog_indices = list(range(6, 26))
    context, prefix_mask = router_module.raw_context(
        horizontal, typewell, hidden_rows
    )
    z_eval = pd.to_numeric(
        horizontal.iloc[hidden_rows]["Z"], errors="coerce"
    ).to_numpy(float)
    last_index = np.flatnonzero(prefix_mask)[-1]
    last_u = (
        float(horizontal.TVT_input.iloc[last_index])
        + float(horizontal.Z.iloc[last_index])
    )

    family_indices = {
        "analog": analog_indices,
        "likPF": likpf_indices,
    }
    family_aggregates = {}
    for family, indices in family_indices.items():
        descriptors, paths = [], []
        for index in indices:
            descriptor = router_module.gr_descriptors(image[index])
            descriptor.update(
                router_module.path_descriptors(
                    candidates[index], flat, z_eval, last_u
                )
            )
            descriptors.append(descriptor)
            paths.append(candidates[index])
        aggregate = router_module.aggregate_family(descriptors, paths)
        prefix = "formation_analog" if family == "analog" else "likPF"
        family_aggregates.update(
            {f"fam_{prefix}_{key}": value for key, value in aggregate.items()}
        )

    analog_median = np.nanmedian(candidates[analog_indices], axis=0)
    likpf_median = np.nanmedian(candidates[likpf_indices], axis=0)
    sampled_analog = router_module.resample(analog_median)
    sampled_likpf = router_module.resample(likpf_median)
    between = float(np.sqrt(np.mean((sampled_analog - sampled_likpf) ** 2)))

    rows = []
    for index in likpf_indices + analog_indices:
        family = "analog" if index in analog_indices else "likPF"
        family_index = (
            analog_indices.index(index)
            if family == "analog"
            else likpf_indices.index(index)
        )
        descriptor = router_module.gr_descriptors(image[index])
        descriptor.update(
            router_module.path_descriptors(candidates[index], flat, z_eval, last_u)
        )
        sampled = router_module.resample(candidates[index])
        analog_features = {
            f"analog_{column}": np.nan
            for column in artifacts["analog_metadata_columns"]
        }
        if family == "analog":
            metadata = analog_metadata[family_index]
            analog_features.update(
                {
                    f"analog_{column}": metadata.get(column, np.nan)
                    for column in artifacts["analog_metadata_columns"]
                }
            )
        candidate_global_index = 22 + family_index if family == "analog" else 17 + family_index
        rows.append(
            {
                "well": well,
                "fold": -1,
                "n_rows": len(hidden_rows),
                "family": family,
                "candidate_name": (
                    f"analog_{family_index:02d}"
                    if family == "analog"
                    else ("likpf_mean" if family_index == 4 else f"likpf{(3, 5, 8, 12)[family_index]}")
                ),
                "candidate_index": candidate_global_index,
                "family_index": family_index,
                **context,
                **family_aggregates,
                **descriptor,
                "distance_to_analog_median": float(
                    np.sqrt(np.mean((sampled - sampled_analog) ** 2))
                ),
                "distance_to_likpf_median": float(
                    np.sqrt(np.mean((sampled - sampled_likpf) ** 2))
                ),
                "distance_between_family_medians": between,
                "candidate_global_index": float(candidate_global_index),
                "candidate_family_index": float(family_index),
                "is_analog": float(family == "analog"),
                "is_likpf": float(family == "likPF"),
                **analog_features,
            }
        )
    return pd.DataFrame(rows).sort_values("candidate_index").reset_index(drop=True)


def rich_matrix(frame, columns):
    frame = frame.copy()
    additions = {}
    for column in columns:
        if column.startswith("relz__"):
            base = column[len("relz__") :]
            values = pd.to_numeric(frame.get(base), errors="coerce")
            additions[column] = (
                (values - values.mean()) / max(float(values.std()), 1e-5)
            ).to_numpy(float)
        elif column.startswith("rank__"):
            base = column[len("rank__") :]
            values = pd.to_numeric(frame.get(base), errors="coerce")
            additions[column] = values.rank(pct=True).to_numpy(float)
    if additions:
        frame = pd.concat(
            [frame, pd.DataFrame(additions, index=frame.index)], axis=1
        )
    return (
        frame.reindex(columns=columns)
        .replace([np.inf, -np.inf], np.nan)
        .astype(np.float32)
    )


def predict_formation_well(
    path,
    test_dir,
    sources,
    artifacts,
    models,
    formation_module,
    render_module,
    router_module,
):
    path = Path(path)
    well = path.name.split("__", 1)[0]
    horizontal = pd.read_csv(path)
    typewell = pd.read_csv(test_dir / f"{well}__typewell.csv")
    query = load_test_query(test_dir, well, formation_module)
    generation = SimpleNamespace(
        max_angle=25.0,
        max_candidates=80,
        prefix_window=512,
        along_weight=0.15,
        score_tail=False,
    )
    generated = formation_module.candidates_for(query, sources, generation)
    generated.sort(key=lambda item: (heuristic_score(item["features"]), item["neighbor"]))
    generated = generated[:20]

    hidden_rows = np.flatnonzero(horizontal.TVT_input.isna().to_numpy())
    last_tvt = float(horizontal.loc[horizontal.TVT_input.notna(), "TVT_input"].iloc[-1])
    flat = np.full(len(hidden_rows), last_tvt, dtype=np.float64)
    analog_paths, analog_metadata = [], []
    for item in generated:
        path_values = np.asarray(item["prediction"], float)
        path_values = np.clip(path_values, flat - 120.0, flat + 120.0)
        analog_paths.append(path_values)
        analog_metadata.append(
            {"hscore": heuristic_score(item["features"]), **item["features"]}
        )
    while len(analog_paths) < 20:
        analog_paths.append(flat.copy() if not analog_paths else analog_paths[-1].copy())
        analog_metadata.append({})

    seed_base = int(hashlib.md5(well.encode()).hexdigest()[:8], 16) % 1_000_000_000
    likpf_paths = likelihood_pf_paths(
        horizontal,
        typewell,
        particles=200,
        seeds=16,
        max_points=200,
        seed_base=seed_base,
    )
    candidates = np.vstack([flat, *likpf_paths, *analog_paths]).astype(np.float32)
    image = render_module.render_well(
        test_dir, well, hidden_rows.astype(np.int32), candidates, 1024
    ).astype(np.float32)
    features = build_feature_frame(
        well,
        horizontal,
        typewell,
        hidden_rows,
        candidates,
        image,
        analog_metadata,
        artifacts,
        router_module,
    )
    analog = features[features.family.eq("analog")].copy()
    risk = models["rich_raw"].predict(
        rich_matrix(analog, artifacts["rich_columns"])
    )
    risk = np.asarray(risk, float)
    risk = (risk - risk.mean()) / max(float(risk.std(ddof=1)), 1e-5)
    order = np.argsort(risk, kind="stable")[:5]
    logits = -risk[order] / 0.25
    logits -= logits.max()
    weights = np.exp(logits)
    weights /= weights.sum()
    formation = weights @ candidates[6 + order]
    diagnostic = {
        "stem": well,
        "formation_candidates": len(generated),
        "formation_risk_min": float(risk[order[0]]),
        "formation_risk_margin": float(
            np.sort(risk)[1] - np.sort(risk)[0]
        ),
        "formation_weight_max": float(weights.max()),
        "formation_neighbors": "|".join(
            str(generated[index]["neighbor"]) for index in order if index < len(generated)
        ),
    }
    return pd.DataFrame(
        {"stem": well, "row_index": hidden_rows, "formation_tvt": formation}
    ), diagnostic


def run_formation_branch(
    train_dir,
    test_dir,
    test_paths,
    artifacts,
    formation_module,
    render_module,
    router_module,
):
    train_dir, test_dir = Path(train_dir), Path(test_dir)
    models = {"rich_raw": decode_booster(artifacts["rich_raw"])}
    records = {}
    train_ids = sorted(
        path.name.split("__", 1)[0]
        for path in train_dir.glob("*__horizontal_well.csv")
    )
    for index, well in enumerate(train_ids, 1):
        record = formation_module.load_well(train_dir, well)
        if record is not None:
            records[well] = record
        if index % 100 == 0 or index == len(train_ids):
            print(f"Formation sources: {index}/{len(train_ids)}", flush=True)
    sources = list(records.values())
    print(f"Usable formation sources: {len(sources)}", flush=True)

    prediction_parts, diagnostics = [], []
    for index, path in enumerate(test_paths, 1):
        prediction, diagnostic = predict_formation_well(
            path,
            test_dir,
            sources,
            artifacts,
            models,
            formation_module,
            render_module,
            router_module,
        )
        prediction_parts.append(prediction)
        diagnostics.append(diagnostic)
        print(
            f"Formation {index}/{len(test_paths)}: {diagnostic['stem']} "
            f"neighbors={diagnostic['formation_candidates']}",
            flush=True,
        )
    return pd.concat(prediction_parts, ignore_index=True), pd.DataFrame(diagnostics)
