# Ochir Dorzhiev
import argparse
import json
from pathlib import Path

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


FAMILIES = ("formation_analog", "likPF", "PF", "fallback")
TARGET_COLUMNS = {f"risk_{name}" for name in FAMILIES}


def family_for_candidate(name):
    if name.startswith("analog_"):
        return "formation_analog"
    if name.startswith("likpf"):
        return "likPF"
    if name.startswith("pf_"):
        return "PF"
    if name == "flat" or name.startswith("beam_"):
        return "fallback"
    return None


def finite(values):
    values = np.asarray(values, dtype=np.float64)
    return values[np.isfinite(values)]


def safe_stat(values, fn, default=np.nan):
    values = finite(values)
    return float(fn(values)) if values.size else float(default)


def corr(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 8:
        return np.nan
    a = a[mask] - np.mean(a[mask])
    b = b[mask] - np.mean(b[mask])
    den = np.sqrt(np.sum(a * a) * np.sum(b * b))
    return float(np.sum(a * b) / den) if den > 1e-12 else np.nan


def slope(values):
    values = np.asarray(values, dtype=np.float64)
    mask = np.isfinite(values)
    if mask.sum() < 8:
        return np.nan
    x = np.linspace(-1.0, 1.0, len(values))[mask]
    return float(np.polyfit(x, values[mask], 1)[0])


def resample(values, size=256):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return np.full(size, np.nan, dtype=np.float64)
    good = np.isfinite(values)
    if good.sum() == 0:
        return np.full(size, np.nan, dtype=np.float64)
    old_x = np.linspace(0.0, 1.0, len(values))
    new_x = np.linspace(0.0, 1.0, size)
    return np.interp(new_x, old_x[good], values[good])


def tail_slope(frame, value, mask, count=256):
    if value not in frame or "MD" not in frame:
        return np.nan
    x = pd.to_numeric(frame.loc[mask, "MD"], errors="coerce").to_numpy(float)[-count:]
    y = pd.to_numeric(frame.loc[mask, value], errors="coerce").to_numpy(float)[-count:]
    good = np.isfinite(x) & np.isfinite(y)
    if good.sum() < 8 or np.ptp(x[good]) < 1e-6:
        return np.nan
    return float(np.polyfit(x[good] - x[good][0], y[good], 1)[0])


def raw_context(horizontal, typewell, rows):
    rows = np.asarray(rows, dtype=int)
    eval_frame = horizontal.iloc[rows]
    first = int(rows[0])
    prefix_mask = np.zeros(len(horizontal), dtype=bool)
    prefix_mask[:first] = True
    if "TVT_input" in horizontal:
        prefix_mask &= horizontal["TVT_input"].notna().to_numpy()

    out = {
        "ctx_eval_rows": float(len(rows)),
        "ctx_log_eval_rows": float(np.log1p(len(rows))),
        "ctx_prefix_rows": float(prefix_mask.sum()),
        "ctx_ps_fraction": float(first / max(len(horizontal) - 1, 1)),
    }

    for col in ("MD", "X", "Y", "Z", "GR"):
        if col not in eval_frame:
            continue
        values = pd.to_numeric(eval_frame[col], errors="coerce").to_numpy(float)
        good = finite(values)
        out[f"ctx_{col}_valid"] = float(np.isfinite(values).mean())
        out[f"ctx_{col}_mean"] = safe_stat(values, np.mean)
        out[f"ctx_{col}_std"] = safe_stat(values, np.std)
        out[f"ctx_{col}_range"] = safe_stat(values, np.ptp)
        out[f"ctx_{col}_drift"] = float(values[-1] - values[0]) if len(values) and np.isfinite(values[[0, -1]]).all() else np.nan
        out[f"ctx_{col}_slope"] = slope(values)
        if good.size > 2:
            out[f"ctx_{col}_grad_std"] = float(np.std(np.diff(good)))

    if {"X", "Y"}.issubset(eval_frame.columns):
        x = pd.to_numeric(eval_frame["X"], errors="coerce").to_numpy(float)
        y = pd.to_numeric(eval_frame["Y"], errors="coerce").to_numpy(float)
        dx = x[-1] - x[0]
        dy = y[-1] - y[0]
        azimuth = np.arctan2(dy, dx)
        out["ctx_xy_distance"] = float(np.hypot(dx, dy))
        out["ctx_azimuth_sin"] = float(np.sin(azimuth))
        out["ctx_azimuth_cos"] = float(np.cos(azimuth))

    known_tvt = pd.to_numeric(horizontal.get("TVT_input"), errors="coerce").to_numpy(float)
    known_z = pd.to_numeric(horizontal.get("Z"), errors="coerce").to_numpy(float)
    out["ctx_known_tvt_slope"] = tail_slope(horizontal, "TVT_input", prefix_mask)
    out["ctx_known_z_slope"] = tail_slope(horizontal, "Z", prefix_mask)
    if prefix_mask.any():
        known_u = known_tvt + known_z
        md = pd.to_numeric(horizontal["MD"], errors="coerce").to_numpy(float)
        good = prefix_mask & np.isfinite(known_u) & np.isfinite(md)
        if good.sum() >= 8:
            idx = np.flatnonzero(good)[-256:]
            out["ctx_known_u_slope"] = float(np.polyfit(md[idx] - md[idx][0], known_u[idx], 1)[0])
        out["ctx_last_tvt"] = safe_stat(known_tvt[prefix_mask][-1:], np.mean)
        out["ctx_last_z"] = safe_stat(known_z[prefix_mask][-1:], np.mean)
        out["ctx_last_u"] = out["ctx_last_tvt"] + out["ctx_last_z"]

    tw_tvt = pd.to_numeric(typewell.get("TVT"), errors="coerce").to_numpy(float)
    tw_gr = pd.to_numeric(typewell.get("GR"), errors="coerce").to_numpy(float)
    out.update({
        "ctx_tw_rows": float(len(typewell)),
        "ctx_tw_tvt_min": safe_stat(tw_tvt, np.min),
        "ctx_tw_tvt_max": safe_stat(tw_tvt, np.max),
        "ctx_tw_tvt_range": safe_stat(tw_tvt, np.ptp),
        "ctx_tw_gr_mean": safe_stat(tw_gr, np.mean),
        "ctx_tw_gr_std": safe_stat(tw_gr, np.std),
        "ctx_tw_gr_range": safe_stat(tw_gr, np.ptp),
    })
    if np.isfinite(out.get("ctx_last_tvt", np.nan)):
        span = max(out["ctx_tw_tvt_range"], 1e-6)
        out["ctx_last_tvt_position"] = (out["ctx_last_tvt"] - out["ctx_tw_tvt_min"]) / span
        out["ctx_tvt_to_bottom"] = out["ctx_tw_tvt_max"] - out["ctx_last_tvt"]


    return out, prefix_mask


def gr_descriptors(image):
    horizontal = np.asarray(image[0], dtype=np.float64)
    typewell = np.asarray(image[1], dtype=np.float64)
    residual = np.asarray(image[2], dtype=np.float64)
    valid = np.asarray(image[3], dtype=np.float64) > 0.5
    mask = valid & np.isfinite(horizontal) & np.isfinite(typewell) & np.isfinite(residual)
    if mask.sum() < 8:
        return {name: np.nan for name in (
            "gr_mae", "gr_rmse", "gr_bias", "gr_std", "gr_corr", "gr_grad_corr",
            "gr_block_min", "gr_block_mean", "gr_block_std", "gr_block_max", "gr_valid",
        )}
    r = residual[mask]
    block_rmse = []
    for block in np.array_split(np.arange(len(residual)), 8):
        block_mask = mask[block]
        if block_mask.sum() >= 4:
            block_rmse.append(float(np.sqrt(np.mean(residual[block][block_mask] ** 2))))
    return {
        "gr_mae": float(np.mean(np.abs(r))),
        "gr_rmse": float(np.sqrt(np.mean(r * r))),
        "gr_bias": float(np.mean(r)),
        "gr_std": float(np.std(r)),
        "gr_corr": corr(horizontal[mask], typewell[mask]),
        "gr_grad_corr": corr(np.diff(horizontal)[mask[1:] & mask[:-1]], np.diff(typewell)[mask[1:] & mask[:-1]]),
        "gr_block_min": safe_stat(block_rmse, np.min),
        "gr_block_mean": safe_stat(block_rmse, np.mean),
        "gr_block_std": safe_stat(block_rmse, np.std),
        "gr_block_max": safe_stat(block_rmse, np.max),
        "gr_valid": float(mask.mean()),
    }


def path_descriptors(candidate, flat, z_eval, last_u):
    candidate = resample(candidate)
    flat = resample(flat)
    z_eval = resample(z_eval)
    delta = candidate - flat
    u_relative = candidate + z_eval - last_u
    u_shape = u_relative - u_relative[0]
    d2 = np.diff(u_shape, n=2)
    return {
        "path_delta_mean": safe_stat(delta, np.mean),
        "path_delta_std": safe_stat(delta, np.std),
        "path_delta_range": safe_stat(delta, np.ptp),
        "path_delta_end": float(delta[-1]),
        "path_delta_slope": slope(delta),
        "path_u_mean": safe_stat(u_relative, np.mean),
        "path_u_std": safe_stat(u_relative, np.std),
        "path_u_range": safe_stat(u_relative, np.ptp),
        "path_u_end": float(u_relative[-1]),
        "path_u_slope": slope(u_shape),
        "path_u_curvature": safe_stat(np.abs(d2), np.mean),
    }


def aggregate_family(candidate_features, paths):
    result = {}
    feature_names = sorted(candidate_features[0])
    for name in feature_names:
        values = [row[name] for row in candidate_features]
        result[f"{name}_min"] = safe_stat(values, np.min)
        result[f"{name}_median"] = safe_stat(values, np.median)
        result[f"{name}_std"] = safe_stat(values, np.std)
        result[f"{name}_max"] = safe_stat(values, np.max)
    sampled = np.vstack([resample(path) for path in paths])
    dispersion = np.nanstd(sampled, axis=0)
    result["family_size"] = float(len(paths))
    result["family_dispersion_mean"] = safe_stat(dispersion, np.mean)
    result["family_dispersion_max"] = safe_stat(dispersion, np.max)
    result["family_endpoint_std"] = safe_stat(sampled[:, -1], np.std)
    return result


def build_dataset(args):
    pool_root = Path(args.pool_root)
    raw_root = Path(args.train_dir)
    metadata = json.loads((pool_root / "fold0" / "metadata.json").read_text())
    candidate_names = metadata["candidate_names"]
    family_indices = {
        family: [i for i, name in enumerate(candidate_names) if family_for_candidate(name) == family]
        for family in FAMILIES
    }
    records = []
    for fold in range(args.folds):
        fold_dir = pool_root / f"fold{fold}"
        quality_table = pd.read_csv(fold_dir / "candidate_quality.csv")
        quality_table = quality_table[quality_table["fold"] == fold].sort_values("well")
        for item_no, item in enumerate(quality_table.itertuples(index=False), 1):
            well = str(item.well)
            packed = np.load(fold_dir / f"{well}.npz")
            image = packed["x"].astype(np.float32)
            candidates = packed["candidates"].astype(np.float32)
            rows = packed["rows"].astype(int)
            quality = packed["quality"].astype(np.float64)
            horizontal = pd.read_csv(raw_root / f"{well}__horizontal_well.csv")
            typewell = pd.read_csv(raw_root / f"{well}__typewell.csv")
            context, prefix_mask = raw_context(horizontal, typewell, rows)
            z_eval = pd.to_numeric(horizontal.iloc[rows]["Z"], errors="coerce").to_numpy(float)
            if prefix_mask.any():
                last_index = np.flatnonzero(prefix_mask)[-1]
                last_tvt = float(horizontal["TVT_input"].iloc[last_index])
                last_z = float(horizontal["Z"].iloc[last_index])
            else:
                last_tvt = float(candidates[0, 0])
                last_z = float(z_eval[0])
            last_u = last_tvt + last_z
            flat = candidates[candidate_names.index("flat")]
            record = {
                "well": well,
                "fold": fold,
                "n_rows": int(item.n_rows),
                **context,
            }
            for family, indices in family_indices.items():
                family_quality = quality[indices]
                record[f"risk_{family}"] = safe_stat(family_quality, np.min, default=1e6)
                descriptors = []
                paths = []
                for index in indices:
                    descriptor = gr_descriptors(image[index])
                    descriptor.update(path_descriptors(candidates[index], flat, z_eval, last_u))
                    descriptors.append(descriptor)
                    paths.append(candidates[index])
                aggregate = aggregate_family(descriptors, paths)
                record.update({f"fam_{family}_{key}": value for key, value in aggregate.items()})
            records.append(record)
            if item_no % 50 == 0 or item_no == len(quality_table):
                print(f"features fold {fold}: {item_no}/{len(quality_table)}", flush=True)
    dataset = pd.DataFrame(records)
    risks = dataset[[f"risk_{family}" for family in FAMILIES]].to_numpy(float)
    dataset["target"] = np.argmin(risks, axis=1)
    return dataset


def model_features(dataset):
    excluded = {"well", "fold", "n_rows", "target"} | TARGET_COLUMNS
    return [column for column in dataset.columns if column not in excluded]


def wide_matrix(dataset, features):
    return dataset[features].replace([np.inf, -np.inf], np.nan).astype(np.float32)


def fit_classifier(train, valid, features, seed):
    model = lgb.LGBMClassifier(
        objective="multiclass",
        num_class=len(FAMILIES),
        n_estimators=350,
        learning_rate=0.025,
        num_leaves=15,
        min_child_samples=20,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.75,
        reg_alpha=0.1,
        reg_lambda=3.0,
        random_state=seed,
        n_jobs=-1,
        verbosity=-1,
    )
    weights = np.sqrt(train["n_rows"].to_numpy(float) / np.median(train["n_rows"]))
    model.fit(wide_matrix(train, features), train["target"], sample_weight=weights)
    return model.predict_proba(wide_matrix(valid, features))


def fit_risk_regression(train, valid, features, seed):
    predictions = np.zeros((len(valid), len(FAMILIES)), dtype=np.float64)
    weights = np.sqrt(train["n_rows"].to_numpy(float) / np.median(train["n_rows"]))
    for family_index, family in enumerate(FAMILIES):
        target = np.log1p(np.clip(train[f"risk_{family}"].to_numpy(float), 0.0, 80.0))
        model = lgb.LGBMRegressor(
            objective="regression_l1",
            n_estimators=350,
            learning_rate=0.025,
            num_leaves=15,
            min_child_samples=20,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.75,
            reg_alpha=0.1,
            reg_lambda=3.0,
            random_state=seed + family_index,
            n_jobs=-1,
            verbosity=-1,
        )
        model.fit(wide_matrix(train, features), target, sample_weight=weights)
        predictions[:, family_index] = np.expm1(model.predict(wide_matrix(valid, features)))
    return -predictions


def long_matrix(dataset):
    rows = []
    context_columns = [column for column in dataset.columns if column.startswith("ctx_")]
    suffixes = sorted({
        column[len(f"fam_{FAMILIES[0]}_"):]
        for column in dataset.columns
        if column.startswith(f"fam_{FAMILIES[0]}_")
    })
    for _, well in dataset.iterrows():
        family_values = {
            suffix: np.array([well.get(f"fam_{family}_{suffix}", np.nan) for family in FAMILIES], dtype=float)
            for suffix in suffixes
        }
        risks = np.array([well[f"risk_{family}"] for family in FAMILIES], dtype=float)
        order = np.argsort(np.argsort(risks))
        for family_index, family in enumerate(FAMILIES):
            row = {column: well[column] for column in context_columns}
            for suffix, values in family_values.items():
                value = values[family_index]
                row[f"own_{suffix}"] = value
                row[f"relative_{suffix}"] = value - np.nanmedian(values)
            for index, family_name in enumerate(FAMILIES):
                row[f"is_{family_name}"] = float(index == family_index)
            row.update({
                "well": well["well"],
                "fold": int(well["fold"]),
                "n_rows": int(well["n_rows"]),
                "family_index": family_index,
                "relevance": int(len(FAMILIES) - 1 - order[family_index]),
            })
            rows.append(row)
    return pd.DataFrame(rows)


def fit_ranker(train_long, valid_long, seed):
    excluded = {"well", "fold", "n_rows", "family_index", "relevance"}
    features = [column for column in train_long.columns if column not in excluded]
    train_long = train_long.sort_values(["well", "family_index"]).reset_index(drop=True)
    valid_long = valid_long.sort_values(["well", "family_index"]).reset_index(drop=True)
    x_train = train_long[features].replace([np.inf, -np.inf], np.nan).astype(np.float32)
    x_valid = valid_long[features].replace([np.inf, -np.inf], np.nan).astype(np.float32)
    weights = np.sqrt(train_long["n_rows"].to_numpy(float) / np.median(train_long["n_rows"]))
    model = lgb.LGBMRanker(
        objective="lambdarank",
        label_gain=[0, 1, 3, 7],
        n_estimators=350,
        learning_rate=0.025,
        num_leaves=15,
        min_child_samples=20,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.75,
        reg_alpha=0.1,
        reg_lambda=3.0,
        random_state=seed,
        n_jobs=-1,
        verbosity=-1,
    )
    model.fit(x_train, train_long["relevance"], group=[4] * (len(train_long) // 4), sample_weight=weights)
    prediction = model.predict(x_valid).reshape(-1, len(FAMILIES))
    well_order = valid_long["well"].drop_duplicates().tolist()
    return prediction, well_order


def normalize_scores(scores):
    scores = np.asarray(scores, dtype=np.float64)
    center = np.mean(scores, axis=1, keepdims=True)
    scale = np.std(scores, axis=1, keepdims=True)
    return (scores - center) / np.maximum(scale, 1e-6)


def pooled_rmse(risk, weights):
    return float(np.sqrt(np.sum(weights * np.asarray(risk) ** 2) / np.sum(weights)))


def evaluate_method(name, scores, risks, weights, target):
    ranking = np.argsort(-scores, axis=1)
    selected = ranking[:, 0]
    selected_risk = risks[np.arange(len(risks)), selected]
    top2_risk = np.min(np.take_along_axis(risks, ranking[:, :2], axis=1), axis=1)
    oracle = np.min(risks, axis=1)
    return {
        "method": name,
        "selected_pooled": pooled_rmse(selected_risk, weights),
        "top2_pooled": pooled_rmse(top2_risk, weights),
        "accuracy": float(np.mean(selected == target)),
        "weighted_accuracy": float(np.average(selected == target, weights=weights)),
        "median_regret": float(np.median(selected_risk - oracle)),
        "p90_regret": float(np.quantile(selected_risk - oracle, 0.9)),
        "selected": selected,
        "selected_risk": selected_risk,
        "top2_risk": top2_risk,
    }


def plot_results(dataset, metrics, best, output):
    target = dataset["target"].to_numpy(int)
    matrix = confusion_matrix(target, best["selected"], labels=np.arange(len(FAMILIES)), normalize="true")
    oracle = np.min(dataset[[f"risk_{family}" for family in FAMILIES]].to_numpy(float), axis=1)
    regret = best["selected_risk"] - oracle
    summary = pd.DataFrame([{key: value for key, value in item.items() if not isinstance(value, np.ndarray)} for item in metrics])

    figure, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    x = np.arange(len(summary))
    axes[0].bar(x - 0.18, summary["selected_pooled"], width=0.36, label="top-1 family")
    axes[0].bar(x + 0.18, summary["top2_pooled"], width=0.36, label="top-2 family oracle")
    axes[0].axhline(pooled_rmse(oracle, dataset["n_rows"].to_numpy(float)), color="black", linestyle="--", label="perfect family")
    axes[0].set_xticks(x, summary["method"], rotation=20, ha="right")
    axes[0].set_ylabel("Pooled RMSE (ft)")
    axes[0].set_title("Router quality")
    axes[0].legend(fontsize=8)

    image = axes[1].imshow(matrix, vmin=0, vmax=1, cmap="Blues")
    for row in range(len(FAMILIES)):
        for column in range(len(FAMILIES)):
            axes[1].text(column, row, f"{matrix[row, column]:.2f}", ha="center", va="center")
    axes[1].set_xticks(range(len(FAMILIES)), FAMILIES, rotation=25, ha="right")
    axes[1].set_yticks(range(len(FAMILIES)), FAMILIES)
    axes[1].set_xlabel("Predicted family")
    axes[1].set_ylabel("Oracle family")
    axes[1].set_title(f"Best router: {best['method']}")
    figure.colorbar(image, ax=axes[1], fraction=0.046)

    axes[2].hist(regret, bins=np.linspace(0, max(30, np.quantile(regret, 0.99)), 31), color="#d95f02", alpha=0.85)
    axes[2].axvline(np.median(regret), color="black", linestyle="--", label=f"median={np.median(regret):.2f}")
    axes[2].set_xlabel("Family routing regret (ft)")
    axes[2].set_ylabel("Wells")
    axes[2].set_title("Where routing still fails")
    axes[2].legend()
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def run(args):
    output = Path(args.out_dir)
    output.mkdir(parents=True, exist_ok=True)
    cache = output / "four_family_router_features_honest.pkl"
    if cache.exists() and not args.rebuild:
        dataset = pd.read_pickle(cache)
        print(f"Loaded feature cache: {cache}")
    else:
        dataset = build_dataset(args)
        dataset.to_pickle(cache)
        print(f"Saved feature cache: {cache}")

    features = model_features(dataset)
    long = long_matrix(dataset)
    method_scores = {
        "classifier": np.zeros((len(dataset), len(FAMILIES)), dtype=np.float64),
        "risk_regression": np.zeros((len(dataset), len(FAMILIES)), dtype=np.float64),
        "ranker": np.zeros((len(dataset), len(FAMILIES)), dtype=np.float64),
    }
    for fold in range(args.folds):
        train_mask = dataset["fold"].to_numpy(int) != fold
        valid_mask = ~train_mask
        train = dataset.loc[train_mask]
        valid = dataset.loc[valid_mask]
        valid_indices = np.flatnonzero(valid_mask)
        classifier_predictions = []
        regression_predictions = []
        ranker_predictions = []
        for ensemble_index in range(args.ensemble):
            seed = args.seed + fold * 100 + ensemble_index * 11
            classifier_predictions.append(fit_classifier(train, valid, features, seed))
            regression_predictions.append(fit_risk_regression(train, valid, features, seed + 1))
            rank_prediction, well_order = fit_ranker(
                long[long["fold"] != fold], long[long["fold"] == fold], seed + 2
            )
            ranker_predictions.append(rank_prediction)
        method_scores["classifier"][valid_indices] = np.mean(classifier_predictions, axis=0)
        method_scores["risk_regression"][valid_indices] = np.mean(regression_predictions, axis=0)
        method_scores["ranker"][valid_indices] = np.mean(ranker_predictions, axis=0)
        print(f"trained OOF fold {fold}: train={len(train)} valid={len(valid)}", flush=True)

    method_scores["blend"] = np.mean(
        [normalize_scores(method_scores[name]) for name in ("classifier", "risk_regression", "ranker")],
        axis=0,
    )
    risks = dataset[[f"risk_{family}" for family in FAMILIES]].to_numpy(float)
    weights = dataset["n_rows"].to_numpy(float)
    target = dataset["target"].to_numpy(int)
    metrics = [evaluate_method(name, scores, risks, weights, target) for name, scores in method_scores.items()]
    best = min(metrics, key=lambda item: item["selected_pooled"])

    oof = dataset[["well", "fold", "n_rows", "target"] + [f"risk_{family}" for family in FAMILIES]].copy()
    for method, scores in method_scores.items():
        for index, family in enumerate(FAMILIES):
            oof[f"score_{method}_{family}"] = scores[:, index]
        oof[f"pred_{method}"] = np.argmax(scores, axis=1)
    oof.to_csv(output / "four_family_router_oof.csv", index=False)

    compact_metrics = pd.DataFrame([
        {key: value for key, value in item.items() if not isinstance(value, np.ndarray)}
        for item in metrics
    ])
    oracle = np.min(risks, axis=1)
    baselines = []
    for family_index, family in enumerate(FAMILIES):
        baselines.append({"method": f"always_{family}", "pooled": pooled_rmse(risks[:, family_index], weights)})
    compact_metrics.to_csv(output / "four_family_router_metrics.csv", index=False)
    pd.DataFrame(baselines).to_csv(output / "four_family_router_baselines.csv", index=False)
    plot_results(dataset, metrics, best, output / "four_family_router_audit.png")

    print("\nFOUR_FAMILY_ROUTER")
    print(f"wells={len(dataset)} features={len(features)} classes={dict(zip(FAMILIES, np.bincount(target, minlength=4)))}")
    print(f"perfect_family_oracle={pooled_rmse(oracle, weights):.4f}")
    print(compact_metrics.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(pd.DataFrame(baselines).to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"best={best['method']} output={output / 'four_family_router_audit.png'}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool-root", default="whole_path_energy_nested_analog")
    parser.add_argument("--train-dir", default="rogii-wellbore-geology-prediction/train")
    parser.add_argument("--out-dir", default="four_family_router_audit_honest")
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--ensemble", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument("--rebuild", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
