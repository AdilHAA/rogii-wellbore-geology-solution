"""ROGII final mix — Andrey Serzhenko and team N & A & O & A & A misis."""

from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import json, math, os, sys, tempfile, shutil, types, warnings

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 200)


FAST_DEBUG = False
N_WORKERS = 2
MAX_TRAIN_WELLS = 40 if FAST_DEBUG else None
MAX_TEST_WELLS = 2 if FAST_DEBUG else None
DATA_ROOT_OVERRIDE = None


ORIENTED_BEAM = 80
print('FAST_DEBUG:', FAST_DEBUG)
print('ORIENTED_BEAM:', ORIENTED_BEAM)


import base64 as _sub12_base64
import lzma as _sub12_lzma
import os as _sub12_os
import pickle as _sub12_pickle
from pathlib import Path as _Sub12Path

def _find_sub12_asset_dir():
    candidates = []
    env_path = _sub12_os.environ.get("SUB12_ASSET_DIR")
    if env_path:
        candidates.append(_Sub12Path(env_path))
    candidates.extend([
        _Sub12Path.cwd() / "sub12_assets",
        _Sub12Path.cwd().parent / "sub12_assets",
        _Sub12Path("/kaggle/input/rogii-sub12-external-assets"),
        _Sub12Path("/kaggle/input/rogii-sub12-external-assets/sub12_assets"),
    ])
    kaggle_input = _Sub12Path("/kaggle/input")
    if kaggle_input.exists():
        candidates.extend(sorted(kaggle_input.glob("*")))
        candidates.extend(sorted(kaggle_input.glob("*/sub12_assets")))
        candidates.extend(path.parent for path in sorted(kaggle_input.rglob("source_blind_moe.py")))
    candidates.append(_Sub12Path("kaggle_rogii_wellbore_geology_prediction/kaggle_upload/rogii_submission12_external_assets_dataset/sub12_assets"))
    for path in candidates:
        if (path / "source_blind_moe.py").is_file() and (path / "segment_moe_checkpoint.pkl.xz").is_file():
            return path.resolve()
    print("sub12 asset search failed. Checked candidates:")
    for path in candidates:
        print(" -", path, "exists=", path.exists())
    if _Sub12Path("/kaggle/input").exists():
        print("/kaggle/input listing:")
        for path in sorted(_Sub12Path("/kaggle/input").glob("*")):
            try:
                sample = [p.name for p in sorted(path.iterdir())[:20]]
            except Exception as exc:
                sample = [f"<iterdir error: {exc}>"]
            print(" *", path, sample)
    raise FileNotFoundError("Could not find sub12_assets directory. Add dataset andrey4522/rogii-sub12-external-assets.")

SUB12_ASSET_DIR = _find_sub12_asset_dir()
print("SUB12_ASSET_DIR:", SUB12_ASSET_DIR)

def _sub12_asset_text(name):
    return (SUB12_ASSET_DIR / name).read_text(encoding="utf-8")

def _sub12_asset_bytes(name):
    return (SUB12_ASSET_DIR / name).read_bytes()

def _sub12_asset_b64(name):
    return _sub12_base64.b64encode(_sub12_asset_bytes(name)).decode("ascii")

def _sub12_asset_pickle_xz(name):
    return _sub12_pickle.loads(_sub12_lzma.decompress(_sub12_asset_bytes(name)))


def find_data_root():
    candidates = []
    if DATA_ROOT_OVERRIDE is not None:
        candidates.append(Path(DATA_ROOT_OVERRIDE))
    candidates.extend([
        Path('/kaggle/input/competitions/rogii-wellbore-geology-prediction'),
        Path.cwd(),
        Path.cwd() / 'rogii-wellbore-geology-prediction',
    ])
    candidates.extend(Path.cwd().parents)
    for root in candidates:
        if (root / 'train').is_dir() and (root / 'test').is_dir() and (root / 'sample_submission.csv').is_file():
            return root.resolve()
    raise FileNotFoundError('Could not find train/, test/, and sample_submission.csv')

DATA_ROOT = find_data_root()
TRAIN_DIR = DATA_ROOT / 'train'
TEST_DIR = DATA_ROOT / 'test'
SAMPLE_SUB_PATH = DATA_ROOT / 'sample_submission.csv'

train_horizontal_paths = sorted(TRAIN_DIR.glob('*__horizontal_well.csv'))
test_horizontal_paths = sorted(TEST_DIR.glob('*__horizontal_well.csv'))
if MAX_TRAIN_WELLS is not None:
    train_horizontal_paths = train_horizontal_paths[:MAX_TRAIN_WELLS]
if MAX_TEST_WELLS is not None:
    test_horizontal_paths = test_horizontal_paths[:MAX_TEST_WELLS]

sample_submission = pd.read_csv(SAMPLE_SUB_PATH)
print('DATA_ROOT:', DATA_ROOT)
print('Train wells:', len(train_horizontal_paths))
print('Test wells:', len(test_horizontal_paths))


from dataclasses import replace

ORIENTED_SOURCE = _sub12_asset_text("oriented.py")
oriented_mod = types.ModuleType("oriented_stride_inline")
sys.modules["oriented_stride_inline"] = oriented_mod
exec(compile(ORIENTED_SOURCE, "oriented_stride_inline.py", "exec"), oriented_mod.__dict__)

OLD_STRIDE_SOURCE = _sub12_asset_text("old_stride.py")
old_mod = types.ModuleType("old_stride_inline")
sys.modules["old_stride_inline"] = old_mod
exec(compile(OLD_STRIDE_SOURCE, "old_stride_inline.py", "exec"), old_mod.__dict__)


ORIENTED_CFG = oriented_mod.OrientedConfig(
    name="coarse_b480",
    endpoint_step=100.0,
    obs_step=25.0,
    lengths_blocks=(1,),
    beam=ORIENTED_BEAM,
    raw_cost_cap=30.0,
    smooth_sigma=0.008,
    use_turn_mixture=False,
    posterior_temp=4.0,
)


DIRECTIONAL_CFG = oriented_mod.OrientedConfig(
    name="cusp075_self075_lateral_directional",
    endpoint_step=100.0,
    obs_step=25.0,
    lengths_blocks=(1,),
    beam=ORIENTED_BEAM,
    raw_cost_cap=30.0,
    smooth_sigma=0.008,
    use_turn_mixture=False,
    posterior_temp=4.0,
    typewell_response="cusp",
    typewell_response_ft=0.75,
    self_weight=0.75,
    self_prefix_mode="lateral_prefer",
    self_directional=True,
)


LF_CFG = old_mod.VariableCfg(
    name="lf_reverse002_full",
    endpoint_step=100.0,
    lengths_blocks=(1, 2, 3, 4, 6, 8),
    obs_step=25.0,
    beam=60,
    alpha=-0.5,
    gamma=12.0,
    persist_weight=0.5,
    rate_weight=5.0,
    length_weight=0.0,
    calibration="offset0.2",
    posterior_temp=2.0,
    tvt_bin=0.5,
    max_same=1,
    derivative_weight=0.3,
    derivative_gamma=12.0,
    motif_corr_weight=0.0,
    reverse_margin_weight=0.02,
    min_corr_points=5,
)


SELFREF_CFG = replace(
    LF_CFG,
    name="lf_cusp075_self_blend050",
    tw_response="cusp",
    tw_smooth_ft=0.75,
    self_weight=0.5,
    self_mode="blend",
)


def parse_json_array(value):
    return np.asarray(json.loads(value), dtype=float)


def rms(a, b):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    return float(np.sqrt(np.mean((a - b) ** 2)))


def _decode_oriented_with_short_tail_fallback(horizontal_path, config):
    result = oriented_mod.decode_well(horizontal_path, config=config)
    if "error" not in result:
        return result
    fallback = replace(
        config,
        name=config.name + "_short_tail",
        endpoint_step=25.0,
        obs_step=10.0,
        beam=min(ORIENTED_BEAM, 160),
        min_final_fraction=0.0,
    )
    return oriented_mod.decode_well(horizontal_path, config=fallback)


def run_decoders(horizontal_path):
    horizontal_path = Path(horizontal_path)
    stem = horizontal_path.name.split("__", 1)[0]
    typewell_path = TEST_DIR / f"{stem}__typewell.csv"
    if not typewell_path.exists():
        raise FileNotFoundError(typewell_path)

    raw = pd.read_csv(horizontal_path)
    if "TVT_input" not in raw.columns:
        raise ValueError(f"{stem}: missing TVT_input")
    tvt_input = raw["TVT_input"].to_numpy(float)
    known = np.flatnonzero(np.isfinite(tvt_input))
    if len(known) == 0:
        raise ValueError(f"{stem}: no known TVT_input")


    dummy_tvt = tvt_input.copy()
    dummy_tvt[~np.isfinite(dummy_tvt)] = tvt_input[known[-1]]
    raw["TVT"] = dummy_tvt

    with tempfile.TemporaryDirectory(prefix="rogii_physics_") as temp_dir:
        temp_dir = Path(temp_dir)
        local_horizontal = temp_dir / horizontal_path.name
        local_typewell = temp_dir / typewell_path.name
        raw.to_csv(local_horizontal, index=False)
        shutil.copy2(typewell_path, local_typewell)

        oriented = _decode_oriented_with_short_tail_fallback(local_horizontal, ORIENTED_CFG)
        directional = _decode_oriented_with_short_tail_fallback(local_horizontal, DIRECTIONAL_CFG)
        legacy = old_mod.decode_well(local_horizontal, LF_CFG)
        selfref = old_mod.decode_well(local_horizontal, SELFREF_CFG)

    outputs = {"O": oriented, "D": directional, "L": legacy, "S": selfref}
    failures = {name: value for name, value in outputs.items() if "error" in value}
    if failures:
        raise RuntimeError(f"Decoder failure for {stem}: {failures}")
    return oriented, legacy, directional, selfref


print("Embedded O/D/L/S decoders loaded.")
print("Physical weights: P = 0.29*S + 0.31*O + 0.40*D")
print("Final weights: prediction = 0.40*A + 0.60*P")


FORMATION_COLS = ['ANCC', 'ASTNU', 'ASTNL', 'EGFDU', 'EGFDL', 'BUDA']
TEMPLATE_GRID = np.arange(-1200.0, 7001.0, 100.0)

def representative_surface(d):
    available = [c for c in FORMATION_COLS if c in d.columns]
    if available:
        return np.nanmedian(d[available].to_numpy(float), axis=1)

    if {'TVT', 'Z'}.issubset(d.columns):
        return d['TVT'].to_numpy(float) + d['Z'].to_numpy(float)
    raise ValueError('Training well has neither formation surfaces nor TVT+Z')

def build_training_libraries(paths, need_gradient=False):
    profiles = []
    heels = []
    vectors = []
    hidden_lengths = []
    prefix_slopes = []
    stems = []
    grad_mids = []
    grad_vecs = []
    grad_slopes = []

    for i, path in enumerate(paths, start=1):
        stem = path.name.split('__', 1)[0]
        d = pd.read_csv(path).sort_values('MD').reset_index(drop=True)
        md = d['MD'].to_numpy(float)
        x = d['X'].to_numpy(float)
        y = d['Y'].to_numpy(float)
        z = d['Z'].to_numpy(float)
        tin = d['TVT_input'].to_numpy(float)
        known = np.flatnonzero(np.isfinite(tin))
        if len(known) < 2:
            continue
        p = int(known[-1]); heel_md = md[p]
        surface = representative_surface(d)

        heel = np.array([x[p], y[p]], float)
        m2 = min(md[-1], heel_md + 800.0)
        vec = np.array([np.interp(m2, md, x) - heel[0], np.interp(m2, md, y) - heel[1]])
        if np.linalg.norm(vec) < 1.0:
            vec = np.array([x[-1] - heel[0], y[-1] - heel[1]])
        vec /= max(np.linalg.norm(vec), 1e-9)

        qmd = heel_md + TEMPLATE_GRID
        ok = (qmd >= md[0]) & (qmd <= md[-1])
        prof = np.full(len(TEMPLATE_GRID), np.nan, dtype=np.float32)
        prof[ok] = np.interp(qmd[ok], md, surface) - surface[p]

        u = tin + z
        lo = max(md[0], heel_md - 800.0)
        mask = (md >= lo) & (md <= heel_md) & np.isfinite(u)
        pslope = float(np.polyfit(md[mask] - heel_md, u[mask], 1)[0]) if mask.sum() >= 10 else 0.0

        profiles.append(prof); heels.append(heel); vectors.append(vec)
        hidden_lengths.append(float(md[-1] - heel_md)); prefix_slopes.append(pslope); stems.append(stem)

        if need_gradient:
            grid = np.arange(md[0], md[-1] + 50.0, 100.0)
            if grid[-1] > md[-1]:
                grid[-1] = md[-1]
            grid = np.unique(grid)
            if len(grid) >= 2:
                gx = np.interp(grid, md, x); gy = np.interp(grid, md, y); gs = np.interp(grid, md, surface)
                ds = np.diff(grid); dx = np.diff(gx); dy = np.diff(gy); hh = np.hypot(dx, dy)
                valid = (ds > 1e-6) & (hh / ds >= 0.35) & np.isfinite(gs[:-1]) & np.isfinite(gs[1:])
                if valid.any():
                    vv = np.column_stack([dx[valid] / hh[valid], dy[valid] / hh[valid]])
                    mm = np.column_stack([(gx[:-1][valid] + gx[1:][valid]) * 0.5,
                                          (gy[:-1][valid] + gy[1:][valid]) * 0.5])
                    sl = np.diff(gs)[valid] / ds[valid]
                    grad_mids.append(mm); grad_vecs.append(vv); grad_slopes.append(sl)

        if i % 100 == 0 or i == len(paths):
            print(f'Built training library: {i}/{len(paths)} wells')

    cache = {
        'prof': np.asarray(profiles, np.float32),
        'heel': np.asarray(heels, float),
        'vec': np.asarray(vectors, float),
        'hlen': np.asarray(hidden_lengths, float),
        'pslope': np.asarray(prefix_slopes, float),
        'grid': TEMPLATE_GRID.copy(),
        'stems': np.asarray(stems, str),
    }
    if need_gradient:
        gradient = {
            'mids': np.concatenate(grad_mids),
            'vecs': np.concatenate(grad_vecs),
            'slopes': np.clip(np.concatenate(grad_slopes), -0.18, 0.18),
        }
        return cache, gradient
    return cache

def weighted_profile(cache, ids, weights):
    pp = cache['prof'][ids].astype(float)
    mask = np.isfinite(pp)
    den = np.sum(weights[:, None] * mask, axis=0)
    num = np.nansum(weights[:, None] * pp, axis=0)
    profile = np.divide(num, den, out=np.full(pp.shape[1], np.nan), where=den > 0)
    grid = cache['grid'].astype(float)
    ok = np.isfinite(profile) & (grid >= 0)
    if ok.sum() < 2:
        raise RuntimeError('Neighbour profile has insufficient support')
    last = np.flatnonzero(ok)[-1]
    if last < len(grid) - 1:
        a = max(np.flatnonzero(ok)[0], last - 3)
        slope = (profile[last] - profile[a]) / max(grid[last] - grid[a], 1e-9)
        profile[last + 1:] = profile[last] + slope * (grid[last + 1:] - grid[last])
    return profile

def query_metadata(d):
    md = d['MD'].to_numpy(float); tin = d['TVT_input'].to_numpy(float)
    p = int(np.flatnonzero(np.isfinite(tin))[-1])
    heel = np.array([d['X'].iloc[p], d['Y'].iloc[p]], float)
    m2 = min(float(md[-1]), float(md[p] + 800.0))
    vec = np.array([np.interp(m2, md, d['X']) - heel[0], np.interp(m2, md, d['Y']) - heel[1]])
    if np.linalg.norm(vec) < 1.0:
        vec = np.array([d['X'].iloc[-1] - heel[0], d['Y'].iloc[-1] - heel[1]], float)
    vec /= max(np.linalg.norm(vec), 1e-9)
    u = tin + d['Z'].to_numpy(float)
    lo = max(float(md[0]), float(md[p] - 800.0))
    mask = (md >= lo) & (md <= md[p]) & np.isfinite(u)
    prefix_slope = float(np.polyfit(md[mask] - md[p], u[mask], 1)[0]) if mask.sum() >= 10 else 0.0
    return p, heel, vec, float(md[-1] - md[p]), prefix_slope

def local_profile(stem, heel, vec, hidden_len, cache, cfg):
    H = cache['heel']; V = cache['vec']; L = cache['hlen']; stems = cache['stems']
    dist = np.linalg.norm(H - heel, axis=1); cosine = V @ vec
    valid = (cosine > cfg['ct']) & (L >= cfg['lr'] * hidden_len) & (stems != stem)
    score = (dist / cfg['dr']) ** 2 + cfg['ap'] * (1.0 - cosine)
    score[~valid] = np.inf
    ids = np.argsort(score)[:int(cfg['k'])]
    if not np.isfinite(score[ids]).all():
        valid = (cosine > 0.0) & (stems != stem)
        score = (dist / cfg['dr']) ** 2 + cfg['ap'] * (1.0 - cosine)
        score[~valid] = np.inf
        ids = np.argsort(score)[:int(cfg['k'])]
    if not np.isfinite(score[ids]).all():
        raise RuntimeError('Not enough valid neighbours')
    weights = np.exp(-(score[ids] - score[ids].min()) / cfg['temp']); weights /= weights.sum()
    profile = weighted_profile(cache, ids, weights)
    diag = {
        'nearest_distance': float(dist[ids[0]]), 'best_cosine': float(cosine[ids[0]]),
        'score0': float(score[ids[0]]), 'score_gap': float(score[ids[-1]] - score[ids[0]]),
        'weight_max': float(weights.max()), 'weight_entropy': float(-np.sum(weights * np.log(weights + 1e-12))),
        'stems': '|'.join(stems[ids].tolist()),
    }
    return profile, diag

template_cache = build_training_libraries(train_horizontal_paths, need_gradient=False)
print('Template library wells:', len(template_cache['stems']))

AGGR_IMPUTER = np.array([4914.075, 4914.075, 0.007831873462800908, 8.607022945774899, 11.895255419518405, 12.561100188954002, 3.6149321621219372, 2.592925772054105, 4.57746891586564, 11.244321777778366, 19.181186288134874, 21.21929620094535, 29.668836021377682, 23.111759500669603, 37.043525039199764, 19.319484069064288, 3.841666666666667, 0.7957878090081759, 2.5892615923639037, 1.2852588519292056, 3.679281285823627, 1.5437133952256181, 26.144664618810427, -0.15881240794956922, 0.10556444001482544, 0.5295472496591208, 0.4597935986094248, 0.9621610145869598], dtype=float)
AGGR_MEAN = np.array([4914.075, 4914.075, 0.007831873462800908, 8.607022945774899, 11.895255419518405, 12.561100188954002, 3.6149321621219372, 2.592925772054105, 4.57746891586564, 11.244321777778366, 19.181186288134874, 21.21929620094535, 29.668836021377682, 23.111759500669603, 37.043525039199764, 19.319484069064288, 3.841666666666667, 0.7957878090081759, 2.5892615923639037, 1.2852588519292056, 3.679281285823627, 1.5437133952256181, 26.144664618810427, -0.15881240794956922, 0.10556444001482544, 0.5295472496591208, 0.4597935986094248, 0.9621610145869598], dtype=float)
AGGR_SCALE = np.array([1194.8785932923618, 1194.8785932923618, 0.03422855811738081, 7.934974031995946, 9.701877266743589, 8.787911209303319, 4.408515764911315, 2.356926152588992, 4.738944404631628, 13.5513390491763, 20.81757989459409, 20.25601965943114, 16.300457371752373, 12.27575359443341, 19.744388616456625, 5.841603865000753, 1.488824107214221, 0.18870312074499943, 0.5195935232792309, 0.9673279965781476, 2.5911853006668983, 0.5744241098610592, 6.9196825135884135, 0.649226490324909, 0.3104136880841924, 1.7421533909555078, 0.18131340923589873, 0.26688003081708117], dtype=float)
AGGR_COEF = np.array([0.0005156819300728533, 0.0005156819300728517, -0.006980593009737202, 0.012428867612238522, 0.01121304059307199, -0.015998829235490605, -0.019056173699983373, 0.021957344328941888, 0.0016883131217663824, 0.007286178792193927, -0.017762038369217647, -0.01580512125798483, -0.013508838861525891, -0.01789250023175713, 0.0037235856755987277, 0.03065272542584903, 0.0031598025248590835, -0.04477969284411039, 0.01133990008629231, -0.030660108037930296, 0.012783598276790225, 0.0013138974300516454, 0.010591946936751183, 0.018296758735156626, -0.006819250631205808, 0.006429834581409109, -0.015622445613589877, -0.01612287811130498], dtype=float)
AGGR_INTERCEPT = 0.24793707973410817

def manual_ridge_predict(values, imputer, mean, scale, coef, intercept):
    x = np.asarray(values, float)
    x = np.where(np.isfinite(x), x, imputer)
    z = (x - mean) / np.where(scale > 0, scale, 1.0)
    return float(intercept + np.dot(coef, z))

AGGR_FEATURES = ['hidden_len', 'nrows', 'prefix_slope', 'd_ol', 'd_on', 'd_ln', 'd_n3n2', 'd_n3n5', 'd_n2n5', 'end_ol', 'end_on', 'end_ln', 'range_o', 'range_l', 'range_n', 'o_beam_spread', 'o_modes', 'o_dom_mass', 'o_cost_per_boundary', 'o_turns', 'lf_beam_spread', 'lf_cost_per_boundary', 'lf_segments', 'lf_cal_b', 'n_score0', 'n_score_gap', 'n_weight_max', 'n_weight_entropy']
AGGR_BASE_WEIGHTS = np.array([0.8132402227864471, 0.18675977721355289], float)
AGGR_BASE_WEIGHTS /= AGGR_BASE_WEIGHTS.sum()
AGGR_NEIGHBOUR_WEIGHTS = np.array([0.345464781661048, 0.1522265423303951, 0.5023086760085569], float)
AGGR_NEIGHBOUR_WEIGHTS /= AGGR_NEIGHBOUR_WEIGHTS.sum()

CFG_N3 = dict(k=3, dr=5000.0, ap=10.0, temp=0.3, lr=0.5, ct=0.7)
CFG_N2 = dict(k=2, dr=10000.0, ap=10.0, temp=0.3, lr=0.5, ct=0.7)
CFG_N5 = dict(k=5, dr=5000.0, ap=10.0, temp=0.3, lr=0.5, ct=0.7)

PHYSICAL_WEIGHTS = np.array([0.29, 0.31, 0.40], dtype=float)
FINAL_FORMATION_WEIGHT = 0.40
FINAL_PHYSICAL_WEIGHT = 0.60
FORMATION_GATE_SCALE = 1.90
FORMATION_GATE_CAP = 0.65


def predict_one_directional_physics(path):
    path = Path(path)
    stem = path.name.split("__", 1)[0]
    raw = pd.read_csv(path)
    raw["_row_index"] = np.arange(len(raw), dtype=np.int64)
    data = raw.sort_values("MD").reset_index(drop=True)

    prefix_end, heel, vector, hidden_len, prefix_slope = query_metadata(data)
    md = data["MD"].to_numpy(float)
    z = data["Z"].to_numpy(float)
    tvt_input = data["TVT_input"].to_numpy(float)
    hidden_md = md[prefix_end + 1:]
    hidden_rows = data["_row_index"].to_numpy(np.int64)[prefix_end + 1:]
    relative_md = hidden_md - md[prefix_end]

    oriented, legacy, directional, selfref = run_decoders(path)
    O = np.interp(
        hidden_md,
        parse_json_array(oriented["boundaries"]),
        parse_json_array(oriented["mean_tvt_boundaries"]),
    )
    L = np.interp(
        hidden_md,
        parse_json_array(legacy["boundaries"]),
        parse_json_array(legacy["mean_tvt_boundaries"]),
    )
    D = np.interp(
        hidden_md,
        parse_json_array(directional["boundaries"]),
        parse_json_array(directional["mean_tvt_boundaries"]),
    )
    S = np.interp(
        hidden_md,
        parse_json_array(selfref["boundaries"]),
        parse_json_array(selfref["mean_tvt_boundaries"]),
    )

    profile3, diag3 = local_profile(stem, heel, vector, hidden_len, template_cache, CFG_N3)
    profile2, _ = local_profile(stem, heel, vector, hidden_len, template_cache, CFG_N2)
    profile5, _ = local_profile(stem, heel, vector, hidden_len, template_cache, CFG_N5)
    base_surface = tvt_input[prefix_end] + z[prefix_end]
    N3 = base_surface + np.interp(relative_md, template_cache["grid"], profile3) - z[prefix_end + 1:]
    N2 = base_surface + np.interp(relative_md, template_cache["grid"], profile2) - z[prefix_end + 1:]
    N5 = base_surface + np.interp(relative_md, template_cache["grid"], profile5) - z[prefix_end + 1:]
    N = (
        AGGR_NEIGHBOUR_WEIGHTS[0] * N3
        + AGGR_NEIGHBOUR_WEIGHTS[1] * N2
        + AGGR_NEIGHBOUR_WEIGHTS[2] * N5
    )
    B = AGGR_BASE_WEIGHTS[0] * O + AGGR_BASE_WEIGHTS[1] * L

    gate_features = {
        "hidden_len": hidden_len,
        "nrows": len(hidden_md),
        "prefix_slope": float(oriented["prefix_surface_slope"]),
        "d_ol": rms(O, L),
        "d_on": rms(O, N),
        "d_ln": rms(L, N),
        "d_n3n2": rms(N3, N2),
        "d_n3n5": rms(N3, N5),
        "d_n2n5": rms(N2, N5),
        "end_ol": abs(O[-1] - L[-1]),
        "end_on": abs(O[-1] - N[-1]),
        "end_ln": abs(L[-1] - N[-1]),
        "range_o": float(np.ptp(O)),
        "range_l": float(np.ptp(L)),
        "range_n": float(np.ptp(N)),
        "o_beam_spread": float(oriented["beam_spread"]),
        "o_modes": float(oriented["n_modes"]),
        "o_dom_mass": float(oriented["dominant_mode_mass"]),
        "o_cost_per_boundary": float(
            oriented["final_cost"] / max(len(parse_json_array(oriented["boundaries"])), 1)
        ),
        "o_turns": float(oriented["posterior_mean_turns"]),
        "lf_beam_spread": float(legacy["beam_spread"]),
        "lf_cost_per_boundary": float(
            legacy["final_cost"] / max(len(parse_json_array(legacy["boundaries"])), 1)
        ),
        "lf_segments": float(legacy["posterior_mean_segments"]),
        "lf_cal_b": float(legacy["cal_b"]),
        "n_score0": diag3["score0"],
        "n_score_gap": diag3["score_gap"],
        "n_weight_max": diag3["weight_max"],
        "n_weight_entropy": diag3["weight_entropy"],
    }
    alpha_raw = manual_ridge_predict(
        [gate_features[column] for column in AGGR_FEATURES],
        AGGR_IMPUTER,
        AGGR_MEAN,
        AGGR_SCALE,
        AGGR_COEF,
        AGGR_INTERCEPT,
    )
    alpha = float(np.clip(FORMATION_GATE_SCALE * alpha_raw, 0.0, FORMATION_GATE_CAP))


    A = (1.0 - alpha) * B + alpha * N

    P = PHYSICAL_WEIGHTS[0] * S + PHYSICAL_WEIGHTS[1] * O + PHYSICAL_WEIGHTS[2] * D
    prediction = FINAL_FORMATION_WEIGHT * A + FINAL_PHYSICAL_WEIGHT * P

    output = pd.DataFrame({
        "stem": stem,
        "row_index": hidden_rows,
        "tvt": prediction,
        "pred_a": A,
        "pred_s": S,
        "pred_o": O,
        "pred_d": D,
    })
    diagnostics = {
        "stem": stem,
        "formation_alpha": alpha,
        "formation_alpha_raw": alpha_raw,
        "nearest_distance": diag3["nearest_distance"],
        "neighbours": diag3["stems"],
        "rms_O_D": rms(O, D),
        "rms_O_S": rms(O, S),
        "rms_A_P": rms(A, P),
        "rms_B_N": rms(B, N),
        "directional_self_coverage": float(directional.get("self_directional_coverage", np.nan)),
        "directional_primary_coverage": float(directional.get("self_primary_coverage", np.nan)),
        "selfref_weight": float(selfref.get("selected_self_weight", np.nan)),
    }
    return output, diagnostics


prediction_parts = []
diagnostics = []
errors = []
remaining = list(test_horizontal_paths)


if remaining:
    first = remaining.pop(0)
    prediction, diagnostic = predict_one_directional_physics(first)
    prediction_parts.append(prediction)
    diagnostics.append(diagnostic)
    print("Compiled and processed:", diagnostic["stem"])

with ThreadPoolExecutor(max_workers=N_WORKERS) as pool:
    futures = {pool.submit(predict_one_directional_physics, path): path for path in remaining}
    for index, future in enumerate(as_completed(futures), start=1):
        source_path = futures[future]
        try:
            prediction, diagnostic = future.result()
            prediction_parts.append(prediction)
            diagnostics.append(diagnostic)
        except Exception as exc:
            errors.append({"stem": source_path.name.split("__", 1)[0], "error": repr(exc)})
        if index % 10 == 0 or index == len(remaining):
            print(f"Processed {index + 1}/{len(test_horizontal_paths)} wells")

diagnostics_df = pd.DataFrame(diagnostics).sort_values("stem")
if errors:
    display(pd.DataFrame(errors))
    raise RuntimeError(f"{len(errors)} wells failed")


import types

FORMATION_SOURCE = _sub12_asset_text("formation.py")
RENDER_SOURCE = _sub12_asset_text("formation_render.py")
ROUTER_SOURCE = _sub12_asset_text("formation_router.py")
FORMATION_RUNTIME_SOURCE = _sub12_asset_text("formation_runtime.py")
FORMATION_ARTIFACTS = _sub12_asset_pickle_xz("formation_artifacts.pkl.xz")


def load_embedded_module(name, source):
    module = types.ModuleType(name)
    module.__file__ = f"<embedded {name}>"
    sys.modules[name] = module
    exec(compile(source, module.__file__, "exec"), module.__dict__)
    return module


formation_module = load_embedded_module("formation_analog_ranker", FORMATION_SOURCE)
render_module = load_embedded_module("materialize_whole_path_energy", RENDER_SOURCE)
router_module = load_embedded_module("audit_four_family_router", ROUTER_SOURCE)
formation_runtime = load_embedded_module("formation_blend_runtime", FORMATION_RUNTIME_SOURCE)

print("Formation production method:", FORMATION_ARTIFACTS["production_method"])


formation_predictions, formation_diagnostics = formation_runtime.run_formation_branch(
    TRAIN_DIR,
    TEST_DIR,
    test_horizontal_paths,
    FORMATION_ARTIFACTS,
    formation_module,
    render_module,
    router_module,
)

if formation_predictions["formation_tvt"].isna().any():
    raise ValueError("Formation branch produced NaN predictions")


import base64, lzma, pickle, torch
if not torch.cuda.is_available():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
SOURCE_BLIND_MOE_SOURCE = _sub12_asset_text("source_blind_moe.py")
SOURCE_BLIND_MOE_PAYLOAD = _sub12_asset_b64("source_blind_moe_checkpoint.pkl.xz")
moe_module = load_embedded_module("source_blind_physical_moe_runtime", SOURCE_BLIND_MOE_SOURCE)
moe_checkpoint = pickle.loads(lzma.decompress(base64.b64decode(SOURCE_BLIND_MOE_PAYLOAD)))
moe_args = moe_checkpoint["args"]
moe_model = moe_module.SourceBlindPhysicalMoE(
    channels=len(moe_checkpoint["feature_names"]),
    hidden=int(moe_args["hidden"]),
    embedding=int(moe_args["embedding"]),
    blocks=int(moe_args["blocks"]),
    dropout=float(moe_args["dropout"]),
    max_logit_delta=float(moe_args["max_logit_delta"]),
    correction_scale_ft=float(moe_args["correction_scale_ft"]),
    base_weights=np.asarray(moe_checkpoint["base_weights"], dtype=np.float32),
    correction_enabled=False,
    weight_mode="global",
    local_smoothing=int(moe_args["local_smoothing"]),
)
moe_state = {}
for name, packed in moe_checkpoint["state"].items():
    array = np.asarray(packed.get("raw")) if "raw" in packed else np.asarray(packed["q"], np.float32) * np.asarray(packed["s"], np.float32)
    moe_state[name] = torch.from_numpy(np.array(array, copy=True))
moe_model.load_state_dict(moe_state)
MOE_DEVICE = torch.device("cpu")
moe_model.to(MOE_DEVICE).eval()
print("Source-blind physical MoE:", MOE_DEVICE)


candidate_predictions = pd.concat(
    prediction_parts,
    ignore_index=True,
)
moe_parts = []
moe_diagnostics = []

with torch.no_grad():
    for stem, group in candidate_predictions.groupby("stem", sort=True):
        group = group.reset_index(drop=True).copy()
        feature_group = group[
            ["pred_a", "pred_s", "pred_o", "pred_d"]
        ].copy()
        feature_group["true_tvt"] = feature_group["pred_a"]
        features, compact_experts, _, feature_names = (
            moe_module.materialize_features(
                stem,
                feature_group,
                TEST_DIR,
                int(moe_args["points"]),
            )
        )
        if feature_names != moe_checkpoint["feature_names"]:
            raise ValueError(f"{stem}: MoE feature channel mismatch")
        feature_tensor = torch.from_numpy(features)[None].to(MOE_DEVICE)
        expert_tensor = torch.from_numpy(compact_experts)[None].to(MOE_DEVICE)
        source_indices = torch.arange(4, device=MOE_DEVICE)[None]
        active = torch.ones(1, 4, device=MOE_DEVICE)
        _, _, posterior, _ = moe_model(
            feature_tensor,
            expert_tensor,
            source_indices,
            active,
        )
        weights = posterior[0].detach().cpu().numpy()
        expert_full = group[
            ["pred_a", "pred_s", "pred_o", "pred_d"]
        ].to_numpy(float)
        source_blind_tvt = expert_full @ weights
        moe_parts.append(
            pd.DataFrame(
                {
                    "stem": stem,
                    "row_index": group["row_index"].to_numpy(np.int64),
                    "source_blind_tvt": source_blind_tvt,
                }
            )
        )
        moe_diagnostics.append(
            {
                "stem": stem,
                "weight_a": float(weights[0]),
                "weight_s": float(weights[1]),
                "weight_o": float(weights[2]),
                "weight_d": float(weights[3]),
            }
        )

source_blind_predictions = pd.concat(moe_parts, ignore_index=True)
moe_diagnostics_df = pd.DataFrame(moe_diagnostics).sort_values("stem")


SEGMENT_MOE_PAYLOAD = _sub12_asset_b64("segment_moe_checkpoint.pkl.xz")
segment_checkpoint = pickle.loads(lzma.decompress(base64.b64decode(SEGMENT_MOE_PAYLOAD)))
segment_args = segment_checkpoint["args"]

class SegmentResidualMoE(moe_module.SourceBlindPhysicalMoE):
    def __init__(self, channels, hidden, embedding, blocks, dropout, max_delta, base_weights, segments):
        super().__init__(
            channels, hidden, embedding, blocks, dropout, max_delta, 20.0,
            base_weights, False, "global", 31,
        )
        self.segments = int(segments)
        self.segment_score_head = torch.nn.Sequential(
            torch.nn.Linear(6 * embedding, hidden),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden, 1),
        )

    def forward(self, features, experts, source_indices, active):
        sequence, embedding = self.encode(features)
        batch, candidates, hidden, points = sequence.shape
        flat = sequence.reshape(batch * candidates, hidden, points)
        mean = torch.nn.functional.adaptive_avg_pool1d(flat, self.segments)
        square = torch.nn.functional.adaptive_avg_pool1d(flat.square(), self.segments)
        std = torch.sqrt(torch.clamp(square - mean.square(), min=1e-6))
        maximum = torch.nn.functional.adaptive_max_pool1d(flat, self.segments)
        local = self.embedding(
            torch.cat((mean, std, maximum), 1).permute(0, 2, 1).reshape(
                batch * candidates * self.segments, 3 * hidden
            )
        ).reshape(batch, candidates, self.segments, -1)
        context = local.mean(1, keepdim=True).expand_as(local)
        global_candidate = embedding[:, :, None, :].expand_as(local)
        global_context = embedding.mean(1)[:, None, None, :].expand_as(local)
        score_input = torch.cat((
            local, context, local - context, (local - context).abs(),
            global_candidate, global_context,
        ), 3)
        delta = self.max_logit_delta * torch.tanh(
            self.segment_score_head(score_input).squeeze(-1)
        )
        logits = self.base_logits[source_indices][:, :, None] + delta
        logits = torch.where(active[:, :, None] > 0, logits, torch.full_like(logits, -30.0))
        point_logits = torch.nn.functional.interpolate(
            logits.reshape(batch * candidates, 1, self.segments),
            size=points, mode="linear", align_corners=True,
        ).reshape(batch, candidates, points)
        weights = torch.softmax(point_logits, 1)
        blend = torch.sum(weights * experts, 1)
        coefficients = torch.zeros(batch, 2, device=features.device, dtype=features.dtype)
        return blend, blend, weights, coefficients

segment_model = SegmentResidualMoE(
    len(segment_checkpoint["feature_names"]),
    int(segment_args["hidden"]),
    int(segment_args["embedding"]),
    int(segment_args["blocks"]),
    float(segment_args["dropout"]),
    float(segment_args["max_logit_delta"]),
    np.asarray(segment_checkpoint["base_weights"], np.float32),
    int(segment_args["segments"]),
)
segment_state = {}
for name, packed in segment_checkpoint["state"].items():
    array = np.asarray(packed.get("raw")) if "raw" in packed else np.asarray(packed["q"], np.float32) * np.asarray(packed["s"], np.float32)
    segment_state[name] = torch.from_numpy(np.array(array, copy=True))
segment_model.load_state_dict(segment_state)
segment_model.to(MOE_DEVICE).eval()
print("Segment residual MoE:", MOE_DEVICE)


segment_input = (
    candidate_predictions
    .merge(source_blind_predictions, on=["stem", "row_index"], validate="one_to_one")
    .merge(formation_predictions, on=["stem", "row_index"], validate="one_to_one")
    .rename(columns={"source_blind_tvt": "pred_base", "formation_tvt": "pred_f"})
)
segment_parts, segment_diagnostics = [], []
segment_columns = ["pred_base", "pred_a", "pred_s", "pred_o", "pred_d", "pred_f"]
old_expert_columns = moe_module.EXPERT_COLUMNS
moe_module.EXPERT_COLUMNS = tuple(segment_columns)
with torch.no_grad():
    for stem, group in segment_input.groupby("stem", sort=True):
        group = group.reset_index(drop=True).copy()
        feature_group = group[segment_columns].copy()
        feature_group["true_tvt"] = feature_group["pred_base"]
        features, compact_experts, _, feature_names = moe_module.materialize_features(
            stem, feature_group, TEST_DIR, int(segment_args["points"])
        )
        if feature_names != segment_checkpoint["feature_names"]:
            raise ValueError(f"{stem}: segment feature channel mismatch")
        feature_tensor = torch.from_numpy(features)[None].to(MOE_DEVICE)
        expert_tensor = torch.from_numpy(compact_experts)[None].to(MOE_DEVICE)
        source_indices = torch.arange(6, device=MOE_DEVICE)[None]
        active = torch.ones(1, 6, device=MOE_DEVICE)
        _, _, compact_weights, _ = segment_model(
            feature_tensor, expert_tensor, source_indices, active
        )
        compact_weights = compact_weights[0].cpu().numpy()
        compact_x = np.linspace(0.0, 1.0, compact_weights.shape[1])
        full_x = np.linspace(0.0, 1.0, len(group))
        full_weights = np.stack([
            np.interp(full_x, compact_x, compact_weights[index])
            for index in range(6)
        ])
        full_weights /= full_weights.sum(0, keepdims=True)
        prediction = np.sum(
            full_weights * group[segment_columns].to_numpy(float).T,
            axis=0,
        )
        segment_parts.append(pd.DataFrame({
            "stem": stem,
            "row_index": group["row_index"].to_numpy(np.int64),
            "segment_tvt": prediction,
        }))
        segment_diagnostics.append({
            "stem": stem,
            **{
                f"weight_{name}": float(full_weights[index].mean())
                for index, name in enumerate(["base", "a", "s", "o", "d", "f"])
            },
        })
moe_module.EXPERT_COLUMNS = old_expert_columns
segment_predictions = pd.concat(segment_parts, ignore_index=True)
segment_diagnostics_df = pd.DataFrame(segment_diagnostics).sort_values("stem")


print("\n=== FIXED BLEND: m1=0.20, sub12=0.525, exp165=0.275, TVT SHIFT +0.4 ===", flush=True)

import gc as _blend_gc
import glob as _blend_glob
import json as _blend_json
import os as _blend_os
import sys as _blend_sys
import time as _blend_time
from pathlib import Path as _BlendPath


FINAL_TVT_SHIFT = +0.4
WEIGHTS = {
    "m1": 0.1600,
    "m2": 0.0,
    "sub12": 0.4200,
    "exp158": 0.0,
    "exp165": 0.2200,
}
NORMALIZE_WEIGHTS = True
FORCE_RUN = {
    "m1m2": False,
    "exp158": False,
    "exp165": False,
}


_blend_output_dir = _BlendPath("/kaggle/working") if _BlendPath("/kaggle/working").is_dir() else _BlendPath.cwd()
_target_cols = [column for column in sample_submission.columns if column != "id"]
if len(_target_cols) != 1:
    raise ValueError(f"Expected one target column, got {_target_cols}")
_target_col = _target_cols[0]
print("blend weights raw:", WEIGHTS, flush=True)


def _save_table(frame, stem):
    csv_path = _blend_output_dir / f"{stem}.csv"
    frame.to_csv(csv_path, index=False)
    parquet_path = _blend_output_dir / f"{stem}.parquet"
    try:
        frame.to_parquet(parquet_path, index=False)
        print(f"saved {csv_path.name} and {parquet_path.name} {frame.shape}", flush=True)
    except Exception as exc:
        print(f"saved {csv_path.name}; parquet skipped: {type(exc).__name__}: {exc}", flush=True)


def _as_submission(frame, pred_col, stem):
    out = sample_submission[["id"]].merge(
        frame[["id", pred_col]].rename(columns={pred_col: _target_col}),
        on="id",
        how="left",
        validate="one_to_one",
    )
    if out[_target_col].isna().any():
        missing = out.loc[out[_target_col].isna(), "id"].head(10).tolist()
        raise ValueError(f"{stem} missing predictions: {missing}")
    out = out[sample_submission.columns]
    _save_table(out, stem)
    return out


components = {}


sub12_detail = segment_predictions.copy()
sub12_detail["id"] = sub12_detail["stem"].astype(str) + "_" + sub12_detail["row_index"].astype(int).astype(str)
_save_table(sub12_detail, "sub12_segment_detail_predictions")
components["sub12"] = _as_submission(sub12_detail, "segment_tvt", "sub12_segment_submission")
print("sub12 ready", components["sub12"].shape, flush=True)


def _need_component(name):
    return abs(float(WEIGHTS.get(name, 0.0))) > 1e-12


def _find_adil_artifacts():
    hits = (
        _blend_glob.glob("/kaggle/input/rogii-artifacts/src/data.py")
        + _blend_glob.glob("/kaggle/input/datasets/*/*/src/data.py")
        + _blend_glob.glob("/kaggle/input/*/src/data.py")
        + _blend_glob.glob("/kaggle/input/*/*/src/data.py")
        + _blend_glob.glob("**/src/data.py", recursive=True)
    )
    hits = [p for p in hits if "rogii-sub12-external-assets" not in p]
    if not hits:
        raise FileNotFoundError("Could not find Adil rogii-artifacts src/data.py. Add dataset adilkhabibullin/rogii-artifacts.")
    return _BlendPath(sorted(hits)[0]).resolve().parents[1]


def _run_adil_exp(exp_name, out_col, cache_name):
    print(f"\n--- running Adil {exp_name} ---", flush=True)
    for _k, _v in [
        ("ROGII_SAVGOL", "31"),
        ("ROGII_SW_TSTEP", "1.0"),
        ("ROGII_SW_THALF", "128"),
        ("ROGII_SW_SCALE", "50"),
        ("ROGII_SW_LOC", "1"),
        ("ROGII_SW_GEO", "1"),
        ("ROGII_SW_MISFIT", "1"),
        ("ROGII_CHUNK_W", "48"),
        ("ROGII_CHUNK_HIST", "8"),
    ]:
        _blend_os.environ.setdefault(_k, _v)

    _modules = {"src", "data", "render", "render_sw", "unet", "predict_sw", "metric", "cv", "hedge", "beam", "rollout2"}
    for _m in [m for m in list(_blend_sys.modules) if m in _modules or m.startswith("src.")]:
        del _blend_sys.modules[_m]

    art = _find_adil_artifacts()
    if str(art) not in _blend_sys.path:
        _blend_sys.path.insert(0, str(art))
    if str(art / "src") not in _blend_sys.path:
        _blend_sys.path.insert(0, str(art / "src"))
    print(f"Adil artifacts: {art}", flush=True)

    import data as _a_data
    _a_data.RAW = str(DATA_ROOT)
    _a_data.CACHE = f"/tmp/{cache_name}_derived"

    import render as _a_render
    import render_sw as _a_render_sw
    from unet import GeoSteerNet as _GeoSteerNet
    from rollout2 import rollout_well as _rollout_well
    from render_sw import SDF_CLAMP as _SDF_CLAMP
    import torch as _torch

    device = "cpu"
    try:
        if _torch.cuda.is_available() and _torch.cuda.get_device_capability(0) >= (7, 0):
            _ = (_torch.ones(8, device="cuda") * 2).sum().item()
            device = "cuda"
    except Exception as exc:
        print(f"{exp_name} GPU unusable: {exc}", flush=True)
    if device == "cpu":
        try:
            _torch.set_num_threads(1)
            _torch.set_num_interop_threads(1)
        except Exception:
            pass
    print(f"{exp_name} DEVICE={device} N_CH={_a_render_sw.N_CH}", flush=True)

    models = []
    for fold in range(5):
        model = _GeoSteerNet(base_ch=32, in_ch=_a_render_sw.N_CH, out_clamp=_SDF_CLAMP, prox_tau=0.5, n_modes=4)
        ckpt = art / "models" / f"{exp_name}_fold{fold}.pt"
        model.load_state_dict(_torch.load(ckpt, map_location=device))
        model.to(device).eval()
        models.append(model)
    print(f"loaded {len(models)} models for {exp_name}", flush=True)

    def _predict_one(wid):
        hw_path = TEST_DIR / f"{wid}__horizontal_well.csv"
        hw = pd.read_csv(hw_path)
        ev_idx = hw.index[hw["TVT_input"].isna()].to_numpy()
        if len(ev_idx) == 0:
            return pd.DataFrame(columns=["id", out_col])
        anchor = float(hw["TVT_input"].dropna().iloc[-1]) if hw["TVT_input"].notna().any() else float(hw["Z"].median())
        try:
            wd = _a_render.load_well_data(wid, split="test")
            if not np.isfinite(wd["h_gr_smooth"]).all():
                fill = float(np.nanmedian(wd["t_gr"])) if np.isfinite(wd["t_gr"]).any() else 0.0
                wd["h_gr_smooth"] = np.nan_to_num(wd["h_gr_smooth"], nan=fill)
            with _torch.no_grad():
                full = _rollout_well(wd, models, device, 48, 8, stride_frac=0.5, rebase_frac=0.75,
                                     tri=True, tau=0.2, lam=1.1, K=4, logits=False)
            pred = np.asarray(full[ev_idx], dtype=float)
            if not np.isfinite(pred).all():
                bad = int((~np.isfinite(pred)).sum())
                print(f"WARN {exp_name} {wid}: {bad} non-finite -> anchor fill", flush=True)
                pred = np.where(np.isfinite(pred), pred, anchor)
        except Exception as exc:
            print(f"WARN {exp_name} {wid}: rollout failed ({type(exc).__name__}: {exc}); anchor fallback", flush=True)
            pred = np.full(len(ev_idx), anchor, dtype=float)
        return pd.DataFrame({"id": [f"{wid}_{int(i)}" for i in ev_idx], out_col: pred})

    wids = sorted(path.name.split("__", 1)[0] for path in TEST_DIR.glob("*__horizontal_well.csv"))
    parts = []
    t0 = _blend_time.time()
    for i, wid in enumerate(wids, 1):
        parts.append(_predict_one(wid))
        print(f"{exp_name} [{i}/{len(wids)}] {wid} elapsed={_blend_time.time()-t0:.1f}s", flush=True)
    pred_df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["id", out_col])
    if pred_df[out_col].isna().any():
        raise ValueError(f"{exp_name} produced NaN predictions")
    _save_table(pred_df, f"{cache_name}_detail_predictions")
    sub = _as_submission(pred_df, out_col, f"{cache_name}_submission")
    del models
    _blend_gc.collect()
    try:
        if device == "cuda":
            _torch.cuda.empty_cache()
    except Exception:
        pass
    return sub


if _need_component("exp158") or FORCE_RUN.get("exp158", False):
    components["exp158"] = _run_adil_exp("exp_158_apc", "exp158_tvt", "exp158_apc")
else:
    print("exp158 skipped: zero weight", flush=True)

if _need_component("exp165") or FORCE_RUN.get("exp165", False):
    components["exp165"] = _run_adil_exp("exp_165_burst", "exp165_tvt", "exp165_burst")
else:
    print("exp165 skipped: zero weight", flush=True)


def _normalize_m1m2_frame(df):
    df = df.copy()
    if "row_id" in df.columns and "id" not in df.columns:
        df = df.rename(columns={"row_id": "id"})
    rename = {}
    for cand in ["m1", "model1", "oubank_model1", "oubank_model1_s42", "pred_m1"]:
        if cand in df.columns and "m1_tvt" not in df.columns:
            rename[cand] = "m1_tvt"
            break
    for cand in ["m2", "model2", "oubank_model2", "oubank_model2_s42", "pred_m2"]:
        if cand in df.columns and "m2_tvt" not in df.columns:
            rename[cand] = "m2_tvt"
            break
    if rename:
        df = df.rename(columns=rename)
    need = ["id"]
    if _need_component("m1"):
        need.append("m1_tvt")
    if _need_component("m2"):
        need.append("m2_tvt")
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"m1/m2 frame missing columns {missing}; columns={list(df.columns)[:30]}")
    return df[[c for c in ["id", "m1_tvt", "m2_tvt"] if c in df.columns]]


def _run_v136_oubank_m1m2():
    print("\n--- running Oubank m1/m2 from v136 feature/inference code ---", flush=True)
    _blend_os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")
    full_src = (Path(__file__).resolve().parents[1] / "components" / "phong" / "v136.py").read_text(encoding="utf-8")
    full_src = full_src.replace("@njit(cache=True)", "@njit(cache=False)")

    def _between(a, b):
        ia = full_src.index(a)
        ib = full_src.index(b, ia)
        return full_src[ia:ib]

    pieces = [
        full_src[:full_src.index("# Locate all files using path searcher helper")],
        _between("def loc(marker", "# Load priors for FE"),
        _between("def _load_v35pfv2_raw", "# Construct PF35V2 map"),
        _between("def _smstd22", "# Load model weights and run"),
        _between("# Run the 14-profile particle filters", "# Beam search parameters"),
        _between("# Beam search parameters", "# Parallel run of beam search selector on test wells"),
        _between("class _DTCNpfb", "# Skip nn_pfb_all14"),
        _between("# ===== V96 form-aware NN TEST INFERENCE", "V96_NN = ["),
    ]
    minimal_src = "\n\n".join(pieces)
    g = {"__name__": "__main__"}
    exec(compile(minimal_src, "v136_oubank_minimal_defs.py", "exec"), g, g)

    driver = r'''
import os, gc, time
import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed
from pathlib import Path

print("[OUBANK] building deployable m1/m2 inputs", flush=True)
# Match rogii_v136_fusion4.py: Oubank PF-bank is designed to run on CUDA when available.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_PFDEV = "cuda" if torch.cuda.is_available() else "cpu"
try:
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
except RuntimeError as e:
    print(f"[OUBANK] torch.set_num_threads skipped: {e}", flush=True)
try:
    torch.set_num_interop_threads(1)
except RuntimeError as e:
    print(f"[OUBANK] torch.set_num_interop_threads skipped: {e}", flush=True)
print(f"[OUBANK] v136 torch device={DEVICE} pf_device={_PFDEV} threads={torch.get_num_threads()}", flush=True)
test_paths = sorted((CFG.dataset_path / "test").glob("*__horizontal_well.csv"))
test_wids = [p.stem.replace("__horizontal_well", "") for p in test_paths]
sample_path = CFG.dataset_path / "sample_submission.csv"
sample_ids = pd.read_csv(sample_path, usecols=["id"])["id"].astype(str)
print(f"[OUBANK] test wells={len(test_wids)} sample rows={len(sample_ids)}", flush=True)

parts = []
for p in test_paths:
    wid = p.stem.replace("__horizontal_well", "")
    hw = _load_v35pfv2_raw(p)
    kn = hw[hw["TVT_input"].notna()]
    ev = hw[hw["TVT_input"].isna()]
    if len(ev) == 0 or len(kn) < 10:
        continue
    last = kn.iloc[-1]
    parts.append(pd.DataFrame({
        "id": [f"{wid}_{int(i)}" for i in ev.index],
        "well": wid,
        "last_known_tvt": np.float64(last["TVT_input"]),
        "md_since": ev["MD"].to_numpy(np.float64) - np.float64(last["MD"]),
    }))
test_df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["id", "well", "last_known_tvt", "md_since"])
test_df["id"] = test_df["id"].astype(str)
base_te = test_df["last_known_tvt"].to_numpy(np.float64)
print(f"[OUBANK] eval rows={len(test_df)}", flush=True)

_pfv2_parts = []
for k, wid in enumerate(test_wids, 1):
    part = _pfv2_one_well(wid)
    if part is not None:
        _pfv2_parts.append(part)
    if k % 25 == 0 or k == len(test_wids):
        print(f"[OUBANK] pf_ancc feature {k}/{len(test_wids)}", flush=True)
_pfv2_df = pd.concat(_pfv2_parts, ignore_index=True) if _pfv2_parts else pd.DataFrame(columns=["id", "pf_ancc"])
PF35V2_MAP = dict(zip(_pfv2_df["id"].astype(str).values, _pfv2_df["pf_ancc"].astype(np.float64).values))
print(f"[OUBANK] PF35V2_MAP rows={len(PF35V2_MAP)}", flush=True)

train_wids = [p.stem.replace("__horizontal_well", "") for p in sorted((CFG.dataset_path / "train").glob("*__horizontal_well.csv"))]
FI = FormationPlaneKNN(train_wids, CFG.dataset_path / "train")
print(f"[OUBANK] FormationPlaneKNN train wells={len(FI.df)}", flush=True)

_tw = test_wids
_t0 = time.time()
_bsel_items = Parallel(n_jobs=4, prefer="processes")(delayed(_bsel_one)(w) for w in _tw)
_bsel = {wid: (bm, var, lt) for wid, bm, var, lt in _bsel_items}
print(f"[OUBANK] beam selector done {time.time()-_t0:.1f}s", flush=True)
_ws = [d for d in (_prep(w) for w in _tw) if d is not None]
_ws.sort(key=lambda d: d["ne"])
print(f"[OUBANK] PF wells prepared={len(_ws)} profiles={len(_PROFS)} seeds={_PFS} particles={_PFN}", flush=True)
_pf = {}
for _c in range(0, len(_ws), _PFCW):
    _pf.update(_pf_mp(_ws[_c:_c+_PFCW], _PROFS))
    if _PFDEV == "cuda":
        torch.cuda.synchronize()
    print(f"[OUBANK] PF chunk {min(_c+_PFCW, len(_ws))}/{len(_ws)} elapsed={time.time()-_t0:.1f}s", flush=True)

PFV = {pid: {} for pid in _PIDS}
PFSTD = {pid: {} for pid in _PIDS}
for w in _ws:
    bm, var, lt = _bsel[w["wid"]]
    cell = _pf[w["wid"]]
    pbs = cell["mean"]
    pstd = cell["std"]
    ne = w["ne"]
    sc, _, _ = _pv(var)
    skey = f"pf_scale_{sc:g}"
    skey = skey if skey in pstd else "pf_scale_8"
    rids = [f"{w['wid']}_{int(i)}" for i in w["ev_idx"]]
    for pi, pid in enumerate(_PIDS):
        pbs_p = {k: v[pi, :ne] for k, v in pbs.items()}
        fin = _av(var, pbs_p, bm, lt)
        sdp = pstd[skey][pi, :ne]
        for r_, fv, sv in zip(rids, fin, sdp):
            PFV[pid][r_] = float(fv)
            PFSTD[pid][r_] = float(sv)
del _pf
gc.collect()
print("[OUBANK] PFV/PFSTD maps ready", flush=True)

_BANK4 = ("BIOU00", "BIOUP06", "BIOUP42", "BIOUP51")
_OU4 = ("OU_P00", "OU_P06", "OU_P23", "OU_P26")

def _align_component(rows, name):
    abs_pred = test_df["id"].astype(str).map(rows).to_numpy(np.float64)
    miss = ~np.isfinite(abs_pred)
    if miss.any():
        abs_pred[miss] = base_te[miss]
    frame = pd.DataFrame({"id": test_df["id"].astype(str).values, name: abs_pred})
    out = pd.DataFrame({"id": sample_ids}).merge(frame, on="id", how="left", validate="one_to_one")
    if out[name].isna().any():
        bad = out.loc[out[name].isna(), "id"].head(10).tolist()
        raise ValueError(f"{name} missing after sample alignment: {bad}")
    print(f"[OUBANK] {name} coverage={(~miss).mean()*100:.1f}%", flush=True)
    return out[name].to_numpy(np.float64)

_m12 = pd.DataFrame({"id": sample_ids})
for _nm, _ckp, _wsharp, _dst in [
    ("oubank_model1", loc("nn_ckpt_m1.pt", prefer="rogii-oubank-ckpts"), "bank_biou4_ou4", "m1_tvt"),
    ("oubank_model2", loc("nn_ckpt_m2.pt", prefer="rogii-oubank-ckpts"), "ou4", "m2_tvt"),
]:
    print(f"[OUBANK] loading {_nm}: {_ckp}", flush=True)
    _c = torch.load(_ckp, map_location="cpu", weights_only=False)
    _profs = _c["profiles"]
    _nf = int(_c["config"]["N_FEATS"])
    _act = _c["config"].get("ACTIVATION", "gelu")
    _dstd = float(_c["norm"]["dtvt_std"])
    _mn = torch.tensor(np.asarray(_c["norm"]["mean"], np.float32), device=DEVICE)
    _sd = torch.tensor(np.asarray(_c["norm"]["std"], np.float32), device=DEVICE)
    _arch = _detect_arch(_c["seed_fold_state_dicts"][0]["sd"])
    _by_seed = {}
    for d in _c["seed_fold_state_dicts"]:
        mm = _DTCNpfb(_nf, act=_act, **_arch).to(DEVICE)
        mm.load_state_dict(d["sd"])
        mm.eval()
        _by_seed.setdefault(int(d["seed"]), []).append(mm)
    print(f"[OUBANK] {_nm}: N_FEATS={_nf} act={_act} arch={_arch} seeds={sorted(_by_seed)}", flush=True)
    _rows = {sd: {} for sd in _by_seed}
    _skip = 0
    _t_model = time.time()
    with torch.no_grad():
        for k, p in enumerate(test_paths, 1):
            w = _load_pfbank(p, _profs)
            if w is None:
                _skip += 1
                continue
            if _wsharp == "bank_biou4_ou4":
                _mid = [_v99_sharp_channels(p, w, _BANK4 + _OU4)]
            elif _wsharp == "ou4":
                _mid = [_v99_sharp_channels(p, w, _OU4)]
            else:
                _mid = []
            feats = np.concatenate([w["feats"]] + _mid + [_form_channels(p)], axis=1).astype(np.float32)
            if feats.shape[1] != _nf:
                raise ValueError(f"{_nm} feature mismatch {feats.shape[1]} != {_nf}")
            xx = ((torch.tensor(feats, device=DEVICE) - _mn) / _sd).t().unsqueeze(0).contiguous()
            pm = torch.tensor(w["pm"], device=DEVICE)
            for sdv in _by_seed:
                acc = np.zeros(len(w["evi"]), np.float64)
                for mm in _by_seed[sdv]:
                    dn = mm(xx)
                    rec = w["atvt"] + torch.cumsum(dn * _dstd * pm, dim=0)
                    acc += rec.cpu().numpy()[w["evi"]].astype(np.float64)
                acc /= len(_by_seed[sdv])
                for r_, v_ in zip(w["rids"], acc):
                    _rows[sdv][r_] = float(v_)
            if k % 25 == 0 or k == len(test_paths):
                print(f"[OUBANK] {_nm} wells {k}/{len(test_paths)} elapsed={time.time()-_t_model:.1f}s", flush=True)
    _seed = 42 if 42 in _rows else sorted(_rows)[0]
    if _seed != 42:
        print(f"[OUBANK] WARNING {_nm}: seed 42 not found, using seed {_seed}", flush=True)
    _m12[_dst] = _align_component(_rows[_seed], _dst)
    del _c, _by_seed, _rows
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    gc.collect()

print(f"[OUBANK] m1/m2 computed {_m12.shape}", flush=True)
'''
    exec(compile(driver, "v136_oubank_driver.py", "exec"), g, g)
    _m12 = g["_m12"]
    _m12.to_parquet(_blend_output_dir / "m1_m2_component_predictions.parquet", index=False)
    _m12.to_csv(_blend_output_dir / "m1_m2_component_predictions.csv", index=False)
    print(f"[OUBANK] saved computed m1/m2 {_m12.shape}", flush=True)
    return _normalize_m1m2_frame(_m12)


if _need_component("m1") or _need_component("m2") or FORCE_RUN.get("m1m2", False):
    m12 = _run_v136_oubank_m1m2()
    _save_table(m12, "m1_m2_component_predictions_normalized")
    if _need_component("m1"):
        components["m1"] = _as_submission(m12, "m1_tvt", "m1_submission")
    if _need_component("m2"):
        components["m2"] = _as_submission(m12, "m2_tvt", "m2_submission")
else:
    print("m1/m2 skipped: zero weights", flush=True)


active = [(name, float(w)) for name, w in WEIGHTS.items() if abs(float(w)) > 1e-12]
missing = [name for name, _ in active if name not in components]
if missing:
    raise RuntimeError(f"Active blend components missing: {missing}. Available={sorted(components)}")
weights = np.array([w for _, w in active], dtype=float)
if NORMALIZE_WEIGHTS:
    s = float(weights.sum())
    if abs(s) < 1e-12:
        raise ValueError("weights sum is zero")
    weights = weights / s
print("active normalized weights:", {name: float(w) for (name, _), w in zip(active, weights)}, flush=True)

blend_components = sample_submission[["id"]].copy()
for name, _ in active:
    comp = components[name].rename(columns={_target_col: f"{name}_tvt"})
    blend_components = blend_components.merge(comp[["id", f"{name}_tvt"]], on="id", how="left", validate="one_to_one")
    if blend_components[f"{name}_tvt"].isna().any():
        raise ValueError(f"NaN after merge for {name}")

pred = np.zeros(len(blend_components), dtype=float)
for (name, _), w in zip(active, weights):
    pred += float(w) * blend_components[f"{name}_tvt"].to_numpy(float)
blend_components["blend_tvt_raw"] = pred
pred = pred + float(FINAL_TVT_SHIFT)
blend_components["blend_tvt"] = pred
print(f"applied FINAL_TVT_SHIFT={FINAL_TVT_SHIFT:+.4f} ft", flush=True)
_save_table(blend_components, "configurable_blend_components")

submission = sample_submission[["id"]].merge(
    blend_components[["id", "blend_tvt"]].rename(columns={"blend_tvt": _target_col}),
    on="id",
    how="left",
    validate="one_to_one",
)
if submission[_target_col].isna().any():
    missing = submission.loc[submission[_target_col].isna(), "id"].head(10).tolist()
    raise ValueError(f"final submission missing predictions: {missing}")
submission = submission[sample_submission.columns]
submission_path = _blend_output_dir / "submission.csv"
submission.to_csv(submission_path, index=False)

summary = {
    "weights_raw": {k: float(v) for k, v in WEIGHTS.items()},
    "weights_active_normalized": {name: float(w) for (name, _), w in zip(active, weights)},
    "final_tvt_shift": float(FINAL_TVT_SHIFT),
    "components": sorted(components.keys()),
    "rows": int(len(submission)),
    "target_col": _target_col,
}
(_blend_output_dir / "configurable_blend_summary.json").write_text(_blend_json.dumps(summary, indent=2), encoding="utf-8")
print("Saved final", submission_path, submission.shape, flush=True)
print(_blend_json.dumps(summary, indent=2), flush=True)


from pathlib import Path as _FinalPath
import os as _final_os, sys as _final_sys, glob as _final_glob
import numpy as _final_np
import pandas as _final_pd

_final_work = _FinalPath('/kaggle/working') if _FinalPath('/kaggle/working').is_dir() else _FinalPath.cwd()
_our_submission_path = _final_work / 'submission.csv'
if not _our_submission_path.is_file():
    raise FileNotFoundError(f'Expected our shifted submission at {_our_submission_path}')
_our_sub = _final_pd.read_csv(_our_submission_path)
_our_cols = [c for c in _our_sub.columns if c != 'id']
if len(_our_cols) != 1:
    raise ValueError(f'Expected one prediction column in our submission, got {_our_cols}')
_final_target_col = _our_cols[0]
_our_sub.to_csv(_final_work / 'our_m1w016_sub12exp165042_shift04_submission.csv', index=False)
print('loaded our shifted submission:', _our_sub.shape, 'target_col=', _final_target_col)

def _final_find_comp_root():
    candidates = []
    if 'DATA_ROOT' in globals():
        candidates.append(_FinalPath(DATA_ROOT))
    candidates.extend([
        _FinalPath('/kaggle/input/competitions/rogii-wellbore-geology-prediction'),
        _FinalPath('/kaggle/input/rogii-wellbore-geology-prediction'),
        _FinalPath.cwd(),
        _FinalPath.cwd() / 'rogii-wellbore-geology-prediction',
    ])
    candidates.extend(_FinalPath.cwd().parents)
    for root in candidates:
        if (root / 'test').is_dir() and (root / 'sample_submission.csv').is_file():
            return root.resolve()
    raise FileNotFoundError('Could not find ROGII competition root with test/ and sample_submission.csv')

_FINAL_COMP = _final_find_comp_root()
_FINAL_TEST_DIR = str(_FINAL_COMP / 'test')
_FINAL_SAMPLE = str(_FINAL_COMP / 'sample_submission.csv')
_final_sample = _final_pd.read_csv(_FINAL_SAMPLE)[['id']]
print('competition root:', _FINAL_COMP, 'sample rows:', len(_final_sample))

def _final_find_support(required_files, preferred=()):
    candidates = []
    for p in preferred:
        candidates.append(_FinalPath(p))
    kin = _FinalPath('/kaggle/input')
    if kin.exists():
        candidates.extend(sorted(kin.glob('*')))
        candidates.extend(sorted(kin.glob('*/*')))

        for hit in sorted(kin.rglob(required_files[0])):
            candidates.append(hit.parent)
    seen = set()
    for path in candidates:
        try:
            key = str(path.resolve())
        except Exception:
            key = str(path)
        if key in seen:
            continue
        seen.add(key)
        if all((path / f).is_file() for f in required_files):
            return path.resolve()
    print('support search failed for', required_files)
    if kin.exists():
        print('/kaggle/input roots:', [p.name for p in sorted(kin.glob('*'))[:80]])
    raise FileNotFoundError(f'Could not find support dir containing {required_files}')

def _final_blend_and_write(alexey_sub, alexey_name, w_our, w_alexey):
    alexey_cols = [c for c in alexey_sub.columns if c != 'id']
    if len(alexey_cols) != 1:
        raise ValueError(f'{alexey_name}: expected one pred column, got {alexey_cols}')
    alexey_col = alexey_cols[0]
    aligned = _final_sample.merge(
        _our_sub[['id', _final_target_col]].rename(columns={_final_target_col: 'our_tvt'}),
        on='id', how='left', validate='one_to_one'
    ).merge(
        alexey_sub[['id', alexey_col]].rename(columns={alexey_col: 'alexey_tvt'}),
        on='id', how='left', validate='one_to_one'
    )
    if aligned['our_tvt'].isna().any():
        raise ValueError(f'our submission missing {int(aligned["our_tvt"].isna().sum())} sample ids')
    if aligned['alexey_tvt'].isna().any():
        miss = int(aligned['alexey_tvt'].isna().sum())
        print(f'{alexey_name}: filling {miss} missing alexey ids by our_tvt fallback')
        aligned['alexey_tvt'] = aligned['alexey_tvt'].fillna(aligned['our_tvt'])
    aligned['blend_tvt'] = float(w_our) * aligned['our_tvt'].astype(float) + float(w_alexey) * aligned['alexey_tvt'].astype(float)
    final = aligned[['id', 'blend_tvt']].rename(columns={'blend_tvt': _final_target_col})
    comp_path = _final_work / f'blend_components_{alexey_name}.csv'
    aligned.to_csv(comp_path, index=False)
    try:
        aligned.to_parquet(_final_work / f'blend_components_{alexey_name}.parquet', index=False)
    except Exception as exc:
        print('component parquet skipped:', repr(exc))
    final.to_csv(_final_work / 'submission.csv', index=False)
    print(f'FINAL BLEND {alexey_name}: w_our={w_our:.12f}, w_{alexey_name}={w_alexey:.12f}')
    print('wrote final submission:', _final_work / 'submission.csv', final.shape)
    print('component stats:')
    print(aligned[['our_tvt','alexey_tvt','blend_tvt']].describe().to_string())
    return final, aligned

print('\n=== RUN ALEXEY SUB22 qadj-soft-blend ===', flush=True)
_SUPPORT_DIR = _final_find_support(
    ['deploy_pf.py', 'deploy_pf_lw.py'],
    preferred=(
        '/kaggle/input/datasets/drzoidberg/sub22-support',
        '/kaggle/input/sub22-support',
    ),
)
print('SUB22 SUPPORT_DIR:', _SUPPORT_DIR)
_final_sys.path.insert(0, str(_SUPPORT_DIR))
import deploy_pf as _sub22_dp, deploy_pf_lw as _sub22_dplw

ctx = _sub22_dp.load_ctx(str(_SUPPORT_DIR), str(_SUPPORT_DIR))
wells = [
    _final_os.path.basename(p).split('__')[0]
    for p in sorted(_final_glob.glob(_final_os.path.join(_FINAL_TEST_DIR, '*__horizontal_well.csv')))
]
print('test wells:', len(wells))
GS_SET = [30, 43, 60]
W_SOFT, W_AHA = 0.8, 0.2
USE_E = False
soft, WD = _sub22_dplw.run_gs_composite_lw(
    ctx, _FINAL_TEST_DIR, wells, GS_SET,
    kappa='kvar', nudge=False, mode='soft', use_E=USE_E,
)
aha = _sub22_dplw.run_adjhauto_lw(ctx, _FINAL_TEST_DIR, wells, WD=WD)
rows = []
for w in wells:
    if w not in WD:
        continue
    idx = _final_np.where(WD[w]['ev'])[0]
    n = len(idx)
    sp = soft[w][1] if (w in soft and len(soft[w][1]) == n) else None
    a = aha[w][1] if (w in aha and len(aha[w][1]) == n) else None
    if sp is not None and a is not None:
        blend = W_SOFT * sp + W_AHA * a
    elif sp is not None:
        blend = sp
    elif a is not None:
        blend = a
    else:
        continue
    for j, ri in enumerate(idx):
        rows.append((f'{w}_{ri}', float(blend[j])))
print('sub22 rows:', len(rows))
_sub22 = _final_pd.DataFrame(rows, columns=['id', _final_target_col])
_sub22 = _final_sample.merge(_sub22, on='id', how='left')
print('sub22 matched %d/%d ids; NaN=%d' % (_sub22[_final_target_col].notna().sum(), len(_sub22), _sub22[_final_target_col].isna().sum()))
if _sub22[_final_target_col].isna().any():
    _sub22[_final_target_col] = _sub22[_final_target_col].fillna(_sub22[_final_target_col].median())
_sub22.to_csv(_final_work / 'alexey_sub22_qadj_soft_submission.csv', index=False)
_final, _aligned = _final_blend_and_write(
    _sub22,
    alexey_name='sub22_qadj_soft',
    w_our=0.85,
    w_alexey=0.15,
)


from pathlib import Path as _OuterPath
import pandas as _outer_pd

_outer_work = _OuterPath('/kaggle/working')
_outer_second = _outer_pd.read_csv(_outer_work / 'submission.csv')
_outer_second.to_csv(_outer_work / 'cv58_stack_branch_submission.csv', index=False)
print('preserved CV-5.8 stack branch:', _outer_second.shape)


import gc
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

COMP = Path("/kaggle/input/competitions/rogii-wellbore-geology-prediction")

def find_dataset(slug, required=None):
    hits = []
    for root in (Path("/kaggle/input"), Path("/kaggle/input/datasets")):
        if root.exists():
            hits.extend(p for p in root.rglob("*") if p.is_dir() and p.name == slug)
    if not hits:
        raise FileNotFoundError(f"Dataset not found: {slug}")
    if required:
        for p in hits:
            if (p / required).exists():
                return p
    return hits[0]

EDITOR_ROOT = find_dataset("rogii-typewell-editor-v2", "rogii_typewell_editor_v2_bundle") / "rogii_typewell_editor_v2_bundle"
EDITOR_CODE = EDITOR_ROOT / "code" / "experiments" / "exp_256_typewell_editor"
EDITOR_WEIGHTS = EDITOR_ROOT / "results" / "raw_affine_prefix_5fold_20ep"
SUPPORT_DIR = find_dataset("sub-30-32-support")
if not (SUPPORT_DIR / "deploy_sota.py").exists():
    SUPPORT_DIR = next(SUPPORT_DIR.rglob("deploy_sota.py")).parent

sys.path.insert(0, str(EDITOR_CODE))
sys.path.insert(0, str(SUPPORT_DIR))
import infer_typewell_editor_v2 as editor_infer
from deploy_sota import run_deployment

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
assert COMP.exists()
assert (EDITOR_WEIGHTS / "fold0" / "best.pt").exists()
print("COMP:", COMP)
print("EDITOR_ROOT:", EDITOR_ROOT)
print("EDITOR_WEIGHTS:", EDITOR_WEIGHTS)
print("SUPPORT_DIR:", SUPPORT_DIR)
print("DEVICE:", DEVICE)


RUNTIME_ROOT = Path("/kaggle/working/typewell_editor_runtime")
RAW_ROOT = RUNTIME_ROOT / "competition" / "raw" / "train"
AVERAGE_TEST = Path("/kaggle/working/sub31_average_typewell_test")
TYPEWELL_OUT = Path("/kaggle/working/average_typewell_outputs")
for p in (RAW_ROOT, AVERAGE_TEST, TYPEWELL_OUT):
    p.mkdir(parents=True, exist_ok=True)

horizontal_files = sorted((COMP / "test").glob("*__horizontal_well.csv"))
well_ids = [p.name.split("__")[0] for p in horizontal_files]
if not well_ids:
    raise FileNotFoundError("No test wells found")
for wid in well_ids:
    for suffix in ("__horizontal_well.csv", "__typewell.csv"):
        src = COMP / "test" / f"{wid}{suffix}"
        if not src.exists():
            raise FileNotFoundError(src)
        shutil.copy2(src, RAW_ROOT / src.name)
        shutil.copy2(src, AVERAGE_TEST / src.name)
print("test wells:", len(well_ids))


@torch.no_grad()
def make_corrected_typewell(wid):
    w = editor_infer.load_inference_well(RUNTIME_ROOT, wid)
    native_tvt = w["t_tvt"]
    fold_curves = []
    for fold in range(5):
        ckpt = torch.load(EDITOR_WEIGHTS / f"fold{fold}" / "best.pt", map_location=DEVICE, weights_only=False)
        a, b = editor_infer.robust_affine(w)
        wf = dict(w)
        wf["t_gr"] = (a + b * w["t_gr"]).astype(np.float32)
        item = editor_infer.exp.build_item(wf, int(ckpt["grid"]), float(ckpt["center"]), float(ckpt["scale"]))
        model = editor_infer.exp.TypewellEditor().to(DEVICE)
        model.load_state_dict(ckpt["model"])
        model.eval()
        x = torch.from_numpy(item["x"])[None].to(DEVICE)
        delta, _ = model(x)
        corrected_grid = (x[:, 0] + delta)[0].cpu().numpy() * float(ckpt["scale"]) + float(ckpt["center"])
        t0, t1 = float(native_tvt[0]), float(native_tvt[-1])
        u_native = np.clip((native_tvt - t0) / max(t1 - t0, 1e-3), 0.0, 1.0)
        curve = editor_infer.exp.interp_np(u_native, item["grid_u"], corrected_grid).astype(np.float32)
        fold_curves.append(curve)
        pd.DataFrame({"TVT": native_tvt, "GR": curve}).to_csv(
            TYPEWELL_OUT / f"{wid}__neural_typewell_fold{fold}.csv", index=False)
        del model, ckpt, x, delta
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
    neural_gr = np.mean(np.stack(fold_curves), axis=0)
    original_path = COMP / "test" / f"{wid}__typewell.csv"
    original = pd.read_csv(original_path)
    if not {"TVT", "GR"}.issubset(original.columns):
        raise ValueError(f"Unexpected typewell columns for {wid}: {list(original.columns)}")
    original_tvt = original["TVT"].to_numpy(np.float64)
    neural_on_original = np.interp(original_tvt, native_tvt, neural_gr).astype(np.float64)
    average = original.copy()
    average["GR"] = 0.5 * original["GR"].to_numpy(np.float64) + 0.5 * neural_on_original
    neural = original.copy()
    neural["GR"] = neural_on_original
    neural_out = TYPEWELL_OUT / f"{wid}__neural_typewell_ensemble.csv"
    average_out = TYPEWELL_OUT / f"{wid}__average_typewell_50_50.csv"
    neural.to_csv(neural_out, index=False)
    average.to_csv(average_out, index=False)
    if not np.allclose(average["TVT"], original["TVT"], rtol=0, atol=0):
        raise AssertionError(f"TVT grid changed for {wid}")
    for col in original.columns:
        if col not in ("GR", "TVT") and not average[col].equals(original[col]):
            raise AssertionError(f"Column {col} changed for {wid}")
    expected_gr = 0.5 * original["GR"].to_numpy(np.float64) + 0.5 * neural_on_original
    if not np.allclose(average["GR"], expected_gr, rtol=0, atol=1e-10):
        raise AssertionError(f"50/50 GR check failed for {wid}")
    return average_out

for i, wid in enumerate(well_ids, 1):
    out = make_corrected_typewell(wid)
    shutil.copy2(out, AVERAGE_TEST / f"{wid}__typewell.csv")
    print(f"average typewell {i}/{len(well_ids)}: {wid}", flush=True)


out_csv = "/kaggle/working/sub31_average_typewell_raw.csv"
submission = run_deployment(
    test_dir=str(AVERAGE_TEST),
    mode="static",
    out_csv=out_csv,
    support_dir=str(SUPPORT_DIR),
    batch_fwd=64,
    batch_bwd=256,
    hmm_budget=8e9,
    sample_submission=str(COMP / "sample_submission.csv"),
)


sample = pd.read_csv(COMP / "sample_submission.csv")
if "id" not in sample.columns or "tvt" not in sample.columns:
    raise ValueError(f"Unexpected sample_submission columns: {list(sample.columns)}")
raw = submission[["id", "tvt"]].copy()
final_submission = sample[["id"]].merge(raw, on="id", how="left", validate="one_to_one")
final_submission = final_submission[["id", "tvt"]]
if len(final_submission) != len(sample):
    raise ValueError("Final submission row count differs from sample_submission")
if not final_submission["id"].equals(sample["id"]):
    raise ValueError("Final submission ids are not in sample_submission order")
if final_submission["tvt"].isna().any():
    raise ValueError("Final submission contains NaN TVT values")

out_csv = "/kaggle/working/submission.csv"
final_submission.to_csv(out_csv, index=False)
print("saved canonical submission:", out_csv)
print("columns:", list(final_submission.columns), "shape:", final_submission.shape)


manifest = {
    "editor": "rogii_typewell_editor_v2",
    "editor_weights": str(EDITOR_WEIGHTS),
    "editor_folds": 5,
    "sub31_support": str(SUPPORT_DIR),
    "test_wells": len(well_ids),
    "typewell_formula": "GR_average = 0.5*GR_original + 0.5*GR_neural_V2",
    "average_typewell_dir": str(TYPEWELL_OUT),
    "submission": out_csv,
    "device": DEVICE,
}
Path("/kaggle/working/sub31_average_typewell_manifest.json").write_text(json.dumps(manifest, indent=2))
print(json.dumps(manifest, indent=2))


from pathlib import Path as _BlendPath
import json as _blend_json
import numpy as _blend_np
import pandas as _blend_pd

W_SUB31_AVG = 0.30
W_CV58_STACK = 0.70
_blend_work = _BlendPath('/kaggle/working')
_blend_sample = _blend_pd.read_csv(COMP / 'sample_submission.csv')
_blend_target_cols = [c for c in _blend_sample.columns if c != 'id']
if _blend_target_cols != ['tvt']:
    raise ValueError(f'Unexpected sample submission columns: {list(_blend_sample.columns)}')

_sub31_avg = _blend_pd.read_csv(_blend_work / 'submission.csv')
_sub31_avg.to_csv(_blend_work / 'sub31_average_typewell_branch_submission.csv', index=False)
_cv58 = _blend_pd.read_csv(_blend_work / 'cv58_stack_branch_submission.csv')

def _canonical_branch(frame, name):
    if list(frame.columns) != ['id', 'tvt']:
        raise ValueError(f'{name}: expected exactly id,tvt; got {list(frame.columns)}')
    if frame['id'].duplicated().any():
        raise ValueError(f'{name}: duplicate ids')
    aligned = _blend_sample[['id']].merge(frame, on='id', how='left', validate='one_to_one')
    if aligned['tvt'].isna().any():
        missing = aligned.loc[aligned['tvt'].isna(), 'id'].head(10).tolist()
        raise ValueError(f'{name}: missing or NaN predictions: {missing}')
    if not aligned['id'].equals(_blend_sample['id']):
        raise ValueError(f'{name}: id order mismatch')
    return aligned

_sub31_avg = _canonical_branch(_sub31_avg, 'sub31_average_typewell')
_cv58 = _canonical_branch(_cv58, 'cv58_stack')

_components = _blend_sample[['id']].copy()
_components['sub31_average_typewell_tvt'] = _sub31_avg['tvt'].to_numpy(float)
_components['cv58_stack_tvt'] = _cv58['tvt'].to_numpy(float)
_components['final_tvt'] = (
    W_SUB31_AVG * _components['sub31_average_typewell_tvt']
    + W_CV58_STACK * _components['cv58_stack_tvt']
)
_expected = 0.30 * _components['sub31_average_typewell_tvt'] + 0.70 * _components['cv58_stack_tvt']
if not _blend_np.allclose(_components['final_tvt'], _expected, rtol=0.0, atol=1e-12):
    raise AssertionError('Final blend formula verification failed')
if not _blend_np.isfinite(_components['final_tvt']).all():
    raise ValueError('Final blend contains non-finite predictions')

_final_submission = _components[['id', 'final_tvt']].rename(columns={'final_tvt': 'tvt'})
_final_submission.to_csv(_blend_work / 'submission.csv', index=False)
_components.to_csv(_blend_work / 'blend_components_sub31avg30_cv58stack70.csv', index=False)
try:
    _components.to_parquet(_blend_work / 'blend_components_sub31avg30_cv58stack70.parquet', index=False)
except Exception as exc:
    print('component parquet skipped:', repr(exc))

_summary = {
    'sub31_average_typewell_weight': W_SUB31_AVG,
    'cv58_stack_weight': W_CV58_STACK,
    'rows': int(len(_final_submission)),
    'formula': '0.30*sub31_average_typewell + 0.70*cv58_stack',
}
(_blend_work / 'blend_summary.json').write_text(_blend_json.dumps(_summary, indent=2), encoding='utf-8')
print(_blend_json.dumps(_summary, indent=2))
print('saved final submission:', _blend_work / 'submission.csv', _final_submission.shape)
