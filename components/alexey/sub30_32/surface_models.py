# Alexey Zakharov (drzoidberg), ROGII team solution.
"""Structural-surface models with trajectory-gradient observations."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

try:
    import cupy as cp
except Exception:
    cp = None


def _integrate(md: np.ndarray, rate: np.ndarray) -> np.ndarray:
    out = np.zeros(len(md), dtype=float)
    if len(md) > 1:
        out[1:] = np.cumsum(0.5 * (rate[1:] + rate[:-1]) * np.diff(md))
    return out


def _slope(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 8:
        return float("nan")
    xc = x - x.mean()
    den = float(xc @ xc)
    if den <= 0:
        return float("nan")
    return float(xc @ (y - y.mean()) / den)


def _fallback_md(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Approximate measured depth from an ordered XY track."""
    distance = np.r_[0.0, np.cumsum(np.hypot(np.diff(x), np.diff(y)))]


    return np.maximum.accumulate(distance + np.arange(len(distance)) * 1e-4)


class DirectionalGradientModel:
    """2:1 anisotropic local GP for directional observations of a gradient."""

    def __init__(
        self,
        xy: np.ndarray,
        direction: np.ndarray,
        values: np.ndarray,
        wells: np.ndarray,
        range_ft: float = 8000.0,
        noise_ratio: float = 0.40,
        anisotropy: float = 2.0,
        neighbors: int = 32,
    ):
        self.xy = np.asarray(xy, float)
        self.a = np.asarray(direction, float)
        values = np.asarray(values, float)
        wells = np.asarray(wells)
        self.range_ft = float(range_ft)
        self.noise_ratio = float(noise_ratio)
        self.anisotropy = float(anisotropy)
        self.neighbors = int(neighbors)

        _, inv, counts = np.unique(wells, return_inverse=True, return_counts=True)
        weight = 1.0 / counts[inv]
        normal = self.a.T @ (weight[:, None] * self.a) + 1e-8 * np.eye(2)
        self.g0 = np.linalg.solve(normal, self.a.T @ (weight * values))
        residual = values - self.a @ self.g0
        self.scale = float(np.sqrt(np.sum(weight * residual**2) / np.sum(weight)))
        self.scale = max(self.scale, 1e-6)
        self.resid = residual / self.scale

        norm = float(np.linalg.norm(self.g0))
        self.dip_axis = self.g0 / norm if norm > 1e-12 else np.array([1.0, 0.0])
        self.strike_axis = np.array([-self.dip_axis[1], self.dip_axis[0]])
        self.metric_xy = self._metric(self.xy)
        self.tree = cKDTree(self.metric_xy)

    def _metric(self, xy: np.ndarray) -> np.ndarray:
        xy = np.asarray(xy, float)
        return np.c_[xy @ self.dip_axis, (xy @ self.strike_axis) / self.anisotropy]

    def predict(
        self, xy: np.ndarray, direction: np.ndarray, chunk: int = 2000
    ) -> tuple[np.ndarray, np.ndarray]:
        xy = np.asarray(xy, float)
        direction = np.asarray(direction, float)
        k = min(self.neighbors, len(self.xy))
        _, ids = self.tree.query(self._metric(xy), k=k, workers=-1)
        ids = np.asarray(ids, dtype=np.int64)
        if ids.ndim == 1:
            ids = ids[:, None]
        pred = np.empty(len(xy), float)
        variance = np.empty(len(xy), float)
        use_gpu = cp is not None and getattr(self, "device", "cuda") == "cuda"

        for start in range(0, len(xy), chunk):
            stop = min(start + chunk, len(xy))
            jj = ids[start:stop]
            sx_np = self.metric_xy[jj].astype(np.float32)
            sa_np = self.a[jj].astype(np.float32)
            sr_np = self.resid[jj].astype(np.float32)
            qx_np = self._metric(xy[start:stop]).astype(np.float32)
            qa_np = direction[start:stop].astype(np.float32)

            xp = cp if use_gpu else np
            sx = xp.asarray(sx_np)
            sa = xp.asarray(sa_np)
            sr = xp.asarray(sr_np)
            qx = xp.asarray(qx_np)
            qa = xp.asarray(qa_np)
            delta = sx[:, :, None, :] - sx[:, None, :, :]
            distance = xp.sqrt(xp.sum(delta * delta, axis=3))
            adot = xp.sum(sa[:, :, None, :] * sa[:, None, :, :], axis=3)
            matrix = xp.exp(-3.0 * distance / self.range_ft) * adot
            diag = xp.arange(k)
            matrix[:, diag, diag] += self.noise_ratio + 1e-5
            qdistance = xp.sqrt(xp.sum((qx[:, None, :] - sx) ** 2, axis=2))
            qdot = xp.sum(qa[:, None, :] * sa, axis=2)
            rhs = xp.exp(-3.0 * qdistance / self.range_ft) * qdot
            try:
                alpha = xp.linalg.solve(matrix, sr[..., None])[..., 0]
            except Exception:
                matrix[:, diag, diag] += 1e-3
                alpha = xp.linalg.solve(matrix, sr[..., None])[..., 0]
            correction = xp.sum(rhs * alpha, axis=1)
            solved_rhs = xp.linalg.solve(matrix, rhs[..., None])[..., 0]
            base_var = xp.sum(qa * qa, axis=1)
            var = xp.maximum(base_var - xp.sum(rhs * solved_rhs, axis=1), 0.0)
            if use_gpu:
                correction = cp.asnumpy(correction)
                var = cp.asnumpy(var)
            pred[start:stop] = direction[start:stop] @ self.g0 + self.scale * correction
            variance[start:stop] = self.scale**2 * var
        return pred, variance


class TrajectoryDipModel:
    """Global trajectory-gradient GP plus Marathon-density isotonic shrinkage."""

    def __init__(self, fit: dict, marathon_xy: np.ndarray, marathon_u: np.ndarray,
                 iso_x: np.ndarray, iso_y: np.ndarray):
        for key in ("g0", "dip_axis", "strike_axis", "range_ft", "noise_ratio", "alpha", "xy", "u"):
            setattr(self, key, np.asarray(fit[key], float) if key not in {"range_ft", "noise_ratio"}
                    else float(fit[key]))
        self.marathon_xy = np.asarray(marathon_xy, float)
        self.marathon_u = np.asarray(marathon_u, float)
        self.iso_x = np.asarray(iso_x, float)
        self.iso_y = np.asarray(iso_y, float)
        self.anisotropy = 2.0
        self.density_bandwidth_ft = 1000.0

    def predict100(self, xy: np.ndarray, direction: np.ndarray) -> np.ndarray:
        xy = np.atleast_2d(np.asarray(xy, float))
        direction = np.atleast_2d(np.asarray(direction, float))
        sx = np.c_[self.xy @ self.dip_axis,
                   (self.xy @ self.strike_axis) / self.anisotropy]
        qx = np.c_[xy @ self.dip_axis,
                   (xy @ self.strike_axis) / self.anisotropy]
        distance = np.sqrt(np.sum((qx[:, None, :] - sx[None, :, :]) ** 2, axis=2))
        dot = direction @ self.u.T
        kernel = np.exp(-3.0 * distance / self.range_ft) * dot
        return direction @ self.g0 + kernel @ self.alpha

    def density(self, xy: np.ndarray, direction: np.ndarray) -> np.ndarray:
        xy = np.atleast_2d(np.asarray(xy, float))
        direction = np.atleast_2d(np.asarray(direction, float))
        distance = np.sqrt(np.sum((xy[:, None, :] - self.marathon_xy[None, :, :]) ** 2, axis=2))
        adot = np.abs(direction @ self.marathon_u.T)
        return np.sum(np.exp(-distance / self.density_bandwidth_ft) * adot**2, axis=1)

    def blend_lambda(self, density: np.ndarray) -> np.ndarray:
        return np.interp(np.asarray(density, float), self.iso_x, self.iso_y)


class _TrackSurface:
    kind = "track_surface"

    def __init__(self, base_surface):
        self.base_surface = base_surface

    def _base(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return self.base_surface.predict_batch(np.asarray(x, float), np.asarray(y, float))

    def predict_well(self, hw) -> tuple[np.ndarray, np.ndarray]:
        required = {"MD", "X", "Y", "TVT_input"}
        missing = required.difference(hw.columns)
        if missing:
            raise ValueError(f"horizontal well is missing columns: {sorted(missing)}")
        md = hw.MD.to_numpy(float)
        if np.any(np.diff(md) <= 0):
            raise ValueError("predict_well requires strictly increasing MD order")
        prefix = hw.TVT_input.notna().to_numpy()
        heel = int(np.flatnonzero(prefix)[-1]) if prefix.any() else 0
        return self._predict_track(md, hw.X.to_numpy(float), hw.Y.to_numpy(float), heel)

    def predict_batch(self, qx, qy, self_wid=None, chunk_size=60_000):
        """Predict gradients along an ordered trajectory."""
        x = np.asarray(qx, float)
        y = np.asarray(qy, float)
        return self._predict_track(_fallback_md(x, y), x, y, 0)


class IsotonicDipSurface(_TrackSurface):
    kind = "isotonic_marathon_dip"

    def __init__(self, base_surface, dip_model: TrajectoryDipModel,
                 min_horizontal_ft: float = 2000.0, max_abs_dip100: float = 8.0):
        super().__init__(base_surface)
        self.dip_model = dip_model
        self.min_horizontal_ft = float(min_horizontal_ft)
        self.max_abs_dip100 = float(max_abs_dip100)

    def _predict_track(self, md: np.ndarray, x: np.ndarray, y: np.ndarray,
                       heel: int) -> tuple[np.ndarray, np.ndarray]:
        base, kvar = self._base(x, y)
        result = np.asarray(base, float).copy()
        tail = np.arange(max(heel + 1, 1), len(md))
        if len(tail) < 20:
            return result, kvar
        xt, yt = x[tail], y[tail]
        ds = np.hypot(np.diff(xt), np.diff(yt))
        along = np.r_[0.0, np.cumsum(ds)]
        length = float(along[-1])
        chord = float(np.hypot(xt[-1] - xt[0], yt[-1] - yt[0]))
        if length < self.min_horizontal_ft or chord < 0.8 * length:
            return result, kvar
        direction = np.array([(xt[-1] - xt[0]) / chord, (yt[-1] - yt[0]) / chord])
        xy = np.array([float(np.mean(xt)), float(np.mean(yt))])
        direct100 = float(self.dip_model.predict100(xy, direction)[0])
        base100 = 100.0 * _slope(along, result[tail])
        if not np.isfinite(direct100 + base100) or abs(direct100) > self.max_abs_dip100:
            return result, kvar
        rho = float(self.dip_model.density(xy, direction)[0])
        lam = float(self.dip_model.blend_lambda(np.array([rho]))[0])
        delta100 = lam * (direct100 - base100)
        distance_from_heel = np.r_[0.0, np.cumsum(np.hypot(np.diff(x[heel:]), np.diff(y[heel:])))]
        result[heel:] += (delta100 / 100.0) * distance_from_heel
        return result, kvar


class RobustDirectDipSurface(_TrackSurface):
    kind = "shipped_robust_direct_dip"

    def __init__(self, base_surface, direct_model: DirectionalGradientModel,
                 blend_lambda: float, qmin: float, rate_cap_ft100: float,
                 drift_cap_ft: float):
        super().__init__(base_surface)
        self.direct_model = direct_model
        self.blend_lambda = float(blend_lambda)
        self.qmin = float(qmin)
        self.rate_cap_ft100 = float(rate_cap_ft100)
        self.drift_cap_ft = float(drift_cap_ft)

    def _predict_track(self, md: np.ndarray, x: np.ndarray, y: np.ndarray,
                       heel: int) -> tuple[np.ndarray, np.ndarray]:
        base, kvar = self._base(x, y)
        base = np.asarray(base, float)
        result = base.copy()
        ax = np.gradient(x, md)
        ay = np.gradient(y, md)
        direction = np.c_[ax, ay]
        direct_rate, variance = self.direct_model.predict(np.c_[x, y], direction)
        prior = self.direct_model.scale**2 * (ax**2 + ay**2)
        reliability = np.clip(1.0 - variance / np.maximum(prior, 1e-12), 0.0, 1.0)
        confidence = np.clip((reliability - self.qmin) / max(1.0 - self.qmin, 1e-9), 0.0, 1.0)
        base_rate = np.gradient(base, md)
        residual = np.clip(
            direct_rate - base_rate,
            -self.rate_cap_ft100 / 100.0,
            self.rate_cap_ft100 / 100.0,
        )
        correction = self.blend_lambda * _integrate(md[heel:], (confidence * residual)[heel:])
        if np.isfinite(self.drift_cap_ft):
            correction = np.clip(correction, -self.drift_cap_ft, self.drift_cap_ft)
        result[heel:] += correction
        return result, kvar


def prepare_model_for_gpu(model):
    """Move the nested scalar kriging model to CuPy without changing the pickle."""
    if cp is None:
        return model
    base = model.base_surface
    base._cx_t = cp.asarray(base.cx.astype(np.float32))
    base._cy_t = cp.asarray(base.cy.astype(np.float32))
    base._resid_t = cp.asarray(base.residuals.astype(np.float32))
    base.device = "cuda"
    if hasattr(model, "direct_model"):
        model.direct_model.device = "cuda"
    return model


def load_surface(path: str | Path, device: str = "cuda"):
    with open(path, "rb") as handle:
        model = pickle.load(handle)
    if device == "cuda":
        prepare_model_for_gpu(model)
    else:
        model.base_surface.device = "cpu"
        if hasattr(model, "direct_model"):
            model.direct_model.device = "cpu"
    return model
