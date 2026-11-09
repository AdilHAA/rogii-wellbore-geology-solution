"""Data preparation from medali1992 (Apache-2.0), adapted by Adil Khabibullin."""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter


SAVGOL_WINDOW = 31


def load_well(data_dir, well_id):
    base = Path(data_dir) / "test" / well_id
    horizontal = pd.read_csv(f"{base}__horizontal_well.csv")
    typewell = pd.read_csv(f"{base}__typewell.csv")
    required = {"GR", "TVT_input", "MD", "X", "Y", "Z"}
    missing = required - set(horizontal.columns)
    if missing:
        raise ValueError(f"{well_id}: missing horizontal columns: {sorted(missing)}")
    if not {"GR", "TVT"}.issubset(typewell.columns):
        raise ValueError(f"{well_id}: typewell needs GR and TVT columns")
    if len(horizontal) < SAVGOL_WINDOW:
        raise ValueError(f"{well_id}: fewer than {SAVGOL_WINDOW} rows")

    tvt_input = horizontal["TVT_input"].to_numpy(dtype=np.float32)
    known = np.isfinite(tvt_input)
    if not known.any():
        raise ValueError(f"{well_id}: no known TVT anchor")
    anchor_row = int(np.flatnonzero(known)[-1])
    if not known[:anchor_row + 1].all():
        raise ValueError(f"{well_id}: TVT_input must have a continuous known prefix")
    if anchor_row < 8 * 32:
        raise ValueError(f"{well_id}: the known prefix must include row 256")

    gr = horizontal["GR"].interpolate(method="linear").bfill().ffill().to_numpy()
    if not np.isfinite(gr).all():
        raise ValueError(f"{well_id}: GR could not be interpolated")
    smooth_gr = savgol_filter(gr, SAVGOL_WINDOW, 2).astype(np.float32)
    trajectory = {
        f"h_{key.lower()}": horizontal[key].to_numpy(dtype=np.float32)
        for key in ("MD", "X", "Y", "Z")
    }
    if not all(np.isfinite(value).all() for value in trajectory.values()):
        raise ValueError(f"{well_id}: non-finite trajectory coordinate")
    t_gr = typewell["GR"].to_numpy(dtype=np.float32)
    t_tvt = typewell["TVT"].to_numpy(dtype=np.float32)
    if len(t_tvt) < 2 or not np.isfinite(t_gr).all() or not np.isfinite(t_tvt).all():
        raise ValueError(f"{well_id}: invalid typewell values")
    if np.any(np.diff(t_tvt) < 0) or t_tvt[-1] - t_tvt[0] < 1.0:
        raise ValueError(f"{well_id}: typewell TVT must be increasing and span at least 1 ft")

    return {
        "well_id": well_id,
        "h_gr_smooth": smooth_gr,
        "h_tvt_input": tvt_input,
        "ps_idx": anchor_row,
        "t_gr": t_gr,
        "t_tvt": t_tvt,
        **trajectory,
    }
