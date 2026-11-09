"""Rendering from medali1992 (Apache-2.0), adapted by Adil Khabibullin."""

import numpy as np


ROWS_PER_COLUMN = 32
TVT_STEP = 1.0
HALF_HEIGHT = 128
HEIGHT = 256
WIDTH = 256


def _resample_typewell(well):
    if "t_grid" not in well:
        tvt, gr = well["t_tvt"], well["t_gr"]
        grid = np.arange(float(tvt.min()), float(tvt.max()), TVT_STEP, dtype=np.float32)
        well["t_grid"] = grid
        well["t_gr_r"] = np.interp(grid, tvt, gr).astype(np.float32)
    return well["t_grid"], well["t_gr_r"]


def build_chunk(well, anchor_row, anchor_tvt, history_tvt):
    history_columns = 8
    first_row = anchor_row - history_columns * ROWS_PER_COLUMN
    if first_row < 0:
        history_columns = anchor_row // ROWS_PER_COLUMN
        first_row = anchor_row - history_columns * ROWS_PER_COLUMN
    n_columns = history_columns + min(48, (len(well["h_gr_smooth"]) - anchor_row) // ROWS_PER_COLUMN)
    if n_columns < history_columns + 2:
        raise ValueError("A chunk requires at least two future columns")
    segment = slice(first_row, first_row + n_columns * ROWS_PER_COLUMN)
    horizontal_gr = well["h_gr_smooth"][segment].reshape(n_columns, ROWS_PER_COLUMN).mean(1).astype(np.float32)

    grid, reference_gr = _resample_typewell(well)
    anchor_index = int(np.abs(grid - anchor_tvt).argmin())
    start, stop = anchor_index - HALF_HEIGHT, anchor_index + HALF_HEIGHT
    pad_left, pad_right = max(0, -start), max(0, stop - len(grid))
    crop = slice(max(0, start), min(len(grid), stop))
    typewell_gr = np.pad(reference_gr[crop], (pad_left, pad_right), "edge")
    typewell_tvt = np.pad(grid[crop], (pad_left, pad_right), "edge").astype(np.float32)

    image = np.zeros((9, HEIGHT, WIDTH), np.float32)
    reference_channel = np.broadcast_to(typewell_gr[:, None], (HEIGHT, n_columns))
    horizontal_channel = np.broadcast_to(horizontal_gr[None, :], (HEIGHT, n_columns))
    image[0, :, :n_columns] = reference_channel
    image[1, :, :n_columns] = horizontal_channel
    for column in range(min(history_columns, len(history_tvt))):
        row = int(np.abs(typewell_tvt - history_tvt[column]).argmin())
        image[2, row, column] = 1.0
    location = (typewell_tvt - anchor_tvt) / (HALF_HEIGHT * TVT_STEP)
    image[3, :, :n_columns] = np.broadcast_to(location[:, None], (HEIGHT, n_columns))

    if "geo_sc" not in well:
        dmd, dz, dx, dy = (
            np.gradient(well[key]) for key in ("h_md", "h_z", "h_x", "h_y")
        )
        dip = np.arctan2(dmd, dz)
        azimuth = np.arctan2(dx, dy)
        well["geo_sc"] = np.stack([
            np.sin(dip), np.cos(dip), np.sin(azimuth), np.cos(azimuth)
        ]).astype(np.float32)
    geometry = well["geo_sc"][:, segment].reshape(4, n_columns, ROWS_PER_COLUMN).mean(2)
    for channel in range(4):
        image[4 + channel, :, :n_columns] = np.broadcast_to(geometry[channel][None, :], (HEIGHT, n_columns))
    image[8, :, :n_columns] = reference_channel - horizontal_channel
    return {
        "image": image,
        "t_seg_tvt": typewell_tvt,
        "n": n_columns,
        "nh": history_columns,
        "c0": first_row,
    }


def _band_logsumexp(values, max_step, penalty):
    result = np.full(len(values), -np.inf)
    for step in range(-max_step, max_step + 1):
        shifted = np.roll(values, step) - penalty * abs(step)
        if step > 0:
            shifted[:step] = -np.inf
        if step < 0:
            shifted[step:] = -np.inf
        result = np.logaddexp(result, shifted)
    return result


def decode_path(sdf, typewell_tvt):
    """Forward-backward mean with transitions limited to four TVT rows."""
    height, n_columns = sdf.shape
    emission = -np.abs(sdf.astype(np.float64)) / 0.2
    forward = np.empty((n_columns, height))
    forward[0] = emission[:, 0]
    for column in range(1, n_columns):
        forward[column] = emission[:, column] + _band_logsumexp(forward[column - 1], 4, 1.1)
    backward = np.empty((n_columns, height))
    backward[-1] = 0.0
    for column in range(n_columns - 2, -1, -1):
        values = emission[:, column + 1] + backward[column + 1]
        result = np.full(height, -np.inf)
        for step in range(-4, 5):
            shifted = np.roll(values, -step) - 1.1 * abs(step)
            if step > 0:
                shifted[height - step:] = -np.inf
            if step < 0:
                shifted[:-step] = -np.inf
            result = np.logaddexp(result, shifted)
        backward[column] = result
    log_probability = forward + backward
    log_probability -= log_probability.max(1, keepdims=True)
    probability = np.exp(log_probability)
    probability /= probability.sum(1, keepdims=True)
    rows = np.arange(height)
    mean_row = (probability * rows[None, :]).sum(1)
    return np.interp(mean_row, rows, typewell_tvt).astype(np.float32)
