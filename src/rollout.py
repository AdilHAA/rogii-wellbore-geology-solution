"""Predict overlapping chunks and re-anchor on the averaged trajectory."""

import numpy as np
import torch

from .render import ROWS_PER_COLUMN, build_chunk, decode_path


@torch.no_grad()
def predict_well(well, models, device):
    known_tvt = well["h_tvt_input"]
    length = len(well["h_gr_smooth"])
    last_known = well["ps_idx"]
    prediction_sum = np.zeros(length)
    prediction_weight = np.zeros(length)
    prediction = np.full(length, np.nan, np.float32)
    prediction[:last_known + 1] = known_tvt[:last_known + 1]
    anchor_row = last_known
    anchor_tvt = float(known_tvt[anchor_row])
    history_tvt = np.array([
        known_tvt[
            max(0, anchor_row - (8 - k) * ROWS_PER_COLUMN):
            max(1, anchor_row - (7 - k) * ROWS_PER_COLUMN)
        ].mean()
        for k in range(8)
    ], np.float32)

    for _ in range(800):
        # The original renderer needs two complete future columns. Carry the
        # last prediction through the remaining rows at the end of the well.
        if length - anchor_row < 2 * ROWS_PER_COLUMN:
            break
        chunk = build_chunk(well, anchor_row, anchor_tvt, history_tvt)
        image = torch.from_numpy(chunk["image"])[None].to(device)
        field_sum = None
        for model in models:
            output = model({"image": image})
            fields = output["sdf"][0].cpu().numpy()
            branch = output["logit"][0].argmax(0).cpu().numpy()
            # Candidate indices are not aligned between independently trained folds.
            field = fields[branch, :, np.arange(fields.shape[2])].T
            field_sum = field if field_sum is None else field_sum + field
        sdf = (field_sum / len(models))[:, :chunk["n"]]
        if not np.isfinite(sdf).all():
            raise ValueError(f"{well['well_id']}: non-finite model output")
        column_tvt = decode_path(sdf, chunk["t_seg_tvt"])
        n_history, n_columns, first_row = chunk["nh"], chunk["n"], chunk["c0"]
        n_future = n_columns - n_history
        for column in range(n_history, n_columns):
            start = max(first_row + column * ROWS_PER_COLUMN, last_known + 1)
            stop = min(first_row + (column + 1) * ROWS_PER_COLUMN, length)
            if start >= stop:
                continue
            position = column - n_history
            weight = 1.0 - abs((position + 0.5) / n_future - 0.5) * 2.0 + 0.1
            prediction_sum[start:stop] += weight * column_tvt[column]
            prediction_weight[start:stop] += weight

        rebase_column = n_history + max(1, int(n_future * 0.75))
        next_anchor = first_row + rebase_column * ROWS_PER_COLUMN
        if next_anchor <= anchor_row or next_anchor + ROWS_PER_COLUMN >= length:
            break
        segment_start = max(last_known + 1, next_anchor - ROWS_PER_COLUMN)
        if prediction_weight[segment_start:next_anchor].sum() > 0:
            anchor_tvt = float((
                prediction_sum[segment_start:next_anchor]
                / np.maximum(prediction_weight[segment_start:next_anchor], 1e-9)
            ).mean())
        else:
            anchor_tvt = float(column_tvt[rebase_column])

        history = []
        for k in range(8):
            start = max(next_anchor - (8 - k) * ROWS_PER_COLUMN, 0)
            stop = max(next_anchor - (7 - k) * ROWS_PER_COLUMN, 1)
            if start <= last_known:
                history.append(float(known_tvt[start:max(stop, start + 1)].mean()))
            elif prediction_weight[start:stop].sum() > 0:
                history.append(float((
                    prediction_sum[start:stop]
                    / np.maximum(prediction_weight[start:stop], 1e-9)
                ).mean()))
            else:
                history.append(anchor_tvt)
        history_tvt = np.array(history, np.float32)
        anchor_row = next_anchor
    else:
        raise RuntimeError(f"{well['well_id']}: exceeded 800 rollout chunks")

    covered = prediction_weight > 0
    prediction[covered] = prediction_sum[covered] / prediction_weight[covered]
    last_value = known_tvt[last_known]
    for row in range(last_known + 1, length):
        if np.isnan(prediction[row]):
            prediction[row] = last_value
        else:
            last_value = prediction[row]
    if not np.isfinite(prediction).all():
        raise ValueError(f"{well['well_id']}: non-finite final prediction")
    return prediction
