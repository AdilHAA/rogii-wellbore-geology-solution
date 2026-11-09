"""Run the five-fold CNN-SDF ensemble and write a Kaggle submission."""

import argparse
from pathlib import Path


WEIGHT_NAMES = [f"exp_217_ancc_ft_full_fold{fold}.pt" for fold in range(5)]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True, help="Competition directory with test/ and sample_submission.csv")
    parser.add_argument("--weights-dir", type=Path, required=True, help="Directory containing the five fold checkpoints")
    parser.add_argument("--output", type=Path, default=Path("submission.csv"))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def main():
    args = parse_args()
    checkpoints = [args.weights_dir / name for name in WEIGHT_NAMES]
    missing = [str(path) for path in checkpoints if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing checkpoints:\n" + "\n".join(missing))
    sample_path = args.data_dir / "sample_submission.csv"
    if not sample_path.is_file():
        raise FileNotFoundError(sample_path)
    horizontal_files = sorted((args.data_dir / "test").glob("*__horizontal_well.csv"))
    if not horizontal_files:
        raise FileNotFoundError(f"No horizontal wells in {args.data_dir / 'test'}")

    import numpy as np
    import pandas as pd
    import torch

    from src.data import load_well
    from src.model import GeoSteerNet
    from src.rollout import predict_well

    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    models = []
    for checkpoint in checkpoints:
        model = GeoSteerNet()
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=True)
        models.append(model.to(device).eval())
    print(f"Loaded {len(models)} models on {device}", flush=True)

    results = []
    for index, path in enumerate(horizontal_files, start=1):
        well_id = path.name.removesuffix("__horizontal_well.csv")
        well = load_well(args.data_dir, well_id)
        rows = np.flatnonzero(np.isnan(well["h_tvt_input"]))
        if len(rows):
            full_prediction = predict_well(well, models, device)
            results.append(pd.DataFrame({
                "id": [f"{well_id}_{row}" for row in rows],
                "tvt": full_prediction[rows].astype(float),
            }))
        print(f"[{index}/{len(horizontal_files)}] {well_id}", flush=True)
    if not results:
        raise ValueError("No rows with missing TVT_input were found")
    predictions = pd.concat(results, ignore_index=True)
    sample = pd.read_csv(sample_path)
    if "id" not in sample.columns or sample["id"].duplicated().any():
        raise ValueError("sample_submission.csv needs unique id values")
    submission = sample[["id"]].merge(predictions, on="id", how="left", validate="one_to_one")
    if not np.isfinite(submission["tvt"]).all():
        raise ValueError("Some submission ids have no finite prediction")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(args.output, index=False)
    print(f"Saved {len(submission)} predictions to {args.output}", flush=True)


if __name__ == "__main__":
    main()
