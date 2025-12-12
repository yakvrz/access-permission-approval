import argparse
import json
from pathlib import Path

import pandas as pd

from access_perm import config as cfg
from access_perm.data import coerce_apps, coerce_users
from access_perm.predict import load_trained_model, predict_requests


def main():
    parser = argparse.ArgumentParser(description="Score new permission requests.")
    parser.add_argument("--config", default="config/default.yaml", help="Path to YAML config.")
    parser.add_argument(
        "--input",
        default=None,
        help="Submission CSV to score (defaults to submission path in config).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Where to save predictions CSV (defaults to reports/predictions.csv).",
    )
    args = parser.parse_args()

    config = cfg.load_config(args.config)
    cfg.ensure_artifact_dirs(config)

    submission_path = args.input or config["data"]["submission"]
    reports_dir = Path(config["artifacts"]["reports_dir"])
    output_path = Path(args.output) if args.output else reports_dir / "predictions.csv"
    model_dir = Path(config["artifacts"]["model_dir"])
    model_path = model_dir / "model.cbm"
    metadata_path = model_dir / "metadata.json"

    if not model_path.exists() or not metadata_path.exists():
        raise FileNotFoundError("Model not found. Run training first.")

    submission_df = pd.read_csv(submission_path)
    users_df = coerce_users(pd.read_csv(config["data"]["users"]))
    apps_df = coerce_apps(pd.read_csv(config["data"]["apps"]))

    with metadata_path.open() as f:
        metadata = json.load(f)

    model = load_trained_model(model_path)
    probas, labels = predict_requests(
        model=model,
        submission_df=submission_df,
        users_df=users_df,
        apps_df=apps_df,
        categorical_columns=metadata["categorical_columns"],
        threshold=metadata["threshold"],
        missing_sentinel=config["features"]["missing_sentinel"],
    )

    output = submission_df.copy()
    output["approval_probability"] = probas
    output["predicted_label"] = labels
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)
    print(json.dumps({"predictions": str(output_path)}, indent=2))


if __name__ == "__main__":
    main()
