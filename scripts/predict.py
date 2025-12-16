import argparse
import json
from pathlib import Path

import pandas as pd

from access_perm import config as cfg
from access_perm.data import coerce_apps, coerce_users
from access_perm.predict import load_trained_model, predict_requests
from access_perm.tracking import log_inference_metrics
from access_perm.validation import validate_submission


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
    validation_dir = Path(config["artifacts"]["validation_dir"]) / "inference"
    model_dir = Path(config["artifacts"]["model_dir"])
    model_path = model_dir / "model.cbm"
    metadata_path = model_dir / "metadata.json"

    if not model_path.exists() or not metadata_path.exists():
        raise FileNotFoundError("Model not found. Run training first.")

    submission_df = pd.read_csv(submission_path)
    users_df = coerce_users(pd.read_csv(config["data"]["users"]))
    apps_df = coerce_apps(pd.read_csv(config["data"]["apps"]))

    validation_cfg = config.get("validation", {})
    if validation_cfg.get("enabled", True):
        validation_result = validate_submission(submission_df, validation_dir, mostly=validation_cfg.get("mostly", 0.95))
        if validation_cfg.get("fail_on_error", True) and not validation_result.success:
            raise ValueError("Submission failed validation. See validation reports.")

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
    approval_rate = float(labels.mean()) if len(labels) else 0.0
    log_inference_metrics(config, count=len(labels), approval_rate=approval_rate, source="batch-cli", prediction_path=output_path)
    print(json.dumps({"predictions": str(output_path), "approval_rate": approval_rate}, indent=2))


if __name__ == "__main__":
    main()
