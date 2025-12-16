import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as cfg
from .data import load_raw_data
from .features import build_features
from .monitoring import save_reference_baseline
from .modeling import TrainingResult, save_model, train_model
from .reporting import (
    plot_calibration,
    plot_confusion,
    plot_roc_pr,
    plot_score_distribution,
    save_metrics,
    compute_segment_metrics,
    plot_shap_importance,
)
from .tracking import log_training_run
from .validation import validate_features, validate_raw_data


def _set_seeds(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)


def run_training(config_path: str) -> TrainingResult:
    def progress(message: str) -> None:
        print(f"[train] {message}", flush=True)

    config = cfg.load_config(config_path)
    cfg.ensure_artifact_dirs(config)
    _set_seeds(42)

    progress(f"Loaded config from {config_path}")
    raw = load_raw_data(config["data"])
    progress("Loaded raw data")
    validation_cfg = config.get("validation", {})
    if validation_cfg.get("enabled", True):
        validation_dir = Path(config["artifacts"]["validation_dir"])
        raw_results = validate_raw_data(raw, validation_dir, mostly=validation_cfg.get("mostly", 0.97))
        if validation_cfg.get("fail_on_error", True) and not all(r.success for r in raw_results):
            raise ValueError("Raw data validation failed. See reports in validation directory.")
        progress("Raw data validation passed")

    missing_sentinel = config["features"]["missing_sentinel"]
    categorical_columns = config["features"]["categorical_columns"]
    model_params = config["model"]["catboost"]
    n_splits = config["model"]["validation"]["n_splits"]
    threshold_strategy = config["model"]["validation"].get("threshold_strategy", "accuracy")

    progress("Building features")
    feature_bundle = build_features(raw, missing_sentinel=missing_sentinel)
    if validation_cfg.get("enabled", True):
        feature_validation = validate_features(feature_bundle, Path(config["artifacts"]["validation_dir"]), mostly=validation_cfg.get("mostly", 0.97))
        if validation_cfg.get("fail_on_error", True) and not feature_validation.success:
            raise ValueError("Feature validation failed. See validation reports.")
        progress("Feature validation passed")

    progress("Starting cross-validation training")
    result = train_model(
        X=feature_bundle.X,
        y=feature_bundle.y,
        groups=feature_bundle.groups,
        categorical_columns=categorical_columns,
        model_params=model_params,
        n_splits=n_splits,
        threshold_strategy=threshold_strategy,
        progress_callback=progress,
    )

    model_dir = Path(config["artifacts"]["model_dir"])
    reports_dir = Path(config["artifacts"]["reports_dir"])
    model_path = model_dir / "model.cbm"
    metadata_path = model_dir / "metadata.json"

    progress(f"Saving model to {model_path}")
    save_model(result.model, model_path)
    metadata = {
        "config_path": str(config_path),
        "categorical_columns": categorical_columns,
        "feature_columns": feature_bundle.feature_columns,
        "threshold": result.threshold,
        "metrics": result.metrics,
        "best_iterations": result.best_iterations,
    }

    progress(f"Writing reports to {reports_dir}")
    save_metrics(result.metrics, reports_dir)
    plot_roc_pr(feature_bundle.y.values, result.oof_pred, reports_dir)
    plot_score_distribution(feature_bundle.y.values, result.oof_pred, reports_dir)
    plot_confusion(feature_bundle.y.values, result.oof_pred, result.threshold, reports_dir)
    plot_calibration(feature_bundle.y.values, result.oof_pred, reports_dir)
    pd.DataFrame({"permission": feature_bundle.y, "score": result.oof_pred}).to_csv(
        reports_dir / "oof_predictions.csv", index=False
    )

    # Segment metrics (top entities with support)
    segment_rows = compute_segment_metrics(
        feature_bundle.X,
        feature_bundle.y.values,
        result.oof_pred,
        result.threshold,
        segments=["appId", "department_filled", "manager_department"],
        top_n=10,
        min_count=100,
    )
    if segment_rows:
        import json as _json

        with (reports_dir / "segment_metrics.json").open("w") as f:
            _json.dump(segment_rows, f, indent=2)

    # SHAP importance on a stratified sample
    sample_size = min(2000, len(feature_bundle.X))
    X_sample = (
        feature_bundle.X.sample(sample_size, random_state=42)
        if sample_size < len(feature_bundle.X)
        else feature_bundle.X
    )
    cat_features = [
        feature_bundle.feature_columns.index(c) for c in categorical_columns if c in feature_bundle.feature_columns
    ]
    plot_shap_importance(
        result.model,
        X_sample,
        cat_features=cat_features,
        feature_names=feature_bundle.feature_columns,
        reports_dir=reports_dir,
        top_n=10,
    )

    baseline_path = Path(config["monitoring"]["reference_baseline"])
    save_reference_baseline(feature_bundle, result.oof_pred, baseline_path)
    metadata["monitoring_reference"] = str(baseline_path)
    cfg.save_metadata(metadata, metadata_path)

    log_training_run(
        config=config,
        config_path=Path(config_path),
        feature_frame=feature_bundle.X,
        scores=result.oof_pred,
        training_result=result,
        reports_dir=reports_dir,
        metadata=metadata,
    )

    progress("Training complete")
    return result


def main():
    parser = argparse.ArgumentParser(description="Train access permission model.")
    parser.add_argument("--config", default="config/default.yaml", help="Path to YAML config.")
    args = parser.parse_args()
    result = run_training(args.config)
    print(json.dumps({"metrics": result.metrics, "threshold": result.threshold}, indent=2))


if __name__ == "__main__":
    main()
