from pathlib import Path
from typing import Any, Dict, Optional

import mlflow
import numpy as np
import pandas as pd
from mlflow.models import infer_signature


def _flatten(prefix: str, d: Dict[str, Any]) -> Dict[str, Any]:
    return {f"{prefix}{k}": v for k, v in d.items()}


def init_mlflow(config: Dict[str, Any]) -> None:
    tracking_cfg = config.get("tracking", {})
    tracking_uri = tracking_cfg.get("mlflow_tracking_uri")
    if tracking_uri:
        if tracking_uri.startswith("file:"):
            Path(tracking_uri.replace("file:", "", 1)).mkdir(parents=True, exist_ok=True)
        elif "://" not in tracking_uri:
            Path(tracking_uri).mkdir(parents=True, exist_ok=True)
        mlflow.set_tracking_uri(tracking_uri)
    experiment = tracking_cfg.get("experiment_name", "access-permission-ml")
    mlflow.set_experiment(experiment)


def log_training_run(
    config: Dict[str, Any],
    config_path: Path,
    feature_frame: pd.DataFrame,
    scores: np.ndarray,
    training_result,
    reports_dir: Path,
    metadata: Dict[str, Any],
) -> Optional[str]:
    tracking_cfg = config.get("tracking", {})
    if not tracking_cfg:
        return None

    init_mlflow(config)
    model_name = tracking_cfg.get("register_model_name")
    run_name = tracking_cfg.get("run_name", "train")
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(_flatten("model.", config.get("model", {}).get("catboost", {})))
        mlflow.log_params(_flatten("validation.", config.get("model", {}).get("validation", {})))
        mlflow.log_param("threshold", training_result.threshold)
        mlflow.log_metric("best_iteration_median", float(np.median(training_result.best_iterations)))
        for key, value in training_result.metrics.items():
            mlflow.log_metric(key, float(value))

        signature = infer_signature(feature_frame, scores)
        from mlflow.catboost import log_model  # imported lazily to avoid hard dependency at import time

        log_model(
            catboost_model=training_result.model,
            artifact_path="model",
            signature=signature,
            registered_model_name=model_name,
        )

        mlflow.log_dict(metadata, "artifacts/metadata.json")
        mlflow.log_artifact(str(config_path), artifact_path="config")
        if reports_dir.exists():
            mlflow.log_artifacts(str(reports_dir), artifact_path="reports")
    return run.info.run_id


def log_inference_metrics(
    config: Dict[str, Any],
    count: int,
    approval_rate: float,
    source: str,
    prediction_path: Optional[Path] = None,
) -> Optional[str]:
    tracking_cfg = config.get("tracking", {})
    if not tracking_cfg or not tracking_cfg.get("log_inference", False):
        return None

    init_mlflow(config)
    with mlflow.start_run(run_name="inference") as run:
        mlflow.log_metric("predictions", count)
        mlflow.log_metric("approval_rate", approval_rate)
        mlflow.log_param("source", source)
        if prediction_path and prediction_path.exists():
            mlflow.log_artifact(str(prediction_path), artifact_path="predictions")
    return run.info.run_id
