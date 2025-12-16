import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from catboost import Pool
import numpy as np
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset
from evidently.pipeline.column_mapping import ColumnMapping
from evidently.report import Report
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from . import config as cfg
from .data import load_raw_data
from .features import FeatureBundle, build_features
from .predict import load_trained_model


def _json_default(obj):
    if isinstance(obj, (np.generic,)):
        return obj.item()
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


@dataclass
class MonitoringOutputs:
    baseline_path: Path
    drift_html: Path
    drift_json: Path
    performance_html: Optional[Path]
    performance_json: Optional[Path]


def save_reference_baseline(bundle: FeatureBundle, scores: np.ndarray, path: Path) -> Path:
    df = bundle.X.copy()
    df["permission"] = bundle.y.values
    df["score"] = scores
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return path


def load_reference_baseline(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Reference baseline not found at {path}. Run training first.")
    return pd.read_parquet(path)


def _column_mapping(df: pd.DataFrame, categorical_columns: List[str], target_col: str, prediction_col: str) -> ColumnMapping:
    cats = [c for c in categorical_columns if c in df.columns]
    numerics = [c for c in df.columns if c not in set(cats + [target_col, prediction_col])]
    return ColumnMapping(
        target=target_col if target_col in df.columns else None,
        prediction=prediction_col if prediction_col in df.columns else None,
        categorical_features=cats,
        numerical_features=numerics,
    )


def _run_drift_report(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    column_mapping: ColumnMapping,
    html_path: Path,
    json_path: Path,
) -> None:
    report = Report(metrics=[DataDriftPreset(), TargetDriftPreset()])
    report.run(reference_data=reference_df, current_data=current_df, column_mapping=column_mapping)
    html_path.parent.mkdir(parents=True, exist_ok=True)
    report.save_html(str(html_path))
    with json_path.open("w") as f:
        json.dump(report.as_dict(), f, indent=2, default=_json_default)


def _compute_performance(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> Dict[str, float]:
    preds = scores >= threshold
    roc = None
    pr_auc = None
    try:
        roc = roc_auc_score(y_true, scores)
    except ValueError:
        roc = None
    try:
        pr_auc = average_precision_score(y_true, scores)
    except ValueError:
        pr_auc = None
    return {
        "roc_auc": float(roc) if roc is not None else None,
        "pr_auc": float(pr_auc) if pr_auc is not None else None,
        "accuracy": float(accuracy_score(y_true, preds)),
        "precision": float(precision_score(y_true, preds, zero_division=0)),
        "recall": float(recall_score(y_true, preds, zero_division=0)),
        "f1": float(f1_score(y_true, preds, zero_division=0)),
    }


def _write_performance_reports(metrics: Dict[str, float], html_path: Path, json_path: Path) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with json_path.open("w") as f:
        json.dump(metrics, f, indent=2)
    html = [
        "<html><head><title>Model performance</title></head><body>",
        "<h2>Current window metrics</h2>",
        "<table border='1' cellpadding='6'>",
        "<tr><th>metric</th><th>value</th></tr>",
    ]
    for k, v in metrics.items():
        display = "n/a" if v is None else f"{v:.4f}"
        html.append(f"<tr><td>{k}</td><td>{display}</td></tr>")
    html.append("</table></body></html>")
    html_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.write_text("\n".join(html))


def run_monitoring(config_path: str, current_interactions: Optional[str] = None) -> MonitoringOutputs:
    config = cfg.load_config(config_path)
    cfg.ensure_artifact_dirs(config)
    monitoring_cfg = config.get("monitoring", {})
    target_column = config.get("model", {}).get("validation", {}).get("target_column", "permission")
    baseline_path = Path(monitoring_cfg["reference_baseline"])
    drift_html = Path(monitoring_cfg["drift_report_html"])
    drift_json = Path(monitoring_cfg["drift_report_json"])
    perf_html = Path(monitoring_cfg["performance_report_html"])
    perf_json = Path(monitoring_cfg["performance_report_json"])

    reference_df = load_reference_baseline(baseline_path)

    data_paths = config["data"].copy()
    if current_interactions:
        data_paths["interactions"] = current_interactions
    raw = load_raw_data(data_paths)
    bundle = build_features(raw, missing_sentinel=config["features"]["missing_sentinel"])

    model_dir = Path(config["artifacts"]["model_dir"])
    metadata_path = model_dir / "metadata.json"
    model_path = model_dir / "model.cbm"
    if not metadata_path.exists() or not model_path.exists():
        raise FileNotFoundError("Model artifacts missing. Run training first.")
    with metadata_path.open() as f:
        metadata = json.load(f)
    model = load_trained_model(model_path)
    cat_features = [bundle.X.columns.get_loc(c) for c in metadata["categorical_columns"] if c in bundle.X.columns]
    scores = model.predict_proba(Pool(bundle.X, cat_features=cat_features))[:, 1]

    current_df = bundle.X.copy()
    current_df["permission"] = bundle.y.values
    current_df["score"] = scores

    col_map = _column_mapping(
        current_df,
        categorical_columns=metadata["categorical_columns"],
        target_col=target_column,
        prediction_col="score",
    )
    _run_drift_report(reference_df, current_df, col_map, drift_html, drift_json)

    metrics = _compute_performance(current_df["permission"].values, scores, metadata["threshold"])
    _write_performance_reports(metrics, perf_html, perf_json)
    return MonitoringOutputs(
        baseline_path=baseline_path,
        drift_html=drift_html,
        drift_json=drift_json,
        performance_html=perf_html,
        performance_json=perf_json,
    )
