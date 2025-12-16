import json
from pathlib import Path

import pandas as pd
import yaml
from fastapi.testclient import TestClient

from access_perm import monitoring, service
from access_perm.config import save_metadata
from access_perm.data import RawData
from access_perm.features import build_features
from access_perm.modeling import save_model, train_model


def _prepare_artifacts(tmp_path: Path) -> Path:
    """Train a tiny model on fixture data and write config + artifacts for smoke tests."""
    base = Path("tests/fixtures/small")
    users = pd.read_csv(base / "users_metadata.csv")
    apps = pd.read_csv(base / "apps_metadata.csv")
    interactions = pd.read_csv(base / "interactions.csv")

    raw = RawData(users=users, apps=apps, interactions=interactions)
    bundle = build_features(raw, missing_sentinel="__UNK__")
    cat_cols = [
        "appId",
        "managerId",
        "category",
        "department_filled",
        "office_filled",
        "manager_department",
        "manager_office",
    ]
    model_params = {"iterations": 20, "learning_rate": 0.2, "depth": 4, "l2_leaf_reg": 3, "random_seed": 42}
    result = train_model(
        X=bundle.X,
        y=bundle.y,
        groups=bundle.groups,
        categorical_columns=cat_cols,
        model_params=model_params,
        n_splits=3,
    )

    model_dir = tmp_path / "model"
    reports_dir = tmp_path / "reports"
    validation_dir = tmp_path / "validation"
    monitoring_dir = tmp_path / "monitoring"
    model_dir.mkdir(parents=True, exist_ok=True)
    save_model(result.model, model_dir / "model.cbm")
    metadata = {
        "categorical_columns": cat_cols,
        "feature_columns": bundle.feature_columns,
        "threshold": result.threshold,
        "metrics": result.metrics,
        "best_iterations": result.best_iterations,
    }
    save_metadata(metadata, model_dir / "metadata.json")

    baseline_path = monitoring_dir / "reference.parquet"
    monitoring.save_reference_baseline(bundle, result.oof_pred, baseline_path)

    config = {
        "data": {
            "users": str(base / "users_metadata.csv"),
            "apps": str(base / "apps_metadata.csv"),
            "interactions": str(base / "interactions.csv"),
            "submission": str(base / "submission.csv"),
        },
        "artifacts": {
            "model_dir": str(model_dir),
            "reports_dir": str(reports_dir),
            "validation_dir": str(validation_dir),
            "monitoring_dir": str(monitoring_dir),
        },
        "features": {
            "missing_sentinel": "__UNK__",
            "categorical_columns": cat_cols,
        },
        "model": {
            "validation": {
                "n_splits": 3,
                "group_column": "userId",
                "target_column": "permission",
                "threshold_strategy": "accuracy",
            }
        },
        "monitoring": {
            "reference_baseline": str(baseline_path),
            "drift_report_html": str(monitoring_dir / "drift_report.html"),
            "drift_report_json": str(monitoring_dir / "drift_report.json"),
            "performance_report_html": str(monitoring_dir / "performance_report.html"),
            "performance_report_json": str(monitoring_dir / "performance_report.json"),
        },
        "tracking": {"log_inference": False},
        "validation": {"enabled": True, "fail_on_error": True, "mostly": 0.9},
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))
    return config_path


def test_service_and_monitoring_smoke(tmp_path, monkeypatch):
    config_path = _prepare_artifacts(tmp_path)

    # Service prediction
    monkeypatch.setenv(service.CONFIG_ENV, str(config_path))
    service._SERVICE_STATE = None  # reset cached state
    client = TestClient(service.app)
    resp = client.post("/predict", json=[{"userId": 1, "appId": 100}])
    assert resp.status_code == 200
    body = resp.json()
    assert "predictions" in body
    assert isinstance(body["predictions"], list)
    assert body["predictions"][0]["predicted_label"] in (0, 1)

    # Monitoring run
    outputs = monitoring.run_monitoring(str(config_path))
    assert outputs.drift_html.exists()
    assert outputs.drift_json.exists()
    assert outputs.performance_html.exists()
    assert outputs.performance_json.exists()
