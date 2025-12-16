import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from . import config as cfg
from .data import coerce_apps, coerce_users
from .predict import load_trained_model, predict_requests
from .tracking import log_inference_metrics
from .validation import validate_submission


CONFIG_ENV = "ACCESS_PERM_CONFIG"


class AccessRequest(BaseModel):
    userId: int
    appId: int
    requestId: Optional[str] = None


class Prediction(BaseModel):
    userId: str
    appId: str
    approval_probability: float
    predicted_label: int
    requestId: Optional[str] = None


def _load_state(config_path: str):
    config = cfg.load_config(config_path)
    cfg.ensure_artifact_dirs(config)
    model_dir = Path(config["artifacts"]["model_dir"])
    model_path = model_dir / "model.cbm"
    metadata_path = model_dir / "metadata.json"
    if not model_path.exists() or not metadata_path.exists():
        raise RuntimeError("Model artifacts not found. Run training first.")
    with metadata_path.open() as f:
        metadata = json.load(f)
    model = load_trained_model(model_path)
    users = coerce_users(pd.read_csv(config["data"]["users"]))
    apps = coerce_apps(pd.read_csv(config["data"]["apps"]))
    return {
        "config": config,
        "model": model,
        "metadata": metadata,
        "users": users,
        "apps": apps,
    }


def _state():
    global _SERVICE_STATE
    if _SERVICE_STATE is None:
        config_path = os.getenv(CONFIG_ENV, "config/default.yaml")
        _SERVICE_STATE = _load_state(config_path)
    return _SERVICE_STATE


_SERVICE_STATE = None
app = FastAPI(title="Access Permission Approval Service")


@app.get("/health")
def health():
    try:
        state = _state()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    model_dir = Path(state["config"]["artifacts"]["model_dir"])
    return {"status": "ok", "model_path": str(model_dir / "model.cbm")}


@app.post("/predict", response_model=Dict[str, List[Prediction]])
def predict(requests: List[AccessRequest]):
    if not requests:
        raise HTTPException(status_code=400, detail="No requests supplied")
    state = _state()
    submission_df = pd.DataFrame([r.dict() for r in requests])
    validation_cfg = state["config"].get("validation", {})
    validation_dir = Path(state["config"]["artifacts"]["validation_dir"]) / "service"
    if validation_cfg.get("enabled", True):
        result = validate_submission(submission_df, validation_dir, mostly=validation_cfg.get("mostly", 0.95))
        if validation_cfg.get("fail_on_error", True) and not result.success:
            raise HTTPException(status_code=400, detail="Payload failed schema validation")

    probas, labels = predict_requests(
        model=state["model"],
        submission_df=submission_df,
        users_df=state["users"],
        apps_df=state["apps"],
        categorical_columns=state["metadata"]["categorical_columns"],
        threshold=state["metadata"]["threshold"],
        missing_sentinel=state["config"]["features"]["missing_sentinel"],
    )
    preds: List[Prediction] = []
    for req, p, label in zip(requests, probas, labels):
        preds.append(
            Prediction(
                userId=str(req.userId),
                appId=str(req.appId),
                approval_probability=float(p),
                predicted_label=int(label),
                requestId=req.requestId,
            )
        )
    approval_rate = float(np.mean(labels)) if len(labels) else 0.0
    log_inference_metrics(
        state["config"],
        count=len(preds),
        approval_rate=approval_rate,
        source="api",
    )
    return {"predictions": preds}
