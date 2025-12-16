import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GroupKFold


@dataclass
class TrainingResult:
    model: CatBoostClassifier
    oof_pred: np.ndarray
    metrics: Dict[str, float]
    threshold: float
    best_iterations: List[int]


def _cat_feature_indices(columns: Sequence[str], categorical_columns: Sequence[str]) -> List[int]:
    return [columns.index(c) for c in categorical_columns if c in columns]


def _choose_threshold(y_true: np.ndarray, scores: np.ndarray, strategy: str = "accuracy") -> float:
    """Select a classification threshold."""
    if strategy not in {"accuracy", "f1"}:
        raise ValueError(f"Unsupported threshold strategy: {strategy}")

    thresholds = np.unique(scores)
    best_threshold, best_score = 0.5, -np.inf

    for t in thresholds:
        preds = scores >= t
        if strategy == "accuracy":
            score = accuracy_score(y_true, preds)
        else:
            score = f1_score(y_true, preds)
        if score > best_score:
            best_score = score
            best_threshold = float(t)
    return float(best_threshold)


def _compute_metrics(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> Dict[str, float]:
    preds = scores >= threshold
    tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()
    return {
        "roc_auc": roc_auc_score(y_true, scores),
        "pr_auc": average_precision_score(y_true, scores),
        "accuracy": accuracy_score(y_true, preds),
        "precision": precision_score(y_true, preds),
        "recall": recall_score(y_true, preds),
        "f1": f1_score(y_true, preds),
        "tpr": tp / (tp + fn) if (tp + fn) else 0.0,
        "fpr": fp / (fp + tn) if (fp + tn) else 0.0,
    }


def train_model(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    categorical_columns: Sequence[str],
    model_params: Dict,
    n_splits: int = 5,
    threshold_strategy: str = "accuracy",
    progress_callback: Optional[Callable[[str], None]] = None,
) -> TrainingResult:
    columns = list(X.columns)
    cat_features = _cat_feature_indices(columns, categorical_columns)
    oof_pred = np.zeros(len(X))
    best_iterations: List[int] = []

    gkf = GroupKFold(n_splits=n_splits)
    for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups=groups)):
        if progress_callback:
            progress_callback(f"Training fold {fold_idx + 1}/{n_splits}")
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = CatBoostClassifier(
            loss_function="Logloss",
            eval_metric="AUC",
            verbose=False,
            **model_params,
        )
        model.fit(
            Pool(X_train, y_train, cat_features=cat_features),
            eval_set=Pool(X_val, y_val, cat_features=cat_features),
            use_best_model=True,
        )
        preds = model.predict_proba(Pool(X_val, cat_features=cat_features))[:, 1]
        oof_pred[val_idx] = preds
        best_iterations.append(model.get_best_iteration())

        if progress_callback:
            progress_callback(f"Completed fold {fold_idx + 1}/{n_splits} (best_iteration={best_iterations[-1]})")

    if progress_callback:
        progress_callback("Selecting threshold and computing metrics")
    threshold = _choose_threshold(y.values, oof_pred, strategy=threshold_strategy)
    metrics = _compute_metrics(y.values, oof_pred, threshold)

    final_iterations = int(np.median(best_iterations)) if best_iterations else model_params.get("iterations", 200)
    if final_iterations < 1:
        final_iterations = max(1, model_params.get("iterations", 200))
    if progress_callback:
        progress_callback(f"Training final model for {final_iterations} iterations")
    final_model = CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="AUC",
        verbose=False,
        iterations=final_iterations,
        **{k: v for k, v in model_params.items() if k != "iterations"},
    )
    final_model.fit(Pool(X, y, cat_features=cat_features))
    if progress_callback:
        progress_callback("Final model trained")
    return TrainingResult(
        model=final_model,
        oof_pred=oof_pred,
        metrics=metrics,
        threshold=threshold,
        best_iterations=best_iterations,
    )


def save_model(model: CatBoostClassifier, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(path)
