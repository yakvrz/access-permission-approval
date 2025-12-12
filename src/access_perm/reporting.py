from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
from catboost import Pool
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    PrecisionRecallDisplay,
    RocCurveDisplay,
    confusion_matrix,
)


def _configure_matplotlib():
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.grid": True,
            "grid.linewidth": 0.4,
            "axes.facecolor": "white",
            "figure.dpi": 110,
        }
    )


def plot_roc_pr(y_true: np.ndarray, scores: np.ndarray, reports_dir: Path) -> None:
    _configure_matplotlib()
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    RocCurveDisplay.from_predictions(y_true, scores, ax=ax[0], name="OOF")
    ax[0].set_title("ROC")
    ax[0].set_xlim(0, 1)
    ax[0].set_ylim(0, 1)
    PrecisionRecallDisplay.from_predictions(y_true, scores, ax=ax[1], name="OOF")
    ax[1].set_title("Precision-Recall")
    ax[1].set_xlim(0, 1)
    ax[1].set_ylim(0, 1)
    reports_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(reports_dir / "roc_pr.png")
    plt.close(fig)


def plot_score_distribution(y_true: np.ndarray, scores: np.ndarray, reports_dir: Path) -> None:
    _configure_matplotlib()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(scores[y_true == 1], bins=30, alpha=0.6, label="approved", color="#1f77b4")
    ax.hist(scores[y_true == 0], bins=30, alpha=0.6, label="denied", color="#d62728")
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Count")
    ax.set_title("Score distribution (OOF)")
    ax.legend()
    reports_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(reports_dir / "score_distribution.png")
    plt.close(fig)


def plot_confusion(y_true: np.ndarray, scores: np.ndarray, threshold: float, reports_dir: Path) -> None:
    _configure_matplotlib()
    preds = scores >= threshold
    cm = confusion_matrix(y_true, preds)
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm, cmap="Blues")
    for (i, j), val in np.ndenumerate(cm):
        ax.text(j, i, int(val), ha="center", va="center", color="black")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks([0, 1], labels=["Denied", "Approved"])
    ax.set_yticks([0, 1], labels=["Denied", "Approved"])
    ax.set_title(f"Confusion @ threshold={threshold:.3f}")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    reports_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(reports_dir / "confusion_matrix.png")
    plt.close(fig)


def plot_calibration(y_true: np.ndarray, scores: np.ndarray, reports_dir: Path) -> None:
    _configure_matplotlib()
    prob_true, prob_pred = calibration_curve(y_true, scores, n_bins=10, strategy="quantile")
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(prob_pred, prob_true, marker="o", label="Model")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect")
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed frequency")
    ax.set_title("Calibration (OOF)")
    ax.legend()
    reports_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(reports_dir / "calibration.png")
    plt.close(fig)


def save_metrics(metrics: Dict[str, float], reports_dir: Path) -> None:
    import json

    reports_dir.mkdir(parents=True, exist_ok=True)
    with (reports_dir / "metrics.json").open("w") as f:
        json.dump(metrics, f, indent=2)


def compute_segment_metrics(
    X,
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    segments: list[str],
    top_n: int = 10,
    min_count: int = 100,
):
    """Return per-segment metrics for selected categorical columns."""
    import pandas as pd
    from sklearn.metrics import roc_auc_score

    preds = scores >= threshold
    rows = []
    y_true = pd.Series(y_true).reset_index(drop=True)
    X = X.reset_index(drop=True)
    scores_s = pd.Series(scores).reset_index(drop=True)

    for col in segments:
        counts = X[col].value_counts().head(top_n)
        for level, count in counts.items():
            if count < min_count:
                continue
            mask = X[col] == level
            y_seg = y_true[mask]
            s_seg = scores_s[mask]
            p_seg = preds[mask]
            acc = (p_seg == y_seg).mean()
            pos_rate = y_seg.mean()
            roc = None
            if y_seg.nunique() > 1:
                roc = roc_auc_score(y_seg, s_seg)
            rows.append(
                {
                    "segment": col,
                    "level": str(level),
                    "count": int(count),
                    "approval_rate": float(pos_rate),
                    "accuracy_at_threshold": float(acc),
                    "roc_auc": float(roc) if roc is not None else None,
                }
            )
    return rows


def plot_shap_importance(model, X_sample, cat_features, feature_names, reports_dir: Path, top_n: int = 10):
    """Compute and plot mean |SHAP| for a sample."""
    reports_dir.mkdir(parents=True, exist_ok=True)
    shap_values = model.get_feature_importance(
        Pool(X_sample, cat_features=cat_features),
        type="ShapValues",
    )
    contrib = shap_values[:, :-1]
    mean_abs = np.abs(contrib).mean(axis=0)
    order = np.argsort(mean_abs)[::-1][:top_n]
    names = [feature_names[i] for i in order]
    values = mean_abs[order]

    _configure_matplotlib()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh(range(len(names)), values[::-1], color="#1f77b4")
    ax.set_yticks(range(len(names)), names[::-1])
    ax.set_xlabel("Mean |SHAP|")
    ax.set_title("Top feature attributions (sample)")
    fig.tight_layout()
    fig.savefig(reports_dir / "shap_importance.png")
    plt.close(fig)
