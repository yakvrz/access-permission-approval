from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool

from .features import build_scoring_matrix


def load_trained_model(path: str | Path) -> CatBoostClassifier:
    model = CatBoostClassifier()
    model.load_model(path)
    return model


def predict_requests(
    model: CatBoostClassifier,
    submission_df: pd.DataFrame,
    users_df: pd.DataFrame,
    apps_df: pd.DataFrame,
    categorical_columns,
    threshold: float,
    missing_sentinel: str,
) -> Tuple[np.ndarray, np.ndarray]:
    X = build_scoring_matrix(submission_df, users_df, apps_df, missing_sentinel)
    cat_features = [X.columns.get_loc(c) for c in categorical_columns if c in X.columns]
    probas = model.predict_proba(Pool(X, cat_features=cat_features))[:, 1]
    labels = (probas >= threshold).astype(int)
    return probas, labels
