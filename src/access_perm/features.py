from dataclasses import dataclass
from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd

from .data import RawData, coerce_apps, coerce_interactions, coerce_users


DEFAULT_MISSING_SENTINEL = "__UNKNOWN__"


@dataclass
class FeatureBundle:
    X: pd.DataFrame
    y: pd.Series
    groups: pd.Series
    feature_columns: List[str]


def _build_manager_lookup(users: pd.DataFrame) -> pd.DataFrame:
    return (
        users[["userId", "department", "officeLocation", "seniority"]]
        .copy()
        .rename(
            columns={
                "userId": "manager_join_key",
                "department": "manager_department",
                "officeLocation": "manager_office",
                "seniority": "manager_seniority",
            }
        )
    )


def _fill_categoricals(df: pd.DataFrame, sentinel: str) -> pd.DataFrame:
    df["category"] = df["category"].fillna(sentinel).astype("string")
    df["manager_department"] = df["manager_department"].fillna(sentinel).astype("string")
    df["manager_office"] = df["manager_office"].fillna(sentinel).astype("string")

    df["department_filled"] = (
        df["department"].fillna(df["manager_department"]).fillna(sentinel).astype("string")
    )
    df["office_filled"] = (
        df["officeLocation"].fillna(df["manager_office"]).fillna(sentinel).astype("string")
    )
    df["managerId"] = df["managerId"].fillna(sentinel).astype("string")
    return df


def _numeric_features(df: pd.DataFrame) -> pd.DataFrame:
    df["seniority"] = df["seniority"].astype("Float64")
    df["manager_seniority"] = df["manager_seniority"].astype("Float64")
    df["manager_seniority_diff"] = df["seniority"] - df["manager_seniority"]
    return df


def _merge_metadata(interactions: pd.DataFrame, users: pd.DataFrame, apps: pd.DataFrame) -> pd.DataFrame:
    merged = (
        interactions.merge(
            users[["userId", "managerId", "department", "officeLocation", "isMachine", "seniority"]],
            on="userId",
            how="left",
        )
        .merge(apps[["appId", "category"]], on="appId", how="left")
    )
    mgr_lookup = _build_manager_lookup(users)
    merged = merged.merge(mgr_lookup, left_on="managerId", right_on="manager_join_key", how="left")
    return merged


def _select_feature_columns(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    feature_columns: List[str] = [
        "appId",
        "managerId",
        "category",
        "department_filled",
        "office_filled",
        "manager_department",
        "manager_office",
        "seniority",
        "manager_seniority",
        "manager_seniority_diff",
        "isMachine",
    ]
    return df[feature_columns].copy(), feature_columns


def build_features(raw: RawData, missing_sentinel: str = DEFAULT_MISSING_SENTINEL) -> FeatureBundle:
    """Create model-ready features, target, and group labels."""
    users = coerce_users(raw.users)
    apps = coerce_apps(raw.apps)
    interactions = coerce_interactions(raw.interactions)

    merged = _merge_metadata(interactions, users, apps)
    merged = _fill_categoricals(merged, missing_sentinel)
    merged = _numeric_features(merged)

    X, feature_columns = _select_feature_columns(merged)
    y = merged["permission"].astype(int)
    groups = merged["userId"]
    return FeatureBundle(X=X, y=y, groups=groups, feature_columns=feature_columns)


def build_scoring_matrix(
    submission_df: pd.DataFrame,
    users: pd.DataFrame,
    apps: pd.DataFrame,
    missing_sentinel: str = DEFAULT_MISSING_SENTINEL,
) -> pd.DataFrame:
    """Prepare inference features from submission requests + metadata."""
    users = coerce_users(users)
    apps = coerce_apps(apps)
    submission_df = submission_df.copy()
    for col in ("userId", "appId"):
        submission_df[col] = (
            pd.to_numeric(submission_df[col], errors="coerce").astype("Int64").astype("string")
        )

    submission_df["permission"] = 0  # placeholder to reuse pipeline
    merged = _merge_metadata(submission_df, users, apps)
    merged = _fill_categoricals(merged, missing_sentinel)
    merged = _numeric_features(merged)

    X, _ = _select_feature_columns(merged)
    return X
