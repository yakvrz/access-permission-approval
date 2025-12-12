from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd


REQUIRED_USER_COLUMNS = {"userId", "managerId", "department", "officeLocation", "isMachine", "seniority"}
REQUIRED_APP_COLUMNS = {"appId", "category"}
REQUIRED_INTERACTION_COLUMNS = {"userId", "appId", "permission"}


@dataclass
class RawData:
    users: pd.DataFrame
    apps: pd.DataFrame
    interactions: pd.DataFrame
    submission: Optional[pd.DataFrame] = None


def _read_csv(path: str | Path, **kwargs) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Expected file not found: {path}")
    return pd.read_csv(path, **kwargs)


def load_raw_data(paths: Dict[str, str | Path]) -> RawData:
    users = _read_csv(paths["users"])
    apps = _read_csv(paths["apps"])
    interactions = _read_csv(paths["interactions"])
    submission = None
    if "submission" in paths and Path(paths["submission"]).exists():
        submission = _read_csv(paths["submission"])
    return RawData(users=users, apps=apps, interactions=interactions, submission=submission)


def _validate_columns(df: pd.DataFrame, required: set, name: str) -> None:
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{name} missing required columns: {sorted(missing)}")


def coerce_users(df: pd.DataFrame) -> pd.DataFrame:
    _validate_columns(df, REQUIRED_USER_COLUMNS, "users")
    out = df.copy()
    for col in ("userId", "managerId"):
        out[col] = pd.to_numeric(out[col], errors="coerce").astype("Int64").astype("string")
    out["department"] = out["department"].astype("string")
    out["officeLocation"] = out["officeLocation"].astype("string")
    out["isMachine"] = (
        out["isMachine"]
        .apply(lambda x: str(x).strip().lower() if pd.notna(x) else "")
        .map({"true": True, "1": True, "yes": True, "false": False, "0": False, "no": False})
        .fillna(False)
        .astype(bool)
    )
    out["seniority"] = pd.to_numeric(out["seniority"], errors="coerce").astype("Int64")
    return out


def coerce_apps(df: pd.DataFrame) -> pd.DataFrame:
    _validate_columns(df, REQUIRED_APP_COLUMNS, "apps")
    out = df.copy()
    out["appId"] = pd.to_numeric(out["appId"], errors="coerce").astype("Int64").astype("string")
    out["category"] = out["category"].astype("string").str.strip().str.lower()
    return out


def coerce_interactions(df: pd.DataFrame) -> pd.DataFrame:
    _validate_columns(df, REQUIRED_INTERACTION_COLUMNS, "interactions")
    out = df.copy()
    for col in ("userId", "appId"):
        out[col] = pd.to_numeric(out[col], errors="coerce").astype("Int64").astype("string")
    out["permission"] = pd.to_numeric(out["permission"], errors="coerce").fillna(0).astype(int)
    return out
