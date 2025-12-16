import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import great_expectations as ge
import numpy as np
import pandas as pd

from .data import (
    REQUIRED_APP_COLUMNS,
    REQUIRED_INTERACTION_COLUMNS,
    REQUIRED_USER_COLUMNS,
    RawData,
)
from .features import FeatureBundle


def _json_default(obj: Any):
    if isinstance(obj, (np.generic,)):
        return obj.item()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


@dataclass
class ValidationResult:
    name: str
    success: bool
    report_path: Path
    statistics: Dict[str, Any]


def _run_validation(ds, name: str, output_dir: Path) -> ValidationResult:
    validation = ds.validate(result_format="SUMMARY")
    result = validation.to_json_dict() if hasattr(validation, "to_json_dict") else validation
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{name}_validation.json"
    with path.open("w") as f:
        json.dump(result, f, indent=2, default=_json_default)
    stats = result.get("statistics", {})
    return ValidationResult(name=name, success=bool(result.get("success")), report_path=path, statistics=stats)


def _expect_required_columns(ds, required: set) -> None:
    for col in required:
        ds.expect_column_to_exist(col)


def _expect_non_null(ds, columns: List[str], mostly: float) -> None:
    for col in columns:
        ds.expect_column_values_to_not_be_null(col, mostly=mostly)


def validate_raw_data(raw: RawData, output_dir: Path, mostly: float = 0.97) -> List[ValidationResult]:
    """Run basic schema checks on raw CSV inputs."""
    results: List[ValidationResult] = []

    users_ds = ge.from_pandas(raw.users)
    _expect_required_columns(users_ds, REQUIRED_USER_COLUMNS)
    _expect_non_null(users_ds, ["userId", "department", "officeLocation"], mostly=mostly)
    users_ds.expect_column_values_to_be_in_set("isMachine", value_set=[True, False], mostly=1.0)
    users_ds.expect_column_values_to_be_between("seniority", min_value=0, mostly=mostly)
    results.append(_run_validation(users_ds, "users", output_dir))

    apps_ds = ge.from_pandas(raw.apps)
    _expect_required_columns(apps_ds, REQUIRED_APP_COLUMNS)
    _expect_non_null(apps_ds, ["appId", "category"], mostly=mostly)
    results.append(_run_validation(apps_ds, "apps", output_dir))

    interactions_ds = ge.from_pandas(raw.interactions)
    _expect_required_columns(interactions_ds, REQUIRED_INTERACTION_COLUMNS)
    _expect_non_null(interactions_ds, ["userId", "appId"], mostly=mostly)
    interactions_ds.expect_column_values_to_be_in_set("permission", value_set=[0, 1], mostly=1.0)
    results.append(_run_validation(interactions_ds, "interactions", output_dir))

    if raw.submission is not None:
        submission_ds = ge.from_pandas(raw.submission)
        _expect_required_columns(submission_ds, {"userId", "appId"})
        _expect_non_null(submission_ds, ["userId", "appId"], mostly=mostly)
        results.append(_run_validation(submission_ds, "submission", output_dir))

    return results


def validate_features(bundle: FeatureBundle, output_dir: Path, mostly: float = 0.97) -> ValidationResult:
    """Validate engineered features and target before training."""
    df = bundle.X.copy()
    df["permission"] = bundle.y.values
    ds = ge.from_pandas(df)
    ds.expect_table_columns_to_contain_set(column_set=list(bundle.feature_columns))
    _expect_non_null(ds, bundle.feature_columns, mostly=mostly)
    ds.expect_column_values_to_be_in_set("isMachine", value_set=[True, False], mostly=1.0)
    if "seniority" in df.columns:
        ds.expect_column_values_to_be_between("seniority", min_value=0, mostly=mostly)
    if "manager_seniority" in df.columns:
        ds.expect_column_values_to_be_between("manager_seniority", min_value=0, mostly=mostly)
    return _run_validation(ds, "features", output_dir)


def validate_submission(df: pd.DataFrame, output_dir: Path, mostly: float = 0.95) -> ValidationResult:
    """Validate a scoring submission payload before inference."""
    ds = ge.from_pandas(df)
    _expect_required_columns(ds, {"userId", "appId"})
    _expect_non_null(ds, ["userId", "appId"], mostly=mostly)
    return _run_validation(ds, "submission", output_dir)
