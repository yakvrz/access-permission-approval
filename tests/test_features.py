import pandas as pd

from access_perm.data import RawData
from access_perm.features import build_features, build_scoring_matrix, DEFAULT_MISSING_SENTINEL


def load_fixture_data():
    base = "tests/fixtures/small"
    users = pd.read_csv(f"{base}/users_metadata.csv")
    apps = pd.read_csv(f"{base}/apps_metadata.csv")
    interactions = pd.read_csv(f"{base}/interactions.csv")
    submission = pd.read_csv(f"{base}/submission.csv")
    return RawData(users=users, apps=apps, interactions=interactions, submission=submission)


def test_manager_backfill_and_sentinel():
    raw = load_fixture_data()
    bundle = build_features(raw, missing_sentinel="__UNK__")
    df = bundle.X.copy()
    df["userId"] = bundle.groups.values

    # User 2 should inherit manager 10's department (engineering)
    user2_department = df.loc[df["userId"] == "2", "department_filled"].iloc[0]
    assert user2_department == "engineering"

    # User 4 has no manager info; should use sentinel
    user4_department = df.loc[df["userId"] == "4", "department_filled"].iloc[0]
    assert user4_department == "__UNK__"

    # Check seniority difference calculation
    user1_diff = df.loc[df["userId"] == "1", "manager_seniority_diff"].iloc[0]
    assert user1_diff == 3 - 6  # user1 seniority 3, manager 6


def test_scoring_matrix_columns_align():
    raw = load_fixture_data()
    X_score = build_scoring_matrix(
        submission_df=raw.submission,
        users=raw.users,
        apps=raw.apps,
        missing_sentinel=DEFAULT_MISSING_SENTINEL,
    )
    train_bundle = build_features(raw, missing_sentinel=DEFAULT_MISSING_SENTINEL)
    assert list(X_score.columns) == train_bundle.feature_columns
    # Ensure no NaNs in categorical columns that should be filled
    for col in ["department_filled", "office_filled", "category", "manager_department", "manager_office"]:
        assert X_score[col].isna().sum() == 0
