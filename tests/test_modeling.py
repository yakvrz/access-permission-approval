import pandas as pd

from access_perm.data import RawData
from access_perm.features import build_features
from access_perm.modeling import train_model
from access_perm.predict import predict_requests


def load_fixture_data():
    base = "tests/fixtures/small"
    users = pd.read_csv(f"{base}/users_metadata.csv")
    apps = pd.read_csv(f"{base}/apps_metadata.csv")
    interactions = pd.read_csv(f"{base}/interactions.csv")
    submission = pd.read_csv(f"{base}/submission.csv")
    return RawData(users=users, apps=apps, interactions=interactions, submission=submission)


def test_training_and_prediction_smoke():
    raw = load_fixture_data()
    sentinel = "__UNK__"
    bundle = build_features(raw, missing_sentinel=sentinel)
    cat_cols = [
        "appId",
        "managerId",
        "category",
        "department_filled",
        "office_filled",
        "manager_department",
        "manager_office",
    ]
    model_params = {"iterations": 30, "learning_rate": 0.1, "depth": 4, "l2_leaf_reg": 3, "random_seed": 42}
    result = train_model(
        X=bundle.X,
        y=bundle.y,
        groups=bundle.groups,
        categorical_columns=cat_cols,
        model_params=model_params,
        n_splits=3,
    )
    assert len(result.oof_pred) == len(bundle.y)
    probas, labels = predict_requests(
        result.model,
        raw.submission,
        raw.users,
        raw.apps,
        cat_cols,
        result.threshold,
        sentinel,
    )
    assert len(probas) == len(raw.submission)
    assert set(labels).issubset({0, 1})
