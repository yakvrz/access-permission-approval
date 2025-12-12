import argparse
import json
from pathlib import Path

import pandas as pd

from . import config as cfg
from .reporting import plot_calibration, plot_confusion, plot_roc_pr, plot_score_distribution


def regenerate_reports(config_path: str) -> None:
    config = cfg.load_config(config_path)
    reports_dir = Path(config["artifacts"]["reports_dir"])
    model_dir = Path(config["artifacts"]["model_dir"])

    metrics_path = reports_dir / "metrics.json"
    oof_path = reports_dir / "oof_predictions.csv"
    metadata_path = model_dir / "metadata.json"

    if not metrics_path.exists() or not oof_path.exists() or not metadata_path.exists():
        raise FileNotFoundError("Run training first to create metrics, metadata, and OOF predictions.")

    oof = pd.read_csv(oof_path)
    with metadata_path.open() as f:
        metadata = json.load(f)

    y_true = oof["permission"].values
    scores = oof["score"].values
    threshold = metadata["threshold"]

    plot_roc_pr(y_true, scores, reports_dir)
    plot_score_distribution(y_true, scores, reports_dir)
    plot_confusion(y_true, scores, threshold, reports_dir)
    plot_calibration(y_true, scores, reports_dir)
    print(json.dumps({"metrics_path": str(metrics_path), "reports_dir": str(reports_dir)}, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Regenerate evaluation plots from saved OOF predictions.")
    parser.add_argument("--config", default="config/default.yaml", help="Path to YAML config.")
    args = parser.parse_args()
    regenerate_reports(args.config)


if __name__ == "__main__":
    main()
