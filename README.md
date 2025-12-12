# Access Permission Approval Prediction

This project predicts the approval outcome of internal app permission requests by combining historical decisions with user, application, and organizational metadata. It trains a CatBoost classifier and evaluates it with user-grouped cross-validation to avoid leakage. The repository includes a Python package with CLI entrypoints, configuration files, tests with synthetic fixtures, and a narrative notebook explaining the approach and results.

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]

# Place CSVs into data/ (see data/README.md)
make train        # trains with GroupKFold CV, saves model + metrics
make report       # regenerates evaluation plots and markdown summary
make test         # runs unit tests on the synthetic fixture set
```

Generated artifacts land in `models/` and `reports/`. The accompanying notebook in `notebooks/` is the portfolio-ready narrative; it references the same pipeline functions used by the CLI.

To sanity-check the pipeline without the private data, run with the bundled synthetic fixture config:

```bash
python -m access_perm.train --config config/sample.yaml
```

## Latest results
- OOF ROC-AUC: 0.932 | PR-AUC: 0.930 | Accuracy: 0.857
- Operating threshold: 0.542 (TPR ~0.854, FPR ~0.140)
- Figures: `docs/figures/roc_pr.png`, `docs/figures/score_distribution.png`, `docs/figures/confusion_matrix.png`, `docs/figures/calibration.png`, `docs/figures/shap_importance.png`

## Repo layout
- `src/access_perm/` — reusable pipeline code (data prep, features, training, prediction, reports)
- `scripts/` — thin CLI wrappers (train, predict, report)
- `config/` — YAML configs (paths, hyperparameters)
- `notebooks/` — write-up notebook (text-first, minimal code)
- `tests/` — pytest suite with synthetic fixtures
- `models/`, `reports/` — saved models/plots/metrics (not tracked)
- `data/` — place raw CSVs here (not tracked)

## Highlights
- **Leakage-safe validation**: GroupKFold by `userId` mirrors the intended deployment setting.
- **Robust handling of missing org data** via sentinel and manager backfill.
- **CatBoost** for high-cardinality categorical features; calibrated probabilities and configurable decision thresholds.
- **Segment coverage & explainability**: per-segment metrics for top apps/departments and SHAP-based feature attributions shipped with reports.
- **Reproducibility**: versioned config, fixed seeds, and saved metadata alongside the model.

## Decisions & assumptions
- Data stays local; the repo ships only schema expectations and synthetic test fixtures.
- Error costs are treated as roughly symmetric; adjust the operating threshold in `config/default.yaml` if precision/recall trade-offs shift.
- The CLI defaults to paths in `config/default.yaml`; override with `--config` or per-flag overrides if you keep data elsewhere.
