# Access Permission Approval Prediction

This project predicts the approval outcome of internal app permission requests by combining historical decisions with user, application, and organizational metadata. It trains a CatBoost classifier and evaluates it with user-grouped cross-validation to avoid leakage. The repository includes a Python package with CLI entrypoints, configuration files, tests with synthetic fixtures, and a narrative notebook explaining the approach and results.

## Quickstart

```bash
uv venv --python 3.11           # or python -m venv .venv
source .venv/bin/activate
uv pip install -e ".[dev]"      # installs app + dev tools

# Place CSVs into data/ (see data/README.md)
make train          # trains with GroupKFold CV, saves model + metrics
make report         # regenerates evaluation plots and markdown summary
make test           # runs unit tests on the synthetic fixture set
make pipeline       # Prefect flow: validate -> train -> monitor
make monitor        # rerun drift/performance monitoring on new data
make serve          # start FastAPI prediction service
```

Generated artifacts land in:
- `models/` — trained CatBoost model + metadata
- `reports/` — plots/metrics from training
- `validation/` — Great Expectations validation results for raw data, features, and scoring payloads
- `monitoring/` — Evidently drift reports + performance summaries
- `mlruns/` — MLflow experiment tracking (local file store by default)

The accompanying notebook in `notebooks/` is the portfolio-ready narrative; it references the same pipeline functions used by the CLI.

To sanity-check the pipeline without the private data, run with the bundled synthetic fixture config:

```bash
python -m access_perm.train --config config/sample.yaml
```

## Production-ish workflow
- **Train & track**: `python -m access_perm.train --config config/default.yaml` logs params/metrics/artifacts to MLflow and saves a monitoring baseline at `monitoring/reference.parquet`.
- **Monitor**: `python -m access_perm.pipeline monitor --config config/default.yaml --current-interactions path/to/new.csv` generates Evidently drift HTML/JSON plus performance metrics in `monitoring/`.
- **Serve**: `uvicorn access_perm.service:app --host 0.0.0.0 --port 8000` (optionally set `ACCESS_PERM_CONFIG` to point at a different YAML). POST to `/predict` with `[{ "userId": ..., "appId": ... }]`.
- **CI hooks**: validations live in `validation/`; the pipeline fails fast if schemas drift from expectations.

## Latest results
- OOF ROC-AUC: 0.932 | PR-AUC: 0.930 | Accuracy: 0.857
- Operating threshold: 0.542 (TPR ~0.854, FPR ~0.140)
- Figures: `docs/figures/roc_pr.png`, `docs/figures/score_distribution.png`, `docs/figures/confusion_matrix.png`, `docs/figures/calibration.png`, `docs/figures/shap_importance.png`

## Repo layout
- `src/access_perm/` — reusable pipeline code (data prep, features, training, prediction, reports)
- `scripts/` — thin CLI wrappers (train, predict, report, pipeline, serve)
- `config/` — YAML configs (paths, hyperparameters)
- `notebooks/` — write-up notebook (text-first, minimal code)
- `tests/` — pytest suite with synthetic fixtures
- `models/`, `reports/` — saved models/plots/metrics (not tracked)
- `validation/` — validation outputs (Great Expectations JSON)
- `monitoring/` — drift/performance reports (Evidently + HTML/JSON)
- `mlruns/` — local MLflow tracking store
- `data/` — place raw CSVs here (not tracked)

## Highlights
- **Leakage-safe validation**: GroupKFold by `userId` mirrors the intended deployment setting.
- **Robust handling of missing org data** via sentinel and manager backfill.
- **CatBoost** for high-cardinality categorical features; calibrated probabilities and configurable decision thresholds.
- **Segment coverage & explainability**: per-segment metrics for top apps/departments and SHAP-based feature attributions shipped with reports.
- **Data contracts baked in**: Great Expectations checks for raw CSVs, engineered features, and scoring payloads; failures block the pipeline.
- **Experiment tracking**: MLflow logs params/metrics/artifacts and registers the model (local file store by default).
- **Monitoring**: Evidently data/target drift reports and rolling performance JSON+HTML in `monitoring/`; reference baseline saved during training.
- **Serving**: FastAPI app (`access_perm.service:app`) with `/health` and `/predict`; optional MLflow inference logging.
- **Reproducibility**: versioned config, fixed seeds, saved metadata/baseline alongside the model.

## Decisions & assumptions
- Data stays local; the repo ships only schema expectations and synthetic test fixtures.
- Error costs are treated as roughly symmetric; adjust the operating threshold in `config/default.yaml` if precision/recall trade-offs shift.
- The CLI defaults to paths in `config/default.yaml`; override with `--config` or per-flag overrides if you keep data elsewhere.
