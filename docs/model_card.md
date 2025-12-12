# Model Card — Access Permission Approval Prediction

## Overview
- **Use case**: Predict whether an internal app permission request will be approved, to assist reviewers or automate low-risk decisions.
- **Model**: CatBoost classifier on mixed tabular data with high-cardinality categoricals.
- **Data**: Historical approvals/denials (116k interactions, 3.7k users, 500 apps) plus user/app/org metadata.
- **Validation**: GroupKFold by `userId` to prevent leakage across the same user in train/test splits.

## Performance (OOF)
- ROC-AUC: ~0.914
- PR-AUC: ~0.913
- Operating threshold (max accuracy, ~0.472): TPR ~84.7%, FPR ~18.4%

## Features
- App identity (`appId`), category
- User org context (department, office, manager office/department, manager seniority delta)
- Missing-value sentinel for sparse org fields
- Group column: `userId` (used only for validation to avoid memorization)

## Data considerations
- Organizational fields contain substantial missingness; manager backfill recovers some context while preserving missingness as signal.
- New apps or org structures may reduce performance until sufficient history accumulates.
- Approval cost asymmetry may require adjusting the decision threshold; probabilities are calibrated to support this.

## Risks & mitigations
- **Leakage**: mitigated by user-grouped CV; `userId` excluded from training features.
- **Distribution shift**: monitor approval rate and app/department mix; retrain/retune threshold as needed.
- **Fairness**: segments (departments/offices) can have varied rates; add segment-level monitoring before production use.
- **Cold start**: expect weaker performance on unseen apps/managers; consider human-in-the-loop for such cases.

## Artifacts
- `models/model.cbm`: trained CatBoost model
- `models/metadata.json`: config, metrics, threshold, feature list
- `reports/roc_pr.png`, `reports/score_distribution.png`, `reports/confusion_matrix.png`, `reports/calibration.png`: evaluation figures from latest run
- `reports/metrics.json`: scalar metrics for quick inspection
- `reports/oof_predictions.csv`: OOF labels and scores used to generate plots
- `reports/segment_metrics.json`: per-segment metrics for top apps/departments (support ≥100)
- `reports/shap_importance.png`: mean |SHAP| bar chart (sample) for interpretability
