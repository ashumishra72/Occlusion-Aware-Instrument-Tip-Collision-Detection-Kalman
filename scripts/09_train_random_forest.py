"""
Random Forest entry in the model-comparison suite. No early stopping (not
applicable to bagged trees), so the validation slice each fold sets aside
for other models is simply unused here - kept anyway so every model sees
the exact same effective training set per fold for a fair comparison.

Usage:
    python scripts/09_train_random_forest.py
"""

import sys
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model_cv_utils import load_features, run_cv, save_model_results

MODEL_NAME = "RandomForest"


def fit_fn(X_tr, y_tr, X_val, y_val):
    model = RandomForestClassifier(
        n_estimators=500, max_depth=10, min_samples_leaf=3,
        class_weight="balanced", random_state=42, n_jobs=-1,
    )
    model.fit(X_tr, y_tr)
    return model


def main():
    df, feature_cols = load_features()
    oof, fold_metrics, threshold = run_cv(df, feature_cols, fit_fn, log_prefix=f"[{MODEL_NAME}] ")
    save_model_results(MODEL_NAME, oof, fold_metrics, threshold)

    split = int(len(df) * 0.85)
    final_model = fit_fn(df[feature_cols].iloc[:split], df["label_collision"].iloc[:split], None, None)
    importance = pd.Series(final_model.feature_importances_, index=feature_cols)
    out_dir = Path(__file__).resolve().parent.parent / "results" / "feature_importance"
    out_dir.mkdir(parents=True, exist_ok=True)
    importance.sort_values(ascending=False).to_csv(
        out_dir / f"{MODEL_NAME.lower()}_importance.csv", header=["importance"], index_label="feature"
    )


if __name__ == "__main__":
    main()
