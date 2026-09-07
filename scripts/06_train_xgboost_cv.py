"""
XGBoost entry in the model-comparison suite. This mirrors the production
model trained by 03_train_classifier.py (same hyperparameters), re-run
through the shared model_cv_utils framework so it's directly comparable to
the other algorithms in results/tables/model_comparison.csv on identical
folds/smoothing/threshold logic.

Usage:
    python scripts/06_train_xgboost_cv.py
"""

import sys
from pathlib import Path

import pandas as pd
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model_cv_utils import load_features, run_cv, save_model_results

MODEL_NAME = "XGBoost"


def fit_fn(X_tr, y_tr, X_val, y_val):
    n_pos, n_neg = int(y_tr.sum()), int((y_tr == 0).sum())
    model = xgb.XGBClassifier(
        n_estimators=800, max_depth=5, learning_rate=0.05, subsample=0.9,
        colsample_bytree=0.9, min_child_weight=2, objective="binary:logistic",
        eval_metric="aucpr", scale_pos_weight=n_neg / max(n_pos, 1),
        early_stopping_rounds=40, random_state=42,
    )
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    return model


def main():
    df, feature_cols = load_features()
    oof, fold_metrics, threshold = run_cv(df, feature_cols, fit_fn, log_prefix=f"[{MODEL_NAME}] ")
    save_model_results(MODEL_NAME, oof, fold_metrics, threshold)

    final_model = fit_fn(df[feature_cols].iloc[:int(len(df) * 0.85)],
                          df["label_collision"].iloc[:int(len(df) * 0.85)],
                          df[feature_cols].iloc[int(len(df) * 0.85):],
                          df["label_collision"].iloc[int(len(df) * 0.85):])
    importance = pd.Series(final_model.feature_importances_, index=feature_cols)
    out_dir = Path(__file__).resolve().parent.parent / "results" / "feature_importance"
    out_dir.mkdir(parents=True, exist_ok=True)
    importance.sort_values(ascending=False).to_csv(
        out_dir / f"{MODEL_NAME.lower()}_importance.csv", header=["importance"], index_label="feature"
    )


if __name__ == "__main__":
    main()
