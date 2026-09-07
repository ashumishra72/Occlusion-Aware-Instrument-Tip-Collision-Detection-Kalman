"""
LightGBM entry in the model-comparison suite. Same StratifiedGroupKFold CV,
smoothing, and threshold logic as the other models (see model_cv_utils.py).

Usage:
    python scripts/07_train_lightgbm.py
"""

import sys
from pathlib import Path

import lightgbm as lgb
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model_cv_utils import load_features, run_cv, save_model_results

MODEL_NAME = "LightGBM"


def fit_fn(X_tr, y_tr, X_val, y_val):
    n_pos, n_neg = int(y_tr.sum()), int((y_tr == 0).sum())
    model = lgb.LGBMClassifier(
        n_estimators=800, max_depth=5, num_leaves=31, learning_rate=0.05,
        subsample=0.9, colsample_bytree=0.9, min_child_samples=10,
        objective="binary", scale_pos_weight=n_neg / max(n_pos, 1),
        random_state=42, verbosity=-1,
    )
    model.fit(
        X_tr, y_tr, eval_set=[(X_val, y_val)], eval_metric="average_precision",
        callbacks=[lgb.early_stopping(40, verbose=False), lgb.log_evaluation(0)],
    )
    return model


def main():
    df, feature_cols = load_features()
    oof, fold_metrics, threshold = run_cv(df, feature_cols, fit_fn, log_prefix=f"[{MODEL_NAME}] ")
    save_model_results(MODEL_NAME, oof, fold_metrics, threshold)

    split = int(len(df) * 0.85)
    final_model = fit_fn(df[feature_cols].iloc[:split], df["label_collision"].iloc[:split],
                          df[feature_cols].iloc[split:], df["label_collision"].iloc[split:])
    importance = pd.Series(final_model.feature_importances_, index=feature_cols)
    out_dir = Path(__file__).resolve().parent.parent / "results" / "feature_importance"
    out_dir.mkdir(parents=True, exist_ok=True)
    importance.sort_values(ascending=False).to_csv(
        out_dir / f"{MODEL_NAME.lower()}_importance.csv", header=["importance"], index_label="feature"
    )


if __name__ == "__main__":
    main()
