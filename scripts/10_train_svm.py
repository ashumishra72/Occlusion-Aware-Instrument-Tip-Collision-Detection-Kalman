"""
SVM (RBF kernel) entry in the model-comparison suite. Features are
standardized (SVMs are scale-sensitive, unlike tree models) inside a
Pipeline so the scaler is fit only on training data per fold - no leakage.
Uses probability=True (Platt scaling) since we need calibrated probabilities
for the same threshold-tuning framework as the other models, and documents
that choice here per the project's evaluation methodology.

Usage:
    python scripts/10_train_svm.py
"""

import sys
from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model_cv_utils import load_features, run_cv, save_model_results

MODEL_NAME = "SVM"


def fit_fn(X_tr, y_tr, X_val, y_val):
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel="rbf", C=1.0, gamma="scale", probability=True,
                     class_weight="balanced", random_state=42)),
    ])
    model.fit(X_tr, y_tr)
    return model


def main():
    df, feature_cols = load_features()
    oof, fold_metrics, threshold = run_cv(df, feature_cols, fit_fn, log_prefix=f"[{MODEL_NAME}] ")
    save_model_results(MODEL_NAME, oof, fold_metrics, threshold,
                        extra={"note": "probability=True uses Platt scaling (5-fold internal CV) "
                                        "for calibrated probabilities; SVM has no native feature "
                                        "importance so none is reported for this model."})


if __name__ == "__main__":
    main()
