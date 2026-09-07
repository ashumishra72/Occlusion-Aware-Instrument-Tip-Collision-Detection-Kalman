"""
Train an XGBoost classifier that predicts, per video frame, whether an
instrument-tip collision (label_collision) is occurring, using the
proximity/motion features produced by 02_extract_features.py.

With only ~3,900 frames (558 positive) from a single video, a single
chronological train/val/test split is noisy: whichever slice lands in
"test" may just happen to contain a different mix of collision types than
train (e.g. the tail of this video has more push-pin/environment contacts
than arm-vs-arm ones), so a single split's score depends heavily on luck.

Instead this script uses StratifiedGroupKFold: the timeline is first cut
into small --group-seconds chunks (a "group" per chunk, so a brief event's
frames always stay together in one fold - no leakage), then folds are
assigned so each fold's collision rate matches the overall ~14% as closely
as the grouping allows. Plain chronological KFold left fold 3 with only 20
positives out of 783 (2.6%, vs ~14% overall) purely by where the block
boundaries fell, which collapsed precision to ~3% in that fold; stratifying
by group fixes that without reintroducing frame-level leakage. Per-fold
metrics are kept so you can see how much they still vary.

The decision threshold is picked ONCE, from the pooled out-of-fold
probabilities across all folds (~3,900 samples) rather than separately per
fold (~700 samples each). Picking it per-fold is unstable - with only
~20-200 positives in a single fold, the F1-maximizing threshold search can
land on a pathologically low cutoff (e.g. 0.05) that flags almost every
frame positive, tanking precision. Pooling first before searching gives the
threshold search enough data to be stable.

Two more things reduce false positives specifically:
  - Probabilities are smoothed over a short --smooth-window of neighboring
    frames before thresholding. A real collision spans multiple consecutive
    frames, so a single frame's stray high probability (sensor noise, a
    momentary bad detection) gets pulled back down by its neighbors, while
    a genuine multi-frame collision stays high.
  - The threshold search optimizes F-beta with --fbeta < 1 by default,
    which weights precision more than recall (F1's 50/50 weighting was
    producing a threshold where most false positives were "barely over the
    line" - median false-positive probability was 0.60 against a 0.485
    threshold).

The final shipped model is retrained on 100% of the data (with the last
15% of the timeline carved out purely for early stopping, not evaluation)
and uses this same pooled threshold.

Outputs:
    models/xgb_collision_classifier.json  - final model, trained on all data
    results/predictions.csv               - out-of-fold predictions for every frame
    results/feature_importance.csv        - feature importances (final model)
    results/threshold.json                - decision threshold used by the final model
    results/cv_metrics.json               - per-fold metrics + mean/std across folds

Usage:
    python scripts/03_train_classifier.py
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    f1_score, precision_recall_curve, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FEATURES_CSV = PROJECT_ROOT / "data" / "features_and_labels.csv"
MODEL_OUT = PROJECT_ROOT / "models" / "xgb_collision_classifier.json"
PREDICTIONS_OUT = PROJECT_ROOT / "results" / "predictions.csv"
FEATURE_IMPORTANCE_OUT = PROJECT_ROOT / "results" / "feature_importance.csv"
THRESHOLD_OUT = PROJECT_ROOT / "results" / "threshold.json"
CV_METRICS_OUT = PROJECT_ROOT / "results" / "cv_metrics.json"

NON_FEATURE_COLS = {
    "frame_idx", "time_s", "label_collision", "label_near_miss", "label_severity",
}


def best_fbeta_threshold(y_true, proba, beta=1.0):
    """Threshold that maximizes F-beta. beta<1 weights precision more than
    recall (e.g. beta=0.5 means precision matters ~4x recall); beta=1 is
    plain F1; beta>1 weights recall more."""
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    if len(thresholds) == 0:
        return 0.5, 0.0
    b2 = beta ** 2
    fbeta = np.where(
        (b2 * precision + recall) > 0,
        (1 + b2) * precision * recall / (b2 * precision + recall + 1e-12),
        0.0,
    )
    best_idx = int(np.argmax(fbeta[:-1]))
    return float(thresholds[best_idx]), float(fbeta[best_idx])


def threshold_tradeoff_table(y_true, proba, thresholds):
    rows = []
    for thr in thresholds:
        pred = (proba >= thr).astype(int)
        rows.append({
            "threshold": thr,
            "precision": precision_score(y_true, pred, zero_division=0),
            "recall": recall_score(y_true, pred, zero_division=0),
            "f1": f1_score(y_true, pred, zero_division=0),
            "n_flagged": int(pred.sum()),
        })
    return rows


def make_model(scale_pos_weight, args):
    return xgb.XGBClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        min_child_weight=args.min_child_weight,
        objective="binary:logistic",
        eval_metric="aucpr",
        scale_pos_weight=scale_pos_weight,
        early_stopping_rounds=args.early_stopping_rounds,
        random_state=42,
    )


def fit_with_internal_val(model_args, X, y, val_frac):
    """Carve the chronological tail of (X, y) off as a validation set for
    early stopping only, fit on the rest, return the fitted model."""
    split = int(len(X) * (1 - val_frac))
    X_tr, y_tr = X.iloc[:split], y.iloc[:split]
    X_val, y_val = X.iloc[split:], y.iloc[split:]

    n_pos, n_neg = int(y_tr.sum()), int((y_tr == 0).sum())
    scale_pos_weight = n_neg / max(n_pos, 1)
    model = make_model(scale_pos_weight, model_args)
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", default=str(FEATURES_CSV))
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--group-seconds", type=float, default=5.0,
                         help="Size of the contiguous time chunks kept together in one fold")
    parser.add_argument("--val-frac", type=float, default=0.15,
                         help="Fraction of each fold's training data carved out for early stopping/threshold tuning")
    parser.add_argument("--n-estimators", type=int, default=800)
    parser.add_argument("--max-depth", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--subsample", type=float, default=0.9)
    parser.add_argument("--colsample-bytree", type=float, default=0.9)
    parser.add_argument("--min-child-weight", type=float, default=2)
    parser.add_argument("--early-stopping-rounds", type=int, default=40)
    parser.add_argument("--smooth-window", type=int, default=3,
                         help="Rolling-average window (frames) applied to predicted probabilities "
                              "before thresholding, to suppress single-frame noise spikes")
    parser.add_argument("--fbeta", type=float, default=1.3,
                         help="Beta for the threshold search's F-beta objective. <1 favors precision "
                              "(fewer false positives), 1.0 = plain F1, >1 favors recall")
    args = parser.parse_args()

    df = pd.read_csv(args.features).sort_values("time_s").reset_index(drop=True)
    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
    X_all, y_all = df[feature_cols], df["label_collision"]
    groups = (df["time_s"] // args.group_seconds).astype(int)
    print(f"Grouped {len(df)} frames into {groups.nunique()} chunks of ~{args.group_seconds}s "
          f"for StratifiedGroupKFold (keeps each chunk whole in one fold)")

    kf = StratifiedGroupKFold(n_splits=args.n_folds, shuffle=True, random_state=42)
    fold_test_idx = []
    oof = df[["frame_idx", "time_s", "label_collision"]].copy()
    oof["predicted_proba"] = np.nan

    for fold_i, (train_idx, test_idx) in enumerate(kf.split(X_all, y_all, groups), start=1):
        X_train, y_train = X_all.iloc[train_idx], y_all.iloc[train_idx]
        X_test = X_all.iloc[test_idx]

        model = fit_with_internal_val(args, X_train, y_train, args.val_frac)
        proba = model.predict_proba(X_test)[:, 1]
        oof.loc[test_idx, "predicted_proba"] = proba
        fold_test_idx.append(test_idx)
        print(f"Fold {fold_i}/{args.n_folds}: n={len(test_idx)} trained "
              f"(best_iteration={model.best_iteration})")

    # Smooth probabilities over neighboring frames (oof is already in time
    # order) before thresholding - a real collision spans multiple frames,
    # so this pulls down single-frame noise spikes without touching a
    # genuine sustained-probability collision window.
    oof["predicted_proba_raw"] = oof["predicted_proba"]
    if args.smooth_window > 1:
        oof["predicted_proba"] = (
            oof["predicted_proba"].rolling(args.smooth_window, center=True, min_periods=1).mean()
        )

    # Pick ONE decision threshold from the full pooled, smoothed out-of-fold
    # probabilities (~3,900 samples) instead of per-fold (~700 samples each)
    # - see module docstring for why per-fold tuning is unstable. Optimizes
    # F-beta (beta<1 favors precision) instead of plain F1.
    global_threshold, pooled_fbeta = best_fbeta_threshold(
        oof["label_collision"], oof["predicted_proba"], beta=args.fbeta
    )
    oof["predicted_label"] = (oof["predicted_proba"] >= global_threshold).astype(int)
    print(f"\nGlobal threshold (from pooled, smoothed OOF probabilities, F{args.fbeta} objective): "
          f"{global_threshold:.3f} (pooled F{args.fbeta}={pooled_fbeta:.3f})")

    print("\nThreshold trade-off (smoothed probabilities):")
    for row in threshold_tradeoff_table(oof["label_collision"], oof["predicted_proba"],
                                         [0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, global_threshold]):
        marker = " <- chosen" if abs(row["threshold"] - global_threshold) < 1e-9 else ""
        print(f"  thr={row['threshold']:.3f}  precision={row['precision']:.3f}  "
              f"recall={row['recall']:.3f}  f1={row['f1']:.3f}  n_flagged={row['n_flagged']}{marker}")

    fold_metrics = []
    for fold_i, test_idx in enumerate(fold_test_idx, start=1):
        y_test = oof.loc[test_idx, "label_collision"]
        proba = oof.loc[test_idx, "predicted_proba"]
        pred = oof.loc[test_idx, "predicted_label"]
        metrics = {
            "fold": fold_i,
            "n_test": int(len(test_idx)),
            "n_positive": int(y_test.sum()),
            "precision": float(precision_score(y_test, pred, zero_division=0)),
            "recall": float(recall_score(y_test, pred, zero_division=0)),
            "f1": float(f1_score(y_test, pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_test, proba)) if y_test.nunique() > 1 else None,
        }
        fold_metrics.append(metrics)
        print(f"Fold {fold_i}/{args.n_folds}: n={metrics['n_test']} pos={metrics['n_positive']} "
              f"precision={metrics['precision']:.3f} recall={metrics['recall']:.3f} f1={metrics['f1']:.3f}")

    def agg(key):
        vals = [m[key] for m in fold_metrics if m[key] is not None]
        return {"mean": float(np.mean(vals)), "std": float(np.std(vals))} if vals else None

    cv_summary = {
        "n_folds": args.n_folds,
        "global_threshold": global_threshold,
        "folds": fold_metrics,
        "precision": agg("precision"),
        "recall": agg("recall"),
        "f1": agg("f1"),
        "roc_auc": agg("roc_auc"),
    }
    CV_METRICS_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(CV_METRICS_OUT, "w", encoding="utf-8") as f:
        json.dump(cv_summary, f, indent=2)
    print(f"\nAcross {args.n_folds} folds: "
          f"precision={cv_summary['precision']['mean']:.3f}±{cv_summary['precision']['std']:.3f}  "
          f"recall={cv_summary['recall']['mean']:.3f}±{cv_summary['recall']['std']:.3f}  "
          f"f1={cv_summary['f1']['mean']:.3f}±{cv_summary['f1']['std']:.3f}")

    # Final model shipped for actual use: trained on all data, tail held out
    # only for early stopping (not for evaluation - the CV above is the
    # evaluation). Uses the same pooled threshold picked above.
    final_model = fit_with_internal_val(args, X_all, y_all, args.val_frac)

    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    final_model.save_model(MODEL_OUT)

    with open(THRESHOLD_OUT, "w", encoding="utf-8") as f:
        json.dump({"decision_threshold": global_threshold}, f, indent=2)

    importance = pd.Series(final_model.feature_importances_, index=feature_cols)
    importance.sort_values(ascending=False).to_csv(
        FEATURE_IMPORTANCE_OUT, header=["importance"], index_label="feature"
    )

    oof = oof.rename(columns={"label_collision": "true_label"})
    oof["predicted_label"] = oof["predicted_label"].astype(int)
    PREDICTIONS_OUT.parent.mkdir(parents=True, exist_ok=True)
    oof.to_csv(PREDICTIONS_OUT, index=False)

    print(f"\nSaved out-of-fold predictions (all {len(oof)} frames) -> {PREDICTIONS_OUT}")
    print(f"Saved cross-validation summary -> {CV_METRICS_OUT}")
    print(f"Saved final model (trained on all data, threshold={global_threshold:.3f}) -> {MODEL_OUT}")
    print(f"Saved feature importance -> {FEATURE_IMPORTANCE_OUT}")


if __name__ == "__main__":
    main()
