"""
Generates the comparison figures and supporting tables from the per-model
results already saved by scripts 06-11 (results/model_comparison/*.json,
*_predictions.csv) and results/tables/model_comparison.csv.

Produces:
    figures/model_comparison/metric_bars.png       - Accuracy/Precision/Recall/F1/ROC-AUC per model
    figures/model_comparison/roc_curves.png         - ROC curve overlay, all models
    figures/model_comparison/pr_curves.png          - Precision-Recall curve overlay, all models
    figures/results/cross_validation_stability.png  - per-fold F1 mean +/- std per model
    figures/results/confusion_matrices.png          - confusion matrix grid, all models
    figures/results/feature_importance_comparison.png - top features, tree models overlaid
    results/error_analysis/error_cases_<best_model>.csv

Usage:
    python scripts/12_model_comparison_report.py
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve

PROJECT_ROOT = Path(__file__).resolve().parent.parent
COMPARISON_DIR = PROJECT_ROOT / "results" / "model_comparison"
TABLES_DIR = PROJECT_ROOT / "results" / "tables"
FEATURE_IMPORTANCE_DIR = PROJECT_ROOT / "results" / "feature_importance"
ERROR_DIR = PROJECT_ROOT / "results" / "error_analysis"
FIG_COMPARISON_DIR = PROJECT_ROOT / "results" / "charts" / "model_comparison"
FIG_RESULTS_DIR = PROJECT_ROOT / "results" / "charts" / "model_comparison"
FEATURES_CSV = PROJECT_ROOT / "data" / "features_and_labels.csv"

MODELS = ["XGBoost", "LightGBM", "CatBoost", "RandomForest", "SVM", "LSTM", "GRU", "TCN"]
TREE_MODELS = ["xgboost", "lightgbm", "catboost", "randomforest"]


def load_all():
    table = pd.read_csv(TABLES_DIR / "model_comparison.csv")
    preds, cvs = {}, {}
    for m in MODELS:
        pred_path = COMPARISON_DIR / f"{m}_predictions.csv"
        cv_path = COMPARISON_DIR / f"{m}_cv_metrics.json"
        if pred_path.exists():
            preds[m] = pd.read_csv(pred_path)
        if cv_path.exists():
            cvs[m] = json.load(open(cv_path))
    return table, preds, cvs


def plot_metric_bars(table, out_path):
    metrics = ["Accuracy", "Precision", "Recall", "F1", "ROC_AUC"]
    x = np.arange(len(table))
    width = 0.15
    fig, ax = plt.subplots(figsize=(12, 6))
    for i, metric in enumerate(metrics):
        ax.bar(x + i * width, table[metric], width, label=metric)
    ax.set_xticks(x + width * (len(metrics) - 1) / 2)
    ax.set_xticklabels(table["Model"], rotation=20)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison Across Metrics (pooled out-of-fold, 5-fold StratifiedGroupKFold CV)")
    ax.legend(loc="upper right", ncol=5, fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_roc_curves(preds, out_path):
    fig, ax = plt.subplots(figsize=(7, 6))
    for m, df in preds.items():
        if df["true_label"].nunique() < 2:
            continue
        fpr, tpr, _ = roc_curve(df["true_label"], df["predicted_proba"])
        from sklearn.metrics import auc
        ax.plot(fpr, tpr, label=f"{m} (AUC={auc(fpr, tpr):.3f})")
    ax.plot([0, 1], [0, 1], "--", color="gray", label="Chance")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves (pooled out-of-fold predictions)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_pr_curves(preds, out_path):
    fig, ax = plt.subplots(figsize=(7, 6))
    for m, df in preds.items():
        precision, recall, _ = precision_recall_curve(df["true_label"], df["predicted_proba"])
        ax.plot(recall, precision, label=m)
    base_rate = next(iter(preds.values()))["true_label"].mean()
    ax.axhline(base_rate, linestyle="--", color="gray", label=f"Base rate ({base_rate:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves (pooled out-of-fold predictions)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_cv_stability(cvs, out_path):
    models = list(cvs.keys())
    means = [cvs[m]["fold_f1"]["mean"] if cvs[m]["fold_f1"] else 0 for m in models]
    stds = [cvs[m]["fold_f1"]["std"] if cvs[m]["fold_f1"] else 0 for m in models]
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(models))
    ax.bar(x, means, yerr=stds, capsize=5, color="steelblue")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=20)
    ax.set_ylabel("F1-score (per-fold mean ± std)")
    ax.set_title("Cross-Validation Stability: Per-Fold F1 Mean ± Std, 5 Folds")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_confusion_matrices(preds, out_path):
    n = len(preds)
    ncols = 4
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    axes = np.array(axes).reshape(-1)
    for ax, (m, df) in zip(axes, preds.items()):
        cm = confusion_matrix(df["true_label"], df["predicted_label"], labels=[0, 1])
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks([0, 1], labels=["No collision", "Collision"], fontsize=8)
        ax.set_yticks([0, 1], labels=["No collision", "Collision"], fontsize=8)
        ax.set_title(m, fontsize=10)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                         color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=9)
    for ax in axes[len(preds):]:
        ax.axis("off")
    fig.suptitle("Confusion Matrices (pooled out-of-fold predictions)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_feature_importance_comparison(out_path, top_n=12):
    frames = {}
    for name in TREE_MODELS:
        path = FEATURE_IMPORTANCE_DIR / f"{name}_importance.csv"
        if path.exists():
            df = pd.read_csv(path).set_index("feature")["importance"]
            frames[name] = df / df.sum()  # normalize so models are comparable
    if not frames:
        print("No tree-model feature importance files found, skipping chart.")
        return
    combined = pd.DataFrame(frames).fillna(0)
    combined["avg"] = combined.mean(axis=1)
    top_features = combined.sort_values("avg", ascending=False).head(top_n).index
    plot_df = combined.loc[top_features, list(frames.keys())]

    fig, ax = plt.subplots(figsize=(9, 6))
    plot_df.iloc[::-1].plot(kind="barh", ax=ax)
    ax.set_xlabel("Normalized importance")
    ax.set_title(f"Top {top_n} Features - Tree Model Comparison")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def error_analysis(preds, table, out_dir):
    best_model = table.sort_values("ROC_AUC", ascending=False).iloc[0]["Model"]
    df = preds[best_model].copy()
    feats = pd.read_csv(FEATURES_CSV)
    df = df.merge(feats, on=["frame_idx", "time_s"], how="left", suffixes=("", "_feat"))

    df["case"] = np.select(
        [
            (df.true_label == 1) & (df.predicted_label == 1),
            (df.true_label == 0) & (df.predicted_label == 0),
            (df.true_label == 0) & (df.predicted_label == 1),
            (df.true_label == 1) & (df.predicted_label == 0),
        ],
        ["TP", "TN", "FP", "FN"],
        default="Unknown",
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    cols = ["frame_idx", "time_s", "true_label", "predicted_label", "predicted_proba", "case",
            "TIPL_present", "TIPR_present", "dist_TIPL_TIPR", "mask_iou_TIPL_TIPR", "label_near_miss"]
    cols = [c for c in cols if c in df.columns]
    df[cols].to_csv(out_dir / f"error_cases_{best_model}.csv", index=False)

    summary = df["case"].value_counts().to_dict()
    both_present_rate = df.groupby("case").apply(
        lambda g: ((g.TIPL_present == 1) & (g.TIPR_present == 1)).mean() if "TIPL_present" in g else None
    )
    print(f"\nError analysis for best model ({best_model} by ROC-AUC): {summary}")
    print(f"Both-tips-detected rate by case:\n{both_present_rate}")
    return best_model, summary


def main():
    FIG_COMPARISON_DIR.mkdir(parents=True, exist_ok=True)
    FIG_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    table, preds, cvs = load_all()
    print(f"Loaded results for {len(preds)} models: {list(preds.keys())}")

    plot_metric_bars(table, FIG_COMPARISON_DIR / "metric_bars.png")
    plot_roc_curves(preds, FIG_COMPARISON_DIR / "roc_curves.png")
    plot_pr_curves(preds, FIG_COMPARISON_DIR / "pr_curves.png")
    plot_cv_stability(cvs, FIG_RESULTS_DIR / "cross_validation_stability.png")
    plot_confusion_matrices(preds, FIG_RESULTS_DIR / "confusion_matrices.png")
    plot_feature_importance_comparison(FIG_RESULTS_DIR / "feature_importance_comparison.png")
    best_model, summary = error_analysis(preds, table, ERROR_DIR)

    print(f"\nAll comparison figures written to {FIG_COMPARISON_DIR} and {FIG_RESULTS_DIR}")
    print(f"Best model by ROC-AUC: {best_model}")


if __name__ == "__main__":
    main()
