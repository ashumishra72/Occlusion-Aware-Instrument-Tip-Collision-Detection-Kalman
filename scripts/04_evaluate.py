"""
Evaluate the trained collision classifier: compute metrics on the held-out
test split (results/predictions.csv) and render diagnostic charts.

Outputs:
    results/metrics.json
    results/charts/confusion_matrix.png
    results/charts/roc_curve.png
    results/charts/precision_recall_curve.png
    results/charts/feature_importance.png

Usage:
    python scripts/04_evaluate.py
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    accuracy_score, average_precision_score, confusion_matrix, f1_score,
    precision_recall_curve, precision_score, recall_score, roc_auc_score,
    roc_curve,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PREDICTIONS_CSV = PROJECT_ROOT / "results" / "predictions.csv"
FEATURE_IMPORTANCE_CSV = PROJECT_ROOT / "results" / "feature_importance.csv"
METRICS_OUT = PROJECT_ROOT / "results" / "metrics.json"
CHARTS_DIR = PROJECT_ROOT / "results" / "charts"


def plot_confusion_matrix(y_true, y_pred, out_path):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], labels=["No collision", "Collision"])
    ax.set_yticks([0, 1], labels=["No collision", "Collision"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_roc_curve(y_true, y_proba, auc, out_path):
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_pr_curve(y_true, y_proba, ap, out_path):
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(recall, precision, label=f"AP = {ap:.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_feature_importance(importance_df, out_path, top_n=15):
    top = importance_df.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.barh(top["feature"], top["importance"])
    ax.set_xlabel("Importance")
    ax.set_title(f"Top {top_n} Feature Importances")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    df = pd.read_csv(PREDICTIONS_CSV)
    y_true, y_pred, y_proba = df["true_label"], df["predicted_label"], df["predicted_proba"]

    metrics = {
        "n_test_samples": int(len(df)),
        "n_positive": int(y_true.sum()),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    if y_true.nunique() > 1:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba))
        metrics["average_precision"] = float(average_precision_score(y_true, y_proba))
    else:
        metrics["roc_auc"] = None
        metrics["average_precision"] = None

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    metrics["confusion_matrix"] = {
        "true_negative": int(cm[0, 0]), "false_positive": int(cm[0, 1]),
        "false_negative": int(cm[1, 0]), "true_positive": int(cm[1, 1]),
    }

    METRICS_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(METRICS_OUT, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    plot_confusion_matrix(y_true, y_pred, CHARTS_DIR / "confusion_matrix.png")
    if y_true.nunique() > 1:
        plot_roc_curve(y_true, y_proba, metrics["roc_auc"], CHARTS_DIR / "roc_curve.png")
        plot_pr_curve(y_true, y_proba, metrics["average_precision"], CHARTS_DIR / "precision_recall_curve.png")

    if FEATURE_IMPORTANCE_CSV.exists():
        importance_df = pd.read_csv(FEATURE_IMPORTANCE_CSV)
        plot_feature_importance(importance_df, CHARTS_DIR / "feature_importance.png")

    print(json.dumps(metrics, indent=2))
    print(f"\nSaved metrics -> {METRICS_OUT}")
    print(f"Saved charts -> {CHARTS_DIR}")


if __name__ == "__main__":
    main()
