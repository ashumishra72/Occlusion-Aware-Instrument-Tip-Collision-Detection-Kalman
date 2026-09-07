"""
Two analyses on the best model's (by ROC-AUC) out-of-fold predictions:

1. Threshold sweep: precision/recall/F1 at thresholds 0.10-0.90 (step 0.05),
   confirming the F-beta-selected threshold is a reasonable choice rather
   than an arbitrary pick.
2. Raw vs. smoothed predictions: does the 3-frame rolling-mean smoothing
   (applied before thresholding in model_cv_utils.run_cv) actually help, or
   was it just theoretical? Computed honestly on real predictions - not
   assumed.

Outputs:
    results/metrics/threshold_analysis.csv
    results/metrics/selected_threshold.json
    results/metrics/raw_vs_smoothed.json
    figures/results/threshold_vs_f1.png
    figures/results/raw_vs_smoothed_predictions.png

Usage:
    python scripts/13_threshold_and_smoothing_analysis.py
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TABLES_DIR = PROJECT_ROOT / "results" / "tables"
COMPARISON_DIR = PROJECT_ROOT / "results" / "model_comparison"
METRICS_DIR = PROJECT_ROOT / "results" / "metrics"
FIG_DIR = PROJECT_ROOT / "results" / "charts" / "model_comparison"


def main():
    table = pd.read_csv(TABLES_DIR / "model_comparison.csv")
    best_model = table.sort_values("ROC_AUC", ascending=False).iloc[0]["Model"]
    df = pd.read_csv(COMPARISON_DIR / f"{best_model}_predictions.csv")
    print(f"Using best model by ROC-AUC: {best_model}")

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # --- 1. Threshold sweep (on smoothed probabilities, the ones actually used) ---
    thresholds = np.round(np.arange(0.10, 0.91, 0.05), 2)
    rows = []
    for thr in thresholds:
        pred = (df["predicted_proba"] >= thr).astype(int)
        rows.append({
            "threshold": thr,
            "precision": precision_score(df["true_label"], pred, zero_division=0),
            "recall": recall_score(df["true_label"], pred, zero_division=0),
            "f1": f1_score(df["true_label"], pred, zero_division=0),
        })
    sweep = pd.DataFrame(rows)
    sweep.to_csv(METRICS_DIR / "threshold_analysis.csv", index=False)

    best_row = sweep.loc[sweep["f1"].idxmax()]
    selected_threshold_from_row = pd.read_csv(TABLES_DIR / "model_comparison.csv")
    actual_used_threshold = float(
        selected_threshold_from_row[selected_threshold_from_row["Model"] == best_model]["Decision_Threshold"].iloc[0]
    )
    with open(METRICS_DIR / "selected_threshold.json", "w", encoding="utf-8") as f:
        json.dump({
            "model": best_model,
            "threshold_used_in_production": actual_used_threshold,
            "threshold_used_selection_method": "F-beta=1.3 maximization on pooled out-of-fold probabilities",
            "coarse_grid_best_f1_threshold": float(best_row["threshold"]),
            "coarse_grid_best_f1": float(best_row["f1"]),
            "note": "The two thresholds differ slightly because the production threshold uses an "
                    "exact search over the precision-recall curve's own thresholds with an F-beta=1.3 "
                    "objective (mildly favors recall), while this file's coarse 0.05-step grid uses "
                    "plain F1 for a simple, independent sanity check.",
        }, f, indent=2)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(sweep["threshold"], sweep["precision"], marker="o", label="Precision")
    ax.plot(sweep["threshold"], sweep["recall"], marker="o", label="Recall")
    ax.plot(sweep["threshold"], sweep["f1"], marker="o", label="F1")
    ax.axvline(actual_used_threshold, linestyle="--", color="gray",
               label=f"Production threshold ({actual_used_threshold:.3f})")
    ax.set_xlabel("Decision threshold")
    ax.set_ylabel("Score")
    ax.set_title(f"Threshold vs. Precision/Recall/F1 ({best_model}, pooled OOF predictions)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "threshold_vs_f1.png", dpi=150)
    plt.close(fig)

    # --- 2. Raw vs. smoothed predictions ---
    def metrics_at(proba_col, thr):
        pred = (df[proba_col] >= thr).astype(int)
        return {
            "precision": precision_score(df["true_label"], pred, zero_division=0),
            "recall": recall_score(df["true_label"], pred, zero_division=0),
            "f1": f1_score(df["true_label"], pred, zero_division=0),
        }

    raw_metrics = metrics_at("predicted_proba_raw", actual_used_threshold)
    smoothed_metrics = metrics_at("predicted_proba", actual_used_threshold)
    n_flips_raw = int((df["predicted_proba_raw"] >= actual_used_threshold).astype(int).diff().abs().sum())
    n_flips_smoothed = int((df["predicted_proba"] >= actual_used_threshold).astype(int).diff().abs().sum())

    comparison = {
        "model": best_model, "threshold": actual_used_threshold,
        "raw": raw_metrics, "smoothed": smoothed_metrics,
        "prediction_flips_raw": n_flips_raw, "prediction_flips_smoothed": n_flips_smoothed,
        "interpretation": (
            "smoothing helped" if smoothed_metrics["f1"] > raw_metrics["f1"] else
            "smoothing did not help" if smoothed_metrics["f1"] < raw_metrics["f1"] else
            "smoothing had no effect on F1"
        ) + f" (fewer flips = more temporally stable predictions: {n_flips_raw} raw vs {n_flips_smoothed} smoothed)",
    }
    with open(METRICS_DIR / "raw_vs_smoothed.json", "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2)

    fig, ax = plt.subplots(figsize=(7, 5))
    labels = ["Precision", "Recall", "F1"]
    x = np.arange(len(labels))
    width = 0.35
    ax.bar(x - width / 2, [raw_metrics[k.lower()] for k in labels], width, label="Raw predictions")
    ax.bar(x + width / 2, [smoothed_metrics[k.lower()] for k in labels], width, label="Smoothed predictions")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1)
    ax.set_title(f"Raw vs. Smoothed Predictions at threshold={actual_used_threshold:.3f} ({best_model})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "raw_vs_smoothed_predictions.png", dpi=150)
    plt.close(fig)

    print(json.dumps(comparison, indent=2))
    print(f"\nSaved threshold sweep -> {METRICS_DIR / 'threshold_analysis.csv'}")
    print(f"Saved selected threshold -> {METRICS_DIR / 'selected_threshold.json'}")
    print(f"Saved raw-vs-smoothed comparison -> {METRICS_DIR / 'raw_vs_smoothed.json'}")


if __name__ == "__main__":
    main()
