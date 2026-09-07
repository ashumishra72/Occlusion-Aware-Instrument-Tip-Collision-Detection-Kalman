"""
Generates figures for the Kalman-tracking paper from the real experiment
outputs in results/kalman_updated/ (metrics.json, protocol.json,
trajectories.csv, *_predictions.csv, *_events.csv). No synthetic/illustrative
data - every plotted value comes from an actual run of 15_kalman_experiment.py.

Outputs (results/charts/kalman/):
    frame_metrics_comparison.png   - Precision/Recall/F1/ROC-AUC, 6 variants
    event_metrics_comparison.png   - detected/missed/false events, 6 variants
    occlusion_conditioned_f1.png   - F1 on missing-tip vs both-visible frames
    trajectory_bridging_example.png - Kalman-filled vs raw-gap tip trajectory
    roc_curves.png                 - ROC overlay, 6 variants
    event_timeline_example.png     - predicted vs ground-truth events, a video segment

Usage:
    python scripts/18_kalman_figures.py
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc

PROJECT_ROOT = Path(__file__).resolve().parent.parent
R = PROJECT_ROOT / "results" / "kalman_updated"
OUT = PROJECT_ROOT / "results" / "charts" / "kalman"

VARIANTS = ["Original43_RF", "Original43_XGB", "Kalman_RF", "Kalman_XGB",
            "KalmanTemporal_RF", "KalmanTemporal_XGB"]
LABELS = {v: v.replace("Original43", "Orig43").replace("KalmanTemporal", "KF+Hist").replace("_", "\n") for v in VARIANTS}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    metrics = json.loads((R / "metrics.json").read_text())
    protocol = json.loads((R / "protocol.json").read_text())

    # --- 1. Frame-level metric comparison ---
    fig, ax = plt.subplots(figsize=(11, 5.5))
    metric_keys = ["precision", "recall", "f1", "roc_auc"]
    x = np.arange(len(VARIANTS))
    width = 0.2
    for i, mk in enumerate(metric_keys):
        vals = [metrics[v]["frame"][mk] for v in VARIANTS]
        ax.bar(x + i * width, vals, width, label=mk.replace("_", "-").upper())
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels([LABELS[v] for v in VARIANTS], fontsize=8)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("Frame-Level Metrics: Original 43 Features vs. Kalman-Tracking Variants\n"
                  "(revised protocol: 24 event-preserving groups, 5 outer folds, training-side threshold)")
    ax.legend(ncol=4, fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "frame_metrics_comparison.png", dpi=150)
    plt.close(fig)

    # --- 2. Event-level metric comparison ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    detected = [metrics[v]["event"]["detected"] for v in VARIANTS]
    missed = [metrics[v]["event"]["missed"] for v in VARIANTS]
    false_ev = [metrics[v]["event"]["false_events"] for v in VARIANTS]
    x = np.arange(len(VARIANTS))
    axes[0].bar(x, detected, label="Detected", color="#2ecc71")
    axes[0].bar(x, missed, bottom=detected, label="Missed", color="#e74c3c")
    axes[0].axhline(protocol["collision_events"], linestyle="--", color="gray",
                     label=f"Total annotated ({protocol['collision_events']})")
    axes[0].set_xticks(x); axes[0].set_xticklabels([LABELS[v] for v in VARIANTS], fontsize=8)
    axes[0].set_title("Event Detection Coverage")
    axes[0].set_ylabel("Number of ground-truth collision events")
    axes[0].legend(fontsize=8)

    axes[1].bar(x, false_ev, color="#e67e22")
    axes[1].set_xticks(x); axes[1].set_xticklabels([LABELS[v] for v in VARIANTS], fontsize=8)
    axes[1].set_title("False (Unmatched) Predicted Events")
    axes[1].set_ylabel("Count")
    fig.suptitle("Event-Level Evaluation (tolerance=0.5s, Hungarian one-to-one matching)")
    fig.tight_layout()
    fig.savefig(OUT / "event_metrics_comparison.png", dpi=150)
    plt.close(fig)

    # --- 3. Occlusion-conditioned F1 ---
    fig, ax = plt.subplots(figsize=(10, 5))
    missing_f1 = [metrics[v]["missing_tip"]["f1"] for v in VARIANTS]
    visible_f1 = [metrics[v]["both_visible"]["f1"] for v in VARIANTS]
    width = 0.35
    ax.bar(x - width / 2, missing_f1, width, label="Frames with a missing tip", color="#c0392b")
    ax.bar(x + width / 2, visible_f1, width, label="Frames with both tips visible", color="#2980b9")
    ax.set_xticks(x); ax.set_xticklabels([LABELS[v] for v in VARIANTS], fontsize=8)
    ax.set_ylabel("F1-score")
    ax.set_title("F1 Conditioned on Tip Visibility\n"
                  "(tests whether Kalman tracking specifically helps the occluded-frame case)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "occlusion_conditioned_f1.png", dpi=150)
    plt.close(fig)

    # --- 4. Trajectory bridging example ---
    traj = pd.read_csv(R / "trajectories.csv")
    feats = pd.read_csv(PROJECT_ROOT / "data" / "features_and_labels.csv")
    merged = traj.merge(feats[["frame_idx", "TIPL_present", "TIPL_x", "TIPL_y"]], on="frame_idx")
    # Find a gap the Kalman filter is actually designed to bridge (<= max_missing_s
    # = 1.5s ~ 7-8 frames at this sampling rate). Some gaps in this video run to
    # 100+ frames (the tip is genuinely out of frame for many seconds, not
    # occluded briefly) - the filter correctly gives up on those (state -> "lost"),
    # so picking the single longest gap would misleadingly illustrate a case the
    # method deliberately does NOT try to bridge. We pick the longest gap that is
    # still within the bridgeable range, which is the honest "this is what
    # bridging looks like when it works" example.
    missing = merged["TIPL_present"].eq(0).to_numpy()
    gaps, cur_start, cur_len = [], None, 0
    for i, m in enumerate(missing):
        if m:
            if cur_start is None:
                cur_start = i
            cur_len += 1
        else:
            if cur_len > 0:
                gaps.append((cur_start, cur_len))
            cur_start, cur_len = None, 0
    if cur_len > 0:
        gaps.append((cur_start, cur_len))
    bridgeable = [(s, l) for s, l in gaps if 2 <= l <= 7]
    best_start, best_len = max(bridgeable, key=lambda x: x[1])
    lo, hi = max(0, best_start - 15), min(len(merged), best_start + best_len + 15)
    window = merged.iloc[lo:hi]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(window["time_s"], window["TIPL_kf_x"], "-", color="#2980b9", label="Kalman-filtered TIPL x")
    observed = window[window["TIPL_present"] == 1]
    ax.scatter(observed["time_s"], observed["TIPL_x"], color="#27ae60", s=25, zorder=5, label="Raw YOLO detection (observed)")
    occluded = window[window["TIPL_present"] == 0]
    if len(occluded):
        ax.axvspan(occluded["time_s"].min(), occluded["time_s"].max(), alpha=0.15, color="red",
                   label=f"Occlusion gap ({best_len} frames, "
                         f"{occluded['time_s'].max() - occluded['time_s'].min():.2f}s)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("TIPL x-position (pixels)")
    ax.set_title("Kalman Filter Bridging a Real Detector Occlusion Gap (TIPL)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "trajectory_bridging_example.png", dpi=150)
    plt.close(fig)

    # --- 5. ROC curves ---
    fig, ax = plt.subplots(figsize=(7, 6))
    for v in VARIANTS:
        pred = pd.read_csv(R / f"{v}_predictions.csv")
        fpr, tpr, _ = roc_curve(pred["true_label"], pred["probability"])
        ax.plot(fpr, tpr, label=f"{v} (AUC={auc(fpr, tpr):.3f})")
    ax.plot([0, 1], [0, 1], "--", color="gray")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves - Kalman-Tracking Experiment (revised protocol)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "roc_curves.png", dpi=150)
    plt.close(fig)

    # --- 6. Event timeline example ---
    gt = pd.read_csv(PROJECT_ROOT / "data" / "ground_truth.csv")
    gt_collisions = gt[gt["is_collision"] == 1]
    events_orig = pd.read_csv(R / "Original43_RF_events.csv")
    events_new = pd.read_csv(R / "KalmanTemporal_RF_events.csv")
    t0, t1 = 40, 120  # a representative segment with several annotated events
    fig, ax = plt.subplots(figsize=(11, 3.5))
    for _, r in gt_collisions[(gt_collisions.end_time_s >= t0) & (gt_collisions.start_time_s <= t1)].iterrows():
        ax.axvspan(r["start_time_s"], max(r["end_time_s"], r["start_time_s"] + 0.15), ymin=0.68, ymax=0.98,
                   color="black", alpha=0.7)
    for _, r in events_orig[(events_orig.end_s >= t0) & (events_orig.start_s <= t1)].iterrows():
        ax.axvspan(r["start_s"], r["end_s"], ymin=0.35, ymax=0.65, color="#e74c3c", alpha=0.7)
    for _, r in events_new[(events_new.end_s >= t0) & (events_new.start_s <= t1)].iterrows():
        ax.axvspan(r["start_s"], r["end_s"], ymin=0.02, ymax=0.32, color="#2980b9", alpha=0.7)
    ax.set_yticks([0.83, 0.5, 0.17])
    ax.set_yticklabels(["Ground truth", "Original43+RF\npredicted events", "Kalman+Hist+RF\npredicted events"], fontsize=8)
    ax.set_xlim(t0, t1)
    ax.set_xlabel("Time (s)")
    ax.set_title(f"Predicted vs. Ground-Truth Collision Events, t={t0}-{t1}s")
    fig.tight_layout()
    fig.savefig(OUT / "event_timeline_example.png", dpi=150)
    plt.close(fig)

    print(f"Wrote 6 figures to {OUT}")


if __name__ == "__main__":
    main()
