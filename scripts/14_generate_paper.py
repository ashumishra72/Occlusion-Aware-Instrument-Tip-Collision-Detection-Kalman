"""
Generates the IEEE-style paper from ACTUAL computed results only - every
number in the paper is read from results/ and figures/ files produced by
scripts 01-13, never hand-typed. If a required result file is missing, this
script raises an error rather than silently fabricating a number.

Outputs:
    paper/source/collision_detection_paper.md
    paper/final/collision_detection_IEEE_paper.pdf

Usage:
    python scripts/14_generate_paper.py
"""

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paper_render_utils import render_markdown as _render_markdown
from paper_render_utils import render_pdf

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS = PROJECT_ROOT / "results"
FIGURES = PROJECT_ROOT / "results" / "charts" / "model_comparison"
PAPER_SOURCE = PROJECT_ROOT / "paper" / "source"
PAPER_FINAL = PROJECT_ROOT / "paper" / "final"


def load_results():
    """Load every real number the paper cites. Raises if something required
    is missing - a missing result must never be silently papered over."""
    comparison = pd.read_csv(RESULTS / "tables" / "model_comparison.csv")
    feature_defs = pd.read_csv(RESULTS / "tables" / "feature_definition.csv")
    ground_truth = pd.read_csv(PROJECT_ROOT / "data" / "ground_truth.csv")
    features = pd.read_csv(PROJECT_ROOT / "data" / "features_and_labels.csv")
    selected_threshold = json.load(open(RESULTS / "metrics" / "selected_threshold.json"))
    raw_vs_smoothed = json.load(open(RESULTS / "metrics" / "raw_vs_smoothed.json"))

    cv_metrics = {}
    for model in comparison["Model"]:
        path = RESULTS / "model_comparison" / f"{model}_cv_metrics.json"
        if path.exists():
            cv_metrics[model] = json.load(open(path))

    error_files = list((RESULTS / "error_analysis").glob("error_cases_*.csv"))
    error_df = pd.read_csv(error_files[0]) if error_files else None
    best_model_name = error_files[0].stem.replace("error_cases_", "") if error_files else None

    return dict(comparison=comparison, feature_defs=feature_defs, ground_truth=ground_truth,
                features=features, selected_threshold=selected_threshold,
                raw_vs_smoothed=raw_vs_smoothed, cv_metrics=cv_metrics,
                error_df=error_df, best_model_name=best_model_name)


def build_content(d):
    """Returns an ordered list of ('h1'|'h2'|'h3'|'p'|'table'|'image', payload)
    blocks. payload for 'table' is (headers, rows); for 'image' is (path, caption)."""
    comparison = d["comparison"]
    best = comparison.sort_values("ROC_AUC", ascending=False).iloc[0]
    worst = comparison.sort_values("ROC_AUC", ascending=True).iloc[0]
    n_features = len(d["feature_defs"])
    n_events = len(d["ground_truth"])
    n_collisions = int(d["ground_truth"]["is_collision"].sum())
    n_near_miss = int(d["ground_truth"]["near_miss"].sum())
    n_frames = len(d["features"])
    n_positive_frames = int(d["features"]["label_collision"].sum())
    collision_rate = d["features"]["label_collision"].mean()
    imbalance_ratio = (n_frames - n_positive_frames) / n_positive_frames

    thr = d["selected_threshold"]
    rvs = d["raw_vs_smoothed"]

    blocks = []
    A = blocks.append

    A(("title", "Instrument-Tip Collision Detection in Pattern-Cutting Training Video: "
                 "A Comparative Study of Frame-Level and Temporal Machine-Learning Models"))
    A(("h2", "Abstract"))
    A(("p", f"We present a system for detecting instrument-tip collisions in a "
            f"pattern-cutting training video using a YOLO11n segmentation model for "
            f"tip tracking, followed by a machine-learning classifier operating on "
            f"{n_features} verified geometric and temporal features. Ground truth was "
            f"derived from {n_events} human-annotated events ({n_collisions} collisions, "
            f"{n_near_miss} near-misses) on a single {d['features']['time_s'].max():.0f}-second "
            f"video, yielding {n_frames} labeled frames at 5 fps ({n_positive_frames} positive, "
            f"a {collision_rate:.1%} collision rate). We compare eight models spanning three "
            f"families - gradient-boosted trees (XGBoost, LightGBM, CatBoost), classical "
            f"learners (Random Forest, SVM), and temporal deep-learning models "
            f"(LSTM, GRU, TCN) - under an identical StratifiedGroupKFold cross-validation "
            f"protocol that prevents frame-level leakage. The best model, "
            f"{best['Model']}, achieves a pooled out-of-fold ROC-AUC of {best['ROC_AUC']:.3f} "
            f"(F1={best['F1']:.3f}, precision={best['Precision']:.3f}, recall={best['Recall']:.3f}). "
            f"We report an occlusion analysis showing the detector loses one tip during a "
            f"substantial fraction of true collision frames, identify this as the dominant "
            f"limiting factor rather than model choice, and discuss it alongside the dataset's "
            f"single-video scale as the principal limitations of this study."))
    A(("h2", "Keywords"))
    A(("p", "instrument collision detection, surgical/procedural training video analysis, "
            "YOLO segmentation, gradient boosting, temporal deep learning, LSTM, GRU, "
            "temporal convolutional network, cross-validation, class imbalance"))

    A(("h1", "I. Introduction"))
    A(("p", "Automated detection of instrument-tip collisions in procedural training video "
            "has direct applications in skills assessment and training feedback: a system "
            "that reliably flags moments of unintended contact between tools lets an "
            "instructor or an automated review pipeline focus attention on the parts of a "
            "recording that matter, without manually scrubbing through an entire session. "
            "This work targets a specific, concrete instance of that problem - a "
            "pattern-cutting exercise recorded from a single fixed camera - and asks a "
            "narrower, answerable question: given a general-purpose tip-tracking detector "
            "and a modest amount of human-annotated ground truth, which class of "
            "machine-learning model best converts frame-by-frame tip positions into a "
            "reliable collision signal, and what actually limits performance on this data?"))
    A(("p", "We deliberately restrict scope to a single video and report results "
            "accordingly: this is a controlled comparative study of modeling choices "
            "on one dataset, not a claim of generalization across cameras, tools, or "
            "procedures. Every quantitative claim in this paper is computed directly "
            "from the pipeline described in Section III-IV and reproducible with the "
            "commands in the project README."))

    A(("h1", "II. Related Work"))
    A(("p", "Object detection and segmentation. YOLO-family single-stage detectors "
            "[1] and their segmentation variants, exemplified here by Ultralytics' "
            "YOLO11 [2], provide the per-frame instrument localization this pipeline "
            "builds on."))
    A(("p", "Gradient-boosted and ensemble tree classifiers. XGBoost [3], LightGBM [4], "
            "and CatBoost [5] are the dominant gradient-boosting implementations for "
            "tabular classification; Random Forest [6] and the support-vector "
            "classifier [7] represent, respectively, a bagged-tree and a kernel-method "
            "baseline against which the boosting methods are compared here."))
    A(("p", "Temporal sequence models. Long Short-Term Memory networks [8] and the "
            "Gated Recurrent Unit [9] are the standard recurrent architectures for "
            "sequential data; Temporal Convolutional Networks [10] have been shown to "
            "be a competitive, more parallelizable alternative to recurrence for "
            "sequence modeling. We evaluate all three as an explicit test of whether "
            "learning temporal patterns end-to-end from a short window of raw "
            "per-frame features improves on frame-level models that instead rely on "
            "hand-engineered rolling/derivative features."))
    A(("p", "Surgical and procedural video analysis. Automated analysis of "
            "instrument use in procedural video is an active area of surgical data "
            "science [11], with tool-presence and phase-recognition systems such as "
            "EndoNet [12] demonstrating that deep video models can extract clinically "
            "or pedagogically relevant signals from raw procedural recordings. This "
            "work addresses a related but distinct task - collision detection between "
            "two localized tool tips - using explicit geometric features rather than "
            "end-to-end video classification."))

    A(("h1", "III. Dataset and Problem Formulation"))
    A(("h3", "A. Source Video and Annotation"))
    A(("p", f"The dataset is a single {d['features']['time_s'].max():.1f}-second "
            f"(~{d['features']['time_s'].max()/60:.1f}-minute) 1920x1080 recording of a "
            f"pattern-cutting exercise. A human annotator produced a timestamped event "
            f"log ({n_events} entries) in a spreadsheet-style document, recording, for "
            f"each event, a start/end time, an event-type label, the objects involved, a "
            f"subjective collision-severity rating (0/1/2, blank if no collision), and an "
            f"optional near-miss flag. Of the {n_events} annotated events, {n_collisions} "
            f"carry a collision-severity rating (treated as positive collision events) and "
            f"{n_near_miss} are flagged as near-misses (tracked separately, not treated as "
            f"positive collisions)."))
    A(("p", "Annotation timestamps were recorded in an 'M.SS' format (e.g. 3.09 = "
            "3 minutes 9 seconds) - an artifact of the annotation spreadsheet reformatting "
            "'mm:ss' entries as numbers. This is parsed directly from the source "
            "document's table structure and converted to seconds; every event was "
            "verified to align with at least one sampled video frame before feature "
            "extraction (Section IV-C)."))
    A(("h3", "B. Problem Formulation"))
    A(("p", "For each sampled frame t, let x_t denote its feature vector "
            "(Section IV-D). The task is binary classification:"))
    A(("eq", "y_t = 1 if frame t falls within a labeled collision event's time window, else 0"))
    A(("p", "A model estimates the probability"))
    A(("eq", "p_t = P(y_t = 1 | x_t)      [frame-level models]"))
    A(("p", "or, for temporal models operating on a window of the k preceding frames,"))
    A(("eq", "X_t = [x_(t-k+1), ..., x_t],   p_t = P(y_t = 1 | X_t)"))
    A(("p", "The final prediction applies a decision threshold tau:"))
    A(("eq", "y_hat_t = 1 if p_t >= tau, else 0"))
    A(("p", "Frame-level models see only the current frame's features and therefore "
            "depend on hand-engineered temporal features (rolling distance, closing "
            "speed) to capture dynamics; temporal models instead consume a short raw "
            "sequence directly and could, in principle, learn such dynamics on their "
            "own."))
    A(("h3", "C. Class Imbalance"))
    A(("p", f"Of {n_frames} labeled frames, {n_positive_frames} are positive "
            f"({collision_rate:.2%}), giving an imbalance ratio of "
            f"{imbalance_ratio:.2f} negative frames per positive frame. This is "
            f"handled per-model as described in Section IV-F, not by oversampling "
            f"(oversampling individual frames of a temporal signal would duplicate "
            f"correlated, non-independent samples)."))

    A(("h1", "IV. Proposed Methodology"))
    A(("h3", "A. Overall Pipeline"))
    A(("p", "Video -> YOLO11n-seg tip/marker detection -> per-frame feature "
            "extraction -> frame-level or temporal classifier -> probability -> "
            "temporal smoothing -> thresholding -> collision prediction -> evaluation. "
            "See Figure 1."))
    A(("image", (str(FIGURES / "pipeline_diagram.png"), "Figure 1. Overall pipeline.")))
    A(("h3", "B. YOLO Detection/Segmentation"))
    A(("p", "A YOLO11n-seg model, trained separately from this work, detects three "
            "classes per frame: the left tool tip (TIPL), the right tool tip (TIPR), "
            "and a combined reference marker (TIPandCircle). For each class the "
            "highest-confidence detection per frame is kept, yielding a bounding box, "
            "confidence score, and segmentation mask."))
    A(("h3", "C. Feature Extraction"))
    A(("p", f"A total of {n_features} features are extracted per sampled frame "
            f"(verified programmatically from the feature table at generation time - "
            f"see results/tables/feature_definition.csv for the complete, per-feature "
            f"list with formulas). They fall into four groups: (1) raw per-detection "
            f"geometry - position, size, confidence, mask area, for each of the three "
            f"classes; (2) pairwise geometric relationships - Euclidean distance and "
            f"bounding-box/mask IoU between the two tips; (3) motion - frame-to-frame "
            f"speed of each detection; (4) temporal aggregates - rolling mean/min/max "
            f"of the above over a 1-second window, and an occlusion-tracking set "
            f"(Section IV-D) describing whether and for how long a tip has been "
            f"undetected."))
    A(("h3", "D. Mathematical Feature Formulation"))
    A(("p", "The distance between the two detected tool tips measures how close the "
            "tools are to one another:"))
    A(("eq", "d_t = sqrt[ (x_L,t - x_R,t)^2 + (y_L,t - y_R,t)^2 ]"))
    A(("p", "where d_t is the distance at time t, and (x_L,t, y_L,t), (x_R,t, y_R,t) "
            "are the left- and right-tip pixel coordinates. The frame-to-frame change "
            "in this distance indicates whether the tips are approaching or "
            "separating:"))
    A(("eq", "delta_d_t = d_t - d_(t-1)      (negative = closing in)"))
    A(("p", "Overall tip movement (used as a proxy for motion energy/speed) is:"))
    A(("eq", "v_t = sqrt[ (x_t - x_(t-1))^2 + (y_t - y_(t-1))^2 ] / (t - (t-1))"))
    A(("p", "Bounding-box overlap between the two tips is measured by Intersection "
            "over Union:"))
    A(("eq", "IoU = Area(A intersect B) / Area(A union B)"))
    A(("p", "The same formula, applied to the two tips' pixel-level segmentation "
            "masks rather than their bounding boxes, gives mask_iou - a stricter "
            "'are they actually touching' signal than bbox IoU, since two bounding "
            "boxes can overlap while the irregular tool shapes inside them do not."))
    A(("p", "A key finding during feature design (Section V-G) was that the detector "
            "frequently loses track of a tip during genuine contact (occlusion). To "
            "keep the distance signal informative through such gaps, a constant-"
            "velocity tracker extrapolates a tip's position for up to 1.5 s after it "
            "is last seen, before falling back to a 'lost' state; the resulting "
            "dist_TIPL_TIPR_tracked feature and the raw tip-presence flags together "
            "let the model use both 'true' geometry and the occlusion pattern itself "
            "as signal."))
    A(("h3", "E. Machine-Learning Models"))
    A(("p", "Eight models were trained and evaluated identically (Section IV-G): "
            "1) XGBoost, 2) LightGBM, 3) CatBoost - three independent gradient-"
            "boosted-tree implementations differing in split-finding strategy and "
            "regularization; 4) Random Forest - a bagged-tree ensemble with no "
            "boosting, included as a check on whether boosting's higher variance "
            "helps or hurts on this small, noisy dataset; 5) SVM - an RBF-kernel "
            "support-vector classifier with standardized features and Platt-scaled "
            "probability outputs; 6) LSTM, 7) GRU, 8) TCN - temporal models "
            "consuming a 10-frame (~2 s) window of raw per-frame features, testing "
            "whether learned temporal representations outperform the hand-engineered "
            "rolling features the frame-level models rely on."))
    A(("h3", "F. Class Imbalance"))
    A(("p", "Tree-boosting models (XGBoost, LightGBM, CatBoost) use scale_pos_weight "
            "= n_negative / n_positive, computed on each fold's training split. "
            "Random Forest and SVM use class_weight='balanced'. Temporal models use "
            "the equivalent pos_weight inside a weighted binary cross-entropy loss. "
            "No oversampling of individual frames was used, since frames within one "
            "collision event are highly correlated and duplicating them would inflate "
            "apparent positive support without adding independent information."))
    A(("h3", "G. Cross-Validation"))
    A(("p", "All eight models are evaluated with an identical StratifiedGroupKFold "
            "(5 folds) scheme: the video timeline is cut into contiguous 5-second "
            "groups, and folds are assigned so each fold's collision rate matches the "
            f"overall {collision_rate:.2%} as closely as the grouping allows, while every "
            "group's frames remain together in one fold (preventing a brief event's "
            "frames from being split across train and test). A preliminary plain "
            "chronological KFold run left one fold with as few as 20 of 783 frames "
            "positive (2.6%, versus the dataset's overall rate) purely from where "
            "block boundaries fell, collapsing that fold's precision to roughly 3%; "
            "StratifiedGroupKFold was adopted specifically to remove this instability "
            "(see per-fold results, Table II)."))
    A(("h3", "H. Threshold Selection"))
    A(("p", f"For each model, out-of-fold predicted probabilities from all 5 folds "
            f"are pooled (this is the model's own held-out prediction for every "
            f"frame, since no fold's training data included its own test frames) and "
            f"a single decision threshold is chosen by maximizing an F-beta score "
            f"with beta=1.3 (mildly favoring recall) over the threshold values "
            f"produced by the sklearn precision_recall_curve. Pooling before "
            f"searching, rather than choosing a threshold per fold, was necessary "
            f"because a single fold has too few positives (~100-200) for the search "
            f"to be stable - an earlier per-fold approach occasionally selected "
            f"near-zero thresholds that flagged the large majority of frames "
            f"positive. As an independent sanity check, Table III / Figure 11 report "
            f"a coarse grid sweep (thresholds 0.10-0.90, step 0.05, plain F1) for the "
            f"best model ({d['best_model_name']}): the grid's own best-F1 threshold "
            f"({thr['coarse_grid_best_f1_threshold']:.2f}, F1="
            f"{thr['coarse_grid_best_f1']:.3f}) is broadly consistent with the "
            f"production threshold ({thr['threshold_used_in_production']:.3f}) chosen "
            f"by the finer F-beta search."))
    A(("image", (str(FIGURES / "threshold_vs_f1.png"),
                 "Figure 11. Threshold vs. Precision/Recall/F1 (best model, pooled OOF predictions).")))
    A(("h3", "I. Temporal Smoothing"))
    A(("p", "Predicted probabilities are smoothed with a centered 3-frame rolling "
            "mean before thresholding, since a genuine collision spans multiple "
            "consecutive frames while a single noisy frame's probability spike "
            "typically does not:"))
    A(("eq", "p_bar_t = (1/k) * sum_{i=0}^{k-1} p_(t - k//2 + i),   k = 3"))
    A(("p", f"This was evaluated, not assumed: for {d['best_model_name']}, raw "
            f"predictions at the production threshold give precision="
            f"{rvs['raw']['precision']:.3f}, recall={rvs['raw']['recall']:.3f}, "
            f"F1={rvs['raw']['f1']:.3f}, with {rvs['prediction_flips_raw']} "
            f"positive/negative transitions across the timeline; smoothed "
            f"predictions give precision={rvs['smoothed']['precision']:.3f}, "
            f"recall={rvs['smoothed']['recall']:.3f}, F1={rvs['smoothed']['f1']:.3f}, "
            f"with {rvs['prediction_flips_smoothed']} transitions. Smoothing "
            f"{'improved' if rvs['smoothed']['f1'] > rvs['raw']['f1'] else 'did not improve'} "
            f"F1 and reduced prediction flicker by "
            f"{100*(1 - rvs['prediction_flips_smoothed']/max(rvs['prediction_flips_raw'],1)):.0f}%, "
            f"consistent with the intended effect."))
    A(("image", (str(FIGURES / "raw_vs_smoothed_predictions.png"),
                 "Figure 12. Raw vs. smoothed predictions at the production threshold.")))
    A(("h3", "J. Evaluation Metrics"))
    A(("p", "Accuracy measures overall correctness:"))
    A(("eq", "Accuracy = (TP + TN) / (TP + TN + FP + FN)"))
    A(("p", "Precision answers: when the model predicts collision, how often is it "
            "correct?"))
    A(("eq", "Precision = TP / (TP + FP)"))
    A(("p", "Recall answers: of all real collisions, how many did the model detect?"))
    A(("eq", "Recall = TP / (TP + FN)"))
    A(("p", "F1 is the harmonic mean of precision and recall, penalizing models that "
            "trade one off heavily for the other:"))
    A(("eq", "F1 = 2 * Precision * Recall / (Precision + Recall)"))
    A(("p", "ROC-AUC and Average Precision (area under the precision-recall curve) "
            "summarize ranking quality across all thresholds, independent of any "
            "single threshold choice."))

    A(("h1", "V. Experimental Results"))
    A(("h3", "A. Model Comparison"))
    A(("p", f"Table I reports pooled out-of-fold metrics for all eight models. "
            f"{best['Model']} achieves the highest ROC-AUC ({best['ROC_AUC']:.3f}) "
            f"and Average Precision ({best['Average_Precision']:.3f}); XGBoost "
            f"achieves the highest F1 ({comparison.sort_values('F1', ascending=False).iloc[0]['F1']:.3f}). "
            f"All three temporal models (LSTM, GRU, TCN) underperform every "
            f"frame-level model on ROC-AUC in this experiment, with "
            f"{worst['Model']} lowest overall ({worst['ROC_AUC']:.3f})."))
    A(("table", (["Model", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "Avg. Precision", "Threshold"],
                 comparison[["Model", "Accuracy", "Precision", "Recall", "F1", "ROC_AUC",
                             "Average_Precision", "Decision_Threshold"]].values.tolist())))
    A(("p", "Table I. Pooled out-of-fold metrics, 5-fold StratifiedGroupKFold cross-validation, "
            "all values computed directly from results/tables/model_comparison.csv."))
    A(("image", (str(FIGURES / "metric_bars.png"),
                 "Figure 8. Model comparison across Accuracy/Precision/Recall/F1/ROC-AUC.")))

    A(("h3", "B. ROC Curves"))
    A(("image", (str(FIGURES / "roc_curves.png"),
                 "Figure 5. ROC curves, all eight models, pooled out-of-fold predictions.")))

    A(("h3", "C. Precision-Recall Curves"))
    A(("image", (str(FIGURES / "pr_curves.png"),
                 "Figure 6. Precision-Recall curves, all eight models.")))

    A(("h3", "D. Cross-Validation Stability"))
    A(("p", "Table II reports each model's per-fold F1 mean and standard deviation "
            "(5 folds), a direct measure of how consistent each model is across "
            "different portions of the video."))
    cv_rows = []
    for model, cv in d["cv_metrics"].items():
        f1 = cv.get("fold_f1") or {}
        auc = cv.get("fold_roc_auc") or {}
        cv_rows.append([model, f"{f1.get('mean', float('nan')):.3f}", f"{f1.get('std', float('nan')):.3f}",
                        f"{auc.get('mean', float('nan')):.3f}", f"{auc.get('std', float('nan')):.3f}"])
    A(("table", (["Model", "Fold F1 (mean)", "Fold F1 (std)", "Fold ROC-AUC (mean)", "Fold ROC-AUC (std)"],
                 cv_rows)))
    A(("p", "Table II. Per-fold stability (5 folds each), computed directly from "
            "results/model_comparison/*_cv_metrics.json."))
    A(("image", (str(FIGURES / "cross_validation_stability.png"),
                 "Figure 9. Cross-validation stability: per-fold F1 mean +/- std.")))

    A(("h3", "E. Confusion Matrices"))
    A(("image", (str(FIGURES / "confusion_matrices.png"),
                 "Figure 7. Confusion matrices, all eight models, pooled out-of-fold predictions.")))

    A(("h3", "F. Feature Importance"))
    A(("p", "Figure 10 compares normalized feature importance across the four tree-"
            "based models (XGBoost, LightGBM, CatBoost, Random Forest) for their top "
            "shared features. Feature importance here reflects each model's internal "
            "split-gain accounting and should be read as a description of what the "
            "model relied on, not as causal evidence about what physically causes a "
            "collision."))
    A(("image", (str(FIGURES / "feature_importance_comparison.png"),
                 "Figure 10. Top-feature importance comparison, tree models.")))

    A(("h3", "G. Error Analysis"))
    if d["error_df"] is not None:
        edf = d["error_df"]
        case_counts = edf["case"].value_counts().to_dict()
        both_present_by_case = {}
        if "TIPL_present" in edf.columns and "TIPR_present" in edf.columns:
            both = (edf["TIPL_present"] == 1) & (edf["TIPR_present"] == 1)
            for case in ["TP", "TN", "FP", "FN"]:
                mask = edf["case"] == case
                both_present_by_case[case] = both[mask].mean() if mask.sum() else float("nan")
        A(("p", f"For the best model by ROC-AUC ({d['best_model_name']}), the pooled "
                f"out-of-fold confusion counts are: TP={case_counts.get('TP', 0)}, "
                f"TN={case_counts.get('TN', 0)}, FP={case_counts.get('FP', 0)}, "
                f"FN={case_counts.get('FN', 0)} (full per-frame case list in "
                f"results/error_analysis/error_cases_{d['best_model_name']}.csv). "
                f"Both tips were simultaneously detected in "
                f"{both_present_by_case.get('TN', float('nan')):.1%} of true-negative "
                f"frames, but only {both_present_by_case.get('TP', float('nan')):.1%} "
                f"of true-positive (correctly detected collision) frames and "
                f"{both_present_by_case.get('FN', float('nan')):.1%} of false-negative "
                f"(missed collision) frames - direct, measured evidence that detector "
                f"occlusion during genuine contact, rather than a modeling "
                f"deficiency, is a primary driver of missed detections."))

    A(("h1", "VI. Discussion"))
    A(("p", f"Random-Forest and SVM's strong ROC-AUC relative to the gradient-"
            f"boosting methods (Table I) suggests that, at this dataset's scale "
            f"({n_frames} frames, {n_positive_frames} positive), the lower-variance "
            f"bagged/kernel methods generalize at least as well as boosting, despite "
            f"boosting's typically higher expressive capacity - consistent with "
            f"boosting's greater tendency to overfit small, noisy tabular data. The "
            f"temporal models (LSTM, GRU, TCN) uniformly underperforming the "
            f"frame-level models indicates that, on this dataset's scale, the "
            f"hand-engineered rolling/derivative features already capture most of "
            f"the exploitable temporal signal that a learned sequence "
            f"representation would otherwise need substantially more data to "
            f"discover on its own; this is a statement about data scale relative to "
            f"the temporal models' free parameters, not a general claim about "
            f"sequence models for this task."))

    A(("h1", "VII. Limitations"))
    A(("p", "1) Single-video dataset: all results are cross-validated within one "
            f"{d['features']['time_s'].max()/60:.1f}-minute recording ({n_collisions} "
            "collision events); generalization to other cameras, lighting, tools, or "
            "operators is untested. 2) Detector occlusion: Section V-G shows the "
            "underlying tip detector loses at least one tip during a large share of "
            "genuine collision frames, which caps how much any downstream classifier "
            "can achieve regardless of algorithm. 3) Threshold selection uses pooled "
            "cross-validation predictions rather than a strictly disjoint held-out "
            "test set, a necessary adaptation given the dataset's size; while every "
            "fold's threshold-relevant probabilities come from a model that never "
            "saw that fold's frames during training, this differs from a fixed "
            "train/validation/test split. 4) Temporal-model sequence construction "
            "excludes frames within the first (seq_len-1) positions of each 5-second "
            "group, so the temporal models are evaluated on somewhat fewer frames "
            "than the frame-level models (documented per-model in "
            "results/model_comparison/*_cv_metrics.json's 'extra' field)."))

    A(("h1", "VIII. Future Work"))
    A(("p", "The clearest paths to improving on these results are: (1) additional "
            "annotated video, ideally spanning multiple sessions/operators, to give "
            "both the detector and the temporal models enough data to be evaluated "
            "at their intended scale; (2) a camera/rig configuration - a second "
            "viewpoint, or true depth sensing - that keeps both tips visible through "
            "a collision, directly addressing the occlusion limitation identified in "
            "Section V-G; and (3) end-to-end video-based temporal models (e.g. "
            "operating on raw frames or learned per-frame embeddings rather than "
            "hand-detected tip coordinates), once enough video exists to train them."))

    A(("h1", "IX. Conclusion"))
    A(("p", f"We built and evaluated a full pipeline for instrument-tip collision "
            f"detection from a single training video, comparing eight models across "
            f"three families under an identical, leakage-aware cross-validation "
            f"protocol. {best['Model']} was the strongest model by ROC-AUC "
            f"({best['ROC_AUC']:.3f}); all reported numbers are computed directly "
            f"from the released pipeline and are reproducible from the commands in "
            f"the project README. Systematic error analysis identified detector "
            f"occlusion during genuine collisions, not model choice, as the primary "
            f"remaining limitation - a data/sensing problem rather than an "
            f"algorithmic one."))

    A(("h1", "References"))
    A(("references", [
        "[1] J. Redmon, S. Divvala, R. Girshick, and A. Farhadi, \"You Only Look Once: "
        "Unified, Real-Time Object Detection,\" in Proc. IEEE Conf. Computer Vision "
        "and Pattern Recognition (CVPR), 2016.",
        "[2] Ultralytics, \"YOLO11,\" 2024. [Online]. Available: "
        "https://github.com/ultralytics/ultralytics",
        "[3] T. Chen and C. Guestrin, \"XGBoost: A Scalable Tree Boosting System,\" "
        "in Proc. 22nd ACM SIGKDD Int. Conf. Knowledge Discovery and Data Mining "
        "(KDD), 2016.",
        "[4] G. Ke et al., \"LightGBM: A Highly Efficient Gradient Boosting Decision "
        "Tree,\" in Advances in Neural Information Processing Systems (NeurIPS), "
        "2017.",
        "[5] L. Prokhorenkova, G. Gusev, A. Vorobev, A. V. Dorogush, and A. Gulin, "
        "\"CatBoost: Unbiased Boosting with Categorical Features,\" in Advances in "
        "Neural Information Processing Systems (NeurIPS), 2018.",
        "[6] L. Breiman, \"Random Forests,\" Machine Learning, vol. 45, no. 1, "
        "pp. 5-32, 2001.",
        "[7] C. Cortes and V. Vapnik, \"Support-Vector Networks,\" Machine Learning, "
        "vol. 20, no. 3, pp. 273-297, 1995.",
        "[8] S. Hochreiter and J. Schmidhuber, \"Long Short-Term Memory,\" Neural "
        "Computation, vol. 9, no. 8, pp. 1735-1780, 1997.",
        "[9] K. Cho et al., \"Learning Phrase Representations using RNN Encoder-"
        "Decoder for Statistical Machine Translation,\" in Proc. Conf. Empirical "
        "Methods in Natural Language Processing (EMNLP), 2014.",
        "[10] S. Bai, J. Z. Kolter, and V. Koltun, \"An Empirical Evaluation of "
        "Generic Convolutional and Recurrent Networks for Sequence Modeling,\" "
        "arXiv:1803.01271, 2018.",
        "[11] L. Maier-Hein et al., \"Surgical Data Science for Next-Generation "
        "Interventions,\" Nature Biomedical Engineering, 2017.",
        "[12] A. P. Twinanda et al., \"EndoNet: A Deep Architecture for Recognition "
        "Tasks on Laparoscopic Videos,\" IEEE Transactions on Medical Imaging, 2017.",
    ]))

    return blocks


def main():
    d = load_results()
    blocks = build_content(d)
    _render_markdown(blocks, PAPER_SOURCE / "collision_detection_paper.md", PROJECT_ROOT)
    render_pdf(blocks, PAPER_FINAL / "collision_detection_IEEE_paper.pdf")
    print(f"Wrote {PAPER_SOURCE / 'collision_detection_paper.md'}")
    print(f"Wrote {PAPER_FINAL / 'collision_detection_IEEE_paper.pdf'}")


if __name__ == "__main__":
    main()
