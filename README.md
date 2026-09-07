# Instrument-Tip Collision Detection — Project Documentation

**Prepared by: Ashutosh Kumar**

This is a detailed, standalone copy of this project's documentation,
covering the research question, results, structure, setup, and every
pipeline stage in detail. It reflects the project as it stands in the main
repository (`collision_project/`) — this folder itself only holds
generated charts and this write-up; **run all commands from the project
root**, not from inside `results/charts/Ashutosh Kumar/`.

For the condensed, always-current version, see the repository's own
`README.md` and `docs/RUN_GUIDE.md`.

---

## 1. Research Question

Given a general-purpose tip-tracking detector (YOLO11n-seg) and a modest
amount of human-annotated ground truth from a single pattern-cutting
video, which class of model — gradient-boosted trees, classical learners,
or temporal deep learning — best converts frame-by-frame tool-tip
positions into a reliable instrument-collision signal, and what actually
limits performance on this data?

A follow-up question, addressed in the project's second phase: if the
detector's own occlusion failures (losing sight of a tip during real
contact) are the main limiting factor, does adding explicit Kalman-filter
motion tracking through those gaps measurably help — at the frame level,
and at the level of whole collision *events*?

## 2. Project Overview / Pipeline

```text
Video + trained detector + human annotations
 -> detections and 43 original geometric/temporal features
 -> original classifiers  OR  Kalman trajectories + motion histories
 -> collision probability scores
 -> threshold selection + event merging
 -> frame-level and event-level evaluation
 -> charts and IEEE-style research papers
```

The project has three phases, each with its own evaluation protocol:

1. **Phase 1 — Original 8-model comparison.** 43 hand-engineered features,
   8 models (XGBoost, LightGBM, CatBoost, Random Forest, SVM, LSTM, GRU,
   TCN), 5-second CV groups, pooled out-of-fold threshold selection.
2. **Phase 2 — Kalman tracking.** Adds an independent Kalman filter per
   tool tip, tested as three feature variants (Original43 / Kalman /
   Kalman+2s-history) × 2 models (RF, XGBoost), under a stricter,
   revised protocol (24 event-preserving groups, boundary-purged splits,
   training-side-only threshold selection).
3. **Phase 3 — Event-F1 tuning.** Reuses Phase 2's classifier scores, but
   re-selects the decision threshold and event-merge gap specifically to
   maximize *event-level* F1 instead of frame-level F1.

**Important:** Phase 1's numbers are **not directly comparable** to
Phases 2-3 — the evaluation protocol changed (smaller groups, different
threshold selection, no boundary guard). Differences between historical
and current scores must not be read as a controlled before/after
comparison.

## 3. Results

### Measured results (reference dataset: 3,917 sampled frames, 558
positive, 64 annotated collision events)

| Comparison | Original-feature RF | Kalman + history RF |
|---|---:|---:|
| Frame F1 (frame-focused extraction) | 0.319 | 0.348 |
| Missing-tip F1 (frame-focused extraction) | 0.448 | 0.504 |
| Event F1 (frame-focused extraction) | 0.385 | 0.342 |
| Event F1 (event-focused extraction) | 0.392 | 0.354 |

Kalman tracking with 2-second history improves frame-level F1 and,
specifically, F1 on frames where a tip went undetected (direct evidence
the tracking helps the occlusion case it targets). It does **not**
straightforwardly improve event-level F1 — this is a genuine, honestly
reported trade-off, not a one-sided win.

### Historical Phase 1 headline numbers (different protocol — context only)

| Model | ROC-AUC | F1 |
|---|---:|---:|
| Random Forest | 0.857 | 0.524 |
| XGBoost | 0.812 | 0.528 |
| SVM | 0.856 (highest Average Precision) | 0.522 |

## 4. Project Structure

```text
collision_project/
├── data/                 video.mp4, ground_truth.csv, features_and_labels.csv
├── docs/                 RUN_GUIDE.md (full script-by-script guide), annotation source .docx
├── models/               trained YOLO11n-seg detector, saved XGBoost classifier
├── configs/              experiment_config.yaml (historical settings)
├── scripts/              35 Python scripts — extraction, training, tuning, figures, papers
├── results/
│   ├── kalman_updated/      Phase 2 trajectories, metrics, predictions, events
│   ├── event_f1_tuned/      Phase 3 tuned metrics, validation grid, events
│   ├── model_comparison/    Phase 1 per-model predictions and CV metrics
│   ├── tables/               model_comparison.csv, feature_definition.csv
│   ├── metrics/, error_analysis/, feature_importance/
│   └── charts/
│       ├── model_comparison/     Phase 1 charts
│       ├── kalman/                Phase 2 charts
│       ├── updated/               end-to-end IEEE paper + its figures
│       ├── comprehensive_evaluation/  11 diagnostic chart pairs
│       ├── event_f1_tuned/        all-evaluations dashboard
│       └── Ashutosh Kumar/        (this folder) — project summary + this documentation
├── paper/final/          all generated PDF papers
├── output/pdf/           the simple event-tuning report and step-by-step Kalman report
└── SIMPLE_GUIDE.md       plain-language walkthrough of the whole project
```

## 5. Installation

Download the project (GitHub **Code > Download ZIP**, or `git clone` your
repository URL), then open a terminal in the project root.

**Windows PowerShell:**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

**Linux/macOS:**
```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell blocks activation, call `.\.venv\Scripts\python.exe` directly
instead of `python`, rather than changing system execution policy.

## 6. Install Project Dependencies

Everything needed is listed in `requirements.txt`:
`opencv-python`, `ultralytics`, `numpy`, `pandas`, `scipy`, `scikit-learn`,
`xgboost`, `lightgbm`, `catboost`, `torch`, `matplotlib`, `reportlab`,
`Pillow`, `pyyaml`. Optional PDF visual-check tooling is in
`requirements-qa.txt` (installs PyMuPDF).

```bash
python -m pip install -r requirements.txt
```

## 7. Verify the Installation

Run a quick sanity check to confirm the interpreter and every dependency
import correctly:

```bash
python -c "import sys; print(sys.executable)"
python -c "import cv2, ultralytics, numpy, pandas, scipy, sklearn, xgboost, lightgbm, catboost, torch, matplotlib, reportlab, PIL; print('Imports OK')"
```

`Imports OK` with no errors means the environment is ready.

## 8. Environment Setup Summary

| Item | Detail |
|---|---|
| Python | 3.10+ syntax required |
| Detector | YOLO11n-seg, trained for `TIPL`, `TIPR`, `TIPandCircle` (generic checkpoints will not work) |
| GPU | Optional; CPU works, runtime varies by hardware |
| Required inputs | `data/video.mp4`, `models/video_tip_circle_yolo11n_seg_best.pt`, `docs/Pattern_Cutting_Annotation_1.docx` |

## 9. Annotation Processing

```bash
python scripts/01_convert_annotations.py
```

Reads the Word document's annotation table (headed `Start Time`, 7
columns: start, end, event type, involved objects, severity, near-miss,
notes). Supports `M.SS` and `MM:SS` time formats (`3.09` = 189 seconds).
Any numeric severity (including 0) sets `is_collision=1`; empty severity
means non-collision. Output: `data/ground_truth.csv`. A matching CSV can
be supplied directly instead, skipping this step.

## 10. Feature Extraction

```bash
python scripts/02_extract_features.py --sample-fps 5
```

Runs the trained detector over every sampled frame, computes 43 original
geometric/temporal features (tip positions, inter-tip distance, IoU,
speed, rolling statistics, etc.), and labels each frame from
`ground_truth.csv` (±0.5s padding around each event). Output:
`data/features_and_labels.csv` — 3,917 rows / 558 positive for the
reference video. A `--max-frames` smoke-test mode exists for quick checks
before a full run.

## 11. Model Training

**Phase 1 (historical, original features):**
```bash
python scripts/03_train_classifier.py      # production XGBoost model
python scripts/06_train_xgboost_cv.py
python scripts/07_train_lightgbm.py
python scripts/08_train_catboost.py
python scripts/09_train_random_forest.py
python scripts/10_train_svm.py
python scripts/11_train_temporal_models.py --arch lstm
python scripts/11_train_temporal_models.py --arch gru
python scripts/11_train_temporal_models.py --arch tcn
```

**Phase 2 (current, Kalman tracking):**
```bash
python scripts/15_kalman_experiment.py
```
Trains RF and XGBoost on each of 3 feature variants (Original43 / Kalman /
Kalman+2s-history) — 6 combinations total.

## 12. Cross-Validation

- **Phase 1**: `StratifiedGroupKFold`, 5 folds, 5-second time groups.
- **Phase 2-3 (current, stricter)**: 5 outer `StratifiedGroupKFold` folds
  over 24 event-preserving time groups (group boundaries are nudged away
  from annotated collision windows). Training data within a fixed
  boundary guard of any held-out group is purged, closing a subtle
  leakage risk in globally-computed rolling features. Each outer fold
  further splits its training data (inner 3-fold grouped CV) to select
  extraction settings without ever touching that fold's own test data.

## 13. Threshold Optimization

- **Phase 1**: one threshold picked from *pooled* out-of-fold
  probabilities/labels across all folds (later found to be slightly
  optimistic, since it reuses evaluation labels for selection).
- **Phase 2**: per-fold threshold selected purely from that fold's
  *inner-validation* split, optimizing frame-level F1 — never touching
  the outer test fold.
- **Phase 3**: same nested, training-side-only selection, but the
  threshold (and the event-merge gap) is chosen to maximize
  **event-level** F1 instead of frame-level F1 (`scripts/17_tune_event_extraction.py`).

## 14. Temporal Smoothing

Predicted probabilities are smoothed with a short rolling window before
thresholding, since a real collision spans multiple consecutive frames
rather than a single noisy spike. This was verified to help, not just
assumed — see `results/charts/*/raw_vs_smoothed*` and the comprehensive
evaluation charts for the measured before/after comparison.

## 15. Evaluation

Two levels are reported throughout:

- **Frame-level**: Accuracy, Precision, Recall, F1, ROC-AUC, Average
  Precision, plus a breakdown of F1 on missing-tip vs. both-tips-visible
  frames (isolates whether Kalman tracking helps the occlusion case).
- **Event-level**: predicted frame-runs are merged into discrete events
  and matched one-to-one against annotated events via the Hungarian
  algorithm (within a time tolerance); reports detected / missed / false
  events, event Precision/Recall/F1, and timing error (start/end/peak MAE).

Run `scripts/12_model_comparison_report.py` (Phase 1) and
`scripts/22_comprehensive_charts.py` / `scripts/23_all_evaluations_summary.py`
(Phases 2-3) to regenerate all evaluation charts and tables.

## 16. Reproducibility

- Use one activated Python environment for every stage.
- `requirements.txt` is not a strict version lock; fixed random seeds
  (`random_state=42` / `torch.manual_seed(42)`) do not guarantee bit-identical
  results across different package versions, hardware, or OS.
- `configs/experiment_config.yaml` documents Phase 1's settings;
  current experiments (Phases 2-3) write their own `protocol.json` instead.
- Full run order, prerequisites, and every script's purpose:
  `docs/RUN_GUIDE.md` (35 scripts documented).
- No Git remote is configured in this workspace; these instructions do
  not perform any GitHub upload.

## 17. Notes

- All results come from **one** ~13-minute video (768.8s, 3,917 sampled
  frames at 5 fps). Generalization to other recordings, cameras, or
  operators is untested.
- The detector loses track of a tip during a large share of genuine
  collision frames — this occlusion, not model choice, is the dominant
  factor limiting further accuracy.
- "Missing tip" is an occlusion *proxy*, not proof of physical occlusion
  for every such frame.
- Historical (Phase 1) and current (Phase 2-3) scores use different,
  non-comparable protocols — never difference them directly.

## 18. Limitations

1. Single-video dataset — no cross-video generalization evidence.
2. The Kalman filter deliberately does not bridge long (multi-second to
   multi-tens-of-seconds) absences where a tip is genuinely out of frame,
   only short occlusion gaps (≤1.5s) — by design, not a bug.
3. Deep temporal models (LSTM/GRU/TCN) were run in Phase 1 only; they have
   not been rerun on the Kalman feature sets.
4. A coupled two-tip Kalman/IMM filter and appearance-based
   re-identification (DeepSORT/ByteTrack-style) remain future work.
5. More annotated video remains the single highest-leverage improvement
   available — not something further tuning can substitute for.

---

*This document and the accompanying figures in this folder were prepared
by Ashutosh Kumar as a project summary. See the repository root for the
live, always-current `README.md` and `docs/RUN_GUIDE.md`.*
