# Complete step-by-step run guide

This guide covers all 35 current Python scripts, including the independently added Kalman paper, event tuning, chart collections and consolidated dashboard. Run commands from the project root. Use one activated Python environment throughout.

## 1. Download and install

Use GitHub **Code > Download ZIP**, extract the files and open a terminal in that folder. Alternatively, replace `YOUR_REPOSITORY_URL` with your actual repository URL:

```bash
git clone YOUR_REPOSITORY_URL collision_project
cd collision_project
```

This local workspace has no Git remote. The placeholder is not an actual URL. The code requires Python 3.10+ syntax support and compatible package wheels for your Python/OS combination. A GPU is optional; runtime depends on hardware.

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Linux/macOS:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Check your interpreter and dependencies:

```bash
python -c "import sys; print(sys.executable)"
python -c "import cv2, ultralytics, numpy, pandas, scipy, sklearn, xgboost, lightgbm, catboost, torch, matplotlib, reportlab, PIL; print('Imports OK')"
```

If PowerShell blocks activation, use `.\.venv\Scripts\python.exe` instead of `python` in subsequent commands. This avoids changing system policy. An existing local `venv/` may also work, but copied virtual environments are not portable between machines.

## 2. Prepare the inputs

Supply these files; a GitHub code download may omit them:

```text
data/video.mp4
models/video_tip_circle_yolo11n_seg_best.pt
docs/Pattern_Cutting_Annotation_1.docx
```

The detector must be trained for `TIPL`, `TIPR` and `TIPandCircle`. A generic YOLO checkpoint does not contain those trained classes. The current scripts do not train the detector.

Convert the supplied annotation document:

```bash
python scripts/01_convert_annotations.py
```

Output: `data/ground_truth.csv`. The converter expects a Word table headed `Start Time`, with seven columns: start, end, event type, involved objects, severity, near-miss text, notes. `M.SS` and `MM:SS` are supported: `3.09` means 189 seconds. A numeric severity, **including zero**, sets `is_collision=1`; empty severity means non-collision. The near-miss field is recognized by the words `near miss`.

Alternatively, supply the CSV directly and skip conversion. CSV times are seconds:

```csv
start_time_s,end_time_s,event_type,involved_objects,collision_severity,near_miss,is_collision,notes
10.0,10.5,contact,"TIPL, TIPR",1,0,1,Example contact
20.0,20.5,near miss,"TIPL, TIPR",,1,0,Example near miss
```

These rows only illustrate the format. Use real annotations matching your video. Require valid start/end times, start <= end, and binary flags. Evaluation needs positive and negative samples and enough groups for the fixed grouped splits. Default frame labels pad collision intervals by +/-0.5 seconds.

## 3. Extract features from the video

Optional smoke test with a separate output:

```bash
python scripts/02_extract_features.py --sample-fps 5 --max-frames 100 --output data/features_smoke.csv
```

This incomplete smoke file may have no collisions. Do not use it for full evaluation. Run full extraction:

```bash
python scripts/02_extract_features.py --sample-fps 5
python -c "import pandas as pd; d=pd.read_csv('data/features_and_labels.csv'); print(d.shape); print(d[['time_s','label_collision']].describe()); print(d.label_collision.value_counts())"
```

Output: `data/features_and_labels.csv`, with frame/time, 43 original features and three labels. The supplied recording produced 3,917 samples and 558 positives; another video will differ. Check that the final timestamp reaches the expected end. If extraction reports an early video-read failure, resolve the file/decoder issue and rerun before training.

For different paths:

```bash
python scripts/02_extract_features.py --video "path/to/video.mp4" --model "path/to/trained_detector.pt" --sample-fps 5
```

The extractor still reads `data/ground_truth.csv`. Most downstream scripts use fixed paths and have no `--video` flag. Use a separate project copy for a new recording, supply matching annotations and regenerate everything. There is no multi-video aggregation or arbitrary-video Kalman-classifier inference command yet.

## 4. Optional detector viewer

```bash
python scripts/05_visualize_detections.py --start-sec 48
```

Use a desktop with a display. Space pauses/resumes; `q` or Esc quits. This shows detector outputs and time, not classifier events. It does not save a rendered collision video. Avoid opening it during extraction when diagnosing video-read failures.

## 5. Current Kalman comparison

The historical eight-model suite is not required for this route.

```bash
python scripts/test_kalman_experiment.py
python scripts/15_kalman_experiment.py
python scripts/audit_kalman_outputs.py
```

Script 15 runs original features, Kalman features and Kalman plus two-second histories with RF and XGBoost. Outputs in `results/kalman_updated/` include trajectories, protocol, metrics and six prediction/event CSV pairs. The protocol records input hash, settings, groups and counts. The audit has a fixed 64-event reference-dataset check; adapt it for another recording.

## 6. Event-F1 threshold and merge-gap tuning

```bash
python scripts/17_tune_event_extraction.py
python scripts/test_event_tuning.py
```

This needs all script-15 outputs for the same cache. It refits inner models, selects threshold/gap on validation event F1, and applies those choices to unchanged outer test scores. Output: `results/event_f1_tuned/`, including metrics, protocol, validation grid, per-fold validation scores and six event/prediction pairs.

If input-hash or saved-threshold checks fail, verify inputs and environment, rerun script 15 completely, then tuning. Do not mix old scores with new features. Tests also contain reference-data assertions. Events include start/end frames, times, peak, confidence and duration. Confidence is an uncalibrated score. Event matching is one-to-one; a true-negative event count is not defined.

## 7. All current charts and reports

Use full filenames: numbers 17 and 18 each correspond to two different scripts. Numeric sorting is not a safe run order.

### Kalman charts and both Kalman reports

Requires script 15:

```bash
python scripts/18_kalman_figures.py
python scripts/17_generate_kalman_paper.py
python scripts/16_generate_updated_pdf.py
```

Outputs: earlier six figures in `results/charts/kalman/`, `paper/final/collision_detection_kalman_IEEE_paper.pdf`, its Markdown source, and `output/pdf/updated.pdf`.

### Simple event report and figures

Requires event tuning:

```bash
python scripts/18_event_figures.py
python scripts/19_event_report.py
```

Outputs under `output/pdf/`: roadmap/results PNG/SVG figures and `event_f1_explained.pdf`.

### End-to-end two-column paper

Requires both new experiments:

```bash
python scripts/20_ieee_figures.py
python scripts/21_build_ieee_paper.py
```

Outputs in `results/charts/updated/`: method/results PNG/SVG figures and `collision_detection_end_to_end_IEEE.pdf`. Source: `paper/source/collision_detection_end_to_end_IEEE.md`. This is an IEEE-style draft, with no invented authors or claim of acceptance.

### Comprehensive diagnostic charts

```bash
python scripts/22_comprehensive_charts.py
```

Output: `results/charts/comprehensive_evaluation/`. It contains 11 PNG/SVG pairs: event metrics/outcomes; full/excerpt timelines; frame metrics/confusion matrices; visibility-conditioned F1/ROC; overall ROC; and separate TIPL/TIPR bridging examples. A README, manifest and plotted data subsets are included.

Script 22 expects the reference recording's 3,917 rows and suitable short gaps. Adapt those assumptions for new videos. Missing detections are an occlusion proxy; predicted gap positions lack ground-truth positions. Report builders also contain reference-dataset narrative values; review them when inputs/settings change. Regenerated tables do not guarantee every sentence stays correct.

## 8. Optional historical eight-model study

This uses the original features and a different evaluation protocol.

First, the separately saved XGBoost classifier:

```bash
python scripts/03_train_classifier.py
python scripts/04_evaluate.py
```

Outputs: `models/xgb_collision_classifier.json`, `results/predictions.csv`, threshold/CV/importance files, `results/metrics.json` and charts directly in `results/charts/`. Script 04 evaluates OOF predictions; the final model is a separate fit. This does not create a complete video-inference application.

Run all historical comparison models:

```bash
python scripts/06_train_xgboost_cv.py
python scripts/07_train_lightgbm.py
python scripts/08_train_catboost.py
python scripts/09_train_random_forest.py
python scripts/10_train_svm.py
python scripts/11_train_temporal_models.py --arch lstm
python scripts/11_train_temporal_models.py --arch gru
python scripts/11_train_temporal_models.py --arch tcn
```

Then supporting analysis and the original paper:

```bash
python scripts/12_model_comparison_report.py
python scripts/13_threshold_and_smoothing_analysis.py
python scripts/build_feature_definition_table.py
python scripts/make_pipeline_figure.py
python scripts/14_generate_paper.py
```

Outputs: `results/model_comparison/`, `results/tables/`, `results/metrics/`, `results/error_analysis/`, `results/charts/model_comparison/`, and `paper/final/collision_detection_IEEE_paper.pdf`.

Historical LSTM/GRU/TCN use original features and exclude frames lacking enough within-group history; they are not Kalman sequence experiments. Five-second chunks can split events; centered smoothing uses future scores; pooled threshold selection reuses evaluation labels. Do not compare historical/current scores as a controlled improvement experiment.

## 9. All-evaluations dashboard

Requires the historical comparison table and both current experiments:

```bash
python scripts/23_all_evaluations_summary.py
```

Output: `results/charts/event_f1_tuned/all_evaluations_summary.png`. This separately added script shows all three phases, whose differing protocols are not a controlled performance progression.

## 10. Optional PDF visual checks

```bash
python -m pip install -r requirements-qa.txt
python scripts/verify_updated_pdf.py
python scripts/verify_event_pdf.py
python scripts/verify_ieee_paper.py
```

Run after generating the respective PDFs. Page images/contact sheets are written under `tmp/pdfs/`. Open them to inspect layout; bounds checks alone are not a full visual review. These scripts do not check the original or separate Kalman paper. Installed PyMuPDF works; the local `tmp/pdf_dependencies` fallback is not required in a fresh checkout.

## 11. Troubleshooting

| Problem | Check |
|---|---|
| Missing Python module | Use the environment interpreter and install requirements there. Pillow provides PIL. |
| Missing input files | Source downloads may omit video/weights/annotations; provide actual files. |
| Annotation table not found | Match the documented Word layout or supply a valid CSV. |
| No positives / insufficient groups | Check seconds/alignment and whether fixed grouped CV fits your data. |
| Script-17 assertion | Regenerate script 15 with matching inputs/environment. |
| Worker communication blocked | Use a permitted local terminal; managed environments may require execution approval. |
| Viewer cannot open GUI | Use a desktop display; check whether headless OpenCV overrides opencv-python. Viewer is optional. |
| Missing paper figures | Run the required figure builder first. |
| New recording fails an audit/chart | Adapt reference counts, example selection and report text. |
| Numbers differ | Check package/device/input/settings differences; fixed seeds do not guarantee equality. |

Capture installed versions after a successful run:

```bash
python -m pip freeze > requirements-local.txt
```

This records a local environment, not a tested cross-platform lock. `configs/experiment_config.yaml` documents historical settings; current scripts do not all read it. Current run settings are in `protocol.json`.

No GitHub upload is performed. Script paths/order were checked against local sources; a fresh installation and full retraining were not repeated for this documentation update.

## 12. Complete script inventory

All 35 current Python files are listed below. Helpers are imported, not run independently.

| Script | Purpose / prerequisite |
|---|---|
| `01_convert_annotations.py` | Word annotations to ground-truth CSV. |
| `02_extract_features.py` | Video + detector + annotations to feature cache. |
| `03_train_classifier.py` | Original XGBoost classifier and OOF results. |
| `04_evaluate.py` | Metrics and charts from script 03. |
| `05_visualize_detections.py` | Optional desktop detector viewer. |
| `06_train_xgboost_cv.py` | Historical XGBoost comparison. |
| `07_train_lightgbm.py` | Historical LightGBM comparison. |
| `08_train_catboost.py` | Historical CatBoost comparison. |
| `09_train_random_forest.py` | Historical RF comparison. |
| `10_train_svm.py` | Historical SVM comparison. |
| `11_train_temporal_models.py` | Historical temporal models; requires --arch lstm/gru/tcn. |
| `12_model_comparison_report.py` | Historical comparison charts and error analysis after 06-11. |
| `13_threshold_and_smoothing_analysis.py` | Historical threshold/smoothing analysis. |
| `14_generate_paper.py` | Original paper after historical results and supporting figures/tables. |
| `15_kalman_experiment.py` | Six original/Kalman/history comparisons. |
| `16_generate_updated_pdf.py` | Step-by-step PDF after script 15. |
| `17_generate_kalman_paper.py` | Separate Kalman paper after 15 and 18_kalman_figures. |
| `17_tune_event_extraction.py` | Event-F1 tuning after matching script-15 outputs. |
| `18_event_figures.py` | Roadmap and tuning figures after event tuning. |
| `18_kalman_figures.py` | Earlier six Kalman figures after script 15. |
| `19_event_report.py` | Simple event PDF after 18_event_figures. |
| `20_ieee_figures.py` | IEEE method/results figures from both new experiments. |
| `21_build_ieee_paper.py` | End-to-end two-column paper after 20_ieee_figures. |
| `22_comprehensive_charts.py` | Eleven diagnostic charts and plotted subsets. |
| `23_all_evaluations_summary.py` | Dashboard after historical table and both new experiments. |
| `audit_kalman_outputs.py` | Audit saved outputs; includes reference-dataset checks. |
| `build_feature_definition_table.py` | Original 43-feature definition CSV. |
| `make_pipeline_figure.py` | Original paper pipeline figure. |
| `model_cv_utils.py` | Imported helper for historical CV. |
| `paper_render_utils.py` | Imported renderer for original and separate Kalman papers. |
| `test_event_tuning.py` | Event behavior and saved-output checks; needs tuning results. |
| `test_kalman_experiment.py` | Tracker/event unit checks; no experiment run needed. |
| `verify_event_pdf.py` | Render/check event explanation PDF. |
| `verify_ieee_paper.py` | Render/check end-to-end IEEE-style PDF. |
| `verify_updated_pdf.py` | Render/check updated PDF. |
