# Comprehensive evaluation charts

All figures use saved held-out experimental outputs. PNG files are 400 dpi; SVG files are editable vectors.
O = original 43 features; K = Kalman (53 features); KT = Kalman plus history (85 features). RF = Random Forest; XGB = XGBoost.
The missing-tip subset is only an occlusion proxy. Event outcome matrices omit true negatives because a true-negative event universe is not defined.

- **event_metrics_comparison**: Precision, recall and F1 before/after event-focused extraction.
- **event_outcome_matrices**: Count matrices; not binary confusion matrices. Detected + missed = 64.
- **frame_metrics_comparison**: Frame precision, recall, F1 and ROC-AUC for both extraction strategies.
- **frame_confusion_matrices**: Binary confusion counts for event-focused thresholds. Rows=true, columns=prediction.
- **occlusion_conditioned_f1**: Visibility-conditioned F1 for both threshold strategies. Missingness is only a proxy.
- **roc_curves**: All-frame ROC from saved held-out scores.
- **occlusion_conditioned_roc_curves**: ROC from saved held-out scores, partitioned by detector visibility.
- **event_timeline_example**: Fixed 40-120 s excerpt; illustrative, not a representative-performance claim.
- **event_timeline_full**: Full recording, all six models and both extraction strategies.
- **trajectory_bridging_example_TIPL**: Longest within-group missing run with >=2 samples, all Kalman states predicted and age <=1.5 s; earliest tie.
- **trajectory_bridging_example_TIPR**: Longest within-group missing run with >=2 samples, all Kalman states predicted and age <=1.5 s; earliest tie.

Reproduce: `venv\Scripts\python.exe scripts\22_comprehensive_charts.py`
Data sources: `results/kalman_updated/`, `results/event_f1_tuned/`, `data/ground_truth.csv`, and `data/features_and_labels.csv`.
Trajectory example selection and exact times are in chart_manifest.json; raw plotted subsets are supplied as CSVs.