# Occlusion-Aware Instrument-Tip Collision Detection: Kalman Trajectories and Event-Focused Evaluation

<b>Abstract-</b>Automatic collision detection in procedural training video is difficult when one instrument tip becomes undetected near contact. We evaluate an occlusion-aware pipeline that combines cached YOLO11n-seg detections, independent Kalman filters, trajectory features, tree classifiers, and event extraction. A single 768.8 s recording provides 3,917 sampled frames and 64 collision annotation rows. Six feature/classifier combinations are evaluated with five event-preserving grouped outer folds and training-side threshold selection. Kalman trajectories with 2 s histories increase Random Forest frame F1 from 0.319 to 0.348 and missing-tip F1 from 0.448 to 0.504, but reduce event recall. Optimizing threshold and merge gap for validation event F1 raises held-out event F1 for this variant from 0.342 to 0.354; the original-feature Random Forest remains higher at 0.392. These pilot results show that frame-level improvement does not establish better event detection and motivate separate optimization of temporal event decisions.

<b>Index Terms-</b>Collision detection, instrument tracking, Kalman filter, occlusion, trajectory features, event evaluation.

I. INTRODUCTION

Instrument-tip collision events can help instructors review pattern-cutting practice. The desired output is an interval that an instructor can inspect, rather than a large set of independent positive frames. This distinction matters because a classifier can improve frame accuracy while producing broad bursts that merge several annotated contacts.

The existing project detects the left tip (TIPL), right tip (TIPR), and a reference marker (TIPandCircle) using YOLO11n-seg. A downstream classifier operates on 43 geometric, confidence, motion, overlap, and tracking features. Those features already include constant-velocity extrapolation and temporal summaries; the baseline is not purely frame-independent.

Our research question is whether explicit uncertainty-aware trajectories improve collision detection, especially when one tip is not detected. We additionally ask whether selecting event-extraction settings for event F1 improves the final collision event output without changing classifier scores.

The study makes three contributions: a reproducible Kalman trajectory implementation; a six-way comparison under an event-preserving grouped protocol; and a controlled comparison of frame-F1 versus event-F1 selection of extraction settings. The contributions are experimental and methodological. We do not claim a validated contact sensor or generalization to unseen trainees.

II. BACKGROUND AND PRIOR PROJECT RESULTS

YOLO segmentation supplies instance boxes, masks, and confidence values [1]. Kalman filtering recursively combines a motion prediction with noisy observations [2]. Random Forest (RF) [3] and XGBoost (XGB) [4] provide practical tabular baselines for the resulting geometric and temporal features.

The companion project compared XGBoost, LightGBM, CatBoost, RF, SVM, LSTM, GRU, and TCN. Its historical RF ROC-AUC was 0.857 and XGBoost F1 was 0.528. Those experiments used smaller temporal groups, centered smoothing, and a threshold selected using pooled out-of-fold labels. They are historical context, not a directly comparable estimate of improvement under the new protocol.

LSTM, GRU, and TCN are established sequence-model families [5]-[7]. They were not rerun on Kalman features in this work: the available dataset remains one video. A new neural-model study should use additional independent recordings and grouped sequence construction before conclusions about generalization are drawn.

III. DATASET AND TARGET DEFINITION

The input is one 1920 x 1080 pattern-cutting recording lasting approximately 768.8 s. The detector cache contains 3,917 samples with median timestamp spacing 0.196 s (approximately 5.10 Hz). Actual timestamps determine velocity and history windows. Human annotations contain 89 rows, including 64 collision rows and seven near-miss rows.

For frame classification, a sample is positive when it lies within any collision interval padded by 0.5 s on each side. This produces 558 positive frames (14.25%). Near misses remain negative unless a collision interval also covers the time. At event level, the 64 original collision rows are retained separately, including point, touching, and overlapping annotations.

![Fig. 1. End-to-end workflow. The original 43-feature branch bypasses Kalman tracking; the K and KT branches use the same cached detector observations. Inner validation selects extraction settings. Outer test groups supply all reported comparison scores.](../../results/charts/updated/fig1_ieee_end_to_end_pipeline.png)

IV. OCCLUSION-AWARE METHODOLOGY

A. Detector measurements and tracking

For each sampled frame, the existing extractor retains the highest-confidence detection per class. A box center serves as the tip-position measurement z = [x, y]<super>T</super>. Box dimensions, confidence, mask area, and precomputed segmentation overlap are reused from the cache. Thus all feature variants share the same detector evidence (Fig. 1).

Each tip has an independent state s = [x, y, v<sub>x</sub>, v<sub>y</sub>]<super>T</super>. With interval dt, F advances x and y by dt times velocity. H selects position. The prediction and measurement correction are

s<sup>-</sup><sub>t</sub> = F s<sub>t-1</sub>, &nbsp; P<sup>-</sup><sub>t</sub> = F P<sub>t-1</sub> F<super>T</super> + Q. &nbsp; (1)

K = P<sup>-</sup> H<super>T</super>(H P<sup>-</sup> H<super>T</super> + R)<super>-1</super>,<br/>s<sub>t</sub> = s<sup>-</sup><sub>t</sub> + K(z<sub>t</sub> - Hs<sup>-</sup><sub>t</sub>). &nbsp; (2)

The implementation solves a linear system for K and uses the Joseph covariance update: P = (I-KH)P<sup>-</sup>(I-KH)<super>T</super> + KRK<super>T</super>. Process noise is Q = 100<super>2</super> GG<super>T</super>, where the rows of G are [dt<super>2</super>/2, 0], [0, dt<super>2</super>/2], [dt, 0], and [0, dt].

Measurement noise is R = [8<super>2</super>/max(c, 0.1)]I for detector confidence c. Initial velocity variance is 10,000 pixels squared per second squared. These fixed noise assumptions are not calibrated detector-error estimates.

B. Occlusion handling and feature construction

When a tip is missing, the filter predicts for at most 1.5 s. A longer gap makes the state unavailable until a detection reinitializes it. Each output records source state (updated, predicted, unavailable), consecutive missing sample count, seconds since detection, and the trace of the position covariance matrix.

d<sub>t</sub> = ||p<sub>L,t</sub> - p<sub>R,t</sub>||<sub>2</sub>, &nbsp; delta d<sub>t</sub> = d<sub>t</sub> - d<sub>t-1</sub>. &nbsp; (3)

Distance rate is delta d/dt; negative values indicate approach. Each tip contributes filtered position, velocity, speed, and speed derivative. Pair features include distance and relative speed. Invalid values remain missing in trajectory exports and use a fixed -1 sentinel for the classifiers, accompanied by availability indicators.

TABLE I. FEATURE VARIANTS

| ID | Features | Definition |
|---|---|---|
| O | 43 | Original cached geometric/temporal features |
| K | 53 | Detector geometry + current Kalman/motion states |
| KT | 85 | K + trailing 2 s summary features |

KT adds mean, minimum, maximum, and standard deviation for eight signals: distance, distance rate, both tip speeds, both position variances, and both tracking-state codes. Windows and Kalman state reset at evaluation-group boundaries. This is causal temporal feature engineering, not a learned neural sequence model.

Segmentation masks are not extrapolated through missing detections. The original CSV stores mask-overlap measurements but not mask pixels or appearance embeddings. A missing tip is an occlusion proxy that can also indicate detector failure or an out-of-view instrument.

V. TRAINING AND EVALUATION PROTOCOL

A. Event-preserving grouped splits

Candidate group boundaries start every 30 s and move beyond collision intervals plus a 4.5 s margin. The resulting 24 contiguous groups keep every padded collision interval intact. Five outer StratifiedGroupKFold splits use seed 42. Four seconds of training samples adjacent to held-out groups are purged to reduce overlap from cached local temporal features.

For each outer fold, a three-way grouped splitter (seed 100 plus zero-based outer fold) supplies its first inner fit/validation split. Inner fit samples are again purged near validation groups. The inner model selects extraction settings; a fresh classifier is fitted to the full purged outer training set for test prediction. Labels, times, frame numbers, group IDs, severity, and near-miss labels are excluded from the features.

Both classifiers use each of the three feature sets in Table I, giving six combinations on identical outer splits. Invalid feature values use a fixed sentinel without learning from test data. All model settings are fixed before this comparison; there is no claim of exhaustive classifier hyperparameter optimization.

TABLE II. CLASSIFIER SETTINGS

| Setting | RF | XGB |
|---|---|---|
| Trees | 300 | 250 |
| Max. depth | 10 | 4 |
| Min. leaf / child | 3 samples | Weight 2 |
| Imbalance | Balanced weights | Negatives / positives |
| Learning rate | - | 0.05 |
| Row / column subsampling | - | 0.9 / 0.9 |
| Seed | 42 | 42 |

B. Frame-focused extraction

A causal mean of the current and up to two previous classifier scores is computed separately in each group. A threshold from 0.05 to 0.95 in steps of 0.01 maximizes inner-validation frame F1. The original extraction then merges neighboring positive samples whose timestamp gap is at most 0.4 s.

The extracted event starts at the first positive timestamp, ends at the last, and peaks at the highest smoothed score inside that interval. Confidence is the peak score; duration is end minus start. Single-sample events are retained with zero duration. Groups are never joined, and event peak/end are finalized only after the event closes.

C. Event-focused extraction

The follow-up experiment optimizes the threshold jointly with a gap from {0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0} s. This produces 637 candidate pairs per inner validation set. The selection objective is validation event F1, with deterministic ties resolved by higher event precision, then shorter gap, then higher threshold.

Previously saved outer classifier scores are reused unchanged. Because inner predictions were not originally saved, the same inner classifiers are fitted again; their frame-F1 thresholds are checked against the saved values before event tuning. This separates extraction changes from changes in classifier predictions.

The gap means positive-to-positive timestamp difference, not the duration of an arbitrary run of negative frames. With approximately 0.196 s sampling, 0.2 s joins adjacent positive samples and 0.4 s can bridge one negative sample. No minimum duration, hysteresis, or new smoothing parameter is optimized here.

D. Event matching and performance metrics

A prediction is eligible for an annotation if it overlaps the annotation expanded by 0.5 s. Maximum-cardinality one-to-one assignment prevents one broad prediction from detecting several annotation rows. Midpoint proximity provides a small tie-break. Unmatched annotations are missed events and unmatched predictions are false events.

F1<sub>event</sub> = 2D / (2D + F + M), &nbsp; (4)

where D, F, and M denote detected, false, and missed event counts. Event precision is D/(D+F), and recall is D/(D+M). We also save strict zero-tolerance evaluation and mean absolute start/end errors for matched pairs. Predicted peak timing is compared with annotation midpoint because human peak-contact times are unavailable.

Frame metrics include precision, recall, F1, ROC-AUC, average precision, and the confusion matrix. A missing-tip subset is defined by at least one absent YOLO tip. Pooled held-out scores are descriptive within-video estimates; individual neighboring frames are not treated as independent experimental replications.

TABLE III. DATA AND SPLIT AUDIT

| Quantity | Verified value |
|---|---|
| Sampled / positive frames | 3,917 / 558 |
| Collision annotation rows | 64 |
| Groups / outer folds | 24 / 5 |
| Missing-tip / positive subset | 1,008 / 315 |
| Input / outer scores | SHA-256 / equality checked |

![Fig. 2. Pooled held-out F1 under frame-focused and event-focused extraction. O: original 43 features; K: Kalman features; KT: Kalman plus 2 s history. RF: Random Forest; XGB: XGBoost. The same outer classifier scores are used in both settings. Gains in event F1 can accompany reduced frame F1.](../../results/charts/updated/fig2_ieee_frame_event_results.png)

VI. EXPERIMENTAL RESULTS

A. Trajectories and frame-focused decisions

TABLE IV. FRAME METRICS WITH FRAME-F1 EXTRACTION

| Model | P | R | F1 | AUC | AP |
|---|---|---|---|---|---|
| O-RF | 0.219 | 0.591 | 0.319 | 0.694 | 0.256 |
| O-XGB | 0.195 | 0.616 | 0.296 | 0.614 | 0.206 |
| K-RF | 0.177 | 0.659 | 0.280 | 0.696 | 0.271 |
| K-XGB | 0.198 | 0.591 | 0.297 | 0.621 | 0.206 |
| KT-RF | 0.232 | 0.695 | 0.348 | 0.706 | 0.294 |
| KT-XGB | 0.226 | 0.602 | 0.328 | 0.661 | 0.272 |

KT-RF increases frame F1 from 0.319 for O-RF to 0.348, while ROC-AUC rises from 0.694 to 0.706. K-RF alone has frame F1 0.280; Kalman tracking by itself therefore does not demonstrate a frame-F1 improvement.

In the missing-tip subset, RF F1 changes from 0.448 to 0.504 for KT. This supports a localized benefit in the detector-missing subset, but the subset does not establish that true physical occlusion caused each missing detection.

B. Event-focused extraction and tradeoffs

TABLE V. EVENT F1 BEFORE AND AFTER EVENT TUNING

| Model | Before | After | Change |
|---|---|---|---|
| O-RF | 0.385 | 0.392 | +0.008 |
| O-XGB | 0.320 | 0.357 | +0.037 |
| K-RF | 0.271 | 0.309 | +0.038 |
| K-XGB | 0.286 | 0.294 | +0.008 |
| KT-RF | 0.342 | 0.354 | +0.012 |
| KT-XGB | 0.275 | 0.304 | +0.029 |

Event F1 increases for all six combinations in this pilot (Fig. 2), but the effect is modest. O-RF remains highest at 0.392. KT-RF rises from 0.342 to 0.354; its detected events fall from 26 to 23, while false events fall from 62 to 43. Event-F1 optimization favors a different precision/recall balance rather than uniformly improving event detection.

Since the outer classifier scores are unchanged, ROC-AUC and average precision are identical before and after extraction tuning. Frame F1 can change because the threshold changes; merge-gap selection affects event grouping. The figure reports both outcomes to avoid interpreting event-F1 gains as improvements in every metric.

C. Event counts and timing

TABLE VI. EVENT-TUNED COUNTS AND START TIMING

| Model | D | M | F | Start MAE, s |
|---|---|---|---|---|
| O-RF | 30 | 34 | 59 | 1.630 |
| O-XGB | 28 | 36 | 65 | 1.679 |
| K-RF | 28 | 36 | 89 | 2.189 |
| K-XGB | 21 | 43 | 58 | 3.150 |
| KT-RF | 23 | 41 | 43 | 2.292 |
| KT-XGB | 21 | 43 | 53 | 2.464 |

D, M, and F are detected, missed, and false events. Timing error is evaluated on matched events only, so a low value cannot compensate for missed collisions. Large predicted intervals can overlap an annotation while having inaccurate boundaries. Complete end and peak/midpoint timing errors are available in the result JSON.

TABLE VII. KT-RF EVENT-FOCUSED SETTINGS

| Outer fold | Threshold | Gap, s |
|---|---|---|
| 1 | 0.280 | 1.5 |
| 2 | 0.360 | 0.2 |
| 3 | 0.090 | 1.0 |
| 4 | 0.300 | 1.0 |
| 5 | 0.290 | 0.2 |

Threshold and gap variation across folds indicates that one universal deployment setting is not established. Inner validation is used for selection; the outer results are not reused to select one final threshold. A production deployment would require a separate training/validation exercise and an independent test recording.

VII. DISCUSSION AND LIMITATIONS

The first result is that temporal trajectory summaries help more than simply substituting Kalman predictions. The second is that frame and event objectives disagree: a broad positive interval can cover many true frames but merge multiple human collision rows. Optimizing event F1 reduces some false bursts, yet it can also miss additional events. For instructional review, the acceptable tradeoff depends on how costly missed contacts are relative to false review prompts.

Only one video is available, and detector training provenance has not been audited. Existing feature histories were cached globally, whereas new Kalman states reset per group; the 4 s guard reduces local overlap but does not make the history implementations identical. This limits attribution of changes specifically to Kalman filtering. A matched causal-history ablation is needed.

The 64 annotation rows include point and overlapping intervals. One-to-one scoring deliberately counts them separately; an alternative human definition of contact episodes would change event metrics. Missing detections are not direct occlusion labels. Two-dimensional box centers do not establish physical contact or depth, and classifier confidence is not calibrated event probability.

Noise settings, expiry, classifiers, and the candidate grid are fixed engineering choices. No independent-video test or significance analysis establishes that these modest differences generalize. Although each tuning step respects held-out groups, the same recording has supported repeated project analysis; results remain exploratory.

VIII. FUTURE WORK AND CONCLUSION

The highest priority is more independently annotated video. With sufficient diversity, rerun LSTM/GRU/TCN on grouped Kalman sequences. Other candidates are coupled two-tip state estimation and appearance-based association. An IMM mixes motion-model estimates [8]; coupling two objects additionally requires meaningful cross-tip dynamics or covariance. DeepSORT uses appearance embeddings [9], whereas standard ByteTrack emphasizes association of high- and low-confidence detections [10]. Neither extension was implemented in this experiment.

In conclusion, Kalman trajectories with short histories improve frame-level scores on this recording, but do not establish superior event detection. Event-focused validation gives small held-out event-F1 gains without changing classifier scores. The original-feature RF still yields the highest event F1. End-to-end assessment must therefore report complete event counts and preserve the distinction between a useful tracking representation and a reliable collision detector.

IX. REPRODUCIBILITY

The detector cache is produced by scripts/02_extract_features.py. Scripts 15_kalman_experiment.py and 17_tune_event_extraction.py produce the new comparisons. Results are saved under results/kalman_updated and results/event_f1_tuned, including input hashes, feature lists, folds, validation grids, probabilities, trajectories, and event matches. Scripts 20_ieee_figures.py and 21_build_ieee_paper.py regenerate these figures and this manuscript.

Deterministic checks cover tracker expiry/reacquisition, constant velocity and covariance, gap bridging, group boundaries, and one-to-one matching. Output audits confirm sample coverage, annotation alignment, disjoint validation/test groups, and unchanged outer scores. All counts and comparison tables are read from saved experiment outputs.

REFERENCES

[1] Ultralytics, &quot;Instance segmentation with Ultralytics YOLO,&quot; documentation, accessed Sep. 2026. [Online]. Available: https://docs.ultralytics.com/tasks/segment

[2] R. E. Kalman, &quot;A new approach to linear filtering and prediction problems,&quot; Journal of Basic Engineering, vol. 82, no. 1, pp. 35-45, 1960. doi: 10.1115/1.3662552.

[3] L. Breiman, &quot;Random forests,&quot; Machine Learning, vol. 45, pp. 5-32, 2001. doi: 10.1023/A:1010933404324.

[4] T. Chen and C. Guestrin, &quot;XGBoost: A scalable tree boosting system,&quot; in Proc. ACM SIGKDD, 2016, pp. 785-794. arXiv:1603.02754.

[5] S. Hochreiter and J. Schmidhuber, &quot;Long short-term memory,&quot; Neural Computation, vol. 9, no. 8, pp. 1735-1780, 1997.

[6] K. Cho et al., &quot;Learning phrase representations using RNN encoder-decoder for statistical machine translation,&quot; in Proc. EMNLP, 2014, pp. 1724-1734.

[7] S. Bai, J. Z. Kolter, and V. Koltun, &quot;An empirical evaluation of generic convolutional and recurrent networks for sequence modeling,&quot; arXiv:1803.01271, 2018.

[8] MathWorks, &quot;trackingIMM: Interacting multiple model filter for object tracking,&quot; documentation, accessed Sep. 2026. [Online]. Available: https://www.mathworks.com/help/fusion/ref/trackingimm.html

[9] N. Wojke, A. Bewley, and D. Paulus, &quot;Simple online and realtime tracking with a deep association metric,&quot; in Proc. IEEE ICIP, 2017, pp. 3645-3649. arXiv:1703.07402.

[10] Y. Zhang et al., &quot;ByteTrack: Multi-object tracking by associating every detection box,&quot; in Proc. ECCV, 2022, pp. 1-21. arXiv:2110.06864.

APPENDIX: IMPLEMENTATION AND OUTPUTS

A. Event export schema

Each predicted event includes group, outer fold, start_frame, end_frame, start_s, end_s, peak_s, confidence, and duration_s. Frame exports include the original frame index, time, group, outer fold, ground-truth label, classifier probability, binary decision, threshold, and merge_gap_s. Collision matches identify both the predicted-event index and the annotation-row index.

B. Reproduction commands

From the project root, run the following scripts using the project Python environment. The paper builder needs ReportLab; the figures need Matplotlib. Detector inference is optional when reusing the verified cache.

scripts/15_kalman_experiment.py

scripts/17_tune_event_extraction.py

scripts/20_ieee_figures.py

scripts/21_build_ieee_paper.py

C. Analysis boundaries

The pipeline uses seconds for temporal derivatives and windows. Coordinates are image pixels; position covariance is in pixels squared. K includes 53 features and KT includes 85. State-code rolling statistics summarize ordered codes rather than categorical probabilities. Estimated trajectories stop after 1.5 s without a detection.

The deployment interpretation is causal for the new features and score smoothing. Event end and peak are retrospective within a completed event. Latency, runtime throughput, annotation agreement, 3D contact, identity-switch rate, and independent-video generalization were not evaluated. Additional research should measure those quantities rather than infer them from F1.

D. Manuscript status

This is an IEEE-style research draft with two-column text, numbered equations, figures, tables, and references. No author names, affiliations, acceptance, copyright transfer, or IEEE endorsement are asserted. Author metadata and a venue-specific submission template can be supplied when the manuscript is prepared for submission.