"""Build the updated research report from measured experiment outputs."""
import csv
import json
import html
from pathlib import Path
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Preformatted
from reportlab.graphics.shapes import Drawing, Rect, String, Line

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'output/pdf'
R=ROOT/'results/kalman_updated'
M=json.loads((R/'metrics.json').read_text())
C=json.loads((R/'protocol.json').read_text())
ST=getSampleStyleSheet()
ST.add(ParagraphStyle(name='TitleNew',fontName='Helvetica-Bold',fontSize=25,leading=29,textColor=colors.HexColor('#12324a'),spaceAfter=16))
ST.add(ParagraphStyle(name='Deck',fontSize=12,leading=18,textColor=colors.HexColor('#405366'),spaceAfter=14))
ST['BodyText'].fontSize=10; ST['BodyText'].leading=14; ST['BodyText'].spaceAfter=9
ST['Heading1'].fontSize=19; ST['Heading1'].leading=23; ST['Heading1'].textColor=colors.HexColor('#12324a')
ST['Heading2'].fontSize=12; ST['Heading2'].leading=16
ST.add(ParagraphStyle(name='Cell',fontSize=8,leading=10))
ST.add(ParagraphStyle(name='MonoNew',fontName='Courier',fontSize=8,leading=11,spaceAfter=12))
story=[]; source=[]

def p(text,style='BodyText'):
    story.append(Paragraph(text,ST[style])); source.append(text)
def h(text): p(text,'Heading2')
def page(title):
    if story: story.append(PageBreak())
    p(title,'Heading1')
def code(text):
    story.append(Preformatted(text,ST['MonoNew'])); source.append('```\n'+text+'\n```')
def table(headers,rows,widths):
    data=[[Paragraph(html.escape(str(x)),ST['Cell']) for x in row] for row in [headers]+rows]
    obj=Table(data,colWidths=widths,repeatRows=1,hAlign='LEFT')
    obj.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#dceaf1')),
        ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#f3f6f8')]),
        ('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),7),
        ('RIGHTPADDING',(0,0),(-1,-1),7),('TOPPADDING',(0,0),(-1,-1),7),
        ('BOTTOMPADDING',(0,0),(-1,-1),7),('LINEBELOW',(0,0),(-1,0),.7,colors.HexColor('#a7bdca'))]))
    story.append(obj); story.append(Spacer(1,12)); source.append(str(headers)+'\n'+'\n'.join(map(str,rows)))
def f(x): return '-' if x is None else f'{x:.3f}'
def label(n): return n.replace('Original43','Original 43').replace('KalmanTemporal','Kalman + history').replace('_',' / ')
def footer(c,doc):
    c.setStrokeColor(colors.HexColor('#c9d7df')); c.line(48,43,564,43)
    c.setFont('Helvetica',8); c.setFillColor(colors.HexColor('#526879'))
    c.drawString(48,30,'COLLISION PROJECT  |  UPDATED EXPERIMENT  |  07 SEPTEMBER 2026')
    c.drawRightString(564,30,str(doc.page))

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    a=M['Original43_RF']; b=M['KalmanTemporal_RF']
    p('OCCLUSION-AWARE COLLISION DETECTION','Deck')
    p('Tracking instrument tips through missing detections','TitleNew')
    p('Updated methodology, runnable implementation and measured results','Deck')
    p('Research question: Can Kalman tracking and short trajectory histories improve instrument-tip collision detection compared with the original 43-feature approach?')
    h('Measured answer')
    p(f"On this single-video experiment, Random Forest with Kalman trajectories and two seconds of history increases frame F1 from <b>{f(a['frame']['f1'])} to {f(b['frame']['f1'])}</b>. Missing-tip frame F1 increases from <b>{f(a['missing_tip']['f1'])} to {f(b['missing_tip']['f1'])}</b>. However, event recall falls from <b>{f(a['event']['recall'])} to {f(b['event']['recall'])}</b>. The evidence supports a modest frame-level benefit, not an overall improvement in event detection.")
    table(['Data','Completed work'],[
        [f"{C['n_frames']} sampled frames; {C['positive_frames']} positive frames",'Two independent Kalman filters; uncertainty and dropout age'],
        [f"{C['collision_events']} collision annotation rows; one video",'Six grouped model comparisons; inner-validation thresholds'],
        [f"{C['n_groups']} temporal groups; five outer folds",'Frame metrics, missing-tip analysis and one-to-one event matching']], [222,294])
    p('The original paper and its results remain available. This report uses a revised evaluation protocol and new experiments; the older headline scores are historical context only.')
    p('Reading guide: pages 2-7 explain each step; pages 8-10 report outcomes and limitations; pages 11-12 provide reproduction instructions and sources.')

    page('1. What changes in the existing project')
    h('Simple explanation')
    p('The detector supplies observations. A tracker estimates each tip position between observations and records how uncertain that estimate becomes. A classifier uses the resulting movement history to score collision likelihood. Neighboring positive scores are then grouped into events.')
    code('Original implementation\nVideo -> YOLO11n-seg -> 43 features -> classifier\n\nUpdated implementation\nCached YOLO detections -> separate TIPL/TIPR Kalman filters\n -> trajectory + motion + missingness + uncertainty\n -> current features or trailing 2-second summaries\n -> Random Forest / XGBoost -> smoothed probability\n -> threshold -> collision events')
    h('The original baseline already contains temporal information')
    p('The 43 features include constant-velocity extrapolation, rolling statistics and motion features. It is therefore not a pure single-frame baseline. The new comparison retains all 43 cached columns in Original43, then replaces the old tracking and temporal terms in the Kalman variants.')
    table(['Variant','Feature count','Purpose'],[[label(n),str(M[n]['feature_count']),purpose] for n,purpose in [
        ('Original43_RF','Rerun existing features under revised evaluation'),
        ('Kalman_RF','Current detector plus Kalman/motion/uncertainty features'),
        ('KalmanTemporal_RF','Add trailing 2-second summaries')]], [155,70,291])
    p('Each feature set is evaluated with both Random Forest and XGBoost. Neural temporal models are deferred: only one labeled video is available, so independent sequence diversity is limited. No new LSTM, GRU, TCN or Transformer result is claimed.')

    page('2. Step 1 - detector inputs and labels')
    h('Simple explanation')
    p('For each sampled frame, keep the highest-confidence detection of each instrument tip. The existing extraction script already supplies the required box centers, bounding boxes, confidence and mask-overlap measurements. Reusing these values holds detector behavior constant between methods.')
    table(['Input','Format and meaning'],[
        ['data/video.mp4','Original video; detector inference can be reproduced with script 02.'],
        ['models/video_tip_circle_yolo11n_seg_best.pt','Existing trained TIPL, TIPR and TIPandCircle detector.'],
        ['data/features_and_labels.csv','One row per sampled frame; frame_idx, time_s, 43 features and three labels.'],
        ['TIPL_* / TIPR_*','present, x, y, w, h, conf, area. Coordinates are in image pixels.'],
        ['mask_iou_TIPL_TIPR','Saved overlap of the detected binary segmentation masks.'],
        ['data/ground_truth.csv','start_time_s, end_time_s, is_collision, severity and near-miss fields.']], [190,326])
    p(f"The cache contains {C['n_frames']} frames, with median sampling interval {C['sample_interval_s']:.3f} s (about {1/C['sample_interval_s']:.2f} Hz). Calculations use actual timestamps rather than assuming exactly 5 Hz.")
    code('center = ((x1 + x2)/2, (y1 + y2)/2)\nbox = (x1, y1, x2, y2)\nmask_iou = count(mask_L & mask_R) / count(mask_L | mask_R)\ny(t) = 1 if any collision interval covers t with +/-0.5 s padding')
    p('Masks themselves were not persisted in the original CSV. Their precomputed overlap is reused; new masks are not fabricated during occlusion. Missing overlap is disambiguated using detection-presence and tracking-state features. The box center is a tip-location proxy, not an annotated physical contact point.')
    p('The segmentation interface exposes per-instance masks, boxes and confidence scores [1]. To regenerate detections, use the existing script before running the new experiment; the completed run used the cache.')

    page('3. Step 2 - one Kalman filter per tip')
    h('Simple explanation')
    p('Each filter remembers position and velocity. It first predicts the next position. If YOLO sees the tip, the filter corrects the prediction using the observation. If the tip is missing, it predicts for at most 1.5 seconds. After that, the position is unavailable until a new detection initializes the track.')
    code('State: s = [x, y, vx, vy]^T         Observation: z = [x, y]^T\n\nF = [[1,0,dt,0], [0,1,0,dt], [0,0,1,0], [0,0,0,1]]\nH = [[1,0,0,0], [0,1,0,0]]\nG = [[dt^2/2,0], [0,dt^2/2], [dt,0], [0,dt]]\nQ = 100^2 * G G^T\nR = (8^2 / max(YOLO_confidence, 0.1)) * I_2\n\nPredict: s_minus = F s; P_minus = F P F^T + Q\nUpdate:  K = P_minus H^T (H P_minus H^T + R)^(-1)\n         s = s_minus + K (z - H s_minus)\n         A = I - K H\n         P = A P_minus A^T + K R K^T')
    p('The Joseph covariance update maintains numerical stability. The code solves a linear system instead of explicitly inverting a matrix. Initial position variance is the measurement variance; initial velocity variance is 10,000 pixels squared per second squared.')
    table(['Field','Interpretation'],[
        ['state = 0 / 1 / 2','YOLO-updated estimate / predicted estimate / unavailable'],
        ['missing_frames and age_s','Consecutive missing sampled frames and seconds since a real detection'],
        ['variance = Pxx + Pyy','Position covariance trace in pixels squared; larger means less certain'],
        ['max_missing_s = 1.5','Expire old tracks and reinitialize on reacquisition']], [190,326])
    p('Process noise, measurement noise and the confidence-to-variance rule are fixed engineering assumptions, not calibrated detector error measurements. Identity follows the TIPL/TIPR class labels; identity swaps and false detections are not corrected by this filter.')

    page('4. Steps 3-4 - trajectories and features')
    h('Simple explanation')
    p('The position estimates form a trajectory for each tip. Useful collision signals include closeness, approach speed, sudden motion changes, overlap, and whether the estimate is based on a visible tip or a prediction.')
    code('T_L = [(x_L(t1), y_L(t1)), ..., (x_L(tn), y_L(tn))]\nT_R = [(x_R(t1), y_R(t1)), ..., (x_R(tn), y_R(tn))]\n\nd(t) = sqrt((x_L-x_R)^2 + (y_L-y_R)^2)\ndelta_d(t) = d(t) - d(t-1)\ndistance_rate(t) = delta_d(t) / dt\nspeed_tip(t) = sqrt(vx(t)^2 + vy(t)^2)\nacceleration_tip(t) = (speed(t) - speed(t-1)) / dt\nrelative_speed(t) = norm(v_L(t) - v_R(t))')
    p('Negative distance rate indicates approach; positive distance rate indicates separation. Acceleration here is the derivative of speed, not a full vector-acceleration estimate. Missing positions stay unavailable in the trajectory export, and missing model inputs use a fixed -1 sentinel alongside state indicators.')
    table(['Feature family','What it contributes'],[
        ['Detector geometry','Tip/marker centers, sizes, confidence, mask area, raw distances and overlap'],
        ['Kalman state','Filtered x/y, vx/vy, speed and acceleration for both tips'],
        ['Pair motion','Distance, distance difference/rate and relative speed'],
        ['Occlusion proxies','Detection presence, track state, missing frame count, age and covariance trace'],
        ['Temporal summaries','Mean, minimum, maximum and standard deviation over the trailing 2 seconds']], [160,356])
    p('Temporal summaries cover distance, distance rate, both speeds, both position variances and both state codes. The means of state codes combine prediction and loss severity; explicit missingness fields remain available. Rolling windows and tracker state reset at evaluation-group boundaries.')
    p('A missing detection is only a proxy for occlusion: it may also reflect detector error or a tip outside the field of view. Two-dimensional proximity cannot establish physical contact in depth.')

    page('5. Steps 5-6 - history and model training')
    h('Simple explanation')
    p('Two seconds of past motion lets a tree classifier use recent approach and separation patterns. This is temporal feature engineering; it is not a neural sequence model. A causal predictor cannot use separation that has not happened yet, so recognition may be delayed until later frames.')
    table(['Setting','Random Forest','XGBoost'],[
        ['Estimators','300','250'],['Maximum depth','10','4'],
        ['Regularization','Minimum leaf size 3','Minimum child weight 2'],
        ['Imbalance handling','Balanced class weights','Training negatives / positives'],
        ['Other','Seed 42','Learning rate .05; row/column sampling .9']], [145,175,196])
    h('Grouped evaluation and threshold selection')
    p(f"Five outer StratifiedGroupKFold splits use {C['n_groups']} contiguous groups. Candidate boundaries begin every 30 seconds and move beyond collision annotations plus a 4.5-second margin. Thus padded event windows are never divided between groups. Four seconds of nearby training samples are purged around held-out groups.")
    p('Inside each outer training set, the first split of a separate three-way grouped splitter forms inner fit and validation sets. The inner model predicts validation scores; a threshold from 0.05 to 0.95 in 0.01 steps maximizes validation F1. A fresh model then fits the full purged outer training set and applies that fixed threshold to the outer test set.')
    code('for outer_train, outer_test in grouped_splits:\n    purge outer_train near outer_test\n    inner_fit, inner_val = grouped_split(outer_train)\n    purge inner_fit near inner_val\n    fit inner_model; score and smooth inner_val\n    threshold = argmax(validation_F1)\n    refit model on outer_train\n    score outer_test; smooth within each group\n    save predictions using the validation threshold')
    p('All six variants use identical outer splits. No labels, timestamps, frame numbers, severity or near-miss labels enter the classifier feature matrix. Group membership remains disjoint between fit, validation and test sets [2].')

    page('6. Steps 7-8 - probability to events')
    h('Simple explanation')
    p('Average each score with up to two preceding scores, apply the validation-selected threshold, and join nearby positive samples into an event. A long burst should produce one event, not one event per frame.')
    code('p_smooth(t) = mean(p(t), p(t-1), p(t-2))\ny_hat(t) = 1[p_smooth(t) >= threshold_from_inner_validation]\n\nWithin each evaluation group:\n  collect positive samples in time order\n  merge if next positive time - previous positive time <= 0.4 s\n  start = first positive timestamp\n  end = last positive timestamp\n  peak = highest smoothed score within the event\n  confidence = score at peak\n  duration = end - start')
    p('At the observed sampling interval, a 0.4-second positive-to-positive gap can bridge one intervening negative sample. Smoothing and merging never cross group boundaries. Single-sample events are retained with duration zero. Event end and peak are finalized after the event closes.')
    h('Example required output')
    table(['start_frame','end_frame','peak_s','confidence','duration_s'],[
        ['105','110','Timestamp of peak','Maximum score','t(110) - t(105)']], [92,92,112,108,112])
    p('This row illustrates the format only. Actual events are saved for every model in results/kalman_updated/*_events.csv with group, start_frame, end_frame, start_s, end_s, peak_s, confidence and duration_s.')
    h('Event matching')
    p('A prediction is eligible if it overlaps a human interval expanded by 0.5 seconds. Maximum-cardinality one-to-one assignment prevents one long prediction from counting as several detected collisions. Unmatched predictions are false events; unmatched annotations are misses. A small midpoint-distance tie-break resolves equally sized assignments.')
    p('All 64 collision annotation rows are counted separately, including touching or overlapping rows. This convention penalizes a merged prediction spanning several annotations. Strict zero-tolerance results are also saved. Peak timing is compared with the annotation midpoint because a human peak-contact time is unavailable.')

    page('7. Measured frame-level results')
    p('These are pooled held-out predictions from the new protocol. Thresholds vary by outer fold and are selected only on inner validation data. RF means Random Forest; XGB means XGBoost.')
    table(['Method','Precision','Recall','F1','ROC-AUC','Avg. prec.'],[
        [label(n)]+[f(r['frame'][k]) for k in ['precision','recall','f1','roc_auc','average_precision']]
        for n,r in M.items()], [176,68,68,68,68,68])
    h('Confusion matrices')
    table(['Method','TN','FP','FN','TP'],[[label(n),r['frame']['confusion_matrix'][0][0],r['frame']['confusion_matrix'][0][1],
        r['frame']['confusion_matrix'][1][0],r['frame']['confusion_matrix'][1][1]] for n,r in M.items()], [216,75,75,75,75])
    p(f"For RF, Kalman + history changes F1 by {b['frame']['f1']-a['frame']['f1']:+.3f}, ROC-AUC by {b['frame']['roc_auc']-a['frame']['roc_auc']:+.3f}, and average precision by {b['frame']['average_precision']-a['frame']['average_precision']:+.3f}. Kalman alone reduces F1 despite a similar ROC-AUC. Threshold choice and temporal features affect the practical tradeoff.")
    h('Why these scores differ from the old paper')
    p('Historical RF ROC-AUC was approximately 0.857 and XGBoost F1 approximately 0.528. Those scores used smaller groups, centered smoothing and a threshold chosen from pooled out-of-fold labels. The new run uses larger event-preserving groups, guarded splits, causal smoothing and inner-validation thresholds, with new fixed training settings. Historical and new scores are not a like-for-like measure of improvement.')

    page('8. Missing-tip and event-level results')
    table(['Method','Missing-tip F1','Visible-tip F1','Missing-tip recall'],[
        [label(n),f(r['missing_tip']['f1']),f(r['both_visible']['f1']),f(r['missing_tip']['recall'])] for n,r in M.items()], [210,102,102,102])
    p(f"The missing-tip subset has {a['missing_tip']['n']} frames, including {a['missing_tip']['positives']} positive frames. It is defined by at least one missing YOLO tip. The RF history variant improves F1 in this subset, but direct occlusion annotations would be needed to isolate true occlusion performance.")
    table(['Method','Detected','Missed','False','Event F1','Recall'],[
        [label(n),r['event']['detected'],r['event']['missed'],r['event']['false_events'],f(r['event']['f1']),f(r['event']['recall'])]
        for n,r in M.items()], [176,68,68,68,68,68])
    table(['Method','Start MAE (s)','End MAE (s)','Peak/midpoint MAE (s)'],[
        [label(n),f(r['event']['start_mae_s']),f(r['event']['end_mae_s']),f(r['event']['peak_midpoint_mae_s'])]
        for n,r in M.items()], [210,102,102,102])
    p('Timing errors are calculated on matched pairs only. A low timing error does not compensate for missed events. Broad predicted bursts can overlap an annotation even when their start and end times are inaccurate; duration and event recall should therefore be reviewed together.')

    page('9. Interpretation and next experiments')
    h('What the completed experiment establishes')
    p(f"The RF history variant detects {b['event']['detected']} of 64 annotations versus {a['event']['detected']} for the original-feature RF, while reducing false events from {a['event']['false_events']} to {b['event']['false_events']}. It improves per-frame classification but merges or misses more individual annotations. It should not replace the baseline on an event-recall objective without further work.")
    h('What remains uncertain')
    p('This is a single-video pilot, not evidence of generalization across trainees or cameras. The detector training/test provenance has not been audited. Missing detections are not verified occlusions, and 2D box centers do not measure physical tip contact. The original cached feature history was computed globally, while new tracks reset by group; the guard reduces local overlap but does not make both history implementations identical.')
    p('Parameters were fixed for this run. The temporal ablation changes both tracking and feature representation; it does not isolate the benefit of a Kalman filter from adding more history features. Scores are uncalibrated classifier outputs, so an event confidence of 0.8 should not be interpreted as a validated 80% probability.')
    h('Prioritized follow-up experiments')
    table(['Experiment','Question and procedure'],[
        ['More independent videos','Use held-out videos/trainees; audit detector training overlap before testing.'],
        ['Matched history ablation','Compare no tracking, constant velocity and Kalman with identical causal histories and resets.'],
        ['Tracker sensitivity','Tune Q, R and expiry (0.5/1.0/1.5 s) only inside training-side validation.'],
        ['Event tuning','Choose smoothing, hysteresis, gap and minimum duration using validation event F1, then evaluate once.'],
        ['Occlusion annotations','Mark true hidden-tip intervals and duration bins; evaluate tracking error with labeled positions.'],
        ['Neural temporal models','With sufficient independent sequences, test small GRU/TCN first; keep complete windows within groups.']], [158,358])
    p('Report paired group/video bootstrap uncertainty in a larger study. Do not treat thousands of neighboring frames as independent samples. Resolve whether touching annotation rows represent separate contacts before standardizing event metrics.')

    page('10. Reproduce the work step by step')
    p('Run these commands from the project root. The completed experiment reused the existing detector cache and left the original paper and results intact.')
    h('1. Optional: regenerate the detector cache')
    code('venv\\Scripts\\python.exe scripts\\02_extract_features.py\n# Default input: data/video.mp4; default sampling: about 5 Hz\n# This command overwrites data/features_and_labels.csv.')
    h('2. Check tracking and event logic')
    code('venv\\Scripts\\python.exe scripts\\test_kalman_experiment.py')
    p('Four deterministic tests pass: expiry/reacquisition, constant-velocity prediction and positive covariance, one-to-one matching, and group boundaries for smoothing/event extraction.')
    h('3. Run all six comparisons')
    code('venv\\Scripts\\python.exe scripts\\15_kalman_experiment.py')
    p('This writes trajectories.csv, metrics.json, protocol.json and six prediction/event CSV pairs to results/kalman_updated. It fits models for evaluation; it does not export a single production classifier. The report reports cross-validation results rather than training-set scores.')
    h('4. Build the PDF')
    code('python scripts\\16_generate_updated_pdf.py')
    p('Use a Python environment with ReportLab for the PDF builder. The experiment uses NumPy, pandas, SciPy, scikit-learn and XGBoost from the project environment. The PDF reads measured JSON results; it does not invent performance values.')
    h('Traceability')
    p('The protocol records group boundaries, parameters, sample counts and the input CSV SHA-256. The feature lists and fold-level metrics/thresholds are recorded for every model. Saved predictions include time, original frame, group, outer fold, ground truth, probability, binary prediction and missing-tip status.')
    code('Input CSV SHA-256:\n'+C['input_sha256'])

    page('11. Implementation map and references')
    table(['Python function','Responsibility'],[
        ['TipKalman.step','Predict/correct a tip state; expire stale tracks; expose uncertainty.'],
        ['make_groups','Move temporal boundaries outside padded collision intervals and history margins.'],
        ['features','Build per-tip trajectories, pair motion and causal 2-second summaries.'],
        ['model / main','Fit RF/XGB with grouped inner validation and five outer test folds.'],
        ['extract_events','Join positive samples within groups and export event timing/confidence.'],
        ['event_metrics','One-to-one event assignment and matched timing errors.']], [165,351])
    h('Minimal tracker usage')
    code('# TipKalman is defined in scripts/15_kalman_experiment.py\nleft = TipKalman(max_missing=1.5)\nright = TipKalman(max_missing=1.5)\n\n# One update per sampled timestamp:\nL = left.step(0.0, [120.0, 80.0], confidence=0.9)\nR = right.step(0.0, [170.0, 85.0], confidence=0.8)\nL_next = left.step(0.2, measurement=None)\n# [x, y, vx, vy, state, missing_frames, age_s, variance]')
    h('Sources and project evidence')
    p('[1] Ultralytics, Instance Segmentation and Prediction documentation. Per-instance boxes and masks are available through Results. <link href="https://docs.ultralytics.com/tasks/segment" color="#17678a">docs.ultralytics.com/tasks/segment</link> and <link href="https://docs.ultralytics.com/modes/predict" color="#17678a">docs.ultralytics.com/modes/predict</link>. Accessed September 2026.')
    p('[2] scikit-learn, Cross-validation: evaluating estimator performance. Grouped evaluation keeps a group out of both train and test simultaneously. <link href="https://scikit-learn.org/stable/modules/cross_validation.html" color="#17678a">scikit-learn.org/stable/modules/cross_validation.html</link>. Accessed September 2026.')
    p('[3] Project sources: scripts/02_extract_features.py; scripts/model_cv_utils.py; configs/experiment_config.yaml; paper/source/collision_detection_paper.md; data/ground_truth.csv; data/features_and_labels.csv.')
    p('[4] New empirical evidence: results/kalman_updated/metrics.json, protocol.json, trajectories.csv and per-model prediction/event exports. Full Python implementation: scripts/15_kalman_experiment.py. All numerical results in this report are loaded from these saved outputs.')
    SimpleDocTemplate(str(OUT/'updated.pdf'),pagesize=(612,792),leftMargin=48,rightMargin=48,
        topMargin=45,bottomMargin=58,title='Updated - Occlusion-aware collision detection',
        author='Collision Project').build(story,onFirstPage=footer,onLaterPages=footer)
    (ROOT/'paper/source/updated_report.md').write_text('\n\n'.join(source),encoding='utf-8')
    print(OUT/'updated.pdf')

if __name__=='__main__': main()
