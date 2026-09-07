"""End-to-end IEEE-style two-column manuscript from existing measured results."""
from pathlib import Path
import json
import html
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import (BaseDocTemplate,PageTemplate,Frame,Paragraph,Table,TableStyle,
    Spacer,PageBreak,FrameBreak,NextPageTemplate)
from reportlab.lib.utils import ImageReader
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/charts/updated'
OLD=json.loads((ROOT/'results/kalman_updated/metrics.json').read_text())
NEW=json.loads((ROOT/'results/event_f1_tuned/metrics.json').read_text())
PROTOCOL=json.loads((ROOT/'results/kalman_updated/protocol.json').read_text())
W=516.; GAP=18.; CW=(W-GAP)/2
S={
 'body':ParagraphStyle('IEEEBody',fontName='Times-Roman',fontSize=10,leading=11.6,alignment=4,spaceAfter=5),
 'abstract':ParagraphStyle('IEEEAbstract',fontName='Times-Bold',fontSize=9,leading=10.6,alignment=4,spaceAfter=7),
 'heading':ParagraphStyle('IEEEHead',fontName='Times-Roman',fontSize=10,leading=12,alignment=1,spaceBefore=9,spaceAfter=6,keepWithNext=True),
 'sub':ParagraphStyle('IEEESub',fontName='Times-Italic',fontSize=10,leading=12,spaceBefore=6,spaceAfter=4,keepWithNext=True),
 'eq':ParagraphStyle('IEEEEq',fontName='Times-Roman',fontSize=9,leading=12,alignment=1,spaceBefore=3,spaceAfter=5),
 'cell':ParagraphStyle('IEEECell',fontName='Times-Roman',fontSize=8,leading=9.2),
 'caption':ParagraphStyle('IEEECaption',fontName='Times-Roman',fontSize=8,leading=9.4,spaceAfter=6),
 'ref':ParagraphStyle('IEEERef',fontName='Times-Roman',fontSize=8.5,leading=9.8,leftIndent=12,firstLineIndent=-12,spaceAfter=4),
}
TITLE='Occlusion-Aware Instrument-Tip Collision Detection:<br/>Kalman Trajectories and Event-Focused Evaluation'
FIGURES={
 'method':('fig1_ieee_end_to_end_pipeline.png','Fig. 1. End-to-end workflow. The original 43-feature branch bypasses Kalman tracking; the K and KT branches use the same cached detector observations. Inner validation selects extraction settings. Outer test groups supply all reported comparison scores.'),
 'result':('fig2_ieee_frame_event_results.png','Fig. 2. Pooled held-out F1 under frame-focused and event-focused extraction. O: original 43 features; K: Kalman features; KT: Kalman plus 2 s history. RF: Random Forest; XGB: XGBoost. The same outer classifier scores are used in both settings. Gains in event F1 can accompany reduced frame F1.')}
story=[]; source=[]

def p(text,style='body'):
    story.append(Paragraph(text,S[style])); source.append(text)
def h(text): p(text,'heading')
def sub(text): p(text,'sub')
def eq(text): p(text,'eq')
def col(): story.append(FrameBreak())
def page(kind):
    story.extend([NextPageTemplate(kind),PageBreak()])
    if kind in FIGURES:
        name,cap=FIGURES[kind]; source.append(f'![{cap}](../../results/charts/updated/{name})')
def f(value): return '-' if value is None else f'{value:.3f}'
def short(n): return n.replace('Original43','O').replace('KalmanTemporal','KT').replace('Kalman','K').replace('_','-')
def table(title,headers,rows,widths=None):
    p(title,'caption')
    widths=widths or [CW/len(headers)]*len(headers)
    data=[[Paragraph(html.escape(str(x)),S['cell']) for x in row] for row in [headers]+rows]
    tab=Table(data,colWidths=widths,repeatRows=1,hAlign='LEFT')
    tab.setStyle(TableStyle([('LINEABOVE',(0,0),(-1,0),.65,colors.black),
        ('LINEBELOW',(0,0),(-1,0),.45,colors.black),('LINEBELOW',(0,-1),(-1,-1),.65,colors.black),
        ('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),3),
        ('RIGHTPADDING',(0,0),(-1,-1),3),('TOPPADDING',(0,0),(-1,-1),4),
        ('BOTTOMPADDING',(0,0),(-1,-1),4)]))
    story.extend([tab,Spacer(1,7)])
    source.append('| '+' | '.join(headers)+' |\n|'+ '|'.join(['---']*len(headers))+'|\n'+
        '\n'.join('| '+' | '.join(map(str,row))+' |' for row in rows))

def frames(top):
    return [Frame(x,48,CW,top-48,id=str(i),leftPadding=0,rightPadding=0,topPadding=0,bottomPadding=0)
            for i,x in enumerate([48,48+CW+GAP])]
def draw(canvas,doc):
    canvas.setFont('Times-Roman',8); canvas.drawCentredString(306,27,str(doc.page))
    kind=doc.pageTemplate.id
    if kind=='first':
        title=Paragraph(TITLE,ParagraphStyle('Title',fontName='Times-Roman',fontSize=21,leading=24,alignment=1))
        _,hh=title.wrap(W,90); title.drawOn(canvas,48,746-hh)
        canvas.setFont('Times-Italic',9)
        canvas.drawCentredString(306,677,'Research manuscript | September 2026')
    elif kind in FIGURES:
        name,caption=FIGURES[kind]
        with Image.open(OUT/name) as im: iw,ih=im.size
        height=W*ih/iw
        canvas.drawImage(ImageReader(str(OUT/name)),48,746-height,width=W,height=height)
        cap=Paragraph(caption,S['caption']); _,ch=cap.wrap(W,60)
        cap.drawOn(canvas,48,746-height-ch-5)

def content():
    source.append('# '+TITLE.replace('<br/>',' '))
    a=OLD['Original43_RF']; b=OLD['KalmanTemporal_RF']; bt=NEW['KalmanTemporal_RF']
    p('<b>Abstract-</b>Automatic collision detection in procedural training video is difficult when one instrument tip becomes undetected near contact. We evaluate an occlusion-aware pipeline that combines cached YOLO11n-seg detections, independent Kalman filters, trajectory features, tree classifiers, and event extraction. A single 768.8 s recording provides 3,917 sampled frames and 64 collision annotation rows. Six feature/classifier combinations are evaluated with five event-preserving grouped outer folds and training-side threshold selection. Kalman trajectories with 2 s histories increase Random Forest frame F1 from 0.319 to 0.348 and missing-tip F1 from 0.448 to 0.504, but reduce event recall. Optimizing threshold and merge gap for validation event F1 raises held-out event F1 for this variant from 0.342 to 0.354; the original-feature Random Forest remains higher at 0.392. These pilot results show that frame-level improvement does not establish better event detection and motivate separate optimization of temporal event decisions.','abstract')
    p('<b>Index Terms-</b>Collision detection, instrument tracking, Kalman filter, occlusion, trajectory features, event evaluation.','abstract')
    h('I. INTRODUCTION')
    p('Instrument-tip collision events can help instructors review pattern-cutting practice. The desired output is an interval that an instructor can inspect, rather than a large set of independent positive frames. This distinction matters because a classifier can improve frame accuracy while producing broad bursts that merge several annotated contacts.')
    p('The existing project detects the left tip (TIPL), right tip (TIPR), and a reference marker (TIPandCircle) using YOLO11n-seg. A downstream classifier operates on 43 geometric, confidence, motion, overlap, and tracking features. Those features already include constant-velocity extrapolation and temporal summaries; the baseline is not purely frame-independent.')
    p('Our research question is whether explicit uncertainty-aware trajectories improve collision detection, especially when one tip is not detected. We additionally ask whether selecting event-extraction settings for event F1 improves the final collision event output without changing classifier scores.')
    col()
    p('The study makes three contributions: a reproducible Kalman trajectory implementation; a six-way comparison under an event-preserving grouped protocol; and a controlled comparison of frame-F1 versus event-F1 selection of extraction settings. The contributions are experimental and methodological. We do not claim a validated contact sensor or generalization to unseen trainees.')
    h('II. BACKGROUND AND PRIOR PROJECT RESULTS')
    p('YOLO segmentation supplies instance boxes, masks, and confidence values [1]. Kalman filtering recursively combines a motion prediction with noisy observations [2]. Random Forest (RF) [3] and XGBoost (XGB) [4] provide practical tabular baselines for the resulting geometric and temporal features.')
    p('The companion project compared XGBoost, LightGBM, CatBoost, RF, SVM, LSTM, GRU, and TCN. Its historical RF ROC-AUC was 0.857 and XGBoost F1 was 0.528. Those experiments used smaller temporal groups, centered smoothing, and a threshold selected using pooled out-of-fold labels. They are historical context, not a directly comparable estimate of improvement under the new protocol.')
    p('LSTM, GRU, and TCN are established sequence-model families [5]-[7]. They were not rerun on Kalman features in this work: the available dataset remains one video. A new neural-model study should use additional independent recordings and grouped sequence construction before conclusions about generalization are drawn.')
    h('III. DATASET AND TARGET DEFINITION')
    p('The input is one 1920 x 1080 pattern-cutting recording lasting approximately 768.8 s. The detector cache contains 3,917 samples with median timestamp spacing 0.196 s (approximately 5.10 Hz). Actual timestamps determine velocity and history windows. Human annotations contain 89 rows, including 64 collision rows and seven near-miss rows.')
    p('For frame classification, a sample is positive when it lies within any collision interval padded by 0.5 s on each side. This produces 558 positive frames (14.25%). Near misses remain negative unless a collision interval also covers the time. At event level, the 64 original collision rows are retained separately, including point, touching, and overlapping annotations.')

    page('method')
    h('IV. OCCLUSION-AWARE METHODOLOGY')
    sub('A. Detector measurements and tracking')
    p('For each sampled frame, the existing extractor retains the highest-confidence detection per class. A box center serves as the tip-position measurement z = [x, y]<super>T</super>. Box dimensions, confidence, mask area, and precomputed segmentation overlap are reused from the cache. Thus all feature variants share the same detector evidence (Fig. 1).')
    p('Each tip has an independent state s = [x, y, v<sub>x</sub>, v<sub>y</sub>]<super>T</super>. With interval dt, F advances x and y by dt times velocity. H selects position. The prediction and measurement correction are')
    eq('s<sup>-</sup><sub>t</sub> = F s<sub>t-1</sub>, &nbsp; P<sup>-</sup><sub>t</sub> = F P<sub>t-1</sub> F<super>T</super> + Q. &nbsp; (1)')
    eq('K = P<sup>-</sup> H<super>T</super>(H P<sup>-</sup> H<super>T</super> + R)<super>-1</super>,<br/>s<sub>t</sub> = s<sup>-</sup><sub>t</sub> + K(z<sub>t</sub> - Hs<sup>-</sup><sub>t</sub>). &nbsp; (2)')
    p('The implementation solves a linear system for K and uses the Joseph covariance update: P = (I-KH)P<sup>-</sup>(I-KH)<super>T</super> + KRK<super>T</super>. Process noise is Q = 100<super>2</super> GG<super>T</super>, where the rows of G are [dt<super>2</super>/2, 0], [0, dt<super>2</super>/2], [dt, 0], and [0, dt].')
    p('Measurement noise is R = [8<super>2</super>/max(c, 0.1)]I for detector confidence c. Initial velocity variance is 10,000 pixels squared per second squared. These fixed noise assumptions are not calibrated detector-error estimates.')
    col()
    sub('B. Occlusion handling and feature construction')
    p('When a tip is missing, the filter predicts for at most 1.5 s. A longer gap makes the state unavailable until a detection reinitializes it. Each output records source state (updated, predicted, unavailable), consecutive missing sample count, seconds since detection, and the trace of the position covariance matrix.')
    eq('d<sub>t</sub> = ||p<sub>L,t</sub> - p<sub>R,t</sub>||<sub>2</sub>, &nbsp; delta d<sub>t</sub> = d<sub>t</sub> - d<sub>t-1</sub>. &nbsp; (3)')
    p('Distance rate is delta d/dt; negative values indicate approach. Each tip contributes filtered position, velocity, speed, and speed derivative. Pair features include distance and relative speed. Invalid values remain missing in trajectory exports and use a fixed -1 sentinel for the classifiers, accompanied by availability indicators.')
    table('TABLE I. FEATURE VARIANTS',['ID','Features','Definition'],[
        ['O','43','Original cached geometric/temporal features'],
        ['K','53','Detector geometry + current Kalman/motion states'],
        ['KT','85','K + trailing 2 s summary features']], [24,44,181])
    p('KT adds mean, minimum, maximum, and standard deviation for eight signals: distance, distance rate, both tip speeds, both position variances, and both tracking-state codes. Windows and Kalman state reset at evaluation-group boundaries. This is causal temporal feature engineering, not a learned neural sequence model.')
    p('Segmentation masks are not extrapolated through missing detections. The original CSV stores mask-overlap measurements but not mask pixels or appearance embeddings. A missing tip is an occlusion proxy that can also indicate detector failure or an out-of-view instrument.')

    page('normal')
    h('V. TRAINING AND EVALUATION PROTOCOL')
    sub('A. Event-preserving grouped splits')
    p('Candidate group boundaries start every 30 s and move beyond collision intervals plus a 4.5 s margin. The resulting 24 contiguous groups keep every padded collision interval intact. Five outer StratifiedGroupKFold splits use seed 42. Four seconds of training samples adjacent to held-out groups are purged to reduce overlap from cached local temporal features.')
    p('For each outer fold, a three-way grouped splitter (seed 100 plus zero-based outer fold) supplies its first inner fit/validation split. Inner fit samples are again purged near validation groups. The inner model selects extraction settings; a fresh classifier is fitted to the full purged outer training set for test prediction. Labels, times, frame numbers, group IDs, severity, and near-miss labels are excluded from the features.')
    p('Both classifiers use each of the three feature sets in Table I, giving six combinations on identical outer splits. Invalid feature values use a fixed sentinel without learning from test data. All model settings are fixed before this comparison; there is no claim of exhaustive classifier hyperparameter optimization.')
    table('TABLE II. CLASSIFIER SETTINGS',['Setting','RF','XGB'],[
        ['Trees','300','250'],['Max. depth','10','4'],['Min. leaf / child','3 samples','Weight 2'],
        ['Imbalance','Balanced weights','Negatives / positives'],['Learning rate','-','0.05'],
        ['Row / column subsampling','-','0.9 / 0.9'],['Seed','42','42']], [104,66,79])
    sub('B. Frame-focused extraction')
    p('A causal mean of the current and up to two previous classifier scores is computed separately in each group. A threshold from 0.05 to 0.95 in steps of 0.01 maximizes inner-validation frame F1. The original extraction then merges neighboring positive samples whose timestamp gap is at most 0.4 s.')
    p('The extracted event starts at the first positive timestamp, ends at the last, and peaks at the highest smoothed score inside that interval. Confidence is the peak score; duration is end minus start. Single-sample events are retained with zero duration. Groups are never joined, and event peak/end are finalized only after the event closes.')
    col()
    sub('C. Event-focused extraction')
    p('The follow-up experiment optimizes the threshold jointly with a gap from {0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0} s. This produces 637 candidate pairs per inner validation set. The selection objective is validation event F1, with deterministic ties resolved by higher event precision, then shorter gap, then higher threshold.')
    p('Previously saved outer classifier scores are reused unchanged. Because inner predictions were not originally saved, the same inner classifiers are fitted again; their frame-F1 thresholds are checked against the saved values before event tuning. This separates extraction changes from changes in classifier predictions.')
    p('The gap means positive-to-positive timestamp difference, not the duration of an arbitrary run of negative frames. With approximately 0.196 s sampling, 0.2 s joins adjacent positive samples and 0.4 s can bridge one negative sample. No minimum duration, hysteresis, or new smoothing parameter is optimized here.')
    sub('D. Event matching and performance metrics')
    p('A prediction is eligible for an annotation if it overlaps the annotation expanded by 0.5 s. Maximum-cardinality one-to-one assignment prevents one broad prediction from detecting several annotation rows. Midpoint proximity provides a small tie-break. Unmatched annotations are missed events and unmatched predictions are false events.')
    eq('F1<sub>event</sub> = 2D / (2D + F + M), &nbsp; (4)')
    p('where D, F, and M denote detected, false, and missed event counts. Event precision is D/(D+F), and recall is D/(D+M). We also save strict zero-tolerance evaluation and mean absolute start/end errors for matched pairs. Predicted peak timing is compared with annotation midpoint because human peak-contact times are unavailable.')
    p('Frame metrics include precision, recall, F1, ROC-AUC, average precision, and the confusion matrix. A missing-tip subset is defined by at least one absent YOLO tip. Pooled held-out scores are descriptive within-video estimates; individual neighboring frames are not treated as independent experimental replications.')
    table('TABLE III. DATA AND SPLIT AUDIT',['Quantity','Verified value'],[
        ['Sampled / positive frames','3,917 / 558'],['Collision annotation rows','64'],
        ['Groups / outer folds','24 / 5'],['Missing-tip / positive subset','1,008 / 315'],
        ['Input / outer scores','SHA-256 / equality checked']], [157,92])

    page('result')
    h('VI. EXPERIMENTAL RESULTS')
    sub('A. Trajectories and frame-focused decisions')
    table('TABLE IV. FRAME METRICS WITH FRAME-F1 EXTRACTION',['Model','P','R','F1','AUC','AP'],[
        [short(n)]+[f(r['frame'][k]) for k in ['precision','recall','f1','roc_auc','average_precision']] for n,r in OLD.items()], [49,40,40,40,40,40])
    p(f"KT-RF increases frame F1 from {f(a['frame']['f1'])} for O-RF to {f(b['frame']['f1'])}, while ROC-AUC rises from {f(a['frame']['roc_auc'])} to {f(b['frame']['roc_auc'])}. K-RF alone has frame F1 {f(OLD['Kalman_RF']['frame']['f1'])}; Kalman tracking by itself therefore does not demonstrate a frame-F1 improvement.")
    p(f"In the missing-tip subset, RF F1 changes from {f(a['missing_tip']['f1'])} to {f(b['missing_tip']['f1'])} for KT. This supports a localized benefit in the detector-missing subset, but the subset does not establish that true physical occlusion caused each missing detection.")
    col()
    sub('B. Event-focused extraction and tradeoffs')
    table('TABLE V. EVENT F1 BEFORE AND AFTER EVENT TUNING',['Model','Before','After','Change'],[
        [short(n),f(r['baseline_event']['f1']),f(r['event']['f1']),f"{r['event']['f1']-r['baseline_event']['f1']:+.3f}"] for n,r in NEW.items()], [63,62,62,62])
    p('Event F1 increases for all six combinations in this pilot (Fig. 2), but the effect is modest. O-RF remains highest at 0.392. KT-RF rises from 0.342 to 0.354; its detected events fall from 26 to 23, while false events fall from 62 to 43. Event-F1 optimization favors a different precision/recall balance rather than uniformly improving event detection.')
    p('Since the outer classifier scores are unchanged, ROC-AUC and average precision are identical before and after extraction tuning. Frame F1 can change because the threshold changes; merge-gap selection affects event grouping. The figure reports both outcomes to avoid interpreting event-F1 gains as improvements in every metric.')

    page('normal')
    sub('C. Event counts and timing')
    table('TABLE VI. EVENT-TUNED COUNTS AND START TIMING',['Model','D','M','F','Start MAE, s'],[
        [short(n),r['event']['detected'],r['event']['missed'],r['event']['false_events'],f(r['event']['start_mae_s'])]
        for n,r in NEW.items()], [58,31,31,31,98])
    p('D, M, and F are detected, missed, and false events. Timing error is evaluated on matched events only, so a low value cannot compensate for missed collisions. Large predicted intervals can overlap an annotation while having inaccurate boundaries. Complete end and peak/midpoint timing errors are available in the result JSON.')
    table('TABLE VII. KT-RF EVENT-FOCUSED SETTINGS',['Outer fold','Threshold','Gap, s'],[
        [r['fold'],f(r['selected']['threshold']),r['selected']['merge_gap_s']] for r in bt['folds']], [83,83,83])
    p('Threshold and gap variation across folds indicates that one universal deployment setting is not established. Inner validation is used for selection; the outer results are not reused to select one final threshold. A production deployment would require a separate training/validation exercise and an independent test recording.')
    h('VII. DISCUSSION AND LIMITATIONS')
    p('The first result is that temporal trajectory summaries help more than simply substituting Kalman predictions. The second is that frame and event objectives disagree: a broad positive interval can cover many true frames but merge multiple human collision rows. Optimizing event F1 reduces some false bursts, yet it can also miss additional events. For instructional review, the acceptable tradeoff depends on how costly missed contacts are relative to false review prompts.')
    p('Only one video is available, and detector training provenance has not been audited. Existing feature histories were cached globally, whereas new Kalman states reset per group; the 4 s guard reduces local overlap but does not make the history implementations identical. This limits attribution of changes specifically to Kalman filtering. A matched causal-history ablation is needed.')
    col()
    p('The 64 annotation rows include point and overlapping intervals. One-to-one scoring deliberately counts them separately; an alternative human definition of contact episodes would change event metrics. Missing detections are not direct occlusion labels. Two-dimensional box centers do not establish physical contact or depth, and classifier confidence is not calibrated event probability.')
    p('Noise settings, expiry, classifiers, and the candidate grid are fixed engineering choices. No independent-video test or significance analysis establishes that these modest differences generalize. Although each tuning step respects held-out groups, the same recording has supported repeated project analysis; results remain exploratory.')
    h('VIII. FUTURE WORK AND CONCLUSION')
    p('The highest priority is more independently annotated video. With sufficient diversity, rerun LSTM/GRU/TCN on grouped Kalman sequences. Other candidates are coupled two-tip state estimation and appearance-based association. An IMM mixes motion-model estimates [8]; coupling two objects additionally requires meaningful cross-tip dynamics or covariance. DeepSORT uses appearance embeddings [9], whereas standard ByteTrack emphasizes association of high- and low-confidence detections [10]. Neither extension was implemented in this experiment.')
    p('In conclusion, Kalman trajectories with short histories improve frame-level scores on this recording, but do not establish superior event detection. Event-focused validation gives small held-out event-F1 gains without changing classifier scores. The original-feature RF still yields the highest event F1. End-to-end assessment must therefore report complete event counts and preserve the distinction between a useful tracking representation and a reliable collision detector.')
    h('IX. REPRODUCIBILITY')
    p('The detector cache is produced by scripts/02_extract_features.py. Scripts 15_kalman_experiment.py and 17_tune_event_extraction.py produce the new comparisons. Results are saved under results/kalman_updated and results/event_f1_tuned, including input hashes, feature lists, folds, validation grids, probabilities, trajectories, and event matches. Scripts 20_ieee_figures.py and 21_build_ieee_paper.py regenerate these figures and this manuscript.')
    p('Deterministic checks cover tracker expiry/reacquisition, constant velocity and covariance, gap bridging, group boundaries, and one-to-one matching. Output audits confirm sample coverage, annotation alignment, disjoint validation/test groups, and unchanged outer scores. All counts and comparison tables are read from saved experiment outputs.')

    page('normal')
    h('REFERENCES')
    refs=[
      '[1] Ultralytics, "Instance segmentation with Ultralytics YOLO," documentation, accessed Sep. 2026. [Online]. Available: https://docs.ultralytics.com/tasks/segment',
      '[2] R. E. Kalman, "A new approach to linear filtering and prediction problems," Journal of Basic Engineering, vol. 82, no. 1, pp. 35-45, 1960. doi: 10.1115/1.3662552.',
      '[3] L. Breiman, "Random forests," Machine Learning, vol. 45, pp. 5-32, 2001. doi: 10.1023/A:1010933404324.',
      '[4] T. Chen and C. Guestrin, "XGBoost: A scalable tree boosting system," in Proc. ACM SIGKDD, 2016, pp. 785-794. arXiv:1603.02754.',
      '[5] S. Hochreiter and J. Schmidhuber, "Long short-term memory," Neural Computation, vol. 9, no. 8, pp. 1735-1780, 1997.',
      '[6] K. Cho et al., "Learning phrase representations using RNN encoder-decoder for statistical machine translation," in Proc. EMNLP, 2014, pp. 1724-1734.',
      '[7] S. Bai, J. Z. Kolter, and V. Koltun, "An empirical evaluation of generic convolutional and recurrent networks for sequence modeling," arXiv:1803.01271, 2018.',
      '[8] MathWorks, "trackingIMM: Interacting multiple model filter for object tracking," documentation, accessed Sep. 2026. [Online]. Available: https://www.mathworks.com/help/fusion/ref/trackingimm.html',
      '[9] N. Wojke, A. Bewley, and D. Paulus, "Simple online and realtime tracking with a deep association metric," in Proc. IEEE ICIP, 2017, pp. 3645-3649. arXiv:1703.07402.',
      '[10] Y. Zhang et al., "ByteTrack: Multi-object tracking by associating every detection box," in Proc. ECCV, 2022, pp. 1-21. arXiv:2110.06864.'
    ]
    for ref in refs: p(html.escape(ref),'ref')
    col()
    h('APPENDIX: IMPLEMENTATION AND OUTPUTS')
    sub('A. Event export schema')
    p('Each predicted event includes group, outer fold, start_frame, end_frame, start_s, end_s, peak_s, confidence, and duration_s. Frame exports include the original frame index, time, group, outer fold, ground-truth label, classifier probability, binary decision, threshold, and merge_gap_s. Collision matches identify both the predicted-event index and the annotation-row index.')
    sub('B. Reproduction commands')
    p('From the project root, run the following scripts using the project Python environment. The paper builder needs ReportLab; the figures need Matplotlib. Detector inference is optional when reusing the verified cache.')
    for command in ['scripts/15_kalman_experiment.py','scripts/17_tune_event_extraction.py','scripts/20_ieee_figures.py','scripts/21_build_ieee_paper.py']:
        p(command,'eq')
    sub('C. Analysis boundaries')
    p('The pipeline uses seconds for temporal derivatives and windows. Coordinates are image pixels; position covariance is in pixels squared. K includes 53 features and KT includes 85. State-code rolling statistics summarize ordered codes rather than categorical probabilities. Estimated trajectories stop after 1.5 s without a detection.')
    p('The deployment interpretation is causal for the new features and score smoothing. Event end and peak are retrospective within a completed event. Latency, runtime throughput, annotation agreement, 3D contact, identity-switch rate, and independent-video generalization were not evaluated. Additional research should measure those quantities rather than infer them from F1.')
    sub('D. Manuscript status')
    p('This is an IEEE-style research draft with two-column text, numbered equations, figures, tables, and references. No author names, affiliations, acceptance, copyright transfer, or IEEE endorsement are asserted. Author metadata and a venue-specific submission template can be supplied when the manuscript is prepared for submission.')

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    content()
    doc=BaseDocTemplate(str(OUT/'collision_detection_end_to_end_IEEE.pdf'),pagesize=(612,792),
        title='Occlusion-Aware Instrument-Tip Collision Detection: Kalman Trajectories and Event-Focused Evaluation',
        author='',leftMargin=48,rightMargin=48,topMargin=46,bottomMargin=48)
    templates=[PageTemplate(id='first',frames=frames(658),onPage=draw),
               PageTemplate(id='normal',frames=frames(746),onPage=draw)]
    for kind,(name,caption) in FIGURES.items():
        with Image.open(OUT/name) as im: iw,ih=im.size
        cap=Paragraph(caption,S['caption']); _,ch=cap.wrap(W,60)
        top=746-W*ih/iw-ch-16
        templates.append(PageTemplate(id=kind,frames=frames(top),onPage=draw))
    doc.addPageTemplates(templates); doc.build(story)
    (ROOT/'paper/source/collision_detection_end_to_end_IEEE.md').write_text('\n\n'.join(source),encoding='utf-8')
    print(OUT/'collision_detection_end_to_end_IEEE.pdf')

if __name__=='__main__': main()
