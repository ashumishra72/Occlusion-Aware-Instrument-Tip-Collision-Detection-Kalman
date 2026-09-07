"""Plain-language companion PDF, using the measured event-F1 tuning results."""
import json
import html
from pathlib import Path
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,PageBreak,Image,Preformatted

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'output/pdf'
M=json.loads((ROOT/'results/event_f1_tuned/metrics.json').read_text())
ST=getSampleStyleSheet()
ST['BodyText'].fontSize=10.5; ST['BodyText'].leading=15; ST['BodyText'].spaceAfter=10
ST['Heading1'].fontSize=22; ST['Heading1'].leading=26; ST['Heading1'].textColor=colors.HexColor('#18374b')
ST['Heading2'].fontSize=13; ST['Heading2'].leading=17; ST['Heading2'].spaceBefore=10
ST['Heading2'].textColor=colors.HexColor('#087f8c')
ST.add(ParagraphStyle(name='CellNew',fontSize=8.5,leading=11))
ST.add(ParagraphStyle(name='CodeNew',fontName='Courier',fontSize=8.5,leading=12,spaceAfter=12))
story=[]
def p(text,style='BodyText'): story.append(Paragraph(text,ST[style]))
def h(text): p(text,'Heading2')
def page(title):
    if story: story.append(PageBreak())
    p(title,'Heading1')
def table(headers,rows,widths):
    data=[[Paragraph(html.escape(str(x)),ST['CellNew']) for x in row] for row in [headers]+rows]
    tab=Table(data,colWidths=widths,repeatRows=1,hAlign='LEFT')
    tab.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#dcebf0')),
        ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#f1f5f7')]),
        ('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),7),
        ('RIGHTPADDING',(0,0),(-1,-1),7),('TOPPADDING',(0,0),(-1,-1),5),
        ('BOTTOMPADDING',(0,0),(-1,-1),5)]))
    story.append(tab); story.append(Spacer(1,12))
def picture(name,width):
    from PIL import Image as PILImage
    with PILImage.open(OUT/name) as im: w,hh=im.size
    story.append(Image(str(OUT/name),width=width,height=width*hh/w)); story.append(Spacer(1,10))
def label(n): return n.replace('Original43','Original 43').replace('KalmanTemporal','Kalman + history').replace('_',' / ')
def f(x): return f'{x:.3f}'
def footer(c,doc):
    c.setStrokeColor(colors.HexColor('#cbd9e0')); c.line(48,44,564,44)
    c.setFont('Helvetica',8); c.setFillColor(colors.HexColor('#607788'))
    c.drawString(48,30,'COLLISION PROJECT | SIMPLE EXPLANATION + EVENT-F1 RESULTS | 07 SEPTEMBER 2026')
    c.drawRightString(564,30,str(doc.page))

def main():
    page('What do these five suggestions mean?')
    p('A simple explanation, two figures, and a new experiment on your existing video.')
    h('1. Judge complete collisions, not just individual frames')
    p('A collision can last for several video frames. The threshold decides how high a score must be before we call a frame positive. The merge gap decides how close positive samples must be to count as one collision. Choose these settings to count whole collisions correctly. <b>This is the experiment completed in this report.</b>')
    h('2. Let a model learn the movement sequence')
    p('LSTM, GRU and TCN models can learn patterns such as approach, contact and separation. The recommendation says to try them again after there are enough independent labeled videos. The project still has one video, so these models were not rerun here. Many nearby frames do not replace many independent examples.')
    h('3. Track both tips as a pair')
    p('Today, each tip has its own tracker. A coupled tracker would represent both tips together, including their relative movement and shared uncertainty. An IMM could combine different motion assumptions, such as steady and rapidly changing motion. This is a proposed extension, not a result from the current experiment.')
    h('4. Remember which tip is which')
    p('When a tip disappears and returns, reconnect it to the correct track. DeepSORT uses visual appearance features to help do this [1]. Standard ByteTrack mainly reconnects tracks using motion and detections with both high and low confidence [2]; it is not inherently an appearance-embedding method.')
    h('5. Add more human-labeled videos')
    p('Collect more examples from different trainees and conditions, including real hidden-tip collisions. Keep separate videos for testing. This would let you assess whether an improvement works beyond the one video already available.')
    p('<b>The passage lists future research directions. It does not say that all five improvements have already been built or proven.</b>')

    page('Figure 1. What can be done now and next')
    picture('figure_improvement_roadmap.png',516)
    p('The green row is the completed experiment. The other rows describe future development or additional data collection. The figure is also saved separately as PNG and editable SVG.')
    h('Two distinctions that matter')
    p('<b>Coupled tracking and IMM are different ideas.</b> Putting two independent states into one vector is not enough to couple them. Cross-tip dynamics, shared process noise, or joint measurements must create a meaningful relationship. An IMM mixes estimates from multiple motion models; it does not automatically couple two objects or prove contact [3].')
    p('<b>Identity and collision are different tasks.</b> Better identity persistence can make trajectories more reliable, but it does not guarantee better collision detection. Evaluate identity switches as well as event F1 when testing association methods.')
    p('For a future coupled design, relative-position uncertainty is P(relative) = P(left) + P(right) - P(left,right) - P(right,left). Independent filters omit the cross-tip covariance terms. Whether coupling helps must be measured, since the instruments can also move independently.')

    page('Figure 2. New measured results')
    picture('figure_event_f1_results.png',516)
    table(['Method','Previous event F1','Event-tuned F1','Change'],[
        [label(n),f(r['baseline_event']['f1']),f(r['event']['f1']),f"{r['event']['f1']-r['baseline_event']['f1']:+.3f}"] for n,r in M.items()], [219,99,99,99])
    a=M['KalmanTemporal_RF']; best=max(M,key=lambda n:M[n]['event']['f1'])
    p(f"For Kalman + history / RF, event F1 changes from <b>{f(a['baseline_event']['f1'])} to {f(a['event']['f1'])}</b>. The highest event F1 in this comparison is <b>{f(M[best]['event']['f1'])}</b>, achieved by {label(best)}. These are modest observed gains on one video, not proof of a general improvement.")
    p('RF = Random Forest; XGB = XGBoost. All values are held-out event scores. Tuning uses inner validation only; outer test labels do not choose the threshold or gap. The plot is not a result from LSTM/GRU/TCN, IMM or appearance association.')

    page('What changed in the event counts?')
    table(['Method','Detected before / after','Missed before / after','False before / after'],[
        [label(n),f"{r['baseline_event']['detected']} / {r['event']['detected']}",
         f"{r['baseline_event']['missed']} / {r['event']['missed']}",
         f"{r['baseline_event']['false_events']} / {r['event']['false_events']}"] for n,r in M.items()], [210,102,102,102])
    p('There are 64 collision annotation rows. A detected event is matched to one annotation. A missed event is an annotation without a match. A false event is a prediction without a match. One long prediction can match only one annotation, even if it covers several contacts.')
    h('How the new settings were chosen')
    p('For each of the five outer folds, train an inner model and test 91 thresholds (0.05-0.95, step 0.01) with seven gaps (0.2, 0.4, 0.6, 0.8, 1.0, 1.5 and 2.0 seconds). Select the pair with the highest event F1 on inner validation. Ties prefer higher event precision, then a shorter gap, then a higher threshold.')
    p('Keep the original held-out classifier scores, feature sets, 24 temporal groups and three-frame causal smoothing unchanged. Apply the selected threshold and gap to the outer test fold. This isolates the effect of changing event extraction, rather than retraining or changing the outer classifier.')
    h('Selected settings: Kalman + history / RF')
    table(['Outer fold','Previous threshold','New threshold','New gap (s)'],[
        [r['fold'],f(r['old_frame_threshold']),f(r['selected']['threshold']),r['selected']['merge_gap_s']]
        for r in M['KalmanTemporal_RF']['folds']], [105,137,137,137])
    p('The gap is the time from one positive sample to the next positive sample. At about 5 Hz, 0.2 seconds joins consecutive positives; 0.4 seconds can bridge one intervening negative sample. There is no single universally validated threshold/gap pair from cross-validation.')
    p('Event F1 = 2 x detected / (2 x detected + false + missed). Matching allows 0.5 seconds around each human annotation. Strict zero-tolerance scores and matched timing errors are also saved in the results JSON.')

    page('Files, reproduction and honest limits')
    h('Saved alongside updated.pdf')
    table(['File','Content'],[
        ['event_f1_explained.pdf','This plain-language report and the two figures.'],
        ['figure_improvement_roadmap.png / .svg','A high-resolution figure explaining the five suggestions.'],
        ['figure_event_f1_results.png / .svg','Actual before/after event F1 for all six models.']], [254,262])
    p('Detailed experiment files are in <b>results/event_f1_tuned/</b>: metrics.json, protocol.json, validation_grid.csv, 30 inner-validation prediction files, and six event/prediction file pairs. Previous results and updated.pdf are preserved.')
    h('Run from the project root')
    story.append(Preformatted('venv\\Scripts\\python.exe scripts\\17_tune_event_extraction.py\nvenv\\Scripts\\python.exe scripts\\18_event_figures.py\npython scripts\\19_event_report.py',ST['CodeNew']))
    h('What has not been established')
    p('No new videos were added. Neural temporal models, coupled Kalman/IMM tracking and appearance association were not implemented in this experiment. The original cache lacks appearance embeddings and discarded low-confidence candidate detections, so association research needs a new detection export from the video. No significance test or independent-video validation establishes that these small gains will persist.')
    p('Model selection remains exploratory because the same video has already been used for earlier analysis. Point and overlapping annotation rows also make event counting sensitive to annotation conventions. The new event tuning is a useful pilot result, not a final deployment setting.')
    h('References for the explanations')
    p('[1] Wojke et al., <i>Simple Online and Realtime Tracking with a Deep Association Metric</i> (2017). <link href="https://arxiv.org/abs/1703.07402" color="#087f8c">arxiv.org/abs/1703.07402</link>.')
    p('[2] Zhang et al., <i>ByteTrack: Multi-Object Tracking by Associating Every Detection Box</i> (2021). <link href="https://arxiv.org/abs/2110.06864" color="#087f8c">arxiv.org/abs/2110.06864</link>.')
    p('[3] MathWorks, <i>trackingIMM: Interacting multiple model filter</i>. <link href="https://www.mathworks.com/help/fusion/ref/trackingimm.html" color="#087f8c">mathworks.com/help/fusion/ref/trackingimm.html</link>. The numbered citations in your pasted passage refer to its own bibliography; this report uses the sources listed here.')
    SimpleDocTemplate(str(OUT/'event_f1_explained.pdf'),pagesize=(612,792),leftMargin=48,rightMargin=48,
        topMargin=45,bottomMargin=58,title='Collision detection - simple explanation and event-F1 results',
        author='Collision Project').build(story,onFirstPage=footer,onLaterPages=footer)
    print(OUT/'event_f1_explained.pdf')

if __name__=='__main__': main()
