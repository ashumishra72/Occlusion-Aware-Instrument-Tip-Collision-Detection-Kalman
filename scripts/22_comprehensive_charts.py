"""Publication-style diagnostic charts from saved held-out predictions only."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve,roc_auc_score,f1_score,confusion_matrix

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/charts/comprehensive_evaluation'
OLD=ROOT/'results/kalman_updated'; NEW=ROOT/'results/event_f1_tuned'
M=json.loads((NEW/'metrics.json').read_text())
NAMES=list(M); LABELS=['O-RF','O-XGB','K-RF','K-XGB','KT-RF','KT-XGB']
COLORS=['#184e77','#cc6600','#16827c','#97428a','#527b28','#ad3540']
P={n:pd.read_csv(NEW/f'{n}_predictions.csv') for n in NAMES}
PO={n:pd.read_csv(OLD/f'{n}_predictions.csv') for n in NAMES}
GT=pd.read_csv(ROOT/'data/ground_truth.csv'); GT=GT[GT.is_collision.eq(1)]
plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman','DejaVu Serif'],
    'font.size':10,'axes.titlesize':11,'axes.labelsize':10,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
manifest={'source':'Saved outer held-out predictions; no model retraining',
    'labels':dict(zip(LABELS,NAMES)),'figures':{},'trajectory_examples':{}}

def save(fig,name,note):
    fig.savefig(OUT/f'{name}.png',dpi=400,bbox_inches='tight',pad_inches=.10)
    fig.savefig(OUT/f'{name}.svg',bbox_inches='tight',pad_inches=.10)
    plt.close(fig); manifest['figures'][name]=note
def title(fig,text,subtitle):
    fig.suptitle(text,fontsize=15,y=.99)
    fig.text(.5,.935,subtitle,ha='center',fontsize=9,color='#444444')
def bars(ax,values,labels,ylim=None):
    x=np.arange(6); w=.8/len(values)
    for i,(key,v) in enumerate(values.items()):
        ax.bar(x-.4+w/2+i*w,v,w,label=key,edgecolor='black',linewidth=.3)
    ax.set_xticks(x,LABELS); ax.grid(axis='y',alpha=.2); ax.set_axisbelow(True)
    if ylim: ax.set_ylim(*ylim)
    ax.legend(fontsize=8,ncol=len(values)); ax.set_ylabel(labels)

def comparisons():
    fig,axs=plt.subplots(1,2,figsize=(11,4.8))
    for ax,field,ttl in zip(axs,['baseline_event','event'],['Frame-focused extraction','Event-focused extraction']):
        bars(ax,{k.title():[M[n][field][k] for n in NAMES] for k in ['precision','recall','f1']},'Event score',(0,.8)); ax.set_title(ttl)
    title(fig,'Event metrics comparison','Same classifiers and test scores; threshold and merge gap differ. Matching tolerance: 0.5 s.')
    fig.subplots_adjust(top=.81,bottom=.12,wspace=.22)
    save(fig,'event_metrics_comparison','Precision, recall and F1 before/after event-focused extraction.')
    fig,axs=plt.subplots(1,2,figsize=(11,4.8))
    for ax,field,ttl in zip(axs,['baseline_event','event'],['Frame-focused extraction','Event-focused extraction']):
        arr=np.array([[M[n][field][k] for k in ['detected','missed','false_events']] for n in NAMES])
        ax.imshow(arr,cmap='Blues',vmin=0,vmax=100,aspect='auto')
        ax.set_xticks(range(3),['Detected','Missed','False events']); ax.set_yticks(range(6),LABELS); ax.set_title(ttl)
        for i in range(6):
            for j in range(3): ax.text(j,i,str(arr[i,j]),ha='center',va='center',color='white' if arr[i,j]>55 else 'black')
    title(fig,'Event outcome matrices','64 annotated collision rows. Event evaluation has no defined true-negative event count.')
    fig.subplots_adjust(top=.81,bottom=.13,wspace=.25)
    save(fig,'event_outcome_matrices','Count matrices; not binary confusion matrices. Detected + missed = 64.')
    fig,axs=plt.subplots(1,2,figsize=(12,4.8))
    for ax,field,ttl in zip(axs,['baseline_frame','frame'],['Frame-focused extraction','Event-focused extraction']):
        bars(ax,{k.replace('_',' ').title():[M[n][field][k] for n in NAMES] for k in ['precision','recall','f1','roc_auc']},'Frame score',(0,1)); ax.set_title(ttl)
    title(fig,'Frame metrics comparison','ROC-AUC is unchanged by threshold/gap tuning because the classifier probabilities are unchanged.')
    fig.subplots_adjust(top=.81,bottom=.12,wspace=.20)
    save(fig,'frame_metrics_comparison','Frame precision, recall, F1 and ROC-AUC for both extraction strategies.')
    fig,axs=plt.subplots(2,3,figsize=(10,7.5))
    for ax,n,label in zip(axs.flat,NAMES,LABELS):
        cm=confusion_matrix(P[n].true_label,P[n].prediction,labels=[0,1])
        ax.imshow(cm,cmap='Blues',vmin=0,vmax=3359)
        for i in range(2):
            for j in range(2): ax.text(j,i,str(cm[i,j]),ha='center',va='center',color='white' if cm[i,j]>1700 else 'black',fontsize=12)
        ax.set_xticks([0,1],['No collision','Collision']); ax.set_yticks([0,1],['No collision','Collision'])
        ax.set_xlabel('Predicted'); ax.set_ylabel('True'); ax.set_title(label)
        assert cm.tolist()==M[n]['frame']['confusion_matrix']
    title(fig,'Frame confusion matrices','Event-focused thresholds; raw held-out sample counts (3,917 samples per model).')
    fig.subplots_adjust(top=.84,hspace=.5,wspace=.55,bottom=.10)
    save(fig,'frame_confusion_matrices','Binary confusion counts for event-focused thresholds. Rows=true, columns=prediction.')

def visibility():
    fig,axs=plt.subplots(1,2,figsize=(11,4.8)); records=[]
    for ax,preds,ttl in zip(axs,[PO,P],['Frame-focused thresholds','Event-focused thresholds']):
        vals={}
        for flag,label in [(1,'At least one tip missing'),(0,'Both tips visible')]:
            scores=[]
            for name in NAMES:
                sub=preds[name][preds[name].missing_tip.eq(flag)]
                value=f1_score(sub.true_label,sub.prediction,zero_division=0); scores.append(value)
                records.append(dict(strategy=ttl,model=name,missing_tip=flag,n=len(sub),positives=int(sub.true_label.sum()),f1=value))
            vals[label]=scores
        bars(ax,vals,'Frame F1',(0,.7)); ax.set_title(ttl)
    title(fig,'Occlusion-conditioned F1 (detector-missing proxy)','Missing detections are not verified occlusion labels; 1,008 missing-tip and 2,909 both-visible samples.')
    fig.subplots_adjust(top=.81,bottom=.12,wspace=.20)
    save(fig,'occlusion_conditioned_f1','Visibility-conditioned F1 for both threshold strategies. Missingness is only a proxy.')
    pd.DataFrame(records).to_csv(OUT/'visibility_metrics.csv',index=False)
    for conditioned in [False,True]:
        fig,axes=plt.subplots(1,2 if conditioned else 1,figsize=(11,5.2) if conditioned else (7,6))
        axs=np.atleast_1d(axes)
        subsets=[(1,'At least one tip missing'),(0,'Both tips visible')] if conditioned else [(None,'All sampled frames')]
        for ax,(flag,ttl) in zip(axs,subsets):
            for name,label,color in zip(NAMES,LABELS,COLORS):
                d=P[name] if flag is None else P[name][P[name].missing_tip.eq(flag)]
                assert d.true_label.nunique()==2
                fp,tp,_=roc_curve(d.true_label,d.probability)
                ax.plot(fp,tp,label=f'{label}: {roc_auc_score(d.true_label,d.probability):.3f}',color=color,lw=1.25)
            ax.plot([0,1],[0,1],'k--',lw=.7); ax.set_xlim(0,1); ax.set_ylim(0,1)
            ax.set_xlabel('False-positive rate'); ax.set_ylabel('True-positive rate')
            ax.set_title(ttl); ax.legend(title='Model: AUC',fontsize=8,loc='lower right'); ax.grid(alpha=.15)
        title(fig,'Occlusion-conditioned ROC curves' if conditioned else 'ROC curves: six model variants',
            'Held-out scores; ROC is identical before/after extraction tuning. Missingness is an occlusion proxy.')
        fig.subplots_adjust(top=.81,bottom=.13,wspace=.25)
        save(fig,'occlusion_conditioned_roc_curves' if conditioned else 'roc_curves','ROC from saved held-out scores, partitioned by detector visibility.' if conditioned else 'All-frame ROC from saved held-out scores.')

def timeline(full=False):
    lo,hi=(0,768.761) if full else (40,120)
    fig,axs=plt.subplots(2,1,figsize=(12,7.5),sharex=True)
    for ax,directory,ttl in zip(axs,[OLD,NEW],['Frame-focused extraction','Event-focused extraction']):
        rows=[('Human annotations',GT,'start_time_s','end_time_s','black')]
        rows += [(label,pd.read_csv(directory/f'{n}_events.csv'),'start_s','end_s',color) for n,label,color in zip(NAMES,LABELS,COLORS)]
        for yy,(label,events,start,end,color) in enumerate(rows):
            for r in events[(events[end]>=lo)&(events[start]<=hi)].to_dict('records'):
                a,b=max(lo,r[start]),min(hi,r[end])
                if b==a: ax.plot([a,a],[yy-.3,yy+.3],color=color,lw=1)
                else: ax.broken_barh([(a,b-a)],(yy-.25,.5),facecolors=color)
        ax.set_yticks(range(7),[r[0] for r in rows]); ax.set_ylim(6.6,-.65); ax.set_xlim(lo,hi)
        ax.set_title(ttl); ax.grid(axis='x',alpha=.2)
    axs[-1].set_xlabel('Video time (s)')
    title(fig,'Collision event timelines: full recording' if full else 'Collision event timelines: 40-120 s excerpt',
        'Human intervals are unpadded; point events are vertical marks. Bars show actual intervals, not stretched durations.')
    fig.subplots_adjust(top=.85,bottom=.09,left=.14,hspace=.30)
    save(fig,'event_timeline_full' if full else 'event_timeline_example',
        'Full recording, all six models and both extraction strategies.' if full else 'Fixed 40-120 s excerpt; illustrative, not a representative-performance claim.')

def trajectories():
    tr=pd.read_csv(OLD/'trajectories.csv'); raw=pd.read_csv(ROOT/'data/features_and_labels.csv')
    assert np.array_equal(tr.frame_idx,raw.frame_idx)
    for tip in ['TIPL','TIPR']:
        missing=raw[tip+'_present'].eq(0).to_numpy(); groups=tr.group.to_numpy(); candidates=[]
        edges=np.diff(np.r_[False,missing,False].astype(int)); starts=np.flatnonzero(edges==1); ends=np.flatnonzero(edges==-1)
        for a,b in zip(starts,ends):
            if a==0 or b==len(tr) or b-a<2: continue
            if groups[a-1]!=groups[b]: continue
            if not tr.iloc[a:b][tip+'_kf_state'].eq(1).all(): continue
            if tr.iloc[a:b][tip+'_kf_age_s'].max()>1.5: continue
            candidates.append((a,b))
        assert candidates,tip
        # Longest entirely predicted missing run, then earliest. No error-based selection.
        a,b=max(candidates,key=lambda ab:(ab[1]-ab[0],-ab[0]))
        ids=np.flatnonzero((groups==groups[a])&(tr.time_s>=tr.time_s.iloc[a]-2)&(tr.time_s<=tr.time_s.iloc[b]+2))
        d=tr.iloc[ids]; r=raw.iloc[ids]; obs=r[tip+'_present'].eq(1).to_numpy()
        predicted=d[tip+'_kf_state'].eq(1).to_numpy()
        fig,axs=plt.subplots(3,1,figsize=(10,7.5),sharex=True)
        for ax,coord in zip(axs[:2],['x','y']):
            ax.plot(d.time_s,d[tip+'_kf_'+coord],color='#184e77',lw=1.3,label='Kalman state estimate')
            ax.scatter(d.time_s[obs],r.loc[obs,tip+'_'+coord],s=18,color='#16827c',label='Observed YOLO position',zorder=3)
            ax.scatter(d.time_s[predicted],d.loc[predicted,tip+'_kf_'+coord],s=26,marker='x',color='#ad3540',label='Prediction without detection',zorder=4)
            ax.set_ylabel(f'{coord}-position (pixels)')
        axs[2].plot(d.time_s,d[tip+'_kf_variance'],color='#97428a',label='Position covariance trace')
        axs[2].set_ylabel('Covariance trace\n(pixels squared)'); axs[2].set_xlabel('Video time (s)')
        # Shade each missing interval separately; never combine separate gaps into one band.
        step=float(tr.time_s.diff().median())
        for ax in axs:
            for aa,bb in zip(starts,ends):
                if bb-1>=ids[0] and aa<=ids[-1]:
                    ax.axvspan(max(d.time_s.min(),tr.time_s.iloc[aa]-step/2),min(d.time_s.max(),tr.time_s.iloc[bb-1]+step/2),color='#ad3540',alpha=.10)
            ax.set_xlim(d.time_s.min(),d.time_s.max()); ax.grid(alpha=.18)
        axs[0].legend(fontsize=8,ncol=3,loc='upper center',bbox_to_anchor=(.5,1.28))
        title(fig,f'Trajectory bridging example: {tip}',
            f'{b-a} missing samples; last-observation to reacquisition interval {tr.time_s.iloc[b]-tr.time_s.iloc[a-1]:.3f} s. Shading = missing detections.')
        fig.text(.5,.015,'Predicted positions during the gap have no position ground truth; this illustrates extrapolation, not verified tracking accuracy.',ha='center',fontsize=9)
        fig.subplots_adjust(top=.80,bottom=.09,hspace=.13)
        name=f'trajectory_bridging_example_{tip}'
        save(fig,name,'Longest within-group missing run with >=2 samples, all Kalman states predicted and age <=1.5 s; earliest tie.')
        manifest['trajectory_examples'][tip]=dict(start_s=float(tr.time_s.iloc[a]),last_missing_s=float(tr.time_s.iloc[b-1]),
            reacquisition_s=float(tr.time_s.iloc[b]),missing_samples=int(b-a),group=int(groups[a]))
        export=pd.concat([d.reset_index(drop=True),r[[tip+'_present',tip+'_x',tip+'_y']].reset_index(drop=True)],axis=1)
        export.to_csv(OUT/f'{name}_data.csv',index=False)

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    for name in NAMES:
        assert len(P[name])==3917
        assert np.array_equal(P[name].probability,PO[name].probability)
    comparisons(); visibility(); timeline(); timeline(True); trajectories()
    (OUT/'chart_manifest.json').write_text(json.dumps(manifest,indent=2))
    lines=['# Comprehensive evaluation charts','',
      'All figures use saved held-out experimental outputs. PNG files are 400 dpi; SVG files are editable vectors.',
      'O = original 43 features; K = Kalman (53 features); KT = Kalman plus history (85 features). RF = Random Forest; XGB = XGBoost.',
      'The missing-tip subset is only an occlusion proxy. Event outcome matrices omit true negatives because a true-negative event universe is not defined.','']
    lines += [f'- **{n}**: {note}' for n,note in manifest['figures'].items()]
    lines += ['', 'Reproduce: `venv\\Scripts\\python.exe scripts\\22_comprehensive_charts.py`',
      'Data sources: `results/kalman_updated/`, `results/event_f1_tuned/`, `data/ground_truth.csv`, and `data/features_and_labels.csv`.',
      'Trajectory example selection and exact times are in chart_manifest.json; raw plotted subsets are supplied as CSVs.']
    (OUT/'README.md').write_text('\n'.join(lines))
    print(f'Saved {len(manifest["figures"])} chart pairs to {OUT}',flush=True)

if __name__=='__main__': main()
