"""Tune event F1 on INNER validation, then evaluate unchanged outer OOF scores.

Keeps the earlier six classifiers, folds, features and smoothing fixed. Fits only
the original inner models again because their validation probabilities were not
saved. Outer test labels never select a threshold or gap.
"""
import importlib.util
import json
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.model_selection import StratifiedGroupKFold

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('base',ROOT/'scripts/15_kalman_experiment.py')
base=importlib.util.module_from_spec(spec); spec.loader.exec_module(base)
OUT=ROOT/'results/event_f1_tuned'
THRESHOLDS=np.round(np.arange(.05,.951,.01),2)
GAPS=[.2,.4,.6,.8,1.,1.5,2.]

def extract(d,p,threshold,groups,gap):
    t=d.time_s.to_numpy(); frames=d.frame_idx.to_numpy()
    events=[]
    for group in np.unique(groups):
        pos=np.flatnonzero((groups==group)&(p>=threshold))
        if not len(pos): continue
        runs=np.split(pos,np.flatnonzero(np.diff(t[pos])>gap+1e-6)+1)
        for run in runs:
            a,b=int(run[0]),int(run[-1]); peak=a+int(np.argmax(p[a:b+1]))
            events.append(dict(group=int(group),start_frame=int(frames[a]),end_frame=int(frames[b]),
                start_s=float(t[a]),end_s=float(t[b]),peak_s=float(t[peak]),
                confidence=float(p[peak]),duration_s=float(t[b]-t[a])))
    return events

def score(events,truth):
    """Same maximum-cardinality rule as base.event_metrics, without timing exports."""
    if not events: return 0.,0.,0,0,len(truth)
    starts=np.array([e['start_s'] for e in events])
    ends=np.array([e['end_s'] for e in events])
    peaks=np.array([e['peak_s'] for e in events])
    eligible=(ends[None,:]>=truth.start_time_s.to_numpy()[:,None]-.5)&(
        starts[None,:]<=truth.end_time_s.to_numpy()[:,None]+.5)
    mid=(truth.start_time_s.to_numpy()+truth.end_time_s.to_numpy())/2
    weights=eligible*(1+.001/(1+np.abs(mid[:,None]-peaks[None,:])))
    i,j=linear_sum_assignment(weights,maximize=True)
    tp=int(np.sum(weights[i,j]>0)); fp=len(events)-tp; fn=len(truth)-tp
    return 2*tp/max(1,2*tp+fp+fn),tp/max(1,tp+fp),tp,fp,fn

def select(d,p,groups,truth):
    rows=[]; best=None
    for threshold in THRESHOLDS:
        for gap in GAPS:
            f1,precision,tp,fp,fn=score(extract(d,p,threshold,groups,gap),truth)
            row=dict(threshold=float(threshold),merge_gap_s=gap,event_f1=f1,
                     event_precision=precision,detected=tp,false_events=fp,missed=fn)
            rows.append(row)
            # Predeclared deterministic tie break: precision, shorter gap, higher threshold.
            key=(f1,precision,-gap,threshold)
            if best is None or key>best[0]: best=(key,row)
    return best[1],rows

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    path=ROOT/'data/features_and_labels.csv'
    protocol=json.loads((ROOT/'results/kalman_updated/protocol.json').read_text())
    assert hashlib.sha256(path.read_bytes()).hexdigest()==protocol['input_sha256']
    d=pd.read_csv(path).sort_values('time_s').reset_index(drop=True)
    gt=pd.read_csv(ROOT/'data/ground_truth.csv')
    truth=gt.loc[gt.is_collision==1].copy()
    groups,bounds=base.make_groups(d,gt)
    assert bounds==protocol['boundaries']
    truth_groups=np.searchsorted(bounds,(truth.start_time_s+truth.end_time_s)/2,side='right')
    y=d.label_collision.to_numpy(); t=d.time_s.to_numpy()
    f=base.features(d,groups)
    available=pd.concat([d,f],axis=1)
    previous=json.loads((ROOT/'results/kalman_updated/metrics.json').read_text())
    splits=list(StratifiedGroupKFold(5,shuffle=True,random_state=42).split(d,y,groups))
    results={}; grid=[]
    for name,prior in previous.items():
        X=available[prior['features']].replace([np.inf,-np.inf],np.nan).fillna(-1.)
        kind=name.rsplit('_',1)[1]
        saved=pd.read_csv(ROOT/'results/kalman_updated'/(name+'_predictions.csv'))
        assert np.array_equal(saved.frame_idx,d.frame_idx)
        p=saved.probability.to_numpy(); pred=np.zeros(len(d),int)
        thresholds=np.zeros(len(d)); gaps=np.zeros(len(d)); events=[]; folds=[]
        for fold,(train,test) in enumerate(splits):
            assert np.all(saved.fold.iloc[test].to_numpy()==fold+1)
            def purge(indices,held):
                keep=np.ones(len(indices),bool)
                for group in np.unique(groups[held]):
                    tt=t[groups==group]
                    keep &= (t[indices]<tt.min()-4)|(t[indices]>tt.max()+4)
                return indices[keep]
            train=purge(train,test)
            a,b=next(StratifiedGroupKFold(3,shuffle=True,random_state=100+fold).split(X.iloc[train],y[train],groups[train]))
            fit,val=purge(train[a],train[b]),train[b]
            assert not set(groups[val])&set(groups[test])
            model=base.model(kind,y[fit]); model.fit(X.iloc[fit],y[fit])
            vp=base.smooth(model.predict_proba(X.iloc[val])[:,1],groups[val])
            vd=d.iloc[val].reset_index(drop=True)
            vg=truth.loc[np.isin(truth_groups,np.unique(groups[val]))]
            # Verify original frame-F1 tuning reproduces the saved setting.
            old_threshold=float(max(np.arange(.05,.951,.01),
                key=lambda z:base.f1_score(y[val],vp>=z,zero_division=0)))
            assert abs(old_threshold-prior['folds'][fold]['threshold'])<1e-8
            best,search=select(vd,vp,groups[val],vg)
            grid.extend([dict(model=name,fold=fold+1,**row) for row in search])
            thresholds[test]=best['threshold']; gaps[test]=best['merge_gap_s']
            pred[test]=(p[test]>=best['threshold']).astype(int)
            fold_events=extract(d.iloc[test].reset_index(drop=True),p[test],best['threshold'],groups[test],best['merge_gap_s'])
            for e in fold_events: e['fold']=fold+1
            events.extend(fold_events)
            folds.append(dict(fold=fold+1,selected=best,n_inner_fit=len(fit),n_validation=len(val),
                validation_truth_events=len(vg),old_frame_threshold=old_threshold))
            pd.DataFrame(dict(frame_idx=vd.frame_idx,time_s=vd.time_s,group=groups[val],
                true_label=y[val],probability=vp)).to_csv(OUT/f'{name}_fold{fold+1}_validation.csv',index=False)
            print(name,'fold',fold+1,'selected',best['threshold'],best['merge_gap_s'],flush=True)
        events.sort(key=lambda e:e['start_s'])
        frame=base.metrics(y,p,pred)
        event=base.event_metrics(events,gt)
        results[name]=dict(frame=frame,event=event,event_strict=base.event_metrics(events,gt,0),
            baseline_frame=prior['frame'],baseline_event=prior['event'],folds=folds)
        # There is no calibration/training change; AUC should be unchanged.
        assert abs(frame['roc_auc']-prior['frame']['roc_auc'])<1e-12
        pd.DataFrame(events).to_csv(OUT/(name+'_events.csv'),index=False)
        saved['prediction']=pred; saved['threshold']=thresholds; saved['merge_gap_s']=gaps
        saved.to_csv(OUT/(name+'_predictions.csv'),index=False)
        (OUT/'metrics.json').write_text(json.dumps(results,indent=2))
        print(name,'event F1',round(prior['event']['f1'],3),'->',round(event['f1'],3),flush=True)
    pd.DataFrame(grid).to_csv(OUT/'validation_grid.csv',index=False)
    config=dict(objective='inner-validation event F1',thresholds=THRESHOLDS.tolist(),gaps_s=GAPS,
        tie_break='higher precision, shorter positive-to-positive gap, higher threshold',
        smoothing='fixed causal 3-frame mean within groups',tolerance_s=.5,
        n_outer_folds=5,n_groups=len(np.unique(groups)),input_sha256=protocol['input_sha256'],
        outer_scores_reused=True,inner_models_refitted=True,neural_models_run=False,
        coupled_filter_run=False,appearance_association_run=False)
    (OUT/'protocol.json').write_text(json.dumps(config,indent=2))
    print('Saved',OUT,flush=True)

if __name__=='__main__': main()
