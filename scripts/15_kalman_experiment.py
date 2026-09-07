"""Occlusion-aware experiment on cached YOLO detections. Original outputs are preserved."""
import json
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, confusion_matrix
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results' / 'kalman_updated'
META = {'frame_idx', 'time_s', 'label_collision', 'label_near_miss', 'label_severity'}
CONFIG = dict(seed=42, folds=5, group_seconds=30, boundary_guard_s=4,
              max_missing_s=1.5, acceleration_std=100, measurement_std=8,
              window_s=2, smoothing_frames=3, merge_gap_s=0.4,
              event_tolerance_s=0.5, rf_trees=300, xgb_trees=250)

class TipKalman:
    """State [x,y,vx,vy], pixel/second units; unavailable coordinates are NaN."""
    def __init__(self, max_missing=1.5):
        self.x = None
        self.P = None
        self.t = self.seen = None
        self.missing = 0
        self.max_missing = max_missing

    def step(self, t, measurement=None, confidence=1.):
        dt = 0 if self.t is None else t - self.t
        if dt < 0:
            raise ValueError('Timestamps must increase')
        self.t = t
        if self.seen is not None and t - self.seen > self.max_missing:
            self.x = None
        if self.x is not None:
            F = np.eye(4); F[0, 2] = F[1, 3] = dt
            G = np.array([[dt*dt/2, 0], [0, dt*dt/2], [dt, 0], [0, dt]])
            self.x = F @ self.x
            self.P = F @ self.P @ F.T + CONFIG['acceleration_std']**2 * G @ G.T
        if measurement is not None:
            R = np.eye(2) * CONFIG['measurement_std']**2 / max(confidence, .1)
            if self.x is None:
                self.x = np.array([*measurement, 0., 0.])
                self.P = np.diag([R[0, 0], R[1, 1], 10000., 10000.])
            else:
                H = np.eye(2, 4)
                K = np.linalg.solve(H @ self.P @ H.T + R, H @ self.P).T
                self.x += K @ (np.asarray(measurement) - H @ self.x)
                A = np.eye(4) - K @ H
                self.P = A @ self.P @ A.T + K @ R @ K.T
            self.seen = t; self.missing = 0
            state = 0
        else:
            self.missing += 1
            state = 1 if self.x is not None else 2
        age = t - self.seen if self.seen is not None else np.nan
        values = list(self.x) if self.x is not None else [np.nan]*4
        uncertainty = np.trace(self.P[:2, :2]) if self.x is not None else np.nan
        return values + [state, self.missing, age, uncertainty]

def make_groups(d, gt):
    # Move boundaries away from complete padded annotations plus history guard.
    boundaries = []
    intervals = [(r.start_time_s-4.5, r.end_time_s+4.5) for r in gt.itertuples() if r.is_collision]
    for candidate in np.arange(30., d.time_s.max(), 30.):
        while True:
            ends = [b for a, b in intervals if a <= candidate <= b]
            if not ends: break
            candidate = max(ends) + .001
        if candidate < d.time_s.max() and (not boundaries or candidate > boundaries[-1]+5):
            boundaries.append(candidate)
    return np.searchsorted(boundaries, d.time_s.to_numpy(), side='right'), boundaries

def features(d, groups):
    all_rows = []
    for group in np.unique(groups):
        sub = d.loc[groups == group]
        trackers = {tip: TipKalman() for tip in ('TIPL', 'TIPR')}
        rows = []
        for r in sub.to_dict('records'):
            out = {}
            for tip in trackers:
                measurement = [r[tip+'_x'], r[tip+'_y']] if r[tip+'_present'] else None
                vals = trackers[tip].step(r['time_s'], measurement, r[tip+'_conf'])
                out.update({tip+'_kf_'+k: v for k, v in zip(
                    ['x', 'y', 'vx', 'vy', 'state', 'missing_frames', 'age_s', 'variance'], vals)})
            rows.append(out)
        f = pd.DataFrame(rows, index=sub.index)
        dt = sub.time_s.diff()
        for tip in trackers:
            f[tip+'_kf_speed'] = np.hypot(f[tip+'_kf_vx'], f[tip+'_kf_vy'])
            f[tip+'_kf_acceleration'] = f[tip+'_kf_speed'].diff()/dt
        f['kf_distance'] = np.hypot(f.TIPL_kf_x-f.TIPR_kf_x, f.TIPL_kf_y-f.TIPR_kf_y)
        f['kf_distance_delta'] = f.kf_distance.diff()
        f['kf_distance_rate'] = f.kf_distance.diff()/dt
        f['kf_relative_speed'] = np.hypot(f.TIPL_kf_vx-f.TIPR_kf_vx, f.TIPL_kf_vy-f.TIPR_kf_vy)
        temporal = {}
        for col in ['kf_distance', 'kf_distance_rate', 'TIPL_kf_speed', 'TIPR_kf_speed',
                    'TIPL_kf_variance', 'TIPR_kf_variance', 'TIPL_kf_state', 'TIPR_kf_state']:
            s = pd.Series(f[col].to_numpy(), index=pd.to_timedelta(sub.time_s, unit='s'))
            roll = s.rolling('2s', min_periods=1)
            for stat in ['mean', 'min', 'max', 'std']:
                temporal[col+'_2s_'+stat] = getattr(roll, stat)().to_numpy()
        all_rows.append(pd.concat([f, pd.DataFrame(temporal, index=f.index)], axis=1))
    return pd.concat(all_rows).sort_index()

def smooth(p, groups):
    s = pd.Series(p)
    return s.groupby(groups).transform(lambda x: x.rolling(3, min_periods=1).mean()).to_numpy()

def model(kind, y):
    if kind == 'RF':
        return RandomForestClassifier(n_estimators=300, max_depth=10, min_samples_leaf=3,
                                      class_weight='balanced', random_state=42, n_jobs=4)
    return XGBClassifier(n_estimators=250, max_depth=4, learning_rate=.05, subsample=.9,
                         colsample_bytree=.9, min_child_weight=2, random_state=42,
                         scale_pos_weight=float((y==0).sum()/max(1,(y==1).sum())), n_jobs=4)

def metrics(y, p, pred):
    return dict(n=int(len(y)), positives=int(np.sum(y)),
                precision=float(precision_score(y,pred,zero_division=0)),
                recall=float(recall_score(y,pred,zero_division=0)),
                f1=float(f1_score(y,pred,zero_division=0)),
                roc_auc=float(roc_auc_score(y,p)) if len(np.unique(y))==2 else None,
                average_precision=float(average_precision_score(y,p)) if np.sum(y) else None,
                confusion_matrix=confusion_matrix(y,pred,labels=[0,1]).tolist())

def extract_events(d, p, pred, groups):
    events = []
    for group in np.unique(groups):
        positives = np.flatnonzero((groups==group) & (pred==1))
        runs = []
        for i in positives:
            if runs and d.time_s.iloc[i]-d.time_s.iloc[runs[-1][-1]] <= CONFIG['merge_gap_s']+1e-6:
                runs[-1].append(i)
            else: runs.append([i])
        for run in runs:
            a,b = run[0],run[-1]
            peak = a+int(np.argmax(p[a:b+1]))
            events.append(dict(group=int(group), start_frame=int(d.frame_idx.iloc[a]),
                end_frame=int(d.frame_idx.iloc[b]), start_s=float(d.time_s.iloc[a]),
                end_s=float(d.time_s.iloc[b]), peak_s=float(d.time_s.iloc[peak]),
                confidence=float(p[peak]), duration_s=float(d.time_s.iloc[b]-d.time_s.iloc[a])))
    return events

def event_metrics(events, gt, tolerance=.5):
    truth = gt.loc[gt.is_collision==1].reset_index(drop=True)
    score = np.zeros((len(truth),len(events)))
    for i,r in enumerate(truth.itertuples()):
        for j,e in enumerate(events):
            if e['end_s'] >= r.start_time_s-tolerance and e['start_s'] <= r.end_time_s+tolerance:
                score[i,j] = 1 + .001/(1+abs(e['peak_s']-(r.start_time_s+r.end_time_s)/2))
    ii,jj = linear_sum_assignment(score,maximize=True)
    matches = [(int(i),int(j)) for i,j in zip(ii,jj) if score[i,j]>0]
    start_errors = [events[j]['start_s']-float(truth.iloc[i].start_time_s) for i,j in matches]
    end_errors = [events[j]['end_s']-float(truth.iloc[i].end_time_s) for i,j in matches]
    peak_errors = [events[j]['peak_s']-float((truth.iloc[i].start_time_s+truth.iloc[i].end_time_s)/2) for i,j in matches]
    tp=len(matches); fp=len(events)-tp; fn=len(truth)-tp
    return dict(detected=tp, missed=fn, false_events=fp, annotated_events=len(truth),
        precision=tp/max(1,tp+fp), recall=tp/max(1,tp+fn), f1=2*tp/max(1,2*tp+fp+fn),
        start_mae_s=float(np.mean(np.abs(start_errors))) if matches else None,
        end_mae_s=float(np.mean(np.abs(end_errors))) if matches else None,
        peak_midpoint_mae_s=float(np.mean(np.abs(peak_errors))) if matches else None,
        matches=[dict(truth_index=i,prediction_index=j,start_error_s=a,end_error_s=b,
                      peak_midpoint_error_s=c) for (i,j),a,b,c in zip(matches,start_errors,end_errors,peak_errors)])

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    path=ROOT/'data/features_and_labels.csv'
    d=pd.read_csv(path).sort_values('time_s').reset_index(drop=True)
    gt=pd.read_csv(ROOT/'data/ground_truth.csv')
    y=d.label_collision.to_numpy(); t=d.time_s.to_numpy()
    groups,bounds=make_groups(d,gt)
    f=features(d,groups)
    pd.concat([d[['frame_idx','time_s']],pd.Series(groups,name='group'),f],axis=1).to_csv(OUT/'trajectories.csv',index=False)
    original=[c for c in d if c not in META]
    assert len(original)==43
    raw=[c for c in original if c.endswith(('_present','_x','_y','_w','_h','_conf','_area')) or c in
         ['dist_TIPL_TIPR','dist_TIPL_CIRCLE','dist_TIPR_CIRCLE','iou_TIPL_TIPR',
          'mask_iou_TIPL_TIPR','mask_overlap_area_TIPL_TIPR','mask_overlap_frac_TIPL_TIPR']]
    current=[c for c in f if '_2s_' not in c]
    sets={'Original43':d[original], 'Kalman':pd.concat([d[raw],f[current]],axis=1),
          'KalmanTemporal':pd.concat([d[raw],f],axis=1)}
    splits=list(StratifiedGroupKFold(5,shuffle=True,random_state=42).split(d,y,groups))
    summary={}
    for variant,X in sets.items():
        X=X.replace([np.inf,-np.inf],np.nan).fillna(-1.)
        for kind in ['RF','XGB']:
            name=variant+'_'+kind
            p=np.full(len(d),np.nan); pred=np.zeros(len(d),int); folds=np.zeros(len(d),int)
            fold_stats=[]
            for fold,(train,test) in enumerate(splits):
                # Guard both sides of each boundary for legacy features computed before splitting.
                def purge(indices, held):
                    keep=np.ones(len(indices),bool)
                    for group in np.unique(groups[held]):
                        tt=t[groups==group]
                        keep &= (t[indices]<tt.min()-4)|(t[indices]>tt.max()+4)
                    return indices[keep]
                train=purge(train,test)
                a,b=next(StratifiedGroupKFold(3,shuffle=True,random_state=100+fold).split(X.iloc[train],y[train],groups[train]))
                fit,val=purge(train[a],train[b]),train[b]
                assert not set(groups[train]) & set(groups[test])
                assert not set(groups[fit]) & set(groups[val])
                m=model(kind,y[fit]); m.fit(X.iloc[fit],y[fit])
                vp=smooth(m.predict_proba(X.iloc[val])[:,1],groups[val])
                thresholds=np.arange(.05,.951,.01)
                threshold=float(max(thresholds,key=lambda z:f1_score(y[val],vp>=z,zero_division=0)))
                m=model(kind,y[train]); m.fit(X.iloc[train],y[train])
                p[test]=smooth(m.predict_proba(X.iloc[test])[:,1],groups[test])
                pred[test]=(p[test]>=threshold).astype(int); folds[test]=fold+1
                fold_stats.append(dict(fold=fold+1,threshold=threshold,n_train=len(train),
                    n_inner_fit=len(fit),n_inner_validation=len(val),**metrics(y[test],p[test],pred[test])))
                print(name,'fold',fold+1,'F1',round(fold_stats[-1]['f1'],3),flush=True)
            assert np.isfinite(p).all()
            events=extract_events(d,p,pred,groups)
            missing=(d.TIPL_present.eq(0)|d.TIPR_present.eq(0)).to_numpy()
            summary[name]=dict(feature_count=X.shape[1], features=list(X.columns),frame=metrics(y,p,pred),
                missing_tip=metrics(y[missing],p[missing],pred[missing]),
                both_visible=metrics(y[~missing],p[~missing],pred[~missing]),
                event=event_metrics(events,gt),event_strict=event_metrics(events,gt,0),folds=fold_stats)
            pd.DataFrame(events).to_csv(OUT/(name+'_events.csv'),index=False)
            pd.DataFrame(dict(frame_idx=d.frame_idx,time_s=t,group=groups,fold=folds,true_label=y,
                probability=p,prediction=pred,missing_tip=missing.astype(int))).to_csv(OUT/(name+'_predictions.csv'),index=False)
            (OUT/'metrics.json').write_text(json.dumps(summary,indent=2))
    provenance=dict(config=CONFIG,boundaries=bounds,n_groups=len(np.unique(groups)),n_frames=len(d),
        positive_frames=int(y.sum()),annotation_rows=len(gt),collision_events=int(gt.is_collision.sum()),
        sample_interval_s=float(np.median(np.diff(t))),input_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        detector_reused=True,deep_temporal_models_run=False)
    (OUT/'protocol.json').write_text(json.dumps(provenance,indent=2))
    print('Saved',OUT,flush=True)

if __name__=='__main__': main()
