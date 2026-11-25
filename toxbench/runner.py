"""Resumable execution of fixed-partition benchmark (test labels used in evaluation only)."""
from __future__ import annotations
import os
for _var in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_var]='1'
os.environ.setdefault('MPLCONFIGDIR',str(__import__('pathlib').Path('tmp/matplotlib').resolve()))
import argparse, hashlib, importlib.metadata, json, platform, time, traceback
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import joblib
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from threadpoolctl import threadpool_limits
from sklearn import set_config
set_config(working_memory=128)
from .modeling import digest, tune_fit, model_score, SigmoidCalibrator


def save_json(path, data):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp'); tmp.write_text(json.dumps(data,indent=2,default=str,allow_nan=False)); tmp.replace(path)


def versions():
    return {p:importlib.metadata.version(p) for p in ['numpy','pandas','scipy','scikit-learn','imbalanced-learn','rdkit','joblib']}


def fingerprint_similarity(X, train_idx, query_idx):
    # RDKit native bulk Tanimoto avoids dense query x training x 2048 intermediates.
    def bitvector(row):
        v=DataStructs.ExplicitBitVect(2048)
        v.SetBitsFromList(np.flatnonzero(row).tolist()); return v
    train=[bitvector(X[i]) for i in train_idx]
    return np.array([max(DataStructs.BulkTanimotoSimilarity(bitvector(X[i]),train)) for i in query_idx])


def run_partition(root, output, config, assay, kind, seed):
    from .evaluation import metrics
    root=Path(root); output=Path(output)
    manifest=root/'partitions'/assay/kind/f'{seed}.csv'
    frame=pd.read_csv(manifest,keep_default_na=False)
    X=np.load(root/'data/processed/fingerprints.npy',mmap_mode='r')
    tr=frame[frame.partition=='train']; ca=frame[frame.partition=='calibration']; te=frame[frame.partition=='test']
    assert set(tr.structure_id).isdisjoint(ca.structure_id) and set(tr.structure_id).isdisjoint(te.structure_id) and set(ca.structure_id).isdisjoint(te.structure_id)
    if kind=='scaffold':
        assert set(tr.scaffold).isdisjoint(ca.scaffold) and set(tr.scaffold).isdisjoint(te.scaffold) and set(ca.scaffold).isdisjoint(te.scaffold)
    code={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((root/'toxbench').glob('*.py')) if p.name in ('modeling.py','runner.py')}
    namespace=digest(config,versions(),code,manifest.read_bytes(),(root/'data/processed/fingerprints.npy').read_bytes())
    base=output/'runs'/assay/kind/str(seed); base.mkdir(parents=True,exist_ok=True)
    metadata=dict(assay=assay,split_kind=kind,seed=seed,namespace=namespace,train_n=len(tr),calibration_n=len(ca),test_n=len(te),train_prevalence=float(tr.label.mean()))
    sim_path=base/'similarity.npz'
    simkey=digest(manifest.read_bytes(),namespace)
    if sim_path.exists() and str(np.load(sim_path)['key'])==simkey:
        sim=np.load(sim_path)['values']
    else:
        sim=fingerprint_similarity(X,tr.fp_index.to_numpy(),te.fp_index.to_numpy())
        np.savez_compressed(sim_path,key=simkey,values=sim)
    for classifier in config['classifiers']:
        for method in config['balancing']:
            dest=base/classifier/method; dest.mkdir(parents=True,exist_ok=True)
            done=dest/'complete.json'
            if done.exists() and json.loads(done.read_text()).get('namespace')==namespace and all((dest/f).exists() for f in ['predictions.csv','calibration_predictions.csv','training.json','model.joblib']):
                print(f'RESUME {assay} {kind} {seed} {classifier} {method}',flush=True); continue
            done.unlink(missing_ok=True)
            started=time.perf_counter()
            try:
                with threadpool_limits(limits=1):
                    model,train_log=tune_fit(X,tr,classifier,method,config,seed,output/'cache',namespace)
                    xc=X[ca.fp_index].astype(np.float32); xt=X[te.fp_index].astype(np.float32)
                    calibrator=SigmoidCalibrator().fit(model_score(model,xc),ca.label.to_numpy())
                    raw=model.predict_proba(xt)[:,1]; calibrated=calibrator.predict(model_score(model,xt))
                pred=te[['structure_id','scaffold','label']].copy()
                pred['raw']=raw; pred['sigmoid']=calibrated; pred['nearest_train_tanimoto']=sim
                pred['constant']=float(tr.label.mean())
                pred.to_csv(dest/'predictions.csv',index=False)
                calpred=ca[['structure_id','label']].copy(); calpred['score']=model_score(model,xc); calpred['raw']=model.predict_proba(xc)[:,1]; calpred['sigmoid']=calibrator.predict(calpred.score)
                calpred.to_csv(dest/'calibration_predictions.csv',index=False)
                joblib.dump(dict(model=model,calibrator=calibrator),dest/'model.joblib',compress=3)
                rows=[]
                for state,p in [('raw',raw),('sigmoid',calibrated),('constant',pred.constant.to_numpy())]:
                    rows.append(dict(metadata,classifier=classifier,method=method,calibration=state,**metrics(te.label.to_numpy(),p,te.structure_id.to_numpy())))
                pd.DataFrame(rows).to_csv(dest/'metrics.csv',index=False)
                # Rank preservation audit uses the same IDs for deterministic ties.
                a=np.lexsort((te.structure_id.to_numpy(),-raw)); b=np.lexsort((te.structure_id.to_numpy(),-calibrated))
                save_json(dest/'training.json',dict(train_log,calibration=calibrator.diagnostics_,calibration_id_hash=digest(ca.structure_id.tolist()),training_id_hash=digest(tr.structure_id.tolist()),test_id_hash=digest(te.structure_id.tolist()),rank_order_identical=bool(np.array_equal(a,b))))
                (dest/'failure.json').unlink(missing_ok=True)
                save_json(done,dict(metadata,classifier=classifier,method=method,seconds=time.perf_counter()-started,cv_fits=train_log['cv_fit_count'],final_fits=1,completed_utc=pd.Timestamp.now(tz='UTC').isoformat()))
                print(f'DONE {assay} {kind} {seed} {classifier} {method} {time.perf_counter()-started:.1f}s',flush=True)
            except Exception as e:
                save_json(dest/'failure.json',dict(metadata,classifier=classifier,method=method,error=repr(e),traceback=traceback.format_exc()))
                print(f'FAILED {assay} {kind} {seed} {classifier} {method}: {e}',flush=True)
    return metadata


def collect(output):
    from .evaluation import metrics
    output=Path(output)
    completed=sorted((output/'runs').glob('*/*/*/*/*/complete.json'))
    records=[]
    for path in completed:
        meta=json.loads(path.read_text()); pred_path=path.parent/'predictions.csv'
        if not pred_path.exists(): raise ValueError(f'Missing predictions for completed run {path}')
        pred=pd.read_csv(pred_path)
        for state in ('raw','sigmoid','constant'):
            records.append(dict(meta,calibration=state,**metrics(pred.label.to_numpy(),pred[state].to_numpy(),pred.structure_id.to_numpy())))
    rows=pd.DataFrame(records)
    output.mkdir(exist_ok=True,parents=True); rows.to_csv(output/'metrics.csv',index=False)
    if len(rows):
        numeric=['average_precision','roc_auc','brier','log_loss','precision_top10','recall_top10','prevalence']
        summary=rows.groupby(['assay','split_kind','classifier','method','calibration'])[numeric].agg(['mean','std','count'])
        summary.columns=['_'.join(c) for c in summary.columns]; summary.reset_index().to_csv(output/'summary.csv',index=False)
    done=[json.loads(p.read_text()) for p in (output/'runs').glob('*/*/*/*/*/complete.json')]
    failures=list((output/'runs').glob('*/*/*/*/*/failure.json'))
    save_json(output/'execution_status.json',dict(completed_models=len(done),cv_fits=sum(d['cv_fits'] for d in done),final_fits=len(done),sum_model_seconds=sum(d['seconds'] for d in done),failure_files=len(failures)))
    return rows


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('command',choices=['prepare','run','summarize','evaluate'])
    ap.add_argument('--root',type=Path,default=Path('.')); ap.add_argument('--config',type=Path,default=Path('configs/full.json')); ap.add_argument('--output',type=Path,default=Path('outputs/full')); ap.add_argument('--workers',type=int); args=ap.parse_args()
    root=args.root.resolve(); config=json.loads(args.config.read_text()); output=args.output.resolve()
    if args.command=='prepare':
        from .data import prepare
        from .splits import make_partitions
        prepare(root); make_partitions(root,seeds=config['seeds']); return
    if args.command=='summarize': collect(output); return
    if args.command=='evaluate':
        from .postprocess import evaluate
        evaluate(output,config); return
    output.mkdir(parents=True,exist_ok=True)
    save_json(output/'config.json',config); save_json(output/'environment.json',dict(python=platform.python_version(),platform=platform.platform(),versions=versions(),cpu_count=os.cpu_count()))
    jobs=[(a,k,s) for a in config['assays'] for k in config['split_kinds'] for s in config['seeds']]
    start=time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers or config['workers']) as ex:
        futures=[ex.submit(run_partition,str(root),str(output),config,*job) for job in jobs]
        for future in as_completed(futures):
            future.result(); collect(output)
    collect(output)
    elapsed=time.perf_counter()-start
    with (output/'invocations.jsonl').open('a') as f: f.write(json.dumps(dict(seconds=elapsed,jobs=len(jobs),utc=pd.Timestamp.now(tz='UTC').isoformat()))+'\n')
    print(f'Execution wall time: {elapsed:.1f}s',flush=True)

if __name__=='__main__': main()
