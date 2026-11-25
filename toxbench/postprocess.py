"""Evaluation-only aggregation, shared bootstrap comparisons and fixed similarity bins."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from .evaluation import metrics, bootstrap_comparisons, create_figures
from .runner import collect, save_json


def evaluate(output, config):
    output=Path(output); collect(output)
    all_ci=[]; similarity=[]; alignment=[]
    for base in sorted((output/'runs').glob('*/*/*/*')):
        if not base.is_dir(): continue
        assay,kind,seed,classifier=base.parts[-4:]
        frames={p.parent.name:pd.read_csv(p,keep_default_na=False) for p in base.glob('*/predictions.csv') if (p.parent/'complete.json').exists()}
        if not frames: continue
        reference=next(iter(frames.values()))
        wide=reference[['structure_id','scaffold','label','nearest_train_tanimoto','constant']].copy()
        for method,frame in frames.items():
            assert np.array_equal(reference[['structure_id','scaffold','label']].values,frame[['structure_id','scaffold','label']].values),f'Alignment failure {base}'
            for state in ('raw','sigmoid'): wide[f'{method}_{state}']=frame[state].to_numpy()
        metadata=dict(assay=assay,split_kind=kind,seed=int(seed),classifier=classifier)
        dest=output/'evaluation'/assay/kind/seed/classifier; dest.mkdir(parents=True,exist_ok=True)
        wide.to_csv(dest/'predictions_wide.csv',index=False)
        ci_key=hashlib.sha256(wide.to_csv(index=False).encode()+str(config['bootstrap_replicates']).encode()+(Path(__file__).parent/'evaluation.py').read_bytes()).hexdigest()
        cp=dest/'bootstrap.csv'; kp=dest/'bootstrap_key.txt'
        if cp.exists() and kp.exists() and kp.read_text()==ci_key:
            ci=pd.read_csv(cp)
        else:
            ci=bootstrap_comparisons(wide,kind,n_bootstrap=config['bootstrap_replicates'],seed=int(hashlib.sha256(f'{assay}/{kind}/{seed}'.encode()).hexdigest()[:8],16))
            for key,value in metadata.items(): ci[key]=value
            ci.to_csv(cp,index=False); kp.write_text(ci_key)
        all_ci.append(ci)
        create_figures(wide,dest,f'{assay} | {kind} | seed {seed} | {classifier}')
        for lo,hi in [(0,.4),(.4,.6),(.6,.8),(.8,1.000001)]:
            sub=wide[(wide.nearest_train_tanimoto>=lo)&(wide.nearest_train_tanimoto<hi)]
            if not len(sub): continue
            for method in frames:
                for state in ('raw','sigmoid'):
                    similarity.append(dict(metadata,method=method,calibration=state,similarity_low=lo,similarity_high=min(hi,1.),**metrics(sub.label.to_numpy(),sub[f'{method}_{state}'].to_numpy(),sub.structure_id.to_numpy())))
        alignment.append(dict(metadata,methods=len(frames),identical_prediction_order=True,test_n=len(wide),test_positive=int(wide.label.sum()),similarity_mean=float(wide.nearest_train_tanimoto.mean()),similarity_median=float(wide.nearest_train_tanimoto.median())))
        print('EVALUATED',assay,kind,seed,classifier,flush=True)
    if all_ci: pd.concat(all_ci,ignore_index=True).to_csv(output/'bootstrap_differences.csv',index=False)
    pd.DataFrame(similarity).to_csv(output/'similarity_metrics.csv',index=False)
    pd.DataFrame(alignment).to_csv(output/'alignment_similarity_audit.csv',index=False)
    save_json(output/'evaluation_status.json',dict(partition_classifier_groups=len(alignment),bootstrap_replicates=config['bootstrap_replicates']))
