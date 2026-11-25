"""Fold-isolated training, cached resampling and monotone sigmoid calibration."""
from __future__ import annotations
import hashlib, json, time, warnings
from pathlib import Path
import joblib
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.exceptions import ConvergenceWarning
from imblearn.under_sampling import RandomUnderSampler, EditedNearestNeighbours
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN


def digest(*parts):
    h = hashlib.sha256()
    for part in parts:
        b = part if isinstance(part, bytes) else json.dumps(part, sort_keys=True, default=str).encode()
        h.update(len(b).to_bytes(8, 'big')); h.update(b)
    return h.hexdigest()


def counts(y):
    return {str(i): int(np.sum(y == i)) for i in (0, 1)}


def resample(X, y, ids, method, seed, cache_dir, namespace):
    """Only caller-supplied training rows enter sampler; key binds X, y and order."""
    X = np.ascontiguousarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int8)
    key = digest(namespace, method, seed, list(ids), X.tobytes(), y.tobytes())
    path = Path(cache_dir) / (key + '.joblib')
    if path.exists():
        xr, yr, log = joblib.load(path)
        return xr, yr, dict(log, cache_hit=True)
    before = counts(y)
    if min(before.values()) == 0:
        raise ValueError('Training fold has only one class')
    log = dict(cache_key=key, input_id_hash=digest(list(ids)), before=before,
               method=method, seed=seed, adjustment=None, dtype=str(X.dtype))
    t = time.perf_counter()
    if method in ('none', 'weights'):
        xr, yr = X, y
    elif method == 'undersample':
        xr, yr = RandomUnderSampler(sampling_strategy=1.0, random_state=seed).fit_resample(X, y)
    elif method == 'smoteenn':
        k = min(5, min(before.values()) - 1)
        if k < 1:
            raise ValueError('SMOTE infeasible: fewer than two minority examples')
        if k != 5:
            log['adjustment'] = f'SMOTE k_neighbors reduced from 5 to {k}'
        smote = SMOTE(sampling_strategy=1.0, k_neighbors=k, random_state=seed)
        sampler = SMOTEENN(smote=smote, enn=EditedNearestNeighbours(sampling_strategy='all', n_neighbors=3, kind_sel='all'), random_state=seed)
        xr, yr = sampler.fit_resample(X, y)
        log['after_smote_expected'] = {str(i): max(before.values()) for i in (0, 1)}
        log['fractional_values'] = int(np.count_nonzero((xr > 0) & (xr < 1)))
    else:
        raise ValueError(method)
    log.update(after=counts(yr), seconds=time.perf_counter()-t, cache_hit=False)
    if min(log['after'].values()) == 0:
        raise ValueError(f'Resampling removed a class: {log}')
    # Avoid caching unchanged matrices; synthetic arrays are shared across classifier/grid fits.
    if method not in ('none', 'weights'):
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix('.tmp'); joblib.dump((xr, yr, log), tmp, compress=3); tmp.replace(path)
    return xr, yr, log


def build_model(classifier, config, params, method, seed):
    weight = 'balanced' if method == 'weights' else None
    if classifier == 'logistic':
        return LogisticRegression(C=params['C'], solver='liblinear', max_iter=3000,
                                  class_weight=weight, random_state=seed)
    return RandomForestClassifier(n_estimators=config['forest_trees'],
        max_depth=params['max_depth'], min_samples_leaf=params['min_samples_leaf'],
        max_features='sqrt', class_weight=weight, random_state=seed, n_jobs=1)


def grid_for(classifier, config):
    if classifier == 'logistic':
        return [{'C': v} for v in config['logistic_C']]
    return [{'max_depth': d, 'min_samples_leaf': l} for d in config['forest_depth'] for l in config['forest_min_leaf']]


def fit_checked(model, X, y):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        model.fit(X, y)
    messages = [str(w.message) for w in caught]
    if any(issubclass(w.category, ConvergenceWarning) for w in caught):
        raise RuntimeError('Estimator did not converge: ' + '; '.join(messages))
    return messages


def tune_fit(X, frame, classifier, method, config, seed, cache_dir, namespace):
    """frame contains training rows only, with fixed cv_fold and fp_index."""
    grid = grid_for(classifier, config)
    trials = []; sampler_logs = []; fits = 0
    # Resample once per fold, reused within this model's grid and by exact-key disk cache.
    folds = []
    for fold in range(3):
        tr = frame[frame.cv_fold != fold]; va = frame[frame.cv_fold == fold]
        if len(va) == 0 or va.label.nunique() < 2:
            raise ValueError(f'CV validation fold {fold} lacks two classes')
        xr, yr, log = resample(X[tr.fp_index], tr.label.to_numpy(), tr.structure_id.tolist(),
                              method, seed+fold, cache_dir, namespace)
        sampler_logs.append(dict(log, stage=f'cv_{fold}'))
        folds.append((xr, yr, X[va.fp_index].astype(np.float32), va.label.to_numpy()))
    for params in grid:
        scores = []; trial = dict(params=params, scores=scores, status='complete')
        try:
            for xr, yr, xv, yv in folds:
                model = build_model(classifier, config, params, method, seed)
                fit_checked(model, xr, yr); fits += 1
                scores.append(float(average_precision_score(yv, model.predict_proba(xv)[:, 1])))
            trial['mean_ap'] = float(np.mean(scores))
        except (ValueError, RuntimeError) as e:
            trial.update(status='failed', error=str(e), mean_ap=None)
        trials.append(trial)
    valid = [v for v in trials if v['status'] == 'complete']
    if not valid:
        raise ValueError(f'No feasible configurations: {trials}')
    best = max(valid, key=lambda v: v['mean_ap'])  # first grid entry wins ties
    del folds
    xr, yr, log = resample(X[frame.fp_index], frame.label.to_numpy(), frame.structure_id.tolist(),
                          method, seed+100, cache_dir, namespace)
    sampler_logs.append(dict(log, stage='refit'))
    model = build_model(classifier, config, best['params'], method, seed)
    fit_checked(model, xr, yr)
    return model, dict(selected=best, trials=trials, sampler_logs=sampler_logs,
                       cv_fit_count=fits, final_fit_count=1,
                       class_weights=({str(i): len(yr)/(2*int(np.sum(yr==i))) for i in (0,1)} if method=='weights' else None))


def model_score(model, X):
    return model.decision_function(X) if hasattr(model, 'decision_function') else model.predict_proba(X)[:,1]


class SigmoidCalibrator:
    """Platt-smoothed binomial loss with nonnegative slope; calibration rows only."""
    def fit(self, scores, y):
        x = np.asarray(scores, dtype=float); y = np.asarray(y, dtype=int)
        if len(np.unique(y)) != 2:
            raise ValueError('Calibration partition must contain both classes')
        self.center_ = float(x.mean()); self.scale_ = float(x.std()) or 1.0
        z = (x-self.center_)/self.scale_
        n1 = int(y.sum()); n0 = len(y)-n1
        target = np.where(y==1, (n1+1)/(n1+2), 1/(n0+2))
        def objective(ab):
            eta=ab[0]*z+ab[1]; residual=expit(eta)-target
            return float(np.mean(np.logaddexp(0,eta)-target*eta)), np.array([np.mean(residual*z), np.mean(residual)])
        opt=minimize(objective, [1., float(np.log(n1/n0))], jac=True, method='L-BFGS-B', bounds=[(0,None),(None,None)], options={'ftol':1e-12,'gtol':1e-9,'maxiter':2000})
        if not opt.success: raise RuntimeError('Calibration failed: '+str(opt.message))
        self.slope_, self.intercept_=map(float,opt.x)
        self.diagnostics_=dict(slope=self.slope_, intercept=self.intercept_, center=self.center_, scale=self.scale_, calibration_n=len(y), calibration_positive=n1, constant_mapping=self.slope_==0, objective=float(opt.fun))
        return self
    def predict(self, scores):
        return expit(self.slope_*(np.asarray(scores)-self.center_)/self.scale_+self.intercept_)
