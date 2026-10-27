"""
Stage 4 driver: cross-validated training with model selection inside each fold.

For fold k (pipeline.evaluate.split): fit the normalizer on train only, fit every grid point on
train, keep the one with the best validation PR-AUC, set the threshold on validation (max-F1 for
supervised models, 99th benign percentile for one-class models), then score the locked test fold
once. Dataset-agnostic: receives the window table, the feature list and View B.
"""
from __future__ import annotations
import os
import random
import time
import numpy as np
import pandas as pd
import torch
from imblearn.over_sampling import SMOTE

from pipeline import evaluate as E
from pipeline import models as M
from pipeline.preprocess import make_normalizer

ONE_CLASS = {"OCSVM", "AE"}
VIEW_B = {"LSTM", "HLSTM"}
RAW_INPUT = {"XGB", "Rule"}          # NaN handled natively; no normalizer


def set_seeds(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _inputs(name, W, S, features, tr, va, te, impute=False):
    if name in VIEW_B:
        sc = M.SeqScaler().fit(S[tr])
        if name == "HLSTM":                       # 2.2 fix 3: View B + normalized View A
            nz = make_normalizer().fit(W[features].to_numpy(float)[tr])
            A = np.nan_to_num(nz.transform(W[features].to_numpy(float))).astype(np.float32)
            return (sc, nz), *((sc.transform(S[m]), A[m]) for m in (tr, va, te))
        return sc, sc.transform(S[tr]), sc.transform(S[va]), sc.transform(S[te])
    X = W[features].to_numpy(float)
    if name in RAW_INPUT and not impute:
        return None, X[tr], X[va], X[te]
    if name in RAW_INPUT:                         # same NaN handling in every imbalance arm (1.3)
        from sklearn.impute import SimpleImputer
        im = SimpleImputer(strategy="median", add_indicator=True).fit(X[tr])
        return im, im.transform(X[tr]), im.transform(X[va]), im.transform(X[te])
    nz = make_normalizer().fit(X[tr])
    return nz, nz.transform(X[tr]), nz.transform(X[va]), nz.transform(X[te])


def select(name, Xtr, ytr, Xva, yva, features, seed=0, imbalance="cost", extra=None, grid=None, q=0.99,
           oc_thr="quantile"):
    """Fit every grid point on train, keep the best validation PR-AUC, set the threshold on validation
    (max-F1 for supervised models, benign q-quantile for one-class models)."""
    cls = getattr(M, name)
    kw = dict(seed=seed, **(extra or {}))
    if name in ("XGB", "LSTM", "HLSTM"):
        kw["weighting"] = "cost" if imbalance == "cost" else "none"
    if name == "Rule":
        kw["names"] = list(features)
    trials = []
    for p in grid or cls.GRID:
        m = cls(**p, **kw).fit(Xtr, ytr, Xva, yva)
        trials.append((M._pr(yva, m.score(Xva)), p, m))
    v, p, m = max(trials, key=lambda r: -np.inf if np.isnan(r[0]) else r[0])
    sva = m.score(Xva)
    one_class_q = name in ONE_CLASS and oc_thr == "quantile"       # oc_thr="f1": 2.2 fix 2
    thr = E.benign_quantile_threshold(yva, sva, q) if one_class_q else E.best_f1_threshold(yva, sva)
    return m, p, v, thr, trials


def run_fold(name, W, features, k, fid, gap, S=None, seed=0, imbalance="cost", extra=None, impute=False,
             grid=None, q=0.99, neg_frac=1.0, drop=None, oc_thr="quantile"):
    """One outer fold. imbalance in {'cost', 'none', 'smote'} (supervised models only); impute=True
    gives XGB median-imputed input with is_missing flags instead of native NaN. grid overrides the
    model's grid (one entry = a fixed configuration); q = benign quantile of the one-class threshold.
    drop = windows removed from training only (2.2 fix 1); oc_thr = 'f1' sets one-class thresholds by
    max-F1 on validation (2.2 fix 2)."""
    set_seeds(seed)
    tau, y = W["tau"].to_numpy(), W["y"].to_numpy()
    tr, va, te = E.split(tau, y, fid, k, gap)
    if neg_frac < 1.0:                            # speed only (1.5), train part only, after split and purge:
        sid = E.sessions(tau)                     # keep every attack window and every window of the attack
        att = np.isin(sid, np.unique(sid[y == 1]))  # session; elsewhere keep every step-th 60 s block
        step = int(round(1 / neg_frac))           # (deterministic, so identical across seeds)
        tr = tr & ((y == 1) | att | (np.floor(tau / 60).astype(int) % step == 0))
    if drop is not None:
        tr = tr & ~drop
    prep, Xtr, Xva, Xte = _inputs(name, W, S, features, tr, va, te, impute)
    ytr, yva, yte = y[tr], y[va], y[te]
    if imbalance == "smote":                      # oversampling baseline (1.3), train rows only
        Xtr, ytr = SMOTE(random_state=seed).fit_resample(Xtr, ytr)
    t0 = time.time()
    m, p, v, thr, trials = select(name, Xtr, ytr, Xva, yva, features, seed, imbalance, extra, grid, q, oc_thr)
    sva, ste = m.score(Xva), m.score(Xte)
    res = dict(model=name, fold=k, seed=seed, imbalance=imbalance, params=str(p), val_PR_AUC=v,
               thr=thr, n_train=int(tr.sum()), pos_train=int(y[tr].sum()),
               n_val=int(va.sum()), pos_val=int(yva.sum()), fit_s=time.time() - t0,
               grid=[(str(g), round(float(s), 4)) for s, g, _ in trials],
               val_R_at_FPR1=E.recall_at_fpr(yva, sva), **E.metrics(yte, ste, thr))
    if name == "Rule":
        res["params"] = f"{m.feature} ({'>=' if m.sign > 0 else '<='})"
    oof = pd.DataFrame(dict(idx=np.where(te)[0], tau=tau[te], y=yte, score=ste, pred=(ste >= thr).astype(int), fold=k))
    VAL_OOF.append(pd.DataFrame(dict(idx=np.where(va)[0], tau=tau[va], y=yva, score=sva,
                                     pred=(sva >= thr).astype(int), fold=k)))
    return res, oof, m, prep


VAL_OOF = []                                      # validation predictions of every fold (2.2 reads them)


def run_cv(name, W, features, fid, gap, S=None, seed=0, imbalance="cost", extra=None, keep_models=False,
           impute=False, grid=None, q=0.99, neg_frac=1.0, drop=None, oc_thr="quantile"):
    rows, oofs, fitted = [], [], []
    for k in range(fid.max() + 1):
        r, o, m, prep = run_fold(name, W, features, k, fid, gap, S, seed, imbalance, extra, impute, grid, q, neg_frac,
                                 drop, oc_thr)
        rows.append(r)
        oofs.append(o)
        if keep_models:
            fitted.append((m, prep))
    return pd.DataFrame(rows), pd.concat(oofs, ignore_index=True), fitted


def pooled(oof: pd.DataFrame, gap: float, seed=0) -> dict:
    """Metrics over all out-of-fold test windows (each window tested exactly once), with a
    block-bootstrap 95% CI for F1 (blocks of length 2 * gap, so a block holds whole windows)."""
    y, p, tau = oof["y"].to_numpy(), oof["pred"].to_numpy(), oof["tau"].to_numpy()
    r = E.metrics(y, p, 1)
    r.pop("ROC_AUC"), r.pop("PR_AUC")
    r["F1_lo"], r["F1_hi"] = E.block_ci(y, p, 1, tau, 2 * gap, seed=seed)
    r["FPR_hi"] = 3 / (r["FP"] + r["TN"]) if r["FP"] == 0 else E.block_ci(y, p, 1, tau, 2 * gap, "FPR", seed=seed)[1]
    return r                                  # FPR_hi: rule of three (95%) when FP = 0, else block bootstrap
