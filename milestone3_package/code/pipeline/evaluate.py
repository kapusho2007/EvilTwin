"""
Stage 5: Evaluation. Dataset-agnostic: receives y, scores and window times only.

  folds()       episode-blocked purged K-fold over windows (Step-8 validation scheme):
                the attack span [first, last positive tau] is cut into K equal time chunks and the
                rest of the timeline into K chunks of equal window count; fold k = attack chunk k U
                benign chunk k. Every fold therefore holds attack and benign windows, and every chunk
                is a contiguous time block (no random shuffling of overlapping windows).
  purge()       drops windows within D of another block: two windows < D apart share frames.
  metrics()     P, R, F1, FPR, FNR, ROC-AUC, PR-AUC, confusion matrix, prevalence pi and the
                constant-classifier baselines (F1 = 2 pi / (1 + pi), PR-AUC = pi).
  block_ci()    95% CI by resampling contiguous D-second blocks (overlapping windows are not
                independent, so a window-level bootstrap would be too narrow).
"""
from __future__ import annotations
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


SESSION_GAP = 600.0                  # s; a capture pause longer than this starts a new session


def sessions(tau: np.ndarray, gap: float = SESSION_GAP) -> np.ndarray:
    """Session id per window: the timeline is cut wherever no window exists for > gap seconds."""
    t = np.sort(np.unique(tau))
    cuts = t[1:][np.diff(t) > gap]
    return np.searchsorted(cuts, tau, side="right")


def folds(tau: np.ndarray, y: np.ndarray, k: int = 5) -> np.ndarray:
    """Fold id 0..k-1 per window: attack chunk k U negative chunk k of every session."""
    pos = tau[y == 1]
    a0, a1 = pos.min(), pos.max()
    span = (tau >= a0) & (tau <= a1)
    f = np.empty(len(tau), int)
    f[span] = np.minimum(((tau[span] - a0) / (a1 - a0 + 1e-9) * k).astype(int), k - 1)
    sid = sessions(tau)
    for s in np.unique(sid):
        rest = np.where(~span & (sid == s))[0]
        order = rest[np.argsort(tau[rest], kind="stable")]
        f[order] = np.arange(len(order)) * k // max(len(order), 1)
    return f


def purge(tau: np.ndarray, keep: np.ndarray, block: np.ndarray, gap: float) -> np.ndarray:
    """keep minus every window whose tau is closer than gap to a window of block."""
    b = np.sort(tau[block])
    i = np.clip(np.searchsorted(b, tau), 1, len(b) - 1)
    d = np.minimum(np.abs(tau - b[i - 1]), np.abs(tau - b[i]))
    return keep & ~block & (d >= gap)


def split(tau, y, fid, k, gap):
    """Fold k = test, fold k+1 = validation (thresholds, early stopping, model selection),
    the remaining folds = train; train and validation are purged against the blocks they touch."""
    K = fid.max() + 1
    te, va = fid == k, fid == (k + 1) % K
    va = purge(tau, va, te, gap)
    tr = purge(tau, purge(tau, ~te & ~va, te, gap), va, gap)
    return tr, va, te


def forward_split(tau, y, frac_tr=0.6, frac_va=0.2, gap=5.0):
    """LABEL-STRATIFIED forward-in-time split. Strictly forward (train < val < test in time), but the
    two boundaries b1 < b2 are placed at quantiles of the POSITIVE taus, so the split uses the attack
    LABELS to decide where to cut. This is necessary when a single short episode would otherwise fall
    entirely in one block; it is NOT a label-independent locked test and NOT generalization to unseen
    devices. A >= gap purge is applied before each boundary. Returns (tr, va, te) boolean masks."""
    tau = np.asarray(tau, float)
    pos = np.sort(tau[np.asarray(y) == 1])
    b1 = float(np.quantile(pos, frac_tr))
    b2 = float(np.quantile(pos, frac_tr + frac_va))
    tr = tau < (b1 - gap)
    va = (tau >= b1) & (tau < (b2 - gap))
    te = tau >= b2
    return tr, va, te


def forward_split_timefrac(tau, frac_tr=0.6, frac_va=0.2, gap=5.0):
    """LABEL-INDEPENDENT forward-in-time split: the two boundaries are placed at fixed fractions of the
    TIME RANGE (no labels used), the pre-specified 60/20/20 chronological design. Feasible only when the
    attack episode spans the boundaries so each block still holds positives (true for AWID3, where the
    episode covers ~half the timeline; NOT for RIDS, where the episode is ~0.2% of the timeline, so a
    pure time split leaves validation and test with zero attack windows). Returns (tr, va, te)."""
    tau = np.asarray(tau, float)
    t0, t1 = tau.min(), tau.max()
    b1 = t0 + frac_tr * (t1 - t0)
    b2 = t0 + (frac_tr + frac_va) * (t1 - t0)
    tr = tau < (b1 - gap)
    va = (tau >= b1) & (tau < (b2 - gap))
    te = tau >= b2
    return tr, va, te


def metrics(y, s, thr) -> dict:
    y = np.asarray(y).astype(int)
    p = (np.asarray(s) >= thr).astype(int)
    tp, fp = int(((p == 1) & (y == 1)).sum()), int(((p == 1) & (y == 0)).sum())
    fn, tn = int(((p == 0) & (y == 1)).sum()), int(((p == 0) & (y == 0)).sum())
    pi = y.mean()
    P = tp / (tp + fp) if tp + fp else 0.0
    R = tp / (tp + fn) if tp + fn else 0.0
    two = len(np.unique(y)) == 2
    return dict(P=P, R=R, F1=2 * P * R / (P + R) if P + R else 0.0,
                FPR=fp / (fp + tn) if fp + tn else 0.0, FNR=fn / (fn + tp) if fn + tp else 0.0,
                ROC_AUC=roc_auc_score(y, s) if two else np.nan,
                PR_AUC=average_precision_score(y, s) if two else np.nan,
                TP=tp, FP=fp, TN=tn, FN=fn, n=len(y), prevalence=pi,
                R_at_FPR1=recall_at_fpr(y, s) if two else np.nan,
                F1_const=2 * pi / (1 + pi), PR_AUC_const=pi)


def recall_at_fpr(y, s, fpr=0.01) -> float:
    """Recall at the threshold that lets through a fraction fpr of the negatives (operating point
    of a SOC alert budget); threshold-free across models, so it is comparable between them."""
    y, s = np.asarray(y), np.asarray(s)
    if y.sum() == 0 or (y == 0).sum() == 0:
        return np.nan
    thr = np.quantile(s[y == 0], 1 - fpr)
    return float((s[y == 1] > thr).mean())


def best_f1_threshold(y, s) -> float:
    """Threshold that maximises F1 on validation scores (never on test)."""
    cand = np.unique(s)
    if len(cand) > 2000:
        cand = np.unique(np.quantile(s, np.linspace(0, 1, 2001)))
    f1 = [metrics(y, s, t)["F1"] for t in cand]
    return float(cand[int(np.argmax(f1))])


def benign_quantile_threshold(y, s, q=0.99) -> float:
    """One-class rule: the q-quantile of benign validation scores (expected FPR = 1 - q)."""
    return float(np.quantile(s[np.asarray(y) == 0], q))


def block_ci(y, s, thr, tau, block: float, stat="F1", B=1000, seed=0):
    """95% percentile CI of a metric by resampling contiguous time blocks of length block."""
    rng = np.random.default_rng(seed)
    b = np.floor((tau - tau.min()) / block).astype(int)
    ids, inv = np.unique(b, return_inverse=True)
    groups = [np.where(inv == i)[0] for i in range(len(ids))]
    vals = []
    for _ in range(B):
        idx = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        if len(np.unique(y[idx])) == 2:
            vals.append(metrics(y[idx], s[idx], thr)[stat])
    return tuple(np.percentile(vals, [2.5, 97.5]))


def _block_counts(y, p, tau, block):
    b = np.floor((tau - tau.min()) / block).astype(int)
    _, inv = np.unique(b, return_inverse=True)
    n = inv.max() + 1
    c = lambda m: np.bincount(inv, weights=m.astype(float), minlength=n)
    return c((p == 1) & (y == 1)), c((p == 1) & (y == 0)), c((p == 0) & (y == 1))


def _f1(tp, fp, fn):
    d = 2 * tp + fp + fn
    return np.where(d > 0, 2 * tp / np.where(d > 0, d, 1), 0.0)


def block_ci_diff(y, pred_a, pred_b, tau, block: float, B=2000, seed=0):
    """Paired block bootstrap of F1(a) - F1(b) on the same out-of-fold windows: point, 95% CI.
    The same resampled blocks are used for both models (paired)."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    A, Bc = _block_counts(y, np.asarray(pred_a), tau, block), _block_counts(y, np.asarray(pred_b), tau, block)
    n = len(A[0])
    W = np.stack([np.bincount(rng.integers(0, n, n), minlength=n) for _ in range(B)])   # B x blocks
    fa, fb = _f1(*(W @ v for v in A)), _f1(*(W @ v for v in Bc))
    ok = (W @ (A[0] + A[2])) > 0                                                         # resample has positives
    point = _f1(*(v.sum() for v in A)) - _f1(*(v.sum() for v in Bc))
    return float(point), *np.percentile((fa - fb)[ok], [2.5, 97.5])
