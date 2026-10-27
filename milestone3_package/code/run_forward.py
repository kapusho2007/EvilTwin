"""
Strictly forward-in-time WITHIN-EPISODE evaluation (reported alongside, not instead of, the
episode-blocked purged CV in run_train.py). Chronological train -> validation -> locked test with a
>= D purge; preprocessing, model selection and thresholds use the earlier blocks only, and the
locked test block is scored once.

Two split designs are run per dataset (evaluate.forward_split_timefrac / forward_split):
  * label_independent - the two boundaries are placed at fixed 60/20/20 fractions of the TIME RANGE,
    using NO labels. This is the genuinely pre-specified chronological split, but it is feasible only
    when the attack episode spans the boundaries. It works for AWID3 (episode ~52% of the timeline)
    and is NOT FEASIBLE for RIDS (episode ~0.2% of a ~72 h capture: a pure time split leaves
    validation and test with zero attack windows), reported honestly rather than fabricated.
  * label_stratified - the boundaries are quantiles of the POSITIVE-window times, so each block keeps
    attacks. Because the cut points are chosen using the attack labels, this is a label-stratified
    forward-in-time evaluation, NOT a fully independent locked test. It is the only feasible forward
    design for RIDS.

Either design probes temporal generalisation within one episode, not generalisation to unseen
devices or episodes. Output: output/forward_counts.csv, output/forward_eval_<ds>.csv (split column).
"""
from __future__ import annotations
import numpy as np
import pandas as pd

import config as C
from pipeline import evaluate as E, features as F, train as T

MODELS = ["RF", "XGB", "OCSVM", "AE", "LSTM"]


def load(ds):
    W = pd.read_parquet(C.OUT_DIR / f"windows_{ds}.parquet")
    S = np.load(C.OUT_DIR / f"sequences_{ds}.npz")["S"]
    return W, S


SPLITS = [("label_independent", lambda tau, y: E.forward_split_timefrac(tau, gap=F.WIN_S)),
          ("label_stratified", lambda tau, y: E.forward_split(tau, y, gap=F.WIN_S))]


def main():
    counts = []
    for ds in ("awid3", "rids"):
        W, S = load(ds)
        tau, y = W["tau"].to_numpy(), W["y"].to_numpy()
        pos = tau[y == 1]
        episode_frac = 100 * (pos.max() - pos.min()) / (tau.max() - tau.min())
        rows = []
        for split_name, split_fn in SPLITS:
            tr, va, te = split_fn(tau, y)
            for part, m in (("train", tr), ("val", va), ("test", te)):
                counts.append(dict(dataset=ds, split=split_name, part=part, n=int(m.sum()),
                                   pos=int(y[m].sum()), neg=int((m & (y == 0)).sum())))
            feasible = y[tr].sum() > 0 and y[va].sum() > 0 and y[te].sum() > 0
            if tr.any() and va.any() and te.any():          # strict forward ordering
                assert tau[tr].max() < tau[va].min() and tau[va].max() < tau[te].min(), "not forward-ordered"
            print(f"[{ds}] {split_name}: pos train/val/test = {int(y[tr].sum())}/{int(y[va].sum())}/"
                  f"{int(y[te].sum())}  feasible={feasible}", flush=True)
            if not feasible:
                print(f"[{ds}] {split_name} NOT FEASIBLE: the episode is {episode_frac:.1f}% of the "
                      f"timeline, so a pure time split leaves a block with no attack windows. Skipped.")
                continue
            for name in MODELS:
                T.set_seeds(C.RANDOM_STATE)
                prep, Xtr, Xva, Xte = T._inputs(name, W, S, F.FEATURES, tr, va, te)
                m, p, v, thr, _ = T.select(name, Xtr, y[tr], Xva, y[va], F.FEATURES, C.RANDOM_STATE)
                ste = m.score(Xte)
                met = E.metrics(y[te], ste, thr)
                lo, hi = E.block_ci(y[te], ste, thr, tau[te], 2 * F.WIN_S)
                rows.append(dict(dataset=ds, split=split_name, model=name, params=str(p),
                                 val_PR_AUC=round(v, 4), F1=round(met["F1"], 4), F1_lo=round(lo, 4),
                                 F1_hi=round(hi, 4), PR_AUC=round(met["PR_AUC"], 4),
                                 ROC_AUC=round(met["ROC_AUC"], 4), FPR=round(met["FPR"], 5),
                                 FNR=round(met["FNR"], 4), R_at_FPR1=round(met["R_at_FPR1"], 4),
                                 TP=met["TP"], FP=met["FP"], FN=met["FN"], TN=met["TN"], n_test=met["n"]))
                print(f"  [{ds}/{split_name}] {name:5s} F1 {met['F1']:.3f} [{lo:.3f},{hi:.3f}] "
                      f"PR-AUC {met['PR_AUC']:.3f} ROC-AUC {met['ROC_AUC']:.3f} FPR {met['FPR']:.4f} "
                      f"FNR {met['FNR']:.3f}", flush=True)
        pd.DataFrame(rows).to_csv(C.OUT_DIR / f"forward_eval_{ds}.csv", index=False)
    pd.DataFrame(counts).to_csv(C.OUT_DIR / "forward_counts.csv", index=False)
    print("\nwrote forward_counts.csv + forward_eval_<ds>.csv (split column: label_independent / label_stratified)")


if __name__ == "__main__":
    main()
