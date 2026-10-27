"""2.1 error analysis from the out-of-fold predictions (no training).
Usage: python run_errors.py [awid3 rids]

Writes per dataset:
  oof_all_<ds>.csv        one row per window: keys, features, label, every model's score and alert
  errors_<ds>.csv         FP / FN counts per model, broken down by the window's dominant frame class
                          ((type, subtype) pair, from View B) and by beacon_rate range
  error_examples_<ds>.csv up to 3 FP and 3 FN windows per model (the highest-score FP, the lowest-score FN)
"""
import sys

import numpy as np
import pandas as pd

import config as C
from pipeline import features as F

MODELS = ["RuleFixed", "Rule", "RF", "XGB", "OCSVM", "AE", "LSTM"]
CLASSES = [s for s in F.STEP_FEATURES if s.startswith("is_")]


def main(ds):
    W = pd.read_parquet(C.OUT_DIR / f"windows_{ds}.parquet")
    S = np.load(C.OUT_DIR / f"sequences_{ds}.npz")["S"]
    share = S[..., [F.STEP_FEATURES.index(c) for c in CLASSES]].sum(1)
    J = W[["ap", "tau", "y"] + F.FEATURES].copy()
    J["ap"] = J["ap"].map(lambda a: ":".join(f"{a:012x}"[i:i + 2] for i in range(0, 12, 2)))
    J["dominant_class"] = np.where(share.sum(1) > 0, np.array([c[3:] for c in CLASSES])[share.argmax(1)], "other")
    J["beacon_rate_bin"] = pd.cut(W["beacon_rate"], [-0.1, 0, 5, 11.6, 20, np.inf],
                                  labels=["0", "(0,5]", "(5,11.6]", "(11.6,20]", ">20"]).astype(str)
    for m in MODELS:
        o = pd.read_csv(C.OUT_DIR / f"oof_{m}_{ds}.csv").set_index("idx")
        J[f"{m}_score"], J[f"{m}_alert"], J["fold"] = o["score"], o["pred"], o["fold"]
    J.to_csv(C.OUT_DIR / f"oof_all_{ds}.csv", index_label="window")

    rows, ex = [], []
    for m in MODELS:
        fp = (J[f"{m}_alert"] == 1) & (J.y == 0)
        fn = (J[f"{m}_alert"] == 0) & (J.y == 1)
        for kind, mask in (("FP", fp), ("FN", fn)):
            base = J.y == (0 if kind == "FP" else 1)
            for by in ("dominant_class", "beacon_rate_bin", "ap"):
                g = pd.DataFrame({"n_err": mask.groupby(J[by]).sum(), "n_class": base.groupby(J[by]).sum()})
                g = g[g.n_err > 0]
                for k, r in g.iterrows():
                    rows.append(dict(model=m, error=kind, by=by, value=k, n=int(r.n_err), of=int(r.n_class),
                                     rate=r.n_err / r.n_class))
            sc = J.loc[mask, f"{m}_score"]
            pick = sc.sort_values(ascending=(kind == "FN")).index[:3]
            for i in pick:
                ex.append(dict(model=m, error=kind, window=i, **J.loc[i, ["ap", "tau", "fold", "y", "dominant_class",
                                                                        f"{m}_score"] + F.FEATURES].to_dict()))
        print(f"[{ds}] {m:9s} FP {int(fp.sum()):4d}  FN {int(fn.sum()):4d}", flush=True)
    E = pd.DataFrame(rows)
    E.to_csv(C.OUT_DIR / f"errors_{ds}.csv", index=False)
    pd.DataFrame(ex).to_csv(C.OUT_DIR / f"error_examples_{ds}.csv", index=False)
    sup = ["RuleFixed", "RF", "XGB", "LSTM"]
    shared = ((J[[f"{m}_alert" for m in sup]] == 1).all(1) & (J.y == 0))
    print(f"[{ds}] FP windows shared by {', '.join(sup)}: {int(shared.sum())}, APs "
          f"{J.loc[shared, 'ap'].value_counts().to_dict()}")
    print(E[E.by == "dominant_class"].pivot_table(index=["model", "error"], columns="value", values="n",
                                                    fill_value=0).to_string())


if __name__ == "__main__":
    for ds in sys.argv[1:] or ["awid3", "rids"]:
        main(ds)
