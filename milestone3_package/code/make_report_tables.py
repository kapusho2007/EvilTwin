"""Markdown tables for the report, computed from output/ only (no training).
Usage: python make_report_tables.py > ../report/tables.md"""
import numpy as np
import pandas as pd

import config as C
from pipeline import evaluate as E

MODELS = ["RuleFixed", "Rule", "RF", "XGB", "OCSVM", "AE", "LSTM"]
NAMES = {"RuleFixed": "Post-hoc rule †", "Rule": "Rule (per fold)", "RF": "RF (ref.)",
         "XGB": "XGBoost", "OCSVM": "OC-SVM", "AE": "Autoencoder", "LSTM": "LSTM"}


def per_fold(ds, m):
    o = pd.read_csv(C.OUT_DIR / f"oof_{m}_{ds}.csv")
    r = [E.metrics(g.y, g.score, np.inf) for _, g in o.groupby("fold")]
    return (np.mean([x["PR_AUC"] for x in r]), np.std([x["PR_AUC"] for x in r]),
            np.mean([x["R_at_FPR1"] for x in r]))


def results():
    print("| Model | Data | F1 [95% CI] | P / R | FPR (95% upper) | PR-AUC (fold SD) | R @ FPR 1% | ΔF1 vs post-hoc rule [95% CI] |")
    print("|---|---|---|---|---|---|---|---|")
    for ds in ("awid3", "rids"):
        tag = "AWID3" if ds == "awid3" else "RIDS"
        b = pd.read_csv(C.OUT_DIR / f"baselines_{ds}.csv").set_index("model")
        for m in MODELS:
            r = b.loc[m]
            pr, sd, rec = per_fold(ds, m)
            d = "–" if m == "RuleFixed" else f"{r.dF1_vs_RuleFixed:+.3f} [{r.dF1_vs_RuleFixed_lo:+.3f}, {r.dF1_vs_RuleFixed_hi:+.3f}]"
            print(f"| {NAMES[m]} | {tag} | {r.F1:.3f} [{r.F1_lo:.3f}, {r.F1_hi:.3f}] | {r.P:.3f} / {r.R:.3f} | "
                  f"{r.FPR:.4f} ({r.FPR_hi:.4f}) | {pr:.3f} ({sd:.3f}) | {rec:.3f} | {d} |")
        s = b.loc["Session"]
        print(f"| Constant / session flag | {tag} | {b.F1_const.iloc[0]:.3f} / {s.F1:.3f} | | | π = {b.prevalence.iloc[0]:.3f} | | |")


def selected():
    print("\n| Model | AWID3 grid point per fold | RIDS grid point per fold |\n|---|---|---|")
    c = {ds: pd.read_csv(C.OUT_DIR / f"cv_main_{ds}.csv") for ds in ("awid3", "rids")}
    for m in ("XGB", "LSTM", "OCSVM", "AE"):
        cell = lambda ds: "; ".join(c[ds][c[ds].model == m].params.str.replace("'", "").str.strip("{}"))
        print(f"| {m} | {cell('awid3')} | {cell('rids')} |")


def folds():
    print("\n| Dataset | Part | Windows | Attack windows | Prevalence |\n|---|---|---|---|---|")
    for ds in ("awid3", "rids"):
        f = pd.read_csv(C.OUT_DIR / f"folds_{ds}.csv")
        for part in ("train", "val", "test"):
            g = f[f.part == part]
            print(f"| {ds.upper() if ds == 'rids' else 'AWID3'} | {part} | {g.n.min():,}–{g.n.max():,} | "
                  f"{g.pos.min()}–{g.pos.max()} | {g.prevalence.min():.3f}–{g.prevalence.max():.3f} |")


def imbalance():
    print("\n| Model | Policy | AWID3 val PR-AUC | AWID3 val R@1% | AWID3 test F1 | RIDS val PR-AUC | RIDS val R@1% | RIDS test F1 |")
    print("|---|---|---|---|---|---|---|---|")
    c = {ds: pd.read_csv(C.OUT_DIR / f"cv_imbalance_{ds}.csv") for ds in ("awid3", "rids")}
    for (m, pol), _ in c["awid3"].groupby(["model", "imbalance"], sort=False):
        cells = []
        for ds in ("awid3", "rids"):
            g = c[ds][(c[ds].model == m) & (c[ds].imbalance == pol)]
            cells += [f"{g.val_PR_AUC.mean():.3f} ± {g.val_PR_AUC.std():.3f}", f"{g.val_R_at_FPR1.mean():.3f}",
                      f"{g.F1.mean():.3f} ± {g.F1.std():.3f}"]
        name = {"none": "none", "cost": "cost weights (ours)", "smote": "SMOTE"}[pol]
        print(f"| {'XGBoost' if m == 'XGB' else m} | {name} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    results()
    selected()
    folds()
    if (C.OUT_DIR / "cv_imbalance_rids.csv").exists():
        imbalance()
