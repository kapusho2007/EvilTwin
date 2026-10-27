"""Section 1.5 checks requested by the Part-1 audit (XGBoost, modal configuration, 5 folds, seed 42,
same RIDS training subsample as run_train.py sensitivity):
  (c) window length D in {2, 10} s: View A rebuilt from the cleaned frames, folds rebuilt, purge gap = D;
  (d) feature-group ablation: remove one group (timing, size, RSSI, SSID context) or add the
      encryption/configuration group (protected_rate, mfpc).
Usage: python run_part1_checks.py [awid3 rids]. Writes output/cv_part1_<dataset>.csv."""
import sys
import time

import numpy as np
import pandas as pd

import config as C
from pipeline import evaluate as E, features as F, ingestion, preprocess, train as T
from run_train import DEFAULT, NEG_FRAC, SSID_CONTEXT, load

GROUPS = {
    "timing": ["n_frames", "beacon_rate", "beacon_iat_std", "duration_mean", "duration_std"],
    "size": ["length_mean", "length_std"],
    "rssi": ["rssi_mean", "rssi_std", "rssi_sibling_diff"],
    "ssid_context": SSID_CONTEXT,
}


def window_mean(df, W, col, D):
    """mean of a frame field over W(a, tau) = {i : bssid_i = a, tau - D <= t_i < tau} (NaN ignored)."""
    d, gid, t_ms, t0 = F._sorted(df)
    aps = np.unique(d["bssid"].values)
    key = gid * F._SPAN + t_ms
    wg = np.searchsorted(aps, W["ap"].values)
    wt = np.round((W["tau"].values - t0) * 1000).astype(np.int64)
    lo = np.searchsorted(key, wg * F._SPAN + wt - D * 1000, side="left")
    hi = np.searchsorted(key, wg * F._SPAN + wt, side="left")
    x = d[col].to_numpy(float)
    ok = ~np.isnan(x)
    cs = lambda v: np.r_[0.0, np.cumsum(v)]
    n, s = cs(ok)[hi] - cs(ok)[lo], cs(np.where(ok, x, 0))[hi] - cs(np.where(ok, x, 0))[lo]
    return np.where(n > 0, s / np.maximum(n, 1), np.nan)


def build(df, D):
    old = F.WIN_S
    F.WIN_S = D
    try:
        return F.build_windows(df)
    finally:
        F.WIN_S = old


def checks(ds):
    W, S, fid = load(ds)
    df = preprocess.clean(ingestion.load(ds))
    W["protected_rate"] = window_mean(df, W, "protected", F.WIN_S)
    rows = []

    def run(tag, knob, value, Wx, feats, fx, gap):
        res, oof, _ = T.run_cv("XGB", Wx, feats, fx, gap, S, seed=C.RANDOM_STATE, grid=[DEFAULT["XGB"]],
                               neg_frac=NEG_FRAC[ds])
        p = T.pooled(oof, gap)
        res = res.assign(experiment=tag, knob=knob, value=str(value), n_windows=len(Wx), n_pos=int(Wx["y"].sum()),
                         pooled_F1=p["F1"], pooled_FPR=p["FPR"], pooled_R=p["R"])
        rows.append(res)
        print(f"[{ds}] {tag:9s} {knob}={value!s:16s} n={len(Wx)} pos={int(Wx['y'].sum())} | val PR-AUC "
              f"{res.val_PR_AUC.mean():.3f} | test PR-AUC {res.PR_AUC.mean():.3f}±{res.PR_AUC.std():.3f} "
              f"R@1% {res.R_at_FPR1.mean():.3f} F1 {res.F1.mean():.3f}±{res.F1.std():.3f} | pooled F1 {p['F1']:.3f} "
              f"FPR {p['FPR']:.4f}", flush=True)

    run("features", "set", "default", W, F.FEATURES, fid, F.WIN_S)
    for g, cols in GROUPS.items():
        run("features", "set", f"-{g}", W, [f for f in F.FEATURES if f not in cols], fid, F.WIN_S)
    for tag, extra in (("+protected", ["protected_rate"]), ("+mfpc", ["mfpc"]),
                       ("+protected+mfpc", ["protected_rate", "mfpc"])):
        run("features", "set", tag, W, F.FEATURES + extra, fid, F.WIN_S)
    for D in (2, 10):
        Wd = build(df, D)
        run("window", "D", D, Wd, F.FEATURES, E.folds(Wd["tau"].to_numpy(), Wd["y"].to_numpy()), D)
    pd.concat(rows).to_csv(C.OUT_DIR / f"cv_part1_{ds}.csv", index=False)


if __name__ == "__main__":
    t0 = time.time()
    for ds in sys.argv[1:] or ["awid3", "rids"]:
        checks(ds)
    print(f"done in {time.time() - t0:.0f}s")
