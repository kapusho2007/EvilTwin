"""Stages 4-5 driver: cross-validated training and evaluation on View A / View B.
Usage: python run_train.py folds|main|rule_gain|transfer|imbalance|sensitivity|random [awid3 rids]

  folds        per-fold composition table and the session-confound check (C1)
  main         every model, seed 42, 5 outer folds; saves fold models to output/models/<dataset>/
  rule_gain    F1 gain over the one-feature rule (block-bootstrap CI), session baseline, attack-session-only metrics
  transfer     train on one dataset, test on the other (H3)
  imbalance    1.3: XGB {none, cost weights, SMOTE}, LSTM {none, cost weights}
  sensitivity  1.5: one-knob parameter sweeps, seed x fold stability, window length D, feature groups
  random       1.4: stratified random split (optimistic upper bound, no purge)
Writes output/cv_<experiment>_<dataset>.csv (one row per fold) and oof_<model>_<dataset>.csv."""
import sys
import time
import warnings

import joblib
import numpy as np
import pandas as pd

import config as C
from pipeline import evaluate as E, features as F, ingestion, preprocess, train as T

# rssi_sibling_diff and ssid_newcomer are undefined (all NaN) on RIDS, which has no second AP per SSID:
# the imputer drops the all-NaN column and nanmedian/nanvar warn. Expected, so silenced here only.
warnings.filterwarnings("ignore", message="Skipping features without any observed values")
warnings.filterwarnings("ignore", message="All-NaN slice")
warnings.filterwarnings("ignore", message="Degrees of freedom <= 0")
warnings.filterwarnings("ignore", message="Mean of empty slice")

MODELS = ["Rule", "RF", "XGB", "OCSVM", "AE", "LSTM"]
SSID_CONTEXT = ["n_bssid_same_ssid", "rssi_sibling_diff", "ssid_newcomer"]
SAME_SIGN = ["deauth_count", "seq_anom", "duration_std", "length_mean", "rssi_mean", "rssi_std"]


def load(ds):
    W = pd.read_parquet(C.OUT_DIR / f"windows_{ds}.parquet")
    S = np.load(C.OUT_DIR / f"sequences_{ds}.npz")["S"]
    return W, S, E.folds(W["tau"].to_numpy(), W["y"].to_numpy())


def log(ds, name, res, oof, tag=""):
    p = T.pooled(oof, F.WIN_S)
    print(f"[{ds}] {name:6s} {tag:22s} F1 {res.F1.mean():.3f}±{res.F1.std():.3f} | pooled F1 {p['F1']:.3f} "
          f"[{p['F1_lo']:.3f}, {p['F1_hi']:.3f}] P {p['P']:.3f} R {p['R']:.3f} FPR {p['FPR']:.4f} | "
          f"PR-AUC {res.PR_AUC.mean():.3f} ROC-AUC {res.ROC_AUC.mean():.3f} | {res.fit_s.sum():.0f}s", flush=True)
    return p


def main(ds):
    W, S, fid = load(ds)
    out = C.OUT_DIR / "models" / ds
    out.mkdir(parents=True, exist_ok=True)
    rows, summary = [], []
    for name in MODELS:
        res, oof, fitted = T.run_cv(name, W, F.FEATURES, fid, F.WIN_S, S, seed=C.RANDOM_STATE, keep_models=True)
        oof.to_csv(C.OUT_DIR / f"oof_{name}_{ds}.csv", index=False)
        for k, (m, prep) in enumerate(fitted):
            joblib.dump(dict(model=m, prep=prep, features=F.FEATURES), out / f"{name}_fold{k}.joblib")
        rows.append(res)
        summary.append(dict(model=name, **log(ds, name, res, oof),
                            F1_fold_mean=res.F1.mean(), F1_fold_sd=res.F1.std(),
                            PR_AUC=res.PR_AUC.mean(), ROC_AUC=res.ROC_AUC.mean(), fit_s=res.fit_s.sum()))
    pd.concat(rows).to_csv(C.OUT_DIR / f"cv_main_{ds}.csv", index=False)
    pd.DataFrame(summary).to_csv(C.OUT_DIR / f"summary_main_{ds}.csv", index=False)


def folds(ds):
    """C1/C3: per-fold composition (before any result) and the session-confound check."""
    from sklearn.metrics import roc_auc_score
    from xgboost import XGBClassifier
    W, S, fid = load(ds)
    tau, y, ap = W["tau"].to_numpy(), W["y"].to_numpy(), W["ap"].to_numpy()
    sid = E.sessions(tau)
    rows = []
    for k in range(fid.max() + 1):
        for part, m in zip(("train", "val", "test"), E.split(tau, y, fid, k, F.WIN_S)):
            rows.append(dict(fold=k, part=part, n=int(m.sum()), pos=int(y[m].sum()), prevalence=y[m].mean(),
                             aps=len(np.unique(ap[m])), aps_pos=len(np.unique(ap[m & (y == 1)])),
                             **{f"neg_session{s}": int((m & (y == 0) & (sid == s)).sum()) for s in np.unique(sid)}))
    t = pd.DataFrame(rows)
    t.to_csv(C.OUT_DIR / f"folds_{ds}.csv", index=False)
    print(t.round(3).to_string(), flush=True)
    neg = y == 0
    if len(np.unique(sid[neg])) > 1:                 # can the features tell sessions apart?
        X, s = W.loc[neg, F.FEATURES].to_numpy(float), sid[neg]
        a, b = np.unique(s)[[0, -1]]
        keep = np.isin(s, [a, b])
        X, s, tt = X[keep], (s[keep] == b).astype(int), tau[neg][keep]
        blk = np.floor((tt - tt.min()) / 60).astype(int) % 5            # 60 s blocks, 5 groups
        sc = np.zeros(len(s))
        for g in range(5):
            m = XGBClassifier(n_estimators=200, max_depth=3, random_state=C.RANDOM_STATE, n_jobs=4)
            sc[blk == g] = m.fit(X[blk != g], s[blk != g]).predict_proba(X[blk == g])[:, 1]
        auc = roc_auc_score(s, sc)
        print(f"[{ds}] session confound: XGB separates negative session {a} from {b}, block-CV AUC = {auc:.3f}")
        pd.DataFrame([dict(session_a=a, session_b=b, auc=auc, n=len(s))]).to_csv(
            C.OUT_DIR / f"session_confound_{ds}.csv", index=False)


def rule_gain(ds):
    """C2: every model against two one-feature rules on the same out-of-fold windows (paired block
    bootstrap of the F1 difference, blocks of 2D):
      Rule        feature, direction chosen on train + validation of each fold (honest baseline);
      RuleFixed   one feature fixed for all folds, threshold still from validation; the feature is the
                  one with the best pooled out-of-fold F1 among all 15 (hindsight: an upper bound on
                  what any single-feature rule can do, not a deployable baseline).
    C1: a 'session' baseline (attack = the window lies in the session that holds the attack) and
    every model's metrics on the attack session alone, where the session cannot help.
    Also rewrites summary_main_<ds>.csv from the OOF files (pooled metrics, CI blocks of 2D)."""
    W, S, fid = load(ds)
    runs = {}
    for f in F.FEATURES:
        if W[f].notna().sum() and W[f].nunique() > 1:
            runs[f] = T.run_cv("Rule", W, F.FEATURES, fid, F.WIN_S, S, seed=C.RANDOM_STATE, extra={"feature": f})[:2]
    f1 = {f: E.metrics(o.y, o.pred, 1)["F1"] for f, (_, o) in runs.items()}
    pd.Series(f1, name="pooled_F1").sort_values(ascending=False).to_csv(C.OUT_DIR / f"rule_per_feature_{ds}.csv")
    best = max(f1, key=f1.get)
    print(f"[{ds}] fixed-feature rules, pooled F1: " + ", ".join(f"{f} {v:.3f}" for f, v in sorted(f1.items(), key=lambda kv: -kv[1])[:5]))
    for name, (res, oof) in (("Rule", T.run_cv("Rule", W, F.FEATURES, fid, F.WIN_S, S, seed=C.RANDOM_STATE)[:2]),
                             ("RuleFixed", runs[best])):
        res.assign(model=name).to_csv(C.OUT_DIR / f"cv_{name}_{ds}.csv", index=False)
        oof.to_csv(C.OUT_DIR / f"oof_{name}_{ds}.csv", index=False)
        print(f"[{ds}] {name}: features per fold {list(res.params)}")
    sid, y = E.sessions(W["tau"].to_numpy()), W["y"].to_numpy()
    att = np.isin(sid, np.unique(sid[y == 1]))
    rows = [dict(model="Session", **E.metrics(y, att.astype(int), 1))]
    print(f"[{ds}] Session baseline F1 = {rows[0]['F1']:.3f}  (attack session: {int(att.sum())} windows, "
          f"{int(y[att].sum())} positive)")
    oofs = {m: pd.read_csv(C.OUT_DIR / f"oof_{m}_{ds}.csv").sort_values("idx") for m in ["RuleFixed"] + MODELS}
    for name, o in oofs.items():
        yy, pp, tt = o.y.to_numpy(), o.pred.to_numpy(), o.tau.to_numpy()
        r = dict(model=name, **T.pooled(o, F.WIN_S))
        for ref in ("Rule", "RuleFixed"):
            d, lo, hi = E.block_ci_diff(yy, pp, oofs[ref].pred.to_numpy(), tt, 2 * F.WIN_S)
            r.update({f"dF1_vs_{ref}": d, f"dF1_vs_{ref}_lo": lo, f"dF1_vs_{ref}_hi": hi})
        a = att[o.idx.to_numpy()]
        m_att = E.metrics(yy[a], pp[a], 1)
        r.update({f"att_{k}": m_att[k] for k in ("F1", "P", "R", "FPR", "n", "prevalence")})
        rows.append(r)
        print(f"[{ds}] {name:9s} pooled F1 {r['F1']:.3f} [{r['F1_lo']:.3f}, {r['F1_hi']:.3f}] | vs Rule "
              f"{r['dF1_vs_Rule']:+.3f} [{r['dF1_vs_Rule_lo']:+.3f}, {r['dF1_vs_Rule_hi']:+.3f}] | vs RuleFixed "
              f"{r['dF1_vs_RuleFixed']:+.3f} [{r['dF1_vs_RuleFixed_lo']:+.3f}, {r['dF1_vs_RuleFixed_hi']:+.3f}] | "
              f"attack session F1 {r['att_F1']:.3f} FPR {r['att_FPR']:.3f}", flush=True)
    pd.DataFrame(rows).to_csv(C.OUT_DIR / f"baselines_{ds}.csv", index=False)


def transfer(_=None):
    """H3 / 2.3: train on one dataset, test on the other without retraining, for every model.
    Source: train = folds 1..4 (purged), validation = fold 0 (model selection, threshold). The
    normalizer / sequence scaler is fitted on the source train block only. Expected (stated before
    the run): the rate features flip sign, so only SAME_SIGN can transfer. A model whose target
    scores are all equal is reported as 'constant', not as chance level. Also writes the per-feature
    shift matrix (transfer_shift.csv) and the single-feature AUC with the source direction."""
    from scipy.stats import ks_2samp
    from sklearn.metrics import roc_auc_score
    data = {ds: load(ds) for ds in ("awid3", "rids")}
    rows = []
    for src, tgt in (("awid3", "rids"), ("rids", "awid3")):
        Ws, Ss, fs = data[src]
        Wt, St, _ = data[tgt]
        ts, ys, yt = Ws["tau"].to_numpy(), Ws["y"].to_numpy(), Wt["y"].to_numpy()
        va = fs == 0
        tr = E.purge(ts, ~va, va, F.WIN_S)
        for name in ("XGB", "OCSVM", "AE", "LSTM"):
            for tag, feats in ((("all_15", F.FEATURES), ("same_sign_6", SAME_SIGN)) if name != "LSTM" else (("view_B", None),)):
                T.set_seeds(C.RANDOM_STATE)
                if name == "LSTM":
                    prep = T.M.SeqScaler().fit(Ss[tr])
                    Xtr, Xva, Xt = prep.transform(Ss[tr]), prep.transform(Ss[va]), prep.transform(St)
                else:
                    Xs, Xt = Ws[feats].to_numpy(float), Wt[feats].to_numpy(float)
                    Xtr, Xva = Xs[tr], Xs[va]
                    if name != "XGB":
                        prep = make_normalizer_for(Xtr)
                        Xtr, Xva, Xt = prep.transform(Xtr), prep.transform(Xva), prep.transform(Xt)
                m, p, v, thr, _ = T.select(name, Xtr, ys[tr], Xva, ys[va], feats or [], C.RANDOM_STATE)
                st = m.score(Xt)
                r = dict(source=src, target=tgt, model=name, features=tag, params=str(p),
                         constant=bool(np.ptp(st) == 0), **E.metrics(yt, st, thr))
                rows.append(r)
                print(f"{src}->{tgt} {name:5s} {tag:12s} ROC-AUC {r['ROC_AUC']:.3f} PR-AUC {r['PR_AUC']:.3f} "
                      f"(const {r['PR_AUC_const']:.3f}) F1 {r['F1']:.3f} FPR {r['FPR']:.3f}"
                      f"{'  CONSTANT OUTPUT' if r['constant'] else ''}", flush=True)
        for f in F.FEATURES:                         # single feature, direction learned on source
            xs, xt = Ws[f].to_numpy(float), Wt[f].to_numpy(float)
            ok_s, ok_t = ~np.isnan(xs), ~np.isnan(xt)
            if (min(ok_s.sum(), ok_t.sum()) < 2 or np.nanstd(xs) == 0 or np.nanstd(xt) == 0
                    or len(np.unique(ys[ok_s])) < 2 or len(np.unique(yt[ok_t])) < 2):
                continue
            sign = 1 if roc_auc_score(ys[ok_s], xs[ok_s]) >= 0.5 else -1
            rows.append(dict(source=src, target=tgt, model="single_feature", features=f,
                             ROC_AUC=roc_auc_score(yt[ok_t], sign * xt[ok_t])))
    pd.DataFrame(rows).to_csv(C.OUT_DIR / "transfer.csv", index=False)
    shift = []
    for f in F.FEATURES:
        r = dict(feature=f)
        vals = {}
        for ds in ("awid3", "rids"):
            x = data[ds][0][f].to_numpy(float)
            vals[ds] = x[~np.isnan(x)]
            q1, med, q3 = np.nanpercentile(x, [25, 50, 75]) if len(vals[ds]) else (np.nan,) * 3
            r.update({f"{ds}_mean": np.nanmean(x) if len(vals[ds]) else np.nan, f"{ds}_median": med,
                      f"{ds}_iqr": q3 - q1, f"{ds}_missing": np.isnan(x).mean()})
        r["ks"] = ks_2samp(vals["awid3"], vals["rids"]).statistic if min(map(len, vals.values())) else np.nan
        shift.append(r)
    pd.DataFrame(shift).to_csv(C.OUT_DIR / "transfer_shift.csv", index=False)
    print(pd.DataFrame(shift).round(3).to_string())


def make_normalizer_for(X):
    from pipeline.preprocess import make_normalizer
    return make_normalizer().fit(X)


def imbalance(ds):
    W, S, fid = load(ds)
    rows, summary = [], []
    for name, modes in (("XGB", ("none", "cost", "smote")), ("LSTM", ("none", "cost"))):
        for mode in modes:
            res, oof, _ = T.run_cv(name, W, F.FEATURES, fid, F.WIN_S, S, seed=C.RANDOM_STATE, imbalance=mode,
                                  impute=True)          # one NaN handling for all arms (SMOTE needs it)
            rows.append(res)
            summary.append(dict(model=name, imbalance=mode, **log(ds, name, res, oof, mode),
                                F1_fold_sd=res.F1.std(), PR_AUC=res.PR_AUC.mean()))
    pd.concat(rows).to_csv(C.OUT_DIR / f"cv_imbalance_{ds}.csv", index=False)
    pd.DataFrame(summary).to_csv(C.OUT_DIR / f"summary_imbalance_{ds}.csv", index=False)


def _windows(ds, D):
    """Rebuild View A with window length D (sensitivity only)."""
    old = F.WIN_S
    F.WIN_S = D
    try:
        return F.build_windows(preprocess.clean(ingestion.load(ds)))
    finally:
        F.WIN_S = old


DEFAULT = {"XGB": dict(max_depth=3, learning_rate=0.05), "LSTM": dict(h1=64, h2=32),
           "OCSVM": dict(nu=0.01, gamma=0.01), "AE": dict(bottleneck=8)}      # modal choice of the main run
SWEEP = {                                                                  # one knob at a time
    "XGB": [("max_depth", v) for v in (2, 3, 4, 5, 6, 8)] + [("learning_rate", v) for v in (0.01, 0.03, 0.05, 0.1, 0.2, 0.3)],
    "OCSVM": [("nu", v) for v in (0.005, 0.01, 0.02, 0.05, 0.1, 0.2)] + [("gamma", v) for v in (0.001, 0.01, 0.1, 1.0, "scale")],
    "AE": [("bottleneck", v) for v in (2, 4, 8, 12)] + [("q", v) for v in (0.95, 0.98, 0.99, 0.995)],
    "LSTM": [("units", v) for v in ((16, 8), (32, 16), (64, 32), (128, 64))] + [("units", (64, 0))]
            + [("lr", v) for v in (3e-4, 1e-3, 3e-3)],
}


NEG_FRAC = {"awid3": 1.0, "rids": 0.25}  # 1.5 only: share of benign training windows kept outside the
                                         # attack session (60 s blocks); validation and test stay complete


QUICK = {"awid3": ("XGB", "OCSVM", "AE"), "rids": ("XGB",)}   # fast cut (user request): LSTM, the RIDS
                                         # one-class sweeps, the bridge and the D rebuilds are left for later
# P0.3: `python run_train.py sensitivity full` runs ALL four core models on BOTH datasets (the complete
# 4x2 matrix). The default keeps the QUICK cut. The window-length D sweep still needs the cleaned raw
# frames (ingestion.load), which are not in this archive, so it stays NOT RUN and is documented.
SENS_PROFILE = "quick"
ALL_SENS = {"awid3": ("XGB", "OCSVM", "AE", "LSTM"), "rids": ("XGB", "OCSVM", "AE", "LSTM")}


def _sens_models(ds):
    return ALL_SENS[ds] if SENS_PROFILE == "full" else QUICK[ds]


def sensitivity(ds):
    """1.5. (a) parameter sensitivity: one knob at a time around DEFAULT, seed fixed, 5 folds, on
    validation (where the choice is made) and test. (b) sampling stability: DEFAULT config, 5 seeds x
    5 folds, test folds; the SD is split into a seed part and a fold part. (c) XGB with window
    length D in {2, 10} s and with feature groups removed or added."""
    W, S, fid = load(ds)
    rows = []

    def run(name, tag, knob, value, params, Wx=W, feats=F.FEATURES, fx=fid, gap=F.WIN_S, seed=C.RANDOM_STATE, q=0.99):
        res, oof, _ = T.run_cv(name, Wx, feats, fx, gap, S, seed=seed, grid=[params], q=q, neg_frac=NEG_FRAC[ds])
        res = res.assign(experiment=tag, knob=knob, value=str(value))
        rows.append(res)
        print(f"[{ds}] {name:5s} {tag:9s} {knob}={value!s:10s} val PR-AUC {res.val_PR_AUC.mean():.3f}±{res.val_PR_AUC.std():.3f} "
              f"val R@1% {res.val_R_at_FPR1.mean():.3f} | test PR-AUC {res.PR_AUC.mean():.3f}±{res.PR_AUC.std():.3f} "
              f"R@1% {res.R_at_FPR1.mean():.3f} F1 {res.F1.mean():.3f} FPR {res.FPR.mean():.4f}", flush=True)

    for name, knobs in SWEEP.items():                                       # (a)
        if name not in _sens_models(ds):
            continue
        for knob, v in knobs:
            p, q = dict(DEFAULT[name]), 0.99
            if knob == "q":
                q = v
            elif knob == "units":
                p.update(h1=v[0], h2=v[1])
            else:
                p[knob] = v
            run(name, "param", knob, v, p, q=q)
    if False:                                                               # bridge to Table 1.4
        for name in ("XGB", "AE", "LSTM"):
            for frac in (1.0, NEG_FRAC[ds]):
                res, _, _ = T.run_cv(name, W, F.FEATURES, fid, F.WIN_S, S, seed=C.RANDOM_STATE,
                                     grid=[DEFAULT[name]], neg_frac=frac)
                rows.append(res.assign(experiment="bridge", knob="neg_frac", value=str(frac)))
                print(f"[{ds}] {name:5s} bridge neg_frac={frac}: test F1 {res.F1.mean():.3f}±{res.F1.std():.3f} "
                      f"PR-AUC {res.PR_AUC.mean():.3f} R@1% {res.R_at_FPR1.mean():.3f}", flush=True)
    for name in DEFAULT:                                                    # (b)
        if name not in _sens_models(ds) or name == "OCSVM":                 # OC-SVM is deterministic: no seed sweep
            continue
        for seed in range(5):
            run(name, "seed", "seed", seed, DEFAULT[name], seed=seed)
    for D in ():                                                            # (c) later
        Wd = _windows(ds, D)
        run("XGB", "window", "D", D, DEFAULT["XGB"], Wx=Wd, fx=E.folds(Wd["tau"].to_numpy(), Wd["y"].to_numpy()), gap=D)
    for tag, feats in (("-ssid_context", [f for f in F.FEATURES if f not in SSID_CONTEXT]),
                       ("+mfpc", F.FEATURES + F.ABLATION_CONFIG)):
        run("XGB", "features", "set", tag, DEFAULT["XGB"], feats=feats)
    pd.concat(rows).to_csv(C.OUT_DIR / f"cv_sensitivity_{ds}.csv", index=False)


OPT_MODELS = ["XGB", "LSTM", "OCSVM", "AE"]
FIXES = [                     # (tag, models it targets, change); applied one at a time, in this order
    ("noise", OPT_MODELS, "drop y=0 windows with ssid_newcomer = 1 from training (suspected label inconsistency, 2.1-1)"),
    ("oc_f1", ["OCSVM", "AE"], "one-class threshold = max-F1 on validation (2.1-2, 1.5)"),
    ("hybrid", ["LSTM"], "LSTM head also gets View A (2.1-4)"),
    ("ap_base", OPT_MODELS, "non-SSID features minus the AP's own median over [t-300, t-5] s (2.1-3)"),
]


def optimise(ds):
    """2.2. Start from the modal configuration (Section 1.5, same RIDS training subsample, seed fixed).
    Each fix is applied on top of the configuration accepted so far and kept if it raises the mean
    pooled validation F1 of the models it targets. Selection uses validation only; the test folds are
    read once, for the start and the final configuration."""
    W0, S, fid = load(ds)
    y = W0["y"].to_numpy()
    based = F.ap_baseline_delta(W0, [f for f in F.FEATURES if f not in SSID_CONTEXT])
    cfg = dict(noise=False, oc_f1=False, hybrid=False, ap_base=False)
    cache, rows = {}, []

    def run(name, c):
        key = (name, c["noise"] and ds_has_noise, c["oc_f1"] and name in T.ONE_CLASS,
               c["hybrid"] and name == "LSTM", c["ap_base"])
        if key not in cache:
            W = based if c["ap_base"] else W0
            drop = ((W0["y"] == 0) & (W0["ssid_newcomer"] == 1)).to_numpy() if c["noise"] else None
            m = "HLSTM" if (name == "LSTM" and c["hybrid"]) else name
            T.VAL_OOF.clear()
            res, oof, _ = T.run_cv(m, W, F.FEATURES, fid, F.WIN_S, S, seed=C.RANDOM_STATE, grid=[DEFAULT[name]],
                                   neg_frac=NEG_FRAC[ds], drop=drop, oc_thr="f1" if c["oc_f1"] else "quantile")
            cache[key] = (T.pooled(pd.concat(T.VAL_OOF), F.WIN_S), T.pooled(oof, F.WIN_S), res)
        return cache[key]

    ds_has_noise = bool(((W0["y"] == 0) & (W0["ssid_newcomer"] == 1)).any())
    for name in OPT_MODELS:
        v, t, r = run(name, cfg)
        rows.append(dict(step="start", model=name, kept=True, part="val", **v))
        print(f"[{ds}] start  {name:5s} val F1 {v['F1']:.3f} FPR {v['FPR']:.4f} FNR {1 - v['R']:.3f}", flush=True)
    for tag, targets, what in FIXES:
        new = dict(cfg, **{tag: True})
        before = {n: run(n, cfg)[0] for n in OPT_MODELS}
        after = {n: run(n, new)[0] for n in OPT_MODELS}
        keep = np.mean([after[n]["F1"] for n in targets]) > np.mean([before[n]["F1"] for n in targets])
        for n in OPT_MODELS:
            rows.append(dict(step=tag, model=n, kept=keep, part="val", **after[n]))
            print(f"[{ds}] {tag:7s}{n:5s} val F1 {before[n]['F1']:.3f} -> {after[n]['F1']:.3f}  FPR {after[n]['FPR']:.4f} "
                  f"FNR {1 - after[n]['R']:.3f}  {'KEEP' if keep else 'drop'}", flush=True)
        if keep:
            cfg = new
    start = dict(noise=False, oc_f1=False, hybrid=False, ap_base=False)
    for n in OPT_MODELS:                                                      # test, once
        for tag, c in (("start", start), ("final", cfg)):
            t = run(n, c)[1]
            rows.append(dict(step=tag, model=n, kept=True, part="test", **t))
            print(f"[{ds}] TEST {tag:6s}{n:5s} F1 {t['F1']:.3f} [{t['F1_lo']:.3f}, {t['F1_hi']:.3f}] FPR {t['FPR']:.4f} "
                  f"FNR {1 - t['R']:.3f} FP {t['FP']} FN {t['FN']}", flush=True)
    print(f"[{ds}] final configuration: {cfg}")
    pd.DataFrame(rows).to_csv(C.OUT_DIR / f"optimise_{ds}.csv", index=False)


def random_split(ds):
    """1.4: the optimistic upper bound. Stratified random 5-fold over windows, no purge: overlapping
    windows of the same 5 s land in train and test."""
    from sklearn.model_selection import StratifiedKFold
    W, S, _ = load(ds)
    y = W["y"].to_numpy()
    fid = np.zeros(len(y), int)
    for k, (_, te) in enumerate(StratifiedKFold(5, shuffle=True, random_state=C.RANDOM_STATE).split(y, y)):
        fid[te] = k
    res, oof, _ = T.run_cv("XGB", W, F.FEATURES, fid, 0.0, S, seed=C.RANDOM_STATE, grid=[DEFAULT["XGB"]])
    res.assign(experiment="random_split").to_csv(C.OUT_DIR / f"cv_random_{ds}.csv", index=False)
    print(f"[{ds}] XGB random split: F1 {res.F1.mean():.3f}±{res.F1.std():.3f} PR-AUC {res.PR_AUC.mean():.3f}")


if __name__ == "__main__":
    argv = sys.argv[1:]
    if "full" in argv:                      # P0.3: full 4x2 sensitivity matrix
        SENS_PROFILE = "full"
        argv = [a for a in argv if a != "full"]
    exp, dss = argv[0], argv[1:] or ["awid3", "rids"]
    dss = dss[:1] if exp == "transfer" else dss
    t0 = time.time()
    for ds in dss:
        {"folds": folds, "main": main, "rule_gain": rule_gain, "transfer": transfer, "imbalance": imbalance,
         "sensitivity": sensitivity, "random": random_split, "optimise": optimise}[exp](ds)
    print(f"done in {time.time() - t0:.0f}s")
