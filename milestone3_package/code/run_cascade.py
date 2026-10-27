"""Part 3 driver: the hybrid cascade on the saved fold models (no retraining).
Usage: python run_cascade.py [awid3 rids]

Per fold k the saved models score the validation and the test fold of pipeline.evaluate.split; every
cascade threshold comes from validation, and the test fold is scored once (pipeline.cascade).
Writes per dataset:
  cascade_windows_<ds>.csv    one row per test window: scores, gate, stage-2 decision, escalation, arbiters
  cascade_<ds>.csv            pooled F1 / P / R / FPR / FNR with block-bootstrap CIs per configuration
  cascade_folds_<ds>.csv      per-fold gate choice, thresholds, gate recall, pass and escalation rates
  cascade_baselines_<ds>.csv  per-fold benign training median / IQR of every feature (LLM context, Part 4)
If llm_decisions_<ds>.csv exists (run_llm.py), the "+ LLM" rows are added."""
import sys

import joblib
import numpy as np
import pandas as pd

import config as C
from pipeline import cascade as K, evaluate as E, features as F

MODELS = ["XGB", "LSTM", "AE", "OCSVM", "Rule"]


def fold_scores(ds, W, S, k, rows):
    X = W[F.FEATURES].to_numpy(float)
    out = {}
    for name in MODELS:
        art = joblib.load(C.OUT_DIR / "models" / ds / f"{name}_fold{k}.joblib")
        m, prep = art["model"], art["prep"]
        Z = S if name == "LSTM" else X
        out[name] = {part: m.score(prep.transform(Z[r]) if prep is not None else Z[r]) for part, r in rows.items()}
    return out


def optimised_gates(ds, W):
    """One-class gate models with the 2.2-final configuration (optimise_<ds>.csv): the fixes kept on
    validation that change the one-class models' training data or inputs (noise: drop y = 0 windows with
    ssid_newcomer = 1 from training; ap_base: per-AP baseline deltas). None kept -> the Table 1.3 models."""
    opt = pd.read_csv(C.OUT_DIR / f"optimise_{ds}.csv")
    kept = set(opt.loc[opt.kept & (opt.part == "val") & (opt.step != "start"), "step"])
    if not kept & {"noise", "ap_base"}:
        return None
    from pipeline import train as T
    from run_train import DEFAULT, SSID_CONTEXT
    Wg = F.ap_baseline_delta(W, [f for f in F.FEATURES if f not in SSID_CONTEXT]) if "ap_base" in kept else W
    drop = ((W["y"] == 0) & (W["ssid_newcomer"] == 1)).to_numpy() if "noise" in kept else None
    fid = E.folds(W["tau"].to_numpy(), W["y"].to_numpy())
    out = {}
    for k in range(fid.max() + 1):
        for name in ("AE", "OCSVM"):
            T.VAL_OOF.clear()
            _, oof, _, _ = T.run_fold(name, Wg, F.FEATURES, k, fid, F.WIN_S, seed=C.RANDOM_STATE,
                                      grid=[DEFAULT[name]], drop=drop)
            out[(k, name)] = dict(val=T.VAL_OOF[-1]["score"].to_numpy(), test=oof["score"].to_numpy())
    print(f"[{ds}] gate from the 2.2-final one-class models (kept fixes: {sorted(kept)})", flush=True)
    return out


def pooled(y, p, tau):
    r = E.metrics(y, p, 1)
    lo, hi = E.block_ci(y, p, 1, tau, 2 * F.WIN_S)
    return dict(F1=r["F1"], F1_lo=lo, F1_hi=hi, P=r["P"], R=r["R"], FPR=r["FPR"], FNR=r["FNR"],
                TP=r["TP"], FP=r["FP"], FN=r["FN"], TN=r["TN"])


def main(ds):
    W = pd.read_parquet(C.OUT_DIR / f"windows_{ds}.parquet")
    S = np.load(C.OUT_DIR / f"sequences_{ds}.npz")["S"]
    cv = pd.read_csv(C.OUT_DIR / f"cv_main_{ds}.csv").set_index(["model", "fold"])["thr"]
    tau, y = W["tau"].to_numpy(), W["y"].to_numpy()
    fid = E.folds(tau, y)
    win, folds, base = [], [], []
    opt_gate = optimised_gates(ds, W)
    for k in range(fid.max() + 1):
        tr, va, te = E.split(tau, y, fid, k, F.WIN_S)
        sc = fold_scores(ds, W, S, k, dict(val=va, test=te))
        if opt_gate is not None:                                             # Table 1.3 AE kept for the 99th-pct row
            sc["AE_1.3"] = sc["AE"]
            sc["AE"], sc["OCSVM"] = opt_gate[(k, "AE")], opt_gate[(k, "OCSVM")]
        yva = y[va]
        g, t_g, pass_va = K.choose_gate(yva, {m: sc[m]["val"] for m in ("AE", "OCSVM")})
        gate = sc[g]["test"] >= t_g
        gate99 = sc.get("AE_1.3", sc["AE"])["test"] >= cv[("AE", k)]       # Table 1.3 AE at its 99th-pct threshold
        t_x, t_l, t_r = cv[("XGB", k)], cv[("LSTM", k)], cv[("Rule", k)]
        a_l = sc["LSTM"]["test"] >= t_l
        stage2, esc = K.route(gate, sc["XGB"]["test"], t_x, a_l)
        _, esc_va = K.route(sc[g]["val"] >= t_g, sc["XGB"]["val"], t_x, sc["LSTM"]["val"] >= t_l)
        yte = y[te]
        folds.append(dict(fold=k, gate=g, gate_source="2.2-final" if opt_gate is not None else "Table 1.3", t_gate=t_g, val_pass=pass_va, val_esc=esc_va.mean(),
                          test_gate_recall=gate[yte == 1].mean(), test_pass=gate.mean(),
                          test_esc=int(esc.sum()), test_esc_pos=int(esc[yte == 1].sum()),
                          gate99_recall=gate99[yte == 1].mean()))
        win.append(pd.DataFrame(dict(
            window=np.where(te)[0], fold=k, tau=tau[te], y=yte, gate_model=g,
            **{f"{m}_score": sc[m]["test"] for m in MODELS},
            XGB_alert=(sc["XGB"]["test"] >= t_x).astype(int), LSTM_alert=a_l.astype(int),
            Rule_alert=(sc["Rule"]["test"] >= t_r).astype(int), XGB_thr=t_x, LSTM_thr=t_l,
            gate=gate.astype(int), gate99=gate99.astype(int), stage2=stage2, escalated=esc.astype(int))))
        b = W.loc[tr & (y == 0), F.FEATURES]
        base.append(pd.DataFrame(dict(fold=k, feature=F.FEATURES, median=b.median().values,
                                      q25=b.quantile(0.25).values, q75=b.quantile(0.75).values)))
        print(f"[{ds}] fold {k}: gate {g} t={t_g:.4g} val pass {pass_va:.3f} | test gate recall "
              f"{folds[-1]['test_gate_recall']:.3f} pass {gate.mean():.3f} escalated {esc.sum()} "
              f"({esc[yte == 1].sum()} attack)", flush=True)
    D = pd.concat(win, ignore_index=True)
    oof = pd.read_csv(C.OUT_DIR / f"oof_all_{ds}.csv").set_index("window")
    for m in ("XGB", "LSTM") + (("AE",) if opt_gate is None else ()):     # save -> reload parity
        a, b = D[f"{m}_score"].values, oof.loc[D.window, f"{m}_score"].values
        dev = np.nanmax(np.abs(a - b) / np.maximum(np.abs(b), 1e-9))        # relative: AE errors reach 1e9
        print(f"[{ds}] parity {m}: max relative |reloaded - stored OOF| = {dev:.1e}")
    llm_path = C.OUT_DIR / f"llm_decisions_{ds}.csv"
    if llm_path.exists():
        L = pd.read_csv(llm_path).set_index("window")["llm_alert"]
        D["llm"] = D.window.map(L)
    D.to_csv(C.OUT_DIR / f"cascade_windows_{ds}.csv", index=False)
    pd.DataFrame(folds).to_csv(C.OUT_DIR / f"cascade_folds_{ds}.csv", index=False)
    pd.concat(base).to_csv(C.OUT_DIR / f"cascade_baselines_{ds}.csv", index=False)

    yv, t, g, s2, e = D.y.values, D.tau.values, D.gate.values == 1, D.stage2.values, D.escalated.values == 1
    e_all = D.XGB_alert.values != D.LSTM_alert.values                       # escalation without the gate
    configs = {
        "XGBoost alone": D.XGB_alert.values,
        "LSTM alone": D.LSTM_alert.values,
        "Autoencoder alone (99th pct.)": D.gate99.values,
        "Gate 99th pct. (Table 1.3 AE) -> XGBoost": (D.gate99.values == 1) & (D.XGB_alert.values == 1),
        "Gate (recall 0.99) -> stage 2, arbiter: trust XGBoost": s2,
        "  arbiter: always benign": K.decide(g, s2, e, 0),
        "  arbiter: trust LSTM": K.decide(g, s2, e, D.LSTM_alert.values),
        "  arbiter: one-feature rule": K.decide(g, s2, e, D.Rule_alert.values),
        "No gate: XGB + LSTM -> rule arbiter": K.decide(np.ones_like(g), D.XGB_alert.values, e_all, D.Rule_alert.values),
    }
    if "llm" in D:
        configs["  arbiter: LLM (abstain -> XGBoost)"] = K.decide(g, s2, e, D.llm.fillna(D.XGB_alert).values.astype(int))
        configs["  arbiter: LLM (abstain -> benign)"] = K.decide(g, s2, e, D.llm.fillna(0).values.astype(int))
    rows = []
    for name, p in configs.items():
        p = np.asarray(p).astype(int)
        d = E.block_ci_diff(yv, p, D.XGB_alert.values, t, 2 * F.WIN_S)
        rows.append(dict(dataset=ds, config=name, **pooled(yv, p, t), dF1_vs_XGB=d[0], dF1_lo=d[1], dF1_hi=d[2]))
    R = pd.DataFrame(rows)
    R["gate_recall"] = g[yv == 1].mean()
    R["pass_rate"] = g.mean()
    R["escalated"] = int(e.sum())
    R["escalated_attack"] = int(e[yv == 1].sum())
    R.to_csv(C.OUT_DIR / f"cascade_{ds}.csv", index=False)
    print(R.round(4).to_string(), flush=True)


if __name__ == "__main__":
    for ds in sys.argv[1:] or ["awid3", "rids"]:
        main(ds)
