"""Complete the RIDS LSTM sensitivity (the only NOT-RUN sensitivity cell) on the cached
windows + sequences, and MERGE into cv_sensitivity_rids.csv (XGB/OC-SVM/AE RIDS already done).
Same knobs/seeds as run_train.sensitivity: LSTM units + learning rate, then 5 seeds at the default."""
import pandas as pd
import config as C
import run_train as R
from pipeline import features as F, train as T

W, S, fid = R.load("rids")
rows = []


def run(tag, knob, value, params, seed=C.RANDOM_STATE):
    res, _, _ = T.run_cv("LSTM", W, F.FEATURES, fid, F.WIN_S, S, seed=seed, grid=[params],
                         neg_frac=R.NEG_FRAC["rids"])
    rows.append(res.assign(experiment=tag, knob=knob, value=str(value)))
    print(f"[rids] LSTM {tag:6s} {knob}={value!s:12s} val PR-AUC {res.val_PR_AUC.mean():.3f} "
          f"test PR-AUC {res.PR_AUC.mean():.3f}+-{res.PR_AUC.std():.3f} F1 {res.F1.mean():.3f} "
          f"R@1% {res.R_at_FPR1.mean():.3f}", flush=True)


for knob, v in R.SWEEP["LSTM"]:                 # (a) parameter sweep: units + lr
    p = dict(R.DEFAULT["LSTM"])
    if knob == "units":
        p.update(h1=v[0], h2=v[1])
    else:
        p[knob] = v
    run("param", knob, v, p)
for seed in range(5):                           # (b) seed stability
    run("seed", "seed", seed, R.DEFAULT["LSTM"], seed=seed)

new = pd.concat(rows, ignore_index=True)
old = pd.read_csv(C.OUT_DIR / "cv_sensitivity_rids.csv")
old = old[old.model != "LSTM"]                  # drop any stale LSTM rows, then add the fresh ones
pd.concat([old, new], ignore_index=True).to_csv(C.OUT_DIR / "cv_sensitivity_rids.csv", index=False)
print("appended RIDS LSTM sensitivity to cv_sensitivity_rids.csv")
