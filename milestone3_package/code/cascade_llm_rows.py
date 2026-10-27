"""Table 3.1 LLM rows from saved outputs, without retraining (same decide() and bootstrap as run_cascade.py):
cascade_windows_<ds>.csv + llm_decisions_<ds>.csv -> cascade_llm.csv."""
import numpy as np
import pandas as pd

import config as C
from pipeline import cascade as K, evaluate as E, features as F
from run_cascade import pooled

rows = []
for ds in ["awid3", "rids"]:
    D = pd.read_csv(C.OUT_DIR / f"cascade_windows_{ds}.csv")
    L = pd.read_csv(C.OUT_DIR / f"llm_decisions_{ds}.csv").set_index("window")["llm_alert"]
    llm = D.window.map(L)
    y, t = D.y.values, D.tau.values
    g, s2, e = D.gate.values, D.stage2.values, D.escalated.values
    for name, a in {"arbiter: LLM (abstain -> XGBoost)": llm.fillna(D.XGB_alert),
                    "arbiter: LLM (abstain -> benign)": llm.fillna(0)}.items():
        p = K.decide(g, s2, e, a.values.astype(int))
        d = E.block_ci_diff(y, p, D.XGB_alert.values, t, 2 * F.WIN_S)
        rows.append(dict(dataset=ds, config=name, **pooled(y, p, t), dF1_vs_XGB=d[0], dF1_lo=d[1], dF1_hi=d[2]))
R = pd.DataFrame(rows)
R.to_csv(C.OUT_DIR / "cascade_llm.csv", index=False)
print(R.round(4).to_string())
