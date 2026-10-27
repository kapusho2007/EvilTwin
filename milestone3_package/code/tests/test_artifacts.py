"""Saved fold models reload and reproduce the stored out-of-fold test scores (artifact parity).
Skipped until run_train.py main has written output/models/."""
import joblib
import numpy as np
import pandas as pd
import pytest

import config as C
from pipeline import evaluate as E, features as F, train as T

CASES = [(ds, m) for ds in ("awid3", "rids") for m in ("RF", "XGB", "OCSVM", "AE", "LSTM")]


@pytest.mark.parametrize("ds,name", CASES)
def test_reload_parity(ds, name):
    path = C.OUT_DIR / "models" / ds / f"{name}_fold0.joblib"
    if not path.exists():
        pytest.skip("no saved models")
    W = pd.read_parquet(C.OUT_DIR / f"windows_{ds}.parquet")
    S = np.load(C.OUT_DIR / f"sequences_{ds}.npz")["S"] if name in T.VIEW_B else None
    tau, y = W["tau"].to_numpy(), W["y"].to_numpy()
    fid = E.folds(tau, y)
    oof = pd.read_csv(C.OUT_DIR / f"oof_{name}_{ds}.csv")
    for k in range(fid.max() + 1):
        art = joblib.load(C.OUT_DIR / "models" / ds / f"{name}_fold{k}.joblib")
        te = E.split(tau, y, fid, k, F.WIN_S)[2]
        if name in T.VIEW_B:
            X = art["prep"].transform(np.asarray(S[te]))
        else:
            X = W.loc[te, art["features"]].to_numpy(float)
            X = X if art["prep"] is None else art["prep"].transform(X)
        ref = oof[oof.fold == k].sort_values("idx")["score"].to_numpy()
        assert np.allclose(art["model"].score(X), ref, rtol=1e-5, atol=1e-6)
