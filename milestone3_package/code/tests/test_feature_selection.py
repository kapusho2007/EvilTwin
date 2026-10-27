"""P0.1 leakage / feature-selection regression tests.
Run: python -m pytest tests -v"""
import numpy as np
import pandas as pd
import pytest

import config as C
from pipeline import features as F

# unified names that are grouping/ordering keys or a priori forbidden shortcuts: never model inputs.
FORBIDDEN = ["t", "src", "src_fallback", "bssid", "seq", "ssid", "protected", "tsf",
             "mfpc", "clock_skew_ppm"]


def test_no_forbidden_field_is_a_feature():
    for f in FORBIDDEN:
        assert f not in F.FEATURES, f"forbidden field {f} leaked into FEATURES"


def test_shortcuts_excluded_but_are_candidates_or_ablation():
    # mfpc is an ablation-only encryption shortcut; clock_skew_ppm is a presence/absence leakage shortcut
    assert "mfpc" in F.ABLATION_CONFIG and "mfpc" not in F.FEATURES
    assert "clock_skew_ppm" in F.CANDIDATES and "clock_skew_ppm" not in F.FEATURES


def test_restored_features_present_and_complement_dropped():
    # P0.1: share_data and beacon_iat_mean were restored (train-fold |rho| < 0.95); the exact
    # complement share_mgmt stays dropped.
    assert "share_data" in F.FEATURES and "beacon_iat_mean" in F.FEATURES
    assert "share_mgmt" not in F.FEATURES


def _trainfold_absrho(ds, a, b):
    from scipy.stats import spearmanr
    from pipeline import evaluate as E
    p = C.OUT_DIR / f"windows_{ds}.parquet"
    if not p.exists():
        pytest.skip("cached windows absent")
    W = pd.read_parquet(p)
    tau, y = W["tau"].to_numpy(), W["y"].to_numpy()
    fid = E.folds(tau, y)
    out = []
    for k in range(fid.max() + 1):
        tr = E.split(tau, y, fid, k, F.WIN_S)[0]
        xa, xb = W.loc[tr, a].to_numpy(float), W.loc[tr, b].to_numpy(float)
        m = ~(np.isnan(xa) | np.isnan(xb))
        if m.sum() > 10 and np.std(xa[m]) > 0 and np.std(xb[m]) > 0:
            out.append(abs(spearmanr(xa[m], xb[m]).statistic))
    return np.array(out)


def test_redundancy_decisions_hold_on_training_folds():
    # share_mgmt is an exact complement on BOTH datasets -> dropped.
    assert np.nanmin(_trainfold_absrho("awid3", "share_mgmt", "share_data")) > 0.95
    assert np.nanmin(_trainfold_absrho("rids", "share_mgmt", "share_data")) > 0.95
    # share_data / beacon_iat_mean are redundant on RIDS but NOT on AWID3 -> kept in the unified list
    # (drop only if redundant on BOTH datasets' training folds).
    assert np.nanmax(_trainfold_absrho("awid3", "share_data", "duration_mean")) < 0.95
    assert np.nanmin(_trainfold_absrho("rids", "share_data", "duration_mean")) > 0.95 - 1e-6
    assert np.nanmax(_trainfold_absrho("awid3", "beacon_iat_mean", "beacon_rate")) < 0.95
    assert np.nanmin(_trainfold_absrho("rids", "beacon_iat_mean", "beacon_rate")) > 0.95
