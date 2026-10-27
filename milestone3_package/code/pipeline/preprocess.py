"""
Stage 2: Preprocessing & Normalization. Dataset-agnostic: reads unified names only.

  clean()       stateless, row-wise, fits nothing (so it may run before the split):
                1. exact duplicate frames removed on (t, src, seq, subtype, length): RIDS stores
                   every frame twice (all 50,258 RogueAP rows are 25,129 pairs), so a random split
                   would put each frame's twin in the test set;
                2. physical range checks on 802.11 fields (out of range -> NaN);
                3. frames with no BSSID (ACK/CTS/RTS) or the broadcast BSSID removed: they cannot
                   be attributed to an AP, and in AWID3 they are all benign (a "no AP = benign"
                   shortcut, the same kind of artifact as the missing-RSSI imputation in the draft).
  Normalizer    median imputation (+ an is_missing indicator per feature with NaNs) + RobustScaler. FIT ON THE TRAINING FOLD ONLY, then applied
                to the test fold (Step-7 leakage rule). One scaler for all data: no
                per-environment MinMax (Milestone 2 feedback, Ch. 5).
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

# Valid ranges from the 802.11 standard / physics. Values outside are capture artifacts -> NaN.
VALID = {
    "signal": (-100.0, 0.0),        # dBm; below the ~-101 dBm thermal floor is impossible (review C4)
    "duration": (0.0, 32767.0),     # 15-bit Duration/ID field, microseconds
    "retry": (0.0, 1.0),
    "protected": (0.0, 1.0),
    "mfpc": (0.0, 1.0),
    "type": (0.0, 3.0),
    "subtype": (0.0, 15.0),
    "ds": (0.0, 3.0),
    "seq": (0.0, 4095.0),           # 12-bit sequence number
    "length": (0.0, 7935.0),        # max 802.11 MPDU
}


BROADCAST = 0xFFFFFFFFFFFF


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates(subset=["t", "src", "seq", "subtype", "length"]).copy()
    for col, (lo, hi) in VALID.items():
        if col in df:
            df.loc[(df[col] < lo) | (df[col] > hi), col] = np.nan
    df = df[(df["bssid"] >= 0) & (df["bssid"] != BROADCAST)]
    return df.sort_values("t", kind="stable").reset_index(drop=True)


def make_normalizer() -> Pipeline:
    """x' = (x - median_train) / IQR_train, after median imputation (both fit on train only).
    Missingness is kept explicit as binary is_missing columns, never hidden in an imputed constant.
    Tree models (XGBoost) skip this and use NaN directly (learned default branch)."""
    return Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", RobustScaler(quantile_range=(25.0, 75.0))),
    ])
