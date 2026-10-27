"""P0.2 impact audit: rebuild View A with the past-only SSID code and compare with the saved
windows_<ds>.parquet (built with the old whole-capture SSID mode). Also counts APs that beacon
more than one SSID. Usage: python audit_ssid_lookahead.py [awid3 rids]
Writes output/audit_ssid_lookahead.csv."""
import sys

import numpy as np
import pandas as pd

import config as C
from pipeline import features as F, ingestion, preprocess

rows = []
for ds in sys.argv[1:] or ["awid3", "rids"]:
    df = preprocess.clean(ingestion.load(ds))
    fc = F.frame_class(df["type"].values, df["subtype"].values)
    b = (fc["is_beacon"] > 0) & (df["src"].values == df["bssid"].values)
    n_ssid = df.loc[b].groupby("bssid")["ssid"].nunique()      # NaN SSIDs not counted
    n_ssid = n_ssid[n_ssid > 0]                                   # APs that beacon at least one SSID
    old = pd.read_parquet(C.OUT_DIR / f"windows_{ds}.parquet")
    new = F.build_windows(df)
    same_shape = old.shape == new.shape and list(old.columns) == list(new.columns)
    diff = {c: int((~np.isclose(old[c].to_numpy(float), new[c].to_numpy(float), equal_nan=True, rtol=0, atol=0)).sum())
            for c in F.CANDIDATES + ["tau", "y"]} if same_shape else {}
    sib = ["n_bssid_same_ssid", "rssi_sibling_diff", "ssid_newcomer"]
    rows.append(dict(dataset=ds, beaconing_aps_with_ssid=len(n_ssid), aps_multi_ssid=int((n_ssid > 1).sum()),
                     windows_old=len(old), windows_new=len(new), same_shape=same_shape,
                     cells_changed_ssid_features=sum(diff.get(c, -1) for c in sib),
                     cells_changed_all=sum(diff.values()) if diff else -1,
                     windows_changed=int(sum((~np.isclose(old[c].to_numpy(float), new[c].to_numpy(float), equal_nan=True, rtol=0, atol=0))
                                             for c in F.CANDIDATES).astype(bool).sum()) if same_shape else -1))
    print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv(C.OUT_DIR / "audit_ssid_lookahead.csv", index=False)
