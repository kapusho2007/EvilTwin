"""Stages 2-3 driver: unified frames -> clean -> AP-window features (View A) -> feature audit.
Usage: python run_features.py awid3 rids
Writes output/windows_<dataset>.parquet/.csv (CANDIDATES + ap, tau, y, frac_attack),
output/sequences_<dataset>.npz (View B, compressed), output/feature_audit_<dataset>.csv and
output/spearman_<dataset>.csv (the evidence for the final pruning)."""
import sys

import numpy as np
import pandas as pd

import config as C
from pipeline import ingestion, preprocess, features

if __name__ == "__main__":
    C.OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ds in sys.argv[1:] or ["awid3", "rids"]:
        raw = ingestion.load(ds)
        df = preprocess.clean(raw)
        W = features.build_windows(df)
        W.to_parquet(C.OUT_DIR / f"windows_{ds}.parquet", index=False)
        W.to_csv(C.OUT_DIR / f"windows_{ds}.csv", index=False)
        np.savez_compressed(C.OUT_DIR / f"sequences_{ds}.npz", S=features.build_sequences(df, W))
        rho = W[features.CANDIDATES].corr(method="spearman")
        pairs = [(a, b, rho.loc[a, b]) for i, a in enumerate(rho) for b in rho.columns[i + 1:]]
        pd.DataFrame(pairs, columns=["feature_a", "feature_b", "spearman_rho"]).sort_values(
            "spearman_rho", key=abs, ascending=False).to_csv(C.OUT_DIR / f"spearman_{ds}.csv", index=False)
        a = features.audit(W[features.CANDIDATES], W["y"].values)
        a.to_csv(C.OUT_DIR / f"feature_audit_{ds}.csv", index=False)
        print(f"[{ds}] frames raw={len(raw):,} clean={len(df):,} attack_frames={int(df.y.sum()):,} | "
              f"APs={W.ap.nunique():,} windows={len(W):,} attack_windows={int(W.y.sum()):,}\n"
              f"{a.round(3).to_string()}\n", flush=True)
