"""Rebuild View B only (windows unchanged): python rebuild_sequences.py [awid3 rids]"""
import sys

import numpy as np
import pandas as pd

import config as C
from pipeline import features, ingestion, preprocess

for ds in sys.argv[1:] or ["awid3", "rids"]:
    W = pd.read_parquet(C.OUT_DIR / f"windows_{ds}.parquet")
    S = features.build_sequences(preprocess.clean(ingestion.load(ds)), W)
    np.savez_compressed(C.OUT_DIR / f"sequences_{ds}.npz", S=S)
    pad = (S[..., -1] == 0).sum(1)
    y = W["y"].to_numpy()
    print(f"[{ds}] {S.shape}; windows with > 64 frames: attack {np.mean(pad[y == 1] == 0):.3f}, "
          f"benign {np.mean(pad[y == 0] == 0):.3f}", flush=True)
