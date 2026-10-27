"""
Stage 1: Data Ingestion. The ONLY dataset-aware module (Step-7 dependency rule).

Reads the raw Wireshark CSVs in chunks and writes one unified frame table per dataset:

    t, src, bssid, length, duration, retry, ds, type, subtype, signal, seq, mfpc, tsf, ssid, protected, y

All later stages read only these unified names.
"""
from __future__ import annotations
import glob
import io
import os
import re
import zlib
import numpy as np
import pandas as pd

import config as C

_TIME_RE = re.compile(r"^(\w{3})\s+(\d+)[-,]?\s+(\d{4})\s+(\d{2}:\d{2}:\d{2}\.\d+)")


def _parse_time(s: pd.Series) -> pd.Series:
    """'Dec 15- 2020 23:03:14.8016 GTB Standard Time' (AWID3) or 'Apr 11, 2022 07:41:05.3668 IST' (RIDS) -> seconds.
    The time zone is ignored: only differences inside one capture are used."""
    p = s.astype(str).str.extract(_TIME_RE)
    txt = p[0] + " " + p[1] + " " + p[2] + " " + p[3].str[:15]
    dt = pd.to_datetime(txt, format="%b %d %Y %H:%M:%S.%f", errors="coerce")
    return (dt - pd.Timestamp("2000-01-01")).dt.total_seconds()


def _mac_to_int(s: pd.Series) -> pd.Series:
    h = s.astype("string").str.replace(":", "", regex=False)
    return h.map(lambda v: int(v, 16) if isinstance(v, str) and len(v) == 12 else -1).astype("int64")


def _num(s: pd.Series) -> pd.Series:
    """Numeric parse; accepts hex strings ('0x02') used by some Wireshark exports."""
    out = pd.to_numeric(s, errors="coerce")
    bad = out.isna() & s.notna()
    if bad.any():
        out[bad] = s[bad].map(lambda v: _hex(v))
    return out.astype("float64")


def _hex(v):
    try:
        return float(int(str(v), 0))
    except ValueError:
        return np.nan


def _crc(v):
    """Stable integer code for an SSID string."""
    return float(zlib.crc32(v.encode())) if isinstance(v, str) and v else np.nan


def _unify(chunk: pd.DataFrame, labels: dict) -> pd.DataFrame:
    chunk = chunk[chunk[C.LABEL_COL].isin(labels)]
    u = pd.DataFrame(index=chunk.index)
    u["t"] = _parse_time(chunk["frame.time"])
    src = chunk["wlan.ta"].fillna(chunk["wlan.sa"])
    u["src"] = _mac_to_int(src)
    u["bssid"] = _mac_to_int(chunk["wlan.bssid"])
    for raw, name in C.RAW_TO_UNIFIED.items():
        if name in ("t", "src", "src_fallback", "bssid"):
            continue
        if raw not in chunk:
            u[name] = np.float32(np.nan)
        elif name == "ssid":
            u[name] = chunk[raw].map(_crc)
        else:
            u[name] = _num(chunk[raw])
    u["y"] = chunk[C.LABEL_COL].map(labels).astype("int8")
    return u.dropna(subset=["t"])


def _read(files, labels, drop_tz=None):
    cols = set(C.RAW_TO_UNIFIED) | {C.LABEL_COL}
    for f in files:
        for chunk in pd.read_csv(f, usecols=lambda c: c in cols, dtype=str,
                                 chunksize=C.CHUNK_ROWS, low_memory=False):
            yield _filter(chunk, labels, drop_tz)


def _read_range(path, k, n, labels, drop_tz=None, block=128 << 20):
    """Byte range k of n of one large CSV: the lines whose first byte lies in [s_k, s_k+1).
    Lets the 10 GB RIDS file be ingested in independent, restartable pieces."""
    cols = set(C.RAW_TO_UNIFIED) | {C.LABEL_COL}
    size = os.path.getsize(path)
    start, end = size * k // n, size * (k + 1) // n
    with open(path, "rb") as f:
        header = f.readline()
        if start > 0:
            f.seek(start - 1)
            f.readline()                                  # finish the line that straddles start
        while f.tell() < end:
            data = f.read(min(block, end - f.tell()))
            if not data.endswith(b"\n"):
                data += f.readline()                      # the last line starting before end
            chunk = pd.read_csv(io.BytesIO(header + data), usecols=lambda c: c in cols,
                                dtype=str, low_memory=False)
            yield _filter(chunk, labels, drop_tz)


def _filter(chunk, labels, drop_tz):
    if drop_tz:
        chunk = chunk[~chunk["frame.time"].str.contains(drop_tz, na=False)]
    return _unify(chunk, labels)


def ingest(dataset: str, k: int = 0, n: int = 1, out_dir=None) -> list:
    """Write piece k of n of the unified table for dataset in {'awid3','rids'}. Returns part paths.
    AWID3: piece k = files k, k+n, ...   RIDS: piece k = byte range k of the single CSV."""
    if dataset == "awid3":
        key = lambda p: int(re.findall(r"(\d+)\.csv$", p)[0])
        files = sorted(glob.glob(C.AWID3_CSV_GLOB), key=key)
        parts = _read(files[k::n], C.AWID3_LABELS)
    elif dataset == "rids":
        parts = _read_range(C.RIDS_CSV, k, n, C.RIDS_LABELS, drop_tz=C.RIDS_FOREIGN_TZ)
    else:
        raise ValueError(dataset)

    out = (out_dir or C.UNIFIED_DIR) / dataset
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob(f"part-{k:03d}-*"):
        old.unlink()                                      # re-running a piece replaces it
    ext = _ext()
    written = []
    for j, u in enumerate(parts):
        if u.empty:
            continue
        p = out / f"part-{k:03d}-{j:04d}.{ext}"
        _write(u, p)
        written.append(p)
        print(f"  [{dataset}] piece {k}/{n} part {j}: {len(u):,} rows, attacks={int(u.y.sum()):,}", flush=True)
    return written


def load(dataset: str, unified_dir=None) -> pd.DataFrame:
    """Read the unified table back, time-ordered."""
    d = (unified_dir or C.UNIFIED_DIR) / dataset
    files = sorted(d.glob("part-*.*"))
    df = pd.concat([_readpart(f) for f in files], ignore_index=True)
    return df.sort_values("t", kind="stable").reset_index(drop=True)


def _ext():
    try:
        import pyarrow  # noqa: F401
        return "parquet"
    except ImportError:
        return "csv.gz"


def _write(df, path):
    if str(path).endswith(".parquet"):
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False, compression="gzip")


def _readpart(path):
    return pd.read_parquet(path) if str(path).endswith(".parquet") else pd.read_csv(path)
