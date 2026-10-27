"""
Stage 3: Feature Selection & Engineering. Dataset-agnostic: reads unified names only.

Unit of analysis (Ch. 6 draft, Sec. 6.0): one access point a (BSSID) in a sliding window
W(a, tau) = { frames i : bssid_i = a, tau - D <= t_i < tau },  D = 5 s, stride 1 s.
A frame-level model only learns "was this frame sent by device X" (Ch. 4); an Evil Twin is a
property of an AP over time and of its relation to the other AP with the same SSID.

  View A  aggregate vector per window       -> XGBoost, OC-SVM, Autoencoder     (build_windows)
  View B  last L frames of the AP per window -> LSTM                            (build_sequences)

Label: y(a, tau) = 1 iff W(a, tau) contains at least one attack frame.

Selection is fixed from Milestone 2 and the leakage audit; absolute time, BSSID/MAC, SSID and raw
wlan.seq are never features, only grouping/ordering keys. All windows end at tau and look back
only, so a window never uses future frames or labels.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

WIN_S = 5            # D: ~49 beacons at the 102.4 ms beacon interval (Ch. 6)
SEQ_LEN = 64         # L for View B (Ch. 6, Model 2)
SEQ_MOD = 4096       # 12-bit 802.11 sequence counter
MIN_FRAMES = 5       # windows with fewer frames carry no window statistics; also removes
                     # "ghost" BSSIDs from bit errors in the address field (1 frame each)
SKEW_MAX_PPM = 200   # |skew| above this = TSF reset, not a clock property
_SPAN = np.int64(10**12)   # composite-key spacing per AP (ms)

CANDIDATES = [
    # traffic mix and rate
    "n_frames", "share_data", "share_mgmt", "beacon_rate", "deauth_count",
    # MAC behaviour
    "beacon_iat_mean", "beacon_iat_std", "seq_anom", "retry_rate",
    "duration_mean", "duration_std", "length_mean", "length_std",
    # physical fingerprint
    "rssi_mean", "rssi_std", "clock_skew_ppm",
    # SSID context (relation to the same-SSID sibling)
    "n_bssid_same_ssid", "rssi_sibling_diff", "mfpc", "mfpc_sibling_diff", "ssid_newcomer",
]
# Pruned by the training-only selection audit (run_feature_audit.py -> feature_selection_manifest.csv,
# feature_redundancy_trainfolds.csv). Rule of Ch. 3.2, made explicit for the unified schema: a
# candidate is dropped only if |Spearman rho| > 0.95 with a kept feature on the TRAINING folds of
# BOTH datasets (never pooled over all windows, never using test labels). Two drops from the earlier
# all-window pass were dataset-specific and did NOT meet this bar, so they are RESTORED to the headline
# set (P0.1); they carry independent information on AWID3:
#   share_data       train-fold |rho| with duration_mean: AWID3 ~0.55 (independent), RIDS ~0.95-0.99
#   beacon_iat_mean  train-fold |rho| with beacon_rate:   AWID3 ~0.92-0.95, RIDS ~0.98
# The remaining drops are an exact complement on both datasets, or a priori forbidden shortcuts.
DROPPED = {
    "share_mgmt": "train-fold |rho| = 1.00 with share_data on both datasets (exact complement; keep share_data)",
    "mfpc_sibling_diff": "train-fold |rho| = 1.00 with mfpc (AWID3); undefined on RIDS (no second AP per SSID)",
    "mfpc": "a priori forbidden shortcut: attacker-set RSN/encryption configuration (same family as "
            "protected). Not in the headline model; added back only in the robustness ablation (+config)",
    "clock_skew_ppm": "a priori leakage shortcut: present in ~95% of attack vs ~5% of benign windows "
            "(RIDS), so its mere presence encodes the class. Excluded a priori, not on a performance score",
}
FEATURES = [c for c in CANDIDATES if c not in DROPPED]
ABLATION_CONFIG = ["mfpc"]   # encryption/RSN group, evaluated as "+config" with protected
KEYS = ["ap", "tau", "y", "frac_attack"]     # bookkeeping columns, never model inputs

# 802.11 frame class from the JOINT (type, subtype) pair: subtype alone is ambiguous,
# e.g. (0,8) = Beacon but (2,8) = QoS Data, (0,12) = Deauth but (2,12) = QoS Null.
FRAME_CLASSES = {
    "beacon": [(0, 8)], "probe_resp": [(0, 5)], "auth": [(0, 11)],
    "assoc": [(0, 0), (0, 1), (0, 2), (0, 3)], "deauth_disassoc": [(0, 10), (0, 12)],
    "data": [(2, 0), (2, 8)], "null": [(2, 4), (2, 12)],
}
STEP_FEATURES = (["log_dt", "dseq", "rssi", "length", "duration", "retry"]
                 + [f"is_{k}" for k in FRAME_CLASSES] + ["present"])


def frame_class(type_, subtype) -> dict:
    """One-hot frame class per frame from the joint (type, subtype) pair."""
    t, st = np.asarray(type_), np.asarray(subtype)
    return {f"is_{k}": np.any([(t == a) & (st == b) for a, b in pairs], axis=0).astype(np.float32)
            for k, pairs in FRAME_CLASSES.items()}


def _sorted(df: pd.DataFrame):
    """Order frames by (AP, time) and build the monotone integer key gid*SPAN + t_ms."""
    order = np.lexsort((df["t"].values, df["bssid"].values))
    d = df.iloc[order].reset_index(drop=True)
    t0 = df["t"].min()
    t_ms = np.round((d["t"].values - t0) * 1000).astype(np.int64)
    gid = np.unique(d["bssid"].values, return_inverse=True)[1].astype(np.int64)
    return d, gid, t_ms, t0


def _window_ends(gid, t_ms):
    """All (AP, tau) with at least one frame in [tau-D, tau): tau on a 1 s grid."""
    sec = t_ms // 1000
    g = np.repeat(gid, WIN_S)
    tau = (np.repeat(sec, WIN_S) + np.tile(np.arange(1, WIN_S + 1), len(sec))) * 1000
    w = np.unique(np.stack([g, tau], 1), axis=0)
    return w[:, 0], w[:, 1]


def _ffill_by(x, gid, shift=False):
    """Last non-NaN value of x within the same AP (strictly before the row if shift)."""
    sr = pd.Series(x)
    if shift:
        sr = sr.groupby(gid).shift(1)
    return sr.groupby(gid).ffill().values


def build_windows(df: pd.DataFrame) -> pd.DataFrame:
    """View A. df: cleaned unified frames. Returns one row per (AP, tau): CANDIDATES + KEYS."""
    d, gid, t_ms, t0 = _sorted(df)
    wg, wt = _window_ends(gid, t_ms)
    key = gid * _SPAN + t_ms
    lo = np.searchsorted(key, wg * _SPAN + wt - WIN_S * 1000, side="left")
    hi = np.searchsorted(key, wg * _SPAN + wt, side="left")

    def s(x):                                       # window sum, NaN ignored
        c = np.r_[0.0, np.cumsum(np.nan_to_num(np.asarray(x, float)))]
        return c[hi] - c[lo]

    def mean_std(x, mask=None):
        x = np.asarray(x, float)
        ok = ~np.isnan(x) if mask is None else (~np.isnan(x) & mask)
        xv = np.where(ok, x, 0.0)
        n = s(ok)
        m = s(xv) / np.maximum(n, 1)
        v = np.maximum(s(xv ** 2) / np.maximum(n, 1) - m ** 2, 0)
        return np.where(n > 0, m, np.nan), np.where(n > 1, np.sqrt(v), np.nan)

    own = d["src"].values == d["bssid"].values               # frames the AP itself sent
    fc = frame_class(d["type"].values, d["subtype"].values)
    mgmt, data = d["type"].values == 0, d["type"].values == 2
    beacon = (fc["is_beacon"] > 0) & own
    deauth = fc["is_deauth_disassoc"] > 0

    # beacon inter-arrival: time since the AP's previous beacon (s)
    bt = np.where(beacon, t_ms, np.nan)
    b_iat = np.where(beacon, (t_ms - _ffill_by(bt, gid, shift=True)) / 1000.0, np.nan)

    # sequence-number continuity of the AP's own frames: 1[(seq_i - seq_prev) mod 4096 > 1]
    seq = np.where(own, d["seq"].values, np.nan)
    gap = np.mod(seq - _ffill_by(seq, gid, shift=True), SEQ_MOD)
    anom = np.where(np.isnan(gap), np.nan, (gap > 1).astype(float))

    # clock skew: least-squares slope of TSF offset vs receive time over the window's beacons
    has = beacon & ~np.isnan(d["tsf"].values)
    t_rel = (t_ms - pd.Series(t_ms).groupby(gid).transform("first").values) / 1000.0   # centred per AP
    x = np.where(has, t_rel, np.nan)
    off = d["tsf"].values / 1e6 - t_ms / 1000.0
    off = off - pd.Series(np.where(has, off, np.nan)).groupby(gid).transform("first").values
    yv = np.where(has, off, np.nan)
    n_b = np.maximum(s(has), 1)
    sx, sy = s(x) / n_b, s(yv) / n_b
    var = s(x * x) / n_b - sx ** 2
    cov = s(x * yv) / n_b - sx * sy
    skew = np.where((s(has) >= 10) & (var > 1e-6), 1e6 * cov / np.where(var > 1e-6, var, 1), np.nan)
    skew[np.abs(skew) > SKEW_MAX_PPM] = np.nan

    n = s(np.ones(len(d)))
    rssi_m, rssi_s = mean_std(d["signal"].values, own)
    dur_m, dur_s = mean_std(d["duration"].values)
    len_m, len_s = mean_std(d["length"].values)
    iat_m, iat_s = mean_std(b_iat)
    anom_m, _ = mean_std(anom)
    mfpc_m, _ = mean_std(d["mfpc"].values, beacon)
    aps = np.unique(d["bssid"].values)

    W = pd.DataFrame({
        "n_frames": n / WIN_S,
        "share_data": s(data) / n,
        "share_mgmt": s(mgmt) / n,
        "beacon_rate": s(beacon) / WIN_S,
        "deauth_count": s(deauth),
        "beacon_iat_mean": iat_m, "beacon_iat_std": iat_s,
        "seq_anom": anom_m,
        "retry_rate": s(d["retry"].values) / n,
        "duration_mean": dur_m, "duration_std": dur_s,
        "length_mean": len_m, "length_std": len_s,
        "rssi_mean": rssi_m, "rssi_std": rssi_s,
        "clock_skew_ppm": skew,
        "mfpc": mfpc_m,
        "ap": aps[wg],
        "tau": t0 + wt / 1000.0,
        "y": (s(d["y"].values) > 0).astype(np.int8),
        "frac_attack": s(d["y"].values) / n,
    })
    W["ssid"] = _ssid_at(d, gid, beacon, hi)
    W["ssid_newcomer"] = _newcomer(W["ssid"].values, wg, d, gid, t_ms, beacon, wt)
    W = _sibling_features(W, s(beacon) > 0)
    W = W[n >= MIN_FRAMES]
    return W[CANDIDATES + KEYS].reset_index(drop=True)


def _ssid_at(d, gid, beacon, hi):
    """SSID of each window = SSID of the AP's latest own beacon before tau (frames < hi only;
    NaN if the AP has not beaconed yet). Past-only: later beacons never change an earlier window."""
    last = _ffill_by(np.where(beacon, d["ssid"].values, np.nan), gid)
    return last[hi - 1]                     # every window holds >= 1 frame of its own AP


def _newcomer(ssid_w, wg, d, gid, t_ms, beacon, wt):
    """1 if another AP had beaconed the window's SSID before this AP first beaconed it.
    Asymmetric cue (the sibling diffs are mirror images for the two APs of a pair). Both first-seen
    times are <= the AP's own first beacon of that SSID, which precedes tau, so no future frame is used.
    0 when the window has no SSID yet."""
    b = pd.DataFrame({"g": gid[beacon], "ssid": d["ssid"].values[beacon], "t": t_ms[beacon]}).dropna()
    first = b.groupby(["g", "ssid"])["t"].min()
    earliest = first.groupby(level="ssid").min()
    ok = ~np.isnan(ssid_w)
    f = np.full(len(ssid_w), np.nan)
    e = np.full(len(ssid_w), np.nan)
    if ok.any():
        idx = pd.MultiIndex.from_arrays([wg[ok], ssid_w[ok]])
        f[ok] = first.reindex(idx).values
        e[ok] = earliest.reindex(ssid_w[ok]).values
    return np.where(ok & (wt > np.nan_to_num(f, nan=np.inf)), (f > e).astype(float), 0.0)


def _sibling_features(W: pd.DataFrame, has_beacon) -> pd.DataFrame:
    """Compare each AP window with the other APs beaconing the same SSID in the same window:
    n_bssid_same_ssid = |{a' : ssid(a') = ssid(a), a' beacons in W}|,
    x_sibling_diff = x(a) - mean_{a' != a} x(a')   (NaN when the AP has no sibling; the
    count feature already says whether a sibling exists). Both are NaN when a does not beacon."""
    W = W.copy()
    adv = np.asarray(has_beacon) & W["ssid"].notna().values
    A = W.loc[adv]
    g = A.groupby(["ssid", "tau"])
    W["n_bssid_same_ssid"] = np.nan                  # undefined when the AP is not beaconing
    W.loc[adv, "n_bssid_same_ssid"] = g["ap"].transform("nunique")
    for col, out in [("rssi_mean", "rssi_sibling_diff"), ("mfpc", "mfpc_sibling_diff")]:
        v = A[col]
        others = g[col].transform("count") - v.notna().astype(int)
        sib = (g[col].transform("sum") - v.fillna(0)) / others.where(others > 0)
        W[out] = np.nan
        W.loc[adv, out] = (v - sib).where(others > 0)
    return W


def build_sequences(df: pd.DataFrame, W: pd.DataFrame) -> np.ndarray:
    """View B: for each window row of W, SEQ_LEN frames of that AP inside the SAME window [tau - D, tau)
    (so the purge gap of D also covers View B): all of them if the window has at most SEQ_LEN frames,
    otherwise SEQ_LEN frames evenly spaced over the window. Returned as a
    (n_windows, SEQ_LEN, len(STEP_FEATURES)) float32 array, zero-padded at the start.
    dseq uses only the AP's own frames (one sequence counter); missing values stay NaN for the
    train-fitted sequence scaler. The last channel is 1 for a real frame, 0 for padding."""
    d, gid, t_ms, t0 = _sorted(df)
    aps = np.unique(d["bssid"].values)
    key = gid * _SPAN + t_ms
    wg = np.searchsorted(aps, W["ap"].values)
    wt = np.round((W["tau"].values - t0) * 1000).astype(np.int64)
    hi = np.searchsorted(key, wg * _SPAN + wt, side="left")
    lo = np.searchsorted(key, wg * _SPAN + wt - WIN_S * 1000, side="left")

    same = np.r_[False, gid[1:] == gid[:-1]]
    own = d["src"].values == d["bssid"].values
    seq = np.where(own, d["seq"].values, np.nan)
    step = np.column_stack([
        np.where(same, np.log1p(np.maximum(np.diff(t_ms, prepend=0), 0) / 1000.0), np.nan),
        np.mod(seq - _ffill_by(seq, gid, shift=True), SEQ_MOD),
        np.where(own, d["signal"].values, np.nan), d["length"].values, d["duration"].values,
        d["retry"].values, *frame_class(d["type"].values, d["subtype"].values).values(),
        np.ones(len(d)),
    ]).astype(np.float32)
    out = np.zeros((len(W), SEQ_LEN, step.shape[1]), np.float32)
    n = hi - lo
    for k in range(SEQ_LEN):                       # position SEQ_LEN-1 = most recent frame
        # n <= SEQ_LEN: all frames, zero-padded at the start; n > SEQ_LEN: SEQ_LEN frames evenly
        # spaced over the whole window (stride n / SEQ_LEN), so the sequence spans [tau - D, tau)
        # instead of only its last few milliseconds on busy APs.
        idx = np.where(n > SEQ_LEN, lo + (k + 1) * n // SEQ_LEN - 1, hi - SEQ_LEN + k)
        ok = idx >= lo
        out[ok, k, :] = step[idx[ok]]
    return out


def audit(X: pd.DataFrame, y: np.ndarray) -> pd.DataFrame:
    """Per feature: class medians, single-feature ROC-AUC (= Mann-Whitney effect size U/(n1 n0)),
    Cliff's delta = 2 AUC - 1, missing share per class. AUC_1 = max(AUC, 1 - AUC) > 0.99 for one
    field is flagged for review, not proof of leakage: each flag is checked semantically and with
    the time-blocked split (a strong physical feature can be ~1.0 legitimately)."""
    from sklearn.metrics import roc_auc_score
    rows = []
    for c in X.columns:
        x = X[c].values.astype(float)
        ok = ~np.isnan(x)
        r = {"feature": c, "missing_benign": 1 - ok[y == 0].mean(), "missing_attack": 1 - ok[y == 1].mean()}
        if ok[y == 1].sum() > 1 and ok[y == 0].sum() > 1 and np.nanstd(x) > 0:
            auc = roc_auc_score(y[ok], x[ok])
            r.update(median_benign=np.median(x[ok & (y == 0)]), median_attack=np.median(x[ok & (y == 1)]),
                     auc=auc, auc_1=max(auc, 1 - auc), cliffs_delta=2 * auc - 1)
        rows.append(r)
    a = pd.DataFrame(rows)
    a["review"] = a.get("auc_1", pd.Series(np.nan, index=a.index)) > 0.99
    return a


def ap_baseline_delta(W: pd.DataFrame, cols, lo_s=300, hi_s=WIN_S) -> pd.DataFrame:
    """2.2 fix 4: x - median of the same AP's windows with tau in [t - lo_s, t - hi_s] (look-back only;
    hi_s = D, so the baseline shares no frame with the window). NaN when the AP has no such history."""
    out = W.copy()
    for ap, g in W.groupby("ap", sort=False):
        g = g.sort_values("tau")
        t = g["tau"].to_numpy(float)
        a = np.searchsorted(t, t - lo_s, side="left")
        b = np.searchsorted(t, t - hi_s, side="right")
        X = g[cols].to_numpy(float)
        med = np.full_like(X, np.nan)
        for i in np.where(b > a)[0]:
            med[i] = np.nanmedian(X[a[i]:b[i]], 0)
        out.loc[g.index, cols] = X - med
    return out
