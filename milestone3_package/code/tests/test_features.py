"""Unit tests for the joint (type, subtype) frame classes and the causal windows.
Run: python -m pytest tests"""
import numpy as np
import pandas as pd

from pipeline import features as F


def test_frame_class_joint_type_subtype():
    t = np.array([0, 0, 0, 2, 2])
    st = np.array([8, 11, 12, 8, 12])
    fc = F.frame_class(t, st)
    assert fc["is_beacon"].tolist() == [1, 0, 0, 0, 0]          # (0,8) Beacon, (2,8) is QoS Data
    assert fc["is_auth"].tolist() == [0, 1, 0, 0, 0]
    assert fc["is_deauth_disassoc"].tolist() == [0, 0, 1, 0, 0]  # (2,12) is QoS Null, not Deauth
    assert fc["is_data"].tolist() == [0, 0, 0, 1, 0]
    assert fc["is_null"].tolist() == [0, 0, 0, 0, 1]


def _frames(n=60, ap=1, start=0.0, period=0.1, y=0):
    t = start + period * np.arange(n)
    return pd.DataFrame({"t": t, "src": ap, "bssid": ap, "length": 100.0, "duration": 0.0,
                         "retry": 0.0, "type": 0.0, "subtype": 8.0, "signal": -40.0,
                         "seq": np.arange(n) % 4096 * 1.0, "mfpc": 1.0, "tsf": t * 1e6,
                         "ssid": 7.0, "protected": 0.0, "y": y})


def test_windows_are_causal_and_count_beacons():
    W = F.build_windows(_frames())
    w = W[W.tau == W.tau.min() + 5].iloc[0]                    # full 5 s window
    assert abs(w.beacon_rate - 10.0) < 0.3                       # 0.1 s period -> ~10 beacons/s
    assert w.seq_anom == 0.0                                     # consecutive sequence numbers
    assert w.y == 0


def test_label_and_future_frames():
    df = pd.concat([_frames(), _frames(n=10, start=8.0, y=1)], ignore_index=True)
    W = F.build_windows(df)
    assert W.loc[W.tau <= df.t.min() + 8.0, "y"].sum() == 0      # no window sees the future attack
    assert W.y.max() == 1


def test_ssid_features_are_past_only():
    """P0.2: AP 1 advertises SSID A, later switches to SSID B. Early windows must be identical
    with or without the later traffic (no whole-capture SSID mode)."""
    a1 = _frames(n=100, ap=1)                                    # 0..10 s, SSID 7
    a2 = _frames(n=100, ap=2, start=2.0); a2["ssid"] = 7.0       # sibling AP, same SSID, later
    late = _frames(n=200, ap=1, start=10.0); late["ssid"] = 9.0  # 10..30 s: AP 1 switches SSID
    early = F.build_windows(pd.concat([a1, a2], ignore_index=True))
    full = F.build_windows(pd.concat([a1, a2, late], ignore_index=True))
    cut = 10.0                                                   # windows [tau-5, tau) before the switch
    e = early[early.tau <= cut].sort_values(["ap", "tau"]).reset_index(drop=True)
    f = full[full.tau <= cut].sort_values(["ap", "tau"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(e, f)
    w = e[(e.ap == 2) & (e.tau == cut)].iloc[0]
    assert w.n_bssid_same_ssid == 2 and w.ssid_newcomer == 1    # AP 2 joined SSID 7 later
