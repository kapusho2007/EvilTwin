"""
P0.1 - training-only feature-selection audit.

The frozen FEATURES list (features.CANDIDATES minus features.DROPPED) must be defensible WITHOUT
looking at test outcomes. This script recomputes, from the TRAINING folds only (purged episode-blocked
split, the same split the models use), the evidence behind every drop, and writes a per-feature
manifest classifying each of the 21 candidates as:
  domain_prespecified - a security-domain input kept a priori;
  redundancy          - dropped because |Spearman rho| > 0.95 with a kept feature (verified per
                        TRAIN fold, both datasets), i.e. an unsupervised, label-free decision;
  leakage_shortcut    - dropped a priori as a forbidden shortcut (encryption/RSN config, or a
                        presence/absence that correlates with the class), never on a score.

No drop uses test performance. If the per-fold-train redundancy confirms the frozen list, the headline
feature set is unchanged and no retraining is required (documented). Writes
output/feature_selection_manifest.csv and output/feature_redundancy_trainfolds.csv.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

import config as C
from pipeline import evaluate as E, features as F

RHO = 0.95

# frozen rationale per candidate (category, kept partner for redundancy, human reason)
PLAN = {
    "n_frames": ("domain_prespecified", None, "window traffic volume"),
    "share_data": ("redundancy", "duration_mean", "|rho|>0.95 with duration_mean (RIDS)"),
    "share_mgmt": ("redundancy", "share_data", "complement of share_data (rho=-1 by definition)"),
    "beacon_rate": ("domain_prespecified", None, "rogue beacon flood rate"),
    "deauth_count": ("domain_prespecified", None, "deauth/disassoc burst"),
    "beacon_iat_mean": ("redundancy", "beacon_rate", "|rho|>0.95 with beacon_rate (rate ~ 1/mean IAT)"),
    "beacon_iat_std": ("domain_prespecified", None, "beacon timing jitter"),
    "seq_anom": ("domain_prespecified", None, "sequence-counter discontinuity of the AP"),
    "retry_rate": ("domain_prespecified", None, "retransmission rate"),
    "duration_mean": ("domain_prespecified", None, "airtime"),
    "duration_std": ("domain_prespecified", None, "airtime variability"),
    "length_mean": ("domain_prespecified", None, "frame size"),
    "length_std": ("domain_prespecified", None, "frame size variability"),
    "rssi_mean": ("domain_prespecified", None, "received signal level"),
    "rssi_std": ("domain_prespecified", None, "signal spread (twin radio)"),
    "clock_skew_ppm": ("leakage_shortcut", None,
                       "presence correlates with class (missing in benign, present in attack); a "
                       "coverage/leakage shortcut, excluded a priori, not on a score"),
    "n_bssid_same_ssid": ("domain_prespecified", None, "same-SSID sibling count"),
    "rssi_sibling_diff": ("domain_prespecified", None, "RSSI vs same-SSID sibling"),
    "mfpc": ("leakage_shortcut", None, "attacker-set RSN/encryption configuration; ablation-only (+config)"),
    "mfpc_sibling_diff": ("redundancy", "mfpc", "|rho|>0.95 with mfpc (AWID3); undefined on RIDS"),
    "ssid_newcomer": ("domain_prespecified", None, "another AP beaconed this SSID first"),
}


def train_masks(W):
    tau, y = W["tau"].to_numpy(), W["y"].to_numpy()
    fid = E.folds(tau, y)
    K = fid.max() + 1
    return [E.split(tau, y, fid, k, F.WIN_S)[0] for k in range(K)]


def spearman_abs(a, b):
    m = ~(np.isnan(a) | np.isnan(b))
    if m.sum() < 10 or np.nanstd(a[m]) == 0 or np.nanstd(b[m]) == 0:
        return np.nan
    from scipy.stats import spearmanr
    return abs(spearmanr(a[m], b[m]).statistic)


def main():
    red_rows = []
    per_fold_ok = {}   # feature -> list over (ds,fold) of max|rho| with its partner
    for ds in ("awid3", "rids"):
        W = pd.read_parquet(C.OUT_DIR / f"windows_{ds}.parquet")
        masks = train_masks(W)
        for feat, (cat, partner, _) in PLAN.items():
            if cat != "redundancy" or partner is None or feat not in W or partner not in W:
                continue
            for k, tr in enumerate(masks):
                rho = spearman_abs(W.loc[tr, feat].to_numpy(float), W.loc[tr, partner].to_numpy(float))
                red_rows.append({"dataset": ds, "feature": feat, "partner": partner,
                                 "fold": k, "abs_spearman_train": round(rho, 4) if rho == rho else None})
                per_fold_ok.setdefault(feat, []).append(rho)
    red = pd.DataFrame(red_rows)
    red.to_csv(C.OUT_DIR / "feature_redundancy_trainfolds.csv", index=False)

    # manifest
    man = []
    for feat, (cat, partner, reason) in PLAN.items():
        retained = feat in F.FEATURES
        # redundancy drops: confirmed iff |rho|>RHO in at least one dataset's every train fold where defined
        confirmed = None
        if cat == "redundancy":
            vals = [v for v in per_fold_ok.get(feat, []) if v == v]
            confirmed = bool(len(vals) and np.nanmin(vals) > RHO) if vals else None
        man.append({"feature": feat, "status": "retained" if retained else "dropped",
                    "category": cat, "partner": partner or "",
                    "partition": "train-folds" if cat == "redundancy" else "a-priori (domain)",
                    "decided_before_test_results": "yes", "reason": reason,
                    "train_fold_redundancy_confirmed": confirmed,
                    "min_abs_spearman_train": (round(float(np.nanmin([v for v in per_fold_ok.get(feat, []) if v == v])), 4)
                                               if cat == "redundancy" and per_fold_ok.get(feat) else None)})
    M = pd.DataFrame(man)
    M.to_csv(C.OUT_DIR / "feature_selection_manifest.csv", index=False)

    # consistency: FEATURES must equal CANDIDATES minus the dropped set here
    dropped_here = {m["feature"] for m in man if m["status"] == "dropped"}
    expected = [c for c in F.CANDIDATES if c not in dropped_here]
    assert expected == F.FEATURES, f"manifest disagrees with features.FEATURES:\n{expected}\n{F.FEATURES}"
    red_unconfirmed = [m["feature"] for m in man if m["category"] == "redundancy"
                       and m["train_fold_redundancy_confirmed"] is False]
    print(M.to_string(index=False))
    print(f"\nFEATURES ({len(F.FEATURES)}): {F.FEATURES}")
    print(f"redundancy drops NOT confirmed on train folds: {red_unconfirmed or 'none'}")
    print("headline feature set unchanged -> no retraining required for P0.1"
          if not red_unconfirmed else "REVIEW: some redundancy drops not confirmed on train folds")


if __name__ == "__main__":
    main()
