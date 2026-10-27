"""
Central configuration. The only place that holds dataset paths, raw column names and labels.
Only pipeline/ingestion.py reads the raw-schema entries (Step-7 dependency rule).

Datasets (Milestone 2, Ch. 5):
  * AWID3 (Dataset A): 802.11 L1+L2, WPA2. Positive class = Evil_Twin.
  * RIDS  (Dataset B): 802.11 L1+L2, WPA3 (Saini et al. 2022). Positive class = RogueAP.
    Its 'Evil_Twin' and 'Krack' rows are AWID3 copies (GTB timestamps) and are removed;
    Deauth and Beacon Flood are other attacks, out of scope, and are removed (never folded into benign).
"""
from pathlib import Path

HERE = Path(__file__).resolve()
ROOT = HERE.parents[1]                                    # .../Milestone3_v2
WORKSHOP = ROOT.parent                                    # .../workshop
DATA_DIR = WORKSHOP / "EvilTwin" / "Data"
UNIFIED_DIR = ROOT / "data_unified"                       # ingestion output (unified schema)
OUT_DIR = ROOT / "output"

AWID3_CSV_GLOB = str(DATA_DIR / "AWID3_Evil_Twin" / "Evil_Twin_*.csv")
RIDS_CSV = str(DATA_DIR / "RIDS_WPA3" / "merged_full.csv")

LABEL_COL = "Label"
AWID3_LABELS = {"Normal": 0, "Evil_Twin": 1}              # anything else is dropped
RIDS_LABELS = {"Normal": 0, "RogueAP": 1}
RIDS_FOREIGN_TZ = "GTB"                                   # AWID3-origin rows inside RIDS (provenance audit)

# Raw 802.11 field -> unified name. Identical names in both datasets (Milestone 2, Ch. 5).
RAW_TO_UNIFIED = {
    "frame.time": "t",                     # parsed to seconds; ordering/window key only, never a feature
    "wlan.ta": "src",                      # transmitter MAC; grouping key only, never a feature
    "wlan.sa": "src_fallback",
    "wlan.bssid": "bssid",                 # AP address; grouping key only (unit = one AP x window)
    "frame.len": "length",
    "wlan.duration": "duration",
    "wlan.fc.retry": "retry",
    "wlan.fc.ds": "ds",
    "wlan.fc.type": "type",
    "wlan.fc.subtype": "subtype",
    "wlan_radio.signal_dbm": "signal",     # physical per-frame dBm (radiotap.dbm_antsignal is a 3-chain sum)
    "wlan.seq": "seq",                     # raw value leaks (Ch. 4); used only as per-source gaps
    "wlan.rsn.capabilities.mfpc": "mfpc",
    "wlan.fixed.timestamp": "tsf",         # beacon TSF (us); used only for the AP clock-skew estimate
    "wlan.ssid": "ssid",                   # hashed; only to count transmitters advertising one SSID
    "wlan.fc.protected": "protected",      # kept only for the robustness ablation, not a model feature
}

RANDOM_STATE = 42
CHUNK_ROWS = 500_000
