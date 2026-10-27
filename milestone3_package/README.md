# Milestone 3 submission: Evil Twin / Rogue AP detection (MITRE T1557.004)

Group 3: Itay Rubanenko, Shaked Sarbstein.
Datasets: AWID3 (Evil_Twin subset) and RIDS (RogueAP class).

## Contents

```
report/   Milestone3_Report.docx   graded report (≤ 15 body pages + appendices)
          Milestone3_Report.pdf    same report as PDF
code/     the pipeline (Section 1.1): run_ingest → run_features → run_train → run_errors → run_cascade → run_llm
          pipeline/   ingestion (only dataset-aware module), preprocess, features, models, train, evaluate, cascade, llm
          tests/      unit tests (python -m pytest tests)
          requirements.txt  pinned versions
output/   every result table, log and out-of-fold prediction cited in the report; window tables (Parquet)
          models/{awid3,rids}/   saved fold models (<model>_fold<k>.joblib = model + fitted normalizer + feature list)
```

## Reproduce

```
pip install -r code/requirements.txt
cd code
python run_ingest.py awid3 && python run_ingest.py rids    # raw CSVs → data_unified/ (paths in config.py)
python run_features.py awid3 rids                          # windows_<ds>.parquet, sequences_<ds>.npz
python run_train.py main                                   # all models, 5 folds → output/, output/models/
# then run each experiment separately (choose any; do NOT pipe them with '|'):
python run_train.py rule_gain
python run_train.py imbalance
python run_train.py sensitivity          # add --profile full for all 4 models x both datasets (P0.3)
python run_train.py transfer
python run_train.py optimise
python run_feature_audit.py                                # P0.1 training-only feature-selection manifest
python run_forward.py                                      # P0.2 forward-only within-episode evaluation
python run_part1_checks.py                                 # feature-group ablation, window length D (needs raw frames)
python run_errors.py
python run_cascade.py
python run_llm.py prepare                                  # escalated windows -> llm_cases_<ds>.jsonl
python run_llm.py check                                    # Ollama model metadata + parser tests
python run_llm.py run                                      # needs a local Ollama server (Section 4.1)
python run_llm.py ablate
python run_llm.py summarise
python run_cascade.py                                      # re-run to add the "+ LLM" rows
python make_report_docx.py                                 # rebuilds the report
```

## Raw data (not included: size and licence)

| | AWID3 (Dataset A) | RIDS (Dataset B) |
|---|---|---|
| Source | AWID3, Evil_Twin CSV subset (request from the AWID project, University of the Aegean) | RIDS WPA3 (Saini et al. 2022), `merged_full.csv` |
| Expected path | `../EvilTwin/Data/AWID3_Evil_Twin/Evil_Twin_<n>.csv` (76 files) | `../EvilTwin/Data/RIDS_WPA3/merged_full.csv` |
| Labels kept | Normal = 0, Evil_Twin = 1 | Normal = 0, RogueAP = 1 |
| Removed at ingest | other labels | Evil_Twin and Krack rows (AWID3 copies, `GTB` timestamps), Deauth, Beacon Flood |
| Frames ingested → after cleaning | 3,778,679 → 1,609,803 (104,826 attack) | 4,443,584 → 1,571,120 (25,129 attack) |

Paths live only in `code/config.py` (`DATA_DIR`); `../` is the folder that holds this one.
Cleaning (`pipeline/preprocess.py`): exact duplicates on (t, src, seq, subtype, length) are dropped
(RIDS stores every frame twice), out-of-range 802.11 values become NaN, frames with no or broadcast BSSID are dropped.
Check the counts after ingest with `python run_features.py awid3 rids` (it prints raw / clean / attack frames;
compare with `output/run_features.log`).

```
cd code
python run_ingest.py awid3                                  # → ../data_unified/awid3/
for k in $(seq 0 9); do python run_ingest.py rids $k 10; done   # 10 restartable pieces of the 10 GB CSV
python run_features.py awid3 rids
```

Not independently verified: a full raw-CSV → report rerun on a second machine. The window tables,
models and logs in `output/` are the ones the report cites.

## Tests

```
cd code && python -m pytest tests -v          # 17 tests; needs pyarrow (Parquet windows) and torch
```
Log of the last run: `output/test_log.txt`. `python audit_ssid_lookahead.py` re-checks that the past-only
SSID features reproduce `windows_<ds>.parquet` (needs `../data_unified/`).
`code/` expects `output/` next to it; this folder keeps that layout.
