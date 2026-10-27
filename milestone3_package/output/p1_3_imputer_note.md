# P1.3 — SimpleImputer "Skipping features without any observed values" on RIDS

Read-only investigation. No code, config, report, or model was modified.

## Summary

The warning is **benign**. It fires because exactly one headline feature,
`rssi_sibling_diff`, is **100% NaN on RIDS** (RIDS has no second AP sharing an
SSID, so the "difference against the same-SSID sibling" is undefined for every
window). `SimpleImputer(add_indicator=True)` cannot compute a median for an
all-NaN column, so it skips the column's imputation, warns, and leaves it NaN.
Input dimensionality is **stable** because both the raw-input imputer and the
normalizer are `fit` on the train fold and `transform` is applied with the same
fitted object: the column count out of `transform` is fixed by `fit`, not by
how many values happen to be observed, so there is no fit-vs-reload shape
mismatch.

## 1. Which feature is at the logged index, and which is all-NaN on RIDS

`pipeline.features.FEATURES` now has **17** features (P0.1 restored `share_data`
and `beacon_iat_mean`). Current order:

```
0  n_frames            6  seq_anom           12 rssi_mean
1  share_data          7  retry_rate         13 rssi_std
2  beacon_rate         8  duration_mean      14 n_bssid_same_ssid
3  deauth_count        9  duration_std       15 rssi_sibling_diff
4  beacon_iat_mean     10 length_mean        16 ssid_newcomer
5  beacon_iat_std      11 length_std
```

- **Current index 13 = `rssi_std`** (NOT all-NaN; 0.542% missing on RIDS).
- The log in `output/test_log.txt` (from `tests/test_artifacts.py`) says
  `Skipping features without any observed values: [13]`. That index refers to
  the **column position in the array passed to `SimpleImputer`**, which is the
  `FEATURES` order at the time the log was written. That log is **stale**: it
  predates P0.1. Before P0.1, `share_data` (now at 1) and `beacon_iat_mean`
  (now at 4) were dropped, so `FEATURES` had 15 entries and
  **index 13 was `rssi_sibling_diff`** — the all-NaN column.
- Re-running `_inputs` on the current 17-feature list reproduces the warning but
  now reports **`[15]`** = `rssi_sibling_diff`. So the live index is 15; the
  logged `[13]` is the pre-P0.1 index for the same underlying feature.

### The only all-NaN feature on RIDS

Per-feature missing rate over `output/windows_rids.parquet`
(38,765 windows) and `output/windows_awid3.parquet` (1,451 windows), for all 17
headline features:

| feature            | RIDS missing | AWID3 missing |
|--------------------|-------------:|--------------:|
| n_frames           |      0.000%  |       0.000%  |
| share_data         |      0.000%  |       0.000%  |
| beacon_rate        |      0.000%  |       0.000%  |
| deauth_count       |      0.000%  |       0.000%  |
| beacon_iat_mean    |      0.601%  |       4.480%  |
| beacon_iat_std     |      0.619%  |       4.480%  |
| seq_anom           |      0.542%  |       0.689%  |
| retry_rate         |      0.000%  |       0.000%  |
| duration_mean      |      0.000%  |       0.000%  |
| duration_std       |      0.000%  |       0.000%  |
| length_mean        |      0.000%  |       0.000%  |
| length_std         |      0.000%  |       0.000%  |
| rssi_mean          |      0.542%  |       0.689%  |
| **rssi_std** (13)  |      0.542%  |       0.689%  |
| n_bssid_same_ssid  |      0.601%  |       4.480%  |
| **rssi_sibling_diff** (15) | **100.000%** |  33.977%  |
| ssid_newcomer      |      0.000%  |       0.000%  |

- **`rssi_sibling_diff` is the only 100%-NaN feature on RIDS** (expected: no
  second AP per SSID, so there is never a sibling to difference against).
- **`ssid_newcomer` is NOT all-NaN on RIDS**: it is 0.000% missing. By design
  `_newcomer` returns `0.0` (not NaN) when a window has no second same-SSID AP,
  so it is defined everywhere and does not trigger the imputer warning. The
  original P1.3 expectation that `ssid_newcomer` would also be undefined on RIDS
  is therefore not borne out: only `rssi_sibling_diff` is all-NaN.
- On AWID3 nothing is all-NaN (`rssi_sibling_diff` is 33.98% missing, since some
  AWID3 windows do have a same-SSID sibling), so the warning does not fire there.

## 2. How an all-NaN feature is handled; dimension stability

Normalizer (`code/pipeline/preprocess.py`, `make_normalizer`):
`Pipeline([SimpleImputer(strategy="median", add_indicator=True), RobustScaler()])`.
Raw-tree path (`train.py::_inputs`, `RAW_INPUT` = {XGB, Rule}): XGB uses NaN
natively by default (no imputer); only when `impute=True` is a
`SimpleImputer(strategy="median", add_indicator=True)` fit for the imbalance
ablation. LSTM/HLSTM use `SeqScaler` on View B; HLSTM additionally normalizes
View A with the same `make_normalizer`.

Handling of the all-NaN column:
- At `fit`, `SimpleImputer` sees `rssi_sibling_diff` has no observed value, so it
  cannot derive a median. It **skips imputing that column** (emits the warning)
  and leaves its values NaN; it does **not drop** the column from the output.
  The transformed matrix keeps all 17 base columns.
- `add_indicator=True` adds an `is_missing` column **only for features that had
  missing values in the TRAIN fold**. On RIDS that fit-time missing set is
  features `[4, 5, 6, 12, 13, 14, 15]` → 7 indicator columns, giving a
  transformed width of `17 + 7 = 24` for the full RIDS table (23 when a feature
  with no train-fold NaN happens to have none in a particular fold). `RobustScaler`
  then scales all columns; the all-NaN column stays NaN after scaling. For the
  non-tree models the NaN is finally zeroed only in HLSTM via
  `np.nan_to_num(nz.transform(...))`; for the median-imputed XGB arm the all-NaN
  column remains NaN and XGB routes it through its learned default branch.

Dimension stability between fit and predict / reload:
- The output column count of `SimpleImputer.transform` is **fixed by `fit`**
  (`n_features_in_` plus the indicator columns chosen at fit), independent of how
  many values are observed at transform time. The same fitted pipeline object is
  reused for train/val/test within a fold, so val/test transforms produce the
  exact same width as train. No shape mismatch can arise from the all-NaN column.
- `FEATURES` is a module-level constant and `W[features].to_numpy(float)` always
  selects the same columns in the same order, so the model input dimension is
  stable across fit, score, and any reload that re-selects `FEATURES`.
- Caveat (not a current bug): because the indicator columns depend on which
  features have train-fold NaNs, the *indicator* count is data-dependent. A
  persisted model must be reloaded together with its fitted preprocessor
  (`prep`), which the pipeline already returns from `run_fold`. Reloading with a
  freshly-fit preprocessor on different data could change the indicator count;
  pairing model+preprocessor avoids that. This is a property of `add_indicator`,
  not of the all-NaN column.

## Conclusion

The warning is cosmetic. It flags the single genuinely-undefined feature
(`rssi_sibling_diff`, all-NaN on RIDS by construction), which the imputer
correctly leaves untouched. No feature is dropped, feature order is fixed, and
the transformed input dimension is stable for fit/predict/reload as long as the
fitted preprocessor is reloaded with the model. The logged index `[13]` is a
stale pre-P0.1 position for that feature; on the current 17-feature list the
same feature sits at index `[15]`.
