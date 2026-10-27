# P1.4 verification note — local-LLM cascade arbitration

Read-only audit of the Part-4 local LLM arbiter (`code/pipeline/llm.py`, `code/run_llm.py`) and its
outputs in `output/`. No code/report edits, no Ollama calls, no retraining.

## 1. Parsing robustness — ROBUST

The final verdict is read from a **structured JSON `verdict` field**, extracted from the *last* valid
JSON object in the reply, and accepted **only** if it is exactly `"attack"` or `"benign"`. It does
*not* classify by substring presence of "EVIL_TWIN"/"RogueAP"/"attack" anywhere in the prose. Parser
(`llm.py`, `parse()`):

```python
def parse(text: str):
    """(verdict 0/1 or None, confidence, status). Only the last JSON object in the reply counts."""
    dec, objs, i = json.JSONDecoder(), [], text.find("{")
    while i >= 0:
        try:
            o, end = dec.raw_decode(text, i)
            objs.append(o)
            i = text.find("{", end)
        except json.JSONDecodeError:
            i = text.find("{", i + 1)
    objs = [o for o in objs if isinstance(o, dict)]
    if not objs:
        return None, None, "no_json"
    o = objs[-1]
    v = str(o.get("verdict", "")).strip().lower()
    if v not in ("attack", "benign"):
        return None, None, "both_or_none" if ("attack" in v and "benign" in v) or not v else "bad_verdict"
    c = o.get("confidence")
    return int(v == "attack"), (float(c) if isinstance(c, (int, float)) else None), "ok"
```

Why robust:
- Keys on `o.get("verdict")` of the final structured object, not on scanning the whole reply.
- A reply whose *reasoning* names both labels is handled correctly: the request runs in Ollama JSON
  mode (`format="json"`), so the reply is a single JSON object; both labels appearing in the
  `reasoning`/`evidence` lists is irrelevant because only the `verdict` key decides.
- Ambiguity only arises if `verdict` itself contains both words (e.g. `"attack or benign"`) or is
  empty, which is explicitly rejected as `both_or_none`; an unexpected string is `bad_verdict`.
  Either way the arbiter abstains and the cascade keeps its Stage-2 decision — it never guesses.
- The design intent and an edge case are self-tested in `run_llm.py` `check()`:
  `'x {"a": 1} {"verdict": "benign"}' -> 0` (takes the last object) and
  `'{"verdict": "attack or benign"}' -> None`.

Empirical confirmation that it keys on the field, not substrings — two near-identical replies in
`llm_raw_awid3_full.jsonl`:
- window 10: reasoning says "a strong indicator of an **attack**" and discusses both; `verdict:"attack"`
  -> parsed **1**.
- window 11: essentially the same reasoning prose; `verdict:"benign"` -> parsed **0**.

A substring classifier would have tagged both as attack. The parser tracked the structured field.

## 2. Negative result (preserved as a negative result)

Escalated set (from `cascade_windows_*.csv`, `llm_summary.csv`, `cascade_llm.csv`):

| metric | AWID3 | RIDS |
|---|---|---|
| escalated cases (n) | 106 | 10 |
| mix: malicious / benign | 58 / 48 | 2 / 8 |
| LLM answered | 106 | 10 |
| parse failures | 0 | 0 |
| abstentions | 0 | 0 |
| errors | 0 | 0 |
| latency p50 / p95 (s) | 22.43 / 27.72 | 24.16 / 30.81 |

Head-to-head on the **identical** escalated set (`cascade_llm.csv`, accuracy on escalated windows):

| arbiter | AWID3 acc (TP,FP,FN,TN) | RIDS acc (TP,FP,FN,TN) |
|---|---|---|
| always-benign | 0.453 (0,0,58,48) | 0.800 (0,0,2,8) |
| Stage-2 alone (trust XGBoost) | 0.991 (58,1,0,47) | 0.200 (2,8,0,0) |
| rule arbitrator (one-feature rule) | **1.000** (58,0,0,48) | **1.000** (2,0,0,8) |
| LLM (abstain -> XGBoost) | 0.679 (56,32,2,16) | 0.200 (2,8,0,0) |

McNemar p (vs LLM): AWID3 one-feature-rule 1.16e-10, trust-XGBoost 2.33e-10 — both significantly
beat the LLM. On RIDS the rule beats the LLM at p = 0.0078.

End-to-end cascade F1 (`cascade_llm.csv` dF1_vs_XGB; same figures in `cascade_llm`/eval tables):
- AWID3: LLM arbiter F1 = 0.9034, **dF1 vs XGBoost = -0.0753** (CI -0.1124, -0.0463): LLM overrides
  *worsen* end-to-end F1, significantly.
- RIDS: LLM arbiter F1 = 0.9842, **dF1 vs XGBoost = -0.0034** (CI -0.0094, 0.0): no improvement
  (upper CI touches 0).

LLM overrides on AWID3: 33 of 106 (`llm_overrides=33`, `llm_fixed=0`, `llm_broken=33`) — every single
override was wrong; zero helped. On RIDS: 0 overrides. The AWID3 LLM added 32 false positives (the
rule added 0). Reasoning-quality audit (same CSV): 15/144 reasoning direction-checks wrong,
14 replies with a wrong-direction claim, 0 hallucinated values.

Conclusion (negative, not recast): the **rule arbitrator is strictly at least as good as the LLM on
both datasets** (perfect 1.000 accuracy on the escalated set vs 0.679 / 0.200), and the LLM's
overrides **do not improve, and in fact worsen, end-to-end F1** (AWID3 -0.075 significant; RIDS -0.003
not significant). The local LLM arbiter is not worth its ~22-25 s/window cost.

## 3. Coverage of escalated cases

The evaluated set **is** the complete set of escalated cases from the final cascade, not a sample:
- `cascade_windows_awid3.csv`: 106 escalated of 1451 windows; `cascade_windows_rids.csv`: 10 escalated
  of 38765. Total escalated = **116**, matching the plan's "~116 historical escalated cases".
- `llm_decisions_awid3.csv` = 106 rows, `llm_decisions_rids.csv` = 10 rows;
  `llm_raw_awid3_full.jsonl` = 106 lines, `llm_raw_rids_full.jsonl` = 10 lines; `scope=all` in
  `llm_summary.csv` reports n=106 / n=10 with answered=106 / 10. Every escalated window was asked and
  answered (the ablation files are a fixed 20/10 subsample of these, not a separate set).

Verdict: P1.4 is correctly implemented and correctly reported as a negative result.
