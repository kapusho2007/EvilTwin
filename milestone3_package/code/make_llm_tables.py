"""Part 4.2-4.3 tables from the LLM replies (run_llm.py run / ablate) and the cascade windows (run_cascade.py).
Usage: python make_llm_tables.py [awid3 rids]
Writes llm_eval.csv (4.3: each arbiter on the same escalated windows, Wilson 95% CIs, LLM overrides of
XGBoost fixed vs broken, statuses, latency) and llm_ablation.csv (4.2: each prompt variant vs the full prompt
on the same subsample, exact McNemar p on the discordant pairs)."""
import json
import re
import sys
from math import comb, sqrt

import pandas as pd

import config as C

VARIANTS = ["full", "no_votes", "no_baselines", "no_indicators"]


def wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return c - h, c + h


def mcnemar_exact(b, c):
    """Two-sided exact binomial test on the discordant pairs (b: only A right, c: only B right)."""
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


NUM = re.compile(r"-?\d+(?:\.\d+)?(?:e-?\d+)?")


def _numbers(o):
    if isinstance(o, dict):
        return [x for v in o.values() for x in _numbers(v)]
    if isinstance(o, list):
        return [x for v in o for x in _numbers(v)]
    return [float(o)] if isinstance(o, (int, float)) and not isinstance(o, bool) else []


def faithful(row):
    """Every evidence item that names a feature must cite, as its first number after the name, a value
    that occurs for that feature in the telemetry (window value, benign median / IQR, earlier windows),
    within rounding (|d| <= 0.01 or 1%). Returns (items checked, items not found) for one reply."""
    try:
        t = json.loads(row["messages"][1]["content"].split("<telemetry>")[1].split("</telemetry>")[0])
        ev = json.loads(row["reply"]).get("evidence", [])
    except (ValueError, IndexError, KeyError, AttributeError, TypeError):
        return 0, 0
    feats = t["window"]["features"]
    checked = bad = 0
    for item in ev if isinstance(ev, list) else []:
        item = str(item)
        names = [f for f in sorted(feats, key=len, reverse=True) if f in item]
        if not names:
            continue
        f = names[0]
        nums = NUM.findall(item.split(f, 1)[1])
        if not nums:
            continue
        x = float(nums[0])
        pool = _numbers(feats[f]) + [d.get(f) for d in t.get("same_AP_earlier", {}).values() if d and d.get(f) is not None]
        checked += 1
        bad += not any(abs(x - v) <= max(0.01, 0.01 * abs(v)) for v in pool)
    return checked, bad


UP = re.compile(r"\b(higher|larger|greater|above|more|increased?)\b", re.I)
DOWN = re.compile(r"\b(lower|smaller|below|less|fewer|decreased?)\b", re.I)


def directions(row):
    """Reasoning steps that name one feature (or its spaced form, e.g. 'beacon rate') and one direction
    word (higher / lower ...): is the window value really on that side of the benign median?
    Returns (steps checked, steps with the wrong direction) for one reply."""
    try:
        t = json.loads(row["messages"][1]["content"].split("<telemetry>")[1].split("</telemetry>")[0])
        steps = json.loads(row["reply"]).get("reasoning", [])
    except (ValueError, IndexError, KeyError, AttributeError, TypeError):
        return 0, 0
    feats = t["window"]["features"]
    checked = wrong = 0
    for st in steps if isinstance(steps, list) else []:
        st = str(st)
        names = [f for f in feats if f in st or f.replace("_", " ") in st.lower()]
        up, down = bool(UP.search(st)), bool(DOWN.search(st))
        x, m = (feats[names[0]].get("value"), feats[names[0]].get("benign_median")) if len(names) == 1 else (None, None)
        if len(names) != 1 or up == down or x is None or m is None or x == m:
            continue
        checked += 1
        wrong += (x > m) != up
    return checked, wrong


def raw(ds, v):
    p = C.OUT_DIR / f"llm_raw_{ds}_{v}.jsonl"
    rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []
    return pd.DataFrame(rows)


def evaluate(ds):
    D = pd.read_csv(C.OUT_DIR / f"cascade_windows_{ds}.csv")
    E = D[D.escalated == 1].set_index("window")
    R = raw(ds, "full").set_index("window")
    fb = R.apply(faithful, axis=1, result_type="expand") if len(R) else pd.DataFrame(columns=[0, 1])
    R["ev_checked"], R["ev_bad"] = fb[0], fb[1]
    dr = R.apply(directions, axis=1, result_type="expand") if len(R) else pd.DataFrame(columns=[0, 1])
    R["dir_checked"], R["dir_wrong"] = dr[0], dr[1]
    E = E.join(R[["verdict", "status", "latency_s", "tokens_out", "ev_checked", "ev_bad", "dir_checked", "dir_wrong"]],
               how="left")
    y = E.y.astype(int)
    rows = []
    arb = {"always benign": 0 * y, "trust XGBoost": E.XGB_alert, "trust LSTM": E.LSTM_alert,
           "one-feature rule": E.Rule_alert, "LLM (abstain -> XGBoost)": E.verdict.fillna(E.XGB_alert).astype(int)}
    for name, p in arb.items():
        p = p.astype(int)
        k = int((p == y).sum())
        lo, hi = wilson(k, len(y))
        tp, fp = int(((p == 1) & (y == 1)).sum()), int(((p == 1) & (y == 0)).sum())
        fn, tn = int(((p == 0) & (y == 1)).sum()), int(((p == 0) & (y == 0)).sum())
        rows.append(dict(dataset=ds, arbiter=name, n=len(y), attack=int(y.sum()), correct=k, acc=k / len(y),
                         acc_lo=lo, acc_hi=hi, TP=tp, FP=fp, FN=fn, TN=tn))
    T = pd.DataFrame(rows)
    ans = E.verdict.notna()
    over = ans & (E.verdict != E.XGB_alert)
    T["llm_answered"] = int(ans.sum())
    T["llm_status"] = json.dumps(E.status.value_counts().to_dict())
    T["llm_overrides"] = int(over.sum())
    T["llm_fixed"] = int((over & (E.verdict == y)).sum())
    T["llm_broken"] = int((over & (E.verdict != y)).sum())
    T["lat_p50"], T["lat_p95"], T["lat_total_s"] = E.latency_s.median(), E.latency_s.quantile(0.95), E.latency_s.sum()
    T["tokens_out_p50"] = E.tokens_out.median()
    hall = ans & (E.ev_bad > 0)
    T["ev_items_checked"], T["ev_items_bad"] = int(E.ev_checked.sum()), int(E.ev_bad.sum())
    T["replies_with_hallucinated_value"] = int(hall.sum())
    T["correct_but_hallucinated"] = int((hall & (E.verdict == y)).sum())
    wd = ans & (E.dir_wrong > 0)
    T["steps_dir_checked"], T["steps_dir_wrong"] = int(E.dir_checked.sum()), int(E.dir_wrong.sum())
    T["replies_with_wrong_direction"] = int(wd.sum())
    T["correct_but_wrong_direction"] = int((wd & (E.verdict == y)).sum())
    k_llm = int((arb["LLM (abstain -> XGBoost)"].astype(int) == y).sum())
    T["mcnemar_p_vs_llm"] = [mcnemar_exact(int(((p.astype(int) == y) & (arb["LLM (abstain -> XGBoost)"].astype(int) != y)).sum()),
                                           int(((p.astype(int) != y) & (arb["LLM (abstain -> XGBoost)"].astype(int) == y)).sum()))
                             for p in arb.values()]
    return T


def ablation(datasets):
    """Per dataset: each variant vs the full prompt on that dataset's fixed ablation subsample."""
    rows = []
    for d in datasets:
        V0 = [raw(d, v) for v in VARIANTS[1:]]
        sub = set(pd.concat(V0).window) if any(len(x) for x in V0) else set()
        F = raw(d, "full")
        for v in VARIANTS:
            V = raw(d, v)
            if V.empty or not sub:
                rows.append(dict(dataset=d, variant=v, n=0))                  # arm not available
                continue
            J = V[V.window.isin(sub)].merge(F[["window", "verdict"]], on="window", suffixes=("", "_full"))
            ok = J.verdict.notna()
            right = (J.verdict == J.y) & ok
            right_f = (J.verdict_full == J.y) & J.verdict_full.notna()
            k, n = int(right.sum()), len(J)
            lo, hi = wilson(k, n)
            rows.append(dict(dataset=d, variant=v, n=n, attack=int(J.y.sum()), correct=k, acc=k / n, acc_lo=lo,
                             acc_hi=hi, attack_verdicts=int((J.verdict == 1).sum()), unanswered=int((~ok).sum()),
                             mcnemar_p_vs_full=mcnemar_exact(int((right & ~right_f).sum()), int((~right & right_f).sum())),
                             lat_p50=J.latency_s.median()))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    dss = sys.argv[1:] or ["awid3", "rids"]
    T = pd.concat([evaluate(d) for d in dss])
    T.to_csv(C.OUT_DIR / "llm_eval.csv", index=False)
    print(T.round(3).to_string())
    A = ablation(dss)
    A.to_csv(C.OUT_DIR / "llm_ablation.csv", index=False)
    print(A.round(3).to_string())
