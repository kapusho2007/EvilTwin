"""Part 4 driver: the local LLM arbiter on the windows the cascade escalated (run_cascade.py first).
Usage: python run_llm.py prepare [awid3 rids]     # pandas: escalated windows -> llm_cases_<ds>.jsonl
       python run_llm.py check                    # next to Ollama: model metadata + parser tests
       python run_llm.py run [awid3 rids]         # next to Ollama: full prompt on every escalated window
       python run_llm.py ablate [awid3 rids]      # next to Ollama: prompt ablations on the fixed subsample
       python run_llm.py summarise [awid3 rids]   # pandas: llm_decisions_<ds>.csv, llm_summary.csv

check / run / ablate use only the standard library, so they run on the analyst's machine beside
`ollama serve`. Every prompt and reply is kept in llm_raw_<ds>_<variant>.jsonl; answered windows are
skipped, so a run can resume."""
import json
import subprocess
import sys

import config as C
from pipeline import llm as LLM

VARIANTS = ["full", "no_votes", "no_baselines", "no_indicators"]
N_ABLATION = 20                       # windows per dataset in the ablation subsample (fixed seed)


def prepare(ds):
    import numpy as np
    import pandas as pd
    from pipeline import features as F
    D = pd.read_csv(C.OUT_DIR / f"cascade_windows_{ds}.csv")
    W = pd.read_parquet(C.OUT_DIR / f"windows_{ds}.parquet")
    B = pd.read_csv(C.OUT_DIR / f"cascade_baselines_{ds}.csv")
    key = {(a, round(t, 3)): i for i, (a, t) in enumerate(zip(W["ap"], W["tau"]))}
    E = D[D.escalated == 1]
    sub = set(np.random.default_rng(42).choice(E.window.values, min(N_ABLATION, len(E)), replace=False).tolist())
    feats = lambda i: {f: (None if pd.isna(v) else float(v)) for f, v in W.loc[i, F.FEATURES].items()}
    with (C.OUT_DIR / f"llm_cases_{ds}.jsonl").open("w") as fh:
        for _, r in E.iterrows():
            i = int(r.window)
            prev = {}
            for lag in (5, 10):
                j = key.get((W.at[i, "ap"], round(W.at[i, "tau"] - lag, 3)))
                prev[f"t-{lag}s"] = feats(j) if j is not None else None
            g = r.gate_model
            fh.write(json.dumps(dict(
                window=i, y=int(r.y), fold=int(r.fold), ablation=i in sub, feat=feats(i),
                base={b.feature: [b.median, b.q25, b.q75] for b in B[B.fold == r.fold].itertuples()},
                prev=prev,
                votes={"XGBoost": [r.XGB_score, r.XGB_thr, int(r.XGB_alert)],
                       "LSTM": [r.LSTM_score, r.LSTM_thr, int(r.LSTM_alert)],
                       f"{g} anomaly gate": [r[f"{g}_score"], None, int(r.gate)]})) + "\n")
    print(f"[{ds}] {len(E)} escalated windows ({int(E.y.sum())} attack), {len(sub)} in the ablation subsample")


def _load(path):
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()] if path.exists() else []


def run(ds, variant="full"):
    cases = _load(C.OUT_DIR / f"llm_cases_{ds}.jsonl")
    if variant != "full":
        cases = [c for c in cases if c["ablation"]]
    path = C.OUT_DIR / f"llm_raw_{ds}_{variant}.jsonl"
    done = {r["window"] for r in _load(path)}
    todo = [c for c in cases if c["window"] not in done]
    print(f"[{ds}] {variant}: {len(todo)} windows to ask ({len(done)} done)", flush=True)
    ab = None if variant == "full" else variant
    with path.open("a") as fh:
        for n, c in enumerate(todo):
            msgs = LLM.messages(LLM.build_context(c["feat"], c["base"], c["prev"], c["votes"], ab), ab)
            text, sec, n_out, n_in, err = LLM.ask(msgs)
            v, conf, status = LLM.parse(text) if err is None else (None, None, err)
            fh.write(json.dumps(dict(window=c["window"], y=c["y"], variant=variant, prompt_version=LLM.PROMPT_VERSION,
                                     messages=msgs, reply=text, verdict=v, confidence=conf, status=status,
                                     latency_s=sec, tokens_out=n_out, tokens_in=n_in)) + "\n")
            fh.flush()
            if n % 10 == 0:
                print(f"  {n + 1}/{len(todo)} {status} {sec:.1f}s", flush=True)


def check():
    tests = {'{"verdict": "attack", "confidence": 0.9}': 1, 'x {"a": 1} {"verdict": "benign"}': 0,
             '{"verdict": "attack or benign"}': None, "no json": None, '{"verdict": "unknown"}': None}
    for t, want in tests.items():
        assert LLM.parse(t)[0] == want, t
    show = subprocess.run(["ollama", "show", LLM.MODEL], capture_output=True, text=True).stdout
    tags = json.loads(subprocess.run(["curl", "-s", "localhost:11434/api/tags"], capture_output=True, text=True).stdout)
    meta = dict(model=LLM.MODEL, options=LLM.OPTIONS, prompt_version=LLM.PROMPT_VERSION, ollama_show=show,
                tags=[m for m in tags.get("models", []) if m["name"] == LLM.MODEL])
    (C.OUT_DIR / "llm_meta.json").write_text(json.dumps(meta, indent=1))
    print(show, "\nparser tests passed; wrote llm_meta.json")


def summarise(datasets):
    import pandas as pd
    rows = []
    for ds in datasets:
        for v in VARIANTS:
            R = pd.DataFrame(_load(C.OUT_DIR / f"llm_raw_{ds}_{v}.jsonl"))
            if R.empty:
                continue
            if v == "full":
                R[["window", "verdict", "confidence", "status", "latency_s"]].rename(
                    columns={"verdict": "llm_alert"}).to_csv(C.OUT_DIR / f"llm_decisions_{ds}.csv", index=False)
            sub = {c["window"] for c in _load(C.OUT_DIR / f"llm_cases_{ds}.jsonl") if c["ablation"]}
            for scope, S in (("all", R), ("ablation", R[R.window.isin(sub)])):
                if v != "full" and scope == "all":
                    continue
                ok = S.verdict.notna()
                y, p = S.y[ok].astype(int), S.verdict[ok].astype(int)
                rows.append(dict(dataset=ds, variant=v, scope=scope, n=len(S), attack=int(S.y.sum()), answered=int(ok.sum()),
                                 TP=int(((p == 1) & (y == 1)).sum()), FP=int(((p == 1) & (y == 0)).sum()),
                                 FN=int(((p == 0) & (y == 1)).sum()), TN=int(((p == 0) & (y == 0)).sum()),
                                 parse_fail=int(S.status.isin(["no_json", "bad_json", "both_or_none", "bad_verdict"]).sum()),
                                 errors=int((~S.status.isin(["ok", "no_json", "bad_json", "both_or_none", "bad_verdict"])).sum()),
                                 p50_s=S.latency_s.median(), p95_s=S.latency_s.quantile(0.95),
                                 tokens_out_p50=S.tokens_out.median()))
    T = pd.DataFrame(rows)
    T.to_csv(C.OUT_DIR / "llm_summary.csv", index=False)
    print(T.round(3).to_string())


if __name__ == "__main__":
    cmd, ds = sys.argv[1], sys.argv[2:] or ["awid3", "rids"]
    if cmd == "check":
        check()
    elif cmd == "summarise":
        summarise(ds)
    else:
        for d in ds:
            if cmd == "prepare":
                prepare(d)
            elif cmd == "run":
                run(d)
            elif cmd == "ablate":
                for v in VARIANTS:
                    run(d, v)
