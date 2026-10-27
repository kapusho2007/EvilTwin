"""
Part 3: hybrid cascade. Dataset-agnostic: receives validation and test scores of the fold models only.

  Stage 1  one-class gate g in {AE, OCSVM}: pass iff s_g >= t_g. t_g is the largest threshold that keeps
           validation recall >= GATE_RECALL; g is the gate that passes fewer validation windows at that
           recall (cheapest filter). Windows below t_g leave as benign (they are never seen again).
  Stage 2  XGBoost and LSTM at their validation max-F1 thresholds (Table 1.3).
  Stage 3  escalation iff the two stage-2 models disagree (MARGIN = 0). An uncertainty band
           |logit p - logit t| < MARGIN is supported but off: XGBoost scores are not calibrated (benign
           windows score ~0.47 under scale_pos_weight, next to the threshold), so a band of 0.25 or 1
           escalated 422-866 windows, almost all of them where both models already agree.
           Agreeing windows take the stage-2 decision; escalated ones go to an arbiter.

Arbiters compared on the same escalated windows: always-benign, stage-2 alone (XGBoost), the fold's
one-feature rule, and the LLM (pipeline.llm).
"""
from __future__ import annotations
import numpy as np

GATE_RECALL = 0.99
MARGIN = 0.0                       # logit units; 0 = escalate on disagreement only


def recall_threshold(y, s, recall=GATE_RECALL) -> float:
    """Largest t with mean(s[y = 1] >= t) >= recall (validation only)."""
    pos = np.sort(np.asarray(s)[np.asarray(y) == 1])
    return float(pos[int(np.floor((1 - recall) * len(pos)))])


def logit(p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def choose_gate(yva, val_scores: dict, recall=GATE_RECALL):
    """Among the one-class models, the gate that passes the fewest validation windows at the recall target."""
    best = None
    for g, s in val_scores.items():
        t = recall_threshold(yva, s, recall)
        passed = float((s >= t).mean())
        if best is None or passed < best[2]:
            best = (g, t, passed)
    return best


def route(gate_pass, p_xgb, t_xgb, a_lstm, margin=MARGIN):
    """Returns (stage-2 decision, escalation mask) for every window; escalation only behind the gate."""
    a_xgb = p_xgb >= t_xgb
    near = np.abs(logit(p_xgb) - logit(t_xgb)) < margin
    esc = gate_pass & ((a_xgb != a_lstm) | near)
    return (gate_pass & a_xgb).astype(int), esc


def decide(gate_pass, stage2, esc, arbiter):
    """Final decision: benign exit below the gate, stage 2 where confident, arbiter where escalated."""
    out = np.where(gate_pass, stage2, 0)
    return np.where(esc, arbiter, out).astype(int)
