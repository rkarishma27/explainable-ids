"""Turning SHAP vectors into analyst-facing alerts."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import FEATURE_DESC


def _fmt(v: float) -> str:
    if abs(v) >= 1e6:
        return f"{v / 1e6:.1f}M"
    if abs(v) >= 1e4:
        return f"{v / 1e3:.1f}k"
    if float(v).is_integer():
        return f"{int(v)}"
    return f"{v:.2f}"


def top_reasons(phi_row, x_row, features, k=3, direction="for"):
    """Top-k (feature, value, contribution) for (`for`) or against the verdict."""
    order = np.argsort(-phi_row if direction == "for" else phi_row)
    out = []
    for i in order[:k]:
        if (direction == "for" and phi_row[i] <= 0) or (direction == "against" and phi_row[i] >= 0):
            break
        out.append((features[i], float(x_row[i]), float(phi_row[i])))
    return out


def reason_text(reasons) -> str:
    return "; ".join(f"{FEATURE_DESC.get(f, f)} = {_fmt(v)} ({c:+.2f})" for f, v, c in reasons)


def build_alert_table(model, X: pd.DataFrame, meta: pd.DataFrame, y_true=None, k=3):
    """One row per non-benign verdict: label, confidence, top-k reasons for it."""
    proba = model.predict_proba(X)
    cls = np.array(model.classes_)
    pred_idx = proba.argmax(1)
    alert = cls[pred_idx] != "BENIGN"
    Xa = X[alert]
    phi, base, kidx = model.explain(Xa, pred_idx[alert])
    feats = list(X.columns)
    rows = []
    for j, (ix, row) in enumerate(Xa.iterrows()):
        r_for = top_reasons(phi[j], row.values, feats, k, "for")
        r_against = top_reasons(phi[j], row.values, feats, 2, "against")
        rows.append({
            "flow_id": meta.loc[ix, "flow_id"], "src_ip": meta.loc[ix, "src_ip"], "dst_ip": meta.loc[ix, "dst_ip"],
            "verdict": cls[pred_idx[alert][j]], "confidence": proba[alert][j].max(),
            "why": reason_text(r_for), "counter_evidence": reason_text(r_against),
            "true_label": None if y_true is None else y_true.loc[ix],
        })
    return pd.DataFrame(rows), phi, kidx


def alert_card_markdown(a: pd.Series) -> str:
    s = f"**Flow {a.flow_id}**  {a.src_ip} -> {a.dst_ip}  |  **{a.verdict}**  (confidence {a.confidence:.0%})\n\n"
    s += f"- Why: {a.why}\n"
    if a.counter_evidence:
        s += f"- Argues against: {a.counter_evidence}\n"
    if a.true_label is not None:
        s += f"- Ground truth (synthetic/test label): {a.true_label}\n"
    return s
