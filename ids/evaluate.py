"""Detection-quality metrics and measurable proxies for analyst usability.

IMPORTANT: no human analysts were involved. The "usability" numbers below are
computable proxies (are explanations faithful? compact? stable? do they help
separate false alerts from true ones?). They are evidence, not a user study.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, precision_recall_fscore_support,
                             roc_auc_score, roc_curve)

from .data import EXPERT_SIGNATURES


# --------------------------------------------------------------------------
# detection quality
# --------------------------------------------------------------------------
def tpr_at_fpr(y_bin, score, fpr_target):
    fpr, tpr, _ = roc_curve(y_bin, score)
    return float(np.interp(fpr_target, fpr, tpr))


def detection_metrics(y_true, proba, classes):
    y_true = np.asarray(y_true)
    classes = list(classes)
    pred = np.array(classes)[proba.argmax(1)]
    p, r, f, s = precision_recall_fscore_support(y_true, pred, labels=classes, zero_division=0)
    per_class = {c: dict(precision=float(p[i]), recall=float(r[i]), f1=float(f[i]), support=int(s[i]))
                 for i, c in enumerate(classes)}
    attack = (y_true != "BENIGN").astype(int)
    # P(any attack), summed directly: 1 - P(benign) saturates to exactly 1.0 in
    # float64 for confident rows and creates ties that corrupt low-FPR metrics.
    score = proba[:, [i for i, c in enumerate(classes) if c != "BENIGN"]].sum(1)
    is_alert = pred != "BENIGN"
    return {
        "accuracy": float((pred == y_true).mean()),
        "macro_f1": float(f.mean()),
        "attack_roc_auc": float(roc_auc_score(attack, score)),
        "attack_pr_auc": float(average_precision_score(attack, score)),
        "attack_tpr_at_1pct_fpr": tpr_at_fpr(attack, score, 0.01),
        "attack_tpr_at_0.1pct_fpr": tpr_at_fpr(attack, score, 0.001),
        "alert_precision": float((attack[is_alert]).mean()) if is_alert.any() else float("nan"),
        "alert_recall": float(is_alert[attack == 1].mean()),
        "benign_false_alert_rate": float(is_alert[attack == 0].mean()),
        "per_class": per_class,
    }


# --------------------------------------------------------------------------
# usability proxies (all operate on a set of alerts with SHAP vectors)
# --------------------------------------------------------------------------
def _top_idx(phi, k, by="pos"):
    key = -phi if by == "pos" else -np.abs(phi)
    return np.argsort(key, axis=1)[:, :k]


def faithfulness(model, X: pd.DataFrame, phi, kidx, k=3, n_random=5, seed=0):
    """Deletion test. Replace the k features SHAP says matter most for the verdict
    with their typical-benign values and measure the fall in the verdict's
    probability. Compared with replacing k random features and k globally-top
    features (same for every alert, i.e. what a global importance chart gives).
    Caveat: edited flows can be off-manifold."""
    rng = np.random.default_rng(seed)
    Xv = X.values.astype(float)
    bg = model.background_.values.astype(float)
    n, F = Xv.shape
    p0 = model.predict_proba(X)[np.arange(n), kidx]

    def drop_with(idx):
        Xe = Xv.copy()
        rows = np.arange(n)[:, None]
        Xe[rows, idx] = bg[idx]
        pr = model.predict_proba(pd.DataFrame(Xe, columns=X.columns))
        return p0 - pr[np.arange(n), kidx], pr.argmax(1) != kidx

    d_shap, flip_shap = drop_with(_top_idx(phi, k))
    d_rand, flip_rand = [], []
    for _ in range(n_random):
        idx = np.stack([rng.choice(F, k, replace=False) for _ in range(n)])
        d, fl = drop_with(idx)
        d_rand.append(d.mean()); flip_rand.append(fl.mean())
    # global importance per predicted class
    gidx = np.zeros((n, k), int)
    for c in np.unique(kidx):
        m = kidx == c
        gidx[m] = np.argsort(-np.abs(phi[m]).mean(0))[:k]
    d_glob, flip_glob = drop_with(gidx)
    return {
        "k": k,
        "mean_prob_drop_shap_top_k": float(d_shap.mean()),
        "mean_prob_drop_class_global_top_k": float(d_glob.mean()),
        "mean_prob_drop_random_k": float(np.mean(d_rand)),
        "verdict_flip_rate_shap_top_k": float(flip_shap.mean()),
        "verdict_flip_rate_class_global_top_k": float(flip_glob.mean()),
        "verdict_flip_rate_random_k": float(np.mean(flip_rand)),
        "n_alerts": int(n),
    }


def compactness(phi, kidx=None):
    """Features needed to cover 80% of an alert's total |SHAP| mass (lower is easier
    to read), and the share of mass in the top 3."""
    a = np.sort(np.abs(phi), axis=1)[:, ::-1]
    cum = np.cumsum(a, axis=1) / np.maximum(a.sum(1, keepdims=True), 1e-12)
    n80 = (cum < 0.8).sum(1) + 1
    return {"median_features_for_80pct": float(np.median(n80)),
            "mean_features_for_80pct": float(n80.mean()),
            "mean_top3_mass_share": float(cum[:, 2].mean()),
            "total_features": int(phi.shape[1])}


def stability(model, X: pd.DataFrame, phi, kidx, k=3, noise=0.05, seed=0):
    """Do near-identical flows get the same explanation? Perturb every continuous
    feature by ~noise (log-normal) and compare top-k sets (Jaccard)."""
    rng = np.random.default_rng(seed)
    Xn = X.copy()
    cols = [c for c in X.columns if c != "dst_port"]
    Xn[cols] = Xn[cols].values * rng.lognormal(0, noise, (len(X), len(cols)))
    phi2, _, _ = model.explain(Xn, kidx)
    a, b = _top_idx(phi, k, "abs"), _top_idx(phi2, k, "abs")
    jac = np.array([len(set(x) & set(y)) / len(set(x) | set(y)) for x, y in zip(a, b)])
    same_pred = (model.predict_proba(Xn).argmax(1) == kidx)
    return {"noise": noise, "mean_top_k_jaccard": float(jac.mean()),
            "mean_top_k_jaccard_when_verdict_unchanged": float(jac[same_pred].mean()) if same_pred.any() else float("nan"),
            "verdict_unchanged_rate": float(same_pred.mean())}


def signature_agreement(phi, X: pd.DataFrame, verdicts, k=3):
    """S = share of the top-k verdict-supporting features that a SOC analyst would
    expect for that attack type (EXPERT_SIGNATURES). Range 0..1."""
    feats = list(X.columns)
    S = np.zeros(len(phi))
    for i in range(len(phi)):
        order = np.argsort(-phi[i])[:k]
        order = [j for j in order if phi[i, j] > 0]
        if order:
            sig = EXPERT_SIGNATURES.get(verdicts[i], set())
            S[i] = np.mean([feats[j] in sig for j in order])
    return S


def confidence(proba):
    """Saturation-free confidence: -log(total probability on the non-top classes)."""
    rest = proba.copy()
    rest[np.arange(len(rest)), proba.argmax(1)] = 0.0
    return -np.log(rest.sum(1) + 1e-300)


def evidence_balance(phi):
    """Share of an alert's |SHAP| mass that supports the verdict (1 = no
    counter-evidence). Exploratory triage signal."""
    pos = np.clip(phi, 0, None).sum(1)
    return pos / np.maximum(np.abs(phi).sum(1), 1e-12)


def _auc_ci(y, s, n_boot=300, seed=0):
    y, s = np.asarray(y), np.asarray(s)
    rng = np.random.default_rng(seed)
    est = roc_auc_score(y, s)
    boots = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() != y[i].max():
            boots.append(roc_auc_score(y[i], s[i]))
    return float(est), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def _rank(a):
    return pd.Series(a).rank(pct=True).values


def triage_simulation(conf, S, is_true_attack, extra=None):
    """Which signal best separates real attacks from false alarms among raised alerts?
    - confidence: all a black-box detector offers an analyst
    - S: agreement between the SHAP explanation and analyst expectations
    - combined: mean of the two percentile ranks
    Also an explicit gating policy: deprioritise alerts whose explanation shares
    nothing with the expected signature (S == 0)."""
    y = np.asarray(is_true_attack).astype(int)
    out = {"n_alerts": int(len(y)), "n_false_alerts": int((y == 0).sum())}
    if y.min() == y.max():
        return out
    out["auc_confidence"] = _auc_ci(y, conf)
    out["auc_explanation_agreement"] = _auc_ci(y, S)
    out["auc_combined"] = _auc_ci(y, 0.5 * _rank(conf) + 0.5 * _rank(S))
    for name, v in (extra or {}).items():
        out[f"auc_{name}"] = _auc_ci(y, v)
        out[f"auc_confidence_plus_{name}"] = _auc_ci(y, 0.5 * _rank(conf) + 0.5 * _rank(v))
    gate = []
    for thr in (1 / 3, 2 / 3):
        keep = S >= thr - 1e-9
        gate.append({"gate": f"S >= {thr:.2f}",
                     "alerts_kept": float(keep.mean()),
                     "false_alerts_removed": float((~keep & (y == 0)).sum() / max((y == 0).sum(), 1)),
                     "true_alerts_lost": float((~keep & (y == 1)).sum() / max((y == 1).sum(), 1)),
                     "precision_before": float(y.mean()),
                     "precision_after": float(y[keep].mean()) if keep.any() else float("nan")})
    out["gating"] = gate
    return out


def global_importance(phi, features):
    """Mean |SHAP| per feature, normalised to shares."""
    m = np.abs(phi).mean(0)
    return pd.Series(m / m.sum(), index=features).sort_values(ascending=False)
