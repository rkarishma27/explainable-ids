"""End-to-end experiment: python run_all.py [--quick] [--seeds 5] [--n-flows 40000]

Produces outputs/: metrics.json, report.md, alerts.csv, alerts_sample.md,
model.joblib and figures/*.png
"""
from __future__ import annotations

import argparse
import json
import os
import time

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

from ids.data import (CLASSES, FEATURE_DESC, FEATURES, add_capture_artifact,
                      generate_flows, load_cicids2017)
from ids.evaluate import (compactness, confidence, detection_metrics, evidence_balance,
                          faithfulness, global_importance, signature_agreement,
                          stability, triage_simulation)
from ids.explain import alert_card_markdown, build_alert_table, reason_text, top_reasons
from ids.models import BlackBoxBaseline, ExplainableIDS, save_bundle

# validated categorical slots 1-3 + text tokens from the dataviz reference palette
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
SURFACE, INK, INK2, GRID, NEUTRAL = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1", "#9a9993"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
FIG = os.path.join(OUT, "figures")


def style_ax(ax, grid_axis="y"):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def new_fig(*a, **k):
    fig, ax = plt.subplots(*a, **k)
    fig.patch.set_facecolor(SURFACE)
    return fig, ax


# --------------------------------------------------------------------------
def load_data(args, seed):
    if args.cicids:
        X, y, meta = load_cicids2017(args.cicids, max_rows=args.n_flows, seed=seed)
        return X, y, meta
    return generate_flows(args.n_flows, seed)


def one_seed(args, seed, keep=False):
    t0 = time.time()
    X, y, meta = load_data(args, seed)
    tr, te = train_test_split(np.arange(len(X)), test_size=0.3, stratify=y, random_state=seed)
    Xtr, ytr, Xte, yte = X.iloc[tr], y.iloc[tr], X.iloc[te], y.iloc[te]
    mte = meta.iloc[te]
    gb = ExplainableIDS(random_state=seed, backend=args.backend).fit(Xtr, ytr)
    mlp = BlackBoxBaseline(random_state=seed).fit(Xtr, ytr)
    pg, pm = gb.predict_proba(Xte), mlp.predict_proba(Xte)
    det = {"explainable_gb": detection_metrics(yte.values, pg, gb.classes_),
           "blackbox_mlp": detection_metrics(yte.values, pm, mlp.classes_)}

    cls = np.array(gb.classes_)
    alerts = np.where(cls[pg.argmax(1)] != "BENIGN")[0]
    rng = np.random.default_rng(seed)
    samp = rng.choice(alerts, min(args.explain_cap, len(alerts)), replace=False)
    Xa, ka = Xte.iloc[samp], pg[samp].argmax(1)
    phi, base, _ = gb.explain(Xa, ka)
    raw = gb.gb_.decision_function(Xa.values)
    add_err = float(np.abs(base + phi.sum(1) - raw[np.arange(len(Xa)), ka]).max())
    if add_err > 1e-6:  # explanations must sum to the model score; never report numbers that don't
        raise RuntimeError(f"seed {seed}: SHAP additivity violated (max err {add_err:.3g})")
    faith = faithfulness(gb, Xa, phi, ka)
    comp = compactness(phi)
    ns = min(args.stab_cap, len(Xa))
    stab = stability(gb, Xa.iloc[:ns], phi[:ns], ka[:ns])
    S = signature_agreement(phi, Xa, cls[ka])
    conf = confidence(pg[samp])
    bal = evidence_balance(phi)
    valid = yte.values[samp] != "BENIGN"
    tri = triage_simulation(conf, S, valid, {"evidence_balance": bal})

    def conf_auc(p, classes):
        c = np.array(classes)
        al = np.where(c[p.argmax(1)] != "BENIGN")[0]
        v = yte.values[al] != "BENIGN"
        return float(roc_auc_score(v, confidence(p)[al])) if 0 < v.sum() < len(v) else float("nan")

    res = {"seed": seed, "detection": det, "faithfulness": faith, "compactness": comp,
           "stability": stab, "triage": tri, "additivity_max_abs_err": add_err,
           "confidence_auc_all_alerts": {"explainable_gb": conf_auc(pg, gb.classes_),
                                         "blackbox_mlp": conf_auc(pm, mlp.classes_)},
           "shap_backend": gb.shap_backend, "leaves_clipped": gb.n_leaves_clipped_, "seconds": round(time.time() - t0, 1),
           "_pool": {"conf": conf, "S": S, "bal": bal, "valid": valid}}
    if keep:
        res["_keep"] = dict(gb=gb, mlp=mlp, Xtr=Xtr, ytr=ytr, Xte=Xte, yte=yte, mte=mte, pg=pg, pm=pm,
                            Xa=Xa, phi=phi, ka=ka, tr=tr, te=te)
    return res


# --------------------------------------------------------------------------
def shortcut_audit(args, K):
    """A dataset artifact (attacks captured in different sessions) that a model can
    exploit. Does the model use it? Can the team find out?"""
    gb0, mlp0 = K["gb"], K["mlp"]
    Xtr, ytr, Xte, yte = K["Xtr"], K["ytr"], K["Xte"], K["yte"]
    art_tr = add_capture_artifact(ytr.values, 101, 0.97).values
    art_te_same = add_capture_artifact(yte.values, 102, 0.97).values
    art_te_shift = add_capture_artifact(yte.values, 103, 0.0).values
    Xtr_l = Xtr.assign(capture_session=art_tr)
    Xte_same = Xte.assign(capture_session=art_te_same)
    Xte_shift = Xte.assign(capture_session=art_te_shift)
    gbl = ExplainableIDS(random_state=0, backend=args.backend).fit(Xtr_l, ytr)
    mlpl = BlackBoxBaseline(random_state=0).fit(Xtr_l, ytr)

    def mf1(m, X):
        return float(f1_score(yte.values, m.predict(X), average="macro", labels=CLASSES, zero_division=0))

    scores = {
        "explainable_gb": {"clean_model_clean_test": mf1(gb0, Xte),
                           "leaky_model_same_distribution_test": mf1(gbl, Xte_same),
                           "leaky_model_shifted_test": mf1(gbl, Xte_shift)},
        "blackbox_mlp": {"clean_model_clean_test": mf1(mlp0, Xte),
                         "leaky_model_same_distribution_test": mf1(mlpl, Xte_same),
                         "leaky_model_shifted_test": mf1(mlpl, Xte_shift)},
    }
    # SHAP audit on the leaky GB: global importance over a sample of its alerts
    pg = gbl.predict_proba(Xte_same)
    cls = np.array(gbl.classes_)
    al = np.where(cls[pg.argmax(1)] != "BENIGN")[0]
    al = np.random.default_rng(0).choice(al, min(args.explain_cap, len(al)), replace=False)
    phi, _, _ = gbl.explain(Xte_same.iloc[al], pg[al].argmax(1))
    imp = global_importance(phi, list(Xte_same.columns))
    # post-hoc permutation importance for the MLP (the black-box's only recourse)
    rng = np.random.default_rng(0)
    sub = rng.choice(len(Xte_same), min(3000, len(Xte_same)), replace=False)
    Xs, ys = Xte_same.iloc[sub], yte.values[sub]
    base_f1 = f1_score(ys, mlpl.predict(Xs), average="macro", labels=CLASSES, zero_division=0)
    perm = {}
    for c in Xs.columns:
        drops = []
        for _ in range(2):
            Xp = Xs.copy(); Xp[c] = rng.permutation(Xp[c].values)
            drops.append(base_f1 - f1_score(ys, mlpl.predict(Xp), average="macro", labels=CLASSES, zero_division=0))
        perm[c] = float(np.mean(drops))
    perm = pd.Series(perm).sort_values(ascending=False)
    return {"macro_f1": scores,
            "shap_global_importance_leaky_gb_top5": {k: float(v) for k, v in imp.head(5).items()},
            "shap_rank_of_artifact": int(list(imp.index).index("capture_session") + 1),
            "shap_share_of_artifact": float(imp["capture_session"]),
            "mlp_permutation_importance_top5_macro_f1_drop": {k: float(v) for k, v in perm.head(5).items()},
            "mlp_permutation_rank_of_artifact": int(list(perm.index).index("capture_session") + 1)}


# --------------------------------------------------------------------------
def agg(vals):
    v = np.array(vals, float)
    return {"mean": float(np.nanmean(v)), "std": float(np.nanstd(v, ddof=1)) if len(v) > 1 else 0.0, "n": int(len(v))}


def summarise(results):
    S = {}
    keys = ["accuracy", "macro_f1", "attack_roc_auc", "attack_pr_auc", "attack_tpr_at_1pct_fpr",
            "attack_tpr_at_0.1pct_fpr", "alert_precision", "alert_recall", "benign_false_alert_rate"]
    S["detection"] = {m: {k: agg([r["detection"][m][k] for r in results]) for k in keys}
                      for m in ("explainable_gb", "blackbox_mlp")}
    S["detection_paired_diff_gb_minus_mlp"] = {}
    for k in keys:
        d = [r["detection"]["explainable_gb"][k] - r["detection"]["blackbox_mlp"][k] for r in results]
        S["detection_paired_diff_gb_minus_mlp"][k] = {**agg(d), "gb_better_in": int(sum(x > 0 for x in d))}
    S["per_class_f1"] = {m: {c: agg([r["detection"][m]["per_class"][c]["f1"] for r in results]) for c in CLASSES}
                         for m in ("explainable_gb", "blackbox_mlp")}
    for blk in ("faithfulness", "compactness", "stability"):
        S[blk] = {k: agg([r[blk][k] for r in results]) for k in results[0][blk]}
    conf = np.concatenate([r["_pool"]["conf"] for r in results])
    Sg = np.concatenate([r["_pool"]["S"] for r in results])
    bal = np.concatenate([r["_pool"]["bal"] for r in results])
    val = np.concatenate([r["_pool"]["valid"] for r in results])
    S["triage_pooled_over_seeds"] = triage_simulation(conf, Sg, val, {"evidence_balance": bal})
    S["confidence_auc_all_alerts"] = {m: agg([r["confidence_auc_all_alerts"][m] for r in results])
                                      for m in ("explainable_gb", "blackbox_mlp")}
    S["additivity_max_abs_err_over_seeds"] = float(max(r["additivity_max_abs_err"] for r in results))
    S["shap_backend"] = results[0]["shap_backend"]
    return S


# --------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------
def fig_detection(S):
    names = [("macro_f1", "Macro F1"), ("attack_pr_auc", "Attack PR-AUC"),
             ("attack_tpr_at_1pct_fpr", "Attack recall\n@ 1% false-alarm rate"), ("alert_precision", "Alert precision")]
    fig, ax = new_fig(figsize=(8.6, 4.2))
    x = np.arange(len(names)); w = 0.36
    for j, (m, lab, col) in enumerate([("explainable_gb", "Explainable GB (+SHAP)", BLUE),
                                       ("blackbox_mlp", "Black-box MLP", ORANGE)]):
        mu = [S["detection"][m][k]["mean"] for k, _ in names]
        sd = [S["detection"][m][k]["std"] for k, _ in names]
        b = ax.bar(x + (j - .5) * w, mu, w - 0.03, yerr=sd, color=col, label=lab,
                   error_kw=dict(ecolor=INK2, lw=1, capsize=3))
        for xi, v in zip(x + (j - .5) * w, mu):
            ax.text(xi, v + 0.012, f"{v:.3f}", ha="center", va="bottom", fontsize=8, color=INK)
    ax.set_xticks(x); ax.set_xticklabels([n for _, n in names], color=INK)
    lo = min(S["detection"][m][k]["mean"] for m in S["detection"] for k, _ in names)
    ax.set_ylim(max(0, lo - 0.12), 1.04)
    ax.set_title(f"Detection quality (mean ± sd over {S['detection']['explainable_gb']['macro_f1']['n']} seeds)",
                 loc="left", color=INK, fontsize=11)
    style_ax(ax); ax.legend(frameon=False, loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=2, labelcolor=INK2, fontsize=9)
    ax.set_ylabel("score (y-axis truncated)", color=INK2, fontsize=9)
    fig.tight_layout(); fig.savefig(f"{FIG}/detection_comparison.png", dpi=160); plt.close(fig)


def fig_alert_cards(K, feats):
    gb, Xa, phi, ka, yte = K["gb"], K["Xa"], K["phi"], K["ka"], K["yte"]
    cls = np.array(gb.classes_)
    truth = yte.loc[Xa.index].values if hasattr(yte, "loc") else None
    truth = np.array([K["yte"].iloc[K["Xte"].index.get_loc(i)] for i in Xa.index])
    rng = np.random.default_rng(7)
    picks = []
    for c in ["DoS", "PortScan", "BruteForce", "Botnet", "WebAttack"]:
        ok = np.where((cls[ka] == c) & (truth == c))[0]
        if len(ok):
            picks.append((rng.choice(ok), "true alert"))
    fp = np.where(truth == "BENIGN")[0]
    if len(fp):
        picks.append((rng.choice(fp), "FALSE alert"))
    fig, axes = plt.subplots(2, 3, figsize=(14, 7)); fig.patch.set_facecolor(SURFACE)
    for ax, (i, tag) in zip(axes.ravel(), picks):
        order = np.argsort(-np.abs(phi[i]))[:7][::-1]
        vals = phi[i][order]
        labels = [f"{FEATURE_DESC.get(feats[j], feats[j])} = {_short(Xa.iloc[i, j])}" for j in order]
        ax.barh(range(len(order)), vals, color=[BLUE if v > 0 else ORANGE for v in vals], height=0.62)
        ax.set_yticks(range(len(order))); ax.set_yticklabels(labels, fontsize=7.5, color=INK)
        ax.axvline(0, color=INK2, lw=0.8)
        pv = K["pg"][K["Xte"].index.get_loc(Xa.index[i]), ka[i]]
        ax.set_title(f"{cls[ka[i]]}: {tag}\np = {pv:.2f}, ground truth {truth[i]}",
                     loc="left", fontsize=9, color=INK)
        style_ax(ax, "x"); ax.grid(axis="y", visible=False)
        ax.set_xlabel("SHAP contribution (log-odds)", fontsize=8, color=INK2)
    for ax in axes.ravel()[len(picks):]:
        ax.axis("off")
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(color=BLUE, label="pushes toward the verdict"), Patch(color=ORANGE, label="pushes against it")],
               loc="lower center", ncol=2, frameon=False, labelcolor=INK2, fontsize=9)
    fig.suptitle("Per-alert explanations (randomly chosen test alerts; top 7 features by |contribution|)",
                 x=0.01, ha="left", color=INK, fontsize=11)
    fig.tight_layout(rect=(0, 0.04, 1, 0.96)); fig.savefig(f"{FIG}/alert_explanations.png", dpi=160); plt.close(fig)


def _short(v):
    v = float(v)
    if abs(v) >= 1e6: return f"{v/1e6:.1f}M"
    if abs(v) >= 1e4: return f"{v/1e3:.0f}k"
    return f"{v:.0f}" if v.is_integer() else f"{v:.1f}"


def fig_global_heatmap(K, feats):
    cls = np.array(K["gb"].classes_)
    phi, ka = K["phi"], K["ka"]
    present = [c for c in cls if c != "BENIGN" and (cls[ka] == c).any()]
    M = np.array([np.abs(phi[cls[ka] == c]).mean(0) for c in present])
    M = M / M.sum(1, keepdims=True)
    top = np.argsort(-M.mean(0))[:12]
    cmap = LinearSegmentedColormap.from_list("seq", ["#f1f6fd", "#9ec1ee", "#2a78d6", "#143f78"])
    fig, ax = new_fig(figsize=(10, 3.8))
    im = ax.imshow(M[:, top], cmap=cmap, aspect="auto")
    ax.set_xticks(range(len(top))); ax.set_xticklabels([feats[j] for j in top], rotation=35, ha="right", fontsize=8, color=INK)
    ax.set_yticks(range(len(present))); ax.set_yticklabels(present, fontsize=9, color=INK)
    for a in range(M.shape[0]):
        for b, j in enumerate(top):
            ax.text(b, a, f"{M[a, j]*100:.0f}", ha="center", va="center", fontsize=7.5,
                    color="white" if M[a, j] > 0.17 else INK)
    ax.set_title("What drives each verdict: share of mean |SHAP| (%), top 12 features", loc="left", color=INK, fontsize=11)
    for s in ax.spines.values(): s.set_visible(False)
    ax.tick_params(length=0)
    fig.tight_layout(); fig.savefig(f"{FIG}/global_drivers_by_class.png", dpi=160); plt.close(fig)


def fig_faithfulness(S):
    f = S["faithfulness"]
    items = [("SHAP top-3\n(this alert)", "mean_prob_drop_shap_top_k", BLUE),
             ("Top-3 by class-level\nimportance", "mean_prob_drop_class_global_top_k", ORANGE),
             ("3 random\nfeatures", "mean_prob_drop_random_k", NEUTRAL)]
    fig, ax = new_fig(figsize=(6.6, 4))
    for i, (lab, k, col) in enumerate(items):
        mu, sd = f[k]["mean"], f[k]["std"]
        ax.bar(i, mu, 0.55, yerr=sd, color=col, error_kw=dict(ecolor=INK2, lw=1, capsize=3))
        ax.text(i, mu + 0.02, f"{mu:.2f}", ha="center", fontsize=9, color=INK)
    ax.set_xticks(range(3)); ax.set_xticklabels([i[0] for i in items], fontsize=9, color=INK)
    ax.set_ylabel("mean drop in verdict probability", fontsize=9, color=INK2); ax.set_ylim(0, 1)
    ax.set_title("Deletion test: reset 3 features to typical-benign values", loc="left", fontsize=11, color=INK)
    style_ax(ax); fig.tight_layout(); fig.savefig(f"{FIG}/faithfulness.png", dpi=160); plt.close(fig)


def fig_shortcut(SC):
    conds = [("clean_model_clean_test", "Trained without artifact"),
             ("leaky_model_same_distribution_test", "Trained with artifact,\ntested with same artifact"),
             ("leaky_model_shifted_test", "Trained with artifact,\nartifact decorrelated at test")]
    fig, ax = new_fig(figsize=(8.6, 4.2)); w = 0.36; x = np.arange(len(conds))
    for j, (m, lab, col) in enumerate([("explainable_gb", "Explainable GB", BLUE), ("blackbox_mlp", "Black-box MLP", ORANGE)]):
        v = [SC["macro_f1"][m][c] for c, _ in conds]
        ax.bar(x + (j - .5) * w, v, w - .03, color=col, label=lab)
        for xi, vi in zip(x + (j - .5) * w, v):
            ax.text(xi, vi + .01, f"{vi:.3f}", ha="center", fontsize=8, color=INK)
    ax.set_xticks(x); ax.set_xticklabels([l for _, l in conds], fontsize=8.5, color=INK)
    lo = min(SC["macro_f1"][m][c] for m in SC["macro_f1"] for c, _ in conds)
    ax.set_ylim(max(0, lo - .1), 1.0); ax.set_ylabel("macro F1 (y-axis truncated)", fontsize=9, color=INK2)
    ax.set_title("Shortcut learning: a dataset artifact inflates in-distribution scores", loc="left", fontsize=11, color=INK)
    style_ax(ax); ax.legend(frameon=False, labelcolor=INK2, fontsize=9, loc="upper right")
    fig.tight_layout(); fig.savefig(f"{FIG}/shortcut_audit.png", dpi=160); plt.close(fig)


# --------------------------------------------------------------------------
def write_report(S, SC, results, args, n_alerts_total):
    d, dd = S["detection"], S["detection_paired_diff_gb_minus_mlp"]
    f, c, st, tri = S["faithfulness"], S["compactness"], S["stability"], S["triage_pooled_over_seeds"]
    pm = lambda a: f"{a['mean']:.3f} ± {a['std']:.3f}"
    ci = lambda t: f"{t[0]:.3f} [{t[1]:.3f}, {t[2]:.3f}]"
    L = []
    L.append("# Results report (auto-generated by run_all.py)\n")
    src = f"CICIDS2017 CSVs at `{args.cicids}`" if args.cicids else "synthetic CICFlowMeter-style flows"
    L.append(f"Data: {src}; {args.n_flows} flows per seed, 70/30 stratified split, {len(results)} seeds. "
             f"SHAP backend: **{S['shap_backend']}**. Max |base + sum(SHAP) − model score| over all explained alerts: "
             f"{S['additivity_max_abs_err_over_seeds']:.1e}.\n")
    L.append("## 1. Detection quality (test set, mean ± sd over seeds)\n")
    L.append("| Metric | Explainable GB | Black-box MLP | GB − MLP (mean ± sd) | GB better in |\n|---|---|---|---|---|")
    for k, lab in [("macro_f1", "Macro F1"), ("accuracy", "Accuracy"), ("attack_roc_auc", "Attack ROC-AUC"),
                   ("attack_pr_auc", "Attack PR-AUC"), ("attack_tpr_at_1pct_fpr", "Attack recall @ 1% FPR"),
                   ("attack_tpr_at_0.1pct_fpr", "Attack recall @ 0.1% FPR"), ("alert_precision", "Alert precision"),
                   ("alert_recall", "Alert recall"), ("benign_false_alert_rate", "Benign flows falsely alerted")]:
        L.append(f"| {lab} | {pm(d['explainable_gb'][k])} | {pm(d['blackbox_mlp'][k])} | "
                 f"{dd[k]['mean']:+.4f} ± {dd[k]['std']:.4f} | {dd[k]['gb_better_in']}/{dd[k]['n']} |")
    L.append("\nPer-class F1:\n\n| Class | Explainable GB | Black-box MLP |\n|---|---|---|")
    for cl in CLASSES:
        L.append(f"| {cl} | {pm(S['per_class_f1']['explainable_gb'][cl])} | {pm(S['per_class_f1']['blackbox_mlp'][cl])} |")
    L.append("\n## 2. Are the explanations trustworthy and readable?\n")
    L.append(f"Explained {n_alerts_total} sampled alerts across seeds.\n")
    L.append("| Property | Result |\n|---|---|")
    L.append(f"| Deletion test: drop in verdict probability when SHAP's top-3 features are reset to typical-benign | {pm(f['mean_prob_drop_shap_top_k'])} |")
    L.append(f"| ...same, top-3 by class-level importance (what a global chart offers) | {pm(f['mean_prob_drop_class_global_top_k'])} |")
    L.append(f"| ...same, 3 random features | {pm(f['mean_prob_drop_random_k'])} |")
    L.append(f"| Verdict flips: SHAP top-3 / class-global top-3 / random | {pm(f['verdict_flip_rate_shap_top_k'])} / {pm(f['verdict_flip_rate_class_global_top_k'])} / {pm(f['verdict_flip_rate_random_k'])} |")
    L.append(f"| Compactness: median features needed for 80% of an alert's |SHAP| mass (of {int(c['total_features']['mean'])}) | {c['median_features_for_80pct']['mean']:.1f} |")
    L.append(f"| Share of |SHAP| mass in the top 3 features | {pm(c['mean_top3_mass_share'])} |")
    L.append(f"| Stability: top-3 Jaccard under ~5% input noise (all / verdict unchanged) | {pm(st['mean_top_k_jaccard'])} / {pm(st['mean_top_k_jaccard_when_verdict_unchanged'])} |")
    L.append("\n## 3. Simulated analyst triage (proxy, not a human study)\n")
    L.append(f"Pooled over seeds: {tri['n_alerts']} alerts, of which {tri['n_false_alerts']} were false alarms "
             "(benign flow flagged). AUROC for telling true attacks from false alarms among raised alerts "
             "(95% bootstrap CI):\n")
    L.append("| Signal available to the analyst | AUROC |\n|---|---|")
    L.append(f"| Model confidence only (all a black box offers) | {ci(tri['auc_confidence'])} |")
    L.append(f"| SHAP-vs-expected-signature agreement | {ci(tri['auc_explanation_agreement'])} |")
    L.append(f"| Both (mean of percentile ranks) | {ci(tri['auc_combined'])} |")
    L.append(f"| *Exploratory, added after the signature signal showed no benefit:* share of |SHAP| mass supporting the verdict | {ci(tri['auc_evidence_balance'])} |")
    L.append(f"| *Exploratory:* confidence + that share | {ci(tri['auc_confidence_plus_evidence_balance'])} |")
    L.append("\nGating policy (deprioritise alerts whose top SHAP features match none/too few of the expected signature features):\n")
    L.append("| Gate | Alerts kept | False alarms removed | True attacks lost | Precision before → after |\n|---|---|---|---|---|")
    for g in tri["gating"]:
        L.append(f"| {g['gate']} | {g['alerts_kept']:.1%} | {g['false_alerts_removed']:.1%} | {g['true_alerts_lost']:.1%} | {g['precision_before']:.3f} → {g['precision_after']:.3f} |")
    ca = S["confidence_auc_all_alerts"]
    L.append(f"\nFor reference, confidence-only AUROC over *all* alerts: GB {pm(ca['explainable_gb'])}, MLP {pm(ca['blackbox_mlp'])}.\n")
    L.append("## 4. Shortcut-learning audit (seed 0)\n")
    m = SC["macro_f1"]
    L.append("| Macro F1 | Explainable GB | Black-box MLP |\n|---|---|---|")
    for key, lab in [("clean_model_clean_test", "Trained without artifact"),
                     ("leaky_model_same_distribution_test", "Trained with artifact, same-artifact test"),
                     ("leaky_model_shifted_test", "Trained with artifact, artifact decorrelated at test")]:
        L.append(f"| {lab} | {m['explainable_gb'][key]:.3f} | {m['blackbox_mlp'][key]:.3f} |")
    L.append(f"\nSHAP global importance of the leaky GB ranks `capture_session` **#{SC['shap_rank_of_artifact']}** "
             f"({SC['shap_share_of_artifact']:.1%} of attribution mass). Top 5: {SC['shap_global_importance_leaky_gb_top5']}.")
    L.append(f"\nThe MLP's only recourse is post-hoc permutation importance, which ranks the artifact **#{SC['mlp_permutation_rank_of_artifact']}** "
             f"(top 5 macro-F1 drops: {SC['mlp_permutation_importance_top5_macro_f1_drop']}). "
             "So a global leak can also be found on a black box with extra tooling; what SHAP adds is the same view "
             "per alert, exact additivity, and no extra scoring passes.\n")
    open(f"{OUT}/report.md", "w", encoding="utf-8").write("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--n-flows", type=int, default=40000)
    ap.add_argument("--explain-cap", type=int, default=1000)
    ap.add_argument("--stab-cap", type=int, default=500)
    ap.add_argument("--backend", default="builtin", choices=["builtin", "shap", "auto"],
                    help="builtin = bundled exact TreeSHAP (verified; default); shap = official package (run verify_shap_backend.py first)")
    ap.add_argument("--cicids", default=None, help="path to CICIDS2017 CSV file/dir (instead of synthetic data)")
    args = ap.parse_args()
    if args.quick:
        args.seeds, args.n_flows, args.explain_cap, args.stab_cap = 2, 12000, 300, 150
    os.makedirs(FIG, exist_ok=True)
    results, K = [], None
    for s in range(args.seeds):
        r = one_seed(args, s, keep=(s == 0))
        if s == 0:
            K = r.pop("_keep")
        results.append(r)
        d = r["detection"]
        print(f"[seed {s}] {r['seconds']}s  macroF1 GB {d['explainable_gb']['macro_f1']:.4f} / MLP {d['blackbox_mlp']['macro_f1']:.4f}", flush=True)
    S = summarise(results)
    print("shortcut audit...", flush=True)
    SC = shortcut_audit(args, K)

    feats = list(K["Xte"].columns)
    print("building alert table...", flush=True)
    atab, _, _ = build_alert_table(K["gb"], K["Xte"], K["mte"], K["yte"])
    atab.to_csv(f"{OUT}/alerts.csv", index=False, encoding="utf-8")
    with open(f"{OUT}/alerts_sample.md", "w", encoding="utf-8") as fh:
        fh.write("# Sample alert cards (seed 0 test set, random sample)\n\n")
        samp = atab.sample(min(12, len(atab)), random_state=3)
        fp = atab[atab.true_label == "BENIGN"].head(3)
        for _, a in pd.concat([samp, fp]).drop_duplicates("flow_id").iterrows():
            fh.write(alert_card_markdown(a) + "\n")
    save_bundle(K["gb"], feats, f"{OUT}/model.joblib")

    fig_detection(S); fig_alert_cards(K, feats); fig_global_heatmap(K, feats)
    fig_faithfulness(S); fig_shortcut(SC)
    n_alerts_total = sum(r["faithfulness"]["n_alerts"] for r in results)
    write_report(S, SC, results, args, n_alerts_total)
    json.dump({"args": vars(args), "summary": S, "shortcut_audit": SC,
               "per_seed": [{k: v for k, v in r.items() if k != "_pool"} for r in results]},
              open(f"{OUT}/metrics.json", "w", encoding="utf-8"), indent=2, default=float)
    print("done ->", OUT, flush=True)


if __name__ == "__main__":
    main()
