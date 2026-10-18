# Explainable Network IDS with SHAP

A network-flow classifier that flags malicious flows (DoS, port scan, brute force, botnet C2,
web attacks) and tells the analyst **why** each flow was flagged, compared against an
unexplainable neural-network baseline on detection quality and on measurable proxies for
analyst usability.

```
BENIGN | DoS | PortScan | BruteForce | Botnet | WebAttack      (25 CICFlowMeter-style features)
```

Example alert (from `score_flows.py`):

```
Flow 13  |  BruteForce (confidence 100%)
- Why: destination port = 21 (+6.11); mean idle time (us) = 0 (+1.97); inter-arrival time std-dev (us) = 100.6k (+1.06)
- Argues against: byte rate (B/s) = 350.03 (-0.34); flow duration (us) = 2.9M (-0.06)
```

## Quick start

```bash
pip install -r requirements.txt          # shap is optional (see below)
python tests/test_treeshap.py            # 6 correctness/regression tests, ~15 s
python run_all.py --quick                # ~2 min smoke run
python run_all.py                        # full run: 5 seeds x 40k flows, ~16 min on 2 cores
python score_flows.py my_flows.csv       # explain alerts on your own flows (first run trains a model, ~1-2 min)
```

Everything lands in `outputs/`: `report.md` (all tables), `metrics.json`, `figures/*.png`,
`alerts.csv` (every test alert with its reasons), `alerts_sample.md`, `model.joblib` (+ `.versions.json`).
The `outputs/` folder in this zip is from the full run described below, minus `model.joblib`:
a pickled model only loads safely on the exact scikit-learn/pandas/numpy versions that wrote it, so
`score_flows.py` trains one locally on first use (~1-2 min) and caches it. `run_all.py` writes a fresh one for your machine.

**Real data:** `python run_all.py --cicids /path/to/CICIDS2017/CSVs --n-flows 200000`
(see Limitations: the loader has not been run on the real files).

## What is in here

| Path | Purpose |
|---|---|
| `ids/data.py` | Synthetic flow generator (6 classes, overlapping "hard" benign look-alikes, 0.3% label noise); CICIDS2017 loader; analyst signature dictionary |
| `ids/treeshap.py` | Exact path-dependent TreeSHAP for scikit-learn gradient boosting (the default backend); official `shap` is opt-in |
| `ids/models.py` | `ExplainableIDS` (gradient boosting + SHAP) and `BlackBoxBaseline` (MLP, same train data, log-scaling + port one-hots) |
| `ids/explain.py` | Turns SHAP vectors into analyst-readable alert text |
| `ids/evaluate.py` | Detection metrics + usability proxies (faithfulness, compactness, stability, triage simulation) |
| `run_all.py` | The experiment: 5 seeds, shortcut-learning audit, figures, report |
| `tests/test_treeshap.py` | TreeSHAP vs brute-force Shapley values, additivity for every class, leaf-bound, string-label MLP, model-save version guard, loader |
| `verify_shap_backend.py` | Checks official `shap.TreeExplainer` against the bundled TreeSHAP on your installed versions |

## Results (synthetic data, 5 seeds, mean ± sd; full tables in `outputs/report.md`)

**Detection: the explainable model is at least as good, with a modest edge.**

| Metric | Explainable GB | Black-box MLP | Seeds where GB is better |
|---|---|---|---|
| Macro F1 | 0.909 ± 0.002 | 0.899 ± 0.008 | 5/5 |
| Attack recall @ 1% false-alarm rate | 0.910 ± 0.006 | 0.885 ± 0.011 | 5/5 |
| Attack recall @ 0.1% false-alarm rate | 0.798 ± 0.017 | 0.775 ± 0.013 | 5/5 |
| Alert precision | 0.961 ± 0.004 | 0.946 ± 0.007 | 5/5 |
| Attack ROC-AUC | 0.991 ± 0.001 | 0.990 ± 0.001 | 3/5 (a tie) |
| Alert recall | 0.913 ± 0.006 | 0.907 ± 0.011 | 3/5 (a tie) |

The macro-F1 gap (+0.010) is about one standard deviation of the seed-to-seed difference,
so read it as "no accuracy is given up for explainability, possibly a small gain", not as a
decisive win. The MLP is slightly better on BruteForce F1 (0.987 vs 0.983); GB is clearly
better on PortScan (0.980 vs 0.948). Neither model was hyperparameter-tuned.

**Explanation quality: the explanations are faithful, compact and stable.**

| Check | Result |
|---|---|
| Deletion test: reset SHAP's top-3 features to typical-benign values, verdict probability drops by | 0.82 (vs 0.69 for top-3 class-level importance, 0.13 for 3 random features) |
| Verdict flips after that deletion | 90% (74% class-level, 15% random) |
| Features needed to cover 80% of an alert's attribution (of 25) | median 5 |
| Top-3 feature overlap under ~5% input noise (Jaccard) | 0.96 |
| Additivity: base + sum(SHAP) vs model score | max error 2.6e-14 |

**Analyst usability: what was and was not shown.**

- The baseline has *no* per-alert explanation, so on the explanation metrics above there is
  nothing to compare against; the claim is only that an explanation exists and is faithful.
- **A simulated-triage experiment did not show an explanation-driven benefit.** Telling real
  attacks from false alarms among raised alerts, model confidence alone scored AUROC 0.934.
  Agreement between SHAP reasons and an analyst-signature dictionary scored 0.293, which is
  *worse than chance*: false alarms are flows that look like the attack, so they match the
  signature even better. Gating alerts on that agreement removed few false alarms or lost
  many true attacks. An exploratory share-of-evidence-for-verdict signal (added after the
  first one failed, so treat it as a post-hoc probe) scored 0.580 alone and lowered the
  combined score. Conclusion: in this simulation, explanations did not improve alert
  prioritisation beyond confidence.
- **Shortcut-learning audit:** adding a fake `capture_session` column that leaks the label
  inflates macro-F1 from 0.910 to 0.974 and collapses it to 0.690 once the leak is removed.
  SHAP's global view ranked the artifact **#3** (13.8% of attribution), so it surfaces the
  problem but did not make it the top item. A post-hoc permutation-importance pass on the
  black-box MLP ranked it **#1**, so the black box is not blind if you add tooling. What SHAP
  gives you is the same information per alert, exact additivity, and no extra scoring passes.

So, against the original goal ("beats an unexplainable baseline on detection quality and
analyst usability"): the detection part is supported (modestly), the usability part is
supported only as "explanations are faithful, compact and stable", not as a demonstrated
analyst-workflow gain. Showing that needs a study with human analysts.

## Limitations (please read)

- **Synthetic data.** No dataset could be downloaded in the build environment. The generator
  is written from general knowledge of these attacks, so class structure is cleaner than real
  traffic and the analyst-signature dictionary and the generator share assumptions. Absolute
  numbers say nothing about your network; run `--cicids` on real data.
- **The CICIDS2017 loader has only been exercised on a tiny fabricated CSV**, using the
  documented column names. Expect to adjust it on the real files.
- **Official `shap` was not used for the reported results.** The build environment could not
  install it (package index blocked), so results use the bundled TreeSHAP, validated against
  brute-force Shapley values and the additivity property. The `--backend shap` path was **never
  executed** by the author; run `verify_shap_backend.py` before relying on it.
- **Library versions:** developed on scikit-learn 1.9.1 / pandas 3.0 / numpy 2.5. It has since been
  fixed for two failures seen on scikit-learn 1.8.0 / pandas 2.3.3 (MLP early stopping with string
  labels; an unloadable pickled model). The MLP fix was verified here to give identical numbers,
  but the author could not install those older versions to reproduce your environment.
- **No human study.** Every usability number is a computable proxy. The deletion test edits
  flows to values that may be off-manifold.
- **scikit-learn gradient-boosting pitfall (handled):** its multiclass Newton step can produce
  leaf values near 1e20 when a leaf's samples are already saturated, which silently corrupts
  log-odds attributions (this showed up as a 1e3 additivity error on one seed). `ExplainableIDS`
  winsorises leaves beyond |100| (22 leaves in the affected model) and `run_all.py` aborts if
  additivity exceeds 1e-6. Healthy leaves reach about ±60, so only pathological ones are touched.
- Explanations describe the *model*, not the attacker: some reasons are uninformative to a
  human (e.g. "byte rate = 0" supporting a DoS verdict). Attributions are in log-odds of the
  predicted class against a training-data background.
- Usability proxies were computed on a random sample of up to 1000 alerts per seed (5000 total).
- `min_samples_leaf=20` and the other hyperparameters were fixed in advance and not tuned on
  the test set.

## Using real `shap`

The default backend is the bundled TreeSHAP, because it is the one validated against brute-force
Shapley values. To use the official package, first check it agrees on your installed versions:

```bash
pip install shap
python verify_shap_backend.py         # prints PASS or MISMATCH per class
python run_all.py --backend shap      # only if it passed
```

The official-`shap` path has not been run by the author (the build environment could not install it).
