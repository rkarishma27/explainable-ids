"""Correctness tests for the built-in TreeSHAP and the CICIDS loader.

Run:  python -m pytest -q tests      (or)      python tests/test_treeshap.py
"""
import itertools
import math
import os
import sys
import tempfile

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sklearn.ensemble import GradientBoostingClassifier  # noqa: E402
from sklearn.tree import DecisionTreeRegressor  # noqa: E402

from ids.data import FEATURES, generate_flows, load_cicids2017  # noqa: E402
from ids.treeshap import ShapBackend, _TreeArrays, tree_shap_one  # noqa: E402


def _value_fn(t: _TreeArrays, x, S):
    """Path-dependent conditional expectation v(S) used by TreeSHAP."""
    def rec(node):
        if t.left[node] == -1:
            return t.value[node]
        f = t.feature[node]
        l, r = t.left[node], t.right[node]
        if f in S:
            return rec(l if x[f] <= t.threshold[node] else r)
        return (t.cover[l] * rec(l) + t.cover[r] * rec(r)) / t.cover[node]
    return rec(0)


def _brute_shapley(t, x, M):
    phi = np.zeros(M)
    for i in range(M):
        others = [j for j in range(M) if j != i]
        for size in range(M):
            w = math.factorial(size) * math.factorial(M - size - 1) / math.factorial(M)
            for S in itertools.combinations(others, size):
                S = set(S)
                phi[i] += w * (_value_fn(t, x, S | {i}) - _value_fn(t, x, S))
    return phi


def test_single_tree_matches_brute_force_shapley():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(600, 6))
    y = X[:, 0] * X[:, 1] + np.sin(X[:, 2]) + 0.5 * (X[:, 3] > 0) + 0.1 * rng.normal(size=600)
    # depth 5 so features repeat along a path (exercises unwind logic)
    t = _TreeArrays(DecisionTreeRegressor(max_depth=5, random_state=0).fit(X, y).tree_)
    for r in range(15):
        x = rng.normal(size=6)
        phi = np.zeros(6)
        tree_shap_one(t, x, phi)
        np.testing.assert_allclose(phi, _brute_shapley(t, x, 6), atol=1e-9)
        # local accuracy for one tree: f(x) = E[f] + sum(phi)
        assert abs(_value_fn(t, x, set(range(6))) - (t.expected_value() + phi.sum())) < 1e-9


def test_gradient_boosting_local_accuracy_every_class():
    X, y, _ = generate_flows(4000, seed=1)
    gb = GradientBoostingClassifier(n_estimators=25, max_depth=3, random_state=0).fit(X.values, y.values)
    be = ShapBackend(gb, backend="builtin", n_jobs=1)
    Xt = X.values[:12]
    raw = gb.decision_function(Xt)
    for k in range(len(gb.classes_)):
        phi = be.shap_values(Xt, k)
        recon = be.base_value(k) + phi.sum(axis=1)
        np.testing.assert_allclose(recon, raw[:, k], atol=1e-8)


def test_leaf_values_are_bounded_and_shap_still_additive():
    """Regression: sklearn's multiclass Newton step can create ~1e20 leaves; the
    model must clip them (and SHAP must still sum to the model score)."""
    from ids.models import ExplainableIDS
    X, y, _ = generate_flows(5000, seed=3)
    m = ExplainableIDS(n_estimators=30, max_abs_leaf=5.0, backend="builtin", n_jobs=1).fit(X, y)
    for e in m.gb_.estimators_.ravel():
        t = e.tree_
        assert np.abs(t.value[t.children_left == -1]).max() <= 5.0 + 1e-12
    Xt = X.iloc[:10]
    phi, base, k = m.explain(Xt)
    raw = m.gb_.decision_function(Xt.values)
    np.testing.assert_allclose(base + phi.sum(1), raw[np.arange(10), k], atol=1e-8)


def test_blackbox_trains_with_string_labels_and_returns_strings():
    """Regression: scikit-learn 1.8's MLP early stopping crashed on string labels."""
    from ids.models import BlackBoxBaseline
    X, y, _ = generate_flows(3000, seed=4)
    m = BlackBoxBaseline(max_iter=30).fit(X, y)
    pred = m.predict(X.iloc[:20])
    assert set(pred) <= set(y) and m.predict_proba(X.iloc[:5]).shape == (5, len(m.classes_))


def test_saved_model_roundtrip_and_version_guard():
    """Regression: a model pickled elsewhere must never be unpickled on different
    library versions (it crashed on pandas 2.3 when written with pandas 3)."""
    import json
    from ids.models import ExplainableIDS, load_bundle, save_bundle
    X, y, _ = generate_flows(3000, seed=5)
    m = ExplainableIDS(n_estimators=10, n_jobs=1).fit(X, y)
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "m.joblib")
        assert load_bundle(p)[0] is None                       # nothing saved yet
        save_bundle(m, list(X.columns), p)
        b, why = load_bundle(p)
        assert b is not None, why
        np.testing.assert_allclose(b["model"].predict_proba(X.iloc[:10]), m.predict_proba(X.iloc[:10]))
        meta = json.load(open(p + ".versions.json"))
        meta["pandas"] = "0.0.1"                               # pretend another pandas wrote it
        json.dump(meta, open(p + ".versions.json", "w"))
        b, why = load_bundle(p)
        assert b is None and "pandas" in why


def test_cicids_loader_on_fabricated_csv():
    cols = ["Destination Port", "Flow Duration", "Total Fwd Packets", "Total Backward Packets",
            "Total Length of Fwd Packets", "Total Length of Bwd Packets", "Fwd Packet Length Mean",
            "Bwd Packet Length Mean", "Fwd Packet Length Max", "Flow Bytes/s", "Flow Packets/s",
            "Flow IAT Mean", "Flow IAT Std", "Fwd IAT Mean", "SYN Flag Count", "ACK Flag Count",
            "RST Flag Count", "PSH Flag Count", "FIN Flag Count", "Init_Win_bytes_forward",
            "Init_Win_bytes_backward", "Down/Up Ratio", "Packet Length Variance", "Active Mean",
            "Idle Mean", "Label"]
    # leading spaces in headers mimic the real files
    rows = [[80, 100, 2, 1, 10, 5, 5, 5, 6, 150.0, 30000.0, 50, 2, 50, 1, 1, 0, 0, 0, 100, 100, 0, 1, 10, 0, "BENIGN"],
            [22, 900, 9, 9, 90, 90, 10, 10, 12, 200.0, 20000.0, 50, 2, 50, 1, 1, 0, 3, 0, 100, 100, 1, 1, 10, 0, "SSH-Patator"],
            [80, 900, 9, 9, 90, 90, 10, 10, 12, float("inf"), 20000.0, 50, 2, 50, 1, 1, 0, 3, 0, 100, 100, 1, 1, 10, 0, "BENIGN"],
            [80, 1, 2, 0, 0, 0, 0, 0, 0, 1.0, 1.0, 1, 1, 1, 1, 0, 1, 0, 0, 100, 0, 0, 0, 0, 0, "Heartbleed"]]
    df = pd.DataFrame(rows, columns=[" " + c for c in cols])
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "x.csv")
        df.to_csv(p, index=False)
        X, y, _ = load_cicids2017(p)
    assert list(X.columns) == FEATURES
    assert list(y) == ["BENIGN", "BruteForce"]  # inf row and unmapped label dropped


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)
