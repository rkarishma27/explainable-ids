"""Exact TreeSHAP for scikit-learn GradientBoostingClassifier.

Implements the path-dependent TreeSHAP algorithm (Lundberg, Erion & Lee 2018,
"Consistent individualized feature attribution for tree ensembles", Algorithm 2)
in plain Python/NumPy, so the project works where the `shap` package cannot be
installed. For the margin (log-odds) of one class k:

    margin_k(x) = base_k + sum_i phi_i(x)          (exact, "local accuracy")

`ShapBackend` uses the real `shap` package when it is importable
(backend="auto"|"shap") and otherwise this implementation (backend="builtin").
Correctness of the built-in version is checked in tests/test_treeshap.py against
brute-force Shapley values and the additivity property.
"""
from __future__ import annotations

import numpy as np
from joblib import Parallel, delayed


class _TreeArrays:
    """Flat arrays for one regression tree (leaf value = raw margin increment)."""

    def __init__(self, sk_tree, scale: float = 1.0):
        t = sk_tree
        self.left = t.children_left.copy()
        self.right = t.children_right.copy()
        self.feature = t.feature.copy()
        self.threshold = t.threshold.copy()
        self.value = t.value[:, 0, 0].astype(float) * scale
        self.cover = t.weighted_n_node_samples.astype(float)

    def expected_value(self) -> float:
        leaf = self.left == -1
        return float(np.sum(self.value[leaf] * self.cover[leaf]) / self.cover[0])


# ---- path bookkeeping (feature, zero_fraction, one_fraction, pweight) -------
def _extend(m, pz, po, pi):
    d = len(m)
    m.append([pi, pz, po, 1.0 if d == 0 else 0.0])
    for i in range(d - 1, -1, -1):
        m[i + 1][3] += po * m[i][3] * (i + 1) / (d + 1)
        m[i][3] = pz * m[i][3] * (d - i) / (d + 1)


def _unwind(m, idx):
    d = len(m) - 1
    po, pz = m[idx][2], m[idx][1]
    nxt = m[d][3]
    for i in range(d - 1, -1, -1):
        if po != 0:
            tmp = m[i][3]
            m[i][3] = nxt * (d + 1) / ((i + 1) * po)
            nxt = tmp - m[i][3] * pz * (d - i) / (d + 1)
        else:
            m[i][3] = m[i][3] * (d + 1) / (pz * (d - i))
    for i in range(idx, d):
        m[i][0], m[i][1], m[i][2] = m[i + 1][0], m[i + 1][1], m[i + 1][2]
    m.pop()


def _unwound_sum(m, idx):
    d = len(m) - 1
    po, pz = m[idx][2], m[idx][1]
    nxt = m[d][3]
    total = 0.0
    for i in range(d - 1, -1, -1):
        if po != 0:
            tmp = nxt * (d + 1) / ((i + 1) * po)
            total += tmp
            nxt = m[i][3] - tmp * pz * (d - i) / (d + 1)
        else:
            total += (m[i][3] / pz) / ((d - i) / (d + 1))
    return total


def tree_shap_one(tree: _TreeArrays, x: np.ndarray, phi: np.ndarray) -> None:
    """Add this tree's SHAP values for sample x into `phi` (in place)."""
    left, right, feat, thr, val, cov = (tree.left, tree.right, tree.feature,
                                        tree.threshold, tree.value, tree.cover)

    def recurse(node, m, pz, po, pi):
        m = [list(e) for e in m]
        _extend(m, pz, po, pi)
        if left[node] == -1:
            for i in range(1, len(m)):
                w = _unwound_sum(m, i)
                phi[m[i][0]] += w * (m[i][2] - m[i][1]) * val[node]
            return
        f = feat[node]
        hot, cold = (left[node], right[node]) if x[f] <= thr[node] else (right[node], left[node])
        iz = io = 1.0
        k = next((j for j in range(1, len(m)) if m[j][0] == f), None)
        if k is not None:
            iz, io = m[k][1], m[k][2]
            _unwind(m, k)
        recurse(hot, m, iz * cov[hot] / cov[node], io, f)
        recurse(cold, m, iz * cov[cold] / cov[node], 0.0, f)

    recurse(0, [], 1.0, 1.0, -1)


def _explain_rows(trees, X, n_features):
    out = np.zeros((X.shape[0], n_features))
    for r in range(X.shape[0]):
        for t in trees:
            tree_shap_one(t, X[r], out[r])
    return out


class ShapBackend:
    """SHAP values of one class's raw margin for a fitted GradientBoostingClassifier."""

    def __init__(self, gb, backend: str = "auto", n_jobs: int = 2):
        self.gb = gb
        self.n_jobs = n_jobs
        self.n_features = gb.n_features_in_
        self.backend = backend
        self._shap = None
        if backend in ("auto", "shap"):
            try:
                import shap  # noqa: F401
                self._shap = shap.TreeExplainer(gb)
                self.backend = "shap"
            except Exception:
                if backend == "shap":
                    raise
                self.backend = "builtin"
        else:
            self.backend = "builtin"
        self._trees = {}
        self._base = {}
        if self.backend == "builtin":
            lr = gb.learning_rate
            for k in range(gb.estimators_.shape[1]):
                self._trees[k] = [_TreeArrays(e.tree_, lr) for e in gb.estimators_[:, k]]

    def base_value(self, k: int) -> float:
        """Expected raw margin of class k over the training distribution."""
        if self.backend == "shap":
            ev = np.atleast_1d(self._shap.expected_value)
            return float(ev[k])
        if k not in self._base:
            # initial raw score = decision_function(x0) - sum of tree outputs at x0
            x0 = np.zeros((1, self.n_features))
            df0 = np.atleast_2d(self.gb.decision_function(x0))[0, k]
            tree_part = self.gb.learning_rate * sum(
                float(e.predict(x0)[0]) for e in self.gb.estimators_[:, k])
            init = float(df0 - tree_part)
            self._base[k] = init + sum(t.expected_value() for t in self._trees[k])
        return self._base[k]

    def shap_values(self, X: np.ndarray, k: int) -> np.ndarray:
        """(n, n_features) attributions of the class-k margin."""
        X = np.asarray(X, dtype=float)
        if self.backend == "shap":
            sv = self._shap.shap_values(X)
            if isinstance(sv, list):
                return np.asarray(sv[k])
            sv = np.asarray(sv)
            return sv[:, :, k] if sv.ndim == 3 else sv
        if len(X) == 0:
            return np.zeros((0, self.n_features))
        trees = self._trees[k]
        if self.n_jobs == 1 or len(X) < 40:
            return _explain_rows(trees, X, self.n_features)
        chunks = np.array_split(np.arange(len(X)), self.n_jobs * 2)
        res = Parallel(n_jobs=self.n_jobs)(
            delayed(_explain_rows)(trees, X[c], self.n_features) for c in chunks if len(c))
        return np.vstack(res)
