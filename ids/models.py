"""The explainable detector (gradient boosting + TreeSHAP) and the black-box baseline (MLP)."""
from __future__ import annotations

import json
import os

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler

from .treeshap import ShapBackend

COMMON_PORTS = [21, 22, 23, 25, 53, 80, 123, 443, 445, 3306, 3389, 8080]


class ExplainableIDS:
    """Gradient-boosted trees; every alert can be decomposed with TreeSHAP.

    `explain(X)` returns, for each flow, the SHAP attribution of the *predicted
    class's* log-odds score: positive values pushed the model toward its verdict.
    """

    def __init__(self, n_estimators=120, max_depth=4, learning_rate=0.1,
                 subsample=0.8, min_samples_leaf=20, max_abs_leaf=100.0,
                 random_state=0, backend="builtin", n_jobs=2):
        # min_samples_leaf matters: with single-sample leaves, scikit-learn's
        # multiclass Newton step divides by ~0 and produces leaf values around
        # 1e7, which saturates probabilities and wrecks log-odds attributions.
        self.params = dict(n_estimators=n_estimators, max_depth=max_depth,
                           learning_rate=learning_rate, subsample=subsample,
                           min_samples_leaf=min_samples_leaf,
                           random_state=random_state)
        self.max_abs_leaf = max_abs_leaf
        self.backend_name = backend
        self.n_jobs = n_jobs

    def _clip_pathological_leaves(self):
        """scikit-learn's multiclass Newton leaf update divides by sum p(1-p),
        which is ~0 when every sample in a leaf is already saturated for a class,
        giving leaf values up to ~1e20 even with min_samples_leaf > 1. Healthy
        leaves (before the 0.1 learning-rate shrinkage) reach about +-60 here,
        so the cap of 100 only touches the pathological ones. Winsorise anything beyond `max_abs_leaf`
        (in place, so predict/predict_proba/SHAP all see the same model)."""
        clipped = 0
        for e in self.gb_.estimators_.ravel():
            t = e.tree_
            v = t.value
            bad = (np.abs(v[:, 0, 0]) > self.max_abs_leaf) & (t.children_left == -1)
            if bad.any():
                clipped += int(bad.sum())
                np.clip(v, -self.max_abs_leaf, self.max_abs_leaf, out=v)
        self.n_leaves_clipped_ = clipped

    def fit(self, X: pd.DataFrame, y):
        self.features_ = list(X.columns)
        self.gb_ = GradientBoostingClassifier(**self.params).fit(X.values, np.asarray(y))
        self._clip_pathological_leaves()
        self.classes_ = list(self.gb_.classes_)
        self.shap_ = ShapBackend(self.gb_, self.backend_name, self.n_jobs)
        med = X[np.asarray(y) == "BENIGN"].median()  # "typical benign flow"
        # object-dtype index so the pickle loads across pandas 2.x / 3.x
        self.background_ = pd.Series(med.values, index=pd.Index(list(med.index), dtype=object))
        return self

    def predict_proba(self, X):
        return self.gb_.predict_proba(_v(X))

    def predict(self, X):
        return self.gb_.predict(_v(X))

    def explain(self, X: pd.DataFrame, class_idx=None):
        """Returns (phi[n, F], base[n], class_idx[n]) for the predicted class
        (or `class_idx`, an int array, if given)."""
        Xv = _v(X)
        if class_idx is None:
            class_idx = self.gb_.predict_proba(Xv).argmax(axis=1)
        class_idx = np.asarray(class_idx)
        phi = np.zeros(Xv.shape)
        base = np.zeros(len(Xv))
        for k in np.unique(class_idx):
            m = class_idx == k
            phi[m] = self.shap_.shap_values(Xv[m], int(k))
            base[m] = self.shap_.base_value(int(k))
        return phi, base, class_idx

    @property
    def shap_backend(self):
        return self.shap_.backend


class BlackBoxBaseline:
    """Opaque neural-network detector. Same inputs, same train data; gives a
    label and a score and nothing else. Heavy-tailed features are log-scaled
    and common ports one-hot encoded so it is not handicapped on preprocessing."""

    def __init__(self, hidden=(128, 64), random_state=0, max_iter=200):
        self.hidden, self.random_state, self.max_iter = hidden, random_state, max_iter

    def _prep(self, X: pd.DataFrame):
        A = X.copy()
        ports = A["dst_port"].values
        onehot = np.stack([(ports == p).astype(float) for p in COMMON_PORTS], axis=1)
        A = A.drop(columns=["dst_port"])
        Z = np.log1p(np.clip(A.values.astype(float), 0, None))
        return np.hstack([Z, onehot, (ports > 1023).astype(float)[:, None]])

    def fit(self, X, y):
        Z = self._prep(X)
        self.scaler_ = StandardScaler().fit(Z)
        self.mlp_ = MLPClassifier(hidden_layer_sizes=self.hidden, early_stopping=True,
                                  max_iter=self.max_iter, random_state=self.random_state)
        # integer-encode labels ourselves: scikit-learn 1.8's MLP early-stopping
        # path raises TypeError (np.isnan on string labels) otherwise
        self.le_ = LabelEncoder().fit(np.asarray(y).astype(str))
        self.mlp_.fit(self.scaler_.transform(Z), self.le_.transform(np.asarray(y).astype(str)))
        self.classes_ = list(self.le_.classes_)
        return self

    def predict_proba(self, X):
        return self.mlp_.predict_proba(self.scaler_.transform(self._prep(X)))

    def predict(self, X):
        return self.le_.inverse_transform(self.mlp_.predict(self.scaler_.transform(self._prep(X))))


def _v(X):
    return X.values if hasattr(X, "values") else np.asarray(X)


# --------------------------------------------------------------------------
# saving / loading: pickles are only trusted on the library versions that wrote them
# --------------------------------------------------------------------------
def _versions():
    return {"sklearn": sklearn.__version__, "pandas": pd.__version__, "numpy": np.__version__}


def _mm(v: str) -> str:
    return ".".join(str(v).split(".")[:2])


def save_bundle(model, features, path):
    joblib.dump({"model": model, "features": list(features)}, path, compress=3)
    with open(path + ".versions.json", "w", encoding="utf-8") as fh:
        json.dump(_versions(), fh)


def load_bundle(path):
    """Returns (bundle, "") or (None, reason). Refuses to unpickle if the file was
    written with a different scikit-learn/pandas/numpy major.minor, because
    cross-version pickles can fail or, worse, load and misbehave silently."""
    meta = path + ".versions.json"
    if not (os.path.exists(path) and os.path.exists(meta)):
        return None, "model file or version file missing"
    with open(meta, encoding="utf-8") as fh:
        saved = json.load(fh)
    cur = _versions()
    for k, v in cur.items():
        if _mm(saved.get(k, "")) != _mm(v):
            return None, f"{k} {saved.get(k, '?')} (saved) vs {v} (installed)"
    try:
        return joblib.load(path), ""
    except Exception as e:  # noqa: BLE001
        return None, f"unpickling failed: {type(e).__name__}"


def train_default_model(n_flows=20000, seed=0, backend="builtin"):
    """Fit the explainable detector on synthetic flows (used when no compatible
    saved model exists). Returns a bundle dict."""
    from .data import generate_flows
    X, y, _ = generate_flows(n_flows, seed)
    model = ExplainableIDS(random_state=seed, backend=backend).fit(X, y)
    return {"model": model, "features": list(X.columns)}
