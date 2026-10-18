"""Check the official `shap.TreeExplainer` against the bundled TreeSHAP.

    pip install shap
    python verify_shap_backend.py

The reported results in this project use the bundled TreeSHAP (verified against
brute-force Shapley values in tests/). The official package should give the same
numbers for scikit-learn gradient boosting; this script tells you whether it does on
YOUR installed versions before you switch with `run_all.py --backend shap`.
"""
import sys

import numpy as np

from ids.data import generate_flows
from ids.models import ExplainableIDS
from ids.treeshap import ShapBackend


def compare(a: ShapBackend, b: ShapBackend, X, n_classes, tol=1e-3):
    ok = True
    for k in range(n_classes):
        pa, pb = a.shap_values(X, k), b.shap_values(X, k)
        d = float(np.abs(pa - pb).max())
        db = abs(a.base_value(k) - b.base_value(k))
        flag = "ok" if (d < tol and db < tol) else "MISMATCH"
        ok &= flag == "ok"
        print(f"  class {k}: max|phi diff| = {d:.2e}   base-value diff = {db:.2e}   {flag}")
    return ok


def main():
    try:
        import shap
    except ImportError:
        raise SystemExit("shap is not installed (pip install shap); nothing to verify.")
    print("shap", shap.__version__)
    X, y, _ = generate_flows(8000, seed=0)
    model = ExplainableIDS(n_estimators=40, backend="builtin", n_jobs=1).fit(X, y)
    try:
        official = ShapBackend(model.gb_, backend="shap")
        res = compare(model.shap_, official, X.values[:100], len(model.classes_))
    except Exception as e:  # noqa: BLE001
        print(f"official shap could not explain this model: {type(e).__name__}: {str(e)[:300]}")
        print("=> keep the default --backend builtin")
        sys.exit(1)
    if res:
        print("PASS: official shap agrees with the bundled TreeSHAP; --backend shap is safe to use.")
    else:
        print("FAIL: values differ. Keep the default --backend builtin (the verified one).")
        sys.exit(1)


if __name__ == "__main__":
    main()
