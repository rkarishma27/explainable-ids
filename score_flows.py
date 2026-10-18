"""Score a CSV of flows with the trained model and print an explained alert for each detection.

    python score_flows.py flows.csv [--model outputs/model.joblib] [--out alerts_out.csv] [--max-print 20]

The CSV needs either this project's 25 feature columns (see ids/data.py: FEATURES)
or the raw CICIDS2017 / CICFlowMeter column names. Rows judged BENIGN are skipped.

If the saved model was written by different scikit-learn / pandas / numpy versions
than the ones installed here, it is NOT unpickled (cross-version pickles can fail or
misbehave silently). Instead a fresh model is trained on synthetic flows (~1-2 min)
and cached next to it as <model>.local.joblib.
"""
import argparse
import warnings

import numpy as np
import pandas as pd

from ids.data import _CIC_COLS
from ids.explain import alert_card_markdown, build_alert_table
from ids.models import load_bundle, save_bundle, train_default_model


def get_model(path):
    bundle, why = load_bundle(path)
    if bundle is not None:
        return bundle
    local = path + ".local.joblib"
    bundle, why_local = load_bundle(local)
    if bundle is not None:
        print(f"[using locally trained model {local}]")
        return bundle
    print(f"[saved model not usable here: {why}]")
    print("[training a fresh model on synthetic flows, ~1-2 minutes...]")
    bundle = train_default_model()
    save_bundle(bundle["model"], bundle["features"], local)
    print(f"[cached as {local}]")
    return bundle


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--model", default="outputs/model.joblib")
    ap.add_argument("--out", default=None)
    ap.add_argument("--max-print", type=int, default=20)
    a = ap.parse_args()

    warnings.filterwarnings("ignore", category=FutureWarning)
    bundle = get_model(a.model)
    model, feats = bundle["model"], bundle["features"]
    df = pd.read_csv(a.csv, encoding="latin-1", skipinitialspace=True)
    df.columns = [c.strip() for c in df.columns]
    df = df.rename(columns=_CIC_COLS)
    miss = [f for f in feats if f not in df.columns]
    if miss:
        raise SystemExit(f"missing feature columns: {miss}")
    X = df[feats].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    meta = pd.DataFrame({"flow_id": np.arange(len(X)),
                         "src_ip": df.get("Source IP", df.get("src_ip", "n/a")),
                         "dst_ip": df.get("Destination IP", df.get("dst_ip", "n/a"))})
    alerts, _, _ = build_alert_table(model, X, meta)
    print(f"{len(alerts)} alerts out of {len(X)} flows")
    for _, row in alerts.head(a.max_print).iterrows():
        print(alert_card_markdown(row))
    if a.out:
        alerts.to_csv(a.out, index=False, encoding="utf-8")
        print("wrote", a.out)


if __name__ == "__main__":
    main()
