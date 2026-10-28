"""W5 CHEAP TRUTH CHECK — does the feature set carry generalizable signal AT ALL?

A supervised probe on the SAME features, splits and horizon as the RL agent:
predict the sign of the next-24h close return, fit ONLY on the train blocks
of all assets (scaler included), then grade AUC per asset on its validation
block — pure walk-forward, no leakage.

Interpretation gate (owner-approved plan):
  OOS AUC > 0.52 stable across assets  -> features carry real cross-time signal;
                                          RL deserves the recalibration run (W3/W4).
  OOS AUC <= 0.50 everywhere          -> no generalizable signal in OHLCV-only
                                          92 features; RL cannot invent one —
                                          report to owner before burning GPU.

Usage: python -m src.evaluation.signal_probe [--model both|xgb|logistic]
"""
import argparse, json
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


def load(cfg):
    import glob
    assets = []
    names = None
    for tr_manifest, split in [("train_manifest.csv", "train"),
                               ("validation_manifest.csv", "validation")]:
        man = pd.read_csv(os.path.join(cfg["processed_dir"], tr_manifest))
        for _, r in man.iterrows():
            df = pd.read_csv(os.path.join(cfg["processed_dir"], r.file))
            if names is None:
                names = [c for c in df.columns if c not in ("datetime", "close")]
            feats = df.drop(columns=["datetime", "close"], errors="ignore").values.astype(np.float32)
            close = df.close.values.astype(np.float64)
            assets.append({"symbol": r.symbol, "features": feats, "close": close,
                           "start": int(r.start_idx), "end": int(r.end_idx),
                           "split": split})
    return assets, names


def build_xy(assets, horizon=24):
    """Per asset: X=features[start..end-horizon), y=sign of close[t+h]/close[t]-1.
    Fit pool = the whole TRAIN block of every asset; OOS = the whole VALIDATION
    block (forward-only, no row of the fit pool comes from validation time)."""
    per = {}
    for a in assets:
        s, e = a["start"], a["end"]
        n = e - s - horizon
        if n < 300:
            continue
        feats, close = a["features"], a["close"]
        X = feats[s:s + n]
        ret = close[s + horizon:e] / close[s:s + n] - 1.0
        y = (ret > 0).astype(int)
        per.setdefault(a["symbol"], {})[a["split"]] = (X, y)
    return per


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config/default.json")
    p.add_argument("--model", default="both", choices=["both", "xgb", "logistic"])
    a = p.parse_args()
    cfg = json.load(open(a.config))
    assets, names = load(cfg)
    per = build_xy(assets)
    dir_idx = names.index("xgb_dir_24") if names and "xgb_dir_24" in names else None
    Xtr = np.concatenate([v["train"][0] for v in per.values()])
    ytr = np.concatenate([v["train"][1] for v in per.values()])
    print(f"train rows {len(ytr):,}  positive rate {ytr.mean():.3f}  assets {len(per)}")
    scaler = StandardScaler().fit(Xtr)
    Xtr_s = scaler.transform(Xtr)

    models = {}
    if a.model in ("both", "logistic"):
        lr = LogisticRegression(max_iter=2000, C=0.1)  # heavy reg: probe, not fit
        lr.fit(Xtr_s, ytr)
        models["logistic"] =(lambda X: lr.predict_proba(scaler.transform(X))[:, 1])
    if a.model in ("both", "xgb"):
        from xgboost import XGBClassifier
        xgb = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                            subsample=0.8, colsample_bytree=0.8,
                            eval_metric="logloss", n_jobs=8, verbosity=0)
        xgb.fit(Xtr_s, ytr)
        models["xgb"] = lambda X: xgb.predict_proba(scaler.transform(X))[:, 1]

    def _summary_row(fn, sym, d):
        Xo, yo = d["validation"]
        pr = fn(Xo)
        return roc_auc_score(yo, pr), len(yo)

    cols = list(models) + (["xgb_dir_24(raw)"] if dir_idx is not None else [])
    print(f"\n{'asset':12s} " + "  ".join(f"{m:>14s}" for m in cols) + "   n_oos")
    summary = {m: [] for m in cols}
    for sym, d in per.items():
        if "validation" not in d:
            continue
        Xo, yo = d["validation"]
        row = []
        for m, fn in models.items():
            auc, n = _summary_row(fn, sym, d)
            summary[m].append(auc)
            row.append(f"{auc:14.3f}")
        if dir_idx is not None:
            auc_raw = roc_auc_score(yo, Xo[:, dir_idx])
            summary["xgb_dir_24(raw)"].append(auc_raw)
            row.append(f"{auc_raw:14.3f}")
            n = len(yo)
        print(f"{sym:12s} " + "  ".join(row) + f"   {n:,}")
    print("\n=== OOS aggregate (validation blocks, forward-only)")
    for m, aucs in summary.items():
        aucs = np.array(aucs)
        print(f"  {m:16s} mean {aucs.mean():.3f}  median {np.median(aucs):.3f}  "
              f">0.52: {100*(aucs>0.52).mean():.0f}%  >0.50: {100*(aucs>0.5).mean():.0f}%")
    print("\nGATE: mean OOS AUC > 0.52 = real signal present; <= 0.50 = none "
          "(report to owner before any GPU spend).")


if __name__ == "__main__":
    main()
