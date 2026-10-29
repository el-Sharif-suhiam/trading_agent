"""P4 TRUTH PROBE — do school-vote channels add generalizable signal?

Same protocol as W5 signal_probe (train-fit, forward-only validation AUC,
next-24h direction) but three feature sets:
  features   : the 92 processed columns (W5 baseline replication)
  votes      : the 8 school votes alone
  combined   : 92 + 8 — the actual P4 observation if we build it

Gate (pre-agreed with the owner): build + distill ONLY if `combined`
improves mean OOS AUC over the features-only baseline in a consistent way.

Usage: python -m src.evaluation.school_probe
"""
import json
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from ..features.schools import NAMES, school_votes

HORIZON = 24


def load(cfg):
    """Per training asset: processed features + school votes aligned by
    datetime (votes computed from RAW OHLCV — the causal source)."""
    out = []
    man = pd.read_csv(os.path.join(cfg["processed_dir"], "train_manifest.csv"))
    man_v = pd.read_csv(os.path.join(cfg["processed_dir"], "validation_manifest.csv"))
    man_all = pd.concat([man.assign(split="train"), man_v.assign(split="validation")])
    for _, r in man_all.iterrows():
        pdf = pd.read_csv(os.path.join(cfg["processed_dir"], r.file),
                          parse_dates=["datetime"])
        raw_name = r.file.replace("_features.csv", ".csv")
        raw = pd.read_csv(os.path.join(cfg["data_dir"], raw_name),
                          parse_dates=["datetime"])
        raw = raw.set_index("datetime")
        feats = pdf.drop(columns=["datetime", "close"], errors="ignore").values.astype(np.float32)
        cols = [c for c in pdf.columns if c not in ("datetime", "close")]
        close = pdf.close.values.astype(np.float64)
        o = raw.open.reindex(pdf.datetime).values.astype(np.float64)
        h = raw.high.reindex(pdf.datetime).values.astype(np.float64)
        l = raw.low.reindex(pdf.datetime).values.astype(np.float64)
        v = raw.volume.reindex(pdf.datetime).values.astype(np.float64)
        votes = school_votes(o, h, l, close, v)
        out.append({"symbol": r.symbol, "split": r.split, "cols": cols,
                    "features": feats, "votes": votes, "close": close,
                    "start": int(r.start_idx), "end": int(r.end_idx)})
    return out


def build_xy(assets, horizon=HORIZON):
    per = {}
    for a in assets:
        s, e = a["start"], a["end"]
        n = e - s - horizon
        if n < 300:
            continue
        Xf = a["features"][s:s + n]
        Xv = a["votes"][s:s + n]
        y = (a["close"][s + horizon:e] / a["close"][s:s + n] - 1.0 > 0).astype(int)
        d = per.setdefault(a["symbol"], {"cols": a["cols"]})
        d[a["split"]] = (Xf, Xv, y)
    return per


def main():
    cfg = json.load(open("config/default.json"))
    assets = load(cfg)
    per = build_xy(assets)
    sets = {
        "features": lambda Xf, Xv: Xf,
        "votes": lambda Xf, Xv: Xv,
        "combined": lambda Xf, Xv: np.hstack([Xf, Xv]),
    }
    print(f"assets {len(per)}  train rows {sum(len(v['train'][2]) for v in per.values()):,}")
    results = {k: [] for k in sets}
    vote_auc = []  # diagnostic: raw MTF vote's own AUC (no fit)
    for sym, d in sorted(per.items()):
        Xf_tr, Xv_tr, ytr = d["train"]
        Xf_o, Xv_o, yo = d["validation"]
        row = [f"{sym:12s}"]
        for name, sel in sets.items():
            Xtr, Xo = sel(Xf_tr, Xv_tr), sel(Xf_o, Xv_o)
            sc = StandardScaler().fit(Xtr)
            lr = LogisticRegression(max_iter=2000, C=0.1)
            lr.fit(sc.transform(Xtr), ytr)
            auc = roc_auc_score(yo, lr.predict_proba(sc.transform(Xo))[:, 1])
            results[name].append(auc)
            row.append(f"{auc:.3f}")
        vote_auc.append(roc_auc_score(yo, Xv_o[:, 4]))   # MTF vote raw AUC
        print("  ".join(row) + f"   mtf_raw={vote_auc[-1]:.3f}")
    print("\n=== OOS aggregate (validation blocks, forward-only) ===")
    for name, aucs in results.items():
        aucs = np.array(aucs)
        print(f"  {name:9s} mean {aucs.mean():.4f}  median {np.median(aucs):.4f}  "
              f">0.52: {100 * (aucs > 0.52).mean():.0f}%  >0.50: {100 * (aucs > 0.5).mean():.0f}%")
    base, comb = np.array(results["features"]), np.array(results["combined"])
    wins = int((comb > base).sum())
    print(f"\nGATE: combined vs features baseline — wins {wins}/{len(base)}, "
          f"mean delta {(comb - base).mean():+.4f}")
    print("BUILD if mean delta >= +0.005 AND wins > half; otherwise report and stop.")


if __name__ == "__main__":
    main()
