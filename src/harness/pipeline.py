"""Live feature pipeline — transforms a raw candle feed into EXACTLY the
model's training diet (95 columns, scaler-fitted on train only).

Design contract (sacred rules #10/#11 respected):
  - the FeatureEngine part is deterministic past-only math: every new candle
    appends one row; historical rows never change (verified by test).
  - the 3 XGBoost columns need models fit on TRAIN data only; at runtime we
    reuse the artifact produced by dataset.build (models/xgb_{SYMBOL}.joblib),
    and — mirroring dataset.build's own fallback for unseen assets — if an
    artifact is missing for a symbol we borrow the first available one
    (shared market grammar) or emit zeros, NEVER crash: a missing indicator
    must degrade gracefully (owner requirement, 2026-09-09 session).
  - scaler comes from data/processed/scaler.joblib (fit on train segments).
"""
import os
import joblib
import numpy as np
import pandas as pd

from ..features.engine import FeatureEngine
from ..features.xgb_ensemble import transform_ensemble


class LivePipeline:
    def __init__(self, processed_dir="data/processed", xgb_dir=None,
                 config_path="config/default.json"):
        import json
        cfg = json.load(open(config_path))
        self.window = cfg.get("window", 160)
        self.scaler = joblib.load(os.path.join(processed_dir, "scaler.joblib"))
        meta = json.load(open(os.path.join(processed_dir, "meta.json")))
        self.feature_names = self._load_feature_names(processed_dir)
        assert len(self.feature_names) == meta["feature_count"], \
            f"processed features {len(self.feature_names)} != meta {meta['feature_count']}"
        self.engine = FeatureEngine()
        # xgb artifacts live next to the scaler (dataset.build saves them
        # there); xgb_dir is a legacy/override location
        self.xgb_dirs = []
        if xgb_dir:
            self.xgb_dirs.append(xgb_dir)
        self.xgb_dirs.append(processed_dir)
        self._xgb_cache = {}          # symbol -> (dir_model, vol_model, names) or None
        self._borrowed = {}           # symbol -> symbol actually used
        self._feat_cache = {}         # symbol -> {"df": raw, "feats": features}
        self._scaled_cache = {}       # symbol -> {"arr": ndarray, "close": ndarray}
        self._cols = None             # exact final column order

    # ---------- feature names (from any processed CSV header) ----------
    @staticmethod
    def _load_feature_names(processed_dir):
        import glob
        files = sorted(glob.glob(os.path.join(processed_dir, "*_features.csv")))
        if not files:
            raise RuntimeError("no processed feature CSVs — run dataset build first")
        head = pd.read_csv(files[0], nrows=1)
        return [c for c in head.columns if c not in ("datetime", "close")]

    # ---------- XGB artifacts (graceful, never crash) ----------
    def _load_xgb(self, symbol):
        if symbol in self._xgb_cache:
            return self._xgb_cache[symbol]
        art = None
        fname = f"xgb_{symbol.replace('/', '_')}.joblib"
        for d in self.xgb_dirs:
            path = os.path.join(d, fname)
            if os.path.exists(path):
                try:
                    cand = joblib.load(path)
                    if len(cand) == 3:
                        art = cand
                        break
                except Exception:
                    continue
        if art is None:
            # borrow the first existing artifact (same policy as dataset.build
            # for unseen assets) — or zeros if none exist at all
            for d in self.xgb_dirs:
                if not os.path.isdir(d):
                    continue
                for other in sorted(os.listdir(d)):
                    if other.startswith("xgb_") and other.endswith(".joblib"):
                        try:
                            cand = joblib.load(os.path.join(d, other))
                            if len(cand) == 3:
                                art = cand
                                self._borrowed[symbol] = other[4:-8].replace("_", "/")
                                break
                        except Exception:
                            continue
                if art is not None:
                    break
        self._xgb_cache[symbol] = art
        return art

    def _append_features(self, symbol, raw_row=None, bulk=False):
        """(Re)compute the features DataFrame for the symbol's full raw
        history. Historical rows are immutable (FeatureEngine math is
        past-only). XGB columns: full set when artifacts exist, zeros when
        they don't (owner: missing indicators must degrade, never crash).
        The FINAL column count is pinned to whatever the scaler expects —
        if training used 95 (with xgb) we always emit 95."""
        cache = self._feat_cache.get(symbol)
        if cache is None:
            cache = {"df": pd.DataFrame()}
            self._feat_cache[symbol] = cache
        if bulk:
            cache["df"] = cache["df"].reset_index(drop=True)
        if raw_row is not None:
            cache["df"] = pd.concat([cache["df"], pd.DataFrame([raw_row])],
                                    ignore_index=True)
        raw = cache["df"]
        if not len(raw):
            cache["feats"] = pd.DataFrame(columns=self._cols or [])
            cache["raw_rows"] = 0
            return cache
        feats = self.engine.transform(raw)
        # --- XGB columns (graceful degradation) ---
        art = self._load_xgb(symbol)
        if art is None:
            zero = pd.DataFrame(0.0, index=feats.index,
                                columns=["xgb_dir_24", "xgb_edge", "xgb_vol"],
                                dtype=np.float32)
            feats = pd.concat([feats, zero], axis=1)
        else:
            dm, vm, names = art
            sig = transform_ensemble(feats, raw.close, dm, vm, names)
            feats = pd.concat([feats, sig], axis=1)
        # --- pin exact column layout the scaler was fit on ---
        n_expected = getattr(self.scaler, "n_features_in_", None)
        if n_expected is not None and feats.shape[1] != n_expected:
            if self._cols is not None:
                feats = feats[self._cols]
            if feats.shape[1] != n_expected:
                raise RuntimeError(
                    f"feature mismatch: pipeline produces {feats.shape[1]} "
                    f"but scaler expects {n_expected} — rebuild dataset or "
                    "provide matching xgb artifacts")
        if self._cols is None:
            self._cols = list(feats.columns)
        cache["feats"] = feats
        cache["raw_rows"] = len(raw)
        return cache

    def append(self, symbol, raw_row):
        """Feed one closed candle. Returns (scaled array [rows, F],
        close prices array) — the ARRAYS the LiveEnv grows on.
        Efficient: only NEW rows are scaled (scaler transform is
        row-independent: z = (x - mean) / std)."""
        symbol = symbol.upper()
        cache = self._append_features(symbol, raw_row)
        raw = cache["df"]
        feats = cache["feats"]
        sc = self._scaled_cache.setdefault(symbol, {})
        prev = sc.get("arr")
        n_new = feats.shape[0] - (prev.shape[0] if prev is not None else 0)
        if n_new <= 0:
            return sc["arr"], sc["close"]
        new_block = feats.values[-n_new:]
        z_new = self.scaler.transform(new_block)
        sc["arr"] = z_new if prev is None else np.vstack([prev, z_new])
        sc["close"] = raw.close.values
        return sc["arr"], sc["close"]

    def warm_start(self, symbol, raw_df):
        """Bulk-load existing history (e.g. the collector CSV) in one shot
        (appending row-by-row would re-transform O(n^2))."""
        symbol = symbol.upper()
        self._feat_cache[symbol] = {"df": raw_df.reset_index(drop=True)}
        cache = self._append_features(symbol, bulk=True)
        raw = cache["df"]
        z = self.scaler.transform(cache["feats"].values)
        sc = self._scaled_cache.setdefault(symbol, {})
        sc["arr"] = z
        sc["close"] = raw.close.values
        return sc["arr"], sc["close"]

    def ready(self, symbol):
        """Do we have enough history for a full observation? (fine window +
        weekly frame reach = 168*8 candles, same margin as dataset.sample)"""
        sc = self._scaled_cache.get(symbol.upper())
        return sc is not None and sc["arr"].shape[0] >= 1344
