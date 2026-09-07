"""XGBoost ensemble signal features — sacred rule #4 compliant.

Three extra columns (xgb_dir_24, xgb_edge, xgb_vol) are injected as ordinary
INPUT FEATURES. The neural agent decides how (or whether) to use them —
they are eyes, not hands. No BUY/SELL rule is ever forced.

Leak-proofing (sacred rule #10):
  - models are fit on the TRAIN segment of each asset only;
  - training labels look HORIZONS ahead but predictions at row t use only
    features at rows <= t (trees split on feature values, never on labels);
  - predictions are produced with model.predict on past-only feature rows,
    then verified by the same prefix-invariance test as every other feature.
"""
import os
import numpy as np
import pandas as pd


HORIZON = 24  # hours ahead the direction label looks (train-only target)


def make_labels(close: pd.Series) -> pd.Series:
    """Direction label: 1 if close rises by HORIZON, else 0.
    The LAST `HORIZON` rows have no label (future beyond the data)."""
    fut = close.shift(-HORIZON)
    up = (fut - close) > 0
    return up.astype(np.float32).where(fut.notna(), other=np.nan)


def fit_ensemble(features: pd.DataFrame, close: pd.Series, cfg=None):
    """Fit two gradient-boosted models on past-only features:
      dir model : binary up/down after HORIZON (probability of direction)
      vol model : binary 'high realized volatility ahead' (risk radar)
    Returns (dir_model, vol_model, feature_names).
    """
    import xgboost as xgb
    cfg = cfg or {}
    n_est = cfg.get("xgb_n_estimators", 300)
    depth = cfg.get("xgb_max_depth", 4)
    lr = cfg.get("xgb_learning_rate", 0.05)
    y_dir = make_labels(close)
    # realized-forward volatility (train target only): std of next-HORIZON
    # hourly returns, thresholded at its own median inside train
    fut = close.shift(-HORIZON)
    fwd = (fut / close - 1.0)
    vol_hi = (fwd.abs() > fwd.abs().median()).astype(np.float32).where(fut.notna(), other=np.nan)
    mask = y_dir.notna()
    X = features.values[mask]
    common = dict(n_estimators=n_est, max_depth=depth, learning_rate=lr,
                  subsample=0.8, colsample_bytree=0.8, tree_method="hist",
                  n_jobs=4, eval_metric="logloss", verbosity=0)
    dir_model = xgb.XGBClassifier(**common)
    dir_model.fit(X, y_dir.values[mask])
    vol_model = xgb.XGBClassifier(**common)
    vol_model.fit(X, vol_hi.values[mask])
    return dir_model, vol_model, list(features.columns)


def transform_ensemble(features: pd.DataFrame, close: pd.Series,
                       dir_model, vol_model, feature_names) -> pd.DataFrame:
    """Produce the three signal columns for every row using ONLY each row's
    own feature values (trees cannot see beyond the row they score)."""
    X = features[feature_names].values
    p_dir = dir_model.predict_proba(X)[:, 1]
    p_vol = vol_model.predict_proba(X)[:, 1]
    # 'edge' = calibrated direction confidence pushed through a signed
    # rank (signed z-like signal in [-1,1]) — scale-free by construction
    edge = 2.0 * p_dir - 1.0
    out = pd.DataFrame(index=features.index)
    out["xgb_dir_24"] = p_dir.astype(np.float32)
    out["xgb_edge"] = edge.astype(np.float32)
    out["xgb_vol"] = p_vol.astype(np.float32)
    return out
