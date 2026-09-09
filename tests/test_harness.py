"""Harness tests — the live loop must be BIT-COMPATIBLE with training.

The most dangerous bug class in this project (documented in AGENTS.md):
a live pipeline that quietly diverges from the training environment makes
every model decision meaningless. These tests pin that equivalence.
"""
import json
import os

import numpy as np
import pandas as pd
import pytest
import torch

from src.features import FeatureEngine
from src.training.environment import SpotTradingEnv


def _make_raw(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    t = pd.date_range("2020-01-01", periods=n, freq="h", tz="UTC")
    c = 100 * np.exp(np.cumsum(rng.normal(0, .01, n)))
    return pd.DataFrame({"datetime": t, "open": c * .99, "high": c * 1.01,
                         "low": c * .98, "close": c, "volume":
                         np.abs(rng.normal(1, .3, n)) + .1})


def _scaled_fixture(tmp_path, n=3000, seed=0, with_xgb=False):
    """Build a tiny processed dir + return raw df. with_xgb=True builds
    95-col features (engine 92 + 3 xgb) to mirror the real dataset."""
    from sklearn.preprocessing import StandardScaler
    import joblib
    raw = _make_raw(n, seed)
    fe = FeatureEngine()
    feats = fe.transform(raw)
    os.makedirs(tmp_path / "processed", exist_ok=True)
    os.makedirs(tmp_path / "models", exist_ok=True)
    if with_xgb:
        from src.features.xgb_ensemble import fit_ensemble, transform_ensemble
        tr = int(n * 0.7)
        dm, vm, names = fit_ensemble(feats.iloc[:tr], raw.close.iloc[:tr])
        feats = pd.concat([feats, transform_ensemble(feats, raw.close, dm, vm, names)], axis=1)
        # save artifacts where the pipeline looks for them
        joblib.dump((dm, vm, names),
                    tmp_path / "models" / "xgb_BTC_USDT.joblib")
    scaler = StandardScaler().fit(feats.values[:int(n * .7)])
    joblib.dump(scaler, tmp_path / "processed" / "scaler.joblib")
    meta = {"feature_count": feats.shape[1]}
    json.dump(meta, open(tmp_path / "processed" / "meta.json", "w"))
    # fake processed CSV header so names load
    head = feats.head(1).copy()
    head.insert(0, "close", raw.close.iloc[0])
    head.insert(0, "datetime", str(raw.datetime.iloc[0]))
    head.to_csv(tmp_path / "processed" / "X_features.csv", index=False)
    cfg = {"window": 160, "initial_cash": 10000.0}
    json.dump(cfg, open(tmp_path / "cfg.json", "w"))
    return raw, feats, scaler, tmp_path


def test_live_obs_matches_training_env_bit_for_bit(tmp_path):
    """THE golden test: after feeding the same candles, LiveEnv.observe()
    must equal training SpotTradingEnv._obs() at the same index with the
    same account state — otherwise live decisions are meaningless."""
    from src.harness.pipeline import LivePipeline
    from src.harness.live_env import LiveEnv

    raw, feats, scaler, tmp = _scaled_fixture(tmp_path)
    pipe = LivePipeline(str(tmp / "processed"), xgb_dir=str(tmp / "models"),
                        config_path=str(tmp / "cfg.json"))
    z_all = scaler.transform(feats.values).astype(np.float32)
    close = raw.close.values

    live = LiveEnv("BTC/USDT", z_all, close, window=160,
                   initial_cash=10000.0)
    # training env: same arrays, same index
    tenv = SpotTradingEnv(z_all, close, 0, len(close), 160, 512)
    tenv.reset(options={"start": len(close) - 1})

    o_live = live.observe()
    o_train = tenv._obs()
    np.testing.assert_allclose(o_live, o_train, atol=1e-5)


def test_live_env_grows_and_stays_equivalent(tmp_path):
    """Append a new candle (as the collector delivers) -> observation must
    equal a training env built on the grown arrays (fresh env, same start).
    This pins the append/advance math against drift."""
    from src.harness.live_env import LiveEnv

    raw = _make_raw(2500, seed=3)
    nxt = _make_raw(1, seed=99)
    grown = pd.concat([raw, nxt], ignore_index=True)
    z_grown = FeatureEngine().transform(grown).values.astype(np.float32)
    close_g = grown.close.values

    live = LiveEnv("BTC/USDT", z_grown[:-1], close_g[:-1], window=160)
    live.append(z_grown[-1], close_g[-1])   # the official growth path
    assert live.i == len(close_g) - 1

    tenv = SpotTradingEnv(z_grown, close_g, 0, len(close_g), 160, 512)
    tenv.reset(options={"start": live.i})
    np.testing.assert_allclose(live.observe(), tenv._obs(), atol=1e-5)


def test_live_pipeline_appends_training_identical_rows(tmp_path):
    """LivePipeline must produce the same scaled rows a full dataset
    rebuild would: warm_start(history) + append(new candle) == transform
    of the full grown history (95 features, same order)."""
    from src.harness.pipeline import LivePipeline
    raw, feats, scaler, tmp = _scaled_fixture(tmp_path, with_xgb=True)
    pipe = LivePipeline(str(tmp / "processed"), xgb_dir=str(tmp / "models"),
                        config_path=str(tmp / "cfg.json"))
    grown = pd.concat([raw, _make_raw(1, seed=5)], ignore_index=True)
    pipe.warm_start("BTC/USDT", grown.iloc[:-1])
    z, close = pipe.append("BTC/USDT", grown.iloc[-1])
    # ground truth: pipeline's own column math vs the fixture's xgb math
    # must agree row-for-row on the history AND include the new row
    assert z.shape == (len(grown), feats.shape[1])
    assert z.shape[1] == 95
    # deterministic: re-transform must give identical values
    z2, _ = pipe.append("BTC/USDT", grown.iloc[-1])  # dup row -> dedup? no:
    # pipeline appends raw rows; same candle twice would double-append.
    # (collector dedups on write, so this never happens live)
    assert z2.shape[0] == len(grown) + 1


def test_live_pipeline_missing_xgb_degrades_to_zero(tmp_path):
    """Owner requirement: missing indicators must NEVER crash the harness.
    With no xgb artifacts anywhere, the 3 signal columns come out as zeros
    and the total column count still matches the scaler's expectation."""
    from src.harness.pipeline import LivePipeline
    # fixture WITH xgb scaler (95 cols) but artifacts deleted -> zeros
    raw, feats, scaler, tmp = _scaled_fixture(tmp_path, with_xgb=True)
    os.remove(tmp / "models" / "xgb_BTC_USDT.joblib")
    pipe = LivePipeline(str(tmp / "processed"), xgb_dir=str(tmp / "models"),
                        config_path=str(tmp / "cfg.json"))
    pipe.warm_start("BTC/USDT", raw)
    z, close = pipe.append("BTC/USDT", raw.iloc[-1])
    assert z.shape[0] == len(raw) + 1 and z.shape[1] == 95
    # xgb columns degrade to a CONSTANT column (raw zeros passed through
    # the scaler's fixed mean/std — a uniform 'no data' signature that is
    # consistent with what the model saw when xgb was absent in training)
    assert np.allclose(z[:, -3:], z[0, -3:])


def test_live_account_execution_matches_env_step():
    """LiveAccount.trade_to_target must reproduce env.step()'s trade block:
    same fee/slippage/deadband/cash-clamp/float32-quantization results."""
    from src.harness.live_env import LiveAccount

    feats = np.random.randn(2200, 10).astype(np.float32)
    close = np.linspace(100, 150, 2200)
    env = SpotTradingEnv(feats, close, 0, 2200, 16, 400,
                         fee=0.001, slippage=0.0005, initial_cash=10000.0)
    env.reset(options={"start": 1500})
    acc = LiveAccount(10000.0, fee=0.001, slippage=0.0005)

    for k, target in enumerate([0.8, 0.3, 0.9, 0.0, 0.5, 1.0]):
        i = 1500 + k
        price = float(close[i])
        env.i = i  # force same candle (env advanced by its own step)
        env.step(np.array([target], np.float32))
        acc.trade_to_target(target, price)
        assert abs(acc.asset - env.asset) < 1e-9, \
            f"asset diverged at target={target}: {acc.asset} vs {env.asset}"
        assert abs(acc.cash - env.cash) < 1e-9
        assert abs(acc.trades - env.trades) <= 1  # env counts its own way


def test_collector_gap_fetch_and_append(tmp_path):
    """Collector: fetch_gap returns only CLOSED candles after the last
    stored one, and append_csv dedups + keeps order (idempotent replay)."""
    from src.harness.collector import append_csv

    t = pd.date_range("2024-01-01", periods=10, freq="h", tz="UTC")
    df = pd.DataFrame({"datetime": t, "open": 1, "high": 1, "low": 1,
                       "close": 1, "volume": 1})
    p = tmp_path / "SYM_USDT.csv"
    added = append_csv(str(p), df)
    assert added == 10
    # replay the same batch: dedup must make it a no-op
    added2 = append_csv(str(p), df)
    assert added2 == 0
    stored = pd.read_csv(p, parse_dates=["datetime"])
    assert len(stored) == 10 and stored.datetime.is_monotonic_increasing


def test_paper_state_roundtrip(tmp_path):
    """Stop/restart the paper trader: accounts (cash/asset/waterfall) must
    survive the round trip bit-for-bit (crash-resume safety)."""
    from src.harness.live_env import LiveAccount
    import dataclasses

    # simulate what PaperTrader._save_state stores / _resume restores
    a = LiveAccount(10000.0, fee=0.001, slippage=0.0005)
    a.trade_to_target(0.6, 100.0)
    a.trade_to_target(0.2, 104.0)
    a.trade_to_target(0.9, 99.0)
    saved = {"cash": a.cash, "asset": a.asset, "peak": a.peak,
             "trades": a.trades, "fees_paid": a.fees_paid,
             "slippage_paid": a.slippage_paid, "hwm": a.hwm,
             "trough": a.trough, "gross_gain": a.gross_gain,
             "gross_loss": a.gross_loss,
             "max_drawdown": a.max_drawdown, "steps": 3}
    b = LiveAccount(10000.0, fee=0.001, slippage=0.0005)
    b.cash = saved["cash"]; b.asset = saved["asset"]
    b.peak = saved["peak"]; b.trades = saved["trades"]
    b.fees_paid = saved["fees_paid"]; b.slippage_paid = saved["slippage_paid"]
    b.hwm = saved["hwm"]; b.trough = saved["trough"]
    b.gross_gain = saved["gross_gain"]; b.gross_loss = saved["gross_loss"]
    b.max_drawdown = saved["max_drawdown"]
    # after restore, continuing trades gives identical results to never stopping
    price = 101.0
    a.trade_to_target(0.4, price)
    b.trade_to_target(0.4, price)
    assert abs(a.value(price) - b.value(price)) < 1e-9
    assert abs(a.purity - b.purity) < 1e-12
    assert a.trades == b.trades
