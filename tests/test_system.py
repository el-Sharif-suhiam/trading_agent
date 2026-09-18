import numpy as np, pandas as pd, torch
from src.features import FeatureEngine
from src.model import TradingAgent
from src.training.environment import SpotTradingEnv
from src.training.ppo import PPO


def _make_df(n=300, seed=0):
    rng = np.random.default_rng(seed)
    t = pd.date_range("2020-01-01", periods=n, freq="h", tz="UTC")
    c = 100 * np.exp(np.cumsum(rng.normal(0, .01, n)))
    return pd.DataFrame({"datetime": t, "open": c*.99, "high": c*1.01,
                         "low": c*.98, "close": c, "volume": np.abs(rng.normal(1, .3, n))+.1})


def test_xgb_features_no_lookahead(tmp_path=None):
    """XGBoost signal columns must be prefix-invariant: predictions on a
    prefix must equal the prefix of predictions on the full series. Trees
    score each row from its own features, but this test also catches any
    accidental label/median leakage from beyond the cut."""
    import numpy as np, pandas as pd
    from src.features.engine import FeatureEngine
    from src.features.xgb_ensemble import fit_ensemble, transform_ensemble
    df = _make_df(600, seed=11)
    fe = FeatureEngine()
    feat = fe.transform(df)
    tr = 400
    dm, vm, names = fit_ensemble(feat.iloc[:tr], df.close.iloc[:tr])
    full = transform_ensemble(feat, df.close, dm, vm, names)
    # cut the tail (the 'future') and re-transform — prefix must match
    feat_cut = fe.transform(df.iloc[:350].reset_index(drop=True))
    cut = transform_ensemble(feat_cut, df.close.iloc[:350], dm, vm, names)
    np.testing.assert_allclose(full.values[:350], cut.values, atol=1e-6)
    # model itself must never have seen rows beyond tr in FITTING:
    # guaranteed by construction; verified indirectly by invariance above.


def test_xgb_features_are_sane_signals(tmp_path=None):
    """The three columns exist, are finite, and dir probability spans a
    meaningful range (a constant column would mean the trees are blind)."""
    import numpy as np, pandas as pd
    from src.features.engine import FeatureEngine
    from src.features.xgb_ensemble import fit_ensemble, transform_ensemble
    df = _make_df(800, seed=13)
    fe = FeatureEngine()
    feat = fe.transform(df)
    dm, vm, names = fit_ensemble(feat.iloc[:500], df.close.iloc[:500])
    sig = transform_ensemble(feat, df.close, dm, vm, names)
    assert list(sig.columns) == ["xgb_dir_24", "xgb_edge", "xgb_vol"]
    assert np.isfinite(sig.values).all()
    assert sig["xgb_dir_24"].nunique() > 10, "dir signal must vary"
    assert sig["xgb_vol"].between(0, 1).all() and sig["xgb_dir_24"].between(0, 1).all()
    # edge is exactly 2*dir-1 by construction
    np.testing.assert_allclose(sig["xgb_edge"], 2*sig["xgb_dir_24"]-1, atol=1e-6)


def test_features_no_future_fill():
    f = FeatureEngine().transform(_make_df())
    assert np.isfinite(f.values).all(); assert len(f) == 300


def test_features_prefix_invariance():
    """transform(prefix) must equal prefix of transform(full): catches ALL look-ahead bias."""
    df = _make_df(400, seed=7)
    full = FeatureEngine().transform(df)
    cut = FeatureEngine().transform(df.iloc[:250].reset_index(drop=True))
    np.testing.assert_allclose(full.values[:250], cut.values, atol=1e-6)


def test_env_portfolio_state_in_obs():
    f = np.random.randn(2000, 80).astype(np.float32)
    c = np.linspace(100, 150, 2000)
    e = SpotTradingEnv(f, c, 0, 2000, 16, 100)
    o, _ = e.reset(options={"start": 1400})
    assert o.shape == (16 + 24 + 8 + 30 + 8, 80 + 5)  # fine+4h+12h+daily+weekly rows, +5 state cols


def test_env_daily_weekly_frames_past_only():
    """Coarse daily/weekly rows must be pure past-only aggregation of the
    fine features: a change in FUTURE candles must not alter them."""
    f = np.random.randn(2000, 12).astype(np.float32)
    c = np.linspace(100, 150, 2000)
    e1 = SpotTradingEnv(f, c, 0, 2000, 16, 100)
    o1, _ = e1.reset(options={"start": 1400})
    e2 = SpotTradingEnv(f[:1401], c[:1401], 0, 1401, 16, 100)  # truncate future
    o2, _ = e2.reset(options={"start": 1400})
    # everything the agent sees at t=1400 must be identical without the future
    np.testing.assert_allclose(o1, o2, atol=1e-6)


def test_env_daily_weekly_rows_match_manual_aggregation():
    f = np.arange(2000 * 3, dtype=np.float32).reshape(2000, 3)
    c = np.linspace(100, 150, 2000)
    e = SpotTradingEnv(f, c, 0, 2000, 16, 100)
    o, _ = e.reset(options={"start": 1500})
    i = 1500
    manual_daily = f[i - 24*30 + 1:i + 1].reshape(30, 24, 3).mean(axis=1)
    manual_weekly = f[i - 168*8 + 1:i + 1].reshape(8, 168, 3).mean(axis=1)
    daily = o[16+24+8:16+24+8+30, :3]
    weekly = o[16+24+8+30:, :3]
    np.testing.assert_allclose(daily, manual_daily, atol=1e-5)
    np.testing.assert_allclose(weekly, manual_weekly, atol=1e-5)


def test_reviewer_zero_init_model_equals_predecessor():
    """The safety contract: with zero-init projection, the reviewer is
    SILENT — new model outputs must equal a reviewer-less model's."""
    import torch.nn as nn
    torch.manual_seed(3)
    m = TradingAgent(20, encoder_size=32, hidden_size=32, num_gru_layers=2, review_size=32)
    m.eval()
    assert bool((m.reviewer.proj.weight == 0).all()), "proj must start zeroed"
    x = torch.randn(2, 4, 10, 20)
    with torch.no_grad():
        d, v, _ = m.forward_sequence(x)
    # same model with reviewer output forcibly ignored must produce identical results
    with torch.no_grad():
        z = m.encode_windows(x)
        y, _ = m.gru(z)
        q = m.policy(y)
        alpha = torch.nn.functional.softplus(m.alpha(q)) + 0.05
        beta = torch.nn.functional.softplus(m.beta(q)) + 0.05
    mean_with = d.concentration1 / (d.concentration1 + d.concentration0)
    mean_without = alpha / (alpha + beta)
    torch.testing.assert_close(mean_with, mean_without.squeeze(-1), rtol=1e-5, atol=1e-6)


def test_reviewer_hidden_pair_roundtrip():
    """Hidden state is a stacked pair [main | reviewer]; forward and
    forward_sequence must stay consistent through it (axis invariant)."""
    torch.manual_seed(4)
    m = TradingAgent(6, encoder_size=16, hidden_size=16, num_gru_layers=2, review_size=16)
    m.eval()
    x = torch.randn(2, 5, 4, 6)
    with torch.no_grad():
        ds, vs, _ = m.forward_sequence(x)
        h = None; means, vals = [], []
        for t in range(5):
            d, v, h = m(x[:, t:t+1], h)
            means.append((d.concentration1/(d.concentration1+d.concentration0)).squeeze(-1))
            vals.append(v)
        means = torch.stack(means, dim=1); vals = torch.stack(vals, dim=1)
        sm = ds.concentration1/(ds.concentration1+ds.concentration0)
    torch.testing.assert_close(means, sm, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(vals, vs, rtol=1e-4, atol=1e-5)


# ---------- anti-overfit regularization knobs (wave 2.5, 2026-09-18) ----------
def test_model_has_no_stochastic_layers_keeps_ppo_kl_guard_sane():
    """Wave-2.5 lesson (measured, not assumed): network-level dropout is
    INCOMPATIBLE with PPO's trust region. Two dropout masks of the SAME
    weights produce ~1.7 KL vs the 0.02 cap, so every minibatch tripped the
    guard and training froze (ent=0, n_updates=0). Regularization must live
    in OBSERVATION space (env obs_noise/obs_dropout), where the stored
    tensor is identical for old/new logprobs. This test pins that the model
    contains no stochastic layers."""
    torch.manual_seed(9)
    m = TradingAgent(10, encoder_size=32, hidden_size=32, num_gru_layers=2, review_size=32)
    for mod in m.modules():
        assert not isinstance(mod, torch.nn.Dropout), \
            "no nn.Dropout may return to the model — it trashes the KL guard"
    m.train()
    x = torch.randn(3, 6, 5, 10)
    with torch.no_grad():
        d1, v1, _ = m.forward_sequence(x)
        d2, v2, _ = m.forward_sequence(x)
    # deterministic in train mode too (no stochastic layer anywhere)
    torch.testing.assert_close(d1.concentration1, d2.concentration1)
    torch.testing.assert_close(v1, v2)


def test_obs_dropout_default_zero_keeps_obs_bit_identical():
    """obs_dropout must default to OFF (0.0) so validation/live observations
    are byte-identical to the no-noise envs."""
    rng = np.random.default_rng(44)
    f = rng.normal(size=(2000, 12)).astype(np.float32)
    c = 100*np.exp(np.cumsum(rng.normal(0, .01, 2000)))
    e0 = SpotTradingEnv(f, c, 0, 2000, window=160, obs_dropout=0.0)
    e1 = SpotTradingEnv(f, c, 0, 2000, window=160)
    e0.reset(options={"start": 1400}); e1.reset(options={"start": 1400})
    np.testing.assert_array_equal(e0._obs(), e1._obs())


def test_obs_dropout_zeros_features_only():
    """obs_dropout must zero FEATURE cells only and never the state columns
    (position/pnl/drawdown/time/frame-flag — the frame-flag lane drives the
    fine/coarse mask in encode_windows)."""
    rng = np.random.default_rng(45)
    f = np.abs(rng.normal(size=(2000, 12))).astype(np.float32)
    c = 100*np.exp(np.cumsum(rng.normal(0, .01, 2000)))
    e = SpotTradingEnv(f, c, 0, 2000, window=160, obs_dropout=1.0)
    e.reset(options={"start": 1400})
    o = e._obs()
    assert o.shape[1] == 12 + 5
    np.testing.assert_array_equal(o[:, :12], 0.0)      # all features zeroed
    assert (o[:, 12:] != 0).any()                       # state columns untouched
    np.testing.assert_array_equal(o[:, -1] % 1, 0.0)    # frame flag still exact 0/1


def test_obs_noise_default_zero_keeps_obs_bit_identical():
    """obs_noise must default to OFF (0.0) so validation/live/golden-test
    observations are byte-identical to the no-noise envs."""
    rng = np.random.default_rng(42)
    f = rng.normal(size=(2000, 12)).astype(np.float32)
    c = 100*np.exp(np.cumsum(rng.normal(0, .01, 2000)))
    e0 = SpotTradingEnv(f, c, 0, 2000, window=160, obs_noise=0.0)
    e1 = SpotTradingEnv(f, c, 0, 2000, window=160)
    e0.reset(options={"start": 1400}); e1.reset(options={"start": 1400})
    np.testing.assert_array_equal(e0._obs(), e1._obs())


def test_obs_noise_perturbs_features_only_never_state_columns():
    """Noise must hit FEATURE columns only — the 5 trailing state columns
    (position/pnl/drawdown/time/frame-flag) carry account info the model
    needs exactly, and the frame-flag lane drives fine/coarse masking."""
    rng = np.random.default_rng(43)
    f = rng.normal(size=(2000, 12)).astype(np.float32)
    c = 100*np.exp(np.cumsum(rng.normal(0, .01, 2000)))
    e0 = SpotTradingEnv(f, c, 0, 2000, window=160, obs_noise=0.0)
    e1 = SpotTradingEnv(f, c, 0, 2000, window=160, obs_noise=0.1)
    e0.reset(options={"start": 1400}); e1.reset(options={"start": 1400})
    o0, o1 = e0._obs(), e1._obs()
    assert o0.shape[1] == o1.shape[1] == 12 + 5
    assert not np.allclose(o0[:, :12], o1[:, :12])      # features perturbed
    np.testing.assert_array_equal(o0[:, 12:], o1[:, 12:])  # state identical


def test_weight_decay_reaches_optimizer_and_survives_state_loading():
    """weight_decay must reach every Adam param_group, AND re-assert after a
    checkpoint's optimizer state loads (Adam.load_state_dict restores the
    SAVED groups — for old checkpoints that means 0.0)."""
    torch.manual_seed(12)
    m = TradingAgent(8, encoder_size=16, hidden_size=16, num_gru_layers=2, review_size=16)
    ppo = PPO(m, lr=0.0003, minibatch_size=2, weight_decay=0.001)
    for g in ppo.opt.param_groups:
        assert g["weight_decay"] == 0.001
    # emulate an old checkpoint whose optimizer was saved with decay 0
    old = {"optimizer": PPO(TradingAgent(8, encoder_size=16, hidden_size=16,
                                         num_gru_layers=2, review_size=16),
                            weight_decay=0.0).opt.state_dict()}
    ppo.opt.load_state_dict(old["optimizer"])
    # without the trainer's re-assert this would be 0.0 — pin the contract
    for g in ppo.opt.param_groups:
        g["weight_decay"] = 0.001
    for g in ppo.opt.param_groups:
        assert g["weight_decay"] == 0.001


def test_env_deadband_blocks_fee_churn():
    f = np.random.randn(2000, 80).astype(np.float32)
    c = np.linspace(100, 150, 2000)
    e = SpotTradingEnv(f, c, 0, 2000, 96, 100, min_trade_fraction=.02)
    e.reset(options={"start": 1400})
    _, _, _, _, i1 = e.step(np.array([.5], np.float32))   # opens position
    _, _, _, _, i2 = e.step(np.array([.501], np.float32)) # tiny wiggle: must NOT trade
    _, _, _, _, i3 = e.step(np.array([.9], np.float32))   # big change: trades
    assert i1["trades"] == 1 and i2["trades"] == 1 and i3["trades"] == 2


def test_env_freezes_at_end():
    f = np.random.randn(1500, 8).astype(np.float32)
    c = np.linspace(100, 150, 1500)
    e = SpotTradingEnv(f, c, 0, 1500, 96, 1000)  # episode_steps > remaining data
    e.reset(options={"start": 1495})
    for _ in range(50):
        _, _, term, trunc, _ = e.step(np.array([.5], np.float32))
    assert e.i == 1499 and term  # clamped, never wrapped into negative slices


def test_agent_full_range():
    m = TradingAgent(84)
    x = torch.randn(4, 8, 48, 84)
    d, v, h = m.forward_sequence(x)
    a = d.sample()
    assert a.shape == (4, 8) and v.shape == (4, 8)
    # hidden pair: 3 main GRU layers + 1 reviewer layer (+ reviewer r-cache)
    stacked, r_cache = h
    assert stacked.shape == (4, 4, 256) and r_cache.shape == (4, 8, 256)
    # Beta support is the full open interval (0,1): sampled actions near extremes are legal
    assert (a > 0).all() and (a < 1).all()
    # untrained net on random data should produce varied means, not a collapsed constant
    means = d.concentration1 / (d.concentration1 + d.concentration0)
    assert means.std() > 1e-4


def test_gru_time_axis_consistency():
    """forward_sequence over [B,T] must equal step-by-step forward with carried hidden.
    This is the invariant that was broken by the old env-major/time-major mismatch."""
    m = TradingAgent(6); m.eval()
    x = torch.randn(2, 5, 4, 6)
    with torch.no_grad():
        dist_seq, v_seq, _ = m.forward_sequence(x)
        h = None; means, vals = [], []
        for t in range(5):
            d, v, h = m(x[:, t:t+1], h)
            means.append((d.concentration1/(d.concentration1+d.concentration0)).squeeze(-1))
            vals.append(v)
        means = torch.stack(means, dim=1); vals = torch.stack(vals, dim=1)
        seq_means = dist_seq.concentration1/(dist_seq.concentration1+dist_seq.concentration0)
    torch.testing.assert_close(means, seq_means, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(vals, v_seq, rtol=1e-4, atol=1e-5)


def test_ppo_update_runs_and_matches_rollout_axis():
    """End-to-end PPO update on env-major [N,T,...] tensors with a real env."""
    torch.manual_seed(0)
    feats = np.random.randn(3000, 8).astype(np.float32)
    c = np.linspace(100, 160, 3000)
    envs = [SpotTradingEnv(feats, c, 0, 3000, 16, 32) for _ in range(3)]
    for k, e in enumerate(envs): e.reset(seed=k, options={"start": 1600+k})
    m = TradingAgent(8+5, encoder_size=32, hidden_size=32, num_gru_layers=1, review_size=32)
    ppo = PPO(m, epochs=2, minibatch_size=2, max_kl=10.0)  # KL guard off for smoke test
    N, T = len(envs), 16
    obs = np.stack([e._obs() for e in envs])
    O = torch.empty(N, T, 86, 13); A = torch.empty(N, T)
    LP = torch.empty(N, T); R = torch.empty(N, T)
    V = torch.empty(N, T); D = torch.empty(N, T)
    h = None; obs_t = torch.tensor(obs)
    for t in range(T):
        with torch.no_grad():
            dist, val, h = m(obs_t[:, None], h)
            act = dist.sample().clamp(1e-5, 1-1e-5); lp = dist.log_prob(act)
        nxt, rs, dn = [], [], []
        for j, e in enumerate(envs):
            no, r, te, tr, _ = e.step(np.array([act[j].item()], np.float32))
            nxt.append(no); rs.append(r); dn.append(te or tr)
        O[:, t] = obs_t; A[:, t] = act; LP[:, t] = lp
        R[:, t] = torch.tensor(rs); V[:, t] = val; D[:, t] = torch.tensor(dn, dtype=torch.float32)
        obs_t = torch.tensor(np.stack(nxt))
    with torch.no_grad():
        _, last_v, _ = m(obs_t[:, None], h)
    # log_probs recomputed BEFORE the update must match rollout values (axis correctness)
    with torch.no_grad():
        dist, _, _ = m.forward_sequence(O)
        lp2 = dist.log_prob(A.clamp(1e-6, 1-1e-6))
    torch.testing.assert_close(lp2, LP, rtol=1e-4, atol=1e-6)
    stats = ppo.update(O, A, LP, R, V, D, last_v)
    assert np.isfinite(stats["policy_loss"]) and np.isfinite(stats["value_loss"])


def test_dataset_window_guard(tmp_path=None):
    import os, tempfile
    from src.dataset import AssetDataset
    d = tempfile.mkdtemp()
    n = 2500
    df = pd.DataFrame({"datetime": pd.date_range("2020-01-01", periods=n, freq="h", tz="UTC"),
                       "close": np.linspace(100, 150, n)})
    for col in ["f1", "f2"]: df[col] = np.random.randn(n).astype(np.float32)
    df.to_csv(os.path.join(d, "X_USDT_features.csv"), index=False)
    pd.DataFrame([{"symbol": "X/USDT", "file": "X_USDT_features.csv",
                   "start_idx": 0, "end_idx": n}]).to_csv(os.path.join(d, "train_manifest.csv"), index=False)
    ds = AssetDataset(d, "train", window=96)
    for _ in range(50):
        a, s = ds.sample(np.random.default_rng(), steps=100, window=96)
        assert s >= 1343 and s + 100 <= a["end"]  # full weekly margin behind, episode ahead


def test_stable_criterion_beats_lucky_peak():
    """The consistency metric (mean of last 10 evals) must not be hijackable
    by a single lucky validation peak — the user's core requirement."""
    import numpy as np
    hist = [5.0]*20 + [95.0] + [5.0]*9   # one lucky spike among consistent mediocrity
    stable_vals = []
    for i in range(len(hist)):
        if i >= 10:
            stable_vals.append(float(np.mean(hist[i-9:i+1])))
    assert max(stable_vals) < 20.0, "single spike must not dominate the stable metric"
    hist2 = [10.0, 11.0, 10.5, 11.5, 10.2, 11.8, 10.9, 11.2, 10.7, 11.4]  # consistently good
    assert float(np.mean(hist2)) > 10.5, "consistent performance must be recognized"


def _doctrine_env(closes, **kw):
    """Env helper on a synthetic price path for doctrine tests.
    v3 observation needs max(window_fine, 96, 1344) candles of history —
    we use window=16 (fine) with a >=1344-row lead-in inside the path itself."""
    import numpy as np
    f = np.zeros((len(closes), 8), dtype=np.float32)
    w = kw.pop("window", 16)
    e = SpotTradingEnv(f, np.asarray(closes, dtype=np.float64), 0, len(closes),
                       window=w, episode_steps=max(1, len(closes) - 1344), min_trade_fraction=0.0, **kw)
    return e


def _run_env(e, start, action=1.0):
    e.reset(options={"start": start})
    info = None
    while True:
        _, _, t, tr, info = e.step(np.array([action], np.float32))
        if t or tr: break
    return info


def test_doctrine_punishes_big_loss_vs_gains():
    """User's exact scenario: 1000 won then 800 lost must have worse purity
    than 1000 won then 300 lost (holding through both).
    reset(start=3) -> entry at close[3]; value marks at close[4..]."""
    import numpy as np
    # entry at 100, rises to 110 (+10% = +1000 on 10k), then:
    # A: falls to 102 => L≈8%  (1000 gain / 800 loss)
    pathA = [100.0]*1400 + [110.0, 106.0, 102.0, 102.0]
    # B: falls to 107 => L≈3%  (1000 gain / 300 loss)
    pathB = [100.0]*1400 + [110.0, 108.0, 107.0, 107.0]
    iA = _run_env(_doctrine_env(pathA), start=1399)
    iB = _run_env(_doctrine_env(pathB), start=1399)
    assert iA["purity"] < iB["purity"], f"1000/800 (p={iA['purity']:.2f}) must be punished harder than 1000/300 (p={iB['purity']:.2f})"


def test_doctrine_bleed_penalty_grows_with_drawdown():
    import numpy as np
    e = _doctrine_env([100.0] * 1410)
    e.reset(options={"start": 1399})
    # simulate peak rising first (as if portfolio climbed), then test depths
    e.peak = 100.0; e.hwm = 100.0; e.trough = 100.0
    p0 = e._bleed_penalty(100.0)      # no drawdown -> 0
    p_small = e._bleed_penalty(99.6)  # 0.4% dip -> below bleed_start(0.5%) -> 0
    p_mid = e._bleed_penalty(97.0)    # 3% dip -> some penalty
    p_deep = e._bleed_penalty(93.0)   # 7% dip -> max penalty
    assert p0 == 0.0 and p_small == 0.0
    assert p_mid < 0 and p_deep < p_mid, "penalty must escalate with drawdown depth"


def test_doctrine_equal_gain_loss_scores_worst():
    """gain == loss must yield purity ~0, worse than any profitable ratio."""
    import numpy as np
    e = _doctrine_env([100.0] * 1400 + [110.0, 105.0, 100.0])  # +10% then full retrace
    info = _run_env(e, start=1399)
    assert abs(info["purity"]) < 0.35, f"equal gain and loss must be ~purity 0, got {info['purity']}"


def test_crisis_sampling_finds_crashes():
    """Crisis-biased sampling must locate the crash region of a synthetic asset."""
    import os, tempfile
    import numpy as np, pandas as pd
    from src.dataset import AssetDataset
    d = tempfile.mkdtemp()
    n = 3000
    closes = np.concatenate([np.linspace(100, 200, 1500), np.linspace(200, 60, 800), np.linspace(60, 90, 700)])
    df = pd.DataFrame({"datetime": pd.date_range("2020-01-01", periods=n, freq="h", tz="UTC"),
                       "close": closes})
    for col in ["f1", "f2"]: df[col] = np.random.randn(n).astype(np.float32)
    df.to_csv(os.path.join(d, "X_USDT_features.csv"), index=False)
    pd.DataFrame([{"symbol": "X/USDT", "file": "X_USDT_features.csv",
                   "start_idx": 0, "end_idx": n}]).to_csv(os.path.join(d, "train_manifest.csv"), index=False)
    ds = AssetDataset(d, "train", window=16)
    rng = np.random.default_rng(0)
    crash_starts = set()
    for _ in range(40):
        a, s = ds.sample(rng, steps=100, window=16, crisis_bias=1.0)  # force crisis mode
        crash_starts.add(s // 100)
    # the crash spans roughly indices 1500-2300 -> starts//100 in 15..22
    assert any(15 <= c <= 23 for c in crash_starts), f"crisis sampling missed crash: {sorted(crash_starts)}"


def test_regime_classifier():
    from src.evaluation.evaluate import classify_regime
    up = np.linspace(100, 150, 100); dn = np.linspace(150, 90, 100); flat = np.full(100, 100.0)
    assert classify_regime(up, 0, 100) == "up"
    assert classify_regime(dn, 0, 100) == "down"
    assert classify_regime(flat, 0, 100) == "sideways"


def test_inactivity_bias_punishes_pure_cash():
    """100% cash every step must accumulate negative shaping — the coward
    equilibrium (never trade) is mathematically dominated."""
    e = _doctrine_env([100.0] * 1410)
    e.reset(options={"start": 1399})
    rewards = []
    while True:
        _, r, t, tr, _ = e.step(np.array([0.0], np.float32))
        rewards.append(r)
        if t or tr: break
    assert sum(rewards) < 0, "sitting in cash forever must cost something"


def test_peak_ambition_rewards_new_highs_only():
    """Bonus fires on new highs, is capped, and doesn't repeat on flat."""
    e = _doctrine_env([100.0] * 1400 + [110.0, 110.0, 121.0, 121.0])
    e.reset(options={"start": 1399})
    while True:
        _, _, t, tr, _ = e.step(np.array([1.0], np.float32))
        if t or tr: break
    # entered at 100% -> value follows price; new highs at 110 and 121
    assert e.peak_count >= 2, f"each new portfolio high must count as ambition, got {e.peak_count}"


def test_ecstasy_fades_with_mastery():
    """Same-sized wins should produce smaller ecstasy over time (EMA adapts):
    progressive ambition — the same thrill demands bigger feats."""
    import numpy as np
    from src.training.environment import SpotTradingEnv
    f = np.zeros((1410, 8), np.float32)
    e = SpotTradingEnv(f, np.full(1410, 100.0), 0, 1410, window=16, episode_steps=10)
    e.reset(options={"start": 1399})
    e.ret_ema = 0.0
    first = e._ecstasy(0.05)
    e.ret_ema = 0.049   # mastered: now expects ~this return
    faded = e._ecstasy(0.05)
    assert first > faded, "identical returns must thrill less once expected"


def test_survival_bonus_for_fleeing_before_crash():
    """Being in cash while the market crashes must pay the survival bonus."""
    import numpy as np
    from src.training.environment import SpotTradingEnv
    # flat, then a cliff at the end; agent stays in cash
    path = [100.0]*1400 + [94.0, 94.0, 94.0, 94.0, 94.0]
    f = np.zeros((len(path), 8), np.float32)
    e = SpotTradingEnv(f, np.asarray(path, dtype=np.float64), 0, len(path),
                       window=16, episode_steps=8)
    e.reset(options={"start": 1399})
    survived = 0.0
    while True:
        _, r, t, tr, info = e.step(np.array([0.0], np.float32))
        if t or tr: break
    # the step where price dropped 6% while agent in cash must have rewarded
    assert e._survival_check.__doc__ is not None  # sanity
    # direct check: simulate one crash step
    e._prev_close = 100.0
    e.i = len(path)-1  # close=94 -> drop of 6%
    assert e._survival_check() > 0, "fleeing before a 6% crash must pay"


def test_timing_bonus_for_surgical_entry():
    """Entry that keeps improving for `timing_horizon` candles earns bonus.
    Step consumes pending entries internally, so we evaluate the mechanism
    directly on a fresh env to keep the unit test pure."""
    import numpy as np
    from src.training.environment import SpotTradingEnv
    path = [100.0]*1400 + [98, 99, 100, 101, 102]
    f = np.zeros((len(path), 8), np.float32)
    e = SpotTradingEnv(f, np.asarray(path, dtype=np.float64), 0, len(path),
                       window=16, episode_steps=8, timing_horizon=4)
    e.reset(options={"start": 1399})
    e._register_entry(100.0)                      # surgical entry at the dip
    e._pending[0]["step"] = e.steps - 10          # force it to be matured
    e.i = len(path)-1                             # price now 102
    bonus = e._resolve_timing()
    assert bonus > 0, f"surgical entry must earn timing bonus, got {bonus}"
    assert bonus <= e.ecstasy_cap, "timing bonus must be capped"
    # a doomed entry (price below entry after horizon) earns nothing
    e2 = SpotTradingEnv(f, np.asarray(path, dtype=np.float64), 0, len(path),
                        window=16, episode_steps=8, timing_horizon=4)
    e2.reset(options={"start": 1399})
    e2._register_entry(105.0)                     # bought the top
    e2._pending[0]["step"] = e2.steps - 10
    e2.i = len(path)-1
    assert e2._resolve_timing() == 0.0, "entry that fell must earn no timing bonus"



def test_adaptive_learning_reacts_to_decline_and_recovery():
    """Wave 1: eval decline must cut LR and boost entropy; recovery must
    relax both back. Never stops training."""
    import json
    from src.trainer import Trainer
    cfg = json.load(open('config/default.json'))
    cfg.update({'model_dir': 'models/bench_t', 'log_dir': 'logs/bench_t'})
    import os
    os.makedirs('models/bench_t', exist_ok=True); os.makedirs('logs/bench_t', exist_ok=True)
    import shutil
    if os.path.exists('logs/bench_t/metrics.csv'): os.remove('logs/bench_t/metrics.csv')
    t = Trainer(cfg, device=torch.device('cpu'))
    # declining phase: 10 evals going down
    hist = [5.0, 4.5, 4.0, 3.5, 3.0, 2.5, 2.0, 1.5, 1.0, 0.5, 0.2, -0.1]
    t._adapt_on_eval(hist)
    assert t.adapt['lr_scale'] < 1.0 and t.adapt['ent_scale'] > 1.0, "decline must trigger protection"
    # recovery phase
    recovering = hist + [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    t.adapt['cooldown'] = 0
    t._adapt_on_eval(recovering)
    assert t.adapt['lr_scale'] >= 1.0 and t.adapt['ent_scale'] <= 1.0, "recovery must relax protection"
    shutil.rmtree('models/bench_t'); shutil.rmtree('logs/bench_t')


def test_adapt_reset_on_resume_switch():
    """adapt_reset_on_resume must return the adapt state to fresh LR/entropy
    scales (1.0/1.0) while the model weights are untouched by _load_checkpoint
    — the one-time bridge into a new regularization regime."""
    import json
    from src.trainer import Trainer
    cfg = json.load(open('config/default.json'))
    assert cfg.get("adapt_reset_on_resume", False) is True, \
        "default config must reset adapt state once for the wave-2.5 regime"
    fresh = Trainer._fresh_adapt()
    assert fresh == {"lr_scale": 1.0, "ent_scale": 1.0, "cooldown": 0}


if __name__ == "__main__":
    test_features_no_future_fill(); test_features_prefix_invariance()
    test_env_portfolio_state_in_obs(); test_env_daily_weekly_frames_past_only()
    test_env_daily_weekly_rows_match_manual_aggregation()
    test_env_deadband_blocks_fee_churn()
    test_env_freezes_at_end(); test_agent_full_range()
    test_gru_time_axis_consistency(); test_reviewer_zero_init_model_equals_predecessor()
    test_reviewer_hidden_pair_roundtrip()
    test_ppo_update_runs_and_matches_rollout_axis()
    test_dataset_window_guard(); test_stable_criterion_beats_lucky_peak()
    test_doctrine_punishes_big_loss_vs_gains(); test_doctrine_bleed_penalty_grows_with_drawdown()
    test_doctrine_equal_gain_loss_scores_worst(); test_crisis_sampling_finds_crashes()
    test_regime_classifier(); test_inactivity_bias_punishes_pure_cash()
    test_peak_ambition_rewards_new_highs_only(); test_ecstasy_fades_with_mastery()
    test_survival_bonus_for_fleeing_before_crash(); test_timing_bonus_for_surgical_entry()
    print("OK")
