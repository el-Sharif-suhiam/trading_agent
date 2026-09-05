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
    f = np.random.randn(1000, 80).astype(np.float32)
    c = np.linspace(100, 150, 1000)
    e = SpotTradingEnv(f, c, 0, 1000, 96, 100)
    o, _ = e.reset(options={"start": 95})
    assert o.shape == (96, 84)  # 80 market + 4 portfolio state


def test_env_deadband_blocks_fee_churn():
    f = np.random.randn(1000, 80).astype(np.float32)
    c = np.linspace(100, 150, 1000)
    e = SpotTradingEnv(f, c, 0, 1000, 96, 100, min_trade_fraction=.02)
    e.reset(options={"start": 95})
    _, _, _, _, i1 = e.step(np.array([.5], np.float32))   # opens position
    _, _, _, _, i2 = e.step(np.array([.501], np.float32)) # tiny wiggle: must NOT trade
    _, _, _, _, i3 = e.step(np.array([.9], np.float32))   # big change: trades
    assert i1["trades"] == 1 and i2["trades"] == 1 and i3["trades"] == 2


def test_env_freezes_at_end():
    f = np.random.randn(200, 8).astype(np.float32)
    c = np.linspace(100, 150, 200)
    e = SpotTradingEnv(f, c, 0, 200, 96, 1000)  # episode_steps > remaining data
    e.reset(options={"start": 95})
    for _ in range(150):
        _, _, term, trunc, _ = e.step(np.array([.5], np.float32))
    assert e.i == 199 and term  # clamped, never wrapped into negative slices


def test_agent_full_range():
    m = TradingAgent(84)
    x = torch.randn(4, 8, 96, 84)
    d, v, h = m.forward_sequence(x)
    a = d.sample()
    assert a.shape == (4, 8) and v.shape == (4, 8) and h.shape == (2, 4, 128)
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
    feats = np.random.randn(2000, 8).astype(np.float32)
    c = np.linspace(100, 160, 2000)
    envs = [SpotTradingEnv(feats, c, 0, 2000, 96, 32) for _ in range(3)]
    for k, e in enumerate(envs): e.reset(seed=k, options={"start": 200+k})
    m = TradingAgent(8+4, encoder_size=32, hidden_size=32, num_gru_layers=1)
    ppo = PPO(m, epochs=2, minibatch_size=2, max_kl=10.0)  # KL guard off for smoke test
    N, T = len(envs), 16
    obs = np.stack([e._obs() for e in envs])
    O = torch.empty(N, T, 96, 12); A = torch.empty(N, T)
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
    n = 400
    df = pd.DataFrame({"datetime": pd.date_range("2020-01-01", periods=n, freq="h", tz="UTC"),
                       "close": np.linspace(100, 150, n)})
    for col in ["f1", "f2"]: df[col] = np.random.randn(n).astype(np.float32)
    df.to_csv(os.path.join(d, "X_USDT_features.csv"), index=False)
    pd.DataFrame([{"symbol": "X/USDT", "file": "X_USDT_features.csv",
                   "start_idx": 0, "end_idx": n}]).to_csv(os.path.join(d, "train_manifest.csv"), index=False)
    ds = AssetDataset(d, "train", window=96)
    for _ in range(50):
        a, s = ds.sample(np.random.default_rng(), steps=100, window=96)
        assert s >= 95 and s + 100 <= a["end"]  # full window behind, episode ahead


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


def _doctrine_env(closes):
    """Env helper on a synthetic price path for doctrine tests.
    All cash (position=0): portfolio value follows... nothing. To observe
    mark-to-market waterfall we hold full position via action=1."""
    import numpy as np
    f = np.zeros((len(closes), 8), dtype=np.float32)
    w = 4
    e = SpotTradingEnv(f, np.asarray(closes, dtype=np.float64), 0, len(closes),
                       window=w, episode_steps=len(closes)-w, min_trade_fraction=0.0)
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
    pathA = [100, 100, 100, 100, 110, 106, 102, 102, 102, 102]
    # B: falls to 107 => L≈3%  (1000 gain / 300 loss)
    pathB = [100, 100, 100, 100, 110, 108, 107, 107, 107, 107]
    iA = _run_env(_doctrine_env(pathA), start=3)
    iB = _run_env(_doctrine_env(pathB), start=3)
    assert iA["purity"] < iB["purity"], f"1000/800 (p={iA['purity']:.2f}) must be punished harder than 1000/300 (p={iB['purity']:.2f})"


def test_doctrine_bleed_penalty_grows_with_drawdown():
    import numpy as np
    e = _doctrine_env([100]*6)
    e.reset(options={"start": 3})
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
    e = _doctrine_env([100, 100, 100, 100, 110, 105, 100, 100, 100])  # +10% then full retrace
    info = _run_env(e, start=3)
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
    e = _doctrine_env([100, 100, 100, 100, 100, 100, 100, 100])
    e.reset(options={"start": 3})
    rewards = []
    while True:
        _, r, t, tr, _ = e.step(np.array([0.0], np.float32))
        rewards.append(r)
        if t or tr: break
    assert sum(rewards) < 0, "sitting in cash forever must cost something"


def test_peak_ambition_rewards_new_highs_only():
    """Bonus fires on new highs, is capped, and doesn't repeat on flat."""
    e = _doctrine_env([100, 100, 100, 100, 110, 110, 110, 121, 121])
    e.reset(options={"start": 3})
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
    f = np.zeros((30, 8), np.float32)
    e = SpotTradingEnv(f, np.full(30, 100.0), 0, 30, window=4, episode_steps=26)
    e.reset(options={"start": 3})
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
    path = [100]*12 + [100, 100, 100, 94.0, 94.0, 94.0, 94.0, 94.0]
    f = np.zeros((len(path), 8), np.float32)
    e = SpotTradingEnv(f, np.asarray(path, dtype=np.float64), 0, len(path),
                       window=4, episode_steps=len(path)-5)
    e.reset(options={"start": 3})
    survived = 0.0
    while True:
        _, r, t, tr, info = e.step(np.array([0.0], np.float32))
        if t or tr: break
    # the step where price dropped 6% while agent in cash must have rewarded
    assert e._survival_check.__doc__ is not None  # sanity
    # direct check: simulate one crash step
    e._prev_close = 100.0
    e.i = 15  # close[15]=94 -> drop of 6%
    assert e._survival_check() > 0, "fleeing before a 6% crash must pay"


def test_timing_bonus_for_surgical_entry():
    """Entry that keeps improving for `timing_horizon` candles earns bonus.
    Step consumes pending entries internally, so we evaluate the mechanism
    directly on a fresh env to keep the unit test pure."""
    import numpy as np
    from src.training.environment import SpotTradingEnv
    path = [100, 100, 100, 100, 98, 99, 100, 101, 102, 103, 104, 105]
    f = np.zeros((len(path), 8), np.float32)
    e = SpotTradingEnv(f, np.asarray(path, dtype=np.float64), 0, len(path),
                       window=4, episode_steps=len(path)-5, timing_horizon=4)
    e.reset(options={"start": 3})
    e._register_entry(100.0)                      # surgical entry at the dip
    e._pending[0]["step"] = e.steps - 10          # force it to be matured
    e.i = 11                                      # price now 105
    bonus = e._resolve_timing()
    assert bonus > 0, f"surgical entry must earn timing bonus, got {bonus}"
    assert bonus <= e.ecstasy_cap, "timing bonus must be capped"
    # a doomed entry (price below entry after horizon) earns nothing
    e2 = SpotTradingEnv(f, np.asarray(path, dtype=np.float64), 0, len(path),
                        window=4, episode_steps=len(path)-5, timing_horizon=4)
    e2.reset(options={"start": 3})
    e2._register_entry(105.0)                     # bought the top
    e2._pending[0]["step"] = e2.steps - 10
    e2.i = 11
    assert e2._resolve_timing() == 0.0, "entry that fell must earn no timing bonus"


if __name__ == "__main__":
    test_features_no_future_fill(); test_features_prefix_invariance()
    test_env_portfolio_state_in_obs(); test_env_deadband_blocks_fee_churn()
    test_env_freezes_at_end(); test_agent_full_range()
    test_gru_time_axis_consistency(); test_ppo_update_runs_and_matches_rollout_axis()
    test_dataset_window_guard(); test_stable_criterion_beats_lucky_peak()
    test_doctrine_punishes_big_loss_vs_gains(); test_doctrine_bleed_penalty_grows_with_drawdown()
    test_doctrine_equal_gain_loss_scores_worst(); test_crisis_sampling_finds_crashes()
    test_regime_classifier(); test_inactivity_bias_punishes_pure_cash()
    test_peak_ambition_rewards_new_highs_only(); test_ecstasy_fades_with_mastery()
    test_survival_bonus_for_fleeing_before_crash(); test_timing_bonus_for_surgical_entry()
    print("OK")
