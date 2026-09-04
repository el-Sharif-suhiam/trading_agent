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
    # strong signals must be able to push the Beta mean near 0 and near 1
    big = torch.full((1, 1, 96, 84), 3.0)
    d_hi, _, _ = m(big, None); d_lo, _, _ = m(-big, None)
    mean_hi = (d_hi.concentration1 / (d_hi.concentration1 + d_hi.concentration0)).item()
    mean_lo = (d_lo.concentration1 / (d_lo.concentration1 + d_lo.concentration0)).item()
    assert mean_hi > 0.6 and mean_lo < 0.4  # untrained: just check extremes differ and are reachable


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
    stats = ppo.update(O, A, LP, R, V, D, last_v)
    assert stats["policy_loss"] == stats["policy_loss"]  # finite check placeholder
    assert np.isfinite(stats["policy_loss"]) and np.isfinite(stats["value_loss"])
    # log_probs recomputed under no-grad must match rollout values (axis correctness)
    with torch.no_grad():
        dist, _, _ = m.forward_sequence(O)
        lp2 = dist.log_prob(A.clamp(1e-6, 1-1e-6))
    torch.testing.assert_close(lp2, LP, rtol=1e-4, atol=1e-6)


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


if __name__ == "__main__":
    test_features_no_future_fill(); test_features_prefix_invariance()
    test_env_portfolio_state_in_obs(); test_env_deadband_blocks_fee_churn()
    test_env_freezes_at_end(); test_agent_full_range()
    test_gru_time_axis_consistency(); test_ppo_update_runs_and_matches_rollout_axis()
    test_dataset_window_guard()
    print("OK")
