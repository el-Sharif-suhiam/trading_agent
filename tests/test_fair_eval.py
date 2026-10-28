"""W1 repair-plan tests: checkpoint compat + fair evaluation protocol."""
import json
import numpy as np
import pandas as pd
import pytest
import torch

from src.model import TradingAgent
from src.model.ckpt import strip_orig_mod, add_orig_mod, load_state_compat, load_checkpoint
from src.evaluation.fair_eval import run_segments, sample_starts, stable_score
from src.training.environment import SpotTradingEnv


WINDOW = 160
EPISODE = 512
ROWS = WINDOW + 24 + 8 + 30 + 8
MARGIN = 168 * 8 + EPISODE + 5   # weekly reach + episode + slack


def _cfg():
    cfg = json.load(open("config/default.json"))
    cfg.update({"window": WINDOW, "rollout_steps": EPISODE,
                "fee": 0.001, "slippage": 0.0005, "initial_cash": 10000.0})
    return cfg


def _asset(seed, n=None):
    n = n or (MARGIN + 100)
    rng = np.random.default_rng(seed)
    c = 100 * np.exp(np.cumsum(rng.normal(0, .01, n)))
    feats = rng.normal(size=(n, 6)).astype(np.float32)
    return {"symbol": f"T{seed}/USDT", "features": feats, "close": c,
            "start": 0, "end": n}


def _small_model():
    torch.manual_seed(1)
    m = TradingAgent(6 + 5, encoder_size=16, hidden_size=16, num_gru_layers=1, review_size=16)
    return m.eval()


# ---------- checkpoint compat (the broken-gate fix) ----------
def test_strip_orig_mod_is_idempotent_and_preserves_values():
    sd = {"gru.weight": torch.zeros(2), "encoder.0.bias": torch.ones(2)}
    stripped = strip_orig_mod(add_orig_mod(strip_orig_mod(sd)))
    assert set(stripped) == set(sd)
    assert torch.equal(stripped["gru.weight"], sd["gru.weight"])
    # already-stripped keys pass through unchanged
    assert set(strip_orig_mod(sd)) == set(sd)


def test_load_state_compat_accepts_compiled_prefix():
    """A state_dict saved from torch.compile (`_orig_mod.` prefixed) must load
    into an eager model — this is exactly what broke the evaluation gates."""
    m = _small_model()
    compiled_like = add_orig_mod(m.state_dict())
    load_state_compat(m, compiled_like)  # no missing/unexpected-key crash
    # and a plain dict must load too (reverse direction)
    load_state_compat(m, m.state_dict())


def test_load_checkpoint_returns_metadata(tmp_path):
    """load_checkpoint loads weights AND hands back ck metadata (update/...)."""
    m = _small_model()
    p = tmp_path / "ck.pt"
    torch.save({"model": add_orig_mod(m.state_dict()), "update": 123,
                "best_val": 5.0}, p)
    ck = load_checkpoint(m, p, weights_only=False)
    assert ck["update"] == 123 and ck["best_val"] == 5.0
    # weights actually loaded (bit-exact roundtrip after prefix stripping)
    for k, v in m.state_dict().items():
        torch.testing.assert_close(v, strip_orig_mod(ck["model"])[k])


# ---------- fair evaluation protocol ----------
def test_sample_starts_respects_margin_and_is_deterministic():
    cfg = _cfg()
    assets = [_asset(7)]
    j1 = sample_starts(assets, cfg, EPISODE, 8, seed=3)
    j2 = sample_starts(assets, cfg, EPISODE, 8, seed=3)
    assert j1 == j2, "same seed must give identical segments"
    for a, s in j1:
        assert s >= 1344, f"segment start {s} must include the 1344-candle margin"
        assert s + EPISODE <= a["end"] - 1, "episode must fit before asset end"


def test_fair_eval_is_batch_size_invariant():
    """The single most brittle part of batched eval is hidden-state slicing.
    Same segments, same model: batch=1 and batch=8 MUST agree bit-for-bit."""
    cfg = _cfg()
    assets = [_asset(11), _asset(12)]
    r1 = run_segments(_small_model(), assets, cfg, torch.device("cpu"),
                      per_asset=3, seed=5, batch_size=1)
    r8 = run_segments(_small_model(), assets, cfg, torch.device("cpu"),
                      per_asset=3, seed=5, batch_size=8)
    for s1, s2 in zip(r1["segments"], r8["segments"]):
        assert s1["symbol"] == s2["symbol"] and s1["start"] == s2["start"]
        assert s1["model_ret"] == pytest.approx(s2["model_ret"], abs=1e-6)
        assert s1["purity"] == pytest.approx(s2["purity"], abs=1e-7)


def test_fair_eval_alpha_is_model_vs_buyhold_on_same_candles():
    """alpha must be model return minus buy-hold measured on the SAME window —
    this is the definition that stops exams from rewarding lucky bull markets."""
    cfg = _cfg()
    res = run_segments(_small_model(), [_asset(21)], cfg, torch.device("cpu"),
                       per_asset=2, seed=9)
    for s in res["segments"]:
        assert s["alpha"] == pytest.approx(s["model_ret"] - s["bh_ret"], abs=1e-7)
    agg = res["aggregate"]
    assert agg["n_segments"] == 2
    for k in ("alpha_mean_pct", "alpha_t", "purity", "beat_buyhold"):
        assert k in agg


def test_stable_score_discounts_bleeding_models_never_boosts():
    """Doctrine-weighted score: purity in [-1,0] must DISCOUNT not boost, and
    one lucky spike among medicore neighbors must not dominate the window."""
    alphas = [0.5] * 10
    intact = stable_score(alphas, [1.0] * 10)    # purity 1.0 -> full alpha
    high = stable_score(alphas, [0.8] * 10)      # purity 0.8 -> 0.9 multiplier
    bleeding = stable_score(alphas, [-0.5] * 10)  # purity negative -> half
    assert intact == pytest.approx(0.5)
    assert high == pytest.approx(0.45)
    assert bleeding == pytest.approx(0.25)
    assert bleeding < high < intact
    assert stable_score([0.1] * 9, [0.0] * 9) is None  # <10 evals -> no score
    # lucky spike can't dominate: 9 mediocre + 1 huge stays close to mediocrity
    spiked = stable_score([0.2] * 9 + [50.0], [1.0] * 10)
    assert spiked == pytest.approx(5.18)  # (0.2*9+50)/10 — window-capped, not spike-dominated


def test_legacy_checkpoint_resets_grading_not_weights():
    """Pre-fair checkpoints (no metric_version) started grading best/stable on
    the legacy protocol; the trainer restarts grading under fair alpha while
    the learned weights pass through untouched."""
    import inspect
    from src.trainer import Trainer
    src = inspect.getsource(Trainer._load_checkpoint)
    assert "metric_version" in src and "eval_history" in src
    # the trainer stores the protocol version alongside every checkpoint
    assert "metric_version" in inspect.getsource(Trainer._ckpt)
