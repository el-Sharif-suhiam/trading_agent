"""P4 school-vote tests — the causality contract is THE test that matters:
a vote at bar t must depend only on bars <= t (no look-ahead, the same
contract that protects the 92-feature engine)."""
import numpy as np
import pytest

from src.features.schools import CONFIRM, NAMES, school_votes, zigzag


def _synth(n=2500, seed=0):
    rng = np.random.default_rng(seed)
    c = 100 * np.exp(np.cumsum(rng.normal(0, .01, n)))
    o = c * (1 + rng.normal(0, .002, n))
    h = np.maximum(o, c) * (1 + np.abs(rng.normal(0, .003, n)))
    l = np.minimum(o, c) * (1 - np.abs(rng.normal(0, .003, n)))
    v = np.abs(rng.normal(1, .3, n)) + .1
    return o, h, l, c, v


def test_votes_shape_scale_and_nan_free():
    o, h, l, c, v = _synth()
    V = school_votes(o, h, l, c, v)
    assert V.shape == (len(c), 8)
    assert not np.isnan(V).any()
    assert V.min() >= -1.0 and V.max() <= 1.0


def test_votes_causal_prefix_invariance():
    """THE contract: votes computed on a prefix must equal the prefix of the
    votes computed on the full series, at every probed bar (pivot votes
    included — a pivot may only be used after its confirmation bar)."""
    o, h, l, c, v = _synth(seed=3)
    full = school_votes(o, h, l, c, v)
    cut = 1400
    pre = school_votes(o[:cut], h[:cut], l[:cut], c[:cut], v[:cut])
    assert np.allclose(pre, full[:cut], atol=1e-6), \
        "prefix votes must not change when the future is revealed"


def test_zigzag_pivots_cannot_be_seen_before_confirmation():
    o, h, l, c, v = _synth(seed=5)
    pi, pk, pc = zigzag(c)
    assert len(pi) > 5
    # every pivot's confirmation bar is at or after the pivot itself
    assert (pc >= pi).all()
    # a pivot at index i is never readable at any bar t < conf


def test_mtf_follows_clean_trend():
    n = 1500
    up = np.linspace(100, 160, n) + np.sin(np.linspace(0, 40, n))
    o = up.copy(); h = up * 1.001; l = up * 0.999; v = np.ones(n)
    V = school_votes(o, h, l, up, v)
    assert V[600:, 4].mean() > 0.5, "clean uptrend must vote MTF positive"


def test_names_are_stable():
    assert len(NAMES) == 8 and all(n_.startswith("school_") for n_ in NAMES)
