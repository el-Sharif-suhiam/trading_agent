"""CAUSAL school-vote channels (P4) — each technical school emits ONE
deterministic decision vote per bar in [-1, +1] (+1=buy, -1=sell, 0=hold).

OWNER DOCTRINE (rule 4, amended 2026-10-30): schools may PROPOSE decisions;
the model experiments and decides what to trust through learning. Nothing
here is a forced rule — these are advisory columns the policy may weigh or
ignore, learned end-to-end from P&L like every other feature.

HARD CAUSALITY CONTRACT (tested): a vote at bar t depends ONLY on bars <= t.
Zigzag pivots carry a confirmation index — a pivot becomes KNOWN only after
price reverses CONFIRM from it; reading it earlier is look-ahead (forbidden).
All votes are computed from raw OHLCV only.

Columns: elliott, wyckoff, gann, cycles, mtf, vsa, classical, harmonic.
These are simplified deterministic proxies of each school — honest, testable,
and cheap; the trust weights belong to the model, not to this file.
"""
import numpy as np
import pandas as pd

CONFIRM = 0.03  # zigzag reversal threshold (pivot confirmed after 3% reversal)
NAMES = ["school_elliott", "school_wyckoff", "school_gann", "school_cycles",
         "school_mtf", "school_vsa", "school_classical", "school_harmonic"]


def zigzag(close, thr=CONFIRM):
    """Causal zigzag. Returns (piv_i, piv_k, piv_c): extreme bar index, kind
    (+1 high / -1 low), and the bar at which the pivot became KNOWN (>= idx)."""
    n = len(close)
    piv_i, piv_k, piv_c = [], [], []
    direction = 0  # 0 unknown, +1 tracking up-leg (next HIGH), -1 next LOW
    hi_i = lo_i = 0
    cand_i, cand_p = 0, close[0]
    for t in range(n):
        p = close[t]
        if direction == 0:
            if p > close[hi_i]: hi_i = t
            if p < close[lo_i]: lo_i = t
            if close[hi_i] >= close[lo_i] * (1 + thr) and hi_i > lo_i:
                piv_i.append(lo_i); piv_k.append(-1); piv_c.append(t)
                direction, cand_i, cand_p = 1, t, p
            elif p <= close[hi_i] * (1 - thr) and hi_i > lo_i:
                piv_i.append(hi_i); piv_k.append(1); piv_c.append(t)
                direction, cand_i, cand_p = -1, t, p
        elif direction == 1:
            if p >= cand_p:
                cand_i, cand_p = t, p
            elif p <= cand_p * (1 - thr):
                piv_i.append(cand_i); piv_k.append(1); piv_c.append(t)
                direction, cand_i, cand_p = -1, t, p
        else:
            if p <= cand_p:
                cand_i, cand_p = t, p
            elif p >= cand_p * (1 + thr):
                piv_i.append(cand_i); piv_k.append(-1); piv_c.append(t)
                direction, cand_i, cand_p = 1, t, p
    return (np.array(piv_i, dtype=np.int64), np.array(piv_k, dtype=np.int8),
            np.array(piv_c, dtype=np.int64))


def _ema(x, span):
    return pd.Series(x).ewm(span=span).mean().values


def _z(x, w=200):
    """Causal rolling z-score (window w)."""
    s = pd.Series(x)
    m = s.rolling(w, min_periods=w // 4).mean()
    sd = s.rolling(w, min_periods=w // 4).std()
    return ((s - m) / sd.replace(0.0, np.nan)).values


def _atr(high, low, close, w=100):
    prev = np.roll(close, 1); prev[0] = close[0]
    tr = np.maximum(high, prev) - np.minimum(low, prev)
    return pd.Series(tr).rolling(w, min_periods=w // 4).mean().values


def _pivot_view(pi, pk, pc):
    """Incremental causal pivot window: yields (pi_so_far, pk_so_far) per bar."""
    j = 0
    n = len(pc)
    for t in range(int(pc[0]) if n else 0, n + 1):
        while j < n and pc[j] <= t:
            j += 1
        yield pi[:j], pk[:j]


# ---------------- direction votes (vectorized, causal) ----------------

def _v_mtf(close):
    """MTF alignment: mean sign of (close - EMA) across 4/12/24/168 bars."""
    signs = np.zeros((len(close), 4), dtype=np.float32)
    for j, s in enumerate((4, 12, 24, 168)):
        signs[:, j] = np.sign(close - _ema(close, s))
    return signs.mean(axis=1)


def _v_gann(close):
    """Gann 1x1 angle proxy: distance from the 45-degree anchor (EMA168)
    normalized by realized 168-bar travel — above the angle = +, below = -."""
    ema = _ema(close, 168)
    travel = pd.Series(close).pct_change(168).rolling(168, min_periods=42).std().values * np.sqrt(168)
    return np.clip((close - ema) / np.maximum(travel * 2.0, 1e-9), -1.0, 1.0)


def _v_wyckoff(high, low, close, volume):
    """Wyckoff phase proxy: range position + effort skew (up-volume vs
    down-volume share) + spring/upthrust bumps at range edges."""
    lo = pd.Series(low).rolling(300, min_periods=75).min().values
    hi = pd.Series(high).rolling(300, min_periods=75).max().values
    pos = (close - lo) / np.maximum(hi - lo, 1e-9)
    ret = np.zeros_like(close); ret[1:] = np.log(close[1:] / close[:-1])
    up = (ret > 0) * volume; dn = (ret <= 0) * volume
    su = pd.Series(up).rolling(100, min_periods=25).sum().values
    sd = pd.Series(dn).rolling(100, min_periods=25).sum().values
    skew = (su - sd) / np.maximum(su + sd, 1e-9)
    v = 0.7 * (pos - 0.5) * 2.0 + 0.3 * skew
    rng = np.maximum(hi - lo, 1e-9)
    v = np.where((low < lo + 0.05 * rng) & (ret > 0), 1.0, v)   # spring
    v = np.where((high > hi - 0.05 * rng) & (ret < 0), -1.0, v)  # upthrust
    return np.clip(v, -1.0, 1.0)


def _v_vsa(high, low, close, volume):
    """VSA: effort (volume z) vs result (body/ATR) — climax reversal and
    confirmation. effort-with-no-result against the trend flips the vote."""
    eff = _z(volume, 200)
    atr = np.maximum(_atr(high, low, close), 1e-9)
    body = (close - np.roll(open_, 1) * 0 + close - close)  # placeholder
    return None


def _v_vsa(high, low, close, volume, open_):
    """VSA (real): effort = volume z-score; result = body/ATR. Effort with
    no result AGAINST the trend = climax (vote flips); effort confirming the
    trend follows it; low-effort pullback = weak signal toward the trend."""
    eff = np.nan_to_num(_z(volume, 200))
    atr = np.maximum(_atr(high, low, close), 1e-9)
    body = (close - open_) / atr
    trend = np.sign(close - _ema(close, 20))
    v = np.zeros_like(close)
    climax = (np.abs(eff) > 2.0) & (np.abs(body) < 0.3)
    v = np.where(climax & (trend > 0), -0.8, v)
    v = np.where(climax & (trend < 0), 0.8, v)
    conf = (np.abs(eff) > 1.0) & (np.abs(body) >= 0.3)
    v = np.where(conf, 0.6 * trend, v)
    weak = (eff < -0.5) & (body < 0)
    v = np.where(weak & (trend > 0), 0.2, v)
    return np.clip(v, -1.0, 1.0)


def _v_cycles(close, refresh=24):
    """Cycle phase proxy: every `refresh` bars take an FFT of the last 512
    log-returns, find the dominant period in [24, 400], locate the last
    trough, and vote by position in the cycle (buy near trough)."""
    n = len(close)
    v = np.zeros(n, dtype=np.float32)
    lr = np.zeros(n); lr[1:] = np.log(close[1:] / close[:-1])
    last_top = -1.0
    for t0 in range(512, n, refresh):
        w = lr[t0 - 512:t0]
        sp = np.abs(np.fft.rfft(w - w.mean()))
        freqs = np.fft.rfftfreq(512, d=1.0)
        lo, hi = 512 // 400, 512 // 24  # periods 400..24 bars
        k = lo + int(np.argmax(sp[lo:hi])) if hi > lo else lo
        per = 1.0 / max(freqs[k], 1e-9)
        seg = w[-int(per):]
        last_top = float(np.argmin(seg)) + 1.0  # trough offset in period
        pos = (last_top % per) / per
        v[t0:t0 + refresh] = np.cos(2 * np.pi * pos) * -1.0
    return v


# ---------------- pivot-structure votes (causal zigzag) ----------------

def _v_elliott(close, pi, pk, t):
    """Elliott proxy: leg A->B is confirmed; the leg B->now is in motion.
    Progress beyond B (in leg units) scales the vote; a break of B = trend
    invalidation = strong vote against."""
    if len(pi) < 2:
        return 0.0
    pa, pb = close[pi[-2]], close[pi[-1]]
    leg = pb - pa
    if abs(leg) < 1e-9:
        return 0.0
    prog = (close[t] - pb) / leg
    if leg > 0:
        if close[t] < pb:
            return -0.7
        return float(np.clip(prog * 2.0, 0.0, 1.0))
    if close[t] > pb:
        return 0.7
    return float(np.clip(prog * 2.0, -1.0, 0.0))


def _v_classical(close, pi, pk, t):
    """Classical chart patterns proxy: head&shoulders (+inverse), double
    top/bottom — from the last confirmed pivots, neckline confirmed."""
    if len(pi) < 5:
        return 0.0
    hs_ = close[pi[-5:]][pk[-5:] == 1]
    ls_ = close[pi[-5:]][pk[-5:] == -1]
    if len(hs_) >= 3:
        s1, head, s2 = hs_[-3], hs_[-2], hs_[-1]
        if head > s1 * 1.01 and head > s2 * 1.01 and abs(s1 - s2) / head < 0.03:
            if len(ls_) and close[t] < ls_[-1]:
                return -1.0
            return -0.4
    if len(ls_) >= 3:
        s1, head, s2 = ls_[-3], ls_[-2], ls_[-1]
        if head < s1 * 0.99 and head < s2 * 0.99 and abs(s1 - s2) / head < 0.03:
            if len(hs_) and close[t] > hs_[-1]:
                return 1.0
            return 0.4
    if len(hs_) >= 2 and len(ls_) >= 1 and abs(hs_[-1] - hs_[-2]) / hs_[-2] < 0.01:
        if close[t] < ls_[-1]:
            return -0.7
    if len(ls_) >= 2 and len(hs_) >= 1 and abs(ls_[-1] - ls_[-2]) / ls_[-2] < 0.01:
        if close[t] > hs_[-1]:
            return 0.7
    return 0.0


def _v_harmonic(close, pi, pk, t):
    """Harmonic proxy (Gartley/Bat zone): X-A-B-C-D from the last 5 pivots;
    ratio windows decide completion; direction from the final pivot kind."""
    if len(pi) < 5:
        return 0.0
    X, A, B, C, D = close[pi[-5:]]
    xa, ab, bc = abs(A - X), abs(B - A), abs(C - B)
    cd = abs(D - C)
    if xa < 1e-9 or ab < 1e-9 or bc < 1e-9:
        return 0.0
    r1, r2, r3 = ab / xa, bc / ab, cd / bc
    if not (0.382 <= r1 <= 0.886 and 0.382 <= r2 <= 0.886 and 1.13 <= r3 <= 1.618):
        return 0.0
    score = min(1.0, max(0.0, 1.0 - abs(r1 - 0.618) / 0.236)
                * (1.0 - abs(r3 - 1.272) / 0.346))
    return float(score * 0.9) if pk[-1] == -1 else float(-score * 0.9)


def school_votes(open_, high, low, close, volume, thr=CONFIRM):
    """All 8 school votes for one asset. Returns (n, 8) float32 in [-1, 1],
    NaN-free (warmup bars vote 0). CAUSALITY: pivot votes see only pivots
    confirmed at or before each bar."""
    n = len(close)
    out = np.zeros((n, 8), dtype=np.float32)
    out[:, 4] = _v_mtf(close)
    out[:, 2] = _v_gann(close)
    out[:, 1] = _v_wyckoff(high, low, close, volume)
    out[:, 5] = _v_vsa(high, low, close, volume, open_)
    out[:, 3] = _v_cycles(close)
    pi, pk, pc = zigzag(close, thr)
    ell = np.zeros(n); cla = np.zeros(n); har = np.zeros(n)
    j = 0
    for t in range(n):
        while j < len(pc) and pc[j] <= t:
            j += 1
        use_i, use_k = pi[max(0, j - 6):j], pk[max(0, j - 6):j]
        ell[t] = _v_elliott(close, use_i, use_k, t)
        cla[t] = _v_classical(close, use_i, use_k, t)
        har[t] = _v_harmonic(close, use_i, use_k, t)
    out[:, 0] = ell; out[:, 6] = cla; out[:, 7] = har
    return np.nan_to_num(out, nan=0.0).astype(np.float32)
