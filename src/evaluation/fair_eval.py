"""FAIR evaluation: episodes shaped exactly like training.

Why this module exists (measured, not assumed): the old validation ran ONE
continuous ~8,000-step episode per asset with the GRU hidden state carried
for a year, while training is 512-step episodes with a reset hidden state.
`best`/`stable` were therefore graded on a distribution the model never
trains on — the consistency metric froze at an update-550 value for 27,000
updates while nobody could tell what the model actually learned.

Fair protocol:
  - N random segments of `seg_steps` (default = rollout_steps = 512) per asset
  - same start margin as training (weekly frame needs 168×8 = 1344 candles)
  - reset from cash, like every training episode
  - alpha = model return MINUS buy-and-hold return on the SAME candles
    (raw return only rewards riding a bull market; alpha isolates skill)
  - deterministic given the seed; batch size must not change results
    (hidden-state slicing is tested bit-for-bit across batch sizes)
"""

import numpy as np
import torch

from ..training.environment import SpotTradingEnv

MARGIN = 1344  # deepest historical reach of the weekly coarse frame


def _start_margin(window):
    return max(window, 96, 168 * 8) - 1


def sample_starts(assets, cfg, seg_steps, per_asset, seed):
    """Deterministic distinct segment starts per asset (respects eval margin)."""
    rng = np.random.default_rng(seed)
    jobs = []
    for a in assets:
        margin = max(_start_margin(cfg["window"]), MARGIN)
        lo = a["start"] + margin
        hi = a["end"] - seg_steps - 1
        if hi <= lo:
            continue
        ks = min(per_asset, hi - lo + 1)
        for s in rng.choice(np.arange(lo, hi + 1), size=ks, replace=False):
            jobs.append((a, int(s)))
    return jobs


def _run_batched(model, jobs, cfg, device, seg_steps, batch_size, ablate_context=False):
    """Run the segment jobs in batches with a carried hidden state per batch
    (the proven run_batch pattern). Results are batch-size invariant."""
    out = []
    B = max(1, batch_size)
    fine = cfg["window"]
    ctx_from = fine + 32  # 24 (4h) + 8 (12h) rows precede the daily block
    for b0 in range(0, len(jobs), B):
        chunk = jobs[b0:b0 + B]
        envs = []
        for a, s in chunk:
            e = SpotTradingEnv(a["features"], a["close"], s, a["end"], cfg["window"],
                               seg_steps, cfg["fee"], cfg["slippage"], cfg["initial_cash"])
            e.reset(seed=1, options={"start": s})
            envs.append(e)
        obs = np.stack([e._obs() for e in envs])
        if ablate_context:
            obs = obs.copy()
            obs[:, ctx_from:, :envs[0].features.shape[1]] = 0.0
        hs = None
        active = list(range(len(envs)))
        infos = [None] * len(envs)
        while active:
            with torch.no_grad():
                dist, _, hs = model(torch.tensor(obs[:, None], device=device), hs)
                acts = (dist.concentration1 / (dist.concentration1 + dist.concentration0)).cpu().numpy()
            nxt, still = [], []
            for k, j in enumerate(active):
                no, r, t, tr, info = envs[j].step(np.array([acts[k]], np.float32))
                if t or tr:
                    infos[j] = info
                else:
                    nxt.append(no)
                    still.append(k)
            if nxt:
                obs = np.stack(nxt)
                if ablate_context:
                    obs = obs.copy()
                    obs[:, ctx_from:, :envs[0].features.shape[1]] = 0.0
                if hs is not None and len(still) < len(active):
                    idx = torch.tensor(still, device=device)
                    if isinstance(hs, tuple):
                        stacked, r_cache = hs
                        stacked = stacked[:, idx]
                        r_cache = r_cache[idx] if r_cache is not None else None
                        hs = (stacked, r_cache)
                    else:
                        hs = hs[:, idx]
            active = [active[k] for k in still]
        for (a, s), info in zip(chunk, infos):
            bh = a["close"][s + seg_steps] / a["close"][s] * (1 - cfg["fee"]) - 1.0
            ret = info["portfolio_value"] / cfg["initial_cash"] - 1.0
            out.append({"symbol": a["symbol"], "start": s,
                        "model_ret": ret, "bh_ret": bh, "alpha": ret - bh,
                        "purity": info.get("purity", 0.0), "dd": info["drawdown"],
                        "trades": info["trades"], "fees": info["fees"]})
    return out


def run_segments(model, assets, cfg, device, seg_steps=None, per_asset=None,
                 seed=0, batch_size=32, ablate_context=False):
    """Evaluate `per_asset` random segments per asset.
    Returns {"aggregate": {...}, "per_asset": [per-symbol means], "segments": [rows]}"""
    seg_steps = int(seg_steps or cfg.get("rollout_steps", 512))
    per_asset = int(per_asset or cfg.get("fair_per_asset", 6))
    jobs = sample_starts(assets, cfg, seg_steps, per_asset, seed)
    if not jobs:
        raise ValueError("no evaluable segment: assets too short for the frame margin")
    rows = _run_batched(model, jobs, cfg, device, seg_steps, batch_size, ablate_context)
    al = np.array([r["alpha"] for r in rows]) * 100.0
    mr = np.array([r["model_ret"] for r in rows]) * 100.0
    bh = np.array([r["bh_ret"] for r in rows]) * 100.0
    pur = np.array([r["purity"] for r in rows])
    dd = np.array([r["dd"] for r in rows])
    n = len(al)
    t = float(al.mean() / (al.std() / np.sqrt(n))) if n > 2 and al.std() > 0 else 0.0
    agg = {"alpha_mean_pct": float(al.mean()), "alpha_median_pct": float(np.median(al)),
           "alpha_std_pct": float(al.std()), "alpha_t": t, "n_segments": n,
           "beat_buyhold": float((al > 0).mean()), "positive": float((mr > 0).mean()),
           "model_ret_pct": float(mr.mean()), "buyhold_ret_pct": float(bh.mean()),
           "purity": float(pur.mean()), "max_drawdown": float(dd.mean())}
    per_asset_out = []
    for sym in sorted({r["symbol"] for r in rows}):
        sel = [r for r in rows if r["symbol"] == sym]
        sa = np.array([r["alpha"] for r in sel]) * 100.0
        sp = np.array([r["purity"] for r in sel])
        per_asset_out.append({"symbol": sym, "n": len(sel), "alpha_pct": float(sa.mean()),
                              "purity": float(sp.mean())})
    return {"aggregate": agg, "per_asset": per_asset_out, "segments": rows}


def stable_score(alphas, purities):
    """Doctrine-weighted consistency score over the last 10 evals:
    mean alpha is discounted (never boosted) by purity — a model whose gains
    dwarf its losses scores full alpha; a bleeding model is worth half."""
    if len(alphas) < 10:
        return None
    m = float(np.mean(alphas[-10:]))
    p = float(np.mean(purities[-10:]))
    return m * (0.5 + 0.5 * max(0.0, p))
