"""W3 REWARD AUDIT — is money the dominant signal, or are shaping terms?

Decomposes the per-step reward of the CURRENT policy into its doctrine
component list (base / bleed / purity / patience / inactivity(cash_drag) /
ambition / ecstasy / timing / survival) over a real rollout from the train
dataset with the given checkpoint.

This is the permanent guard against the measured root cause of the stale
training: the bleed penalty used to carry ~22x the variance of the money
term (−0.138/step vs money std 0.0074), so PPO optimized drawdown-avoidance
instead of profit. Config recalibration (bleed 0.35→0.004, cash_drag 0.002,
timing ×0.25) must flip that — verify here every time a coefficient moves.

Exit code 0 when money dominates, 1 otherwise (CI-friendly).

Usage: python -m src.evaluation.reward_audit [--checkpoint models/latest.pt]
"""
import argparse, json, sys
import numpy as np
import torch

from ..dataset import AssetDataset
from ..model import TradingAgent
from ..model.ckpt import load_checkpoint
from ..training.environment import SpotTradingEnv


def audit(model, cfg, train_ds, device, n_envs=16, steps=256):
    """Roll out the current policy on sampled train episodes; log every
    doctrine term per step via wrapped env methods. Returns dict of stats."""
    envs = []
    for k in range(n_envs):
        rng = np.random.default_rng(1000 + k)
        asset, s = train_ds.sample(rng, cfg["rollout_steps"], cfg["window"],
                                   crisis_bias=cfg.get("crisis_bias", 0.15))
        e = SpotTradingEnv(asset["features"], asset["close"], asset["start"], asset["end"],
                           cfg["window"], cfg["rollout_steps"], cfg["fee"], cfg["slippage"],
                           cfg["initial_cash"],
                           bleed_start=cfg.get("bleed_start", .012), bleed_max=cfg.get("bleed_max", .05),
                           bleed_penalty=cfg.get("bleed_penalty", .004),
                           purity_coef=cfg.get("purity_coef", .25),
                           trend_patience_coef=cfg.get("trend_patience_coef", .15),
                           ecstasy_cap=cfg.get("ecstasy_cap", .2),
                           timing_horizon=cfg.get("timing_horizon", 8),
                           survival_bonus=cfg.get("survival_bonus", .3),
                           peak_ambition_coef=cfg.get("peak_ambition_coef", .5),
                           obs_noise=cfg.get("obs_noise", 0.0),
                           obs_dropout=cfg.get("obs_dropout", 0.0),
                           cash_drag=cfg.get("cash_drag", 0.0),
                           timing_scale=cfg.get("timing_scale", 1.0),
                           ecstasy_scale=cfg.get("ecstasy_scale", 1.0))
        e.reset(seed=1, options={"start": s})
        envs.append(e)
    TERM = {"bleed": "_bleed_penalty", "purity": "_purity_term",
            "patience": "_trend_patience", "inactivity": "_inactivity_bias",
            "ambition": "_peak_ambition", "ecstasy": "_ecstasy",
            "timing": "_resolve_timing", "survival": "_survival_check"}
    ACC = {k: [] for k in ["base", *TERM]}
    for e in envs:  # wrap methods to record their per-step outputs
        for name, attr in TERM.items():
            fn = getattr(e, attr)
            def mk(fn, name):
                def g(*ar, **kw):
                    r = fn(*ar, **kw); ACC[name].append(r); return r
                return g
            setattr(e, attr, mk(fn, name))
    obs = np.stack([e._obs() for e in envs])
    hs = None
    done = [False] * n_envs
    for _loop in range(steps):
        with torch.no_grad():
            dist, _, hs = model(torch.tensor(obs[:, None], device=device), hs)
            acts = (dist.concentration1 / (dist.concentration1 + dist.concentration0)).cpu().numpy()
        nxt = []; any_left = False
        for j, e in enumerate(envs):
            if done[j]:
                continue
            v0 = e._value()
            no, r, t, tr, _ = e.step(np.array([acts[j]], np.float32))
            ACC["base"].append(np.log(max(e._value(), 1e-12) / max(v0, 1e-12)))
            _ = r  # per-term values already recorded by wraps; money term recomputed
            if t or tr:
                done[j] = True
            else:
                nxt.append(no); any_left = True
        if not any_left:
            break
        obs = np.stack(nxt)
        if hs is not None:
            live = np.array([j for j in range(n_envs) if not done[j]])
            if isinstance(hs, tuple):
                st, rc = hs; hs = (st[:, live], rc[live])
            else:
                hs = hs[:, live]
    stats = {}
    for k, v in ACC.items():
        v = np.array(v, dtype=np.float64)
        stats[k] = {"mean": float(v.mean()), "std": float(v.std()),
                    "absmean": float(np.abs(v).mean())}
    total_std = np.sqrt(sum(stats[k]["std"] ** 2 for k in stats if k != "base"))
    money = stats["base"]
    stats["_summary"] = {
        "money_std_share": float(money["std"] ** 2 / (money["std"] ** 2 + total_std ** 2 + 1e-30)),
        "dominance_ok": bool(all(money["std"] >= 2.0 * stats[k]["std"] for k in stats
                                 if k not in ("base", "_summary")))}
    return stats


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config/default.json")
    p.add_argument("--checkpoint", default="models/latest.pt")
    a = p.parse_args()
    cfg = json.load(open(a.config))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = AssetDataset(cfg["processed_dir"], "train", cfg["window"])
    model = TradingAgent(ds.feature_count + 5, cfg["encoder_size"], cfg["hidden_size"],
                         cfg["num_gru_layers"]).to(device)
    ck = load_checkpoint(model, a.checkpoint, map_location=device, weights_only=False)
    model.eval()
    st = audit(model, cfg, ds, device,
               n_envs=cfg["num_envs"], steps=min(cfg["rollout_steps"], 512))
    print(f"[reward audit] {a.checkpoint} | update={ck.get('update')}")
    print(f"{'term':11s} {'mean/step':>12s} {'std':>10s} {'abs mean':>10s}")
    for k, v in st.items():
        if not isinstance(v, dict) or k == "_summary":
            continue
        print(f"{k:11s} {v['mean']:+12.6f} {v['std']:10.6f} {v['absmean']:10.6f}")
    s = st["_summary"]
    print(f"\nmoney-term variance share: {s['money_std_share']*100:.1f}%")
    print(f"dominance check (money std >= 2x every non-money term): "
          f"{'PASS' if s['dominance_ok'] else 'FAIL'}")
    sys.exit(0 if s["dominance_ok"] else 1)


if __name__ == "__main__":
    main()
