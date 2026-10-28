import argparse, json
import numpy as np, torch
from ..dataset import AssetDataset
from ..model import TradingAgent
from ..model.ckpt import load_checkpoint
from ..training.environment import SpotTradingEnv
from .fair_eval import run_segments as fair_run_segments

def _start_margin(window):
    """History margin an observation needs: fine window + weekly frame reach
    (168h x 8 rows). The env itself clamps short history safely now, but a
    correct margin keeps eval segments comparable to training starts."""
    return max(window, 96, 168 * 8) - 1


def run(model, asset, cfg, device):
    steps = asset["end"] - asset["start"] - cfg["window"] - 1
    start = asset["start"] + _start_margin(cfg["window"])
    env = SpotTradingEnv(asset["features"], asset["close"], asset["start"], asset["end"],
                         cfg["window"], max(1, steps), cfg["fee"], cfg["slippage"],
                         cfg["initial_cash"])
    obs, _ = env.reset(seed=1, options={"start": start})
    h = None
    done = False
    while not done:
        with torch.no_grad():
            dist, _, h = model(torch.tensor(obs[None, None], device=device), h)
            act = (dist.concentration1 / (dist.concentration1 + dist.concentration0)).item()
        obs, r, t, tr, info = env.step(np.array([act], np.float32))
        done = t or tr
    buyhold = cfg["initial_cash"] * (asset["close"][asset["end"] - 1] / asset["close"][start]) * (1 - cfg["fee"])
    return {"final": info["portfolio_value"], "return_pct": (info["portfolio_value"] / cfg["initial_cash"] - 1) * 100,
            "max_drawdown": info["drawdown"], "trades": info["trades"], "fees": info["fees"],
            "buy_hold": buyhold, "beat_buyhold": bool(info["portfolio_value"] >= buyhold),
            "purity": info.get("purity", 0.0)}


def run_batch(model, assets, cfg, device, batch_size=16):
    """Evaluate several assets in parallel by batching their envs step-by-step.
    Produces identical results to run() (deterministic mean policy, same envs)."""
    envs, starts, remaining = [], [], []
    for asset in assets:
        steps = asset["end"] - asset["start"] - cfg["window"] - 1
        s = asset["start"] + _start_margin(cfg["window"])
        env = SpotTradingEnv(asset["features"], asset["close"], asset["start"], asset["end"],
                             cfg["window"], max(1, steps), cfg["fee"], cfg["slippage"],
                             cfg["initial_cash"])
        env.reset(seed=1, options={"start": s})
        envs.append(env); starts.append(s)
    N = len(envs)
    hs = None
    active = list(range(N))
    infos = [None] * N
    obs = np.stack([e._obs() for e in envs])
    while active:
        with torch.no_grad():
            x = torch.tensor(obs[:, None], device=device)  # [N, 1, W, F+4]
            dist, _, hs = model(x, hs)
            mean = (dist.concentration1 / (dist.concentration1 + dist.concentration0))
            acts = mean.cpu().numpy()
        nxt, still = [], []
        for k, j in enumerate(active):
            no, r, t, tr, info = envs[j].step(np.array([acts[k]], np.float32))
            if t or tr: infos[j] = info
            else: nxt.append(no); still.append(k)  # keep batch position k, not env id
        if nxt:
            obs = np.stack(nxt)
            if hs is not None and len(still) < len(active):
                idx = torch.tensor(still, device=device)
                if isinstance(hs, tuple):
                    stacked, r_cache = hs
                    # stacked GRU hidden: [layers, B, R] -> index dim 1
                    stacked = stacked[:, idx]
                    # reviewer r-cache: [B, past, R] -> index dim 0
                    if r_cache is not None:
                        r_cache = r_cache[idx]
                    hs = (stacked, r_cache)
                else:
                    hs = hs[:, idx]
        active = [active[k] for k in still]
    results = []
    for j, asset in enumerate(assets):
        buyhold = cfg["initial_cash"] * (asset["close"][asset["end"] - 1] / asset["close"][starts[j]]) * (1 - cfg["fee"])
        info = infos[j]
        results.append({"final": info["portfolio_value"],
                        "return_pct": (info["portfolio_value"] / cfg["initial_cash"] - 1) * 100,
                        "max_drawdown": info["drawdown"], "trades": info["trades"],
                        "fees": info["fees"], "buy_hold": buyhold,
                        "beat_buyhold": bool(info["portfolio_value"] >= buyhold),
                        "purity": info.get("purity", 0.0)})
    return results

def classify_regime(close, start, end):
    """Classify an evaluation segment as up/down/sideways from price change."""
    span = end - start
    if span < 10:
        return "sideways"
    first = float(np.mean(close[start:start + span // 10]))
    last = float(np.mean(close[end - span // 10:end]))
    chg = last / max(first, 1e-12) - 1.0
    if chg > 0.10: return "up"
    if chg < -0.10: return "down"
    return "sideways"


def run_regime(model, asset, cfg, device, n_segments=3, seg_steps=512):
    """Evaluate over several random segments of the asset and aggregate by
    regime (up/down/sideways). Returns dict with per-regime return means."""
    rng = np.random.default_rng(7)
    segs = []
    for _ in range(n_segments):
        lo = asset["start"] + _start_margin(cfg["window"])
        hi = asset["end"] - seg_steps - 1
        if hi <= lo:
            continue
        s = int(rng.integers(lo, hi + 1))
        segs.append(s)
    results = {"up": [], "down": [], "sideways": []}
    for s in segs:
        regime = classify_regime(asset["close"], s, s + seg_steps)
        env = SpotTradingEnv(asset["features"], asset["close"], asset["start"], asset["end"],
                             cfg["window"], seg_steps, cfg["fee"], cfg["slippage"],
                             cfg["initial_cash"])
        obs, _ = env.reset(seed=1, options={"start": s})
        h = None; done = False
        while not done:
            with torch.no_grad():
                dist, _, h = model(torch.tensor(obs[None, None], device=device), h)
                act = (dist.concentration1 / (dist.concentration1 + dist.concentration0)).item()
            obs, r, t, tr, info = env.step(np.array([act], np.float32))
            done = t or tr
        results[regime].append(info["portfolio_value"] / cfg["initial_cash"] - 1.0)
    return {k: (float(np.mean(v)) if v else None) for k, v in results.items()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config/default.json")
    p.add_argument("--checkpoint", default="models/latest.pt")
    p.add_argument("--split", default="validation", choices=["validation", "test", "unseen_test"])
    p.add_argument("--regimes", action="store_true", help="also report per-regime (up/down/sideways) results")
    p.add_argument("--mode", default="full", choices=["full", "segments"],
                   help="full = one continuous episode per asset (legacy); "
                        "segments = N×512h fair episodes matching training (RECOMMENDED)")
    a = p.parse_args(); cfg = json.load(open(a.config))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = AssetDataset(cfg["processed_dir"], a.split, cfg["window"])
    model = TradingAgent(ds.feature_count + 5, cfg["encoder_size"], cfg["hidden_size"], cfg["num_gru_layers"]).to(device)
    ck = load_checkpoint(model, a.checkpoint, map_location=device, weights_only=False)
    model.eval()
    if a.mode == "segments":
        res = fair_run_segments(model, ds.assets, cfg, device,
                                batch_size=cfg.get("fair_batch_size", 32))
        agg = res["aggregate"]
        print(f"[fair] {a.checkpoint} | {a.split} | update={ck.get('update')} "
              f"| {agg['n_segments']} segments × {cfg.get('rollout_steps', 512)}h")
        print(f"  alpha mean {agg['alpha_mean_pct']:+.2f}%  median {agg['alpha_median_pct']:+.2f}%  "
              f"t={agg['alpha_t']:+.2f}")
        print(f"  beat buyhold {100*agg['beat_buyhold']:.0f}%  positive {100*agg['positive']:.0f}%  "
              f"purity {agg['purity']:+.3f}  dd {agg['max_drawdown']*100:.1f}%")
        print(f"  model {agg['model_ret_pct']:+.2f}%  buyhold {agg['buyhold_ret_pct']:+.2f}%")
        for pa in res["per_asset"]:
            print(f"    {pa['symbol']:12s} n={pa['n']:2d} alpha={pa['alpha_pct']:+7.2f}% purity={pa['purity']:+.2f}")
        return
    for x in ds.assets:
        print(x["symbol"], run(model, x, cfg, device))
        if a.regimes:
            print("   regimes:", run_regime(model, x, cfg, device))

if __name__ == "__main__": main()
