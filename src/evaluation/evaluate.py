import argparse, json
import numpy as np, torch
from ..dataset import AssetDataset
from ..model import TradingAgent
from ..training.environment import SpotTradingEnv

def run(model, asset, cfg, device):
    steps = asset["end"] - asset["start"] - cfg["window"] - 1
    start = asset["start"] + cfg["window"] - 1
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
            "buy_hold": buyhold, "beat_buyhold": bool(info["portfolio_value"] >= buyhold)}


def run_batch(model, assets, cfg, device, batch_size=16):
    """Evaluate several assets in parallel by batching their envs step-by-step.
    Produces identical results to run() (deterministic mean policy, same envs)."""
    envs, starts, remaining = [], [], []
    for asset in assets:
        steps = asset["end"] - asset["start"] - cfg["window"] - 1
        s = asset["start"] + cfg["window"] - 1
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
                    hs = tuple(h[:, idx] for h in hs)
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
                        "beat_buyhold": bool(info["portfolio_value"] >= buyhold)})
    return results

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config/default.json")
    p.add_argument("--checkpoint", default="models/latest.pt")
    p.add_argument("--split", default="validation", choices=["validation", "test", "unseen_test"])
    a = p.parse_args(); cfg = json.load(open(a.config))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = AssetDataset(cfg["processed_dir"], a.split, cfg["window"])
    model = TradingAgent(ds.feature_count + 4, cfg["encoder_size"], cfg["hidden_size"], cfg["num_gru_layers"]).to(device)
    ck = torch.load(a.checkpoint, map_location=device)
    model.load_state_dict(ck["model"]); model.eval()
    for x in ds.assets: print(x["symbol"], run(model, x, cfg, device))

if __name__ == "__main__": main()
