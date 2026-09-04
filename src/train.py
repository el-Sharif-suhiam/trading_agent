import argparse, json, os, random, time
import numpy as np, torch
from .dataset import AssetDataset
from .model import TradingAgent
from .training.environment import SpotTradingEnv
from .training.ppo import PPO


def make_envs(ds, cfg, update, seed):
    envs = []
    for k in range(cfg["num_envs"]):
        # per-env deterministic RNG so episodes differ across envs but are reproducible
        rng = np.random.default_rng(seed + update * 1000 + k)
        for _ in range(10):
            asset, s = ds.sample(rng, cfg["episode_steps"], cfg["window"])
            if s >= cfg["window"] - 1:
                break
        env = SpotTradingEnv(asset["features"], asset["close"], asset["start"], asset["end"],
                             cfg["window"], cfg["episode_steps"], cfg["fee"], cfg["slippage"],
                             cfg["initial_cash"])
        env.reset(seed=seed + update * 1000 + k, options={"start": s})
        env.i = s
        envs.append(env)
    return envs


def rollout(model, envs, ppo, cfg, device, update):
    """Collect one rollout. Tensors are stacked env-major [N, T, ...] so the
    GRU correctly runs along the time axis during PPO updates."""
    N = len(envs); T = cfg["rollout_steps"]
    obs_t = torch.tensor(np.stack([e._obs() for e in envs]), device=device)
    O = torch.empty(N, T, *obs_t.shape[1:], device=device)
    A = torch.empty(N, T, device=device)
    LP = torch.empty(N, T, device=device)
    R = torch.empty(N, T, device=device)
    V = torch.empty(N, T, device=device)
    D = torch.empty(N, T, device=device)
    h = None
    for t in range(T):
        with torch.no_grad():
            dist, val, h = model(obs_t[:, None], h)
            act = dist.sample().clamp(1e-5, 1 - 1e-5)
            lp = dist.log_prob(act)
        next_obs, rs, dones = [], [], []
        for j, e in enumerate(envs):
            no, r, term, trunc, _ = e.step(np.array([act[j].item()], dtype=np.float32))
            next_obs.append(no); rs.append(r); dones.append(term or trunc)
        O[:, t] = obs_t; A[:, t] = act; LP[:, t] = lp
        R[:, t] = torch.tensor(rs, device=device)
        V[:, t] = val
        D[:, t] = torch.tensor(dones, dtype=torch.float32, device=device)
        obs_t = torch.tensor(np.stack(next_obs), device=device)
        # envs keep stepping even after done inside rollout; SpotTradingEnv clamps at end
    with torch.no_grad():
        _, last_v, _ = model(obs_t[:, None], h)
    return O, A, LP, R, V, D, last_v


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config/default.json")
    p.add_argument("--resume", default="")
    a = p.parse_args()
    cfg = json.load(open(a.config))
    os.makedirs(cfg["model_dir"], exist_ok=True); os.makedirs(cfg["log_dir"], exist_ok=True)

    random.seed(cfg["seed"]); np.random.seed(cfg["seed"]); torch.manual_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = AssetDataset(cfg["processed_dir"], "train", cfg["window"])
    model = TradingAgent(ds.feature_count + 4, cfg["encoder_size"], cfg["hidden_size"], cfg["num_gru_layers"]).to(device)
    ppo = PPO(model, cfg["learning_rate"], cfg["gamma"], cfg["gae_lambda"], cfg["clip_coef"],
              cfg["entropy_coef"], cfg["value_coef"], cfg["max_grad_norm"], cfg["ppo_epochs"],
              cfg["minibatch_size"])
    start = 0
    if a.resume:
        ck = torch.load(a.resume, map_location=device)
        model.load_state_dict(ck["model"]); ppo.opt.load_state_dict(ck["optimizer"])
        start = ck.get("update", 0)
        print("Resumed from update", start)

    print("device", device, "| assets", len(ds.assets), "| features", ds.feature_count, "(+4 portfolio state)")

    for update in range(start, cfg["updates"]):
        t0 = time.time()
        envs = make_envs(ds, cfg, update, cfg["seed"])
        O, A, LP, R, V, D, last_v = rollout(model, envs, ppo, cfg, device, update)
        stats = ppo.update(O, A, LP, R, V, D, last_v)
        final_vals = [e._value() for e in envs]
        trades = sum(e.trades for e in envs)
        msg = (f"update {update+1}/{cfg['updates']} "
               f"portfolio_mean={np.mean(final_vals):.2f} trades={trades} "
               f"sec={time.time()-t0:.1f} {stats}")
        print(msg, flush=True)
        ck = {"model": model.state_dict(), "optimizer": ppo.opt.state_dict(),
              "update": update + 1, "config": cfg}
        torch.save(ck, os.path.join(cfg["model_dir"], "latest.pt"))
        if (update + 1) % 10 == 0:
            torch.save(ck, os.path.join(cfg["model_dir"], f"update_{update+1:05d}.pt"))

    torch.save({"model": model.state_dict(), "optimizer": ppo.opt.state_dict(),
                "update": cfg["updates"], "config": cfg},
               os.path.join(cfg["model_dir"], "latest.pt"))


if __name__ == "__main__": main()
