"""Shared training engine used by both CLI and future harness integrations."""
import csv, json, os, random, time
import numpy as np
import torch

from .dataset import AssetDataset
from .model import TradingAgent
from .training.environment import SpotTradingEnv
from .training.ppo import PPO
from .evaluation.evaluate import run as eval_run


class Trainer:
    def __init__(self, cfg, device=None, resume_path=""):
        self.cfg = cfg
        random.seed(cfg["seed"]); np.random.seed(cfg["seed"]); torch.manual_seed(cfg["seed"])
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        os.makedirs(cfg["model_dir"], exist_ok=True)
        os.makedirs(cfg["log_dir"], exist_ok=True)

        self.train_ds = AssetDataset(cfg["processed_dir"], "train", cfg["window"])
        self.val_ds = AssetDataset(cfg["processed_dir"], "validation", cfg["window"])
        in_size = self.train_ds.feature_count + 4
        self.model = TradingAgent(in_size, cfg["encoder_size"], cfg["hidden_size"], cfg["num_gru_layers"]).to(self.device)
        self.ppo = PPO(self.model, cfg["learning_rate"], cfg["gamma"], cfg["gae_lambda"], cfg["clip_coef"],
                       cfg["entropy_coef"], cfg["value_coef"], cfg["max_grad_norm"], cfg["ppo_epochs"],
                       cfg["minibatch_size"], cfg.get("max_kl", 0.02))
        self.update_i = 0
        self.best_val = -np.inf
        if resume_path:
            self._load_checkpoint(resume_path)

        self.csv_path = os.path.join(cfg["log_dir"], "metrics.csv")
        if not os.path.exists(self.csv_path) or self.update_i == 0:
            with open(self.csv_path, "w", newline="") as fh:
                csv.writer(fh).writerow(["update", "portfolio_mean", "trades", "reward_mean",
                                         "entropy", "value_loss", "policy_loss", "ratio_clip_frac",
                                         "kl", "sec_per_update", "val_return_pct", "best_val_return_pct"])

    # ---------- rollout ----------
    def _make_envs(self):
        cfg = self.cfg; envs = []
        for k in range(cfg["num_envs"]):
            rng = np.random.default_rng(cfg["seed"] + self.update_i * 1000 + k)
            asset, s = self.train_ds.sample(rng, cfg["episode_steps"], cfg["window"])
            env = SpotTradingEnv(asset["features"], asset["close"], asset["start"], asset["end"],
                                 cfg["window"], cfg["rollout_steps"], cfg["fee"], cfg["slippage"],
                                 cfg["initial_cash"])
            env.reset(seed=cfg["seed"] + self.update_i * 1000 + k, options={"start": s})
            envs.append(env)
        return envs

    def _rollout(self, envs):
        cfg = self.cfg
        N = len(envs); T = cfg["rollout_steps"]
        obs_t = torch.tensor(np.stack([e._obs() for e in envs]), device=self.device)
        O = torch.empty(N, T, *obs_t.shape[1:], device=self.device)
        A = torch.empty(N, T, device=self.device)
        LP = torch.empty(N, T, device=self.device)
        R = torch.empty(N, T, device=self.device)
        V = torch.empty(N, T, device=self.device)
        D = torch.empty(N, T, device=self.device)
        h = None
        for t in range(T):
            with torch.no_grad():
                dist, val, h = self.model(obs_t[:, None], h)
                act = dist.sample().clamp(1e-5, 1 - 1e-5)
                lp = dist.log_prob(act)
            next_obs, rs, dones = [], [], []
            for j, e in enumerate(envs):
                no, r, term, trunc, _ = e.step(np.array([act[j].item()], dtype=np.float32))
                next_obs.append(no); rs.append(r); dones.append(term or trunc)
            O[:, t] = obs_t; A[:, t] = act; LP[:, t] = lp
            R[:, t] = torch.tensor(rs, device=self.device)
            V[:, t] = val
            D[:, t] = torch.tensor(dones, dtype=torch.float32, device=self.device)
            obs_t = torch.tensor(np.stack(next_obs), device=self.device)
        with torch.no_grad():
            _, last_v, _ = self.model(obs_t[:, None], h)
        return O, A, LP, R, V, D, last_v

    # ---------- validation ----------
    def validate(self):
        rets = [eval_run(self.model, x, self.cfg, self.device) for x in self.val_ds.assets]
        return float(np.mean([r["return_pct"] for r in rets]))

    # ---------- checkpoints ----------
    def _ckpt(self):
        return {"model": self.model.state_dict(), "optimizer": self.ppo.opt.state_dict(),
                "update": self.update_i, "best_val": self.best_val, "config": self.cfg}

    def _save(self, name):
        torch.save(self._ckpt(), os.path.join(self.cfg["model_dir"], name))

    def _save_if_due(self, last_time):
        if time.time() - last_time >= self.cfg.get("checkpoint_minutes", 5) * 60:
            self._save("latest.pt")
            return time.time()
        return last_time

    def _load_checkpoint(self, path):
        ck = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ck["model"])
        self.ppo.opt.load_state_dict(ck["optimizer"])
        self.update_i = ck.get("update", 0)
        self.best_val = ck.get("best_val", -np.inf)
        print(f"resumed from {path} at update {self.update_i}")

    # ---------- main loop ----------
    def train(self, time_budget_hours=None):
        cfg = self.cfg
        budget = time_budget_hours if time_budget_hours is not None else cfg.get("time_budget_hours", 0)
        deadline = time.time() + budget * 3600 if budget > 0 else None
        last_ckpt = time.time()
        patience = cfg.get("patience_updates", 0)
        stale = 0
        print(f"device={self.device} assets={len(self.train_ds.assets)} "
              f"features={self.train_ds.feature_count}+4 updates={cfg['updates']} "
              f"budget={budget or 'inf'}h")
        while self.update_i < cfg["updates"]:
            t0 = time.time()
            envs = self._make_envs()
            O, A, LP, R, V, D, last_v = self._rollout(envs)
            stats = self.ppo.update(O, A, LP, R, V, D, last_v)
            self.update_i += 1
            self.ppo.set_lr(self._lr(cfg["learning_rate"], cfg.get("lr_final_factor", .1),
                                     self.update_i / cfg["updates"]))
            vals = [e._value() for e in envs]
            trades = sum(e.trades for e in envs)
            row = [self.update_i, round(np.mean(vals), 2), trades, round(R.mean().item(), 6),
                   stats.get("entropy"), stats.get("value_loss"), stats.get("policy_loss"),
                   stats.get("ratio_clip_frac"), stats.get("approx_kl"),
                   round(time.time() - t0, 1), "", self.best_val if self.best_val > -np.inf else ""]
            # ---------- periodic validation / best checkpoint ----------
            if self.update_i % cfg.get("eval_every", 25) == 0:
                vret = self.validate()
                row[10] = round(vret, 3)
                if vret > self.best_val:
                    self.best_val = vret; stale = 0
                    self._save("best.pt")
                else:
                    stale += cfg.get("eval_every", 25)
                if patience and stale >= patience:
                    print(f"early stop: no val improvement in {stale} updates (best {self.best_val:.2f}%)")
                    row and None
            with open(self.csv_path, "a", newline="") as fh:
                csv.writer(fh).writerow(row)
            print(f"update {self.update_i}/{cfg['updates']} portfolio={row[1]:.0f} "
                  f"val={row[10] if row[10] != '' else '-'} best={self.best_val:.2f}% "
                  f"ent={stats.get('entropy')} kl={stats.get('approx_kl')} {row[9]}s", flush=True)
            last_ckpt = self._save_if_due(last_ckpt)
            if deadline and time.time() > deadline:
                print(f"time budget of {budget}h reached at update {self.update_i}; stopping")
                break
        self._save("latest.pt")
        return self.update_i

    def _lr(self, base, final_factor, frac):
        return base * (1 + (final_factor - 1) * frac)


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config/default.json")
    p.add_argument("--resume", default="")
    p.add_argument("--time-budget-hours", type=float, default=None)
    a = p.parse_args()
    cfg = json.load(open(a.config))
    Trainer(cfg, resume_path=a.resume).train(time_budget_hours=a.time_budget_hours)


if __name__ == "__main__":
    main()
