"""Shared training engine used by both CLI and future harness integrations."""
import csv, json, os, random, time
import numpy as np
import torch

from .dataset import AssetDataset
from .model import TradingAgent
from .training.environment import SpotTradingEnv
from .training.ppo import PPO
from .evaluation.evaluate import run_batch as eval_run_batch


class Trainer:
    def __init__(self, cfg, device=None, resume_path=""):
        self.cfg = cfg
        random.seed(cfg["seed"]); np.random.seed(cfg["seed"]); torch.manual_seed(cfg["seed"])
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # TF32: free 20-40% speedup on Ampere+ GPUs, no meaningful precision
        # loss for RL training (mixed-scale rewards are robust to tf32)
        if self.device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.set_float32_matmul_precision("high")
        os.makedirs(cfg["model_dir"], exist_ok=True)
        os.makedirs(cfg["log_dir"], exist_ok=True)

        self.train_ds = AssetDataset(cfg["processed_dir"], "train", cfg["window"])
        self.val_ds = AssetDataset(cfg["processed_dir"], "validation", cfg["window"])
        in_size = self.train_ds.feature_count + 5
        self.model = TradingAgent(in_size, cfg["encoder_size"], cfg["hidden_size"], cfg["num_gru_layers"]).to(self.device)
        if cfg.get("torch_compile", False) and hasattr(torch, "compile"):
            try:
                self.model = torch.compile(self.model)  # 1.3-2x after warmup
                print("torch.compile enabled (first updates slower — JIT warmup)")
            except Exception as exc:
                print(f"torch.compile unavailable ({exc}); running eager")
        self.base_entropy = cfg["entropy_coef"]
        self.base_lr = cfg["learning_rate"]
        self.ppo = PPO(self.model, cfg["learning_rate"], cfg["gamma"], cfg["gae_lambda"], cfg["clip_coef"],
                       cfg["entropy_coef"], cfg["value_coef"], cfg["max_grad_norm"], cfg["ppo_epochs"],
                       cfg["minibatch_size"], cfg.get("max_kl", 0.02),
                       weight_decay=cfg.get("weight_decay", 0.0))
        self.update_i = 0
        self.best_val = -np.inf
        self.stable_val = -np.inf  # best 10-eval rolling mean (consistency criterion)
        # adaptive learning state: recovers from overfitting stagnation
        # (wave 1 of the agreed anti-overfit plan)
        self.adapt = self._fresh_adapt()
        if resume_path:
            self._load_checkpoint(resume_path)

        self.csv_path = os.path.join(cfg["log_dir"], "metrics.csv")
        if not os.path.exists(self.csv_path) or self.update_i == 0:
            with open(self.csv_path, "w", newline="") as fh:
                csv.writer(fh).writerow(["update", "portfolio_mean", "trades", "reward_mean",
                                         "entropy", "value_loss", "policy_loss", "ratio_clip_frac",
                                         "kl", "sec_per_update", "val_return_pct", "stable_val_return_pct"])

    # ---------- rollout ----------
    def _make_envs(self):
        cfg = self.cfg; envs = []
        crisis_bias = cfg.get("crisis_bias", 0.15)
        for k in range(cfg["num_envs"]):
            rng = np.random.default_rng(cfg["seed"] + self.update_i * 1000 + k)
            asset, s = self.train_ds.sample(rng, cfg["episode_steps"], cfg["window"],
                                            crisis_bias=crisis_bias)
            env = SpotTradingEnv(asset["features"], asset["close"], asset["start"], asset["end"],
                                 cfg["window"], cfg["rollout_steps"], cfg["fee"], cfg["slippage"],
                                 cfg["initial_cash"],
                                 bleed_start=cfg.get("bleed_start", 0.012),
                                 bleed_max=cfg.get("bleed_max", 0.05),
                                 bleed_penalty=cfg.get("bleed_penalty", 0.35),
                                 purity_coef=cfg.get("purity_coef", 0.25),
                                 trend_patience_coef=cfg.get("trend_patience_coef", 0.15),
                                 ecstasy_cap=cfg.get("ecstasy_cap", 0.2),
                                 timing_horizon=cfg.get("timing_horizon", 8),
                                 survival_bonus=cfg.get("survival_bonus", 0.3),
                                 peak_ambition_coef=cfg.get("peak_ambition_coef", 0.5),
                                 obs_noise=cfg.get("obs_noise", 0.0),
                                 obs_dropout=cfg.get("obs_dropout", 0.0))
            env.reset(seed=cfg["seed"] + self.update_i * 1000 + k, options={"start": s})
            envs.append(env)
        return envs

    def _rollout(self, envs):
        cfg = self.cfg
        N = len(envs); T = cfg["rollout_steps"]
        obs_np = SpotTradingEnv.batch_obs(envs)
        obs_t = torch.from_numpy(obs_np).to(self.device, non_blocking=True)
        # rollout buffers live on CPU (pin for fast H2D) — GPU holds only the
        # streaming obs + activations. The 8GB card cannot host the full
        # N×T×230×F buffer plus PPO backward passes at once.
        O = torch.empty(N, T, *obs_t.shape[1:], device="cpu", pin_memory=True)
        A = torch.empty(N, T, device="cpu", pin_memory=True)
        LP = torch.empty(N, T, device="cpu", pin_memory=True)
        R = torch.empty(N, T, device="cpu", pin_memory=True)
        V = torch.empty(N, T, device="cpu", pin_memory=True)
        D = torch.empty(N, T, device="cpu", pin_memory=True)
        h = None
        for t in range(T):
            with torch.no_grad():
                dist, val, h = self.model(obs_t[:, None], h)
                act = dist.sample().clamp(1e-5, 1 - 1e-5)
                lp = dist.log_prob(act)
            acts_np = act.cpu().numpy()
            # obs for the NEXT step comes from the env returns directly —
            # no second batch_obs pass (halves obs-building work per step)
            next_obs, rs, dones = [], [], []
            for j, e in enumerate(envs):
                no, r, term, trunc, _ = e.step(np.array([acts_np[j]], dtype=np.float32))
                next_obs.append(no); rs.append(r); dones.append(term or trunc)
            O[:, t] = obs_t.cpu(); A[:, t] = act.cpu(); LP[:, t] = lp.cpu()
            R[:, t] = torch.tensor(rs)
            V[:, t] = val.cpu()
            D[:, t] = torch.tensor(dones, dtype=torch.float32)
            obs_np = np.stack(next_obs)  # envs always share shape/dtype
            obs_t = torch.from_numpy(np.ascontiguousarray(obs_np)).to(self.device, non_blocking=True)
        with torch.no_grad():
            _, last_v, _ = self.model(obs_t[:, None], h)
        return O, A, LP, R, V, D, last_v

    @staticmethod
    def _fresh_adapt():
        return {"lr_scale": 1.0, "ent_scale": 1.0, "cooldown": 0}

    # ---------- validation ----------
    def validate(self):
        """Returns (mean_return_pct, mean_purity). Purity = (G-L)/(G+L) averaged
        over assets — the user's doctrine: gains must dwarf losses."""
        # keep the model deterministic for exams (defensive: obs_noise/dropout
        # live in the ENV, which eval envs construct with 0.0 — this also pins
        # any future stochastic layer to stay out of grading).
        was_training = self.model.training
        self.model.eval()
        try:
            rets = eval_run_batch(self.model, self.val_ds.assets, self.cfg, self.device)
        finally:
            if was_training:
                self.model.train()
        # purity comes from env info; run_batch exposes it via final infos
        pur = [r.get("purity", 0.0) for r in rets]
        return (float(np.mean([r["return_pct"] for r in rets])),
                float(np.mean(pur)) if pur else 0.0)

    # ---------- checkpoints ----------
    def _ckpt(self):
        return {"model": self.model.state_dict(), "optimizer": self.ppo.opt.state_dict(),
                "update": self.update_i, "best_val": self.best_val,
                "stable_val": self.stable_val, "config": self.cfg,
                "adapt": self.adapt,
                "feature_count": self.train_ds.feature_count,
                "state_dim": self.train_ds.feature_count + 5}

    def _save(self, name):
        torch.save(self._ckpt(), os.path.join(self.cfg["model_dir"], name))

    def _save_if_due(self, last_time):
        if time.time() - last_time >= self.cfg.get("checkpoint_minutes", 5) * 60:
            self._save("latest.pt")
            return time.time()
        return last_time

    def _load_checkpoint(self, path):
        ck = torch.load(path, map_location=self.device)
        state = ck["model"]
        # strip torch.compile's _orig_mod. prefix if present (checkpoint
        # may come from a compiled or eager session)
        if any(k.startswith("_orig_mod.") for k in state):
            state = {k.replace("_orig_mod.", "", 1): v for k, v in state.items()}
        # or add it if the running model is compiled
        if any(not k.startswith("_orig_mod.") for k in state) and hasattr(self.model, "_orig_mod"):
            state = {f"_orig_mod.{k}": v for k, v in state.items()}
        self.model.load_state_dict(state)
        if "optimizer" in ck:  # distilled checkpoints have no optimizer state
            self.ppo.opt.load_state_dict(ck["optimizer"])
            # Adam.load_state_dict restores the SAVED param_groups, which for
            # checkpoints trained before weight_decay held 0.0 — re-assert the
            # configured value or the new regularizer would silently never fire.
            for g in self.ppo.opt.param_groups:
                g["weight_decay"] = self.cfg.get("weight_decay", 0.0)
        self.update_i = ck.get("update", 0)
        self.best_val = ck.get("best_val", -np.inf)
        self.stable_val = ck.get("stable_val", -np.inf)
        self.adapt = ck.get("adapt", self.adapt)
        if self.cfg.get("adapt_reset_on_resume", False):
            # One-time hard reset of the ADAPT hyper-parameter state (LR/entropy
            # scales) — NOT the learned weights. Used when a new regularization
            # regime begins (obs_noise/dropout/weight_decay, wave 2.5): the old
            # adaptive state was stuck at LR-floor ×1/16 + entropy-cap ×4 from a
            # memorization era; letting it carry over would starve the fresh
            # regularized training of learning rate. `stable`/`best` metrics are
            # preserved (doctrine: continuity over peaks).
            self.adapt = self._fresh_adapt()
            print("  [adapt] hyper-param state reset on resume (fresh LR/entropy)",
                  flush=True)
        print(f"resumed from {path} at update {self.update_i} "
              f"(best {self.best_val:.2f}%, stable {self.stable_val if self.stable_val > -np.inf else 0:.2f}%)")

    # ---------- adaptive learning (anti-overfitting, wave 1) ----------
    def _adapt_on_eval(self, val_hist):
        """Wave 1 of the agreed plan: when the mean of the last 5 evals is
        DECLINING vs the previous 5-eval window, the model is likely memorizing
        instead of learning. Response (bounded, reversible):
          - cut LR (halve, floor 1/16 of base)   -> smaller memorization steps
          - boost entropy coefficient (x2, cap x4) -> stronger exploration push
        When evals recover, scales gradually return to 1.0.
        Never stops training (sacred rule: no early stop)."""
        cfg = self.cfg
        window = cfg.get("adapt_window", 5)
        if len(val_hist) < 2 * window:
            return
        recent = float(np.mean(val_hist[-window:]))
        prev = float(np.mean(val_hist[-2*window:-window]))
        if self.adapt["cooldown"] > 0:
            self.adapt["cooldown"] -= 1
            return
        if recent < prev - 0.5:  # meaningful decline (>=0.5pp)
            self.adapt["lr_scale"] = max(self.adapt["lr_scale"] * 0.5, 1/16)
            self.adapt["ent_scale"] = min(self.adapt["ent_scale"] * 2.0, 4.0)
            self.adapt["cooldown"] = cfg.get("adapt_cooldown", 10)  # evals before next change
            print(f"  [adapt] eval decline detected ({prev:.1f}→{recent:.1f}%): "
                  f"lr_scale={self.adapt['lr_scale']:.3f} ent_scale={self.adapt['ent_scale']:.1f}", flush=True)
        elif recent > prev + 0.5:  # recovering: gently relax toward normal
            self.adapt["lr_scale"] = min(self.adapt["lr_scale"] * 2.0, 1.0)
            self.adapt["ent_scale"] = max(self.adapt["ent_scale"] * 0.5, 1.0)
            self.adapt["cooldown"] = cfg.get("adapt_cooldown", 10)
        # entropy coefficient applied to PPO for subsequent updates
        self.ppo.entropy_coef = self.base_entropy * self.adapt["ent_scale"]

    # ---------- main loop ----------
    def train(self, time_budget_hours=None):
        cfg = self.cfg
        stop_file = os.path.join(cfg["log_dir"], "STOP")
        if os.path.exists(stop_file):
            os.remove(stop_file)  # clear stale stop request from previous session
        budget = time_budget_hours if time_budget_hours is not None else cfg.get("time_budget_hours", 0)
        deadline = time.time() + budget * 3600 if budget > 0 else None
        last_ckpt = time.time()
        val_hist = []  # recent validation returns (consistency tracking)
        print(f"device={self.device} assets={len(self.train_ds.assets)} "
              f"features={self.train_ds.feature_count}+4 updates={cfg['updates']} "
              f"budget={budget or 'inf'}h  (touch {stop_file} to stop safely)")
        while self.update_i < cfg["updates"]:
            if os.path.exists(stop_file):
                print(f"stop requested via {stop_file}; saving and exiting cleanly")
                break
            t0 = time.time()
            # free previous cycle's buffers BEFORE building new ones —
            # pinned rollout tensors (1.5GB) held through rebind caused
            # double-buffer RAM spikes and slower cycles
            O = A = LP = R = V = D = None
            envs = None
            import gc as _gc; _gc.collect()
            envs = self._make_envs()
            O, A, LP, R, V, D, last_v = self._rollout(envs)
            stats = self.ppo.update(O, A, LP, R, V, D, last_v)
            self.update_i += 1
            # LR schedule combined with adaptive scale (wave 1 anti-overfit)
            scheduled = self._lr(cfg["learning_rate"], cfg.get("lr_final_factor", .1),
                                 self.update_i / cfg["updates"])
            self.ppo.set_lr(scheduled * self.adapt["lr_scale"])
            vals = [e._value() for e in envs]
            trades = sum(e.trades for e in envs)
            row = [self.update_i, round(np.mean(vals), 2), trades, round(R.mean().item(), 6),
                   stats.get("entropy"), stats.get("value_loss"), stats.get("policy_loss"),
                   stats.get("ratio_clip_frac"), stats.get("approx_kl"),
                   round(time.time() - t0, 1), "",
                   self.stable_val if self.stable_val > -np.inf else ""]
            # ---------- periodic validation / best checkpoint ----------
            if self.update_i % cfg.get("eval_every", 25) == 0:
                vret, vpurity = self.validate()
                row[10] = round(vret, 3)
                val_hist.append(vret)
                if vret > self.best_val:
                    self.best_val = vret; self._save("best.pt")
                # consistency criterion: mean of last 10 evals.
                # A model that scores ~the same every exam is worth far more
                # than one that hit a lucky peak once.
                if len(val_hist) >= 10:
                    m = float(np.mean(val_hist[-10:]))
                    # doctrine-weighted score: return must come with purity
                    # (gains dwarfing losses). Purity in [-1,1] boosts or
                    # discounts the raw return mean.
                    score = m * (0.5 + 0.5 * max(0.0, vpurity))
                    if score > self.stable_val:
                        self.stable_val = score
                        self._save("stable.pt")
                # adaptive learning: react to decline/recovery (wave 1)
                self._adapt_on_eval(val_hist)
                # NOTE: no early stop. The system NEVER decides it cannot
                # improve — it saves its best state and keeps trying with
                # everything it has (user's explicit requirement).
            with open(self.csv_path, "a", newline="") as fh:
                csv.writer(fh).writerow(row)
            print(f"update {self.update_i}/{cfg['updates']} portfolio={row[1]:.0f} "
                  f"val={row[10] if row[10] != '' else '-'} best={self.best_val:.2f}% "
                  f"stable={self.stable_val:.2f}% "
                  f"ent={stats.get('entropy')} kl={stats.get('approx_kl')} {row[9]}s", flush=True)
            last_ckpt = self._save_if_due(last_ckpt)
            if deadline and time.time() > deadline:
                print(f"time budget of {budget}h reached at update {self.update_i}; stopping")
                break
            if os.path.exists(stop_file):
                print("stop requested; saving and exiting cleanly")
                break
        self._save("latest.pt")
        os.path.exists(stop_file) and os.remove(stop_file)
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
