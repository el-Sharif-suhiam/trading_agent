"""Distillation: the current (v2) genius teaches the new (v3) brain.

Why: sacred rule #12 — never retrain from scratch. The v2 model holds
thousands of updates of market understanding; distillation transfers it
by matching the new model's action distributions and value estimates to
the teacher's on thousands of real windows, BEFORE the new model starts
its own PPO training. Nothing is lost; the successor starts as a young
genius with an elder's memory.

v3 upgrades over v2 (reviewer + context ladder):
  - Reviewer unit (second thought): GRU 256 + self-attention, zero-init
    residual — silent at birth, earns influence only if it helps.
  - Daily (30 rows) + weekly (8 rows) context frames — ~3 months of
    structure in 38 extra rows.

Usage:
    python -m src.distill --teacher models/stable.pt --student-out models/distilled.pt
"""
import argparse, json, os
import numpy as np
import torch
from torch.distributions import Beta

from .dataset import AssetDataset
from .model import TradingAgent


def sample_batch(ds, rng, batch, seq_len, window):
    """Sample [B, T, W, F] observation tensors from real training windows.
    v2 windows were fine-only (default 160) rows + 5 state cols; coarse
    frames are rebuilt from the same fine rows by the obs builder."""
    B, T, W = batch, seq_len, window
    Fdim = ds.feature_count + 5
    out = np.empty((B, T, W, Fdim), dtype=np.float32)
    starts = []
    for b in range(B):
        a, s = ds.sample(rng, steps=T + W + 8, window=W)
        starts.append((a, s))
    for t in range(T):
        for b, (a, s) in enumerate(starts):
            i = s + t
            if i + 1 >= a["end"]:
                i = a["end"] - 2
            feat = a["features"][i - W + 1:i + 1]
            # portfolio-state columns: neutral (flat, no position) — the
            # action-distribution transfer is about market reading.
            state = np.zeros((W, 5), dtype=np.float32)
            state[:, 2] = 0.0; state[:, 3] = t / T * 2 - 1
            out[b, t] = np.concatenate([feat, state], axis=1)
    return out


def v2_agent(feature_count, cfg):
    """Rebuild the v2 architecture (no reviewer, 3-frame obs) to load the
    teacher checkpoint. v3's real definition is the current agent.py."""
    import torch.nn as nn
    from .model.agent import TimeframeGating

    class V2(nn.Module):
        def __init__(self, input_size, encoder_size=256, hidden_size=256, num_gru_layers=3):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(input_size, encoder_size), nn.LayerNorm(encoder_size), nn.Tanh(),
                nn.Linear(encoder_size, encoder_size), nn.Tanh())
            self.frame_gate = TimeframeGating(encoder_size)
            self.attn_q = nn.Linear(encoder_size, 64)
            self.attn_k = nn.Linear(encoder_size, 64)
            self.gru = nn.GRU(encoder_size, hidden_size, num_layers=num_gru_layers, batch_first=True)
            self.policy = nn.Sequential(nn.Linear(hidden_size, 64), nn.Tanh())
            self.alpha = nn.Linear(64, 1)
            self.beta = nn.Linear(64, 1)
            self.value = nn.Sequential(nn.Linear(hidden_size, 64), nn.Tanh(), nn.Linear(64, 1))
            self.num_layers = num_gru_layers
            self.apply(self._init)

        @staticmethod
        def _init(m):
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight); nn.init.constant_(m.bias, 0)

        def encode_windows(self, x):
            b, t, w, f = x.shape
            flat = x.reshape(b * t, w, f)
            e = self.encoder(flat)
            fine_mask = (flat[..., -1] < 0.5)
            n_fine = fine_mask.sum(1, keepdim=True).clamp(min=1)
            n_coarse = (w - fine_mask.sum(1, keepdim=True)).clamp(min=1)
            fine = (e * fine_mask.unsqueeze(-1)).sum(1) / n_fine
            coarse = (e * (~fine_mask).unsqueeze(-1)).sum(1) / n_coarse
            mixed = self.frame_gate(fine, coarse)
            q = self.attn_q(mixed); k = self.attn_k(coarse)
            aff = torch.sigmoid((q * k).sum(-1))
            attended = mixed + aff.unsqueeze(-1) * coarse
            return attended.reshape(b, t, -1)

        def forward(self, x, h=None):
            z = self.encode_windows(x)
            y, h = self.gru(z, h)
            q = self.policy(y[:, -1])
            alpha = torch.nn.functional.softplus(self.alpha(q)) + 0.05
            beta = torch.nn.functional.softplus(self.beta(q)) + 0.05
            value = self.value(y[:, -1]).squeeze(-1)
            return torch.distributions.Beta(alpha.squeeze(-1), beta.squeeze(-1)), value, h

        def forward_sequence(self, x, h=None):
            z = self.encode_windows(x)
            y, h = self.gru(z, h)
            q = self.policy(y)
            alpha = torch.nn.functional.softplus(self.alpha(q)) + 0.05
            beta = torch.nn.functional.softplus(self.beta(q)) + 0.05
            value = self.value(y).squeeze(-1)
            return torch.distributions.Beta(alpha.squeeze(-1), beta.squeeze(-1)), value, h

    return V2(feature_count + 5,
              cfg.get("encoder_size", 256), cfg.get("hidden_size", 256),
              cfg.get("num_gru_layers", 3))


def distill(teacher_path, out_path, cfg, device, epochs=6, batches=200,
            batch=32, seq_len=32, lr=3e-4):
    ds = AssetDataset(cfg["processed_dir"], "train", cfg["window"])
    ck = torch.load(teacher_path, map_location=device, weights_only=False)
    # the teacher must be built with ITS OWN config (v2: reviewer-less,
    # 3 frames) and ITS OWN feature count (the checkpoint predates the
    # XGB columns — it expects 92 market features + 5 state cols).
    tcfg = ck.get("config", cfg)
    teacher_sd = ck.get("state_dim", ds.feature_count + 5)  # teacher obs width
    teacher = v2_agent(teacher_sd - 5, tcfg).to(device)
    state = {k.replace("_orig_mod.", "", 1): v for k, v in ck["model"].items()}
    teacher.load_state_dict(state); teacher.eval()

    student = TradingAgent(ds.feature_count + 5, cfg["encoder_size"],
                           cfg["hidden_size"], cfg["num_gru_layers"],
                           review_size=cfg.get("review_size", 256)).to(device)
    opt = torch.optim.Adam(student.parameters(), lr=lr)
    rng = np.random.default_rng(cfg.get("seed", 42))

    print(f"distilling {teacher_path} -> {out_path} | "
          f"batches={batches} batch={batch} seq={seq_len}")
    for ep in range(epochs):
        tot = 0.0
        for it in range(batches):
            win = sample_batch(ds, rng, batch, seq_len, cfg["window"])
            x = torch.from_numpy(win).to(device)
            with torch.no_grad():
                td, tv, _ = teacher.forward_sequence(build_teacher_obs(x, teacher_sd))
                t_mean = td.concentration1 / (td.concentration1 + td.concentration0)
            sd, sv, _ = student.forward_sequence(x)
            s_mean = sd.concentration1 / (sd.concentration1 + sd.concentration0)
            # match means (action), match concentration scale softly, match values.
            # value magnitudes dwarf action means (log-return scale), so the
            # value loss is scaled to keep both terms in comparable regimes.
            act_loss = (s_mean - t_mean).pow(2).mean()
            conc_loss = ((sd.concentration1 + sd.concentration0) -
                         (td.concentration1 + td.concentration0).clamp(min=0.1)).abs().mean() * 0.1
            val_loss = (sv - tv).pow(2).mean() * 0.05
            loss = act_loss + conc_loss + val_loss
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            opt.step()
            tot += loss.item()
        print(f"epoch {ep+1}/{epochs}: distill loss {tot/batches:.5f}")
    torch.save({"model": student.state_dict(), "update": 0,
                "best_val": -float('inf'), "stable_val": -float('inf'),
                "config": cfg, "distilled_from": teacher_path,
                "feature_count": ds.feature_count,
                "state_dim": ds.feature_count + 5},
               out_path)
    print(f"saved distilled student -> {out_path}")


def build_teacher_obs(x, teacher_state_dim):
    """Student windows [B,T,160,F+5] carry the FULL feature set (which may
    include new XGB signal columns the teacher never saw). The v2 teacher
    expects its own obs width (legacy feature_count + 5): fine rows +
    rebuilt coarse frames. We feed the teacher only the columns it knows."""
    B, T, W, F = x.shape
    fm = x[..., :teacher_state_dim]  # legacy market + state columns
    c4 = fm[:, :, -96:, :].reshape(B, T, 24, 4, teacher_state_dim).mean(dim=3)
    c12 = fm[:, :, -96:, :].reshape(B, T, 8, 12, teacher_state_dim).mean(dim=3)
    c4[..., -1] = 1.0; c12[..., -1] = 1.0
    return torch.cat([fm, c4, c12], dim=2)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--teacher", default="models/stable.pt")
    p.add_argument("--student-out", default="models/distilled.pt")
    p.add_argument("--config", default="config/default.json")
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--batches", type=int, default=200)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--seq-len", type=int, default=32)
    a = p.parse_args()
    cfg = json.load(open(a.config))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    distill(a.teacher, a.student_out, cfg, device, epochs=a.epochs,
            batches=a.batches, batch=a.batch, seq_len=a.seq_len)


if __name__ == "__main__":
    main()
