"""Distillation: the old (v1) genius teaches the new (v2) brain.

Why: sacred rule #12 — never retrain from scratch. The v1 model holds
27,389 updates of market understanding; distillation transfers it by
matching the new model's action distributions and value estimates to the
teacher's on thousands of real windows, BEFORE the new model starts its
own PPO training. Nothing is lost; the successor starts as a young genius
with an elder's memory.

Usage:
    python -m src.distill --teacher models/stable.pt --student-out models/distilled.pt
"""
import argparse, json, os
import numpy as np
import torch
from torch.distributions import Beta

from .dataset import AssetDataset
from .model import TradingAgent


def old_agent(feature_count, cfg):
    """Rebuild the v1 architecture (window-GRU encoder, 2-layer decision GRU,
    128 cells) to load the teacher checkpoint. Kept compact — distillation
    only. The v1's real definition lives in git history (commit e26c993)."""
    import torch.nn as nn

    class V1(nn.Module):
        def __init__(self, input_size, encoder_size=128, hidden_size=128, num_gru_layers=2):
            super().__init__()
            self.window_gru = nn.GRU(input_size, encoder_size, batch_first=True)
            self.gru = nn.GRU(encoder_size, hidden_size, num_layers=num_gru_layers, batch_first=True)
            self.policy = nn.Sequential(nn.Linear(hidden_size, 64), nn.Tanh())
            self.alpha = nn.Linear(64, 1)
            self.beta = nn.Linear(64, 1)
            self.value = nn.Sequential(nn.Linear(hidden_size, 64), nn.Tanh(), nn.Linear(64, 1))
            self.num_layers = num_gru_layers

        def forward(self, x, h=None):
            b, t, w, f = x.shape
            flat = x.reshape(b * t, w, f)
            e, _ = self.window_gru(flat)
            z = e[:, -1].reshape(b, t, -1)
            y, h = self.gru(z, h)
            q = self.policy(y[:, -1])
            alpha = torch.nn.functional.softplus(self.alpha(q)) + 0.05
            beta = torch.nn.functional.softplus(self.beta(q)) + 0.05
            value = self.value(y[:, -1]).squeeze(-1)
            return Beta(alpha.squeeze(-1), beta.squeeze(-1)), value, h

        def forward_sequence(self, x, h=None):
            b, t, w, f = x.shape
            flat = x.reshape(b * t, w, f)
            e, _ = self.window_gru(flat)
            z = e[:, -1].reshape(b, t, -1)
            y, h = self.gru(z, h)
            q = self.policy(y)
            alpha = torch.nn.functional.softplus(self.alpha(q)) + 0.05
            beta = torch.nn.functional.softplus(self.beta(q)) + 0.05
            value = self.value(y).squeeze(-1)
            return Beta(alpha.squeeze(-1), beta.squeeze(-1)), value, h

    return V1(feature_count + 4,
              tcfg_enc(cfg), tcfg_hid(cfg), tcfg_layers(cfg))


def tcfg_enc(cfg): return cfg.get("encoder_size", 128)
def tcfg_hid(cfg): return cfg.get("hidden_size", 128)
def tcfg_layers(cfg): return cfg.get("num_gru_layers", 2)


def sample_batch(ds, rng, batch, seq_len, window):
    """Sample [B, T, W, F] observation tensors from real training windows.
    v1 windows were fine-only 96 rows + 4 state cols."""
    B, T, W = batch, seq_len, window
    Fdim = ds.feature_count + 4
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
            # v1's 4 portfolio-state columns were generated live by the env;
            # for distillation we use neutral state (flat, no position) —
            # the action distribution transfer is about market reading.
            state = np.zeros((W, 4), dtype=np.float32)
            state[:, 2] = 0.0; state[:, 3] = t / T * 2 - 1
            out[b, t] = np.concatenate([feat, state], axis=1)
    return out


def distill(teacher_path, out_path, cfg, device, epochs=3, batches=200,
            batch=16, seq_len=32, lr=1e-4):
    ds = AssetDataset(cfg["processed_dir"], "train", cfg["window"])
    ck = torch.load(teacher_path, map_location=device, weights_only=False)
    # the teacher must be built with ITS OWN config (v1: 128 cells, 2 layers)
    tcfg = ck.get("config", cfg)
    teacher = old_agent(ds.feature_count, tcfg).to(device)
    teacher.load_state_dict(ck["model"]); teacher.eval()

    student = TradingAgent(ds.feature_count + 5, cfg["encoder_size"],
                           cfg["hidden_size"], cfg["num_gru_layers"]).to(device)
    opt = torch.optim.Adam(student.parameters(), lr=lr)
    rng = np.random.default_rng(cfg.get("seed", 42))

    print(f"distilling {teacher_path} -> {out_path} | "
          f"batches={batches} batch={batch} seq={seq_len}")
    for ep in range(epochs):
        tot = 0.0
        for it in range(batches):
            win = sample_batch(ds, rng, batch, seq_len, 96)  # teacher sees v1 windows
            x = torch.from_numpy(win).to(device)
            with torch.no_grad():
                td, tv, _ = teacher.forward_sequence(x)
                t_mean = td.concentration1 / (td.concentration1 + td.concentration0)
            xs = build_student_obs(x, device)
            sd, sv, _ = student.forward_sequence(xs)
            s_mean = sd.concentration1 / (sd.concentration1 + sd.concentration0)
            # match means (action), match concentration scale softly, match values
            act_loss = (s_mean - t_mean).pow(2).mean()
            conc_loss = ((sd.concentration1 + sd.concentration0) -
                         (td.concentration1 + td.concentration0).clamp(min=0.1)).abs().mean() * 0.1
            val_loss = (sv - tv).pow(2).mean()
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


def build_student_obs(x, device):
    """Convert teacher windows [B,T,96,F+4] into student v2 windows
    [B,T,192,F+5]: fine 160 rows aren't available from a 96-row window, so
    we pad by repeating the earliest row (state-only padding — market rows
    come from the same window). Frame flag appended: fine=0, coarse=1.
    Coarse rows: 24 (4h) and 8 (12h) built by masked averaging of fine rows.
    """
    B, T, W, F = x.shape
    out_fine = x.new_zeros(B, T, 160, F + 1)
    out_fine[:, :, :W, :F] = x
    if W < 160:
        out_fine[:, :, W:, :F] = x[:, :, :1, :]  # pad with earliest row
        out_fine[:, :, W:, F] = 0  # padded fine rows flag stays fine
    out_fine[:, :, :, F] = 0.0
    # coarse 4h: average consecutive groups of 4 fine rows (padded repeats)
    fine = out_fine[:, :, :, :F]
    c4 = fine.reshape(B, T, 40, 4, F).mean(dim=3)  # 40 groups from 160 rows
    c4 = c4[:, :, -24:]
    c12 = fine.reshape(B, T, 16, 10, F).mean(dim=3)  # ~12h groups of 12: use 10-row groups
    c12 = c12[:, :, -8:]
    def flag(rows, val):
        fl = torch.full((rows.shape[0], rows.shape[1], rows.shape[2], 1), val,
                        device=rows.device, dtype=rows.dtype)
        return torch.cat([rows, fl], dim=-1)
    return torch.cat([out_fine, flag(c4, 1.0), flag(c12, 1.0)], dim=2)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--teacher", default="models/stable.pt")
    p.add_argument("--student-out", default="models/distilled.pt")
    p.add_argument("--config", default="config/default.json")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batches", type=int, default=200)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--seq-len", type=int, default=32)
    a = p.parse_args()
    cfg = json.load(open(a.config))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    distill(a.teacher, a.student_out, cfg, device, epochs=a.epochs,
            batches=a.batches, batch=a.batch, seq_len=a.seq_len)


if __name__ == "__main__":
    main()
