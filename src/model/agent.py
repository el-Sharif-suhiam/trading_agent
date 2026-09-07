import torch
from torch import nn
from torch.distributions import Beta


class TimeframeGating(nn.Module):
    """Dynamic frame switching: learns, from the encoded window state, how
    much weight to give the fine-grained (1h) view vs the coarse aggregated
    (4h/12h) views. In calm regimes the agent leans on structure (coarse);
    in volatile moments it leans on detail (fine). The gate is learned —
    not a hand-coded rule (sacred rule #4)."""
    def __init__(self, encoder_size):
        super().__init__()
        self.g = nn.Sequential(nn.Linear(encoder_size, 32), nn.Tanh(),
                               nn.Linear(32, 2))  # logits for [fine, coarse]

    def forward(self, fine_emb, coarse_emb):
        # fine_emb/coarse_emb: [B, T, E]
        w = torch.softmax(self.g(fine_emb), dim=-1)     # [B, T, 2]
        return w[..., 0:1] * fine_emb + w[..., 1:2] * coarse_emb


class TradingAgent(nn.Module):
    """Genius architecture v2 — the distilled successor brain.

    Upgrades over v1 (301K params, single-frame window-GRU):
      - window 192 candles (8 days) incl. embedded aggregated rows (4h/12h)
      - dynamic timeframe gating (switches attention between frames)
      - encoder 256, decision GRU 256 x 3 layers (deeper abstraction stack)
      - cross-timeframe attention: fine rows query coarse rows

    The model is frame-agnostic in inputs: rows arrive with a 'frame' flag
    among portfolio state; it learns where to look.
    """
    def __init__(self, input_size, encoder_size=256, hidden_size=256, num_gru_layers=3):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, encoder_size), nn.LayerNorm(encoder_size), nn.Tanh(),
            nn.Linear(encoder_size, encoder_size), nn.Tanh())
        self.frame_gate = TimeframeGating(encoder_size)
        # single query attention: fine embeddings attend to coarse embeddings
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
        """x: [B, T, W, F] -> [B, T, E]; each window is embedded, then fine
        and coarse frame rows are pooled separately (mask-aware), mixed by
        the learned dynamic frame gate, and refined with a cheap bilinear
        cross-frame affinity.

        Row layout contract (produced by the environment):
          last 5 columns of F are [position, pnl, drawdown, timefrac, frame]
          frame flag: 0 = fine (1h), 1 = coarse (4h/12h)
        """
        b, t, w, f = x.shape
        flat = x.reshape(b * t, w, f)
        e = self.encoder(flat)                        # [B*T, W, E]
        fine_mask = (flat[..., -1] < 0.5)              # [B*T, W]
        n_fine = fine_mask.sum(1, keepdim=True).clamp(min=1)
        n_coarse = (w - fine_mask.sum(1, keepdim=True)).clamp(min=1)
        fine = (e * fine_mask.unsqueeze(-1)).sum(1) / n_fine      # [B*T, E]
        coarse = (e * (~fine_mask).unsqueeze(-1)).sum(1) / n_coarse  # [B*T, E]
        mixed = self.frame_gate(fine, coarse)          # [B*T, E]
        # cross-timeframe affinity: scalar per window, fine vs coarse query
        q = self.attn_q(mixed)                          # [B*T, 64]
        k = self.attn_k(coarse)                         # [B*T, 64]
        aff = torch.sigmoid((q * k).sum(-1))            # [B*T] in (0,1)
        attended = mixed + aff.unsqueeze(-1) * coarse    # [B*T, E]
        return attended.reshape(b, t, -1)

    def forward(self, x, h=None):
        z = self.encode_windows(x)                   # [B, T, E]
        y, h = self.gru(z, h)
        q = self.policy(y[:, -1])
        alpha = torch.nn.functional.softplus(self.alpha(q)) + 0.05
        beta = torch.nn.functional.softplus(self.beta(q)) + 0.05
        value = self.value(y[:, -1]).squeeze(-1)
        return Beta(alpha.squeeze(-1), beta.squeeze(-1)), value, h

    def forward_sequence(self, x, h=None):
        z = self.encode_windows(x)                   # [B, T, E]
        y, h = self.gru(z, h)
        q = self.policy(y)
        alpha = torch.nn.functional.softplus(self.alpha(q)) + 0.05
        beta = torch.nn.functional.softplus(self.beta(q)) + 0.05
        value = self.value(y).squeeze(-1)
        return Beta(alpha.squeeze(-1), beta.squeeze(-1)), value, h
