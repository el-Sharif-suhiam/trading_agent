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


class Reviewer(nn.Module):
    """The Reviewer (second thought): a light GRU that re-reads the main
    decision stream before the heads act, plus a single-head CAUSAL
    self-attention that lets each timestep ask 'which PAST moments matter
    for THIS decision?'. The causal mask is not optional: without it the
    sequence-training view would attend to future steps that step-by-step
    rollout (carried hidden state) can never see — breaking the
    step-vs-sequence invariant AND leaking future information (sacred
    rule #10). With the mask, position t attends strictly to positions
    <= t, so one-step forward with carried state == sequence forward.

    Safety contract (owner's bottleneck objection, resolved mathematically):
      - RESIDUAL, not a gate: output = main_y + proj(reviewer_y).
        The main stream reaches the heads UNDILUTED — no information bottleneck.
      - ZERO-INIT: proj starts as exact zeros, so at step 0 the new model is
        BIT-FOR-BIT the old model. The reviewer can only earn influence
        through gradients that demonstrably reduce the loss. Worst case ==
        predecessor, never worse.
      - No veto power: it adds a marginal note on the page; it never rewrites
        the page.
    """
    def __init__(self, hidden_size, review_size=256):
        super().__init__()
        self.gru = nn.GRU(hidden_size, review_size, num_layers=1, batch_first=True)
        # single-head CAUSAL self-attention over the decision stream
        self.attn_q = nn.Linear(review_size, 64)
        self.attn_k = nn.Linear(review_size, 64)
        self.proj = nn.Linear(review_size, hidden_size)
        nn.init.zeros_(self.proj.weight); nn.init.zeros_(self.proj.bias)
        self.num_layers = 1

    def forward(self, y, h):
        """h = (gru_hidden, r_cache or None). r_cache holds the reviewer's
        past hidden states r[0..t-1] so that step-by-step rollout can run
        the SAME causal attention as sequence mode (which sees the whole
        prefix). Without the cache, a one-step call would attend only to
        itself — an irreproducible mismatch vs training."""
        gru_h, r_cache = (h if isinstance(h, tuple) else (h, None))
        r, gru_h = self.gru(y, gru_h)                     # [B,T,R]
        B, T, _ = r.shape
        if r_cache is not None and T > 0:
            r_all = torch.cat([r_cache, r], dim=1)         # [B, past+T, R]
        else:
            r_all = r
        q = self.attn_q(r); k = self.attn_k(r_all)         # q: [B,T,64], k: [B,past+T,64]
        scores = q @ k.transpose(-2, -1) / (64 ** 0.5)     # [B, T, past+T]
        # causal: position t (global) attends to all cached/past <= t
        past = r_all.shape[1] - T
        Tq = scores.shape[2]
        col = torch.arange(Tq, device=r.device)
        row = torch.arange(T, device=r.device) + past
        mask = col[None, :] <= row[:, None]                # [T, past+T]
        scores = scores.masked_fill(~mask, float("-inf"))
        aff = torch.softmax(scores, dim=-1)
        attended = r + torch.matmul(aff, r_all)            # residual attention
        out = y + self.proj(attended)                       # zero-init residual
        return out, (gru_h, r_all)


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
    def __init__(self, input_size, encoder_size=256, hidden_size=256, num_gru_layers=3,
                 review_size=256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, encoder_size), nn.LayerNorm(encoder_size), nn.Tanh(),
            nn.Linear(encoder_size, encoder_size), nn.Tanh())
        self.frame_gate = TimeframeGating(encoder_size)
        # single query attention: fine embeddings attend to coarse embeddings
        self.attn_q = nn.Linear(encoder_size, 64)
        self.attn_k = nn.Linear(encoder_size, 64)
        self.gru = nn.GRU(encoder_size, hidden_size, num_layers=num_gru_layers, batch_first=True)
        # the Reviewer: second thought before the heads act (zero-init residual)
        self.reviewer = Reviewer(hidden_size, review_size)
        self.policy = nn.Sequential(nn.Linear(hidden_size, 64), nn.Tanh())
        self.alpha = nn.Linear(64, 1)
        self.beta = nn.Linear(64, 1)
        self.value = nn.Sequential(nn.Linear(hidden_size, 64), nn.Tanh(), nn.Linear(64, 1))
        self.num_layers = num_gru_layers
        self.apply(self._init)
        # re-zero the reviewer projection AFTER self.apply (orthogonal init
        # would otherwise break the silent-start guarantee)
        nn.init.zeros_(self.reviewer.proj.weight); nn.init.zeros_(self.reviewer.proj.bias)

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

    def _split_h(self, h):
        """Hidden state is (stacked [main | reviewer] tensor, r_cache).
        Accepts None, the bare stacked tensor (e.g. from old checkpoints),
        or the full pair."""
        if h is None:
            return None, (None, None)
        if isinstance(h, tuple):
            stacked, r_cache = h
        else:
            stacked, r_cache = h, None
        return stacked[:self.num_layers], (stacked[self.num_layers:], r_cache)

    def forward(self, x, h=None):
        """h = stacked (main GRU layers, reviewer GRU layer). The reviewer's
        r-cache lives inside the reviewer's tuple state (see Reviewer.forward)
        — the stacked tensor form carries only recurrent parts, so step-wise
        rollout also needs the cache: it is returned as part of `h` as a
        (stacked_tensor, r_cache) pair from this method."""
        z = self.encode_windows(x)                   # [B, T, E]
        hm, (hr_tensor, r_cache) = self._split_h(h)
        y, hm = self.gru(z, hm)
        y, (hr_tensor, r_cache) = self.reviewer(y, (hr_tensor, r_cache))
        q = self.policy(y[:, -1])
        alpha = torch.nn.functional.softplus(self.alpha(q)) + 0.05
        beta = torch.nn.functional.softplus(self.beta(q)) + 0.05
        value = self.value(y[:, -1]).squeeze(-1)
        return Beta(alpha.squeeze(-1), beta.squeeze(-1)), value, \
            (torch.cat([hm, hr_tensor]), r_cache)

    def forward_sequence(self, x, h=None, chunk=8):
        """x: [B, T, W, F]. Processes the sequence in temporal chunks so the
        encoder activations of the whole T×W×E tensor never live in memory
        at once (8GB GPU budget). The GRU carries state across chunks, so
        the result is IDENTICAL to unchunked processing."""
        hm, (hr_tensor, r_cache) = self._split_h(h)
        z_chunks = []
        for st in range(0, x.shape[1], chunk):
            z_chunks.append(self.encode_windows(x[:, st:st+chunk]))
        z = torch.cat(z_chunks, dim=1) if len(z_chunks) > 1 else z_chunks[0]
        y, hm = self.gru(z, hm)
        y, (hr_tensor, r_cache) = self.reviewer(y, (hr_tensor, r_cache))
        q = self.policy(y)
        alpha = torch.nn.functional.softplus(self.alpha(q)) + 0.05
        beta = torch.nn.functional.softplus(self.beta(q)) + 0.05
        value = self.value(y).squeeze(-1)
        return Beta(alpha.squeeze(-1), beta.squeeze(-1)), value, \
            (torch.cat([hm, hr_tensor]), r_cache)
