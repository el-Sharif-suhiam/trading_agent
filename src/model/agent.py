import torch
from torch import nn
from torch.distributions import Beta

class TradingAgent(nn.Module):
    """Window encoder -> GRU temporal memory -> Beta policy + value head.

    Each observation is a window [W, F] of candles. The encoder embeds the
    whole window into one vector **in candle order** (small GRU, not mean
    pooling), then the decision GRU consumes the sequence of window embeddings
    across decision steps. The policy head outputs Beta(alpha, beta) with
    alpha, beta > 0, so target allocation can reach the full [0, 1] range.
    """
    def __init__(self, input_size, encoder_size=128, hidden_size=128, num_gru_layers=2):
        super().__init__()
        self.window_gru = nn.GRU(input_size, encoder_size, batch_first=True)
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
        # x: [B, T, W, F] -> window embeddings [B, T, E]
        b, t, w, f = x.shape
        flat = x.reshape(b * t, w, f)
        e, _ = self.window_gru(flat)
        return e[:, -1].reshape(b, t, -1)

    def forward(self, x, h=None):
        # x: [B, 1, W, F] single decision step
        z = self.encode_windows(x)          # [B, 1, E]
        y, h = self.gru(z, h)               # [B, 1, H], h carries memory
        q = self.policy(y[:, -1])
        alpha = torch.nn.functional.softplus(self.alpha(q)) + 0.05
        beta = torch.nn.functional.softplus(self.beta(q)) + 0.05
        value = self.value(y[:, -1]).squeeze(-1)
        return Beta(alpha.squeeze(-1), beta.squeeze(-1)), value, h

    def forward_sequence(self, x, h=None):
        # x: [B, T, W, F]; identical computation to T sequential forward() calls
        z = self.encode_windows(x)           # [B, T, E]
        y, h = self.gru(z, h)               # [B, T, H]
        q = self.policy(y)
        alpha = torch.nn.functional.softplus(self.alpha(q)) + 0.05
        beta = torch.nn.functional.softplus(self.beta(q)) + 0.05
        value = self.value(y).squeeze(-1)
        return Beta(alpha.squeeze(-1), beta.squeeze(-1)), value, h
