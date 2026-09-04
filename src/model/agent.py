import torch
from torch import nn
from torch.distributions import Beta

class TradingAgent(nn.Module):
    def __init__(self,input_size,encoder_size=128,hidden_size=128,num_gru_layers=2):
        super().__init__(); self.encoder=nn.Sequential(nn.Linear(input_size,encoder_size),nn.LayerNorm(encoder_size),nn.Tanh(),nn.Linear(encoder_size,encoder_size),nn.Tanh()); self.gru=nn.GRU(encoder_size,hidden_size,num_layers=num_gru_layers,batch_first=True); self.policy=nn.Sequential(nn.Linear(hidden_size,64),nn.Tanh()); self.alpha=nn.Linear(64,1); self.beta=nn.Linear(64,1); self.value=nn.Sequential(nn.Linear(hidden_size,64),nn.Tanh(),nn.Linear(64,1)); self.num_layers=num_gru_layers
        self.apply(self._init)
    @staticmethod
    def _init(m):
        if isinstance(m,nn.Linear): nn.init.orthogonal_(m.weight); nn.init.constant_(m.bias,0)
    def encode_windows(self,x):
        # [B,T,W,F] -> [B,T,E]
        b,t,w,f=x.shape; z=self.encoder(x.reshape(b*t,w,f)); z=z.mean(dim=1); return z.reshape(b,t,-1)
    def forward(self,x,h=None):
        z=self.encode_windows(x); y,h=self.gru(z,h); q=self.policy(y[:,-1]); alpha=torch.nn.functional.softplus(self.alpha(q))+1.0; beta=torch.nn.functional.softplus(self.beta(q))+1.0; value=self.value(y[:,-1]).squeeze(-1); return Beta(alpha.squeeze(-1),beta.squeeze(-1)),value,h
    def forward_sequence(self,x,h=None):
        z=self.encode_windows(x); y,h=self.gru(z,h); q=self.policy(y); alpha=torch.nn.functional.softplus(self.alpha(q))+1.0; beta=torch.nn.functional.softplus(self.beta(q))+1.0; value=self.value(y).squeeze(-1); return Beta(alpha.squeeze(-1),beta.squeeze(-1)),value,h
