import torch
from torch import nn

class PPO:
    def __init__(self, model, lr=3e-4, gamma=.99, gae_lambda=.95, clip_coef=.2,
                 entropy_coef=.005, value_coef=.5, max_grad_norm=.5, epochs=4,
                 minibatch_size=1024, max_kl=0.02):
        self.model=model
        self.opt=torch.optim.Adam(model.parameters(),lr=lr,eps=1e-5)
        self.gamma=gamma; self.lam=gae_lambda; self.clip=clip_coef
        self.entropy_coef=entropy_coef; self.value_coef=value_coef
        self.max_grad_norm=max_grad_norm
        self.max_kl=max_kl
        self.epochs=epochs
        self.minibatch_size=minibatch_size
        self.base_lr = lr

    def set_lr(self, lr):
        for g in self.opt.param_groups: g["lr"] = lr

    def update(self, obs, actions, old_logprobs, rewards, values, dones, last_value):
        """All inputs are env-major: obs [N,T,W,F], scalars [N,T].
        Minibatches are slices of envs; the GRU runs along T inside each slice,
        which matches the time axis used during rollout collection."""
        N, T = actions.shape
        device = obs.device
        # GAE over the time axis, per env
        last_value = last_value.to(device).reshape(-1)
        adv = torch.zeros_like(rewards)
        gae = torch.zeros(N, device=device)
        for t in range(T-1, -1, -1):
            nv = last_value if t == T-1 else values[:, t+1]
            nonterm = 1. - dones[:, t]
            delta = rewards[:, t] + self.gamma * nv * nonterm - values[:, t]
            gae = delta + self.gamma * self.lam * nonterm * gae
            adv[:, t] = gae
        returns = adv + values
        flat = adv.reshape(-1)
        flat = (flat - flat.mean()) / (flat.std() + 1e-8)
        adv = flat.reshape(N, T)
        stats = {}
        approx_kl = 0.0
        stop_early = False
        for _ in range(self.epochs):
            if stop_early: break
            order = torch.randperm(N, device=device)
            for st in range(0, N, self.minibatch_size):
                idx = order[st:st+self.minibatch_size]
                if idx.numel() == 0: continue
                dist, val, _ = self.model.forward_sequence(obs[idx])
                lp = dist.log_prob(actions[idx].clamp(1e-6, 1.-1e-6))
                with torch.no_grad():
                    approx_kl = (old_logprobs[idx] - lp).mean().item()
                if approx_kl > self.max_kl:
                    stop_early = True
                    break
                ratio = torch.exp(lp - old_logprobs[idx])
                a = adv[idx]
                policy_loss = -torch.min(ratio*a, torch.clamp(ratio, 1-self.clip, 1+self.clip)*a).mean()
                value_loss = .5 * (returns[idx] - val).pow(2).mean()
                ent = dist.entropy().mean()
                loss = policy_loss + self.value_coef*value_loss - self.entropy_coef*ent
                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.opt.step()
                stats = {"policy_loss": round(policy_loss.item(), 5),
                         "value_loss": round(value_loss.item(), 5),
                         "entropy": round(ent.item(), 5),
                         "ratio_clip_frac": round((torch.abs(ratio-1) > self.clip).float().mean().item(), 4),
                         "approx_kl": round(approx_kl, 5)}
        return stats
