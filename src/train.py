import argparse,json,os,random
import numpy as np, torch
from .dataset import AssetDataset
from .model import TradingAgent
from .training.environment import SpotTradingEnv
from .training.ppo import PPO

def main():
    p=argparse.ArgumentParser(); p.add_argument("--config",default="config/default.json"); p.add_argument("--resume",default=""); a=p.parse_args(); cfg=json.load(open(a.config)); os.makedirs(cfg["model_dir"],exist_ok=True); os.makedirs(cfg["log_dir"],exist_ok=True)
    random.seed(cfg["seed"]); np.random.seed(cfg["seed"]); torch.manual_seed(cfg["seed"]); device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds=AssetDataset(cfg["processed_dir"],"train",cfg["window"]); model=TradingAgent(ds.feature_count,cfg["encoder_size"],cfg["hidden_size"],cfg["num_gru_layers"]).to(device)
    ppo=PPO(model,cfg["learning_rate"],cfg["gamma"],cfg["gae_lambda"],cfg["clip_coef"],cfg["entropy_coef"],cfg["value_coef"],cfg["max_grad_norm"],cfg["ppo_epochs"],cfg["minibatch_size"])
    start=0
    if a.resume:
        ck=torch.load(a.resume,map_location=device); model.load_state_dict(ck["model"]); ppo.opt.load_state_dict(ck["optimizer"]); start=ck.get("update",0); print("Resumed",start)
    print("device",device,"assets",len(ds.assets),"features",ds.feature_count)
    for update in range(start,cfg["updates"]):
        envs=[]
        for _ in range(cfg["num_envs"]):
            asset,s=ds.sample(np.random.default_rng(),cfg["episode_steps"]); envs.append(SpotTradingEnv(asset["features"],asset["close"],asset["start"],asset["end"],cfg["window"],cfg["episode_steps"],cfg["fee"],cfg["slippage"],cfg["initial_cash"])); envs[-1].reset(seed=cfg["seed"]+update)
            envs[-1].i=s
        obs=np.stack([e._obs() for e in envs]); obs_t=torch.tensor(obs,device=device)
        O=[];A=[];LP=[];R=[];V=[];D=[]; infos=[]
        h=None
        for _ in range(cfg["rollout_steps"]):
            with torch.no_grad(): dist,val,h=model(obs_t[:,None],h)
            act=dist.sample().clamp(1e-5,1-1e-5); vals=val; lp=dist.log_prob(act)
            next_obs=[]; rs=[]; ds_done=[]
            for j,e in enumerate(envs):
                no,r,term,trunc,info=e.step(np.array([act[j].item()],dtype=np.float32)); next_obs.append(no); rs.append(r); ds_done.append(term or trunc); infos.append(info)
            O.append(obs_t.cpu()); A.append(act.detach().cpu()); LP.append(lp.detach().cpu()); R.append(torch.tensor(rs)); V.append(vals.detach().cpu()); D.append(torch.tensor(ds_done,dtype=torch.float32)); obs_t=torch.tensor(np.stack(next_obs),device=device)
        obs_seq=torch.stack(O).to(device); actions=torch.stack(A).to(device); old_lp=torch.stack(LP).to(device); rewards=torch.stack(R).to(device); values=torch.stack(V).to(device); dones=torch.stack(D).to(device)
        with torch.no_grad(): _,last_v,h_last=model(obs_t[:,None],h)
        stats=ppo.update(obs_seq,actions,old_lp,rewards,values,dones,last_v)
        vals=[x["portfolio_value"] for x in infos[-cfg["num_envs"]:]]; trades=sum(x["trades"] for x in infos[-cfg["num_envs"]:])
        if update%10==0:
            ck={"model":model.state_dict(),"optimizer":ppo.opt.state_dict(),"update":update+1,"config":cfg}; torch.save(ck,os.path.join(cfg["model_dir"],"latest.pt")); print(f"update {update+1}/{cfg['updates']} portfolio_mean={np.mean(vals):.2f} trades={trades} {stats}")
    torch.save({"model":model.state_dict(),"optimizer":ppo.opt.state_dict(),"update":cfg["updates"],"config":cfg},os.path.join(cfg["model_dir"],"latest.pt"))

if __name__=="__main__": main()
