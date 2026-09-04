import argparse,json,os
import numpy as np, torch
from ..dataset import AssetDataset
from ..model import TradingAgent
from ..training.environment import SpotTradingEnv

def run(model,asset,cfg,device):
    steps=asset["end"]-asset["start"]-cfg["window"]-1; start=asset["start"]+cfg["window"]-1
    env=SpotTradingEnv(asset["features"],asset["close"],asset["start"],asset["end"],cfg["window"],max(1,steps),cfg["fee"],cfg["slippage"],cfg["initial_cash"]); obs,_=env.reset(seed=1,options={"start":start}); h=None
    done=False
    while not done:
        with torch.no_grad(): dist,_,h=model(torch.tensor(obs[None,None],device=device),h); act=(dist.concentration1/(dist.concentration1+dist.concentration0)).item()
        obs,r,t,tr,info=env.step(np.array([act],np.float32)); done=t or tr
    buyhold=cfg["initial_cash"]*(asset["close"][asset["end"]-1]/asset["close"][start])*(1-cfg["fee"])
    return {"final":info["portfolio_value"],"return_pct":(info["portfolio_value"]/cfg["initial_cash"]-1)*100,"max_drawdown":info["drawdown"],"trades":info["trades"],"fees":info["fees"],"buy_hold":buyhold}

def main():
    p=argparse.ArgumentParser(); p.add_argument("--config",default="config/default.json"); p.add_argument("--checkpoint",default="models/latest.pt"); p.add_argument("--split",default="validation",choices=["validation","test","unseen_test"]); a=p.parse_args(); cfg=json.load(open(a.config)); device=torch.device("cuda" if torch.cuda.is_available() else "cpu" ); ds=AssetDataset(cfg["processed_dir"],a.split,cfg["window"]); model=TradingAgent(ds.feature_count,cfg["encoder_size"],cfg["hidden_size"],cfg["num_gru_layers"]).to(device); ck=torch.load(a.checkpoint,map_location=device); model.load_state_dict(ck["model"]); model.eval()
    for x in ds.assets: print(x["symbol"],run(model,x,cfg,device))
if __name__=="__main__": main()
