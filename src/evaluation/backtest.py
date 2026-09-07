import argparse,json,numpy as np,torch
from .evaluate import run
from ..dataset import AssetDataset
from ..model import TradingAgent

def main():
 p=argparse.ArgumentParser(); p.add_argument("--config",default="config/default.json"); p.add_argument("--checkpoint",default="models/latest.pt"); p.add_argument("--split",default="test",choices=["validation","test","unseen_test"]); a=p.parse_args(); cfg=json.load(open(a.config)); device=torch.device("cuda" if torch.cuda.is_available() else "cpu"); ds=AssetDataset(cfg["processed_dir"],a.split,cfg["window"]); model=TradingAgent(ds.feature_count+5,cfg["encoder_size"],cfg["hidden_size"],cfg["num_gru_layers"]).to(device); model.load_state_dict(torch.load(a.checkpoint,map_location=device)["model"]); model.eval(); results=[run(model,x,cfg,device) for x in ds.assets];
 print("mean final",np.mean([r["final"] for r in results]),"mean return %",np.mean([r["return_pct"] for r in results])); [print(x["symbol"],r) for x,r in zip(ds.assets,results)]
if __name__=="__main__": main()
