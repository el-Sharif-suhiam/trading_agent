import argparse, glob, json, os
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from .features import FeatureEngine

class AssetDataset:
    def __init__(self, processed_dir, split, window):
        self.processed_dir=processed_dir; self.split=split; self.window=window
        self.manifest=pd.read_csv(os.path.join(processed_dir,f"{split}_manifest.csv"))
        self.assets=[]
        for _,r in self.manifest.iterrows():
            f=os.path.join(processed_dir,r.file)
            df=pd.read_csv(f,parse_dates=["datetime"])
            feats=df.drop(columns=["datetime","close"],errors="ignore").values.astype(np.float32)
            close=df.close.values.astype(np.float64)
            self.assets.append({"symbol":r.symbol,"features":feats,"close":close,"start":int(r.start_idx),"end":int(r.end_idx)})
        self.feature_count=self.assets[0]["features"].shape[1]

    def sample(self, rng, steps, window=0):
        """Random episode start guaranteed to keep a full observation window
        in the past and `steps` candles ahead, so no negative/wrapping slices."""
        for _ in range(100):
            a = self.assets[int(rng.integers(len(self.assets)))]
            lo = a["start"] + window - 1
            hi = a["end"] - steps - 1
            if hi >= lo:
                return a, int(rng.integers(lo, hi + 1))
        raise RuntimeError(
            f"No asset long enough for window={window} + steps={steps}; "
            "download more history or reduce window/rollout_steps"
        )


def build(raw_dir, processed_dir, train_frac=.70, val_frac=.15, unseen_symbols=None):
    os.makedirs(processed_dir,exist_ok=True)
    files=sorted(glob.glob(os.path.join(raw_dir,"*.csv")))
    if not files: raise RuntimeError("No raw CSV files found")
    unseen_symbols={x.replace("/","_").upper() for x in (unseen_symbols or [])}
    engine=FeatureEngine(); prepared=[]; unseen_prepared=[]; scaler=StandardScaler()
    for f in files:
        raw=pd.read_csv(f,parse_dates=["datetime"])
        raw=raw.sort_values("datetime").drop_duplicates("datetime").reset_index(drop=True)
        if len(raw)<1000: print("Skipping short asset",f); continue
        feat=engine.transform(raw)
        n=len(raw); tr=int(n*train_frac); va=int(n*(train_frac+val_frac))
        if os.path.basename(f).replace(".csv","").upper() in unseen_symbols:
            unseen_prepared.append((f,raw,feat,tr,va))
        else:
            scaler.partial_fit(feat.iloc[:tr].values); prepared.append((f,raw,feat,tr,va))
    if not prepared: raise RuntimeError("No usable training assets")
    joblib.dump(scaler,os.path.join(processed_dir,"scaler.joblib"))
    manifests={"train":[],"validation":[],"test":[],"unseen_test":[]}
    for f,raw,feat,tr,va in prepared:
        z=scaler.transform(feat.values).astype(np.float32)
        out=pd.DataFrame(z,columns=feat.columns); out.insert(0,"close",raw.close.values); out.insert(0,"datetime",raw.datetime.astype(str))
        name=os.path.basename(f).replace(".csv","_features.csv"); out.to_csv(os.path.join(processed_dir,name),index=False)
        symbol=os.path.basename(f).replace(".csv","").replace("_","/")
        for split,s,e in [("train",0,tr),("validation",tr,va),("test",va,n:=len(raw))]:
            manifests[split].append({"symbol":symbol,"file":name,"start_idx":s,"end_idx":e})
    for f,raw,feat,tr,va in unseen_prepared:
        z=scaler.transform(feat.values).astype(np.float32); out=pd.DataFrame(z,columns=feat.columns); out.insert(0,"close",raw.close.values); out.insert(0,"datetime",raw.datetime.astype(str)); name=os.path.basename(f).replace(".csv","_features.csv"); out.to_csv(os.path.join(processed_dir,name),index=False); symbol=os.path.basename(f).replace(".csv","").replace("_","/"); manifests["unseen_test"].append({"symbol":symbol,"file":name,"start_idx":0,"end_idx":len(raw)})
    for split,rows in manifests.items(): pd.DataFrame(rows).to_csv(os.path.join(processed_dir,f"{split}_manifest.csv"),index=False)
    with open(os.path.join(processed_dir,"meta.json"),"w") as fh: json.dump({"feature_count":len(prepared[0][2].columns),"assets":[r[0] for r in prepared]},fh,indent=2,default=str)
    print("Built",len(prepared),"assets; features",len(prepared[0][2].columns))

if __name__=="__main__":
    p=argparse.ArgumentParser(); sub=p.add_subparsers(dest="cmd",required=True); b=sub.add_parser("build")
    b.add_argument("--raw-dir",default="data/raw"); b.add_argument("--processed-dir",default="data/processed"); b.add_argument("--unseen-symbols",nargs="*",default=[])
    a=b.parse_args(); build(a.raw_dir,a.processed_dir,unseen_symbols=a.unseen_symbols)
