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

    def sample(self, rng, steps, window=0, crisis_bias=0.0):
        """Random episode start; `window` here is the FINE-window history
        requirement (160 for the v2 multi-frame observation). The weekly
        frame needs 1344 candles — the largest margin of all frames."""
        for _ in range(100):
            a = self.assets[int(rng.integers(len(self.assets)))]
            lo = a["start"] + max(window, 1344)
            hi = a["end"] - steps - 1
            if hi < lo:
                continue
            if crisis_bias > 0 and rng.random() < crisis_bias:
                s = self._crisis_start(rng, a, steps, max(window, 1344))
                if s is not None:
                    return a, s
            return a, int(rng.integers(lo, hi + 1))
        raise RuntimeError(
            f"No asset long enough for window={window} + steps={steps}; "
            "download more history or reduce window/rollout_steps"
        )

    def _crisis_start(self, rng, a, steps, window):
        """Locate severe drawdown regions in this asset's close series and
        sample an episode start inside one. Returns None if no crisis found
        (e.g. mostly flat asset)."""
        if not hasattr(self, "_crisis_cache"):
            self._crisis_cache = {}
        key = (a["symbol"],)
        if key not in self._crisis_cache:
            c = a["close"]
            # rolling max drawdown over ~30 days (720h) on the train range only
            lookback = 720
            s = a["start"]; e = a["end"]
            idx = np.arange(s, e)
            if len(idx) < lookback + steps:
                self._crisis_cache[key] = None
                return None
            roll_max = np.maximum.accumulate(c[idx])
            dd = 1.0 - c[idx]/np.maximum(roll_max, 1e-12)
            # crisis = regions where drawdown exceeded 25% from running peak
            crisis_mask = dd > 0.25
            self._crisis_cache[key] = (idx, crisis_mask)
        cached = self._crisis_cache[key]
        if cached is None:
            return None
        idx, mask = cached
        candidates = idx[mask]
        # need a full window behind and steps ahead within the asset range
        valid = [i for i in candidates
                 if i - (window-1) >= a["start"] and i + steps < a["end"]]
        if not valid:
            return None
        return int(valid[rng.integers(len(valid))])


def build(raw_dir, processed_dir, train_frac=.70, val_frac=.15, unseen_symbols=None):
    os.makedirs(processed_dir,exist_ok=True)
    files=sorted(glob.glob(os.path.join(raw_dir,"*.csv")))
    if not files: raise RuntimeError("No raw CSV files found")
    unseen_symbols={x.replace("/","_").upper() for x in (unseen_symbols or [])}
    engine=FeatureEngine(); prepared=[]; unseen_prepared=[]; scaler=StandardScaler()
    use_xgb = os.environ.get("USE_XGB_FEATURES", "1") == "1"
    xgb_models = {}
    for f in files:
        raw=pd.read_csv(f,parse_dates=["datetime"])
        raw=raw.sort_values("datetime").drop_duplicates("datetime").reset_index(drop=True)
        if len(raw)<1000: print("Skipping short asset",f); continue
        feat=engine.transform(raw)
        n=len(raw); tr=int(n*train_frac); va=int(n*(train_frac+val_frac))
        sym=os.path.basename(f).replace(".csv","").upper()
        if use_xgb and sym not in unseen_symbols:
            # fit ensemble on THIS asset's train segment only (rule #10),
            # predict past-only rows -> 3 extra feature columns
            from .features.xgb_ensemble import fit_ensemble, transform_ensemble
            dm, vm, names = fit_ensemble(feat.iloc[:tr], raw.close.iloc[:tr])
            sig = transform_ensemble(feat, raw.close, dm, vm, names)
            feat = pd.concat([feat, sig], axis=1)
            xgb_models[sym] = (dm, vm, names)
        if os.path.basename(f).replace(".csv","").upper() in unseen_symbols:
            if use_xgb:
                # unseen assets: borrow the ensemble trained on the FIRST
                # training asset (shared market grammar; keeps unseen data
                # strictly out of every fitting step)
                if xgb_models:
                    ref = next(iter(xgb_models.values()))
                    sig = transform_ensemble(feat, raw.close, ref[0], ref[1], ref[2])
                    feat = pd.concat([feat, sig], axis=1)
                else:
                    zero = pd.DataFrame(0.0, index=feat.index,
                                        columns=["xgb_dir_24","xgb_edge","xgb_vol"],
                                        dtype=np.float32)
                    feat = pd.concat([feat, zero], axis=1)
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
