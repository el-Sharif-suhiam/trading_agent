import numpy as np, pandas as pd, torch
from src.features import FeatureEngine
from src.model import TradingAgent
from src.training.environment import SpotTradingEnv

def test_features_no_future_fill():
    n=300; t=pd.date_range("2020-01-01",periods=n,freq="h",tz="UTC"); c=np.linspace(100,130,n); df=pd.DataFrame({"datetime":t,"open":c*.99,"high":c*1.01,"low":c*.98,"close":c,"volume":np.ones(n)})
    f=FeatureEngine().transform(df); assert np.isfinite(f.values).all(); assert len(f)==n

def test_agent():
    m=TradingAgent(80); x=torch.randn(4,8,96,80); d,v,h=m.forward_sequence(x); a=d.sample(); assert a.shape==(4,8) and v.shape==(4,8) and h.shape==(2,4,128)

def test_env():
    f=np.random.randn(1000,80).astype(np.float32); c=np.linspace(100,150,1000); e=SpotTradingEnv(f,c,0,1000,96,100); o,_=e.reset(options={"start":95}); no,r,t,tr,i=e.step(np.array([.5],np.float32)); assert no.shape==(96,80); assert np.isfinite(r)

if __name__=="__main__":
    test_features_no_future_fill(); test_agent(); test_env(); print("OK")
