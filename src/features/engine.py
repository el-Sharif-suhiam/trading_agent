import numpy as np
import pandas as pd


def _rsi(s, n=14):
    d=s.diff(); up=d.clip(lower=0).ewm(alpha=1/n, adjust=False).mean(); dn=(-d.clip(upper=0)).ewm(alpha=1/n, adjust=False).mean()
    return 100 - 100/(1 + up/(dn+1e-12))

def _atr(df,n=14):
    pc=df.close.shift(); tr=pd.concat([(df.high-df.low),(df.high-pc).abs(),(df.low-pc).abs()],axis=1).max(axis=1)
    return tr.ewm(alpha=1/n,adjust=False).mean()

def _z(s,n):
    m=s.rolling(n).mean(); sd=s.rolling(n).std(); return (s-m)/(sd+1e-12)

def _slope(s,n):
    return (s-s.shift(n))/((s.shift(n)).abs()+1e-12)

class FeatureEngine:
    """All features use current/past candles only. They are scale-free where possible."""
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        x=df.copy(); c=x.close; v=x.volume; ret=c.pct_change()
        out=pd.DataFrame(index=x.index)
        # Raw normalized candle geometry
        out["oc"]=(x.open-c)/(c+1e-12); out["hc"]=(x.high-c)/(c+1e-12); out["lc"]=(x.low-c)/(c+1e-12)
        out["body"]=(c-x.open)/(x.high-x.low+1e-12); out["range"]=(x.high-x.low)/(c+1e-12)
        for n in [1,2,3,6,12,24,48,96]: out[f"ret_{n}"]=c.pct_change(n)
        # Trend / momentum
        for n in [5,10,20,50,100,200]:
            ma=c.ewm(span=n,adjust=False).mean(); out[f"ema_dist_{n}"]=(c-ma)/(c+1e-12); out[f"ema_slope_{n}"]=_slope(ma,min(n,20))
        rsi=_rsi(c); out["rsi"]=(rsi-50)/50
        e12=c.ewm(span=12,adjust=False).mean(); e26=c.ewm(span=26,adjust=False).mean(); macd=e12-e26; sig=macd.ewm(span=9,adjust=False).mean()
        out["macd"] = macd/(c+1e-12); out["macd_signal"]=(macd-sig)/(c+1e-12)
        ll=x.low.rolling(14).min(); hh=x.high.rolling(14).max(); out["stoch"]=(c-ll)/(hh-ll+1e-12)*2-1; out["williams"]=(hh-c)/(hh-ll+1e-12)*-2+1
        tp=(x.high+x.low+c)/3; sma_tp=tp.rolling(20).mean(); out["cci"]=(tp-sma_tp)/(0.015*(tp-sma_tp).abs().rolling(20).mean()+1e-12)
        atr=_atr(x); out["atr_pct"]=atr/(c+1e-12); out["atr_change"]=atr.pct_change()
        mid=c.rolling(20).mean(); sd=c.rolling(20).std(); out["bb_pos"]=(c-mid)/(2*sd+1e-12); out["bb_width"]=4*sd/(mid.abs()+1e-12)
        # Volume
        out["vol_z"]=_z(np.log1p(v),48); out["vol_ratio_20"]=v/(v.rolling(20).mean()+1e-12)
        obv=(np.sign(ret.fillna(0))*v).cumsum(); out["obv_z"]=_z(obv,48)
        vwap=(tp*v).rolling(48).sum()/(v.rolling(48).sum()+1e-12); out["vwap_dist"]=(c-vwap)/(c+1e-12)
        # Price action / structure
        for n in [12,24,48,96]:
            rh=x.high.rolling(n).max(); rl=x.low.rolling(n).min(); out[f"dist_high_{n}"]=(c-rh)/(c+1e-12); out[f"dist_low_{n}"]=(c-rl)/(c+1e-12); out[f"range_pos_{n}"]=(c-rl)/(rh-rl+1e-12)
        out["up_break_20"]=(c-x.high.shift(1).rolling(20).max())/(atr+1e-12); out["down_break_20"]=(c-x.low.shift(1).rolling(20).min())/(atr+1e-12)
        # Confirmed swing points only: a 5-candle swing needs 2 candles after it,
        # so compare against the swing computed up to candle t-2 (no look-ahead).
        swing_hi=x.high.rolling(5).max().shift(2); swing_lo=x.low.rolling(5).min().shift(2)
        out["swing_high_flag"]=(x.high>=swing_hi).astype(float); out["swing_low_flag"]=(x.low<=swing_lo).astype(float)
        out["hh_hl_proxy"]=_slope(x.high.rolling(10).max(),10); out["ll_lh_proxy"]=_slope(x.low.rolling(10).min(),10)
        # Elliott-style: swing legs and Fibonacci relationships, expressed as hypotheses
        leg=c-c.shift(5); prev=c.shift(5)-c.shift(10); out["elliott_impulse_proxy"]=leg/(atr*5+1e-12); out["elliott_alternation"]=np.tanh((leg.abs()-prev.abs())/(atr*5+1e-12))
        out["fib_038"]=(c-(c.shift(10)+0.382*(c-c.shift(10))))/(atr*5+1e-12); out["fib_0618"]=(c-(c.shift(10)+0.618*(c-c.shift(10))))/(atr*5+1e-12); out["fib_1618"]=(c-(c.shift(10)+1.618*(c-c.shift(10))))/(atr*5+1e-12)
        # Wyckoff-style effort/result, springs/upthrusts as continuous signals
        effort=(v/(v.rolling(20).mean()+1e-12)); result=(x.high-x.low)/(atr+1e-12); out["wyckoff_effort_result"]=effort/(result+1e-12); out["wyckoff_climax"]=(effort-1)*np.tanh((result-1)); out["wyckoff_spring"]=(x.low-x.low.rolling(20).min().shift(1))/(atr+1e-12); out["wyckoff_upthrust"]=(x.high-x.high.rolling(20).max().shift(1))/(atr+1e-12)
        # Gann/time-price relationships: normalized angle-like and square/cycle distances
        for n in [9,18,36,72]: out[f"gann_time_price_{n}"]=(c-c.shift(n))/(atr*n+1e-12)
        out["gann_sqrt_distance"]=(np.sqrt(c.clip(lower=1e-12))-np.sqrt(c.shift(1).clip(lower=1e-12)))/(np.sqrt(c.shift(1).clip(lower=1e-12))+1e-12)
        # Cycles / time
        for n in [24,48,96,168]:
            out[f"cycle_ret_corr_{n}"]=ret.rolling(n).corr(ret.shift(n)).fillna(0); out[f"cycle_phase_{n}"]=np.sin(2*np.pi*(np.arange(len(x))%n)/n); out[f"cycle_phase_cos_{n}"]=np.cos(2*np.pi*(np.arange(len(x))%n)/n)
        # Multi-timeframe aggregation from 1h data
        for n in [4,12,24]:
            rr=c.pct_change(n); out[f"mtf_ret_{n}"]=rr; out[f"mtf_vol_{n}"]=rr.rolling(n).std(); out[f"mtf_rsi_{n}"]=(_rsi(c,n)-50)/50
        out=out.replace([np.inf,-np.inf],np.nan).fillna(0).astype(np.float32)
        return out
