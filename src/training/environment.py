import numpy as np
import gymnasium as gym
from gymnasium import spaces

class SpotTradingEnv(gym.Env):
    """Single-asset Spot environment. Action is target asset allocation in [0,1].

    Trading fills happen at the close of the decision candle (same candle the
    agent observes) with fee+slippage applied, then the portfolio is marked at
    the next candle's close. Observations include portfolio state so the
    agent can learn risk management, not only market patterns.
    """
    metadata={"render_modes":[]}

    def __init__(self, features, close, start, end, window=96, episode_steps=512,
                 fee=.001, slippage=.0005, initial_cash=10000., min_trade_fraction=0.02):
        self.features=features; self.close=close; self.start=start; self.end=end
        self.window=window; self.episode_steps=episode_steps
        self.fee=fee; self.slippage=slippage; self.initial_cash=initial_cash
        self.min_trade_fraction=min_trade_fraction
        n_state = 4  # position, unrealized pnl, drawdown, time fraction
        self.observation_space=spaces.Box(-np.inf,np.inf,shape=(window,features.shape[1]+n_state),dtype=np.float32)
        self.action_space=spaces.Box(0.,1.,shape=(1,),dtype=np.float32)

    def _obs(self):
        market=self.features[self.i-self.window+1:self.i+1]
        v=self._value(); pos=self.asset*self.close[self.i]/max(v,1e-12)
        pnl=(v/self.initial_cash-1.0)
        dd=v/max(self.peak,1e-12)-1.0
        tf=self.steps/max(self.episode_steps,1)
        state=np.full((self.window,4),[pos*2-1, float(np.tanh(pnl*5)), dd*10, tf*2-1],dtype=np.float32)
        return np.concatenate([market,state],axis=1)

    def _value(self): return self.cash+self.asset*self.close[self.i]

    def reset(self,seed=None,options=None):
        super().reset(seed=seed)
        max_start=self.end-self.episode_steps-1
        if options and "start" in options: self.i=int(options["start"])
        else: self.i=int(self.np_random.integers(self.start+self.window-1,max_start+1)) if max_start>self.start+self.window-1 else self.start+self.window-1
        self.cash=self.initial_cash; self.asset=0.; self.steps=0; self.trades=0; self.fees=0.; self.slippage_paid=0.; self.peak=self.initial_cash; self.max_drawdown=0.
        return self._obs(),{}

    def step(self,action):
        target=float(np.clip(action[0],0,1)); price=float(self.close[self.i]); value=self._value()
        desired_asset=value*target/price
        delta=desired_asset-self.asset
        if abs(delta)*price > max(self.min_trade_fraction*value,1e-9):
            if delta>0:
                gross=delta*price*(1+self.slippage); cost=gross/(1-self.fee)
                if cost>self.cash:
                    delta=self.cash*(1-self.fee)/(price*(1+self.slippage)); gross=delta*price*(1+self.slippage); cost=gross/(1-self.fee)
                self.cash-=cost; self.asset+=delta; self.fees+=gross*self.fee; self.slippage_paid+=delta*price*self.slippage; self.trades+=1
            else:
                qty=-delta; gross=qty*price*(1-self.slippage); net=gross*(1-self.fee)
                self.asset-=qty; self.cash+=net; self.fees+=gross*self.fee; self.slippage_paid+=qty*price*self.slippage; self.trades+=1
        old=value; self.i=min(self.i+1, self.end-1); self.steps+=1; new=self._value()
        self.peak=max(self.peak,new); self.max_drawdown=min(self.max_drawdown,new/self.peak-1)
        reward=float(np.log(max(new,1e-12)/max(old,1e-12)))
        terminated=self.i>=self.end-1
        truncated=self.steps>=self.episode_steps and not terminated
        info={"portfolio_value":new,"trades":self.trades,"fees":self.fees,
              "drawdown":self.max_drawdown,
              "position":self.asset*self.close[self.i]/max(new,1e-12)}
        return self._obs(),reward,terminated,truncated,info
