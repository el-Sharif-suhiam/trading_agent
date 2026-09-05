import numpy as np
import gymnasium as gym
from gymnasium import spaces

class SpotTradingEnv(gym.Env):
    """Single-asset Spot environment. Action is target asset allocation in [0,1].

    Core philosophy (planted doctrine):
      1. Total return still matters (log portfolio change after fee/slippage).
      2. BLEEDING PENALTY: losses are punished progressively harder — the deeper
         the drawdown from episode peak, the bigger the extra penalty. This
         teaches fast loss-cutting ("small losses, as if they never existed").
      3. PURITY BONUS: gross gains vs gross losses ratio is rewarded — a model
         that wins 1000 and loses 800 is punished even though net positive
         (user's explicit doctrine: loss value must stay far below gain value).
      4. TREND PATIENCE BONUS: holding a profitable position in an uptrend is
         rewarded with a small per-step bonus so winners are let to run.

    Trading fills happen at the close of the decision candle with fee+slippage,
    then the portfolio is marked at the next candle's close.
    """
    metadata={"render_modes":[]}

    def __init__(self, features, close, start, end, window=96, episode_steps=512,
                 fee=.001, slippage=.0005, initial_cash=10000., min_trade_fraction=0.02,
                 bleed_start=0.005, bleed_max=0.05, bleed_penalty=1.5,
                 purity_coef=0.8, trend_patience_coef=0.15):
        self.features=features; self.close=close; self.start=start; self.end=end
        self.window=window; self.episode_steps=episode_steps
        self.fee=fee; self.slippage=slippage; self.initial_cash=initial_cash
        self.min_trade_fraction=min_trade_fraction
        # doctrine params
        self.bleed_start=bleed_start        # drawdown below which penalty starts (normal breathing)
        self.bleed_max=bleed_max            # drawdown at which penalty reaches max
        self.bleed_penalty=bleed_penalty     # max extra penalty magnitude
        self.purity_coef=purity_coef         # weight of the purity shaping term
        self.trend_patience_coef=trend_patience_coef
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
        self.cash=self.initial_cash; self.asset=0.; self.steps=0; self.trades=0
        self.fees=0.; self.slippage_paid=0.
        self.peak=self.initial_cash; self.max_drawdown=0.
        # doctrine trackers
        self.hwm=None; self.trough=None            # set on first post-fee mark
        self.gross_gain=0.0; self.gross_loss=0.0
        return self._obs(),{}

    # ---- doctrine shaping terms ----
    def _bleed_penalty(self, value):
        """Progressive penalty on drawdown below episode peak.
        0 until bleed_start (normal breathing), grows smoothly to
        bleed_penalty at bleed_max and beyond."""
        dd = 1.0 - value/max(self.peak,1e-12)       # current drawdown fraction
        if dd <= self.bleed_start: return 0.0
        x = min((dd - self.bleed_start)/(self.bleed_max - self.bleed_start), 1.0)
        return -self.bleed_penalty * x * x          # quadratic ramp: gentle then harsh

    def _track_waterfall(self, value, basis=None):
        """Waterfall gain/loss accounting: record new highs as gross gains and
        new lows below the last high-water mark as gross losses. On the first
        mark, `basis` (pre-decision portfolio value) seeds the HWM so the first
        step's move is counted honestly as gain or loss (user doctrine:
        1000 win / 800 loss must be punished vs 1000 / 300)."""
        if self.hwm is None and basis is not None:
            self.hwm = float(basis); self.trough = float(basis)
        elif self.hwm is None:
            self.hwm = value; self.trough = value
        elif value > self.hwm:
            self.gross_gain += value - self.hwm
            self.hwm = value; self.trough = value
        elif value < self.trough:
            self.gross_loss += self.trough - value
            self.trough = value
        G, L = self.gross_gain, self.gross_loss
        if G + L < 1e-12: return 0.0
        purity = (G - L)/(G + L)
        return self.purity_coef * purity * 0.01

    def _trend_patience(self, old_value, new_value):
        """Small bonus for staying invested while the position is profitable
        and the market keeps rising — teaches 'let winners run'."""
        if self.asset <= 0: return 0.0
        if new_value <= old_value: return 0.0
        invested = self.asset*self.close[self.i]/max(new_value,1e-12)
        if invested < 0.5: return 0.0
        return self.trend_patience_coef * invested * 0.005

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
        # ---- reward = doctrine ----
        base=float(np.log(max(new,1e-12)/max(old,1e-12)))
        bleed=self._bleed_penalty(new)
        # waterfall basis: the pre-decision value (post previous fees) is the
        # honest starting mark — the first step's move counts as gain or loss.
        purity=self._track_waterfall(new, basis=old if self.hwm is None else None)
        patience=self._trend_patience(old,new)
        reward=base+bleed+purity+patience
        terminated=self.i>=self.end-1
        truncated=self.steps>=self.episode_steps and not terminated
        info={"portfolio_value":new,"trades":self.trades,"fees":self.fees,
              "drawdown":self.max_drawdown,
              "position":self.asset*self.close[self.i]/max(new,1e-12),
              "purity": ((self.gross_gain-self.gross_loss)/(self.gross_gain+self.gross_loss))
                        if (self.gross_gain+self.gross_loss)>1e-12 else 0.0,
              "gross_gain":self.gross_gain,"gross_loss":self.gross_loss,}
        return self._obs(),reward,terminated,truncated,info
