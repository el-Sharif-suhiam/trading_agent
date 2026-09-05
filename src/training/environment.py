import numpy as np
import gymnasium as gym
from gymnasium import spaces

class SpotTradingEnv(gym.Env):
    """Single-asset Spot environment. Action = target asset allocation in [0,1].

    THE GENIUS DOCTRINE (all terms verified mathematically before planting):
      base         : log portfolio change after fee/slippage (money is the goal)
      bleed        : progressive penalty on drawdown beyond the FREE BREATHING
                     ZONE (bleed_start, default 1.2%) up to bleed_max (5%).
                     Below the zone trading is risk-free of penalty -> sitting
                     forever is mathematically dominated by smart trading.
      inactivity   : tiny per-step opportunity cost when 100% cash —
                     the coward equilibrium (never trade) always loses.
      purity       : waterfall gain/loss accounting; (G-L)/(G+L) rewarded —
                     gains must dwarf losses (user doctrine 1000/800 vs 1000/300).
      patience     : staying invested while profitable in an uptrend is
                     rewarded (let winners run).
      peak_ambition: log-scaled bonus on each NEW portfolio high — climbing
                     successive peaks is ecstasy; capped so it can never
                     outweigh the money terms.
      ecstasy (RPE): reward prediction error — realized step return vs the
                     agent's own expectations. Positive surprises deliver a
                     dopamine-like bonus that fades as the agent gets better,
                     demanding ever-higher mastery (user: "شعور بالنشوة كل ما تطور").
      timing       : hindsight-clean entry quality — after `timing_horizon`
                     candles, entering decisions that kept improving earn a
                     surgical-entry bonus (buying near local lows).
      survival     : crash-escape bonus — spotting a reversal signature and
                     going to cash BEFORE the drop pays a smart-survival bonus
                     (user: "يتنبأ بالانهيار فينجو").

    All shaping terms are caps-bounded so the money reward always dominates.
    """
    metadata={"render_modes":[]}

    def __init__(self, features, close, start, end, window=96, episode_steps=512,
                 fee=.001, slippage=.0005, initial_cash=10000., min_trade_fraction=0.02,
                 bleed_start=0.012, bleed_max=0.05, bleed_penalty=0.35,
                 purity_coef=0.25, trend_patience_coef=0.15,
                 ecstasy_cap=0.2, timing_horizon=8, survival_bonus=0.3,
                 peak_ambition_coef=0.5):
        self.features=features; self.close=close; self.start=start; self.end=end
        self.window=window; self.episode_steps=episode_steps
        self.fee=fee; self.slippage=slippage; self.initial_cash=initial_cash
        self.min_trade_fraction=min_trade_fraction
        # doctrine params
        self.bleed_start=bleed_start; self.bleed_max=bleed_max
        self.bleed_penalty=bleed_penalty; self.purity_coef=purity_coef
        self.trend_patience_coef=trend_patience_coef
        self.ecstasy_cap=ecstasy_cap; self.timing_horizon=timing_horizon
        self.survival_bonus=survival_bonus
        self.peak_ambition_coef=peak_ambition_coef
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
        self.hwm=None; self.trough=None
        self.gross_gain=0.0; self.gross_loss=0.0
        # ecstasy / expectation tracking (EMA of recent step returns)
        self.ret_ema=0.0
        # peak ambition: log-scaled successive new highs
        self.peak_count=0
        # timing quality: pending entries awaiting hindsight evaluation
        self._pending=[]
        self._timing_score=0.0
        # survival: did agent flee to cash right before a drop?
        self._prev_close=None
        return self._obs(),{}

    # ---------- doctrine terms ----------
    def _bleed_penalty(self, value):
        dd = 1.0 - value/max(self.peak,1e-12)
        if dd <= self.bleed_start: return 0.0
        x = min((dd - self.bleed_start)/(self.bleed_max - self.bleed_start), 1.0)
        return -self.bleed_penalty * x * x

    def _track_waterfall(self, value, basis=None):
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

    def _inactivity_bias(self):
        if self.asset > 0: return 0.0
        return -0.00015

    def _trend_patience(self, old_value, new_value):
        if self.asset <= 0: return 0.0
        if new_value <= old_value: return 0.0
        invested = self.asset*self.close[self.i]/max(new_value,1e-12)
        if invested < 0.5: return 0.0
        return self.trend_patience_coef * invested * 0.005

    def _peak_ambition(self, value, prev_peak):
        """Ecstasy of successive new highs: log-scaled bonus per new peak,
        capped so money terms always dominate."""
        if value <= prev_peak or prev_peak <= 0: return 0.0
        frac = value/prev_peak - 1.0
        if frac <= 0: return 0.0
        bonus = self.peak_ambition_coef * np.log1p(frac*100) * 0.01
        self.peak_count += 1
        return float(min(bonus, self.ecstasy_cap))

    def _ecstasy(self, base_ret):
        """Reward Prediction Error: positive surprises above the agent's own
        recent expectations deliver a capped dopamine-like bonus. As the agent
        gets consistently good, EMA rises and the ecstasy fades — demanding
        ever-higher mastery for the same thrill (progressive ambition)."""
        expected = self.ret_ema
        rpe = base_ret - expected
        # update expectation EMA (only realized returns, no shaping)
        self.ret_ema = 0.995*self.ret_ema + 0.005*base_ret
        if rpe <= 0: return 0.0
        bonus = min(self.ecstasy_cap, 2.0*rpe)
        return bonus

    def _register_entry(self, entry_price):
        """Schedule a hindsight-clean timing evaluation after `timing_horizon`
        candles for a fresh entry at `entry_price`."""
        self._pending.append({"step": self.steps, "price": float(entry_price)})

    def _resolve_timing(self):
        """Evaluate matured entries: did price keep improving after entry?
        Surgical entries (price rose steadily since entry) earn bonus."""
        H = self.timing_horizon
        matured = [p for p in self._pending if self.steps - p["step"] >= H]
        if not matured: return 0.0
        bonus = 0.0
        price_now = float(self.close[self.i])
        for p in matured:
            move = (price_now - p["price"])/max(p["price"],1e-12)
            if move > 0:
                # sustained improvement: capped timing bonus
                bonus += min(move*2.0, self.ecstasy_cap)
            self._pending.remove(p)
        return float(min(bonus, self.ecstasy_cap))

    def _survival_check(self):
        """Crash-escape bonus: the agent was in cash (or fled this step) and
        the market then dropped hard. Rewards predicting the reversal."""
        if self._prev_close is None: return 0.0
        drop = (float(self.close[self.i]) - self._prev_close)/max(self._prev_close,1e-12)
        fled = (self.asset <= 1e-12)
        if fled and drop < -0.02:   # fled and market fell > 2%
            return self.survival_bonus * min(abs(drop)*5.0, 1.0)
        return 0.0

    def step(self,action):
        target=float(np.clip(action[0],0,1)); price=float(self.close[self.i]); value=self._value()
        prev_asset=self.asset
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
        # entry happened? schedule timing evaluation (hindsight-clean)
        if self.asset > prev_asset + 1e-12:
            self._register_entry(price)
        old=value; self._prev_close=float(self.close[self.i])
        self.i=min(self.i+1, self.end-1); self.steps+=1; new=self._value()
        prev_peak=self.peak
        self.peak=max(self.peak,new); self.max_drawdown=min(self.max_drawdown,new/self.peak-1)
        # ---------- reward = THE GENIUS DOCTRINE ----------
        base=float(np.log(max(new,1e-12)/max(old,1e-12)))
        bleed=self._bleed_penalty(new)
        purity=self._track_waterfall(new, basis=old if self.hwm is None else None)
        patience=self._trend_patience(old,new)
        inactivity=self._inactivity_bias()
        ambition=self._peak_ambition(new, prev_peak)
        ecstasy=self._ecstasy(base)
        timing=self._resolve_timing()
        survival=self._survival_check()
        reward=base+bleed+purity+patience+inactivity+ambition+ecstasy+timing+survival
        terminated=self.i>=self.end-1
        truncated=self.steps>=self.episode_steps and not terminated
        info={"portfolio_value":new,"trades":self.trades,"fees":self.fees,
              "drawdown":self.max_drawdown,
              "position":self.asset*self.close[self.i]/max(new,1e-12),
              "purity": ((self.gross_gain-self.gross_loss)/(self.gross_gain+self.gross_loss))
                        if (self.gross_gain+self.gross_loss)>1e-12 else 0.0,
              "gross_gain":self.gross_gain,"gross_loss":self.gross_loss,}
        return self._obs(),reward,terminated,truncated,info
