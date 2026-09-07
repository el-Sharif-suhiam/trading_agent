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
        # multi-frame window layout: fine(1h) + 4h + 12h + 1d + weekly rows
        # `window` means the FINE (1h) portion of the observation.
        # Context ladder: 160h (~6.7d) + 4d(4h) + 4d(12h) + 30d(daily)
        # + ~56d(weekly) — the agent sees ~3 months of structure in
        # ~230 rows instead of bloating the fine window.
        self.window_fine = window
        self.n_daily = 30   # 30 daily rows = 30 days of context
        self.n_weekly = 8   # 8 weekly rows  = ~8 weeks of context
        self.obs_rows = window + 24 + 8 + self.n_daily + self.n_weekly
        n_state = 5  # position, pnl, drawdown, time fraction, frame flag
        self.observation_space=spaces.Box(-np.inf,np.inf,shape=(self.obs_rows,features.shape[1]+n_state),dtype=np.float32)
        self.action_space=spaces.Box(0.,1.,shape=(1,),dtype=np.float32)

    def _coarse_frame(self, rows, group, cache_attr):
        """Sliding coarse frame via cumulative sums — EXACTLY equivalent to
        the legacy `features[i-g*rows+1:i+1].reshape(rows,g,-1).mean(1)`
        layout (groups anchor at i and slide each step; the v2 model was
        trained on this layout so bit-compatibility matters), computed in
        O(rows*F) instead of O(rows*group*F) via a running cumsum."""
        C = self._csum
        lo0 = self._csum_lo  # absolute index of _csum[0]
        i = self.i
        # group k (0=oldest..rows-1=last): consecutive non-overlapping groups,
        # the last one ending exactly at i. chunk k spans
        # [i-g*(rows-k)+1 .. i-g*(rows-k-1)]  (inclusive ends).
        ends = i - group * (rows - 1 - np.arange(rows))      # inclusive
        hi = C[ends - lo0]
        lo = np.empty_like(hi)
        lo[1:] = C[ends[:-1] - lo0]          # start_k - 1 = end_{k-1}
        lo[0] = C[ends[0] - group - lo0]     # start_0 - 1 (always >= lo0 by margin)
        return ((hi - lo) / group).astype(np.float32)

    def _obs(self):
        # fine frame: last `window_fine` 1h candles (default 160 = ~6.7 days)
        wf = getattr(self, 'window_fine', 160)
        market = self.features[self.i - wf + 1:self.i + 1]
        # coarse frames: past-only aggregation (no look-ahead), incrementally
        # cached per group boundary — each coarse row is the mean of its
        # underlying fine features (scale-free already).
        f4 = self._coarse_frame(24, 4, "f4")
        f12 = self._coarse_frame(8, 12, "f12")
        fd = self._coarse_frame(self.n_daily, 24, "fd")
        fw = self._coarse_frame(self.n_weekly, 168, "fw")
        v = self._value(); pos = self.asset * self.close[self.i] / max(v, 1e-12)
        pnl = (v / self.initial_cash - 1.0)
        dd = v / max(self.peak, 1e-12) - 1.0
        tf = self.steps / max(self.episode_steps, 1)
        # portfolio state row — built once, tiled to all frames (one array
        # op instead of five np.full calls per frame per step)
        srow = np.array([pos*2-1, float(np.tanh(pnl*5)), dd*10, tf*2-1], dtype=np.float32)
        row_fine = np.concatenate([srow, np.zeros(1, dtype=np.float32)])
        row_coarse = np.concatenate([srow, np.ones(1, dtype=np.float32)])
        state_fine = np.broadcast_to(row_fine, (wf, 5))
        state_c4 = np.broadcast_to(row_coarse, (24, 5))
        state_c12 = np.broadcast_to(row_coarse, (8, 5))
        state_fd = np.broadcast_to(row_coarse, (self.n_daily, 5))
        state_fw = np.broadcast_to(row_coarse, (self.n_weekly, 5))
        return np.concatenate([
            np.concatenate([market, state_fine], axis=1),
            np.concatenate([f4, state_c4], axis=1),
            np.concatenate([f12, state_c12], axis=1),
            np.concatenate([fd, state_fd], axis=1),
            np.concatenate([fw, state_fw], axis=1),
        ], axis=0)

    @staticmethod
    def batch_obs(envs):
        """Build observations for many envs in ONE call (hot path)."""
        n = len(envs)
        rows = envs[0].obs_rows
        cols = envs[0].features.shape[1] + 5
        out = np.empty((n, rows, cols), dtype=np.float32)
        for j, e in enumerate(envs):
            out[j] = e._obs()
        return out

    def _value(self): return self.cash+self.asset*self.close[self.i]

    def reset(self,seed=None,options=None):
        super().reset(seed=seed)
        # the weekly frame needs 168*8=1344 candles of history; fine 160,
        # 12h 96, daily 720 — the weekly margin dominates.
        need = max(self.window_fine, 96, 168 * self.n_weekly)
        max_start=self.end-self.episode_steps-1
        if options and "start" in options: self.i=int(options["start"])
        else: self.i=int(self.np_random.integers(self.start+need-1,max_start+1)) if max_start>self.start+need-1 else self.start+need-1
        self.cash=self.initial_cash; self.asset=0.; self.steps=0; self.trades=0
        self.fees=0.; self.slippage_paid=0.
        self.peak=self.initial_cash; self.max_drawdown=0.
        self._coarse_cache = {}
        # running cumulative sum of features: _csum[k - base] = sum(features[0..k])
        # for k in [base, end]. base = reset index — only the span this
        # episode can ever touch is materialized (full-asset arrays x 32 envs
        # were the RAM leak that degraded every cycle after the first).
        # _csum_prev holds sum(features[0..base-1]) for subtractions below base.
        self._csum_base = self.i
        # margin needed below base: the deepest reach of any coarse frame
        # (weekly: 168*8 rows) plus one for the subtraction row
        margin = 168 * 8 + 1
        lo_base = max(0, self._csum_base - margin)
        span = max(1, self.end - lo_base)
        self._csum = np.empty((span, self.features.shape[1]), dtype=np.float64)
        # rows BELOW base are needed only as subtraction prefixes — one row
        # per hour of margin is wasteful, but margin (1345) is tiny vs
        # episode length and keeps absolute indexing exact.
        self._csum_lo = lo_base  # absolute index of _csum[0]
        np.cumsum(self.features[lo_base:self.i + 1].astype(np.float64),
                  axis=0, out=self._csum[:self.i + 1 - lo_base])
        self._csum_valid = self.i
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
        # extend the running cumsum for the (possibly clamped) new index
        if self._csum_valid < self.i:
            s = self._csum_valid + 1 - self._csum_lo
            e = self.i + 1 - self._csum_lo
            block = self.features[self._csum_valid + 1:self.i + 1].astype(np.float64)
            np.cumsum(block, axis=0, out=self._csum[s:e])
            self._csum[s:e] += self._csum[s - 1] if s > 0 else 0.0
            self._csum_valid = self.i
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
