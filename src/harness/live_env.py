"""LiveEnv — the live trading body around the trained brain.

It wraps SpotTradingEnv's EXACT observation math (sacred: the model was
trained on _obs() from environment.py — any drift here = meaningless
decisions) while replacing the training-only parts:
  - features/close arrays grow as new candles arrive (append, never rewrite)
  - `advance()` moves the clock one candle WITHOUT letting the training
    `step()` trade on it — the live account is updated by the harness,
    not by the env
  - `execute()` applies the model's target position to the live account
    with the SAME fee/slippage/deadband math as training, so live behavior
    == training behavior
  - position/pnl/drawdown state columns are recomputed from the live
    account, and `steps` grows without episode truncation (the live market
    never truncates)

One LiveEnv per symbol; each keeps its own model hidden state (the brain
carries memory across candles exactly like evaluation does).
"""
import numpy as np

from ..training.environment import SpotTradingEnv


class LiveAccount:
    """Cash + asset position with training-identical execution math."""

    def __init__(self, initial_cash=10000.0, fee=0.001, slippage=0.0005,
                 min_trade_fraction=0.02):
        self.initial_cash = float(initial_cash)
        self.cash = float(initial_cash)
        self.asset = 0.0
        self.fee = fee
        self.slippage = slippage
        self.min_trade_fraction = min_trade_fraction
        self.peak = float(initial_cash)
        self.trades = 0
        self.fees_paid = 0.0
        self.slippage_paid = 0.0
        # per-trade waterfall purity tracking (same as training env, mv3):
        # average entry basis of the current position (cost incl. fees) and
        # realized round-trip G/L accumulation
        self.basis_qty = 0.0
        self.basis_cost = 0.0
        self.gross_gain = 0.0
        self.gross_loss = 0.0
        self.max_drawdown = 0.0

    def value(self, price):
        return self.cash + self.asset * price

    def target_position(self, price):
        v = self.value(price)
        return self.asset * price / max(v, 1e-12)

    def trade_to_target(self, target, price):
        """Exactly the env.step() trade block: same deadband, same fees,
        same slippage. `target` is quantized to float32 FIRST — training
        receives actions as float32 (np.float32(0.8) == 0.80000001...) so
        live must apply the same quantization for identical results."""
        target = float(np.clip(np.float32(target), 0.0, 1.0))
        value = self.value(price)
        desired_asset = value * target / price
        delta = desired_asset - self.asset
        executed = 0.0
        if abs(delta) * price > max(self.min_trade_fraction * value, 1e-9):
            if delta > 0:
                gross = delta * price * (1 + self.slippage)
                cost = gross / (1 - self.fee)
                if cost > self.cash:
                    delta = self.cash * (1 - self.fee) / (price * (1 + self.slippage))
                    gross = delta * price * (1 + self.slippage)
                    cost = gross / (1 - self.fee)
                self.cash -= cost
                self.asset += delta
                executed = delta
                # merge into average entry basis (identical to env.step())
                self.basis_cost += cost
                self.basis_qty += delta
                self.fees_paid += gross * self.fee
                self.slippage_paid += delta * price * self.slippage
                self.trades += 1
            else:
                qty = -delta
                gross = qty * price * (1 - self.slippage)
                net = gross * (1 - self.fee)
                # realized round-trip P&L vs average entry basis (identical
                # to env.step() — the golden bit-for-bit parity depends on it)
                avg = self.basis_cost / max(self.basis_qty, 1e-12)
                realized = net - qty * avg
                if realized > 0:
                    self.gross_gain += realized
                elif realized < 0:
                    self.gross_loss += -realized
                self.basis_cost -= qty * avg
                self.basis_qty = max(0.0, self.basis_qty - qty)
                if self.basis_qty < 1e-12:
                    self.basis_cost = 0.0
                self.asset -= qty
                self.cash += net
                executed = -qty
                self.fees_paid += gross * self.fee
                self.slippage_paid += qty * price * self.slippage
                self.trades += 1
        v = self.value(price)
        self.peak = max(self.peak, v)
        self.max_drawdown = min(self.max_drawdown, v / self.peak - 1.0)
        return executed

    @property
    def purity(self):
        g, l = self.gross_gain, self.gross_loss
        if g + l < 1e-12:
            return 0.0
        return (g - l) / (g + l)

    def snapshot(self, price):
        v = self.value(price)
        return {
            "cash": self.cash, "asset": self.asset, "value": v,
            "position": self.asset * price / max(v, 1e-12),
            "pnl_pct": (v / self.initial_cash - 1.0) * 100.0,
            "max_drawdown": self.max_drawdown, "trades": self.trades,
            "fees_paid": self.fees_paid, "purity": self.purity,
        }


class LiveEnv:
    """One symbol's live environment: growing arrays + env's exact _obs()
    math + model hidden state. The harness drives it candle by candle."""

    def __init__(self, symbol, features, close, window=160, fee=0.001,
                 slippage=0.0005, initial_cash=10000.0,
                 min_trade_fraction=0.02, episode_steps=512):
        self.symbol = symbol
        # underlying arrays GROW: we keep a reference the env reads through
        self._features = features          # [rows, 95] (scaled)
        self._close = close                # [rows]
        # dummy env instance purely to reuse its _obs()/frame math; we set
        # the reset()-computed fields MANUALLY (reset() itself would redo
        # the whole cumsum materialization — wasteful and it also randomizes
        # `i`). Fields _obs() needs: i, plus the cumsum frame machinery.
        self._env = SpotTradingEnv(
            self._features, self._close, 0, len(self._close),
            window, episode_steps, fee, slippage, initial_cash,
            min_trade_fraction,
        )
        self.window_fine = self._env.window_fine
        self.account = LiveAccount(initial_cash, fee, slippage,
                                   min_trade_fraction)
        self.hidden = None                  # model memory (per-symbol)
        self.i = len(self._close) - 1      # current candle index
        self.steps = 0
        self._init_env_cursors()

    def _init_env_cursors(self):
        """Set exactly the fields _obs() reads, without reset()'s sampling:
        cursor, portfolio trackers, and the coarse-frame cumsum cache."""
        env = self._env
        env.i = self.i
        env.steps = self.steps
        env.cash = self.account.cash
        env.asset = self.account.asset
        env.initial_cash = self.account.initial_cash
        env.peak = self.account.peak
        env.fees = 0.0; env.slippage_paid = 0.0; env.trades = 0
        # coarse-frame cache: materialize the cumsum over the span the
        # frames can reach (weekly frame reaches back 168*8 rows from `i`)
        margin = 168 * 8 + 1
        lo = max(0, self.i - margin)
        span = max(1, self.i - lo + 1)
        env._csum = np.empty((span, self._features.shape[1]), dtype=np.float64)
        np.cumsum(self._features[lo:self.i + 1].astype(np.float64),
                  axis=0, out=env._csum)
        env._csum_lo = lo
        env._csum_valid = self.i
        env.basis_qty = 0.0; env.basis_cost = 0.0
        env.gross_gain = 0.0; env.gross_loss = 0.0
        env.ret_ema = 0.0
        env.peak_count = 0
        env._pending = []
        env._timing_score = 0.0
        env._prev_close = None

    # ---------- growth ----------
    def append(self, scaled_row, close_price):
        """One new closed candle (already scaled by LivePipeline.append)."""
        self._features = np.vstack([self._features, scaled_row[None].astype(np.float32)])
        self._close = np.append(self._close, float(close_price))
        self.i = len(self._close) - 1
        # keep the env's internal references pointed at the grown arrays
        self._env.features = self._features
        self._env.close = self._close
        self._env.end = self.i + 1
        # advance the env's internal cursor to the new candle WITHOUT trading
        self._env.i = self.i
        self._env.steps = self.steps
        # extend the env's cumsum cache incrementally (same math as its step)
        if self._env._csum_valid < self.i:
            s = self._env._csum_valid + 1 - self._env._csum_lo
            e = self.i + 1 - self._env._csum_lo
            block = self._features[self._env._csum_valid + 1:self.i + 1].astype(np.float64)
            if self._env._csum.shape[0] < e:
                grown = np.zeros((e, self._features.shape[1]), dtype=np.float64)
                grown[:self._env._csum.shape[0]] = self._env._csum
                self._env._csum = grown
            np.cumsum(block, axis=0, out=self._env._csum[s:e])
            self._env._csum[s:e] += self._env._csum[s - 1] if s > 0 else 0.0
            self._env._csum_valid = self.i

    # ---------- observation (training-identical) ----------
    def _sync_account_state(self):
        price = float(self._close[self.i])
        v = self.account.value(price)
        # copy account state INTO the env trackers so _obs() state columns
        # reflect the LIVE account, not a training episode
        self._env.asset = self.account.asset
        self._env.cash = self.account.cash
        self._env.initial_cash = self.account.initial_cash
        self._env.peak = self.account.peak
        self._env.steps = self.steps

    def observe(self):
        """Build the observation for the model. Identical math to env._obs():
        fine window + coarse frames + live account state columns."""
        self._sync_account_state()
        return self._env._obs()

    # ---------- decision + execution ----------
    def act(self, model, device):
        """Run the model on the current observation; returns target in [0,1]
        (Beta mean policy — deterministic, same as evaluation)."""
        import torch
        obs = self.observe()
        with torch.no_grad():
            x = torch.tensor(obs[None, None], device=device)
            dist, _, self.hidden = model(x, self.hidden)
            mean = (dist.concentration1 /
                    (dist.concentration1 + dist.concentration0)).item()
        return float(np.clip(mean, 0.0, 1.0))

    def execute(self, target):
        """Apply the model's target to the live account at current price.
        Returns executed delta (quantity, signed)."""
        price = float(self._close[self.i])
        executed = self.account.trade_to_target(target, price)
        self.steps += 1
        return executed
    def snapshot(self):
        price = float(self._close[self.i])
        snap = self.account.snapshot(price)
        snap["symbol"] = self.symbol
        snap["candle_time"] = None  # filled by the harness (it knows df)
        snap["price"] = price
        snap["steps"] = self.steps
        return snap
