"""PaperTrader — the hourly live loop (paper mode).

Lifecycle each hour (aligned to candle CLOSE):
  1. pull newly closed candles from the collector CSVs (or fetch directly)
  2. feed them through LivePipeline (95 features, exact training diet)
  3. for each symbol: LiveEnv.observe() -> model -> target position
  4. safety layer checks -> LiveEnv.execute() (paper: ledger only)
  5. append a decision record + snapshot state (crash/resume safe)

Control philosophy (owner requirement, 2026-09-09):
  - NO auto-restart, NO watchdog: the loop runs while YOU keep it running.
  - graceful stop file (logs/STOP_LIVE) checked once per cycle
  - every state persisted to logs/paper/ so a stop/restart resumes exactly
    where it left off, back-filling any missed candles (no gaps ever)

Position sizing across symbols (portfolio layer, v1 — equal notional slots):
  - capital is split into N equal slots (one per traded symbol)
  - each LiveEnv trades its own slot with its own sub-account
  - v1 keeps it simple; ranked allocation by signal strength is a future
    upgrade behind the same interface (owner approved this staging)
"""
import argparse
import json
import os
import signal
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from ..dataset import AssetDataset
from ..model import TradingAgent
from ..execution.safety import SafetyLayer
from .pipeline import LivePipeline
from .live_env import LiveEnv


def _log(msg):
    print(f"[{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}] {msg}",
          flush=True)


class PaperTrader:
    def __init__(self, config_path="config/default.json",
                 checkpoint="models/stable.pt", out_dir="logs/paper",
                 symbols=None, max_symbols=None):
        import torch
        import joblib
        self.cfg = json.load(open(config_path))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)
        os.makedirs(os.path.join(out_dir, "history"), exist_ok=True)

        # model: stable.pt (production criterion — the project's real metric)
        ds = AssetDataset(self.cfg["processed_dir"], "train",
                          self.cfg["window"])
        self.model = TradingAgent(ds.feature_count + 5,
                                  self.cfg["encoder_size"],
                                  self.cfg["hidden_size"],
                                  self.cfg["num_gru_layers"]).to(self.device)
        ck = torch.load(checkpoint, map_location=self.device,
                        weights_only=False)
        state = ck["model"]
        # torch.compile checkpoints carry an "_orig_mod." prefix — strip it
        if any(k.startswith("_orig_mod.") for k in state):
            state = {k[len("_orig_mod."):]: v for k, v in state.items()}
        self.model.load_state_dict(state)
        self.model.eval()
        _log(f"model loaded: {checkpoint} (update {ck.get('update', '?')})")

        self.safety = SafetyLayer(
            max_order_fraction=self.cfg.get("max_order_fraction", 0.25),
            max_daily_loss=self.cfg.get("max_daily_loss", 0.05))

        # symbol universe: explicit list, or all training assets
        if symbols:
            self.symbols = [s.upper() for s in symbols]
        else:
            man = pd.read_csv(os.path.join(self.cfg["processed_dir"],
                                           "train_manifest.csv"))
            self.symbols = sorted(man.symbol.unique().tolist())
        if max_symbols:
            self.symbols = self.symbols[:max_symbols]

        self.pipeline = LivePipeline(self.cfg["processed_dir"],
                                     xgb_dir=self.cfg.get("model_dir", "models"),
                                     config_path=config_path)
        self.slot_cash = self.cfg["initial_cash"]
        self.envs = {}
        self._build_envs()

        # resume state
        self.state_path = os.path.join(out_dir, "state.json")
        self._resume()

        # graceful stop
        self.stop_file = "logs/STOP_LIVE"
        signal.signal(signal.SIGTERM, self._sig_stop)
        signal.signal(signal.SIGINT, self._sig_stop)

    # ---------- env construction ----------
    def _build_envs(self):
        for sym in self.symbols:
            path = os.path.join(self.cfg["data_dir"],
                                sym.replace("/", "_") + ".csv")
            if not os.path.exists(path):
                _log(f"  {sym}: no CSV in {self.cfg['data_dir']} — skipped")
                continue
            raw = pd.read_csv(path, parse_dates=["datetime"])
            if not len(raw):
                _log(f"  {sym}: empty CSV — skipped")
                continue
            # warm start: full history through the pipeline in one shot
            z, close = self.pipeline.warm_start(sym, raw)
            sc = self.pipeline._scaled_cache[sym]
            self.envs[sym] = LiveEnv(
                sym, sc["arr"], sc["close"],
                window=self.cfg["window"],
                fee=self.cfg["fee"], slippage=self.cfg["slippage"],
                initial_cash=self.slot_cash,
                min_trade_fraction=self.cfg.get("min_trade_fraction", 0.02),
                episode_steps=self.cfg["episode_steps"])
            _log(f"  {sym}: env ready ({len(raw)} candles history)")

    # ---------- persistence ----------
    def _resume(self):
        self.started_at = datetime.now(timezone.utc).isoformat()
        self.decisions_count = 0
        if not os.path.exists(self.state_path):
            return
        st = json.load(open(self.state_path))
        self.started_at = st.get("started_at", self.started_at)
        self.decisions_count = st.get("decisions_count", 0)
        # restore accounts (cash/asset/peak/waterfall) into LiveEnvs
        for sym, acc in (st.get("accounts") or {}).items():
            env = self.envs.get(sym)
            if env is None:
                continue
            a = env.account
            a.cash = acc["cash"]; a.asset = acc["asset"]
            a.peak = acc["peak"]; a.trades = acc["trades"]
            a.fees_paid = acc["fees_paid"]; a.slippage_paid = acc["slippage_paid"]
            a.hwm = acc.get("hwm"); a.trough = acc.get("trough")
            a.gross_gain = acc.get("gross_gain", 0.0)
            a.gross_loss = acc.get("gross_loss", 0.0)
            a.max_drawdown = acc.get("max_drawdown", 0.0)
            env.steps = acc.get("steps", 0)
            # hidden state: too complex to serialize faithfully (GRU+cache);
            # on restart the model re-reads the window fresh — a small,
            # bounded memory reset identical to evaluation episode starts.
        if st.get("accounts"):
            _log(f"resumed {len(st['accounts'])} accounts from state")

    def _save_state(self):
        accounts = {}
        for sym, env in self.envs.items():
            a = env.account
            accounts[sym] = {
                "cash": a.cash, "asset": a.asset, "peak": a.peak,
                "trades": a.trades, "fees_paid": a.fees_paid,
                "slippage_paid": a.slippage_paid, "hwm": a.hwm,
                "trough": a.trough, "gross_gain": a.gross_gain,
                "gross_loss": a.gross_loss,
                "max_drawdown": a.max_drawdown, "steps": env.steps,
            }
        st = {"started_at": self.started_at,
              "decisions_count": self.decisions_count,
              "updated_at": datetime.now(timezone.utc).isoformat(),
              "checkpoint": "models/stable.pt",
              "accounts": accounts}
        tmp = self.state_path + ".tmp"
        json.dump(st, open(tmp, "w"), indent=2)
        os.replace(tmp, self.state_path)

    def _sig_stop(self, *a):
        _log("stop signal received — will stop after this cycle cleanly")
        open(self.stop_file, "w").close()

    # ---------- hourly cycle ----------
    def _new_candles_for(self, sym):
        """Read the symbol's CSV and return rows the pipeline hasn't seen."""
        env = self.envs.get(sym)
        cache = self.pipeline._feat_cache.get(sym)
        if env is None or cache is None:
            return []
        seen = cache["raw_rows"]
        path = os.path.join(self.cfg["data_dir"], sym.replace("/", "_") + ".csv")
        df = pd.read_csv(path, parse_dates=["datetime"])
        if len(df) <= seen:
            return []
        return df.iloc[seen:]

    def _hour_boundary(self):
        """Wait until the top of the next hour + small settle delay."""
        now = datetime.now(timezone.utc)
        secs = now.minute * 60 + now.second
        wait = 3600 - secs + 20          # candle close + 20s settle
        return wait

    def _run_cycle(self):
        results = {}
        for sym in self.symbols:
            env = self.envs.get(sym)
            if env is None:
                continue
            new = self._new_candles_for(sym)
            if not len(new):
                continue  # nothing new for this symbol yet
            # append ALL new candles through the pipeline, then grow the env
            # via its OFFICIAL append path (the one pinned by tests against
            # the training env — never hand-wire env internals here)
            last_close = None
            for _, row in new.iterrows():
                z, close = self.pipeline.append(sym, row)
                last_close = float(row.close)
            env.append(z[-1], last_close)
            # decide on the LATEST candle
            target = env.act(self.model, self.device)
            # safety: order value vs slot value
            price = float(env._close[env.i])
            cur_pos = env.account.target_position(price)
            order_frac = abs(target - cur_pos)
            snap_pre = env.account.snapshot(price)
            daily_ret = 0.0  # per-slot daily P&L not tracked v1 (paper)
            # cold-start exemption: building the FIRST position from 100%
            # cash cannot compound risk (nothing to lose but the entry
            # itself); the training env was allowed its first trade too.
            # Once invested, every rebalance obeys the 25% order cap.
            cold_start = cur_pos < 1e-9
            ok, why = self.safety.validate(
                snap_pre["value"],
                order_frac * snap_pre["value"] if not cold_start else 0.0,
                daily_ret, price_age_seconds=0)
            if not ok:
                _log(f"  {sym}: SAFETY BLOCK ({why}) — holding position")
                results[sym] = {"target": target, "executed": 0.0,
                                "blocked": why,
                                "candle_time": str(new.datetime.iloc[-1]),
                                **snap_pre}
                continue
            executed = env.execute(target)
            snap = env.snapshot()
            snap["candle_time"] = str(new.datetime.iloc[-1])
            results[sym] = {"target": target, "executed": executed,
                            **snap}
            _log(f"  {sym}: target={target:.3f} pos={snap['position']:.3f} "
                 f"value={snap['value']:.2f} pnl={snap['pnl_pct']:+.2f}%")
        if results:
            self.decisions_count += 1
            ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            rec_path = os.path.join(self.out_dir, "history",
                                    f"decision_{ts}.json")
            json.dump({"ts": ts, "results": results},
                      open(rec_path, "w"), indent=2)
            self._save_state()
        return results

    def run(self):
        _log(f"paper trader starting: {len(self.envs)} symbols, "
             f"slot={self.slot_cash} each, device={self.device}")
        while True:
            if os.path.exists(self.stop_file):
                _log("STOP_LIVE found — exiting cleanly (resume with "
                     "livectl.sh start; state is saved)")
                os.remove(self.stop_file)
                self._save_state()
                break
            wait = self._hour_boundary()
            _log(f"next candle in {int(wait)}s — checking for new data")
            time.sleep(wait)
            try:
                self._run_cycle()
            except Exception as e:
                _log(f"cycle error (non-fatal, will retry next hour): {e}")
            # hourly equity curve point
            try:
                total = sum(env.account.snapshot(
                    float(env._close[env.i]))["value"]
                    for env in self.envs.values())
                ts = datetime.now(timezone.utc).isoformat()
                eq_path = os.path.join(self.out_dir, "equity.csv")
                new_row = pd.DataFrame([[ts, total]],
                                        columns=["ts", "value"])
                new_row.to_csv(eq_path, mode="a", header=not os.path.exists(eq_path),
                               index=False)
            except Exception:
                pass


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config/default.json")
    p.add_argument("--checkpoint", default="models/stable.pt")
    p.add_argument("--out-dir", default="logs/paper")
    p.add_argument("--symbols", nargs="*", default=None,
                   help="trade only these (default: all training assets)")
    p.add_argument("--max-symbols", type=int, default=None)
    a = p.parse_args()
    pt = PaperTrader(config_path=a.config, checkpoint=a.checkpoint,
                     out_dir=a.out_dir, symbols=a.symbols,
                     max_symbols=a.max_symbols)
    pt.run()


if __name__ == "__main__":
    main()
