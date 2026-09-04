# General Spot Trading Agent

One shared recurrent PPO model learns from many cryptocurrencies as separate episodes. It does not trade multiple assets simultaneously.

## Design
- Spot only: no futures, margin, leverage, shorting, or withdrawals.
- One shared model across all assets; asset identity is not an input.
- Rich, scale-free features: OHLCV, trend, momentum, volatility, volume, price action/structure, Elliott-style swing relationships, Wyckoff-style volume/price states, Gann/time-price relationships, cycles, and multi-timeframe features.
- Features are signals, not hard-coded trading rules. The model decides.
- Hierarchical model: candle encoder -> GRU decision memory -> Beta policy + value head.
- PPO reward: log portfolio-value change after fees/slippage.
- Train/validation/test are chronological per asset; scaler is fitted only on training rows.
- Separate unseen-asset evaluation is supported.
- Harness/execution/MCP are isolated from training and can be enabled later.
- Environment observations include portfolio state (position, pnl, drawdown, time), so the agent learns risk management, not only chart patterns.
- Trades below a deadband (2% of portfolio) are skipped to avoid fee churn.
- PPO updates run the GRU along the true time axis (env-major tensors), with KL guard, LR decay, and checkpoints every 5 minutes.

## Quick start

```bash
cd ~/trading-agent
source .venv/bin/activate
pip install -r requirements.txt

# Download training assets (add many symbols for a richer general model)
python -m src.data download --symbols BTC/USDT ETH/USDT SOL/USDT XRP/USDT --timeframe 1h --since 2020-01-01

# Build feature files and manifests (hold out symbols for unseen-asset testing)
python -m src.dataset build --raw-dir data/raw --processed-dir data/processed --unseen-symbols ADA/USDT

# Train the shared model (default time budget: 8h; pass 6/12 or any hours)
python -m src.train --config config/default.json --time-budget-hours 6

# Resume training (safe after power loss; checkpoints every 5 minutes)
python -m src.train --config config/default.json --resume models/latest.pt

# Evaluate checkpoints: best.pt is picked by validation return
python -m src.evaluation.evaluate --config config/default.json --checkpoint models/best.pt
python -m src.evaluation.evaluate --split test --checkpoint models/best.pt
python -m src.evaluation.evaluate --split unseen_test --checkpoint models/best.pt
python -m src.evaluation.backtest --split test --checkpoint models/best.pt
```

## Monitoring
- `logs/metrics.csv` — per-update training metrics (portfolio, entropy, KL, val return).
- `models/best.pt` — highest validation return so far.
- `models/latest.pt` — most recent checkpoint (resume point).

## Session training
- `python -m src.train --time-budget-hours 6` — bounded session, clean stop, latest.pt saved.
- Early stopping: training stops if validation return hasn't improved for `patience_updates` updates (see config).
- Deterministic seeds across envs and updates; resumable checkpoints.

## Live path
The production path is intentionally separated:
`model -> harness -> safety -> execution -> exchange adapter`.

MCP exposes monitoring/control tools to OpenCode without making OpenCode a dependency of live trading.

Live trading is disabled by default.

## Tests
```bash
python -m pytest tests/ -q
# or: python tests/test_system.py
```
