# General Spot Trading Agent

One shared recurrent PPO model learns from many cryptocurrencies as separate episodes. It does not trade multiple assets simultaneously.

## Design
- Spot only: no futures, margin, leverage, shorting, or withdrawals.
- One shared model across all assets; asset identity is not an input.
- Rich, scale-free features: OHLCV, trend, momentum, volatility, volume, price action/structure, Elliott-style swing relationships, Wyckoff-style volume/price states, Gann/time-price relationships, cycles, and multi-timeframe features.
- Features are signals, not hard-coded trading rules. The model decides.
- Hierarchical model: candle/window encoder -> GRU decision memory -> Beta policy + value head.
- PPO reward: log portfolio-value change after fees/slippage.
- Train/validation/test are chronological per asset; scaler is fitted only on training rows.
- Separate unseen-asset evaluation is supported.
- Harness/execution/MCP are isolated from training and can be enabled later.

## Quick start

```bash
cd ~/trading-agent
source .venv/bin/activate
pip install -r requirements.txt

# Download training assets
python -m src.data download --symbols BTC/USDT ETH/USDT SOL/USDT XRP/USDT --timeframe 1h --since 2020-01-01

# Build feature files and manifests
python -m src.dataset build --raw-dir data/raw --processed-dir data/processed

# Train shared model
python -m src.train --config config/default.json

# Evaluate a checkpoint
python -m src.evaluation.evaluate --config config/default.json --checkpoint models/best.pt
```

The downloader accepts arbitrary Spot symbols. Add more assets later and rerun dataset building/training; the model remains shared.

## Live path

The production path is intentionally separated:
`model -> harness -> safety -> execution -> exchange adapter`.
MCP exposes monitoring/control tools to OpenCode without making OpenCode a dependency of live trading.

Live trading is disabled by default.
