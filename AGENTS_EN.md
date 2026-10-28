# AGENTS_EN.md — Successor Agent Guide (English Version)

> **Who is this file for?** Any AI agent (or human developer) who inherits this project after "z-ai/glm" who built it with the owner (alsharif). Read it fully before touching any line. The owner holds you accountable for results; his trust is built on explicit commitments documented here — **never violate them.**
> For the Arabic master version with full context, see `AGENTS.md`. This file is the operational English reference.

---

## 0. Sacred Non-Negotiable Rules

These are the owner's explicit conditions across entire sessions. Violating any = breach of trust:

1. **Spot only. No Futures, no Margin, no Leverage, no Short, no derivatives of any kind.** Reason: the owner's religious conviction (futures are haram). Do not discuss, do not "evolve" toward it, do not suggest it even as an option. Any leverage talk = absolutely forbidden.
2. **No interest of any form**: no staking, no lending, no yield, no lending out the portfolio, no interest on cash. Cash in the system = raw idle cash.
3. **One shared general model** learning from many coins as separate training episodes — NOT a model per coin, NOT a portfolio traded together. One coin per episode.
4. **Technical schools (Elliott/Wyckoff/Gann/Cycles/MTF) are feature inputs — NOT hard-coded BUY/SELL rules.** The model discovers their relationships itself. Manual decision rules are forbidden.
5. **Objective = portfolio dollar value after fees and slippage.** No intermediate technical metric.
6. **No early-stopping, ever**: the system NEVER decides it cannot improve — it saves its best states (best/stable) and keeps trying always. (Owner's literal request after an early-stop ruined a training session.)
7. **Consistency over peak**: one lucky exam peak is worthless. The metric: mean of last 10 evals (`stable`). A consistent +8% model beats a one-off +60%.
8. **Profit purity doctrine**: winning 1000 and losing 800 = **guilty** even if net positive. Losses must stay small relative to gains. (Owner's literal example — implemented via waterfall accounting.)
9. **Losses are bad, profit is required** — but calibrated as a "balanced genius", not a "frozen coward": free breathing zone (1.2% no penalty) + activation bonus against sitting in cash forever. (Historical lesson: harsh penalty dose 1.5 produced a frozen non-trading model — commit 2752c9d, recalibrated in e26c993.)
10. **Validation/test/unseen data is never touched during training** — scaler is fitted on train only. Any future leakage = all results void.
11. **Unseen coins (ADA/DOT/LTC) are reserved for generalization evaluation only** — never enter training.
12. **Any new idea builds on existing learning** — resume from checkpoint — never retrain from scratch (sole exception: radical architecture change — and even then distillation is discussed first).
13. **MCP/OpenCode is an optional control interface** — the system runs independently 24/7 without them.
14. **Live is disabled by default** — enabled explicitly only after passing: Training → Validation → Unseen → Backtest → Paper Trading → small Live Spot then gradual scaling.
15. **Living money is entirely outside the system** — the system trades growth money only. (Owner's agreed financial-independence model: grow capital gradually, no withdrawals before 6 live green months.)

---

## 1. Project State (at time of writing)

| Component | Status |
|---|---|
| Data | 20 coins 1h from 2020 (17 train + 3 unseen) in `data/` — ICP starts 2022-01-11 (first 8 months dropped by owner's request) |
| Features | 92 features/candle (`src/features/engine.py`) — future-leak free (prefix-invariance tested) |
| Model | window-GRU encoder + 2-layer decision GRU + Beta policy & value heads — 301,635 params — `src/model/agent.py` |
| Training | **Paused at update 25,852 / 40,000** (hit 12h session budget, clean stop by design). `best=64.66%` (old early peak), `stable≈0.5-2%` (the real metric) |
| Doctrine | Fully planted (commit e26c993): free breathing 1.2%, quadratic bleed 0.35, purity 0.25, peak ambition, RPE ecstasy, hindsight-clean surgical timing, crash survival, activation bonus, **no early-stop** |
| Tests | **20 passing** — `.venv/bin/python -m pytest tests/ -q` — must stay green always |
| Control | `./trainctl.sh {start|resume|stop|status|watch|ui|log|kill|help}` + curses dashboard `tui.py` |
| Not built yet | Live harness loop (controller.py skeleton), executive OrderManager (skeleton), CCXT live adapter (intentionally locked), any DEX adapter, paper trading loop |

**Diagnostic note**: last 15 evals mostly negative (-2 to -10%) after an early 40-60% jump in the first thousand updates — **probable overfitting pattern**: memorizing training periods instead of understanding markets. Agreed 3-wave counter-plan — wave 1 (adaptive) built & live, wave 2 (fresh 2025-2026 data) implemented 2026-09-18, **wave 2.5 (obs_noise + obs_dropout + weight_decay, owner-approved on math evidence; dropout moved to observation space after network dropout broke the PPO KL guard — see waves) implemented 2026-09-18, training running** — with architecture upgrade (wave 3) reserved for the owner's decision. **Start here when resuming training.**

### Repair wave W — 2026-10-28 (deep analysis + fixes, owner-approved)

**Measured diagnosis (now the reference)**: 27,743 updates (~123 GPU-hours) produced **zero out-of-sample alpha**.
- Reward decomposed term-by-term (reward_audit): the bleed penalty carried **22× the variance of the money signal** (−0.138/step vs money std 0.0074) → PPO learned drawdown-avoidance, not money-making; inactivity_bias 0.00015 = 0.1% of the bleed cost → "cowardice" nearly free — exactly the inversion of doctrine #9 (commit 2752c9d).
- Measurement was broken: the exam was ONE ~8,000-step episode (mismatched with 512-step training) + evaluate/backtest **crashed** on any post-torch.compile checkpoint → the gates were never actually run on these models.
- **Truth matrix (fair, 512h segments)**: latest train alpha +12.5% (t=5.1) vs validation/test/unseen ≈ 0 → **memorization mathematically confirmed**. purity negative in every evaluation without exception.
- signal_probe (W5): a thin but real 24h directional signal (OOS AUC 0.522 mean, 17/17 assets >0.50) → continued training is worth it; hope lies in fee-aware execution management, not richer prediction.

**Fixes implemented** (each unit: tests + commit):
1. **W1 Measurement**: `src/model/ckpt.py` (compat loader) + `src/evaluation/fair_eval.py` (N×512h segments matching training, alpha vs buy-hold on the same candles, bit-exact batch invariance) + trainer.validate() now fair-alpha + `eval_history` persisted inside checkpoints + metric_version (legacy checkpoints restart grading at −inf, weights untouched) + legacy metrics.csv rotated to metrics_pre_fair.csv.
2. **W2 Data**: collector running, gap-filled to 2026-10-28; final Holdout = Oct+ (never trained on; data must be ≤7 days old at any gate).
3. **W3 Doctrine recalibration**: bleed_penalty 0.35→**0.004**, new cash_drag **0.002** (sitting in cash finally costs), timing ×0.10, ecstasy ×0.25 — **money-dominance gate PASS (money variance share 73.7%)** — re-run `reward_audit` after any coefficient change.
4. **W4 recalibration run**: resumed from latest.pt (never from scratch — rule 12) with pre-agreed kill criteria: fair validation alpha ≥ +0.5% and t > 2, purity > 0, train↔OOS gap < 5pp. Failure = data for the report, not for spin.

**Reference**: `logs/eval_baseline.md` (pre-repair numbers).

## 2. Code Map

```
config/default.json       — all settings (all doctrine params live here:
                            bleed_start/max/penalty, purity_coef, ecstasy_cap,
                            timing_horizon, survival_bonus, peak_ambition_coef,
                            crisis_bias, entropy_coef=0.02, num_envs=32,
                            updates=40000, time_budget_hours=8)
src/data.py               — OHLCV download from Binance spot via CCXT
                            (cmd: python -m src.data download ...)
                            Historical note: load_markets works ONLY with
                            options.fetchMarkets=['spot'] (avoids fapi/dapi failures).
src/dataset.py            — manifests + scaler (train only) + chronological
                            70/15/15 split per coin + sample() with window guard
                            + crisis_bias (samples from dynamically-detected
                            >=25% drawdown regions of each close series).
                            Build cmd: python -m src.dataset --raw-dir ... --processed-dir ... --unseen-symbols ADA/USDT DOT/USDT LTC/USDT
src/features/engine.py    — the 92 features. Absolute ban: any look-ahead
                            (rolling center=True is forbidden — was removed once).
                            Tested by prefix-invariance — keep that test green.
src/model/agent.py        — TradingAgent: window-GRU encoder, then decision GRU.
                            Critical invariant: forward([B,1,W,F]) ==
                            forward_sequence([B,T,W,F]) step by step (tested —
                            this was the worst bug in project history: the GRU
                            ran across the env axis instead of time).
src/training/environment.py — env + full doctrine (read the docstring:
                            base/bleed/inactivity/purity/patience/ambition/
                            ecstasy/timing/survival). Every term capped so
                            money always dominates.
src/training/ppo.py       — PPO with correct env-major axis [N,T,...],
                            KL-guard (max_kl=0.02), advantage normalization.
src/trainer.py            — session engine: Trainer class + resume + time
                            budget + validation every 25 updates (eval_every)
                            + best.pt (peak) + stable.pt (mean of last 10 evals
                            × purity — **this is the production file**) +
                            metrics.csv + NO EARLY STOP (sacred rule #6).
src/train.py              — thin CLI wrapper over trainer.py.
src/evaluation/evaluate.py — run() sequential, run_batch() parallel, classify_regime
                            + run_regime (up/down/sideways classified evals —
                            use --regimes), purity in results.
                            **--mode segments (ALWAYS use it for decisions)**:
                            the fair 512h exam matching training (from fair_eval).
src/evaluation/fair_eval.py — W1: fair exam protocol (N×512h rotating segments,
                            alpha vs buy-hold on identical candles, bit-exact
                            batch invariance, stable_score = window-10 × purity).
src/model/ckpt.py          — W1: torch.compile-compatible checkpoint loading
                            (strip/add _orig_mod) — every reader goes through it.
src/evaluation/reward_audit.py — W3: per-step reward decomposition + the
                            money-dominance gate (money std >= 2× every term;
                            must always be PASS).
src/evaluation/signal_probe.py — W5: cheap truth check (24h-direction OOS AUC
                            — before burning GPU).
src/evaluation/backtest.py — evaluation on a specific split.
src/execution/           — safety.py (limits: max_order_fraction,
                            max_daily_loss, stale data), order_manager.py
                            (planning skeleton), ccxt_adapter.py (intentionally
                            locked with enable_live=False until explicit
                            enablement after the gates), exchange.py (ABC).
src/harness/              — state.py (JSON StateStore), decision.py
                            (DecisionEngine: predict(window, hidden) — ready
                            to use), controller.py (skeleton — build the live
                            loop here), lifecycle.py (phase constants).
src/mcp/server.py         — MCP tools: get_status, pause_trading, emergency_stop.
src/live.py               — intentional placeholder. Live loop builds on harness.
trainctl.sh               — all training control from terminal (read help).
tui.py                    — curses dashboard (q quit, p pause display only).
tests/test_system.py      — 20 tests. Rule: any change without a new covering
                            test = forbidden. Any red test = stop and fix now.
```

## 3. Roadmap

### First: resume training (immediate)
```bash
./trainctl.sh resume 12        # continues from 25,852 toward 40,000 (~11-12h GPU left)
```
Watch `stable` (not `best`): consistently climbing above 8-15% = real progress. Stuck near zero or negative after thousands of updates despite the waves below = move to serious overfitting work.

### Anti-overfitting waves (agreed, evolving order):
1. **Adaptive learning**: when mean of last 5 evals declines, lower LR automatically and raise entropy temporarily (**built & live** — `_adapt_on_eval` in trainer.py, controlled by adapt_window/adapt_cooldown in config; alone it hit its floor ~×1/16 LR and ×4 entropy with evals stuck at ~-6% — a structural, not dynamic, problem).
2. **Fresh data**: re-download latest 2025-2026 months for all coins (same data.py commands — they append after deleting old files) then rebuild dataset — model resumes itself on newer data. **Implemented 2026-09-18**: collector filled all 20 coins to 2026-09-18 15:00, dataset rebuilt, training restarted (waves above re-documented in AGENTS.md §2).
3. **Wave 2.5 — explicit regularization** (implemented 2026-09-18 as the structural response to the stuck −6% evals, owner approved on evidence; then corrected by direct measurement):
   - `obs_noise` (default 0.05): additive Gaussian on FEATURE columns only, training envs only (live/validation/golden tests stay at 0.0 → bit-for-bit clean). Never touches the 5 state columns — position/pnl/drawdown/time/frame-flag must stay exact.
   - `obs_dropout` (default 0.1): random zeroing of a fraction of FEATURE cells (dropout in observation space), training envs only, default 0.0. **Measured lesson, not assumption**: we first tried NETWORK dropout (GRU inter-layer + decision-stream head, 0.15) and discovered it BREAKS PPO's KL guard — two dropout masks over the SAME weights yield KL≈1.7 vs the 0.02 cap, so every minibatch stopped early (ent=0.0, n_updates=0 — a frozen "training" worse than none). Regularization must live in OBSERVATION space, where it bakes into the stored obs tensor and the guard sees identical inputs for old/new logprobs, measuring only true drift (after the fix: same-weights KL == 0.0 exactly).
   - `weight_decay` (default 1e-4): Adam L2. **Gotcha wired**: `Adam.load_state_dict` restores the SAVED param_groups (old checkpoints hold 0.0) — the trainer re-asserts the configured value right after resume.
   - `adapt_reset_on_resume` (default true, ONE-time): resets the adaptive LR/entropy state to fresh 1.0/1.0 on resume — the old state was stuck at floor-×1/16 + cap-×4 from the memorization era and would starve the fresh regularized training; learned WEIGHTS are untouched (`stable`/`best` preserved).
   All keys default to 0.0 in the constructors so old checkpoints load unchanged (verified: 0 extra/missing state_dict keys; the model has NO stochastic layers — test `test_model_has_no_stochastic_layers_keeps_ppo_kl_guard_sane` forbids dropout from returning to the network). New unit tests: 41 total green.
   **Immediate-recovery proof**: after the fix `kl` returned from 1.7 to 0.017–0.020 (inside the trust region) and `ent` from 0 to -0.7..-0.9; proper learning resumed from checkpoint 26547.
4. **If still stagnant** (owner's decision only): architecture upgrade (hidden 128→256) with distillation — the old model teaches the new one (teacher logits) before replacement. **Owner's decision required first — not yours.**

### Then — the gates (mandatory order):
```
1. python -m src.evaluation.evaluate --checkpoint models/stable.pt --split validation
2. ... --split test          (never seen by the model)
3. ... --split unseen_test --regimes   (ADA/DOT/LTC + up/down/sideways)
   Agreed success criteria: positive return on test + stable beats buy-hold
   on a reasonable number of assets + positive consistent purity + decent
   behavior in downtrends (small losses — no bottom-chasing)
4. Failing gates = deeper training cycle / more data — NEVER go live.
```

### Building the live loop (Paper first — always):
1. Build the harness loop: hourly candle fetch → FeatureEngine → scaler.transform → last 96 rows (including the 4 portfolio-state columns!) → DecisionEngine.predict → OrderManager.plan → (paper: log only) → StateStore.
2. **Critical warning**: observation = features + 4 state columns (position, pnl, drawdown, time) — copy normalization formulas LITERALLY from `environment.py::_obs()` or decisions are meaningless.
3. Paper for weeks (4-8) — compare live performance to historical. Then present the small-Live decision (200-500 USD) to the owner — **his call is final, not your judgment.**

### Every new change:
A unit test covering it + `pytest tests/ -q` green + descriptive commit. **The model resumes over its learning — never demolish.**

## 4. Exchange Integration — Full Guide (CEX & DEX)

### 4.1 Architecture — why integration is isolated by design

```
Trained model (models/stable.pt)
        ↓  (the model does not know or care which exchange it trades on)
Harness layer (operation loop)
        ↓
Safety layer (the owner's hard limits)
        ↓
Execution / ExchangeAdapter   ←── ONLY this piece changes per exchange
        ↓
Any exchange in the world
```

The adapter contract (`src/execution/exchange.py`):
```python
class ExchangeAdapter(ABC):
    def balance(self): ...          # current funds
    def ticker(self, symbol): ...   # latest price
    def target_position(self, symbol, target): ...  # rebalance to target fraction
    def cancel_all(self, symbol=None): ...          # emergency flatten
```
**Any new exchange = implement these four methods. Nothing else changes.**

### 4.2 Connecting to Binance (primary CEX)

```python
from src.execution.ccxt_adapter import CCXTSpotAdapter
adapter = CCXTSpotAdapter(
    exchange_id="binance",
    api_key=API_KEY,            # trade-only permissions!
    secret=API_SECRET,
    enable_live=True            # explicit opt-in after the gates
)
```

Setup steps:
1. Create API key at Binance → **enable ONLY "Enable Reading" + "Enable Spot Trading"** — NEVER enable withdrawals.
2. Store keys in environment variables (BINANCE_API_KEY / BINANCE_API_SECRET) or a git-ignored file — never commit them.
3. The adapter loads markets; pull `precision`, `limits.amount.min`, `minNotional` from `markets` info and enforce them in OrderManager (round order sizes to valid steps, skip below-minimum orders).
4. Set `defaultType: spot` in exchange options — the codebase historically fails if fapi/dapi endpoints are touched (see src/data.py note).
5. Safety limits before first live order: max_order_fraction (e.g. 0.25), max_daily_loss (e.g. 0.03 → pause 24h), stale-price check (never act on data older than ~2 minutes).

Fees/slippage calibration: Binance spot taker ~0.1% (0.075% with BNB discount — but holding BNB for discounts is optional, not a yield program). Slippage on majors ~0.01-0.05% for reasonable sizes; the training env uses fee=0.001, slippage=0.0005 — verify live fills match, adjust config if needed.

### 4.3 Connecting to Bybit, OKX, KuCoin, Coinbase, Kraken, or ANY CEX

The whole point of CCXT: **the same adapter, just a different exchange_id.**

```python
adapter = CCXTSpotAdapter(exchange_id="bybit", api_key=..., secret=..., enable_live=True)
adapter = CCXTSpotAdapter(exchange_id="okx",    api_key=..., secret=..., enable_live=True)
adapter = CCXTSpotAdapter(exchange_id="kucoin", api_key=..., secret=..., enable_live=True)
```

Per-exchange checklist (do ALL before live):
1. **Spot markets only** — filter `markets` to spot; some exchanges expose derivatives through the same API surface. Enforce `defaultType: spot` and verify the symbol exists as spot.
2. **Fees differ**: Bybit spot 0.1%, OKX 0.08-0.1%, KuCoin 0.1% (0.06% with KCS — again, optional holding, not yield). Update config `fee`/`slippage` to the actual venue and re-run a short evaluation — do not carry Binance's assumptions to a new venue blindly.
3. **Precision/minNotional differ** per exchange — always derive from `market['precision']` and `market['limits']`, never hard-code.
4. **Rate limits differ** — CCXT handles this if `enableRateLimit: true` (already set in the adapter).
5. **Test with a tiny live order first** (5-20 USD) — verify the order path, fill reporting, and balance math end-to-end before scaling.
6. **Historical data for retraining**: the model was trained on Binance 1h candles. Different venues have slightly different candles (minor deviations). For live trading this is fine (features are scale-free). If you retrain on a new venue's data, rebuild the dataset from that venue's downloads for consistency.

### 4.4 Connecting to DEXs (decentralized exchanges) — ICP and beyond

There is NO CCXT for ICP DEXs — each adapter is built manually. General DEX adapter plan:

1. **Pre-flight (mandatory before ANY build)**:
   - Does the venue have documented APIs (REST/canister calls)? No API = no integration, period.
   - Are the smart contracts audited and open?
   - Is liquidity mature enough that slippage for your order size is acceptable?
   - If any check fails: do not integrate, whatever the owner's eagerness — report the findings instead. (The owner awaits an unverified platform called "MultiDex" on ICP — do NOT build on it before it exists and passes these checks.)
2. **Data adapter first (training enrichment)** — lower risk, high value:
   - Fetch OHLCV/trades from the DEX API → write to `data/raw/` in the same CSV format → rebuild dataset with the coin added (or as new unseen assets for generalization testing).
   - DEX data quirks: gappy candles (thin liquidity hours), wilder spikes, shorter history (2-4 years vs 6+). Handle: drop ultra-thin candles, mark regime, or use trade-aggregation instead of raw candles if provided.
3. **Execution adapter** — implement the same 4-method contract:
   - Authentication is THE hard part on DEXs: Internet Identity is human-interactive — for a 24/7 agent you need either a Plug wallet signing flow, a cold operations key with tightly-scoped allowances, or the venue's service-agent mechanism if one exists. Design this with the owner explicitly; never store raw keys in the repo.
   - Order format: many ICP DEXs (ICDex) are order-book style — map `target_position` to limit/market orders; AMM-style ones (ICPSwap) need swap quotes with expected-slippage bounds.
   - Slippage on DEXs is materially higher than CEXs: measure real fills, update config `slippage` upward for honest training, and re-run paper trading on the new venue before any live funds.
4. **Fees on DEXs**: swap fees commonly 0.25-0.3% + gas (cycles on ICP, cheap) — update the env's fee config to the venue's real numbers for any retraining/tuning cycle done specifically for that venue.

### 4.5 The golden sequence for ANY new venue (CEX or DEX)

```
1. Paper trade on the venue's live data (its own candles) for 2-4 weeks.
2. Compare live results vs historical backtest expectations — if materially
   worse (fees/slippage worse than assumed), recalibrate and repeat.
3. Go live with the smallest viable size (owner's explicit approval required).
4. Scale gradually. Owner decides every size increase.
```

## 5. Historical Traps — Never Repeat (paid for in full)

| Trap | Lesson | Protection |
|---|---|---|
| Inverted PPO axis | GRU ran across envs axis instead of time — mathematically corrupt training with no visible symptoms | `test_gru_time_axis_consistency` — never break it |
| Look-ahead features | `rolling(center=True)` saw the future — fake results | `test_features_prefix_invariance` |
| Negative slices | sampling without window margin = random crashes | `dataset.sample()` guard |
| Lethal penalty dose | bleed_penalty=1.5 + purity=0.8 produced a **frozen model** (ent=-15, zero trades, val=0 forever) | current calibration (0.35/0.25) + inactivity tests — if you see ent<-8 and val=0 for long: penalty overdosed again |
| Hidden dominant penalty | bleed 0.35/step carried 22× the money variance across 27k updates with zero OOS alpha (diagnosed 2026-10-28) | reward_audit + money-dominance gate — must always PASS |
| Measuring on the wrong distribution | exam = one 8,000-step episode (training is 512) → best/stable frozen since update 550 | fair_eval (512h segments) — legacy measurement retired |
| Broken gate tool | evaluate/backtest crashed on compiled checkpoints (torch.compile prefix) → gates never actually ran | `src/model/ckpt.py` — every checkpoint read passes through it |
| Nearly-free cowardice | inactivity 0.00015 vs bleed −0.138: sitting in cash cost 0.1% of the pain | cash_drag=0.002 (W3) — doctrine #9 finally balanced |
| Lucky peak locksout | old early-stop compared against the all-time best → an early 26% killed a session | now: no early-stop at all + stable metric |
| Window mean-pooling | lost candle order within the window | current window-GRU encoder |
| Trusting best= as success | passing peaks mean nothing | watch stable only (10-eval mean × purity) |

## 6. Daily Ops Reference

```bash
./trainctl.sh start 12          # start 12h session (auto-resumes if latest.pt exists)
./trainctl.sh stop              # graceful stop (finishes update, saves, exits)
./trainctl.sh status            # state + GPU + last progress + last checkpoint
./trainctl.sh ui                # live dashboard (q to quit)
./trainctl.sh log 20            # last 20 lines
.venv/bin/python -m pytest tests/ -q          # 49 tests — all must be green
# Evaluations (post-training phase — ALL in segments mode):
.venv/bin/python -m src.evaluation.evaluate --checkpoint models/stable.pt --split test --mode segments
.venv/bin/python -m src.evaluation.evaluate --checkpoint models/stable.pt --split unseen_test --mode segments --regimes
# money-dominance gate (must be PASS after any coefficient change):
.venv/bin/python -m src.evaluation.reward_audit --checkpoint models/latest.pt
# quick eval trend check (col 11 = fair alpha, 12 = stable, 13 = purity):
awk -F',' 'NR>1 && $11!="" {print $1, $11}' logs/metrics.csv | tail -20
```

**Reading the progress line**:
```
update N/40000 portfolio=X val=Y best=Z stable=W ent=E kl=K 2.5s
portfolio: mean of training-episode portfolios (starts 10K — natural wobble)
val: latest exam (every 25 updates) | best: all-time peak (comparison only!)
stable: **THE real metric** — mean of last 10 exams × purity — watch this one
ent: exploration vitality. Healthy: -0.5 to -4. Dead: < -8 (check penalty dose)
kl: update stability. |value| < 0.02 typically
```

## 7. Owner Interaction Protocol

- The owner thinks in Arabic and wants human-language explanation before any technical execution.
- **Present the plan → wait for approval word → execute + test + commit.** Never execute a major principle without approval.
- Always give real numbers — no flattery, no exaggeration (he explicitly rejected "guaranteed 90% profit").
- Money decisions (capital size, withdrawals, going live, architecture changes) = **his exclusively**. Your role: honest diagnosis and engineering options.
- His religious commitments (rules 1-2) are non-negotiable in any "developmental" form.
- If you lose contact with him: the system is designed to run alone (expired sessions stop cleanly — nothing breaks).
- The reference for any ambiguity: the sacred rules documented here.

## 8. The Project in One Sentence (its soul)

**A committed spot genius**: it learns alone from all the coins, chases wealth with mathematical ecstasy (RPE) and escalating ambition (successive peaks), but its iron doctrine: losses must always stay small (profit purity), the system never surrenders (no early-stop), and the owner decides the money — while the agent (you now) guards the doctrine and builds upon it, never instead of it.

*Written by glm (z-ai) after ~17 continuous building sessions with the owner alsharif — 2026-09. Last state: training paused at 25,852/40,000 awaiting resume, all tests green, full doctrine alive.*
