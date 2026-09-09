#!/usr/bin/env bash
# livectl.sh — MANUAL control of the live data collector + paper trader.
# Philosophy: the owner starts/stops everything by hand. No auto-restart,
# no watchdog. Stopping is always clean; state is saved and resumable.
#
# Usage:
#   ./livectl.sh collect          # start the continuous data collector
#   ./livectl.sh paper [syms..]   # start paper trading (default: all training symbols)
#   ./livectl.sh stop             # graceful stop collector + paper
#   ./livectl.sh kill             # hard kill (last resort)
#   ./livectl.sh status           # what's running + last decisions + data freshness
#   ./livectl.sh log [N]          # last N paper-trader lines
#   ./livectl.sh dlog [N]         # last N collector lines
#   ./livectl.sh help

set -u
cd "$(dirname "$0")"
PY=.venv/bin/python
PAPER_LOG=logs/paper.log
COLLECT_LOG=logs/collector.log
STOP=logs/STOP_LIVE
PAPER_DIR=logs/paper

paper_running() { pgrep -f "src.harness.paper_trader" > /dev/null 2>&1; }
collect_running() { pgrep -f "src.harness.collector" > /dev/null 2>&1; }

case "${1:-help}" in
  collect)
    if collect_running; then echo "collector already running"; exit 0; fi
    rm -f logs/STOP_COLLECT
    nohup $PY -m src.harness.collector >> "$COLLECT_LOG" 2>&1 &
    echo "collector started (pid $!) — hourly candles flow into data/raw continuously"
    ;;
  paper)
    if paper_running; then echo "paper trader already running"; exit 0; fi
    shift 0 2>/dev/null || true
    SYMS=""
    [ "$#" -gt 1 ] && SYMS="$*"
    rm -f "$STOP"
    if [ -n "$SYMS" ]; then
      nohup $PY -m src.harness.paper_trader --symbols $SYMS >> "$PAPER_LOG" 2>&1 &
    else
      nohup $PY -m src.harness.paper_trader >> "$PAPER_LOG" 2>&1 &
    fi
    echo "paper trader started (pid $!) — hourly decisions, paper money only"
    echo "log: $PAPER_LOG   state: $PAPER_DIR/state.json"
    ;;
  stop)
    [ -f "$STOP" ] || touch "$STOP"
    [ -f logs/STOP_COLLECT ] || touch logs/STOP_COLLECT
    if paper_running; then
      echo "paper: stop requested — exits after current cycle (clean save)"
    else
      echo "paper: not running"
    fi
    if collect_running; then
      pkill -INT -f "src.harness.collector" 2>/dev/null && echo "collector: stopping (may take a cycle)"
    else
      echo "collector: not running"
    fi
    ;;
  kill)
    pkill -9 -f "src.harness.paper_trader" 2>/dev/null
    pkill -9 -f "src.harness.collector" 2>/dev/null
    echo "hard-killed. state on disk is from the last completed cycle (resume-safe)"
    ;;
  status)
    echo "=== paper trader"
    if paper_running; then
      echo "RUNNING (pid $(pgrep -f 'src.harness.paper_trader' | head -1))"
      [ -f "$PAPER_DIR/state.json" ] && $PY - <<'EOF'
import json
try:
    s = json.load(open("logs/paper/state.json"))
    accs = s.get("accounts") or {}
    tot = 0.0
    for sym, a in sorted(accs.items()):
        # recompute value needs a price; use stored cash+asset*0 fallback markers
        print(f"  {sym}: cash={a['cash']:.0f} asset={a['asset']:.4f} trades={a['trades']}")
    print("  decisions so far:", s.get("decisions_count", 0))
    print("  last update:", s.get("updated_at", "?"))
EOF
    else
      echo "NOT RUNNING"
    fi
    echo "=== collector"
    if collect_running; then echo "RUNNING (pid $(pgrep -f 'src.harness.collector' | head -1))"; else echo "NOT RUNNING"; fi
    echo "=== data freshness (training universe)"
    for f in data/raw/BTC_USDT.csv data/raw/ETH_USDT.csv; do
      [ -f "$f" ] && echo "  $(basename $f): last candle $(tail -1 $f | cut -d, -f1)"
    done
    ;;
  log)
    tail -"${2:-20}" "$PAPER_LOG" 2>/dev/null || echo "(no paper log yet)"
    ;;
  dlog)
    tail -"${2:-15}" "$COLLECT_LOG" 2>/dev/null || echo "(no collector log yet)"
    ;;
  *)
    cat <<'EOF'
livectl.sh — manual control of the live harness (paper mode)

Commands:
  collect         Start the continuous data collector (runs forever, fills gaps
                  automatically; safe alongside training)
  paper [syms]    Start the paper trader (hourly decisions; default all training
                  symbols; e.g. ./livectl.sh paper BTC/USDT ETH/USDT)
  stop            Graceful stop of both (paper exits after current cycle)
  kill            Hard kill (last resort — resume-safe either way)
  status          Running state + accounts + data freshness
  log [N]         Last N paper-trader log lines
  dlog [N]        Last N collector log lines

Notes:
  - NO auto-restart: if stopped, it STAYS stopped until you run start again.
  - Paper state saves to logs/paper/ — stop/restart resumes accounts exactly.
  - Live real-money trading stays disabled behind the gates (sacred rule #14).
EOF
    ;;
esac
