#!/usr/bin/env bash
# trainctl.sh — full control of training from the terminal, no external tools.
# Usage:
#   ./trainctl.sh start [hours]   # launch training (default hours from config: 8)
#   ./trainctl.sh stop            # graceful stop: finishes current update, saves, exits
#   ./trainctl.sh status          # is it running? + last progress line
#   ./trainctl.sh watch           # live progress (Ctrl+C to exit, does NOT stop training)
#   ./trainctl.sh kill            # hard kill (last resort; resume still works)
#   ./trainctl.sh resume [hours]  # continue from models/latest.pt
#   ./trainctl.sh log [N]         # last N progress lines (default 15)

set -u
cd "$(dirname "$0")"
PY=.venv/bin/python
TRAIN_LOG=logs/train.log
STOP=logs/STOP

is_running() { pgrep -f "src.trainer" > /dev/null 2>&1 || pgrep -f "src.train" > /dev/null 2>&1; }

case "${1:-help}" in
  start|resume)
    HOURS="${2:-}"
    if is_running; then echo "already running (pid $(pgrep -f src.train | head -1)) — use stop first"; exit 1; fi
    ARGS="--config config/default.json"
    if [ "$1" = resume ] || [ -f models/latest.pt ]; then
      # resume if asked OR if a previous session left a checkpoint (prevents accidental restart-from-zero)
      ARGS="$ARGS --resume models/latest.pt"
      [ "$1" = start ] && echo "found models/latest.pt — resuming previous training (use kill+delete latest.pt to truly restart)"
    fi
    [ -n "$HOURS" ] && ARGS="$ARGS --time-budget-hours $HOURS"
    rm -f "$STOP"
    nohup $PY -m src.train $ARGS >> "$TRAIN_LOG" 2>&1 &
    echo "training started (pid $!) log: $TRAIN_LOG"
    echo "opening dashboard (q to exit dashboard — training keeps running)..."
    sleep 2
    $PY tui.py
    ;;
  stop)
    if ! is_running; then echo "not running"; exit 1; fi
    touch "$STOP"
    echo "stop requested — finishing current update, saving checkpoint, then exiting"
    echo "(takes up to ~1 min; check with: ./trainctl.sh status)"
    ;;
  kill)
    pkill -9 -f "src.trainer" 2>/dev/null; pkill -9 -f "src.train" 2>/dev/null
    echo "hard-killed. resume later with: ./trainctl.sh resume"
    ;;
  status)
    if is_running; then
      echo "RUNNING (pid $(pgrep -f 'src.train|src.trainer' | head -1))"
      nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>/dev/null | sed 's/^/GPU: /'
    else
      echo "NOT RUNNING"
      [ -f models/latest.pt ] && echo "last checkpoint: update $($PY -c "import torch;print(torch.load('models/latest.pt',map_location='cpu',weights_only=False).get('update','?'))" 2>/dev/null)"
    fi
    echo "--- last progress:"
    grep "^update" "$TRAIN_LOG" 2>/dev/null | tail -1 || echo "(no progress yet)"
    ;;
  watch)
    tail -f "$TRAIN_LOG" 2>/dev/null || echo "no log yet — start training first"
    ;;
  ui)
    if ! command -v nvidia-smi > /dev/null; then echo "(nvidia-smi not found; GPU row will be hidden)"; fi
    $PY tui.py
    ;;
  log)
    grep "^update" "$TRAIN_LOG" 2>/dev/null | tail -"${2:-15}" || echo "(no progress yet)"
    ;;
  help|--help|-h)
    cat <<'EOF'
trainctl.sh — control the training loop from the terminal

Usage: ./trainctl.sh <command> [args]

Commands:
  start [hours]   Start training. Optional hours limit (default: config time_budget_hours)
  resume [hours]  Resume from models/latest.pt (after stop/crash/power-off)
  stop            Graceful stop: finishes current update, saves checkpoint, exits (~1 min)
  kill            Hard kill (last resort; resume still works afterwards)
  status          Running? + GPU usage + last progress line + last checkpoint update
  watch           Live progress log (Ctrl+C exits watch, training keeps running)
  ui              Graphical live dashboard in the terminal (charts, GPU, stats)
  log [N]         Show last N progress lines (default 15)
  help            This help screen

Examples:
  ./trainctl.sh start 12     # train up to 12 hours
  ./trainctl.sh stop         # safe stop
  ./trainctl.sh resume 6     # continue where it stopped
  ./trainctl.sh status       # check state anytime
  ./trainctl.sh ui           # beautiful live dashboard
EOF
    ;;
  *)
    sed -n '2,10p' "$0"
    ;;
esac
