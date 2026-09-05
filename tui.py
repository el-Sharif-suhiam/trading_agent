#!/usr/bin/env python3
"""Live training dashboard (TUI). Usage: .venv/bin/python tui.py [--interval 2]

Keys:  q or Esc = quit  |  p = pause scrolling
Safe: reading only — never stops training.
"""
import curses, os, sys, time
import torch, psutil

ROOT = os.path.dirname(os.path.abspath(__file__))
LOG = os.path.join(ROOT, "logs", "train.log")
CSV = os.path.join(ROOT, "logs", "metrics.csv")
MODELS = os.path.join(ROOT, "models")


def training_pid():
    for p in psutil.process_iter(["pid", "cmdline"]):
        try:
            cl = p.info["cmdline"] or []
            if len(cl) >= 3 and "src.train" in " ".join(cl):
                return p
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return None


def last_lines(path, n=40):
    try:
        with open(path, "rb") as fh:
            fh.seek(0, os.SEEK_END); size = fh.tell()
            fh.seek(max(0, size - 40000))
            return fh.read().decode(errors="replace").splitlines()[-n:]
    except OSError:
        return []


def parse_progress(lines):
    for line in reversed(lines):
        if line.startswith("update "):
            parts = line.split()
            d = {"raw": line}
            try:
                upd, _, tot = parts[1].partition("/")
                d["update"] = upd; d["total"] = tot
                for kv in parts[2:]:
                    k, _, v = kv.partition("=")
                    if k in ("portfolio", "val", "best", "stable", "ent", "kl"):
                        d[k] = v
                d["sec"] = parts[-1]
            except Exception:
                pass
            return d
    return None


def checkpoint_update():
    try:
        ck = torch.load(os.path.join(MODELS, "latest.pt"), map_location="cpu",
                        weights_only=False)
        return ck.get("update", "?")
    except Exception:
        return "-"


def read_csv_history(limit=400):
    rows = []
    try:
        with open(CSV) as fh:
            lines = fh.read().splitlines()[1:]  # header
        for ln in lines[-limit:]:
            f = ln.split(",")
            if len(f) >= 11:
                rows.append(f)
    except OSError:
        pass
    return rows


def val_series(rows):
    xs, ys = [], []
    for i, f in enumerate(rows):
        if len(f) >= 11 and f[10] and f[10] not in ("", "-"):
            try:
                xs.append(i); ys.append(float(f[10]))
            except ValueError:
                pass
    return xs, ys


def spark(vals, width, height):
    """ASCII vertical bar chart of val series."""
    if not vals or width < 8 or height < 2:
        return ["(no validation data yet — first eval at update 25)"]
    vals = vals[-width:]
    lo, hi = min(vals), max(vals)
    span = (hi - lo) or 1.0
    grid = [[" "] * len(vals) for _ in range(height)]
    for col, v in enumerate(vals):
        h = int((v - lo) / span * (height - 1))
        grid[height - 1 - h][col] = "█"
    return ["".join(r) for r in grid]


def fit_color(win, y, x, txt, color, attr=0):
    h, w = win.getmaxyx()
    if y < h and x < w:
        try:
            win.addstr(y, min(x, w - 1), txt[: max(1, w - x - 1)], curses.color_pair(color) | attr)
        except curses.error:
            pass


def draw(stdscr, interval):
    curses.curs_set(0)
    stdscr.nodelay(True)
    curses.start_color(); curses.use_default_colors()
    curses.init_pair(1, curses.COLOR_GREEN, -1)
    curses.init_pair(2, curses.COLOR_RED, -1)
    curses.init_pair(3, curses.COLOR_CYAN, -1)
    curses.init_pair(4, curses.COLOR_YELLOW, -1)
    curses.init_pair(5, curses.COLOR_WHITE, -1)
    paused = False
    while True:
        if not paused:
            stdscr.erase()
            h, w = stdscr.getmaxyx()
            p = training_pid()
            lines = last_lines(LOG)
            prog = parse_progress(lines)
            rows = read_csv_history()
            xs, ys = val_series(rows)

            fit_color(stdscr, 0, 2, " GENERAL SPOT TRADING AGENT — LIVE DASHBOARD ", 3, curses.A_BOLD | curses.A_REVERSE)
            # --- status box ---
            if p:
                try:
                    cpu = p.cpu_percent(None); mem = p.memory_percent()
                except Exception:
                    cpu = mem = 0.0
                fit_color(stdscr, 1, 2, f" STATUS: RUNNING  (pid {p.pid})", 1, curses.A_BOLD)
                fit_color(stdscr, 1, 34, f" CPU {cpu:.0f}%  RAM {mem:.0f}%", 5)
            else:
                fit_color(stdscr, 1, 2, " STATUS: NOT RUNNING — start with: ./trainctl.sh start", 2, curses.A_BOLD)
            try:
                import subprocess
                g = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total",
                                    "--format=csv,noheader"], capture_output=True, text=True, timeout=5).stdout.strip()
                fit_color(stdscr, 2, 2, f" GPU: {g}", 4)
            except Exception:
                pass
            fit_color(stdscr, 2, 60, f" checkpoint update: {checkpoint_update()}", 5)

            # --- progress numbers ---
            y0 = 4
            if prog:
                fit_color(stdscr, y0, 2, f" update {prog.get('update','?')}/{prog.get('total','?')}", 5, curses.A_BOLD)
                try: pf = float(prog.get('portfolio') or 10000)
                except ValueError: pf = 10000.0
                fit_color(stdscr, y0, 24, f"portfolio {prog.get('portfolio','?')}", 1 if pf >= 10000 else 2)
                fit_color(stdscr, y0, 42, f"val {prog.get('val','-')}%", 4)
                sv = prog.get('stable', '-')
                try: svf = float(sv)
                except (ValueError, TypeError): svf = -999
                fit_color(stdscr, y0, 54, f"stable {sv}%", (1 if svf > 0 else 2), curses.A_BOLD)
                fit_color(stdscr, y0, 70, f"best {prog.get('best','?')}%", 5)
                fit_color(stdscr, y0, 84, f"ent {prog.get('ent','?')}", 5)
            else:
                fit_color(stdscr, y0, 2, "(no progress lines yet — training just started or log empty)", 5)

            # --- validation chart ---
            chart_h = max(3, (h - 12) // 2)
            chart_w = w - 6
            fit_color(stdscr, 6, 2, " VALIDATION RETURN % (per eval, last 400 evals) ", 3, curses.A_BOLD)
            for i, ln in enumerate(spark(ys, chart_w, chart_h)):
                fit_color(stdscr, 7 + i, 2, ln, 1)
            if ys:
                fit_color(stdscr, 7 + chart_h, 2, f" min {min(ys):.1f}%   max {max(ys):.1f}%   last {ys[-1]:.1f}%", 5)

            # --- live log tail ---
            ly = 8 + chart_h
            fit_color(stdscr, ly, 2, " LAST UPDATES ", 3, curses.A_BOLD)
            shown = lines[-max(1, h - ly - 3):]
            for i, ln in enumerate(shown):
                col = 5
                if " early stop" in ln or "budget" in ln or "stop" in ln: col = 2
                elif ln.startswith("update"): col = 5
                fit_color(stdscr, ly + 1 + i, 2, ln, col)

            fit_color(stdscr, h - 2, 2, " q: quit   p: pause   (dashboard only — training unaffected)", 4)
            fit_color(stdscr, h - 1, 2, f" refreshing every {interval}s — {time.strftime('%H:%M:%S')}", 5)
        stdscr.refresh()
        stop = time.time() + interval
        while time.time() < stop:
            ch = stdscr.getch()
            if ch in (ord('q'), 27):
                return
            if ch == ord('p'):
                paused = not paused
                if paused:
                    fit_color(stdscr, 0, 0, " PAUSED ", 2, curses.A_BOLD)
            time.sleep(0.1)


if __name__ == "__main__":
    interval = float(sys.argv[sys.argv.index("--interval") + 1]) if "--interval" in sys.argv else 2.0
    os.environ.setdefault("TERM", "xterm-256color")
    curses.wrapper(draw, interval)
