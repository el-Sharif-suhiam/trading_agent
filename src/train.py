"""Thin CLI wrapper around the shared Trainer engine (src/trainer.py)."""
import argparse, json
from .trainer import Trainer


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config/default.json")
    p.add_argument("--resume", default="")
    p.add_argument("--time-budget-hours", type=float, default=None,
                   help="Stop training cleanly after this many hours (e.g. 6 or 12).")
    a = p.parse_args()
    cfg = json.load(open(a.config))
    Trainer(cfg, resume_path=a.resume).train(time_budget_hours=a.time_budget_hours)


if __name__ == "__main__": main()
