"""Checkpoint compatibility utilities.

torch.compile renames every state_dict key with a `_orig_mod.` prefix, so a
checkpoint may come from a compiled OR an eager session — and the reader
(evaluate / backtest / harness) may be running the other way. Every module
that loads a saved model MUST go through here; the duplicated strip logic
that existed before broke the documented evaluation gates (evaluate.py
crashed with missing/unexpected keys on any post-compile checkpoint).
"""

_PREFIX = "_orig_mod."


def strip_orig_mod(state):
    """Remove the torch.compile `_orig_mod.` prefix from every key (idempotent)."""
    return {k[len(_PREFIX):] if k.startswith(_PREFIX) else k: v for k, v in state.items()}


def add_orig_mod(state):
    """Add the torch.compile `_orig_mod.` prefix to every key."""
    return {f"{_PREFIX}{k}": v for k, v in state.items()}


def load_state_compat(model, state):
    """Load a state_dict into `model` regardless of which side was compiled."""
    state = strip_orig_mod(state)
    if hasattr(model, "_orig_mod"):  # running under torch.compile
        state = add_orig_mod(state)
    model.load_state_dict(state)
    return model


def load_checkpoint(model, path, map_location="cpu", weights_only=False):
    """torch.load a checkpoint file AND load its model weights into `model`.
    Returns the raw checkpoint dict (update/best_val/config/... metadata
    still accessible to the caller)."""
    import torch
    ck = torch.load(path, map_location=map_location, weights_only=weights_only)
    load_state_compat(model, ck["model"])
    return ck
