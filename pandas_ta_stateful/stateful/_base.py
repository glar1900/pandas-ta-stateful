# -*- coding: utf-8 -*-
"""pandas-ta stateful – shared base: state classes, helpers, registries.

All category modules (``_overlap``, ``_momentum``, …) import from here
and populate the registries at load time.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import math
import warnings

NAN = float("nan")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_nan(x: Any) -> bool:
    """True when *x* is None or a float NaN."""
    return x is None or (isinstance(x, float) and math.isnan(x))


def _param(params: Dict[str, Any], key: str, default: Any) -> Any:
    """Pull *key* from *params*; treat None as missing → default."""
    value = params.get(key, default)
    return default if value is None else value


def _as_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _as_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


# ---------------------------------------------------------------------------
# Shared state classes
# ---------------------------------------------------------------------------

@dataclass
class EMAState:
    """Reusable for EMA / RMA / Wilder / SMMA.

    EMA  -> alpha = 2 / (length + 1)   via ``ema_make``
    RMA  -> alpha = 1 / length          via ``rma_make``

    presma=True  ->  first output = SMA(x[0:length])   (TA-Lib default)
    presma=False ->  first output = x[0]
    """
    length: int
    alpha: float
    last: Optional[float] = None
    presma: bool = True
    _warmup_sum: float = 0.0
    _warmup_count: int = 0


@dataclass
class ATRState:
    """ATR with Wilder (RMA, alpha=1/n) + SMA seed.

    TA-Lib default: first ATR = mean(TR[0:length]), then Wilder smoothing.
    """
    length: int
    percent: bool = False
    prev_close: Optional[float] = None
    atr: Optional[float] = None
    _tr_sum: float = 0.0
    _tr_count: int = 0


# ---------------------------------------------------------------------------
# Low-level update helpers
# ---------------------------------------------------------------------------

def ema_make(length: int, presma: bool = True) -> EMAState:
    """EMA state – alpha = 2 / (length + 1)."""
    return EMAState(length=length, alpha=2.0 / (length + 1.0), presma=presma)


def rma_make(length: int, presma: bool = True) -> EMAState:
    """RMA / Wilder state – alpha = 1 / length."""
    return EMAState(length=length, alpha=1.0 / length, presma=presma)


def ema_update_raw(state: EMAState, x: float) -> Tuple[Optional[float], EMAState]:
    """Single-step EMA / RMA update.  Returns (value | None, state).

    Returns None while warming up (presma mode, fewer than *length*
    samples seen).
    """
    if state.last is None:
        if state.presma:
            state._warmup_sum += x
            state._warmup_count += 1
            if state._warmup_count < state.length:
                return None, state
            state.last = state._warmup_sum / state.length   # SMA seed
            return state.last, state
        else:
            state.last = x
            return state.last, state
    state.last = state.alpha * x + (1.0 - state.alpha) * state.last
    return state.last, state


def atr_update_raw(state: ATRState, high: float, low: float, close: float) -> Tuple[Optional[float], ATRState]:
    """Single-step ATR (Wilder).  Returns (atr | None, state)."""
    # True Range
    if state.prev_close is None:
        tr = high - low
    else:
        tr = max(high - low, abs(high - state.prev_close), abs(low - state.prev_close))
    state.prev_close = close

    if state.atr is None:
        state._tr_sum += tr
        state._tr_count += 1
        if state._tr_count < state.length:
            return None, state
        state.atr = state._tr_sum / state.length       # SMA seed
    else:
        state.atr = (state.atr * (state.length - 1) + tr) / state.length

    value: float = state.atr
    if state.percent and close != 0.0:
        value = value * 100.0 / close
    return value, state


# ---------------------------------------------------------------------------
# Indicator descriptor & registries  (populated by category modules)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class StatefulIndicator:
    """Immutable descriptor for a single stateful indicator."""
    kind:         str
    inputs:       Tuple[str, ...]
    init:         Callable[[Dict[str, Any]], Any]
    update:       Callable[[Any, Dict[str, Any], Dict[str, Any]],
                           Tuple[List[Optional[float]], Any]]
    output_names: Callable[[Dict[str, Any]], List[str]]


# Populated by category modules at import time.
STATEFUL_REGISTRY:  Dict[str, StatefulIndicator] = {}
SEED_REGISTRY:      Dict[str, Callable] = {}            # kind -> seed_fn(inputs, params) -> State
LOOKAHEAD_REGISTRY: Dict[str, StatefulIndicator] = {}   # conditional / approximate lookahead


# ---------------------------------------------------------------------------
# Generic seed helper
# ---------------------------------------------------------------------------

def replay_seed(kind: str, inputs: Dict[str, Any], params: Dict[str, Any]) -> Any:
    """Generic seed: replay the stateful update over historical Series.

    *inputs* values must be ``pd.Series`` (or any indexable with ``.iloc``).
    Returns the final *State* after processing all rows.

    For output_only / internal_series indicators this is the default
    seed implementation registered in ``SEED_REGISTRY``.
    """
    import pandas as pd          # lazy – pandas not required at module load
    # Check both STATEFUL_REGISTRY and LOOKAHEAD_REGISTRY
    indicator = STATEFUL_REGISTRY.get(kind) or LOOKAHEAD_REGISTRY.get(kind)
    if indicator is None:
        raise ValueError(f"Indicator '{kind}' not found in STATEFUL_REGISTRY or LOOKAHEAD_REGISTRY")
    state = indicator.init(params)
    keys = list(inputs.keys())
    if not keys:
        return state
    n = len(inputs[keys[0]])
    for i in range(n):
        bar: Dict[str, float] = {}
        valid = True
        for k in keys:
            v = inputs[k].iloc[i]
            if pd.isna(v):
                valid = False
                break
            bar[k] = float(v)
        if not valid:
            continue
        _, state = indicator.update(state, bar, params)
    return state


# ---------------------------------------------------------------------------
# Output-name helpers
# ---------------------------------------------------------------------------

STATEFUL_SPEC_EXCLUDES = frozenset({
    "kind", "append", "prefix", "suffix", "delimiter",
    "col_names", "state_key", "returns", "returns_state",
    "timed", "verbose", "ordered", "cores", "chunksize",
    "name", "description",
})


def build_state_key(kind: str, spec: Dict[str, Any]) -> str:
    """Deterministic cache-key from *kind* + non-meta params."""
    parts = sorted(
        ((k, v) for k, v in spec.items() if k not in STATEFUL_SPEC_EXCLUDES),
        key=lambda x: x[0],
    )
    payload = "|".join(f"{k}={repr(v)}" for k, v in parts)
    return f"{kind}|{payload}" if payload else kind


def resolve_output_names(
        base_names: List[str], spec: Dict[str, Any]
) -> Tuple[Optional[List[str]], Optional[str]]:
    """Apply prefix / suffix / col_names overrides from *spec*."""
    names = list(base_names)
    delimiter = spec.get("delimiter", "_")
    prefix = spec.get("prefix") or ""
    suffix = spec.get("suffix") or ""
    if prefix:
        prefix = f"{prefix}{delimiter}"
    if suffix:
        suffix = f"{delimiter}{suffix}"
    if prefix or suffix:
        names = [f"{prefix}{n}{suffix}" for n in names]
    col_names = spec.get("col_names")
    if col_names is not None:
        if not isinstance(col_names, tuple):
            col_names = (col_names,)
        if len(col_names) < len(names):
            return None, f"[!] col_names too short: {len(col_names)} < {len(names)}"
        names = list(col_names[: len(names)])
    return names, None


def stateful_supported_kinds(include_lookahead: bool = True) -> List[str]:
    """Return sorted list of supported indicator kinds.

    When include_lookahead=True, include conditional/approximate lookahead kinds.
    """
    kinds = set(STATEFUL_REGISTRY.keys())
    if include_lookahead:
        kinds.update(LOOKAHEAD_REGISTRY.keys())
    return sorted(kinds)
