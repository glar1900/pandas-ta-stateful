# -*- coding: utf-8 -*-
"""pandas-ta.stateful – streaming / stateful indicator package.

Category modules populate STATEFUL_REGISTRY, SEED_REGISTRY, and
LOOKAHEAD_REGISTRY at import time.  This package re-exports them
plus the shared base API.
"""
from __future__ import annotations

# Base API (always available)
from ._base import (
    NAN,
    EMAState,
    ATRState,
    StatefulIndicator,
    STATEFUL_REGISTRY,
    SEED_REGISTRY,
    LOOKAHEAD_REGISTRY,
    ema_update_raw,
    ema_make,
    rma_make,
    atr_update_raw,
    replay_seed,
    build_state_key,
    resolve_output_names,
    stateful_supported_kinds,
    STATEFUL_SPEC_EXCLUDES,
    _is_nan,
    _param,
    _as_int,
    _as_float,
)

# ---------------------------------------------------------------------------
# Category modules – each populates the shared registries on import
# ---------------------------------------------------------------------------
from . import _candle       # noqa: F401  ha
from . import _performance  # noqa: F401  drawdown
from . import _cycle        # noqa: F401  ebsw, reflex
from . import _overlap      # noqa: F401  ema, rma, smma, kama, …
from . import _momentum     # noqa: F401  rsi, macd, …
from . import _volatility   # noqa: F401  atr, …
from . import _trend        # noqa: F401  adx, psar, …
from . import _volume       # noqa: F401  obv, ad, …
from . import _lookahead    # noqa: F401  dpo, ichimoku, …

__all__ = [
    # base
    "NAN",
    "EMAState",
    "ATRState",
    "StatefulIndicator",
    "STATEFUL_REGISTRY",
    "SEED_REGISTRY",
    "LOOKAHEAD_REGISTRY",
    "ema_update_raw",
    "ema_make",
    "rma_make",
    "atr_update_raw",
    "replay_seed",
    "build_state_key",
    "resolve_output_names",
    "stateful_supported_kinds",
]
