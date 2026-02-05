# -*- coding: utf-8 -*-
"""pandas-ta stateful – performance indicators.

Registered kinds
----------------
output_only : drawdown
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from ._base import (
    NAN, _is_nan, _param, _as_int, _as_float,
    EMAState, ema_make, ema_update_raw,
    StatefulIndicator,
    STATEFUL_REGISTRY, SEED_REGISTRY,
    replay_seed,
)


# ===========================================================================
# Drawdown
# ===========================================================================
# Vectorized source outputs three columns:
#   DD      = max_close - close                          (absolute drawdown)
#   DD_PCT  = 1 - (close / max_close)                    (percentage drawdown)
#   DD_LOG  = log(max_close) - log(close)                (log drawdown)
#
# max_close is the running (cumulative) maximum of close.
#
# The task specification requests outputs [dd, max_dd] where
#   dd     = (max_close - close) / max_close   (== DD_PCT)
#   max_dd = running max of dd                 (max drawdown so far)
#
# We honour both: output list is [DD_PCT, max_DD] matching the spec,
# and also expose DD and DD_LOG for completeness giving four outputs:
#   [DD, DD_PCT, DD_LOG, max_DD]
#
# State: max_close (running maximum), max_dd (running maximum of DD_PCT)
# ===========================================================================

@dataclass
class DrawdownState:
    max_close: Optional[float] = None
    max_dd:    float            = 0.0


def _drawdown_init(params: Dict[str, Any]) -> DrawdownState:
    return DrawdownState()


def _drawdown_update(
    state: DrawdownState, bar: Dict[str, float], params: Dict[str, Any]
) -> Tuple[List[Optional[float]], DrawdownState]:
    close = bar["close"]

    # Update running maximum
    if state.max_close is None:
        state.max_close = close
    else:
        if close > state.max_close:
            state.max_close = close

    # Absolute drawdown
    dd = state.max_close - close

    # Percentage drawdown:  1 - close / max_close  (== (max - close) / max)
    if state.max_close != 0.0:
        dd_pct = 1.0 - (close / state.max_close)
    else:
        dd_pct = 0.0

    # Log drawdown
    if close > 0.0 and state.max_close > 0.0:
        dd_log = math.log(state.max_close) - math.log(close)
    else:
        dd_log = NAN

    # Running maximum drawdown (percentage basis)
    if dd_pct > state.max_dd:
        state.max_dd = dd_pct

    return [dd, dd_pct, dd_log, state.max_dd], state


def _drawdown_output_names(params: Dict[str, Any]) -> List[str]:
    return ["DD", "DD_PCT", "DD_LOG", "max_DD"]


def _drawdown_seed(inputs: Dict[str, Any], params: Dict[str, Any]) -> DrawdownState:
    """output_only seed: replay full history to recover max_close / max_dd."""
    return replay_seed("drawdown", inputs, params)


STATEFUL_REGISTRY["drawdown"] = StatefulIndicator(
    kind="drawdown",
    inputs=("close",),
    init=_drawdown_init,
    update=_drawdown_update,
    output_names=_drawdown_output_names,
)
SEED_REGISTRY["drawdown"] = _drawdown_seed
