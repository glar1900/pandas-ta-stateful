# -*- coding: utf-8 -*-
"""pandas-ta stateful – candle indicators.

Registered kinds
----------------
output_only : ha (Heikin-Ashi)
"""
from __future__ import annotations

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
# HA – Heikin-Ashi
# ===========================================================================
# Vectorized source (np_ha):
#   ha_close[i] = 0.25 * (open[i] + high[i] + low[i] + close[i])
#   ha_open[0]  = 0.5  * (open[0] + close[0])
#   ha_open[i]  = 0.5  * (ha_open[i-1] + ha_close[i-1])     for i > 0
#   ha_high[i]  = max(high[i], ha_open[i], ha_close[i])
#   ha_low[i]   = min(low[i],  ha_open[i], ha_close[i])
#
# State: prev_ha_open, prev_ha_close  (both None before first bar)
# First bar initialisation:
#   ha_open  = 0.5 * (open + close)      <- same as vectorized ha_open[0]
#   ha_close = 0.25 * (O + H + L + C)
# ===========================================================================

@dataclass
class HAState:
    prev_ha_open:  Optional[float] = None
    prev_ha_close: Optional[float] = None


def _ha_init(params: Dict[str, Any]) -> HAState:
    return HAState()


def _ha_update(
    state: HAState, bar: Dict[str, float], params: Dict[str, Any]
) -> Tuple[List[Optional[float]], HAState]:
    o = bar["open"]
    h = bar["high"]
    l = bar["low"]
    c = bar["close"]

    # ha_close is always the simple average of the raw candle
    ha_close = 0.25 * (o + h + l + c)

    if state.prev_ha_open is None:
        # First bar: ha_open = 0.5 * (open + close)  (vectorized convention)
        ha_open = 0.5 * (o + c)
    else:
        ha_open = 0.5 * (state.prev_ha_open + state.prev_ha_close)

    ha_high = max(h, ha_open, ha_close)
    ha_low  = min(l, ha_open, ha_close)

    # Persist for next bar
    state.prev_ha_open  = ha_open
    state.prev_ha_close = ha_close

    return [ha_open, ha_high, ha_low, ha_close], state


def _ha_output_names(params: Dict[str, Any]) -> List[str]:
    return ["HA_open", "HA_high", "HA_low", "HA_close"]


def _ha_seed(inputs: Dict[str, Any], params: Dict[str, Any]) -> HAState:
    """output_only seed: replay full history to recover final HA state."""
    return replay_seed("ha", inputs, params)


STATEFUL_REGISTRY["ha"] = StatefulIndicator(
    kind="ha",
    inputs=("open", "high", "low", "close"),
    init=_ha_init,
    update=_ha_update,
    output_names=_ha_output_names,
)
SEED_REGISTRY["ha"] = _ha_seed
