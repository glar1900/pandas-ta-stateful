# -*- coding: utf-8 -*-
"""pandas-ta stateful -- cycle indicators.

Each section follows the pattern:
  1. State dataclass  (if beyond what _base already provides)
  2. init / update / output_names helpers
  3. STATEFUL_REGISTRY["<kind>"] = StatefulIndicator(...)
  4. SEED_REGISTRY["<kind>"]     = seed_fn          (output_only / internal_series)
     (replay_only indicators omit step 4)

Seed-method legend used throughout:
  output_only     -- seed_fn extracts the final output value(s) from the
                     vectorised result and reconstructs the minimal state
                     needed to keep updating.
  internal_series -- seed_fn additionally captures intermediate series
                     (e.g. EMA1 for DEMA) so that the recursive chain
                     stays numerically identical.
  replay_only     -- no seed_fn; the only way to initialise is to replay
                     every bar through update().
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from ._base import (
    NAN,
    _is_nan,
    _param,
    _as_int,
    _as_float,
    StatefulIndicator,
    STATEFUL_REGISTRY,
    SEED_REGISTRY,
    replay_seed,
)


# ===========================================================================
# EBSW  (internal_series) -- Even Better SineWave
# ===========================================================================
# State: lastHP, lastClose, filtHist[3] (filter history for SuperSmoother)
# Default: length=40, bars=10, initial_version=False

@dataclass
class EBSWState:
    length: int
    bars: int
    initial_version: bool
    lastHP: float = 0.0
    lastClose: float = 0.0
    filtHist: List[float] = None
    # Pre-computed constants
    alpha1: float = 0.0
    a1: float = 0.0
    c1: float = 0.0
    c2: float = 0.0
    c3: float = 0.0
    _warmup_count: int = 0

    def __post_init__(self):
        if self.filtHist is None:
            self.filtHist = [0.0, 0.0, 0.0]


def _ebsw_init(params: Dict[str, Any]) -> EBSWState:
    length = _as_int(_param(params, "length", 40), 40)
    bars = _as_int(_param(params, "bars", 10), 10)
    initial_version = bool(_param(params, "initial_version", False))

    state = EBSWState(length=length, bars=bars, initial_version=initial_version)

    # Pre-compute constants
    pi = math.pi
    if initial_version:
        state.alpha1 = (1 - math.sin(360 / length)) / math.cos(360 / length)
        state.a1 = math.exp(-math.sqrt(2) * pi / bars)
        state.c2 = 2 * state.a1 * math.cos(math.sqrt(2) * 180 / bars)
        state.c3 = -state.a1 * state.a1
        state.c1 = 1 - state.c2 - state.c3
    else:  # Default version
        angle = 2 * pi / length
        state.alpha1 = (1 - math.sin(angle)) / math.cos(angle)
        ang = 2 ** 0.5 * pi / bars
        state.a1 = math.exp(-ang)
        state.c2 = 2 * state.a1 * math.cos(ang)
        state.c3 = -state.a1 ** 2
        state.c1 = 1 - state.c2 - state.c3

    return state


def _ebsw_update(
    state: EBSWState, bar: Dict[str, Any], params: Dict[str, Any]
) -> Tuple[List[Optional[float]], EBSWState]:
    close = bar["close"]

    # Warmup period
    state._warmup_count += 1
    if state._warmup_count < state.length:
        return [None], state

    # First valid output at warmup_count == length
    if state._warmup_count == state.length:
        state.lastClose = close
        return [0.0], state

    # HighPass filter cyclic components
    hp = 0.5 * (1 + state.alpha1) * (close - state.lastClose) + state.alpha1 * state.lastHP

    # Smooth with a Super Smoother Filter
    if state.initial_version:
        # Initial version uses list append/pop
        filter_ = 0.5 * state.c1 * (hp + state.lastHP) + state.c2 * state.filtHist[1] + state.c3 * state.filtHist[0]
        # 3 Bar average of wave amplitude and power
        wave = (filter_ + state.filtHist[1] + state.filtHist[0]) / 3.0
        power = (filter_ * filter_ + state.filtHist[1] * state.filtHist[1] + state.filtHist[0] * state.filtHist[0]) / 3.0
        # Normalize the Average Wave to Square Root of the Average Power
        if power > 0:
            wave = wave / math.sqrt(power)
        else:
            wave = 0.0
        # Update filter history
        state.filtHist.append(filter_)
        state.filtHist.pop(0)
    else:  # Default version
        # Rotate filters to overwrite oldest value
        state.filtHist[0] = state.filtHist[1]
        state.filtHist[1] = state.filtHist[2]
        state.filtHist[2] = 0.5 * state.c1 * (hp + state.lastHP) + state.c2 * state.filtHist[1] + state.c3 * state.filtHist[0]

        # Wave calculation
        wave = (state.filtHist[0] + state.filtHist[1] + state.filtHist[2]) / 3.0
        rms = math.sqrt((state.filtHist[0] ** 2 + state.filtHist[1] ** 2 + state.filtHist[2] ** 2) / 3.0)
        if rms > 0:
            wave = wave / rms
        else:
            wave = 0.0

    # Update past values
    state.lastHP = hp
    state.lastClose = close

    return [wave], state


def _ebsw_output_names(params: Dict[str, Any]) -> List[str]:
    length = _as_int(_param(params, "length", 40), 40)
    bars = _as_int(_param(params, "bars", 10), 10)
    return [f"EBSW_{length}_{bars}"]


def _ebsw_seed(series: Dict[str, Any], params: Dict[str, Any]) -> EBSWState:
    """internal_series: use replay_seed to reconstruct full state."""
    return replay_seed("ebsw", series, params)


STATEFUL_REGISTRY["ebsw"] = StatefulIndicator(
    kind="ebsw",
    inputs=("close",),
    init=_ebsw_init,
    update=_ebsw_update,
    output_names=_ebsw_output_names,
)
SEED_REGISTRY["ebsw"] = _ebsw_seed


# ===========================================================================
# REFLEX  (internal_series) -- Reflex Cycle Indicator
# ===========================================================================
# State: _f (SuperSmoother series buffer), _ms (last EMA value)
# Default: length=20, smooth=20, alpha=0.04, pi=3.14159, sqrt2=1.414

@dataclass
class REFLEXState:
    length: int
    smooth: int
    alpha: float
    pi: float
    sqrt2: float
    # SuperSmoother filter buffer (need length values for slope calculation)
    _f_buffer: List[float] = None
    # Mean square EMA state
    _ms: float = 0.0
    # Previous close for SuperSmoother
    _close_prev: float = 0.0
    # Pre-computed constants
    _a: float = 0.0
    _b: float = 0.0
    _c: float = 0.0
    _warmup_count: int = 0

    def __post_init__(self):
        if self._f_buffer is None:
            # Need to store last length+1 values for slope calculation
            self._f_buffer = []


def _reflex_init(params: Dict[str, Any]) -> REFLEXState:
    length = _as_int(_param(params, "length", 20), 20)
    smooth = _as_int(_param(params, "smooth", 20), 20)
    alpha = _as_float(_param(params, "alpha", 0.04), 0.04)
    pi = _as_float(_param(params, "pi", 3.14159), 3.14159)
    sqrt2 = _as_float(_param(params, "sqrt2", 1.414), 1.414)

    state = REFLEXState(
        length=length,
        smooth=smooth,
        alpha=alpha,
        pi=pi,
        sqrt2=sqrt2
    )

    # Pre-compute SuperSmoother constants
    ratio = 2 * sqrt2 / smooth
    state._a = math.exp(-pi * ratio)
    state._b = 2 * state._a * math.cos(180 * ratio)
    state._c = state._a * state._a - state._b + 1

    return state


def _reflex_update(
    state: REFLEXState, bar: Dict[str, Any], params: Dict[str, Any]
) -> Tuple[List[Optional[float]], REFLEXState]:
    close = bar["close"]

    state._warmup_count += 1

    # SuperSmoother filter: _f[i] = 0.5 * c * (x[i] + x[i-1]) + b * _f[i-1] - a*a * _f[i-2]
    if state._warmup_count == 1:
        # First bar
        _f_current = 0.0
        state._f_buffer.append(_f_current)
        state._close_prev = close
        return [None], state

    if state._warmup_count == 2:
        # Second bar
        _f_current = 0.0
        state._f_buffer.append(_f_current)
        state._close_prev = close
        return [None], state

    # Calculate SuperSmoother filter from bar 3 onwards
    _f_prev1 = state._f_buffer[-1]
    _f_prev2 = state._f_buffer[-2] if len(state._f_buffer) >= 2 else 0.0
    _f_current = 0.5 * state._c * (close + state._close_prev) + state._b * _f_prev1 - state._a * state._a * _f_prev2

    state._f_buffer.append(_f_current)
    state._close_prev = close

    # Keep only the last length+1 values to save memory
    if len(state._f_buffer) > state.length + 1:
        state._f_buffer.pop(0)

    # Need `length` bars before computing reflex
    if state._warmup_count <= state.length:
        return [None], state

    # Calculate slope and sum
    # slope = (_f[i - n] - _f[i]) / n
    n = state.length
    i = len(state._f_buffer) - 1  # current index in buffer

    # _f[i-n] is at buffer index (i-n) if we have full history, otherwise buffer[0]
    if len(state._f_buffer) > n:
        f_old = state._f_buffer[i - n]
    else:
        f_old = state._f_buffer[0]

    slope = (f_old - _f_current) / n

    # sum over j=1 to n-1: _f[i] - _f[i-j] + j*slope
    _sum = 0.0
    for j in range(1, n):
        if i - j >= 0 and i - j < len(state._f_buffer):
            _sum += _f_current - state._f_buffer[i - j] + j * slope
    _sum /= n

    # Update mean square with EMA
    state._ms = state.alpha * _sum * _sum + (1 - state.alpha) * state._ms

    # Compute result
    if state._ms > 0:
        result = _sum / math.sqrt(state._ms)
    else:
        result = 0.0

    return [result], state


def _reflex_output_names(params: Dict[str, Any]) -> List[str]:
    length = _as_int(_param(params, "length", 20), 20)
    smooth = _as_int(_param(params, "smooth", 20), 20)
    alpha = _as_float(_param(params, "alpha", 0.04), 0.04)
    return [f"REFLEX_{length}_{smooth}_{alpha}"]


def _reflex_seed(series: Dict[str, Any], params: Dict[str, Any]) -> REFLEXState:
    """internal_series: use replay_seed to reconstruct full state.

    The reflex indicator requires the full history of the SuperSmoother filter (_f)
    to compute the slope and sum accurately. Therefore, we use replay_seed.
    """
    return replay_seed("reflex", series, params)


STATEFUL_REGISTRY["reflex"] = StatefulIndicator(
    kind="reflex",
    inputs=("close",),
    init=_reflex_init,
    update=_reflex_update,
    output_names=_reflex_output_names,
)
SEED_REGISTRY["reflex"] = _reflex_seed
