# -*- coding: utf-8 -*-
"""pandas-ta stateful -- overlap indicators.

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
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from collections import deque
from math import exp, cos, sqrt, log, atan, pow as _pow

from ._base import (
    NAN,
    _is_nan,
    _param,
    _as_int,
    _as_float,
    EMAState,
    ATRState,
    ema_make,
    rma_make,
    ema_update_raw,
    atr_update_raw,
    StatefulIndicator,
    STATEFUL_REGISTRY,
    SEED_REGISTRY,
    replay_seed,
)


# ===========================================================================
# EMA  (output_only)
# ===========================================================================
# State: reuses EMAState directly.  alpha = 2/(length+1), presma=True.
# Default length = 10.

def _ema_init(params: Dict[str, Any]) -> EMAState:
    length = _as_int(_param(params, "length", 10), 10)
    return ema_make(length, presma=True)


def _ema_update(
    state: EMAState, bar: Dict[str, Any], params: Dict[str, Any]
) -> Tuple[List[Optional[float]], EMAState]:
    x = bar["close"]
    val, state = ema_update_raw(state, x)
    return [val], state


def _ema_output_names(params: Dict[str, Any]) -> List[str]:
    length = _as_int(_param(params, "length", 10), 10)
    return [f"EMA_{length}"]


def _ema_seed(series: Dict[str, Any], params: Dict[str, Any]) -> EMAState:
    """Reconstruct EMAState from the last valid output value."""
    length = _as_int(_param(params, "length", 10), 10)
    state = ema_make(length, presma=True)
    col = _ema_output_names(params)[0]
    s = series.get(col)
    if s is not None:
        last_valid = s.dropna()
        if len(last_valid) > 0:
            state.last = float(last_valid.iloc[-1])
            # warmup already done -- mark as seeded
            state._warmup_count = length
            state._warmup_sum = 0.0
    return state


STATEFUL_REGISTRY["ema"] = StatefulIndicator(
    kind="ema",
    inputs=("close",),
    init=_ema_init,
    update=_ema_update,
    output_names=_ema_output_names,
)
SEED_REGISTRY["ema"] = _ema_seed


# ===========================================================================
# RMA  (output_only)  -- Wilder's MA, alpha = 1/length, presma=True
# ===========================================================================
# Default length = 10.  Vectorised source uses ewm(alpha=1/n, adjust=False)
# which is equivalent to presma=False seed (first value = x[0]).
# However the spec table says presma=True (SMA seed).  We honour the spec.

def _rma_init(params: Dict[str, Any]) -> EMAState:
    length = _as_int(_param(params, "length", 10), 10)
    return rma_make(length, presma=True)


def _rma_update(
    state: EMAState, bar: Dict[str, Any], params: Dict[str, Any]
) -> Tuple[List[Optional[float]], EMAState]:
    x = bar["close"]
    val, state = ema_update_raw(state, x)
    return [val], state


def _rma_output_names(params: Dict[str, Any]) -> List[str]:
    length = _as_int(_param(params, "length", 10), 10)
    return [f"RMA_{length}"]


def _rma_seed(series: Dict[str, Any], params: Dict[str, Any]) -> EMAState:
    length = _as_int(_param(params, "length", 10), 10)
    state = rma_make(length, presma=True)
    col = _rma_output_names(params)[0]
    s = series.get(col)
    if s is not None:
        last_valid = s.dropna()
        if len(last_valid) > 0:
            state.last = float(last_valid.iloc[-1])
            state._warmup_count = length
            state._warmup_sum = 0.0
    return state


STATEFUL_REGISTRY["rma"] = StatefulIndicator(
    kind="rma",
    inputs=("close",),
    init=_rma_init,
    update=_rma_update,
    output_names=_rma_output_names,
)
SEED_REGISTRY["rma"] = _rma_seed


# ===========================================================================
# SMMA  (output_only)  -- SMoothed MA
# ===========================================================================
# Formula: smma[i] = ((length-1)*smma[i-1] + x[i]) / length
# Seed: SMA of first `length` bars.  Default length = 7.

@dataclass
class SMMState:
    length: int
    prev: Optional[float] = None
    _warmup_sum: float = 0.0
    _warmup_count: int = 0


def _smma_init(params: Dict[str, Any]) -> SMMState:
    length = _as_int(_param(params, "length", 7), 7)
    return SMMState(length=length)


def _smma_update(
    state: SMMState, bar: Dict[str, Any], params: Dict[str, Any]
) -> Tuple[List[Optional[float]], SMMState]:
    x = bar["close"]
    n = state.length
    if state.prev is None:
        state._warmup_sum += x
        state._warmup_count += 1
        if state._warmup_count < n:
            return [None], state
        # SMA seed
        state.prev = state._warmup_sum / n
        return [state.prev], state
    state.prev = ((n - 1) * state.prev + x) / n
    return [state.prev], state


def _smma_output_names(params: Dict[str, Any]) -> List[str]:
    length = _as_int(_param(params, "length", 7), 7)
    return [f"SMMA_{length}"]


def _smma_seed(series: Dict[str, Any], params: Dict[str, Any]) -> SMMState:
    length = _as_int(_param(params, "length", 7), 7)
    state = SMMState(length=length)
    col = _smma_output_names(params)[0]
    s = series.get(col)
    if s is not None:
        last_valid = s.dropna()
        if len(last_valid) > 0:
            state.prev = float(last_valid.iloc[-1])
            state._warmup_count = length  # mark warmup done
    return state


STATEFUL_REGISTRY["smma"] = StatefulIndicator(
    kind="smma",
    inputs=("close",),
    init=_smma_init,
    update=_smma_update,
    output_names=_smma_output_names,
)
SEED_REGISTRY["smma"] = _smma_seed


# ===========================================================================
# KAMA  (output_only)  -- Kaufman Adaptive MA
# ===========================================================================
# Defaults: length=10, fast=2, slow=30
# Seed value = SMA(close, length).  After warmup:
#   er   = abs(close - close[length ago]) / sum(abs(close[i]-close[i-1])) over length
#   sc   = (er*(fr-sr) + sr)^2      fr=2/(fast+1), sr=2/(slow+1)
#   kama = sc * close + (1-sc) * prev_kama
# State needs: prev_kama, a deque of last `length` closes (for er numerator)
#   and a deque of last `length` abs-diffs (for er denominator).

@dataclass
class KAMAState:
    length: int
    fast: int
    slow: int
    fr: float       # 2/(fast+1)
    sr: float       # 2/(slow+1)
    prev_kama: Optional[float] = None
    # ring buffer of raw closes (size = length+1 to cover the shift)
    closes: deque = field(default_factory=deque)
    # ring buffer of abs(close[i]-close[i-1]) last `length` values
    abs_diffs: deque = field(default_factory=deque)
    _warmup_count: int = 0


def _kama_init(params: Dict[str, Any]) -> KAMAState:
    length = _as_int(_param(params, "length", 10), 10)
    fast  = _as_int(_param(params, "fast",  2), 2)
    slow  = _as_int(_param(params, "slow", 30), 30)
    fr = 2.0 / (fast + 1.0)
    sr = 2.0 / (slow + 1.0)
    return KAMAState(
        length=length, fast=fast, slow=slow, fr=fr, sr=sr,
        closes=deque(maxlen=length + 1),
        abs_diffs=deque(maxlen=length),
    )


def _kama_update(
    state: KAMAState, bar: Dict[str, Any], params: Dict[str, Any]
) -> Tuple[List[Optional[float]], KAMAState]:
    x = bar["close"]
    n = state.length

    # accumulate abs_diffs once we have at least 2 closes
    if len(state.closes) > 0:
        state.abs_diffs.append(abs(x - state.closes[-1]))

    state.closes.append(x)
    state._warmup_count += 1

    # need `length` bars before first output (index length-1 in 0-based)
    if state._warmup_count < n:
        return [None], state

    if state.prev_kama is None:
        # SMA seed on the first `length` closes
        state.prev_kama = sum(state.closes) / n
        return [state.prev_kama], state

    # ER calculation
    # numerator: abs(close_now - close[length bars ago])
    # closes deque has maxlen=length+1; after warmup it is full
    change = abs(x - state.closes[0])  # closes[0] is the oldest (length bars ago)
    vol = sum(state.abs_diffs)  # sum of last `length` abs diffs
    er = change / vol if vol != 0.0 else 0.0

    sc_base = er * (state.fr - state.sr) + state.sr
    sc = sc_base * sc_base

    state.prev_kama = sc * x + (1.0 - sc) * state.prev_kama
    return [state.prev_kama], state


def _kama_output_names(params: Dict[str, Any]) -> List[str]:
    length = _as_int(_param(params, "length", 10), 10)
    fast  = _as_int(_param(params, "fast",  2), 2)
    slow  = _as_int(_param(params, "slow", 30), 30)
    return [f"KAMA_{length}_{fast}_{slow}"]


def _kama_seed(series: Dict[str, Any], params: Dict[str, Any]) -> KAMAState:
    """output_only seed: extract last KAMA value.
    closes / abs_diffs buffers cannot be recovered from output alone;
    we fill them from the raw close series passed in series["close"]
    if available, otherwise leave empty (caller may use replay_seed).
    """
    length = _as_int(_param(params, "length", 10), 10)
    fast  = _as_int(_param(params, "fast",  2), 2)
    slow  = _as_int(_param(params, "slow", 30), 30)
    state = _kama_init(params)
    col = _kama_output_names(params)[0]
    s = series.get(col)
    if s is not None:
        last_valid = s.dropna()
        if len(last_valid) > 0:
            state.prev_kama = float(last_valid.iloc[-1])
            state._warmup_count = length  # mark warmup done
    # Try to recover closes buffer from "close" series
    close_s = series.get("close")
    if close_s is not None and len(close_s) >= length + 1:
        tail = close_s.iloc[-(length + 1):]
        state.closes = deque(
            (float(v) for v in tail.values),
            maxlen=length + 1,
        )
        for i in range(1, len(tail)):
            state.abs_diffs.append(abs(float(tail.iloc[i]) - float(tail.iloc[i - 1])))
    return state


STATEFUL_REGISTRY["kama"] = StatefulIndicator(
    kind="kama",
    inputs=("close",),
    init=_kama_init,
    update=_kama_update,
    output_names=_kama_output_names,
)
SEED_REGISTRY["kama"] = _kama_seed


# ===========================================================================
# MCGD  (output_only)  -- McGinley Dynamic
# ===========================================================================
# Vectorised: rolling(2).apply(_mcgd) where
#   _mcgd(x[prev, cur], n, k):
#       d = k * n * (cur/prev)^4
#       result = prev + (cur - prev) / d
# Stateful equivalent: track prev_mcgd.  First bar = close (no output yet).
# Second bar onwards the formula fires.  Default length=10, c=1.

@dataclass
class MCGDState:
    length: int
    c: float
    prev: Optional[float] = None


def _mcgd_init(params: Dict[str, Any]) -> MCGDState:
    length = _as_int(_param(params, "length", 10), 10)
    c_raw = _param(params, "c", 1.0)
    c = float(c_raw) if isinstance(c_raw, (int, float)) and 0 < float(c_raw) <= 1 else 1.0
    return MCGDState(length=length, c=c)


def _mcgd_update(
    state: MCGDState, bar: Dict[str, Any], params: Dict[str, Any]
) -> Tuple[List[Optional[float]], MCGDState]:
    x = bar["close"]
    if state.prev is None:
        # first bar -- no rolling-2 output yet
        state.prev = x
        return [None], state
    prev = state.prev
    # guard against division by zero
    if prev == 0.0:
        state.prev = x
        return [x], state
    ratio = x / prev
    d = state.c * state.length * (ratio ** 4)
    if d == 0.0:
        val = x
    else:
        val = prev + (x - prev) / d
    state.prev = val
    return [val], state


def _mcgd_output_names(params: Dict[str, Any]) -> List[str]:
    length = _as_int(_param(params, "length", 10), 10)
    return [f"MCGD_{length}"]


def _mcgd_seed(series: Dict[str, Any], params: Dict[str, Any]) -> MCGDState:
    state = _mcgd_init(params)
    col = _mcgd_output_names(params)[0]
    s = series.get(col)
    if s is not None:
        last_valid = s.dropna()
        if len(last_valid) > 0:
            state.prev = float(last_valid.iloc[-1])
    return state


STATEFUL_REGISTRY["mcgd"] = StatefulIndicator(
    kind="mcgd",
    inputs=("close",),
    init=_mcgd_init,
    update=_mcgd_update,
    output_names=_mcgd_output_names,
)
SEED_REGISTRY["mcgd"] = _mcgd_seed


# ===========================================================================
# ALLIGATOR  (output_only)  -- Bill Williams
# ===========================================================================
# jaw  = SMMA(hlc3, 13)  -- but vectorised source passes `close` not hlc3.
# The vectorised alligator.py calls smma(close, jaw/teeth/lips).
# The input column the vectorised code uses is `close`.
# Defaults: jaw=13, teeth=8, lips=5.
# Each line is an independent SMMA; we reuse SMMState.

@dataclass
class AlligatorState:
    jaw_state:   SMMState
    teeth_state: SMMState
    lips_state:  SMMState


def _alligator_init(params: Dict[str, Any]) -> AlligatorState:
    jaw   = _as_int(_param(params, "jaw",   13), 13)
    teeth = _as_int(_param(params, "teeth",  8),  8)
    lips  = _as_int(_param(params, "lips",   5),  5)
    return AlligatorState(
        jaw_state=SMMState(length=jaw),
        teeth_state=SMMState(length=teeth),
        lips_state=SMMState(length=lips),
    )


def _alligator_update(
    state: AlligatorState, bar: Dict[str, Any], params: Dict[str, Any]
) -> Tuple[List[Optional[float]], AlligatorState]:
    x = bar["close"]

    # --- jaw ---
    js = state.jaw_state
    if js.prev is None:
        js._warmup_sum += x
        js._warmup_count += 1
        if js._warmup_count >= js.length:
            js.prev = js._warmup_sum / js.length
            jaw_val: Optional[float] = js.prev
        else:
            jaw_val = None
    else:
        js.prev = ((js.length - 1) * js.prev + x) / js.length
        jaw_val = js.prev

    # --- teeth ---
    ts = state.teeth_state
    if ts.prev is None:
        ts._warmup_sum += x
        ts._warmup_count += 1
        if ts._warmup_count >= ts.length:
            ts.prev = ts._warmup_sum / ts.length
            teeth_val: Optional[float] = ts.prev
        else:
            teeth_val = None
    else:
        ts.prev = ((ts.length - 1) * ts.prev + x) / ts.length
        teeth_val = ts.prev

    # --- lips ---
    ls = state.lips_state
    if ls.prev is None:
        ls._warmup_sum += x
        ls._warmup_count += 1
        if ls._warmup_count >= ls.length:
            ls.prev = ls._warmup_sum / ls.length
            lips_val: Optional[float] = ls.prev
        else:
            lips_val = None
    else:
        ls.prev = ((ls.length - 1) * ls.prev + x) / ls.length
        lips_val = ls.prev

    return [jaw_val, teeth_val, lips_val], state


def _alligator_output_names(params: Dict[str, Any]) -> List[str]:
    jaw   = _as_int(_param(params, "jaw",   13), 13)
    teeth = _as_int(_param(params, "teeth",  8),  8)
    lips  = _as_int(_param(params, "lips",   5),  5)
    _props = f"_{jaw}_{teeth}_{lips}"
    return [f"AGj{_props}", f"AGt{_props}", f"AGl{_props}"]


def _alligator_seed(series: Dict[str, Any], params: Dict[str, Any]) -> AlligatorState:
    jaw   = _as_int(_param(params, "jaw",   13), 13)
    teeth = _as_int(_param(params, "teeth",  8),  8)
    lips  = _as_int(_param(params, "lips",   5),  5)
    state = _alligator_init(params)
    names = _alligator_output_names(params)
    # jaw
    s = series.get(names[0])
    if s is not None:
        lv = s.dropna()
        if len(lv) > 0:
            state.jaw_state.prev = float(lv.iloc[-1])
            state.jaw_state._warmup_count = jaw
    # teeth
    s = series.get(names[1])
    if s is not None:
        lv = s.dropna()
        if len(lv) > 0:
            state.teeth_state.prev = float(lv.iloc[-1])
            state.teeth_state._warmup_count = teeth
    # lips
    s = series.get(names[2])
    if s is not None:
        lv = s.dropna()
        if len(lv) > 0:
            state.lips_state.prev = float(lv.iloc[-1])
            state.lips_state._warmup_count = lips
    return state


STATEFUL_REGISTRY["alligator"] = StatefulIndicator(
    kind="alligator",
    inputs=("close",),
    init=_alligator_init,
    update=_alligator_update,
    output_names=_alligator_output_names,
)
SEED_REGISTRY["alligator"] = _alligator_seed


# ===========================================================================
# ZLMA  (output_only)  -- Zero-Lag MA
# ===========================================================================
# lag = (length-1) // 2
# adjusted = 2 * close - close[lag bars ago]
# Then apply inner MA (default "ema") to adjusted.
# State: inner MA state + deque of last `lag` raw closes.
# We only implement the stateful variant for mamode="ema" (default).
# For other mamodes the user should use replay_seed.

@dataclass
class ZLMAState:
    length: int
    lag: int
    mamode: str
    # history of raw closes, size = lag  (oldest first)
    prev_closes: deque  # maxlen = lag
    # inner MA state (EMAState for default "ema")
    inner: Any = None


def _zlma_init(params: Dict[str, Any]) -> ZLMAState:
    length  = _as_int(_param(params, "length", 10), 10)
    mamode  = str(_param(params, "mamode", "ema"))
    lag     = (length - 1) // 2
    inner   = ema_make(length, presma=True)  # default ema inner
    return ZLMAState(
        length=length, lag=lag, mamode=mamode,
        prev_closes=deque(maxlen=max(lag, 1)),
        inner=inner,
    )


def _zlma_update(
    state: ZLMAState, bar: Dict[str, Any], params: Dict[str, Any]
) -> Tuple[List[Optional[float]], ZLMAState]:
    close = bar["close"]

    # compute adjusted close
    if len(state.prev_closes) < state.lag:
        # not enough history yet to shift; adjusted = 2*close - close (=close)
        # but we still feed it to the inner MA for warmup
        adjusted = close
    else:
        # prev_closes[0] is the oldest = close[lag bars ago]
        adjusted = 2.0 * close - state.prev_closes[0]

    # append current close to history (after reading the lagged value)
    state.prev_closes.append(close)

    # feed adjusted into inner EMA
    val, state.inner = ema_update_raw(state.inner, adjusted)
    return [val], state


def _zlma_output_names(params: Dict[str, Any]) -> List[str]:
    length = _as_int(_param(params, "length", 10), 10)
    mamode = str(_param(params, "mamode", "ema"))
    # vectorised names: ZL_EMA_<length>  etc.
    return [f"ZL_{mamode.upper()}_{length}"]


def _zlma_seed(series: Dict[str, Any], params: Dict[str, Any]) -> ZLMAState:
    """Seed from vectorised output + raw close tail."""
    length = _as_int(_param(params, "length", 10), 10)
    lag    = (length - 1) // 2
    state  = _zlma_init(params)
    col    = _zlma_output_names(params)[0]
    s      = series.get(col)
    if s is not None:
        lv = s.dropna()
        if len(lv) > 0:
            state.inner.last = float(lv.iloc[-1])
            state.inner._warmup_count = length
            state.inner._warmup_sum  = 0.0
    # recover close history for lag calculation
    close_s = series.get("close")
    if close_s is not None and len(close_s) >= lag:
        tail = close_s.iloc[-lag:]
        state.prev_closes = deque(
            (float(v) for v in tail.values),
            maxlen=max(lag, 1),
        )
    return state


STATEFUL_REGISTRY["zlma"] = StatefulIndicator(
    kind="zlma",
    inputs=("close",),
    init=_zlma_init,
    update=_zlma_update,
    output_names=_zlma_output_names,
)
SEED_REGISTRY["zlma"] = _zlma_seed


# ===========================================================================
# HILO  (internal_series)  -- Gann HiLo Activator
# ===========================================================================
# Outputs: HILO, HILOl (long), HILOs (short)
# Defaults: high_length=13, low_length=21, mamode="sma"
# high_ma = MA(high, high_length), low_ma = MA(low, low_length)
# Logic (vectorised uses i-1 comparison):
#   if close > high_ma_prev  ->  hilo = low_ma   (long)
#   elif close < low_ma_prev ->  hilo = high_ma  (short)
#   else                     ->  hilo = hilo_prev
# For the stateful version we implement the MAs as EMA (internal EMAState)
# to keep it streaming-friendly.  The vectorised default is "sma" which
# is a window MA; we store a deque for SMA if mamode=="sma".

@dataclass
class _SMAWindow:
    """Simple rolling-window SMA for stateful use."""
    length: int
    buf: deque = field(default_factory=deque)

    def update(self, x: float) -> Optional[float]:
        self.buf.append(x)
        if len(self.buf) < self.length:
            return None
        return sum(self.buf) / self.length


@dataclass
class HILOState:
    high_length: int
    low_length: int
    high_ma_state: _SMAWindow
    low_ma_state: _SMAWindow
    prev_high_ma: Optional[float] = None
    prev_low_ma:  Optional[float] = None
    prev_hilo:    Optional[float] = None


def _hilo_init(params: Dict[str, Any]) -> HILOState:
    hl = _as_int(_param(params, "high_length", 13), 13)
    ll = _as_int(_param(params, "low_length",  21), 21)
    return HILOState(
        high_length=hl, low_length=ll,
        high_ma_state=_SMAWindow(length=hl, buf=deque(maxlen=hl)),
        low_ma_state =_SMAWindow(length=ll, buf=deque(maxlen=ll)),
    )


def _hilo_update(
    state: HILOState, bar: Dict[str, Any], params: Dict[str, Any]
) -> Tuple[List[Optional[float]], HILOState]:
    high  = bar["high"]
    low   = bar["low"]
    close = bar["close"]

    cur_high_ma = state.high_ma_state.update(high)
    cur_low_ma  = state.low_ma_state.update(low)

    hilo_val: Optional[float] = None
    long_val: Optional[float] = None
    short_val: Optional[float] = None

    if cur_high_ma is not None and cur_low_ma is not None:
        if state.prev_high_ma is not None and state.prev_low_ma is not None:
            # decision logic uses previous bar's MA values
            if close > state.prev_high_ma:
                hilo_val = cur_low_ma
                long_val = cur_low_ma
            elif close < state.prev_low_ma:
                hilo_val = cur_high_ma
                short_val = cur_high_ma
            else:
                hilo_val = state.prev_hilo if state.prev_hilo is not None else cur_high_ma
                long_val  = hilo_val
                short_val = hilo_val
        else:
            # first bar with both MAs valid -- initialise to high_ma
            hilo_val = cur_high_ma

        state.prev_hilo = hilo_val

    # always advance the "previous" MA snapshots
    if cur_high_ma is not None:
        state.prev_high_ma = cur_high_ma
    if cur_low_ma is not None:
        state.prev_low_ma = cur_low_ma

    return [hilo_val, long_val, short_val], state


def _hilo_output_names(params: Dict[str, Any]) -> List[str]:
    hl = _as_int(_param(params, "high_length", 13), 13)
    ll = _as_int(_param(params, "low_length",  21), 21)
    _props = f"_{hl}_{ll}"
    return [f"HILO{_props}", f"HILOl{_props}", f"HILOs{_props}"]


def _hilo_seed(series: Dict[str, Any], params: Dict[str, Any]) -> HILOState:
    """internal_series seed: recover MA states and last hilo from series."""
    hl = _as_int(_param(params, "high_length", 13), 13)
    ll = _as_int(_param(params, "low_length",  21), 21)
    state = _hilo_init(params)
    names = _hilo_output_names(params)
    # recover prev_hilo
    s = series.get(names[0])
    if s is not None:
        lv = s.dropna()
        if len(lv) > 0:
            state.prev_hilo = float(lv.iloc[-1])
    # recover SMA windows from raw high/low if available
    high_s = series.get("high")
    low_s  = series.get("low")
    if high_s is not None and len(high_s) >= hl:
        tail = high_s.iloc[-hl:]
        state.high_ma_state.buf = deque((float(v) for v in tail.values), maxlen=hl)
        state.prev_high_ma = sum(state.high_ma_state.buf) / hl
    if low_s is not None and len(low_s) >= ll:
        tail = low_s.iloc[-ll:]
        state.low_ma_state.buf = deque((float(v) for v in tail.values), maxlen=ll)
        state.prev_low_ma = sum(state.low_ma_state.buf) / ll
    return state


STATEFUL_REGISTRY["hilo"] = StatefulIndicator(
    kind="hilo",
    inputs=("high", "low", "close"),
    init=_hilo_init,
    update=_hilo_update,
    output_names=_hilo_output_names,
)
SEED_REGISTRY["hilo"] = _hilo_seed


# ===========================================================================
# DEMA  (internal_series)  -- Double EMA
# ===========================================================================
# DEMA = 2*EMA1 - EMA2  where EMA2 = EMA(EMA1, length)
# Default length = 10.

@dataclass
class DEMAState:
    ema1: EMAState
    ema2: EMAState


def _dema_init(params: Dict[str, Any]) -> DEMAState:
    length = _as_int(_param(params, "length", 10), 10)
    return DEMAState(
        ema1=ema_make(length, presma=True),
        ema2=ema_make(length, presma=True),
    )


def _dema_update(
    state: DEMAState, bar: Dict[str, Any], params: Dict[str, Any]
) -> Tuple[List[Optional[float]], DEMAState]:
    x = bar["close"]
    ema1_val, state.ema1 = ema_update_raw(state.ema1, x)
    dema_val: Optional[float] = None
    if ema1_val is not None:
        ema2_val, state.ema2 = ema_update_raw(state.ema2, ema1_val)
        if ema2_val is not None:
            dema_val = 2.0 * ema1_val - ema2_val
    return [dema_val], state


def _dema_output_names(params: Dict[str, Any]) -> List[str]:
    length = _as_int(_param(params, "length", 10), 10)
    return [f"DEMA_{length}"]


def _dema_seed(series: Dict[str, Any], params: Dict[str, Any]) -> DEMAState:
    """internal_series: recover ema1 and ema2 last values."""
    length = _as_int(_param(params, "length", 10), 10)
    state  = _dema_init(params)
    # ema1 series may be named EMA_<length> in the vectorised output
    ema1_col = f"EMA_{length}"
    s1 = series.get(ema1_col)
    if s1 is not None:
        lv = s1.dropna()
        if len(lv) > 0:
            state.ema1.last = float(lv.iloc[-1])
            state.ema1._warmup_count = length
            state.ema1._warmup_sum  = 0.0
    # ema2 series is EMA of ema1; name convention varies.
    # Try common patterns or fall back to DEMA output back-calculation.
    dema_col = _dema_output_names(params)[0]
    dema_s = series.get(dema_col)
    if dema_s is not None and s1 is not None:
        # at the last valid point: dema = 2*ema1 - ema2  => ema2 = 2*ema1 - dema
        dema_lv = dema_s.dropna()
        ema1_lv = s1.dropna()
        if len(dema_lv) > 0 and len(ema1_lv) > 0:
            last_ema1 = float(ema1_lv.iloc[-1])
            last_dema = float(dema_lv.iloc[-1])
            state.ema2.last = 2.0 * last_ema1 - last_dema
            state.ema2._warmup_count = length
            state.ema2._warmup_sum  = 0.0
    return state


STATEFUL_REGISTRY["dema"] = StatefulIndicator(
    kind="dema",
    inputs=("close",),
    init=_dema_init,
    update=_dema_update,
    output_names=_dema_output_names,
)
SEED_REGISTRY["dema"] = _dema_seed


# ===========================================================================
# TEMA  (output_only)  -- Triple EMA
# ===========================================================================
# TEMA = 3*ema1 - 3*ema2 + ema3
# ema2 = EMA(ema1), ema3 = EMA(ema2).  Default length = 10.

@dataclass
class TEMAState:
    ema1: EMAState
    ema2: EMAState
    ema3: EMAState


def _tema_init(params: Dict[str, Any]) -> TEMAState:
    length = _as_int(_param(params, "length", 10), 10)
    return TEMAState(
        ema1=ema_make(length, presma=True),
        ema2=ema_make(length, presma=True),
        ema3=ema_make(length, presma=True),
    )


def _tema_update(
    state: TEMAState, bar: Dict[str, Any], params: Dict[str, Any]
) -> Tuple[List[Optional[float]], TEMAState]:
    x = bar["close"]
    e1, state.ema1 = ema_update_raw(state.ema1, x)
    tema_val: Optional[float] = None
    if e1 is not None:
        e2, state.ema2 = ema_update_raw(state.ema2, e1)
        if e2 is not None:
            e3, state.ema3 = ema_update_raw(state.ema3, e2)
            if e3 is not None:
                tema_val = 3.0 * e1 - 3.0 * e2 + e3
    return [tema_val], state


def _tema_output_names(params: Dict[str, Any]) -> List[str]:
    length = _as_int(_param(params, "length", 10), 10)
    return [f"TEMA_{length}"]


def _tema_seed(series: Dict[str, Any], params: Dict[str, Any]) -> TEMAState:
    """output_only seed.  Back-calculate ema1/ema2 from TEMA + EMA series
    when available; otherwise recover only from output (lossy for ema1/2).
    """
    length = _as_int(_param(params, "length", 10), 10)
    state  = _tema_init(params)

    ema1_col  = f"EMA_{length}"
    tema_col  = _tema_output_names(params)[0]

    s_ema1 = series.get(ema1_col)
    s_tema = series.get(tema_col)

    # If ema1 series available, seed ema1 directly
    if s_ema1 is not None:
        lv = s_ema1.dropna()
        if len(lv) > 0:
            state.ema1.last = float(lv.iloc[-1])
            state.ema1._warmup_count = length
            state.ema1._warmup_sum  = 0.0

    # TEMA = 3*e1 - 3*e2 + e3.  With e1 known and TEMA known we still
    # need e2 or e3.  As a pragmatic fallback we replay for exact state.
    # Here we just seed ema1 and leave ema2/ema3 to be warmed up from
    # the first ema1 values that flow through.
    # If the caller needs exact numeric parity, use replay_seed().
    return state


STATEFUL_REGISTRY["tema"] = StatefulIndicator(
    kind="tema",
    inputs=("close",),
    init=_tema_init,
    update=_tema_update,
    output_names=_tema_output_names,
)
SEED_REGISTRY["tema"] = _tema_seed


# ===========================================================================
# T3  (output_only)  -- Tillson T3
# ===========================================================================
# 6 cascaded EMAs.  T3 = c1*e6 + c2*e5 + c3*e4 + c4*e3
# c1 = -a^3,  c2 = 3a^2 + 3a^3,  c3 = -6a^2 - 3a - 3a^3,  c4 = 1 + 3a + a^3 + 3a^2
# Default: length=10, a=0.7

@dataclass
class T3State:
    e1: EMAState
    e2: EMAState
    e3: EMAState
    e4: EMAState
    e5: EMAState
    e6: EMAState
    c1: float
    c2: float
    c3: float
    c4: float


def _t3_init(params: Dict[str, Any]) -> T3State:
    length = _as_int(_param(params, "length", 10), 10)
    a_raw  = _param(params, "a", 0.7)
    a      = float(a_raw) if isinstance(a_raw, (int, float)) and 0 < float(a_raw) < 1 else 0.7
    c1 =  -(a ** 3)
    c2 =  3 * a**2 + 3 * a**3
    c3 = -6 * a**2 - 3 * a - 3 * a**3
    c4 =  1 + 3 * a + a**3 + 3 * a**2
    return T3State(
        e1=ema_make(length), e2=ema_make(length),
        e3=ema_make(length), e4=ema_make(length),
        e5=ema_make(length), e6=ema_make(length),
        c1=c1, c2=c2, c3=c3, c4=c4,
    )


def _t3_update(
    state: T3State, bar: Dict[str, Any], params: Dict[str, Any]
) -> Tuple[List[Optional[float]], T3State]:
    x = bar["close"]
    v1, state.e1 = ema_update_raw(state.e1, x)
    t3_val: Optional[float] = None
    if v1 is not None:
        v2, state.e2 = ema_update_raw(state.e2, v1)
        if v2 is not None:
            v3, state.e3 = ema_update_raw(state.e3, v2)
            if v3 is not None:
                v4, state.e4 = ema_update_raw(state.e4, v3)
                if v4 is not None:
                    v5, state.e5 = ema_update_raw(state.e5, v4)
                    if v5 is not None:
                        v6, state.e6 = ema_update_raw(state.e6, v5)
                        if v6 is not None:
                            t3_val = (state.c1 * v6 + state.c2 * v5
                                      + state.c3 * v4 + state.c4 * v3)
    return [t3_val], state


def _t3_output_names(params: Dict[str, Any]) -> List[str]:
    length = _as_int(_param(params, "length", 10), 10)
    a_raw  = _param(params, "a", 0.7)
    a      = float(a_raw) if isinstance(a_raw, (int, float)) and 0 < float(a_raw) < 1 else 0.7
    return [f"T3_{length}_{a}"]


def _t3_seed(series: Dict[str, Any], params: Dict[str, Any]) -> T3State:
    """output_only seed.  Recovering all 6 EMA states from output alone
    is not feasible without the intermediate series.  We seed e1 if an
    EMA series is available; full parity requires replay_seed().
    """
    state = _t3_init(params)
    length = _as_int(_param(params, "length", 10), 10)
    ema1_col = f"EMA_{length}"
    s = series.get(ema1_col)
    if s is not None:
        lv = s.dropna()
        if len(lv) > 0:
            state.e1.last = float(lv.iloc[-1])
            state.e1._warmup_count = length
            state.e1._warmup_sum  = 0.0
    return state


STATEFUL_REGISTRY["t3"] = StatefulIndicator(
    kind="t3",
    inputs=("close",),
    init=_t3_init,
    update=_t3_update,
    output_names=_t3_output_names,
)
SEED_REGISTRY["t3"] = _t3_seed


# ===========================================================================
# SSF  (internal_series)  -- Ehlers 2-pole Super Smoother Filter
# ===========================================================================
# Ehlers (non-everget) variant:
#   ratio = sqrt2 / length
#   a     = exp(-pi * ratio)
#   b     = 2 * a * cos(180 * ratio)   <-- note: 180 degrees in the original
#   c     = a*a - b + 1
#   ssf[i] = 0.5*c*(x[i]+x[i-1]) + b*ssf[i-1] - a*a*ssf[i-2]
#
# Everget variant replaces 180*ratio with pi*sqrt2/n (radians).
# Defaults: length=20, pi=3.14159, sqrt2=1.414, everget=False

@dataclass
class SSFState:
    length:  int
    everget: bool
    a: float
    b: float
    c: float       # 0.5 * (a*a - b + 1)
    a2: float      # a*a
    prev_x:  Optional[float] = None   # x[i-1]
    prev1:   Optional[float] = None   # ssf[i-1]
    prev2:   Optional[float] = None   # ssf[i-2]


def _ssf_init(params: Dict[str, Any]) -> SSFState:
    length  = _as_int(_param(params, "length", 20), 20)
    pi_val  = _as_float(_param(params, "pi", 3.14159), 3.14159)
    sqrt2   = _as_float(_param(params, "sqrt2", 1.414), 1.414)
    everget = bool(_param(params, "everget", False))

    if everget:
        arg = pi_val * sqrt2 / length
        a   = exp(-arg)
        b   = 2.0 * a * cos(arg)
    else:
        ratio = sqrt2 / length
        a     = exp(-pi_val * ratio)
        b     = 2.0 * a * cos(180.0 * ratio)   # Ehlers uses degrees here

    c  = 0.5 * (a * a - b + 1.0)
    a2 = a * a
    return SSFState(length=length, everget=everget, a=a, b=b, c=c, a2=a2)


def _ssf_update(
    state: SSFState, bar: Dict[str, Any], params: Dict[str, Any]
) -> Tuple[List[Optional[float]], SSFState]:
    x = bar["close"]

    if state.prev1 is None:
        # bar 0: result[0] = x[0]  (copy(x) seed)
        state.prev2 = None
        state.prev1 = x
        state.prev_x = x
        return [x], state

    if state.prev2 is None:
        # bar 1: result[1] = x[1]  (copy(x) seed)
        state.prev2 = state.prev1
        state.prev1 = x
        state.prev_x = x
        return [x], state

    # bar >= 2: recursive formula
    val = (state.c * (x + state.prev_x)
           + state.b * state.prev1
           - state.a2 * state.prev2)
    state.prev2 = state.prev1
    state.prev1 = val
    state.prev_x = x
    return [val], state


def _ssf_output_names(params: Dict[str, Any]) -> List[str]:
    length  = _as_int(_param(params, "length", 20), 20)
    everget = bool(_param(params, "everget", False))
    return [f"SSF{'e' if everget else ''}_{length}"]


def _ssf_seed(series: Dict[str, Any], params: Dict[str, Any]) -> SSFState:
    """internal_series seed: recover prev1, prev2, prev_x from tails."""
    state = _ssf_init(params)
    col   = _ssf_output_names(params)[0]
    ssf_s = series.get(col)
    close_s = series.get("close")
    if ssf_s is not None and len(ssf_s) >= 2:
        state.prev1 = float(ssf_s.iloc[-1])
        state.prev2 = float(ssf_s.iloc[-2])
    if close_s is not None and len(close_s) >= 1:
        state.prev_x = float(close_s.iloc[-1])
    return state


STATEFUL_REGISTRY["ssf"] = StatefulIndicator(
    kind="ssf",
    inputs=("close",),
    init=_ssf_init,
    update=_ssf_update,
    output_names=_ssf_output_names,
)
SEED_REGISTRY["ssf"] = _ssf_seed


# ===========================================================================
# SSF3  (internal_series)  -- Ehlers 3-pole Super Smoother Filter
# ===========================================================================
# Coefficients (from nb_ssf3):
#   a  = exp(-pi / n)
#   b  = 2*a*cos(-pi*sqrt3/n)      <-- note negative sign in cos arg
#   c  = a*a
#   d4 = c*c
#   d3 = -c*(1+b)
#   d2 = b + c
#   d1 = 1 - d2 - d3 - d4
#   ssf3[i] = d1*x[i] + d2*ssf3[i-1] + d3*ssf3[i-2] + d4*ssf3[i-3]
# First 3 bars: result = copy(x)  (i.e. result[0..2] = x[0..2]).
# Defaults: length=20, pi=3.14159, sqrt3=1.732

@dataclass
class SSF3State:
    d1: float
    d2: float
    d3: float
    d4: float
    prev1: Optional[float] = None   # ssf3[i-1]
    prev2: Optional[float] = None   # ssf3[i-2]
    prev3: Optional[float] = None   # ssf3[i-3]
    _count: int = 0                 # bars seen


def _ssf3_init(params: Dict[str, Any]) -> SSF3State:
    length = _as_int(_param(params, "length", 20), 20)
    pi_val = _as_float(_param(params, "pi", 3.14159), 3.14159)
    sqrt3  = _as_float(_param(params, "sqrt3", 1.732), 1.732)

    a  = exp(-pi_val / length)
    b  = 2.0 * a * cos(-pi_val * sqrt3 / length)
    c  = a * a
    d4 = c * c
    d3 = -c * (1.0 + b)
    d2 = b + c
    d1 = 1.0 - d2 - d3 - d4
    return SSF3State(d1=d1, d2=d2, d3=d3, d4=d4)


def _ssf3_update(
    state: SSF3State, bar: Dict[str, Any], params: Dict[str, Any]
) -> Tuple[List[Optional[float]], SSF3State]:
    x = bar["close"]
    state._count += 1

    if state._count <= 3:
        # first 3 bars: output = x (copy seed)
        # shift history
        state.prev3 = state.prev2
        state.prev2 = state.prev1
        state.prev1 = x
        return [x], state

    # recursive
    val = (state.d1 * x
           + state.d2 * state.prev1
           + state.d3 * state.prev2
           + state.d4 * state.prev3)
    state.prev3 = state.prev2
    state.prev2 = state.prev1
    state.prev1 = val
    return [val], state


def _ssf3_output_names(params: Dict[str, Any]) -> List[str]:
    length = _as_int(_param(params, "length", 20), 20)
    return [f"SSF3_{length}"]


def _ssf3_seed(series: Dict[str, Any], params: Dict[str, Any]) -> SSF3State:
    state = _ssf3_init(params)
    col   = _ssf3_output_names(params)[0]
    ssf_s = series.get(col)
    if ssf_s is not None and len(ssf_s) >= 3:
        state.prev1 = float(ssf_s.iloc[-1])
        state.prev2 = float(ssf_s.iloc[-2])
        state.prev3 = float(ssf_s.iloc[-3])
        state._count = len(ssf_s)   # mark past warmup
    return state


STATEFUL_REGISTRY["ssf3"] = StatefulIndicator(
    kind="ssf3",
    inputs=("close",),
    init=_ssf3_init,
    update=_ssf3_update,
    output_names=_ssf3_output_names,
)
SEED_REGISTRY["ssf3"] = _ssf3_seed


# ===========================================================================
# SUPERTREND  (internal_series)
# ===========================================================================
# Outputs: SUPERT (trend), SUPERTd (direction), SUPERTl (long), SUPERTs (short)
# Defaults: length=7, multiplier=3.0
# Uses ATR internally (Wilder / RMA, SMA-seeded).
# hl2 = (high + low) / 2
# ub = hl2 + multiplier * atr
# lb = hl2 - multiplier * atr
# Direction logic (vectorised):
#   if close > ub_prev  -> dir = 1
#   elif close < lb_prev -> dir = -1
#   else dir = dir_prev; clamp lb/ub accordingly

@dataclass
class SUPERTRENDState:
    multiplier: float
    atr_state:  ATRState
    dir_prev:   int                    # 1 or -1
    lb_prev:    Optional[float] = None
    ub_prev:    Optional[float] = None
    _started:   bool = False           # True once first valid ATR produced


def _supertrend_init(params: Dict[str, Any]) -> SUPERTRENDState:
    length     = _as_int(_param(params, "length", 7), 7)
    atr_length = _as_int(_param(params, "atr_length", length), length)
    mult       = _as_float(_param(params, "multiplier", 3.0), 3.0)
    atr_state  = ATRState(length=atr_length)
    return SUPERTRENDState(multiplier=mult, atr_state=atr_state, dir_prev=1)


def _supertrend_update(
    state: SUPERTRENDState, bar: Dict[str, Any], params: Dict[str, Any]
) -> Tuple[List[Optional[float]], SUPERTRENDState]:
    high  = bar["high"]
    low   = bar["low"]
    close = bar["close"]

    atr_val, state.atr_state = atr_update_raw(state.atr_state, high, low, close)

    if atr_val is None:
        # ATR still warming up
        return [None, None, None, None], state

    hl2  = (high + low) / 2.0
    matr = state.multiplier * atr_val
    lb   = hl2 - matr
    ub   = hl2 + matr

    if not state._started:
        # first bar with valid ATR -- initialise direction
        state._started = True
        state.dir_prev = 1
        state.lb_prev  = lb
        state.ub_prev  = ub
        trend  = lb
        long_v = lb
        short_v: Optional[float] = None
        dir_v  = 1
    else:
        # direction logic
        if close > state.ub_prev:
            direction = 1
        elif close < state.lb_prev:
            direction = -1
        else:
            direction = state.dir_prev
            if direction > 0 and lb < state.lb_prev:
                lb = state.lb_prev
            if direction < 0 and ub > state.ub_prev:
                ub = state.ub_prev

        state.dir_prev = direction
        state.lb_prev  = lb
        state.ub_prev  = ub

        if direction > 0:
            trend   = lb
            long_v  = lb
            short_v = None
            dir_v   = 1
        else:
            trend   = ub
            long_v  = None
            short_v = ub
            dir_v   = -1

    return [trend, float(dir_v), long_v, short_v], state


def _supertrend_output_names(params: Dict[str, Any]) -> List[str]:
    length = _as_int(_param(params, "length", 7), 7)
    mult   = _as_float(_param(params, "multiplier", 3.0), 3.0)
    _props = f"_{length}_{mult}"
    return [
        f"SUPERT{_props}",
        f"SUPERTd{_props}",
        f"SUPERTl{_props}",
        f"SUPERTs{_props}",
    ]


def _supertrend_seed(series: Dict[str, Any], params: Dict[str, Any]) -> SUPERTRENDState:
    """internal_series seed: recover ATR state and direction/band state."""
    length     = _as_int(_param(params, "length", 7), 7)
    atr_length = _as_int(_param(params, "atr_length", length), length)
    state      = _supertrend_init(params)
    names      = _supertrend_output_names(params)

    # direction from SUPERTd
    d_s = series.get(names[1])
    if d_s is not None:
        lv = d_s.dropna()
        if len(lv) > 0:
            state.dir_prev = int(lv.iloc[-1])
            state._started = True

    # long / short bands
    l_s = series.get(names[2])
    s_s = series.get(names[3])
    if l_s is not None:
        lv = l_s.dropna()
        if len(lv) > 0:
            state.lb_prev = float(lv.iloc[-1])
    if s_s is not None:
        lv = s_s.dropna()
        if len(lv) > 0:
            state.ub_prev = float(lv.iloc[-1])

    # ATR state: recover from high/low/close tails via replay
    # (ATR internal state is too complex for simple extraction)
    high_s  = series.get("high")
    low_s   = series.get("low")
    close_s = series.get("close")
    if high_s is not None and low_s is not None and close_s is not None:
        n = len(high_s)
        atr_st = ATRState(length=atr_length)
        for i in range(n):
            import pandas as _pd
            h = high_s.iloc[i]
            lo = low_s.iloc[i]
            cl = close_s.iloc[i]
            if _pd.isna(h) or _pd.isna(lo) or _pd.isna(cl):
                continue
            _, atr_st = atr_update_raw(atr_st, float(h), float(lo), float(cl))
        state.atr_state = atr_st

    return state


STATEFUL_REGISTRY["supertrend"] = StatefulIndicator(
    kind="supertrend",
    inputs=("high", "low", "close"),
    init=_supertrend_init,
    update=_supertrend_update,
    output_names=_supertrend_output_names,
)
SEED_REGISTRY["supertrend"] = _supertrend_seed


# ===========================================================================
# VIDYA  (internal_series)  -- Variable Index Dynamic Average
# ===========================================================================
# alpha_base = 2 / (length + 1)
# CMO is computed as a rolling window stat:
#   mom[i] = close[i] - close[i-1]   (drift=1)
#   pos_sum = sum of max(mom, 0) over last `length` bars
#   neg_sum = sum of abs(min(mom, 0)) over last `length` bars
#   cmo = (pos_sum - neg_sum) / (pos_sum + neg_sum)
#   abs_cmo = abs(cmo)
# vidya[i] = alpha_base * abs_cmo * close[i] + (1 - alpha_base*abs_cmo) * vidya[i-1]
# Vectorised: first `length` bars of vidya = 0 (then NaN-replaced).
# State: deque of last `length` momentum values, prev_vidya, prev_close.
# Default: length=14, drift=1

@dataclass
class VIDYAState:
    length:     int
    alpha_base: float
    prev_vidya: float                            # starts at 0.0 per vectorised
    prev_close: Optional[float] = None
    # ring buffer of momentum values (last `length`)
    mom_buf:    deque = field(default_factory=deque)
    _count:     int = 0                          # total bars processed


def _vidya_init(params: Dict[str, Any]) -> VIDYAState:
    length = _as_int(_param(params, "length", 14), 14)
    alpha  = 2.0 / (length + 1.0)
    return VIDYAState(
        length=length, alpha_base=alpha,
        prev_vidya=0.0,
        mom_buf=deque(maxlen=length),
    )


def _vidya_update(
    state: VIDYAState, bar: Dict[str, Any], params: Dict[str, Any]
) -> Tuple[List[Optional[float]], VIDYAState]:
    close = bar["close"]
    state._count += 1

    if state.prev_close is not None:
        mom = close - state.prev_close
        state.mom_buf.append(mom)

    state.prev_close = close

    # Need at least `length` momentum values (requires length+1 bars)
    if len(state.mom_buf) < state.length:
        # vidya stays 0 during warmup; output None (will be NaN)
        return [None], state

    # CMO from rolling window
    pos_sum = 0.0
    neg_sum = 0.0
    for m in state.mom_buf:
        if m > 0:
            pos_sum += m
        else:
            neg_sum += abs(m)
    denom = pos_sum + neg_sum
    cmo   = (pos_sum - neg_sum) / denom if denom != 0.0 else 0.0
    abs_cmo = abs(cmo)

    eff_alpha = state.alpha_base * abs_cmo
    state.prev_vidya = eff_alpha * close + (1.0 - eff_alpha) * state.prev_vidya

    val = state.prev_vidya if state.prev_vidya != 0.0 else None
    return [val], state


def _vidya_output_names(params: Dict[str, Any]) -> List[str]:
    length = _as_int(_param(params, "length", 14), 14)
    return [f"VIDYA_{length}"]


def _vidya_seed(series: Dict[str, Any], params: Dict[str, Any]) -> VIDYAState:
    """internal_series: replay for exact parity."""
    return replay_seed("vidya", series, params)


STATEFUL_REGISTRY["vidya"] = StatefulIndicator(
    kind="vidya",
    inputs=("close",),
    init=_vidya_init,
    update=_vidya_update,
    output_names=_vidya_output_names,
)
SEED_REGISTRY["vidya"] = _vidya_seed


# ===========================================================================
# JMA  (internal_series)  -- Jurik Moving Average
# ===========================================================================
# This is a faithful port of the vectorised loop in jma.py.
# Static derived constants are computed once in init.
# Mutable per-bar state: ma1, ma2, det0, det1, kv, uBand, lBand, v_sum, volty_prev
# Default: length=7, phase=0

@dataclass
class JMAState:
    # derived constants (computed once)
    sum_length: int
    length_half: float   # 0.5 * (length-1)
    pr:         float
    length1:    float
    pow1:       float
    length2:    float
    bet:        float
    beta:       float
    # mutable per-bar state
    ma1:        float = 0.0
    ma2:        float = 0.0
    det0:       float = 0.0
    det1:       float = 0.0
    kv:         float = 0.0
    uBand:      float = 0.0
    lBand:      float = 0.0
    prev_jma:   float = 0.0
    # v_sum rolling buffer (sum_length=10 window)
    v_sum_buf:  deque = field(default_factory=deque)   # stores volty values
    v_sum_val:  float = 0.0                            # current v_sum
    # avg_volty window (65 bars of v_sum)
    vsum_history: deque = field(default_factory=deque)  # stores v_sum values
    _count:     int = 0                                # bars seen (0-based idx)


def _jma_init(params: Dict[str, Any]) -> JMAState:
    _length = _as_int(_param(params, "length", 7), 7)
    phase   = _as_float(_param(params, "phase", 0.0), 0.0)

    sum_length = 10
    length_half = 0.5 * (_length - 1)
    pr = 0.5 if phase < -100 else (2.5 if phase > 100 else 1.5 + phase * 0.01)
    length1 = max((log(sqrt(length_half)) / log(2.0)) + 2.0, 0.0) if length_half > 0 else 2.0
    pow1    = max(length1 - 2.0, 0.5)
    length2 = length1 * sqrt(length_half) if length_half > 0 else 0.0
    bet     = length2 / (length2 + 1.0) if (length2 + 1.0) != 0 else 0.0
    beta    = 0.45 * (_length - 1) / (0.45 * (_length - 1) + 2.0)

    return JMAState(
        sum_length=sum_length,
        length_half=length_half,
        pr=pr, length1=length1, pow1=pow1,
        length2=length2, bet=bet, beta=beta,
        v_sum_buf=deque(maxlen=sum_length),
        vsum_history=deque(maxlen=66),  # need 65+1 for the window
    )


def _jma_update(
    state: JMAState, bar: Dict[str, Any], params: Dict[str, Any]
) -> Tuple[List[Optional[float]], JMAState]:
    price = bar["close"]

    if state._count == 0:
        # i == 0 in vectorised: jma[0] = ma1 = uBand = lBand = close[0]
        state.ma1   = price
        state.uBand = price
        state.lBand = price
        state.prev_jma = price
        state.v_sum_buf.append(0.0)
        state.v_sum_val = 0.0
        state.vsum_history.append(0.0)
        state._count = 1
        _length = _as_int(_param(params, "length", 7), 7)
        # first _length-1 bars are NaN in vectorised output
        return [price if _length <= 1 else None], state

    state._count += 1

    # Price volatility
    del1 = price - state.uBand
    del2 = price - state.lBand
    volty = max(abs(del1), abs(del2)) if abs(del1) != abs(del2) else 0.0

    # v_sum: rolling sum of volty over sum_length window
    # v_sum[i] = v_sum[i-1] + (volty[i] - volty[i - sum_length]) / sum_length
    # The oldest volty that should be subtracted is volty[i-sum_length].
    # We store the last sum_length volty values in v_sum_buf.
    old_volty = state.v_sum_buf[0] if len(state.v_sum_buf) >= state.sum_length else 0.0
    state.v_sum_val += (volty - old_volty) / state.sum_length
    state.v_sum_buf.append(volty)

    # avg_volty = average of v_sum over last 66 entries (indices max(i-65,0) .. i)
    state.vsum_history.append(state.v_sum_val)
    avg_volty = sum(state.vsum_history) / len(state.vsum_history)

    d_volty   = 0.0 if avg_volty == 0.0 else volty / avg_volty
    r_volty_max = state.length1 ** (1.0 / state.pow1) if state.pow1 != 0 else state.length1
    r_volty   = max(1.0, min(r_volty_max, d_volty))

    # Jurik volatility bands
    pow2 = r_volty ** state.pow1
    state.kv   = state.bet ** (sqrt(pow2))
    state.uBand = price if del1 > 0 else price - (state.kv * del1)
    state.lBand = price if del2 < 0 else price - (state.kv * del2)

    # Jurik dynamic factor
    power = r_volty ** state.pow1
    alpha = state.beta ** power

    # 1st stage
    state.ma1 = (1.0 - alpha) * price + alpha * state.ma1

    # 2nd stage (Kalman)
    state.det0 = (1.0 - state.beta) * (price - state.ma1) + state.beta * state.det0
    state.ma2  = state.ma1 + state.pr * state.det0

    # 3rd stage (Jurik adaptive)
    state.det1 = ((state.ma2 - state.prev_jma) * (1.0 - alpha) * (1.0 - alpha)
                  + alpha * alpha * state.det1)
    state.prev_jma = state.prev_jma + state.det1

    _length = _as_int(_param(params, "length", 7), 7)
    # vectorised sets jma[0:_length-1] = NaN
    out_val: Optional[float] = state.prev_jma if state._count >= _length else None
    return [out_val], state


def _jma_output_names(params: Dict[str, Any]) -> List[str]:
    _length = _as_int(_param(params, "length", 7), 7)
    phase   = _as_float(_param(params, "phase", 0.0), 0.0)
    return [f"JMA_{_length}_{phase}"]


def _jma_seed(series: Dict[str, Any], params: Dict[str, Any]) -> JMAState:
    """internal_series: replay for exact parity."""
    return replay_seed("jma", series, params)


STATEFUL_REGISTRY["jma"] = StatefulIndicator(
    kind="jma",
    inputs=("close",),
    init=_jma_init,
    update=_jma_update,
    output_names=_jma_output_names,
)
SEED_REGISTRY["jma"] = _jma_seed


# ===========================================================================
# HWMA  (replay_only)  -- Holt-Winter Moving Average
# ===========================================================================
# F, V, A triple-exponential state.  No SEED_REGISTRY entry.
# Defaults: na=0.2, nb=0.1, nc=0.1
# Vectorised initialisation: last_a=0, last_v=0, last_f=close[0]
# Output = F + V + 0.5*A

@dataclass
class HWMAState:
    na: float
    nb: float
    nc: float
    last_f: Optional[float] = None
    last_v: float = 0.0
    last_a: float = 0.0


def _hwma_init(params: Dict[str, Any]) -> HWMAState:
    na = _as_float(_param(params, "na", 0.2), 0.2)
    nb = _as_float(_param(params, "nb", 0.1), 0.1)
    nc = _as_float(_param(params, "nc", 0.1), 0.1)
    # clamp to (0,1) per vectorised validation
    na = na if 0 < na < 1 else 0.2
    nb = nb if 0 < nb < 1 else 0.1
    nc = nc if 0 < nc < 1 else 0.1
    return HWMAState(na=na, nb=nb, nc=nc)


def _hwma_update(
    state: HWMAState, bar: Dict[str, Any], params: Dict[str, Any]
) -> Tuple[List[Optional[float]], HWMAState]:
    x = bar["close"]

    if state.last_f is None:
        # very first bar: initialise last_f = close[0], last_v=0, last_a=0
        # then run the formula for i=0
        state.last_f = x
        state.last_v = 0.0
        state.last_a = 0.0

    F = (1.0 - state.na) * (state.last_f + state.last_v + 0.5 * state.last_a) + state.na * x
    V = (1.0 - state.nb) * (state.last_v + state.last_a) + state.nb * (F - state.last_f)
    A = (1.0 - state.nc) * state.last_a + state.nc * (V - state.last_v)

    out = F + V + 0.5 * A

    state.last_a = A
    state.last_f = F
    state.last_v = V

    return [out], state


def _hwma_output_names(params: Dict[str, Any]) -> List[str]:
    na = _as_float(_param(params, "na", 0.2), 0.2)
    nb = _as_float(_param(params, "nb", 0.1), 0.1)
    nc = _as_float(_param(params, "nc", 0.1), 0.1)
    na = na if 0 < na < 1 else 0.2
    nb = nb if 0 < nb < 1 else 0.1
    nc = nc if 0 < nc < 1 else 0.1
    return [f"HWMA_{na}_{nb}_{nc}"]


# replay_only: no SEED_REGISTRY entry
STATEFUL_REGISTRY["hwma"] = StatefulIndicator(
    kind="hwma",
    inputs=("close",),
    init=_hwma_init,
    update=_hwma_update,
    output_names=_hwma_output_names,
)


# ===========================================================================
# MAMA  (internal_series)  -- MESA Adaptive Moving Average
# ===========================================================================
# Ehlers Hilbert Transform based.  Outputs: MAMA, FAMA.
# Defaults: fastlimit=0.5, slowlimit=0.05, prenan=3
# State mirrors all the arrays in nb_mama as scalar prev values.
# The vectorised loop starts at i=3 (first 3 bars are seed = 0).

@dataclass
class MAMAState:
    fastlimit: float
    slowlimit: float
    # Hilbert Transform history buffers (need i, i-1, i-2, i-3, i-4, i-6)
    # We keep a deque of the last 7 raw prices
    x_buf:    deque = field(default_factory=deque)  # maxlen=7
    # Per-step intermediate scalars that require history (i-2, i-4, i-6)
    # We store the last 7 values of each intermediate series
    wma4_buf:  deque = field(default_factory=deque)  # maxlen=7
    dt_buf:    deque = field(default_factory=deque)  # maxlen=7
    q1_buf:    deque = field(default_factory=deque)  # maxlen=7
    i1_buf:    deque = field(default_factory=deque)  # maxlen=7
    # Smoothed scalars (only need prev)
    i2_prev:   float = 0.0
    q2_prev:   float = 0.0
    re_prev:   float = 0.0
    im_prev:   float = 0.0
    period_prev: float = 0.0
    smp_prev:  float = 0.0
    phase_prev: float = 0.0
    mama_prev: float = 0.0
    fama_prev: float = 0.0
    _count:    int = 0


def _mama_init(params: Dict[str, Any]) -> MAMAState:
    fl = _as_float(_param(params, "fastlimit", 0.5), 0.5)
    sl = _as_float(_param(params, "slowlimit", 0.05), 0.05)
    return MAMAState(
        fastlimit=fl, slowlimit=sl,
        x_buf=deque(maxlen=7),
        wma4_buf=deque(maxlen=7),
        dt_buf=deque(maxlen=7),
        q1_buf=deque(maxlen=7),
        i1_buf=deque(maxlen=7),
    )


def _buf_get(buf: deque, offset: int) -> float:
    """Get value at `offset` positions back from end.  offset=0 = latest.
    Returns 0.0 if not enough history."""
    idx = len(buf) - 1 - offset
    return float(buf[idx]) if idx >= 0 else 0.0


def _mama_update(
    state: MAMAState, bar: Dict[str, Any], params: Dict[str, Any]
) -> Tuple[List[Optional[float]], MAMAState]:
    x = bar["close"]
    state.x_buf.append(x)
    state._count += 1  # 1-based count after append

    # Constants
    a_c, b_c = 0.0962, 0.5769
    p_w      = 0.2
    smp_w    = 0.33
    smp_w_c  = 0.67

    i = state._count - 1  # 0-based index

    if i < 3:
        # bars 0, 1, 2: all intermediates stay 0; mama/fama = 0
        # Push zeros into history buffers to keep alignment
        state.wma4_buf.append(0.0)
        state.dt_buf.append(0.0)
        state.q1_buf.append(0.0)
        state.i1_buf.append(0.0)
        prenan = _as_int(_param(params, "prenan", 3), 3)
        return [None, None], state  # prenan covers at least first 3

    # --- WMA(x,4) ---
    # wma4 = 0.4*x[i] + 0.3*x[i-1] + 0.2*x[i-2] + 0.1*x[i-3]
    x0 = _buf_get(state.x_buf, 0)
    x1 = _buf_get(state.x_buf, 1)
    x2 = _buf_get(state.x_buf, 2)
    x3 = _buf_get(state.x_buf, 3)
    wma4 = 0.4 * x0 + 0.3 * x1 + 0.2 * x2 + 0.1 * x3
    state.wma4_buf.append(wma4)

    # adj_prev_period
    adj_prev_period = 0.075 * state.period_prev + 0.54

    # --- Detrended WMA ---
    # dt[i] = adj * (a*wma4[i] + b*wma4[i-2] - b*wma4[i-4] - a*wma4[i-6])
    wma4_0 = _buf_get(state.wma4_buf, 0)
    wma4_2 = _buf_get(state.wma4_buf, 2)
    wma4_4 = _buf_get(state.wma4_buf, 4)
    wma4_6 = _buf_get(state.wma4_buf, 6)
    dt = adj_prev_period * (a_c * wma4_0 + b_c * wma4_2 - b_c * wma4_4 - a_c * wma4_6)
    state.dt_buf.append(dt)

    # --- Q1 and I1 ---
    dt_0 = _buf_get(state.dt_buf, 0)
    dt_2 = _buf_get(state.dt_buf, 2)
    dt_4 = _buf_get(state.dt_buf, 4)
    dt_6 = _buf_get(state.dt_buf, 6)
    q1 = adj_prev_period * (a_c * dt_0 + b_c * dt_2 - b_c * dt_4 - a_c * dt_6)
    state.q1_buf.append(q1)

    # i1[i] = dt[i-3]
    i1 = _buf_get(state.dt_buf, 3)
    state.i1_buf.append(i1)

    # --- Phase Q1 and I1 by 90 degrees ---
    i1_0 = _buf_get(state.i1_buf, 0)
    i1_2 = _buf_get(state.i1_buf, 2)
    i1_4 = _buf_get(state.i1_buf, 4)
    i1_6 = _buf_get(state.i1_buf, 6)
    ji   = adj_prev_period * (a_c * i1_0 + b_c * i1_2 - b_c * i1_4 - a_c * i1_6)

    q1_0 = _buf_get(state.q1_buf, 0)
    q1_2 = _buf_get(state.q1_buf, 2)
    q1_4 = _buf_get(state.q1_buf, 4)
    q1_6 = _buf_get(state.q1_buf, 6)
    jq   = adj_prev_period * (a_c * q1_0 + b_c * q1_2 - b_c * q1_4 - a_c * q1_6)

    # --- Phasor Addition ---
    i2 = i1 - jq
    q2 = q1 + ji

    # --- Smooth I2 & Q2 ---
    i2 = p_w * i2 + (1.0 - p_w) * state.i2_prev
    q2 = p_w * q2 + (1.0 - p_w) * state.q2_prev

    # --- Homodyne Discriminator ---
    re = state.i2_prev * i2 + state.q2_prev * q2   # note: vectorised uses i2[i]*i2[i-1] etc
    # Correction: vectorised is re[i] = i2[i]*i2[i-1] + q2[i]*q2[i-1]
    # At this point i2/q2 are the NEW smoothed values; i2_prev/q2_prev are from last bar.
    # But we already overwrote i2/q2 with smoothed. The vectorised code overwrites in-place
    # BEFORE computing re. So re uses the new i2[i] (smoothed) and old i2[i-1] (prev smoothed).
    re = i2 * state.i2_prev + q2 * state.q2_prev
    im = i2 * state.q2_prev - q2 * state.i2_prev
    # vectorised: im[i] = i2[i]*q2[i-1] + q2[i]*i2[i-1]  -- note the sign
    # Actually: im[i] = i2[i]*q2[i-1] + q2[i]*i2[i-1]  (addition, not subtraction)
    im = i2 * state.q2_prev + q2 * state.i2_prev

    # --- Smooth Re & Im ---
    re = p_w * re + (1.0 - p_w) * state.re_prev
    im = p_w * im + (1.0 - p_w) * state.im_prev

    # --- Period ---
    if im != 0.0 and re != 0.0:
        period = 360.0 / atan(im / re)
    else:
        period = 0.0

    if period > 1.5 * state.period_prev:
        period = 1.5 * state.period_prev
    if period < 0.67 * state.period_prev:
        period = 0.67 * state.period_prev
    if period < 6.0:
        period = 6.0
    if period > 50.0:
        period = 50.0

    period = p_w * period + (1.0 - p_w) * state.period_prev
    smp    = smp_w * period + smp_w_c * state.smp_prev

    # --- Phase ---
    if i1 != 0.0:
        phase = atan(q1 / i1)
    else:
        phase = 0.0

    dphase = state.phase_prev - phase
    if dphase < 1.0:
        dphase = 1.0

    alpha = state.fastlimit / dphase
    if alpha > state.fastlimit:
        alpha = state.fastlimit
    if alpha < state.slowlimit:
        alpha = state.slowlimit

    # --- MAMA & FAMA ---
    state.mama_prev = alpha * x + (1.0 - alpha) * state.mama_prev
    state.fama_prev = 0.5 * alpha * state.mama_prev + (1.0 - 0.5 * alpha) * state.fama_prev

    # Update prev state
    state.i2_prev     = i2
    state.q2_prev     = q2
    state.re_prev     = re
    state.im_prev     = im
    state.period_prev = period
    state.smp_prev    = smp
    state.phase_prev  = phase

    prenan = _as_int(_param(params, "prenan", 3), 3)
    if state._count <= prenan:
        return [None, None], state

    return [state.mama_prev, state.fama_prev], state


def _mama_output_names(params: Dict[str, Any]) -> List[str]:
    fl = _as_float(_param(params, "fastlimit", 0.5), 0.5)
    sl = _as_float(_param(params, "slowlimit", 0.05), 0.05)
    _props = f"_{fl}_{sl}"
    return [f"MAMA{_props}", f"FAMA{_props}"]


def _mama_seed(series: Dict[str, Any], params: Dict[str, Any]) -> MAMAState:
    """internal_series: replay for exact parity."""
    return replay_seed("mama", series, params)


STATEFUL_REGISTRY["mama"] = StatefulIndicator(
    kind="mama",
    inputs=("close",),
    init=_mama_init,
    update=_mama_update,
    output_names=_mama_output_names,
)
SEED_REGISTRY["mama"] = _mama_seed


# ===========================================================================
# PIVOTS  (internal_series)  -- Pivot Points
# ===========================================================================
# Streaming pivots: detect session boundaries via timestamp.
# When a new session starts the previous session's OHLC is used to compute
# pivot levels which are then held constant until the next session change.
# Defaults: method="traditional", anchor="D"
#
# Output columns depend on method:
#   traditional / classic / camarilla / woodie : P, S1-S4, R1-R4  (9 cols)
#   fibonacci                                  : P, S1-S3, R1-R3  (7 cols)
#   demark                                     : P, S1, R1        (3 cols)

@dataclass
class PIVOTSState:
    method: str
    anchor: str
    # current session accumulator
    cur_open:  Optional[float] = None
    cur_high:  Optional[float] = None
    cur_low:   Optional[float] = None
    cur_close: Optional[float] = None
    # previous session OHLC (used to compute active pivot levels)
    prev_open:  Optional[float] = None
    prev_high:  Optional[float] = None
    prev_low:   Optional[float] = None
    prev_close: Optional[float] = None
    # last session key (date string / period label)
    session_key: Optional[str] = None
    # cached pivot levels (None until first session rollover)
    levels: Optional[List[float]] = None


def _pivots_init(params: Dict[str, Any]) -> PIVOTSState:
    method = str(_param(params, "method", "traditional"))
    anchor = str(_param(params, "anchor", "D"))
    return PIVOTSState(method=method, anchor=anchor)


def _session_key_from_bar(bar: Dict[str, Any], anchor: str) -> str:
    """Extract a session identifier from the bar.
    If 'timestamp' is present use it; otherwise use a generic counter.
    For anchor='D' we use the date portion of the timestamp.
    """
    ts = bar.get("timestamp")
    if ts is not None:
        # handle pandas Timestamp or datetime
        try:
            if anchor.upper() in ("D",):
                return str(ts)[:10]  # YYYY-MM-DD
            elif anchor.upper() in ("W", "WE"):
                # ISO week
                import datetime
                if hasattr(ts, "isocalendar"):
                    iso = ts.isocalendar()
                    return f"{iso[0]}-W{iso[1]}"
                return str(ts)[:10]
            elif anchor.upper() in ("M", "ME"):
                return str(ts)[:7]  # YYYY-MM
            elif anchor.upper() in ("Y", "YE"):
                return str(ts)[:4]  # YYYY
        except Exception:
            pass
        return str(ts)[:10]
    # fallback: no timestamp -- cannot detect session change
    return ""


def _compute_pivots(method: str, o: float, h: float, lo: float, cl: float) -> List[float]:
    """Compute pivot levels from previous session OHLC.
    Returns [P, S1, S2, S3, S4, R1, R2, R3, R4] with NaN for unused slots.
    Order matches output_names ordering.
    """
    tp = (h + lo + cl) / 3.0
    hl_range = h - lo if h != lo else 1e-10  # non_zero_range guard

    if method == "traditional":
        s1 = 2*tp - h;         r1 = 2*tp - lo
        s2 = tp - hl_range;    r2 = tp + hl_range
        s3 = tp - 2*hl_range;  r3 = tp + 2*hl_range
        s4 = tp - 2*hl_range;  r4 = tp + 2*hl_range  # vectorised duplicates s3/r3
        return [tp, s1, s2, s3, s4, r1, r2, r3, r4]

    elif method == "classic":
        s1 = 2*tp - h;         r1 = 2*tp - lo
        s2 = tp - hl_range;    r2 = tp + hl_range
        s3 = tp - 2*hl_range;  r3 = tp + 2*hl_range
        s4 = tp - 3*hl_range;  r4 = tp + 3*hl_range
        return [tp, s1, s2, s3, s4, r1, r2, r3, r4]

    elif method == "camarilla":
        s1 = cl - 11.0/120.0 * hl_range;  r1 = cl + 11.0/120.0 * hl_range
        s2 = cl - 11.0/60.0  * hl_range;  r2 = cl + 11.0/60.0  * hl_range
        s3 = cl - 0.275      * hl_range;  r3 = cl + 0.275      * hl_range
        s4 = cl - 0.55       * hl_range;  r4 = cl + 0.55       * hl_range
        return [tp, s1, s2, s3, s4, r1, r2, r3, r4]

    elif method == "fibonacci":
        s1 = tp - 0.382 * hl_range;  r1 = tp + 0.382 * hl_range
        s2 = tp - 0.618 * hl_range;  r2 = tp + 0.618 * hl_range
        s3 = tp - hl_range;          r3 = tp + hl_range
        return [tp, s1, s2, s3, r1, r2, r3]

    elif method == "woodie":
        tp_w = (2*o + h + lo) / 4.0
        s1 = 2*tp_w - h;          r1 = 2*tp_w - lo
        s2 = tp_w - hl_range;     r2 = tp_w + hl_range
        s3 = lo - 2*(h - tp_w);   r3 = h + 2*(tp_w - lo)
        s4 = s3 - hl_range;       r4 = r3 + hl_range
        return [tp_w, s1, s2, s3, s4, r1, r2, r3, r4]

    elif method == "demark":
        if o == cl:
            tp_d = 0.25 * (h + lo + 2*cl)
        elif cl > o:
            tp_d = 0.25 * (2*h + lo + cl)
        else:
            tp_d = 0.25 * (h + 2*lo + cl)
        s1 = 2*tp_d - h
        r1 = 2*tp_d - lo
        return [tp_d, s1, r1]

    # fallback = traditional
    return _compute_pivots("traditional", o, h, lo, cl)


def _pivots_update(
    state: PIVOTSState, bar: Dict[str, Any], params: Dict[str, Any]
) -> Tuple[List[Optional[float]], PIVOTSState]:
    o  = bar["open"]
    h  = bar["high"]
    lo = bar["low"]
    cl = bar["close"]

    key = _session_key_from_bar(bar, state.anchor)

    if state.session_key is None:
        # very first bar
        state.session_key = key
        state.cur_open  = o
        state.cur_high  = h
        state.cur_low   = lo
        state.cur_close = cl
        # no pivot levels yet
        n_out = len(_pivots_output_names_for_method(state.method, state.anchor))
        return [None] * n_out, state

    if key != state.session_key and key != "":
        # session rollover: finalise previous session
        state.prev_open  = state.cur_open
        state.prev_high  = state.cur_high
        state.prev_low   = state.cur_low
        state.prev_close = state.cur_close

        # start new session accumulator
        state.cur_open  = o
        state.cur_high  = h
        state.cur_low   = lo
        state.cur_close = cl
        state.session_key = key

        # compute pivot levels from previous session
        state.levels = _compute_pivots(
            state.method,
            state.prev_open, state.prev_high, state.prev_low, state.prev_close,
        )
    else:
        # same session: update accumulator
        if state.cur_high is None or h > state.cur_high:
            state.cur_high = h
        if state.cur_low is None or lo < state.cur_low:
            state.cur_low = lo
        state.cur_close = cl

    # output current levels (or None if no rollover has happened yet)
    if state.levels is None:
        n_out = len(_pivots_output_names_for_method(state.method, state.anchor))
        return [None] * n_out, state
    return list(state.levels), state


def _pivots_output_names_for_method(method: str, anchor: str) -> List[str]:
    _props = f"PIVOTS_{method[:4].upper()}_{anchor}"
    if method == "demark":
        return [f"{_props}_P", f"{_props}_S1", f"{_props}_R1"]
    elif method == "fibonacci":
        return [
            f"{_props}_P",
            f"{_props}_S1", f"{_props}_S2", f"{_props}_S3",
            f"{_props}_R1", f"{_props}_R2", f"{_props}_R3",
        ]
    else:
        # traditional, classic, camarilla, woodie
        return [
            f"{_props}_P",
            f"{_props}_S1", f"{_props}_S2", f"{_props}_S3", f"{_props}_S4",
            f"{_props}_R1", f"{_props}_R2", f"{_props}_R3", f"{_props}_R4",
        ]


def _pivots_output_names(params: Dict[str, Any]) -> List[str]:
    method = str(_param(params, "method", "traditional"))
    anchor = str(_param(params, "anchor", "D"))
    return _pivots_output_names_for_method(method, anchor)


def _pivots_seed(series: Dict[str, Any], params: Dict[str, Any]) -> PIVOTSState:
    """internal_series seed: recover session state from OHLC tails."""
    state = _pivots_init(params)
    # Try to recover current pivot levels from the output series
    names = _pivots_output_names(params)
    # P column
    p_s = series.get(names[0])
    if p_s is not None:
        lv = p_s.dropna()
        if len(lv) > 0:
            # levels are all held constant within a session; extract last row
            state.levels = []
            for n in names:
                col = series.get(n)
                if col is not None and len(col) > 0:
                    state.levels.append(float(col.iloc[-1]))
                else:
                    state.levels.append(NAN)

    # Recover current session OHLC from raw series tails
    # (last bar of each raw series)
    for raw_key, attr in [("open", "cur_open"), ("high", "cur_high"),
                          ("low", "cur_low"), ("close", "cur_close")]:
        raw_s = series.get(raw_key)
        if raw_s is not None and len(raw_s) > 0:
            setattr(state, attr, float(raw_s.iloc[-1]))

    # session_key: try to extract from index
    # If the series have a DatetimeIndex we can get the last date
    for raw_key in ("close", "open"):
        raw_s = series.get(raw_key)
        if raw_s is not None and len(raw_s) > 0:
            idx = raw_s.index[-1]
            try:
                state.session_key = str(idx)[:10]
            except Exception:
                pass
            break

    return state


STATEFUL_REGISTRY["pivots"] = StatefulIndicator(
    kind="pivots",
    inputs=("open", "high", "low", "close"),
    init=_pivots_init,
    update=_pivots_update,
    output_names=_pivots_output_names,
)
SEED_REGISTRY["pivots"] = _pivots_seed
