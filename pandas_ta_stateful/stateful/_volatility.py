# -*- coding: utf-8 -*-
"""pandas-ta stateful – volatility indicators.

Registered kinds
----------------
atr, natr, accbands, thermo, aberration, atrts,
chandelier_exit, kc, massi, rvi, hwc
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from collections import deque
import math

from ._base import (
    NAN, _is_nan, _param, _as_int, _as_float,
    EMAState, ATRState,
    ema_make, rma_make, ema_update_raw, atr_update_raw,
    StatefulIndicator,
    STATEFUL_REGISTRY, SEED_REGISTRY,
    replay_seed,
)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _sma_buf_value(buf: deque) -> Optional[float]:
    """Return the current SMA from a full deque, else None."""
    if len(buf) < buf.maxlen:  # type: ignore[arg-type]
        return None
    return sum(buf) / len(buf)


def _ma_state_make(mamode: str, length: int) -> Any:
    """Factory: returns EMAState for ema/rma, or a fresh deque for sma."""
    mode = mamode.lower()
    if mode == "rma":
        return rma_make(length)
    elif mode == "sma":
        return deque(maxlen=length)
    else:
        # ema (default) and any other exponential variant
        return ema_make(length)


def _ma_state_update(state: Any, x: float, mamode: str) -> Tuple[Optional[float], Any]:
    """Single-step update dispatched on mamode.  Returns (value|None, state)."""
    mode = mamode.lower()
    if mode == "sma":
        state.append(x)                          # state is a deque
        return _sma_buf_value(state), state
    else:
        # EMAState (ema / rma)
        return ema_update_raw(state, x)


def _stdev_from_buf(buf: deque) -> float:
    """Sample standard deviation (ddof=1) from a full deque.  Matches pandas default."""
    n = len(buf)
    if n < 2:
        return 0.0
    mean = sum(buf) / n
    variance = sum((x - mean) ** 2 for x in buf) / (n - 1)
    return math.sqrt(variance)


# ===========================================================================
# ATR
# ===========================================================================

def _atr_init(params: Dict[str, Any]) -> ATRState:
    length = _as_int(_param(params, "length", 14), 14)
    percent = bool(_param(params, "percent", False))
    return ATRState(length=length, percent=percent)


def _atr_update(
    state: ATRState,
    bar: Dict[str, float],
    params: Dict[str, Any],
) -> Tuple[List[Optional[float]], ATRState]:
    value, state = atr_update_raw(state, bar["high"], bar["low"], bar["close"])
    return [value], state


def _atr_output_names(params: Dict[str, Any]) -> List[str]:
    length = _as_int(_param(params, "length", 14), 14)
    percent = bool(_param(params, "percent", False))
    return [f"ATRr{'p' if percent else ''}_{length}"]


def _atr_seed(series: Dict[str, "pd.Series"], params: Dict[str, Any]) -> ATRState:  # noqa: F821
    """Reconstruct ATRState by replaying over the raw input series."""
    return replay_seed("atr", series, params)


STATEFUL_REGISTRY["atr"] = StatefulIndicator(
    kind="atr",
    inputs=("high", "low", "close"),
    init=_atr_init,
    update=_atr_update,
    output_names=_atr_output_names,
)
SEED_REGISTRY["atr"] = _atr_seed


# ===========================================================================
# NATR  (Normalized ATR)
# ===========================================================================
# NATR = scalar * ATR / close.  Internally shares ATRState with ATR.
# The ATRState itself does NOT store percent; we derive NATR in the update.

def _natr_init(params: Dict[str, Any]) -> ATRState:
    length = _as_int(_param(params, "length", 14), 14)
    # percent flag on ATRState must be False; we apply scalar/close ourselves.
    return ATRState(length=length, percent=False)


def _natr_update(
    state: ATRState,
    bar: Dict[str, float],
    params: Dict[str, Any],
) -> Tuple[List[Optional[float]], ATRState]:
    scalar = _as_float(_param(params, "scalar", 100), 100.0)
    atr_val, state = atr_update_raw(state, bar["high"], bar["low"], bar["close"])
    if atr_val is None:
        return [None], state
    close = bar["close"]
    natr_val = (scalar / close) * atr_val if close != 0.0 else NAN
    return [natr_val], state


def _natr_output_names(params: Dict[str, Any]) -> List[str]:
    length = _as_int(_param(params, "length", 14), 14)
    return [f"NATR_{length}"]


def _natr_seed(series: Dict[str, "pd.Series"], params: Dict[str, Any]) -> ATRState:  # noqa: F821
    return replay_seed("natr", series, params)


STATEFUL_REGISTRY["natr"] = StatefulIndicator(
    kind="natr",
    inputs=("high", "low", "close"),
    init=_natr_init,
    update=_natr_update,
    output_names=_natr_output_names,
)
SEED_REGISTRY["natr"] = _natr_seed


# ===========================================================================
# ACCBANDS  (Acceleration Bands)
# ===========================================================================
# lower = MA( low  * (1 - hl_ratio) , length )
# mid   = MA( close                 , length )
# upper = MA( high * (1 + hl_ratio) , length )
# hl_ratio = (high - low) / (high + low) * c

@dataclass
class AccbandsState:
    length: int
    c: float
    mamode: str
    lower_ma: Any   # EMAState or deque
    mid_ma: Any
    upper_ma: Any


def _accbands_init(params: Dict[str, Any]) -> AccbandsState:
    length = _as_int(_param(params, "length", 20), 20)
    c = _as_float(_param(params, "c", 4), 4.0)
    mamode = str(_param(params, "mamode", "sma"))
    return AccbandsState(
        length=length, c=c, mamode=mamode,
        lower_ma=_ma_state_make(mamode, length),
        mid_ma=_ma_state_make(mamode, length),
        upper_ma=_ma_state_make(mamode, length),
    )


def _accbands_update(
    state: AccbandsState,
    bar: Dict[str, float],
    params: Dict[str, Any],
) -> Tuple[List[Optional[float]], AccbandsState]:
    high, low, close = bar["high"], bar["low"], bar["close"]
    denom = high + low
    hl_ratio = ((high - low) / denom * state.c) if denom != 0.0 else 0.0

    raw_lower = low * (1.0 - hl_ratio)
    raw_upper = high * (1.0 + hl_ratio)

    lower_val, state.lower_ma = _ma_state_update(state.lower_ma, raw_lower, state.mamode)
    mid_val,   state.mid_ma   = _ma_state_update(state.mid_ma,   close,     state.mamode)
    upper_val, state.upper_ma = _ma_state_update(state.upper_ma, raw_upper, state.mamode)

    return [lower_val, mid_val, upper_val], state


def _accbands_output_names(params: Dict[str, Any]) -> List[str]:
    length = _as_int(_param(params, "length", 20), 20)
    return [f"ACCBL_{length}", f"ACCBM_{length}", f"ACCBU_{length}"]


def _accbands_seed(series: Dict[str, "pd.Series"], params: Dict[str, Any]) -> AccbandsState:  # noqa: F821
    return replay_seed("accbands", series, params)


STATEFUL_REGISTRY["accbands"] = StatefulIndicator(
    kind="accbands",
    inputs=("high", "low", "close"),
    init=_accbands_init,
    update=_accbands_update,
    output_names=_accbands_output_names,
)
SEED_REGISTRY["accbands"] = _accbands_seed


# ===========================================================================
# THERMO  (Elder's Thermometer)
# ===========================================================================
# thermoL  = abs(low  - low_prev)
# thermoH  = abs(high - high_prev)
# thermo   = max(thermoL, thermoH)
# thermo_ma = MA(thermo, length)
# thermo_long  = 1 if thermo < thermo_ma * long   else 0
# thermo_short = 1 if thermo > thermo_ma * short  else 0

@dataclass
class ThermoState:
    length: int
    long: float
    short: float
    mamode: str
    asint: bool
    prev_high: Optional[float] = None
    prev_low: Optional[float] = None
    ma_state: Any = None          # EMAState or deque


def _thermo_init(params: Dict[str, Any]) -> ThermoState:
    length  = _as_int(_param(params, "length", 20), 20)
    long    = _as_float(_param(params, "long", 2), 2.0)
    short   = _as_float(_param(params, "short", 0.5), 0.5)
    mamode  = str(_param(params, "mamode", "ema"))
    asint   = bool(_param(params, "asint", True))
    return ThermoState(
        length=length, long=long, short=short,
        mamode=mamode, asint=asint,
        ma_state=_ma_state_make(mamode, length),
    )


def _thermo_update(
    state: ThermoState,
    bar: Dict[str, float],
    params: Dict[str, Any],
) -> Tuple[List[Optional[float]], ThermoState]:
    high, low = bar["high"], bar["low"]

    if state.prev_high is None or state.prev_low is None:
        # First bar: no previous -> thermo is undefined (NaN equivalent).
        # Still update prev; outputs are all None.
        state.prev_high = high
        state.prev_low  = low
        return [None, None, None, None], state

    thermoL = abs(low  - state.prev_low)
    thermoH = abs(high - state.prev_high)
    thermo  = max(thermoL, thermoH)

    state.prev_high = high
    state.prev_low  = low

    ma_val, state.ma_state = _ma_state_update(state.ma_state, thermo, state.mamode)

    if ma_val is None:
        return [thermo, None, None, None], state

    long_flag  = 1 if thermo < ma_val * state.long  else 0
    short_flag = 1 if thermo > ma_val * state.short else 0

    if not state.asint:
        long_flag  = float(long_flag)   # type: ignore[assignment]
        short_flag = float(short_flag)  # type: ignore[assignment]

    return [thermo, ma_val, long_flag, short_flag], state


def _thermo_output_names(params: Dict[str, Any]) -> List[str]:
    length = _as_int(_param(params, "length", 20), 20)
    long   = _as_float(_param(params, "long", 2), 2.0)
    short  = _as_float(_param(params, "short", 0.5), 0.5)
    _props = f"_{length}_{long}_{short}"
    return [f"THERMO{_props}", f"THERMOma{_props}",
            f"THERMOl{_props}", f"THERMOs{_props}"]


def _thermo_seed(series: Dict[str, "pd.Series"], params: Dict[str, Any]) -> ThermoState:  # noqa: F821
    return replay_seed("thermo", series, params)


STATEFUL_REGISTRY["thermo"] = StatefulIndicator(
    kind="thermo",
    inputs=("high", "low"),
    init=_thermo_init,
    update=_thermo_update,
    output_names=_thermo_output_names,
)
SEED_REGISTRY["thermo"] = _thermo_seed


# ===========================================================================
# ABERRATION
# ===========================================================================
# atr_  = ATR(high, low, close, atr_length)   -- Wilder RMA
# jg    = (high + low + close) / 3            -- HLC3
# zg    = SMA(jg, length)
# sg    = zg + atr_
# xg    = zg - atr_
# outputs: ZG (mid), SG (upper), XG (lower), ATR

@dataclass
class AberrationState:
    length: int
    atr_length: int
    atr_state: ATRState
    sma_buf: deque   # deque(maxlen=length) holding hlc3 values


def _aberration_init(params: Dict[str, Any]) -> AberrationState:
    length     = _as_int(_param(params, "length", 5), 5)
    atr_length = _as_int(_param(params, "atr_length", 15), 15)
    return AberrationState(
        length=length,
        atr_length=atr_length,
        atr_state=ATRState(length=atr_length),
        sma_buf=deque(maxlen=length),
    )


def _aberration_update(
    state: AberrationState,
    bar: Dict[str, float],
    params: Dict[str, Any],
) -> Tuple[List[Optional[float]], AberrationState]:
    high, low, close = bar["high"], bar["low"], bar["close"]

    atr_val, state.atr_state = atr_update_raw(state.atr_state, high, low, close)

    hlc3 = (high + low + close) / 3.0
    state.sma_buf.append(hlc3)
    zg = _sma_buf_value(state.sma_buf)

    # Both zg and atr must be ready for sg / xg
    if zg is None or atr_val is None:
        return [zg, None, None, atr_val], state

    sg = zg + atr_val
    xg = zg - atr_val
    return [zg, sg, xg, atr_val], state


def _aberration_output_names(params: Dict[str, Any]) -> List[str]:
    length     = _as_int(_param(params, "length", 5), 5)
    atr_length = _as_int(_param(params, "atr_length", 15), 15)
    _props = f"_{length}_{atr_length}"
    return [f"ABER_ZG{_props}", f"ABER_SG{_props}",
            f"ABER_XG{_props}", f"ABER_ATR{_props}"]


def _aberration_seed(series: Dict[str, "pd.Series"], params: Dict[str, Any]) -> AberrationState:  # noqa: F821
    return replay_seed("aberration", series, params)


STATEFUL_REGISTRY["aberration"] = StatefulIndicator(
    kind="aberration",
    inputs=("high", "low", "close"),
    init=_aberration_init,
    update=_aberration_update,
    output_names=_aberration_output_names,
)
SEED_REGISTRY["aberration"] = _aberration_seed


# ===========================================================================
# ATRTS  (ATR Trailing Stop)
# ===========================================================================
# atr_  = ATR(h, l, c, length)  * k
# ma_   = MA(close, ma_length)
# direction: close > ma_ -> up (True), else dn (False)
# Trailing stop (streaming):
#   if up  : trail = max(prev_trail,  close - atr_)
#   if dn  : trail = min(prev_trail,  close + atr_)
# On direction flip the trail resets to the new-side raw value.

@dataclass
class AtrtsState:
    length: int
    ma_length: int
    k: float
    mamode: str
    atr_state: ATRState
    ma_state: Any                      # EMAState or deque for close MA
    prev_trail: Optional[float] = None
    prev_up: Optional[bool] = None     # last known direction; None = no history


def _atrts_init(params: Dict[str, Any]) -> AtrtsState:
    length    = _as_int(_param(params, "length", 14), 14)
    ma_length = _as_int(_param(params, "ma_length", 20), 20)
    k         = _as_float(_param(params, "k", 3.0), 3.0)
    mamode    = str(_param(params, "mamode", "ema"))
    return AtrtsState(
        length=length, ma_length=ma_length, k=k, mamode=mamode,
        atr_state=ATRState(length=length),
        ma_state=_ma_state_make(mamode, ma_length),
    )


def _atrts_update(
    state: AtrtsState,
    bar: Dict[str, float],
    params: Dict[str, Any],
) -> Tuple[List[Optional[float]], AtrtsState]:
    high, low, close = bar["high"], bar["low"], bar["close"]

    atr_val, state.atr_state = atr_update_raw(state.atr_state, high, low, close)
    ma_val,  state.ma_state  = _ma_state_update(state.ma_state, close, state.mamode)

    if atr_val is None or ma_val is None:
        return [None], state

    atr_k = atr_val * state.k
    is_up = close > ma_val          # current direction

    if state.prev_trail is None or state.prev_up is None:
        # First valid bar: initialise trail
        trail = (close - atr_k) if is_up else (close + atr_k)
    elif is_up != state.prev_up:
        # Direction flip: reset trail to the raw value for the new side
        trail = (close - atr_k) if is_up else (close + atr_k)
    else:
        # Same direction: clamp
        if is_up:
            trail = max(state.prev_trail, close - atr_k)
        else:
            trail = min(state.prev_trail, close + atr_k)

    state.prev_trail = trail
    state.prev_up    = is_up
    return [trail], state


def _atrts_output_names(params: Dict[str, Any]) -> List[str]:
    length    = _as_int(_param(params, "length", 14), 14)
    ma_length = _as_int(_param(params, "ma_length", 20), 20)
    k         = _as_float(_param(params, "k", 3.0), 3.0)
    mamode    = str(_param(params, "mamode", "ema"))
    _props = f"ATRTS{mamode[0]}_{length}_{ma_length}_{k}"
    return [_props]


def _atrts_seed(series: Dict[str, "pd.Series"], params: Dict[str, Any]) -> AtrtsState:  # noqa: F821
    return replay_seed("atrts", series, params)


STATEFUL_REGISTRY["atrts"] = StatefulIndicator(
    kind="atrts",
    inputs=("high", "low", "close"),
    init=_atrts_init,
    update=_atrts_update,
    output_names=_atrts_output_names,
)
SEED_REGISTRY["atrts"] = _atrts_seed


# ===========================================================================
# CHANDELIER EXIT
# ===========================================================================
# atr_mult = ATR(h, l, c, atr_length) * multiplier
# When use_close=False (default):
#   long  = max(high, high_length)  - atr_mult      <- "long stop" (support)
#   short = min(low,  low_length)   + atr_mult      <- "short stop" (resistance)
# When use_close=True:
#   long  = max(close, roll_length) - atr_mult
#   short = min(close, roll_length) + atr_mult
# direction (per-bar, ffill logic):
#   close > prev_long  -> +1
#   close < prev_short -> -1
#   otherwise          -> repeat prev_direction

@dataclass
class ChandelierState:
    high_length: int
    low_length: int
    atr_length: int
    multiplier: float
    use_close: bool
    atr_state: ATRState
    high_buf: deque          # rolling window for high (or close)
    low_buf: deque           # rolling window for low  (or close)
    prev_long: Optional[float] = None
    prev_short: Optional[float] = None
    direction: int = -1      # initial direction = -1 (falling) per spec


def _chandelier_init(params: Dict[str, Any]) -> ChandelierState:
    high_length = _as_int(_param(params, "high_length", 22), 22)
    low_length  = _as_int(_param(params, "low_length", 22), 22)
    atr_length  = _as_int(_param(params, "atr_length", 14), 14)
    multiplier  = _as_float(_param(params, "multiplier", 2.0), 2.0)
    use_close   = bool(_param(params, "use_close", False))
    roll_length = max(high_length, low_length)
    return ChandelierState(
        high_length=high_length,
        low_length=low_length,
        atr_length=atr_length,
        multiplier=multiplier,
        use_close=use_close,
        atr_state=ATRState(length=atr_length),
        high_buf=deque(maxlen=high_length if not use_close else roll_length),
        low_buf=deque(maxlen=low_length  if not use_close else roll_length),
    )


def _chandelier_update(
    state: ChandelierState,
    bar: Dict[str, float],
    params: Dict[str, Any],
) -> Tuple[List[Optional[float]], ChandelierState]:
    high, low, close = bar["high"], bar["low"], bar["close"]

    atr_val, state.atr_state = atr_update_raw(state.atr_state, high, low, close)

    # Buffer feed
    if state.use_close:
        state.high_buf.append(close)
        state.low_buf.append(close)
    else:
        state.high_buf.append(high)
        state.low_buf.append(low)

    if atr_val is None:
        return [None, None, float(state.direction)], state

    atr_mult = atr_val * state.multiplier

    highest = max(state.high_buf)
    lowest  = min(state.low_buf)

    long_stop  = highest - atr_mult
    short_stop = lowest  + atr_mult

    # Direction logic (vectorized source uses shift(drift=1) comparison,
    # meaning it compares current close to *previous* long/short).
    if state.prev_long is not None and state.prev_short is not None:
        if close > state.prev_long:
            state.direction = 1
        elif close < state.prev_short:
            state.direction = -1
        # else: keep (ffill)

    state.prev_long  = long_stop
    state.prev_short = short_stop

    return [long_stop, short_stop, float(state.direction)], state


def _chandelier_output_names(params: Dict[str, Any]) -> List[str]:
    high_length = _as_int(_param(params, "high_length", 22), 22)
    low_length  = _as_int(_param(params, "low_length", 22), 22)
    atr_length  = _as_int(_param(params, "atr_length", 14), 14)
    multiplier  = _as_float(_param(params, "multiplier", 2.0), 2.0)
    use_close   = bool(_param(params, "use_close", False))
    _name = "CHDLREXT"
    _props = f"_{high_length}_{low_length}_{atr_length}_{multiplier}"
    if use_close:
        _props = f"_CLOSE{_props}"
    return [f"{_name}l{_props}", f"{_name}s{_props}", f"{_name}d{_props}"]


def _chandelier_seed(series: Dict[str, "pd.Series"], params: Dict[str, Any]) -> ChandelierState:  # noqa: F821
    return replay_seed("chandelier_exit", series, params)


STATEFUL_REGISTRY["chandelier_exit"] = StatefulIndicator(
    kind="chandelier_exit",
    inputs=("high", "low", "close"),
    init=_chandelier_init,
    update=_chandelier_update,
    output_names=_chandelier_output_names,
)
SEED_REGISTRY["chandelier_exit"] = _chandelier_seed


# ===========================================================================
# KC  (Keltner Channel)
# ===========================================================================
# range_  = TR(h,l,c)  if tr else (high - low)
# basis   = MA(close,  length)
# band    = MA(range_, length)
# lower   = basis - scalar * band
# upper   = basis + scalar * band
# TR needs prev_close (same logic as atr_update_raw True Range sub-step).

@dataclass
class KCState:
    length: int
    scalar: float
    use_tr: bool          # "tr" param in vectorized source
    mamode: str
    basis_ma: Any         # EMAState or deque
    band_ma: Any          # EMAState or deque
    prev_close: Optional[float] = None


def _kc_init(params: Dict[str, Any]) -> KCState:
    length = _as_int(_param(params, "length", 20), 20)
    scalar = _as_float(_param(params, "scalar", 2), 2.0)
    use_tr = bool(_param(params, "tr", True))
    mamode = str(_param(params, "mamode", "ema"))
    return KCState(
        length=length, scalar=scalar, use_tr=use_tr, mamode=mamode,
        basis_ma=_ma_state_make(mamode, length),
        band_ma=_ma_state_make(mamode, length),
    )


def _kc_update(
    state: KCState,
    bar: Dict[str, float],
    params: Dict[str, Any],
) -> Tuple[List[Optional[float]], KCState]:
    high, low, close = bar["high"], bar["low"], bar["close"]

    # Compute range
    if state.use_tr:
        if state.prev_close is None:
            rng = high - low
        else:
            rng = max(high - low,
                      abs(high - state.prev_close),
                      abs(low  - state.prev_close))
    else:
        rng = high - low

    state.prev_close = close

    basis_val, state.basis_ma = _ma_state_update(state.basis_ma, close, state.mamode)
    band_val,  state.band_ma  = _ma_state_update(state.band_ma,  rng,   state.mamode)

    if basis_val is None or band_val is None:
        return [None, basis_val, None], state

    lower = basis_val - state.scalar * band_val
    upper = basis_val + state.scalar * band_val
    return [lower, basis_val, upper], state


def _kc_output_names(params: Dict[str, Any]) -> List[str]:
    length = _as_int(_param(params, "length", 20), 20)
    scalar = _as_float(_param(params, "scalar", 2), 2.0)
    mamode = str(_param(params, "mamode", "ema"))
    _m = mamode.lower()[0] if mamode else ""
    _props = f"{_m}_{length}_{scalar}"
    return [f"KCL{_props}", f"KCB{_props}", f"KCU{_props}"]


def _kc_seed(series: Dict[str, "pd.Series"], params: Dict[str, Any]) -> KCState:  # noqa: F821
    return replay_seed("kc", series, params)


STATEFUL_REGISTRY["kc"] = StatefulIndicator(
    kind="kc",
    inputs=("high", "low", "close"),
    init=_kc_init,
    update=_kc_update,
    output_names=_kc_output_names,
)
SEED_REGISTRY["kc"] = _kc_seed


# ===========================================================================
# MASSI  (Mass Index)
# ===========================================================================
# hl      = high - low   (non_zero_range -> max(high-low, 0) effectively; hl >= 0)
# ema1    = EMA(hl,  fast)
# ema2    = EMA(ema1, fast)   -- second EMA on ema1, same fast period
# ratio   = ema1 / ema2
# massi   = rolling_sum(ratio, slow)

@dataclass
class MassiState:
    fast: int
    slow: int
    ema1_state: EMAState
    ema2_state: EMAState
    ratio_buf: deque          # deque(maxlen=slow) of ratio values


def _massi_init(params: Dict[str, Any]) -> MassiState:
    fast = _as_int(_param(params, "fast", 9), 9)
    slow = _as_int(_param(params, "slow", 25), 25)
    # Enforce fast <= slow (mirrors vectorized swap)
    if slow < fast:
        fast, slow = slow, fast
    return MassiState(
        fast=fast, slow=slow,
        ema1_state=ema_make(fast),
        ema2_state=ema_make(fast),
        ratio_buf=deque(maxlen=slow),
    )


def _massi_update(
    state: MassiState,
    bar: Dict[str, float],
    params: Dict[str, Any],
) -> Tuple[List[Optional[float]], MassiState]:
    high, low = bar["high"], bar["low"]
    hl = max(high - low, 0.0)          # non_zero_range: ensures >= 0

    ema1_val, state.ema1_state = ema_update_raw(state.ema1_state, hl)

    if ema1_val is None:
        # ema2 cannot be fed yet
        return [None], state

    ema2_val, state.ema2_state = ema_update_raw(state.ema2_state, ema1_val)

    if ema2_val is None:
        return [None], state

    ratio = ema1_val / ema2_val if ema2_val != 0.0 else NAN
    state.ratio_buf.append(ratio)

    if len(state.ratio_buf) < state.ratio_buf.maxlen:  # type: ignore[arg-type]
        return [None], state

    massi_val = sum(state.ratio_buf)
    return [massi_val], state


def _massi_output_names(params: Dict[str, Any]) -> List[str]:
    fast = _as_int(_param(params, "fast", 9), 9)
    slow = _as_int(_param(params, "slow", 25), 25)
    if slow < fast:
        fast, slow = slow, fast
    return [f"MASSI_{fast}_{slow}"]


def _massi_seed(series: Dict[str, "pd.Series"], params: Dict[str, Any]) -> MassiState:  # noqa: F821
    return replay_seed("massi", series, params)


STATEFUL_REGISTRY["massi"] = StatefulIndicator(
    kind="massi",
    inputs=("high", "low"),
    init=_massi_init,
    update=_massi_update,
    output_names=_massi_output_names,
)
SEED_REGISTRY["massi"] = _massi_seed


# ===========================================================================
# RVI  (Relative Volatility Index)
# ===========================================================================
# Stateful implementation covers the default single-source (close) mode only
# (refined=False, thirds=False).
#
# std     = stdev(close, length)       -- sample stdev over rolling window
# pos     = max(close - close_prev, 0)
# neg     = abs(min(close - close_prev, 0))
# pos_std = pos * std
# neg_std = neg * std
# pos_avg = MA(pos_std, length)
# neg_avg = MA(neg_std, length)
# rvi     = scalar * pos_avg / (pos_avg + neg_avg + 1e-10)

@dataclass
class RviState:
    length: int
    scalar: float
    mamode: str
    close_buf: deque          # deque(maxlen=length) of close values -> stdev
    prev_close: Optional[float] = None
    pos_ma: Any = None        # EMAState or deque for pos_std MA
    neg_ma: Any = None        # EMAState or deque for neg_std MA


def _rvi_init(params: Dict[str, Any]) -> RviState:
    length = _as_int(_param(params, "length", 14), 14)
    scalar = _as_float(_param(params, "scalar", 100), 100.0)
    mamode = str(_param(params, "mamode", "ema"))
    return RviState(
        length=length, scalar=scalar, mamode=mamode,
        close_buf=deque(maxlen=length),
        pos_ma=_ma_state_make(mamode, length),
        neg_ma=_ma_state_make(mamode, length),
    )


def _rvi_update(
    state: RviState,
    bar: Dict[str, float],
    params: Dict[str, Any],
) -> Tuple[List[Optional[float]], RviState]:
    close = bar["close"]
    state.close_buf.append(close)

    if state.prev_close is None:
        # First bar: no diff available
        state.prev_close = close
        return [None], state

    diff = close - state.prev_close
    state.prev_close = close

    pos = max(diff, 0.0)
    neg = abs(min(diff, 0.0))

    # stdev needs a full window
    if len(state.close_buf) < state.close_buf.maxlen:  # type: ignore[arg-type]
        # Not enough data for stdev; feed 0 * pos/neg into MAs to keep them
        # in lockstep, but output None.
        _, state.pos_ma = _ma_state_update(state.pos_ma, 0.0, state.mamode)
        _, state.neg_ma = _ma_state_update(state.neg_ma, 0.0, state.mamode)
        return [None], state

    std = _stdev_from_buf(state.close_buf)

    pos_std = pos * std
    neg_std = neg * std

    pos_avg, state.pos_ma = _ma_state_update(state.pos_ma, pos_std, state.mamode)
    neg_avg, state.neg_ma = _ma_state_update(state.neg_ma, neg_std, state.mamode)

    if pos_avg is None or neg_avg is None:
        return [None], state

    rvi_val = state.scalar * pos_avg / (pos_avg + neg_avg + 1e-10)
    return [rvi_val], state


def _rvi_output_names(params: Dict[str, Any]) -> List[str]:
    length = _as_int(_param(params, "length", 14), 14)
    return [f"RVI_{length}"]


def _rvi_seed(series: Dict[str, "pd.Series"], params: Dict[str, Any]) -> RviState:  # noqa: F821
    return replay_seed("rvi", series, params)


STATEFUL_REGISTRY["rvi"] = StatefulIndicator(
    kind="rvi",
    inputs=("close",),
    init=_rvi_init,
    update=_rvi_update,
    output_names=_rvi_output_names,
)
SEED_REGISTRY["rvi"] = _rvi_seed


# ===========================================================================
# HWC  (Holt-Winters Channel)  –  replay_only
# ===========================================================================
# Iterative triple-exponential (Holt-Winters) with variance channel.
# State is fully sequential; no closed-form seed is possible.
# -> STATEFUL_REGISTRY only.  SEED_REGISTRY is NOT populated.
#
# Per hwc.py source:
#   F = (1 - na)*(last_f + last_v + 0.5*last_a) + na*close
#   V = (1 - nb)*(last_v + last_a)              + nb*(F - last_f)
#   A = (1 - nc)*last_a                         + nc*(V - last_v)
#   result = F + V + 0.5*A
#   var    = (1 - nd)*last_var + nd*(last_price - last_result)^2
#   stddev = sqrt(last_var)                     -- NOTE: uses *last_var* (previous)
#   upper  = result + scalar*stddev
#   lower  = result - scalar*stddev
#   (channels mode also outputs width and pct_width)

@dataclass
class HWCState:
    scalar: float
    channels: bool
    na: float
    nb: float
    nc: float
    nd: float
    last_a: float = 0.0
    last_v: float = 0.0
    last_f: float = 0.0
    last_var: float = 0.0
    last_price: Optional[float] = None
    last_result: Optional[float] = None
    _initialised: bool = False


def _hwc_init(params: Dict[str, Any]) -> HWCState:
    scalar   = _as_float(_param(params, "scalar", 1), 1.0)
    channels = bool(_param(params, "channels", True))
    na       = _as_float(_param(params, "na", 0.2), 0.2)
    nb       = _as_float(_param(params, "nb", 0.1), 0.1)
    nc       = _as_float(_param(params, "nc", 0.1), 0.1)
    nd       = _as_float(_param(params, "nd", 0.1), 0.1)
    return HWCState(
        scalar=scalar, channels=channels,
        na=na, nb=nb, nc=nc, nd=nd,
    )


def _hwc_update(
    state: HWCState,
    bar: Dict[str, float],
    params: Dict[str, Any],
) -> Tuple[List[Optional[float]], HWCState]:
    close = bar["close"]

    if not state._initialised:
        # First bar: seed all accumulators to close (mirrors hwc.py lines 57-58)
        state.last_f       = close
        state.last_price   = close
        state.last_result  = close
        state.last_a       = 0.0
        state.last_v       = 0.0
        state.last_var     = 0.0
        state._initialised = True

    # Compute F, V, A (using last_* from *previous* iteration)
    F = (1.0 - state.na) * (state.last_f + state.last_v + 0.5 * state.last_a) \
        + state.na * close
    V = (1.0 - state.nb) * (state.last_v + state.last_a) \
        + state.nb * (F - state.last_f)
    A = (1.0 - state.nc) * state.last_a \
        + state.nc * (V - state.last_v)

    result = F + V + 0.5 * A

    # Variance uses *last_var* and *last_price / last_result* (previous bar values)
    var    = (1.0 - state.nd) * state.last_var \
             + state.nd * (state.last_price - state.last_result) ** 2
    stddev = math.sqrt(state.last_var)          # sqrt of PREVIOUS var (matches source)

    upper = result + state.scalar * stddev
    lower = result - state.scalar * stddev

    # Advance state
    state.last_a       = A
    state.last_f       = F
    state.last_v       = V
    state.last_var     = var
    state.last_price   = close
    state.last_result  = result

    if state.channels:
        width     = upper - lower
        pct_width = (close - lower) / (width + 2.220446049250313e-16)  # float epsilon
        return [result, lower, upper, width, pct_width], state
    else:
        return [result, lower, upper], state


def _hwc_output_names(params: Dict[str, Any]) -> List[str]:
    scalar   = _as_float(_param(params, "scalar", 1), 1.0)
    channels = bool(_param(params, "channels", True))
    _props = f"_{scalar}"
    names = [f"HWM{_props}", f"HWL{_props}", f"HWU{_props}"]
    if channels:
        names.extend([f"HWW{_props}", f"HWPCT{_props}"])
    return names


# replay_only: STATEFUL_REGISTRY only, no SEED_REGISTRY entry.
STATEFUL_REGISTRY["hwc"] = StatefulIndicator(
    kind="hwc",
    inputs=("close",),
    init=_hwc_init,
    update=_hwc_update,
    output_names=_hwc_output_names,
)
