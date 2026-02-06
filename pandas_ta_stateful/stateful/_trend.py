# -*- coding: utf-8 -*-
"""pandas-ta stateful -- trend indicators.

Registered kinds
----------------
decay, adx, alphatrend, amat, cksp, ht_trendline, qstick, rwi, trendflex, psar, zigzag
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from collections import deque
import math
import numpy as np

from ._base import (
    NAN, _is_nan, _param, _as_int, _as_float,
    EMAState, ATRState,
    ema_make, rma_make, ema_update_raw, atr_update_raw,
    StatefulIndicator,
    STATEFUL_REGISTRY, SEED_REGISTRY,
    replay_seed,
)
from pandas_ta_stateful.maps import Imports

# ---------------------------------------------------------------------------
# Internal helpers (from _volatility.py pattern)
# ---------------------------------------------------------------------------

def _sma_buf_value(buf: deque) -> Optional[float]:
    """Return the current SMA from a full deque, else None."""
    if len(buf) < buf.maxlen:  # type: ignore[arg-type]
        return None
    return sum(buf) / len(buf)


def _fmt_num(val: Any) -> Any:
    if isinstance(val, float) and float(val).is_integer():
        return int(val)
    return val


def _ma_state_make(mamode: str, length: int) -> Any:
    """Factory: returns EMAState for ema/rma, or a fresh deque for sma."""
    mode = mamode.lower()
    if mode == "rma":
        return rma_make(length, presma=True)
    elif mode == "sma":
        return deque(maxlen=length)
    else:
        # ema (default) and any other exponential variant
        return ema_make(length, presma=True)


def _ma_state_update(state: Any, x: float, mamode: str) -> Tuple[Optional[float], Any]:
    """Single-step update dispatched on mamode.  Returns (value|None, state)."""
    mode = mamode.lower()
    if mode == "sma":
        state.append(x)                          # state is a deque
        return _sma_buf_value(state), state
    else:
        # EMAState (ema / rma)
        return ema_update_raw(state, x)


# ===========================================================================
# DECAY  (output_only)
# ===========================================================================
# State: just the previous decay value
# Modes: "linear" (default) or "exp" (exponential)

@dataclass
class DecayState:
    length: int
    mode: str
    prev_decay: float = 0.0


def _decay_init(params: Dict[str, Any]) -> DecayState:
    length = _as_int(_param(params, "length", 1), 1)
    mode = str(_param(params, "mode", "linear")).lower()
    if mode not in ["linear", "exp", "exponential"]:
        mode = "linear"
    return DecayState(length=length, mode=mode)


def _decay_update(
    state: DecayState, bar: Dict[str, Any], params: Dict[str, Any]
) -> Tuple[List[Optional[float]], DecayState]:
    x = bar["close"]

    if state.mode in ["exp", "exponential"]:
        # Exponential decay: rate = 1.0 - (1.0 / n)
        rate = 1.0 - (1.0 / state.length)
        state.prev_decay = max(0.0, x, state.prev_decay * rate)
    else:
        # Linear decay: rate = 1.0 / n
        rate = 1.0 / state.length
        state.prev_decay = max(0.0, x, state.prev_decay - rate)

    return [state.prev_decay], state


def _decay_output_names(params: Dict[str, Any]) -> List[str]:
    length = _as_int(_param(params, "length", 1), 1)
    mode = str(_param(params, "mode", "linear")).lower()
    if mode not in ["linear", "exp", "exponential"]:
        mode = "linear"
    prefix = "EXPDECAY" if mode in ["exp", "exponential"] else "LDECAY"
    return [f"{prefix}_{length}"]


def _decay_seed(series: Dict[str, Any], params: Dict[str, Any]) -> DecayState:
    """Reconstruct DecayState from the last output value."""
    length = _as_int(_param(params, "length", 1), 1)
    mode = str(_param(params, "mode", "linear")).lower()
    if mode not in ["linear", "exp", "exponential"]:
        mode = "linear"
    state = DecayState(length=length, mode=mode)

    col = _decay_output_names(params)[0]
    s = series.get(col)
    if s is not None:
        last_valid = s.dropna()
        if len(last_valid) > 0:
            state.prev_decay = float(last_valid.iloc[-1])

    return state


STATEFUL_REGISTRY["decay"] = StatefulIndicator(
    kind="decay",
    inputs=("close",),
    init=_decay_init,
    update=_decay_update,
    output_names=_decay_output_names,
)
SEED_REGISTRY["decay"] = _decay_seed


# ===========================================================================
# ADX  (internal_series)
# ===========================================================================
# State: atr_state, pos_ma_state, neg_ma_state, dx_ma_state
# Default: length=14, signal_length=14, mamode="rma", scalar=100

@dataclass
class ADXState:
    length: int
    signal_length: int
    adxr_length: int
    scalar: float
    mamode: str
    use_talib: bool
    atr_state: ATRState
    pos_ma_state: Any       # EMAState or deque
    neg_ma_state: Any
    dx_ma_state: Any
    adx_buf: deque
    prev_high: Optional[float] = None
    prev_low: Optional[float] = None


def _adx_init(params: Dict[str, Any]) -> ADXState:
    length = _as_int(_param(params, "length", 14), 14)
    signal_length = _as_int(_param(params, "signal_length", length), length)
    adxr_length = _as_int(_param(params, "adxr_length", 2), 2)
    scalar = _as_float(_param(params, "scalar", 100), 100.0)
    mamode = str(_param(params, "mamode", "rma"))
    use_talib = bool(_param(params, "talib", True)) and Imports.get("talib", False)
    if use_talib:
        mamode = "rma"

    return ADXState(
        length=length,
        signal_length=signal_length,
        adxr_length=adxr_length,
        scalar=scalar,
        mamode=mamode,
        use_talib=use_talib,
        atr_state=ATRState(length=length, percent=False),
        pos_ma_state=_ma_state_make(mamode, length),
        neg_ma_state=_ma_state_make(mamode, length),
        dx_ma_state=_ma_state_make(mamode, signal_length),
        adx_buf=deque(maxlen=adxr_length + 1),
    )


def _adx_update(
    state: ADXState, bar: Dict[str, Any], params: Dict[str, Any]
) -> Tuple[List[Optional[float]], ADXState]:
    high, low, close = bar["high"], bar["low"], bar["close"]

    # ATR calculation
    atr_val, state.atr_state = atr_update_raw(state.atr_state, high, low, close)

    if atr_val is None or atr_val == 0.0:
        return [None, None, None], state

    k = state.scalar / atr_val

    # Directional movement
    if state.prev_high is not None:
        up = high - state.prev_high
        dn = state.prev_low - low

        pos = up if (up > dn and up > 0) else 0.0
        neg = dn if (dn > up and dn > 0) else 0.0
    else:
        pos = neg = 0.0

    state.prev_high = high
    state.prev_low = low

    # Smooth directional movements
    dmp_val, state.pos_ma_state = _ma_state_update(state.pos_ma_state, pos, state.mamode)
    dmn_val, state.neg_ma_state = _ma_state_update(state.neg_ma_state, neg, state.mamode)

    if dmp_val is None or dmn_val is None:
        return [None, None, None, None], state

    if state.use_talib and state.mamode.lower() == "rma":
        # TA-Lib PLUS_DM/MINUS_DM uses Wilder-smoothed SUM. Multiply by length.
        dmp_val = dmp_val * state.length
        dmn_val = dmn_val * state.length

    dmp = k * dmp_val
    dmn = k * dmn_val

    # DX calculation
    denom = dmp + dmn
    dx = state.scalar * abs(dmp - dmn) / denom if denom != 0.0 else 0.0

    # ADX (smoothed DX)
    adx_val, state.dx_ma_state = _ma_state_update(state.dx_ma_state, dx, state.mamode)
    adxr_val: Optional[float] = None
    if adx_val is not None:
        state.adx_buf.append(adx_val)
        if len(state.adx_buf) >= state.adxr_length + 1:
            adxr_val = 0.5 * (adx_val + state.adx_buf[0])

    return [adx_val, adxr_val, dmp, dmn], state


def _adx_output_names(params: Dict[str, Any]) -> List[str]:
    length = _as_int(_param(params, "length", 14), 14)
    signal_length = _as_int(_param(params, "signal_length", length), length)
    adxr_length = _as_int(_param(params, "adxr_length", 2), 2)
    return [
        f"ADX_{signal_length}",
        f"ADXR_{signal_length}_{adxr_length}",
        f"DMP_{length}",
        f"DMN_{length}",
    ]


def _adx_seed(series: Dict[str, Any], params: Dict[str, Any]) -> ADXState:
    """Reconstruct ADXState by replaying over the raw input series."""
    return replay_seed("adx", series, params)


STATEFUL_REGISTRY["adx"] = StatefulIndicator(
    kind="adx",
    inputs=("high", "low", "close"),
    init=_adx_init,
    update=_adx_update,
    output_names=_adx_output_names,
)
SEED_REGISTRY["adx"] = _adx_seed


# ===========================================================================
# ALPHATREND  (internal_series)
# ===========================================================================
# Vectorized reference uses:
#   ATR(high, low, close, length, mamode)
#   MFI if volume provided, else RSI(src, length, mamode)
#   lag = 2
# Outputs: ALPHAT, ALPHATl

@dataclass
class AlphatrendState:
    length: int
    multiplier: float
    threshold: float
    lag: int
    mamode: str
    src: str
    tr_ma_state: Any
    prev_close: Optional[float] = None
    # RSI components (used when no volume)
    prev_src: Optional[float] = None
    gain_state: Any = None
    loss_state: Any = None
    # MFI components (used when volume is provided)
    prev_tp: Optional[float] = None
    pos_buf: deque = field(default_factory=deque)
    neg_buf: deque = field(default_factory=deque)
    pos_sum: float = 0.0
    neg_sum: float = 0.0
    # Alphatrend history
    prev_at: Optional[float] = None
    at_hist: deque = field(default_factory=deque)


def _alphatrend_init(params: Dict[str, Any]) -> AlphatrendState:
    length = _as_int(_param(params, "length", 14), 14)
    multiplier = _as_float(_param(params, "multiplier", 1), 1.0)
    threshold = _as_float(_param(params, "threshold", 50), 50.0)
    lag = _as_int(_param(params, "lag", 2), 2)
    mamode = str(_param(params, "mamode", "sma"))
    src = str(_param(params, "src", "close")).lower()

    state = AlphatrendState(
        length=length,
        multiplier=multiplier,
        threshold=threshold,
        lag=lag,
        mamode=mamode,
        src=src,
        tr_ma_state=_ma_state_make(mamode, length),
        gain_state=_ma_state_make(mamode, length),
        loss_state=_ma_state_make(mamode, length),
        pos_buf=deque(maxlen=length),
        neg_buf=deque(maxlen=length),
        at_hist=deque(maxlen=lag + 1),
    )
    return state


def _alphatrend_update(
    state: AlphatrendState, bar: Dict[str, Any], params: Dict[str, Any]
) -> Tuple[List[Optional[float]], AlphatrendState]:
    open_ = bar["open"]
    high, low, close = bar["high"], bar["low"], bar["close"]
    volume = bar.get("volume")

    # ATR calculation (TR -> MA)
    if state.prev_close is None:
        tr = high - low
    else:
        tr = max(high - low, abs(high - state.prev_close), abs(low - state.prev_close))
    state.prev_close = close
    atr_val, state.tr_ma_state = _ma_state_update(state.tr_ma_state, tr, state.mamode)

    # Momentum (MFI if volume present, else RSI on src)
    momo_val: Optional[float] = None
    if volume is not None:
        tp = (high + low + close) / 3.0
        if state.prev_tp is not None:
            mf = tp * volume
            pos_flow = mf if tp > state.prev_tp else 0.0
            neg_flow = mf if tp < state.prev_tp else 0.0
            if len(state.pos_buf) == state.pos_buf.maxlen:  # type: ignore[arg-type]
                state.pos_sum -= state.pos_buf[0]
            if len(state.neg_buf) == state.neg_buf.maxlen:  # type: ignore[arg-type]
                state.neg_sum -= state.neg_buf[0]
            state.pos_buf.append(pos_flow)
            state.neg_buf.append(neg_flow)
            state.pos_sum += pos_flow
            state.neg_sum += neg_flow
            if len(state.pos_buf) >= state.length:
                denom = state.pos_sum + state.neg_sum
                momo_val = 100.0 * state.pos_sum / denom if denom != 0.0 else 50.0
        state.prev_tp = tp
    else:
        src_val = close
        if state.src == "open":
            src_val = open_
        elif state.src == "high":
            src_val = high
        elif state.src == "low":
            src_val = low
        if state.prev_src is not None:
            diff = src_val - state.prev_src
            gain = diff if diff > 0 else 0.0
            loss = -diff if diff < 0 else 0.0
            avg_gain, state.gain_state = _ma_state_update(state.gain_state, gain, state.mamode)
            avg_loss, state.loss_state = _ma_state_update(state.loss_state, loss, state.mamode)
            if avg_gain is not None and avg_loss is not None:
                denom = avg_gain + avg_loss
                momo_val = 100.0 * avg_gain / denom if denom != 0.0 else 50.0
        state.prev_src = src_val

    if atr_val is None or momo_val is None:
        # Maintain lag alignment even during warmup
        state.at_hist.append(None)
        atl = state.at_hist[0] if len(state.at_hist) >= state.lag + 1 else None
        return [None, atl], state

    lower_atr = low - atr_val * state.multiplier
    upper_atr = high + atr_val * state.multiplier

    momo_threshold = momo_val >= state.threshold

    at_val: Optional[float]
    if state.prev_at is None:
        # First valid value -> set prev, but output NaN (matches vectorized)
        state.prev_at = lower_atr if momo_threshold else upper_atr
        at_val = None
    else:
        if momo_threshold:
            at_val = state.prev_at if lower_atr < state.prev_at else lower_atr
        else:
            at_val = state.prev_at if upper_atr > state.prev_at else upper_atr
        state.prev_at = at_val

    state.at_hist.append(at_val)
    atl = state.at_hist[0] if len(state.at_hist) >= state.lag + 1 else None
    return [at_val, atl], state


def _alphatrend_output_names(params: Dict[str, Any]) -> List[str]:
    length = _as_int(_param(params, "length", 14), 14)
    multiplier = _fmt_num(_as_float(_param(params, "multiplier", 1), 1.0))
    threshold = _fmt_num(_as_float(_param(params, "threshold", 50), 50.0))
    lag = _as_int(_param(params, "lag", 2), 2)
    props = f"_{length}_{multiplier}_{threshold}"
    return [f"ALPHAT{props}", f"ALPHATl{props}_{lag}"]


def _alphatrend_seed(series: Dict[str, Any], params: Dict[str, Any]) -> AlphatrendState:
    """Reconstruct AlphatrendState by replaying over the raw input series."""
    return replay_seed("alphatrend", series, params)


STATEFUL_REGISTRY["alphatrend"] = StatefulIndicator(
    kind="alphatrend",
    inputs=("open", "high", "low", "close", "volume"),
    init=_alphatrend_init,
    update=_alphatrend_update,
    output_names=_alphatrend_output_names,
)
SEED_REGISTRY["alphatrend"] = _alphatrend_seed


# ===========================================================================
# AMAT  (internal_series)
# ===========================================================================
# State: fast_ma_state, slow_ma_state
# Default: fast=8, slow=21, lookback=2, mamode="ema"

@dataclass
class AmatState:
    fast: int
    slow: int
    lookback: int
    mamode: str
    fast_ma_state: Any      # EMAState or deque
    slow_ma_state: Any
    fast_history: deque = field(default_factory=lambda: deque(maxlen=3))
    slow_history: deque = field(default_factory=lambda: deque(maxlen=3))


def _amat_init(params: Dict[str, Any]) -> AmatState:
    fast = _as_int(_param(params, "fast", 8), 8)
    slow = _as_int(_param(params, "slow", 21), 21)
    lookback = _as_int(_param(params, "lookback", 2), 2)
    mamode = str(_param(params, "mamode", "ema"))

    return AmatState(
        fast=fast,
        slow=slow,
        lookback=lookback,
        mamode=mamode,
        fast_ma_state=_ma_state_make(mamode, fast),
        slow_ma_state=_ma_state_make(mamode, slow),
    )


def _amat_update(
    state: AmatState, bar: Dict[str, Any], params: Dict[str, Any]
) -> Tuple[List[Optional[float]], AmatState]:
    close = bar["close"]

    fast_val, state.fast_ma_state = _ma_state_update(state.fast_ma_state, close, state.mamode)
    slow_val, state.slow_ma_state = _ma_state_update(state.slow_ma_state, close, state.mamode)

    if fast_val is None or slow_val is None:
        return [None, None], state

    state.fast_history.append(fast_val)
    state.slow_history.append(slow_val)

    # Need lookback+1 values to compute trends
    if len(state.fast_history) < state.lookback + 1:
        return [None, None], state

    # Long run: fast increasing AND (slow decreasing OR slow increasing)
    fast_inc = all(state.fast_history[i] < state.fast_history[i+1]
                   for i in range(len(state.fast_history)-1))
    slow_dec = all(state.slow_history[i] > state.slow_history[i+1]
                   for i in range(len(state.slow_history)-1))
    slow_inc = all(state.slow_history[i] < state.slow_history[i+1]
                   for i in range(len(state.slow_history)-1))

    long_run = 1 if (fast_inc and (slow_dec or slow_inc)) else 0

    # Short run: fast decreasing AND (slow increasing OR slow decreasing)
    fast_dec = all(state.fast_history[i] > state.fast_history[i+1]
                   for i in range(len(state.fast_history)-1))

    short_run = 1 if (fast_dec and (slow_inc or slow_dec)) else 0

    return [long_run, short_run], state


def _amat_output_names(params: Dict[str, Any]) -> List[str]:
    fast = _as_int(_param(params, "fast", 8), 8)
    slow = _as_int(_param(params, "slow", 21), 21)
    lookback = _as_int(_param(params, "lookback", 2), 2)
    mamode = str(_param(params, "mamode", "ema"))
    prefix = mamode[0].lower() if mamode else "e"
    props = f"_{fast}_{slow}_{lookback}"
    return [f"AMAT{prefix}_LR{props}", f"AMAT{prefix}_SR{props}"]


def _amat_seed(series: Dict[str, Any], params: Dict[str, Any]) -> AmatState:
    """Reconstruct AmatState by replaying over the raw input series."""
    return replay_seed("amat", series, params)


STATEFUL_REGISTRY["amat"] = StatefulIndicator(
    kind="amat",
    inputs=("close",),
    init=_amat_init,
    update=_amat_update,
    output_names=_amat_output_names,
)
SEED_REGISTRY["amat"] = _amat_seed


# ===========================================================================
# CKSP  (internal_series)
# ===========================================================================
# State: atr_state
# Default: p=10, x=1, q=9, mamode="rma" (tvmode=True)

@dataclass
class CkspState:
    p: int
    x: float
    q: int
    mamode: str
    tr_ma_state: Any
    prev_close: Optional[float] = None
    high_buf: deque = field(default_factory=lambda: deque(maxlen=10))
    low_buf: deque = field(default_factory=lambda: deque(maxlen=10))
    long_stop_buf: deque = field(default_factory=lambda: deque(maxlen=9))
    short_stop_buf: deque = field(default_factory=lambda: deque(maxlen=9))


def _cksp_init(params: Dict[str, Any]) -> CkspState:
    tvmode_param = params.get("tvmode", None)
    mode_tv = bool(_param(params, "tvmode", True))
    p = _as_int(_param(params, "p", 10), 10)
    x_param = params.get("x", None)
    q_param = params.get("q", None)
    x = float(x_param) if isinstance(x_param, float) and x_param > 0 else (1.0 if tvmode_param is True else 3.0)
    q = int(q_param) if isinstance(q_param, float) and q_param > 0 else (9 if tvmode_param is True else 20)
    mamode = str(_param(params, "mamode", "rma" if mode_tv else "sma"))

    state = CkspState(
        p=p, x=x, q=q, mamode=mamode,
        tr_ma_state=_ma_state_make(mamode, p),
    )
    state.high_buf = deque(maxlen=p)
    state.low_buf = deque(maxlen=p)
    state.long_stop_buf = deque(maxlen=q)
    state.short_stop_buf = deque(maxlen=q)

    return state


def _cksp_update(
    state: CkspState, bar: Dict[str, Any], params: Dict[str, Any]
) -> Tuple[List[Optional[float]], CkspState]:
    high, low, close = bar["high"], bar["low"], bar["close"]

    # ATR calculation (TR -> MA)
    if state.prev_close is None:
        tr = high - low
    else:
        tr = max(high - low, abs(high - state.prev_close), abs(low - state.prev_close))
    state.prev_close = close
    atr_val, state.tr_ma_state = _ma_state_update(state.tr_ma_state, tr, state.mamode)

    state.high_buf.append(high)
    state.low_buf.append(low)

    if atr_val is None or len(state.high_buf) < state.p:
        return [None, None], state

    # First stop: max(high[p]) - x * atr
    max_high = max(state.high_buf)
    min_low = min(state.low_buf)

    long_stop_ = max_high - state.x * atr_val
    short_stop_ = min_low + state.x * atr_val

    state.long_stop_buf.append(long_stop_)
    state.short_stop_buf.append(short_stop_)

    if len(state.long_stop_buf) < state.q:
        return [None, None], state

    # Second stop: max/min of first stops
    long_stop = max(state.long_stop_buf)
    short_stop = min(state.short_stop_buf)

    return [long_stop, short_stop], state


def _cksp_output_names(params: Dict[str, Any]) -> List[str]:
    p = _as_int(_param(params, "p", 10), 10)
    tvmode_param = params.get("tvmode", None)
    x_param = params.get("x", None)
    q_param = params.get("q", None)
    x = float(x_param) if isinstance(x_param, float) and x_param > 0 else (1.0 if tvmode_param is True else 3.0)
    q = int(q_param) if isinstance(q_param, float) and q_param > 0 else (9 if tvmode_param is True else 20)
    x = _fmt_num(x)
    props = f"_{p}_{x}_{q}"
    return [f"CKSPl{props}", f"CKSPs{props}"]


def _cksp_seed(series: Dict[str, Any], params: Dict[str, Any]) -> CkspState:
    """Reconstruct CkspState by replaying over the raw input series."""
    return replay_seed("cksp", series, params)


STATEFUL_REGISTRY["cksp"] = StatefulIndicator(
    kind="cksp",
    inputs=("high", "low", "close"),
    init=_cksp_init,
    update=_cksp_update,
    output_names=_cksp_output_names,
)
SEED_REGISTRY["cksp"] = _cksp_seed


# ===========================================================================
# HT_TRENDLINE  (internal_series)
# ===========================================================================
# State: wma4, dt, q1, i1, ji, jq, i2, q2, re, im, period, smp, i_trend
# Hilbert Transform with many internal series

@dataclass
class HtTrendlineState:
    wma4: float = 0.0
    dt: float = 0.0
    q1: float = 0.0
    i1: float = 0.0
    ji: float = 0.0
    jq: float = 0.0
    i2: float = 0.0
    q2: float = 0.0
    re: float = 0.0
    im: float = 0.0
    period: float = 0.0
    smp: float = 0.0
    i_trend: float = 0.0
    # History buffers for lookback
    wma4_buf: deque = field(default_factory=lambda: deque([0.0]*7, maxlen=7))
    dt_buf: deque = field(default_factory=lambda: deque([0.0]*7, maxlen=7))
    i1_buf: deque = field(default_factory=lambda: deque([0.0]*7, maxlen=7))
    q1_buf: deque = field(default_factory=lambda: deque([0.0]*7, maxlen=7))
    i2_buf: deque = field(default_factory=lambda: deque([0.0]*2, maxlen=2))
    q2_buf: deque = field(default_factory=lambda: deque([0.0]*2, maxlen=2))
    re_buf: deque = field(default_factory=lambda: deque([0.0]*2, maxlen=2))
    im_buf: deque = field(default_factory=lambda: deque([0.0]*2, maxlen=2))
    period_buf: deque = field(default_factory=lambda: deque([0.0]*2, maxlen=2))
    i_trend_buf: deque = field(default_factory=lambda: deque([0.0]*4, maxlen=4))
    close_buf: deque = field(default_factory=lambda: deque([0.0]*4, maxlen=4))
    warmup: int = 0
    use_talib: bool = False
    close_hist: list = field(default_factory=list)


def _ht_trendline_init(params: Dict[str, Any]) -> HtTrendlineState:
    # Default to internal incremental implementation for performance.
    use_talib = bool(_param(params, "talib", False)) and Imports.get("talib", False)
    return HtTrendlineState(use_talib=use_talib)


def _ht_trendline_update(
    state: HtTrendlineState, bar: Dict[str, Any], params: Dict[str, Any]
) -> Tuple[List[Optional[float]], HtTrendlineState]:
    x = bar["close"]
    if _is_nan(x):
        return [None], state

    if state.use_talib:
        # Accuracy-first: delegate to TA-Lib using full history buffer.
        state.close_hist.append(float(x))
        try:
            import talib  # type: ignore
        except Exception:
            # Fallback to custom incremental if TA-Lib unavailable at runtime.
            state.use_talib = False
        else:
            arr = np.asarray(state.close_hist, dtype=float)
            out = talib.HT_TRENDLINE(arr)
            last = out[-1] if len(out) else np.nan
            if np.isnan(last):
                return [None], state
            return [float(last)], state

    state.close_buf.append(x)
    state.warmup += 1

    # Need at least 13 bars (following TALib)
    if state.warmup < 13:
        return [None], state

    a, b = 0.0962, 0.5769

    # WMA4
    if len(state.close_buf) >= 4:
        state.wma4 = (0.4 * state.close_buf[-1] + 0.3 * state.close_buf[-2] +
                      0.2 * state.close_buf[-3] + 0.1 * state.close_buf[-4])
    state.wma4_buf.append(state.wma4)

    if state.warmup < 7:
        return [None], state

    # Detrend
    adj_prev_period = 0.075 * state.period_buf[-1] + 0.54
    state.dt = adj_prev_period * (a * state.wma4_buf[-1] + b * state.wma4_buf[-3] -
                                   b * state.wma4_buf[-5] - a * state.wma4_buf[-7])
    state.dt_buf.append(state.dt)

    # Q1
    state.q1 = adj_prev_period * (a * state.dt_buf[-1] + b * state.dt_buf[-3] -
                                   b * state.dt_buf[-5] - a * state.dt_buf[-7])
    state.q1_buf.append(state.q1)

    # I1
    state.i1 = state.dt_buf[-4] if len(state.dt_buf) >= 4 else 0.0
    state.i1_buf.append(state.i1)

    # JI, JQ
    state.ji = adj_prev_period * (a * state.i1_buf[-1] + b * state.i1_buf[-3] -
                                   b * state.i1_buf[-5] - a * state.i1_buf[-7])
    state.jq = adj_prev_period * (a * state.q1_buf[-1] + b * state.q1_buf[-3] -
                                   b * state.q1_buf[-5] - a * state.q1_buf[-7])

    # I2, Q2
    i2_raw = state.i1 - state.jq
    q2_raw = state.q1 + state.ji

    state.i2 = 0.2 * i2_raw + 0.8 * state.i2_buf[-1]
    state.q2 = 0.2 * q2_raw + 0.8 * state.q2_buf[-1]
    state.i2_buf.append(state.i2)
    state.q2_buf.append(state.q2)

    # RE, IM
    re_raw = state.i2 * state.i2_buf[-2] + state.q2 * state.q2_buf[-2]
    im_raw = state.i2 * state.q2_buf[-2] - state.q2 * state.i2_buf[-2]

    state.re = 0.2 * re_raw + 0.8 * state.re_buf[-1]
    state.im = 0.2 * im_raw + 0.8 * state.im_buf[-1]
    state.re_buf.append(state.re)
    state.im_buf.append(state.im)

    # Period
    if state.re != 0 and state.im != 0:
        period_raw = 360.0 / (math.degrees(math.atan(state.im / state.re)))
        if period_raw > 1.5 * state.period_buf[-1]:
            period_raw = 1.5 * state.period_buf[-1]
        if period_raw < 0.67 * state.period_buf[-1]:
            period_raw = 0.67 * state.period_buf[-1]
        if period_raw < 6.0:
            period_raw = 6.0
        if period_raw > 50.0:
            period_raw = 50.0
        state.period = 0.2 * period_raw + 0.8 * state.period_buf[-1]
    state.period_buf.append(state.period)

    # SMP
    state.smp = 0.33 * state.period + 0.67 * state.smp

    # DC Period average
    dc_period = int(state.smp + 0.5)
    if dc_period > 0 and len(state.close_buf) >= dc_period:
        dcp_avg = sum(state.close_buf[-i] for i in range(1, min(dc_period+1, len(state.close_buf)+1))) / dc_period
    else:
        dcp_avg = x

    state.i_trend = dcp_avg
    state.i_trend_buf.append(state.i_trend)

    # Final result: WMA of i_trend
    if len(state.i_trend_buf) >= 4:
        result = (0.4 * state.i_trend_buf[-1] + 0.3 * state.i_trend_buf[-2] +
                  0.2 * state.i_trend_buf[-3] + 0.1 * state.i_trend_buf[-4])
    else:
        result = None

    return [result], state


def _ht_trendline_output_names(params: Dict[str, Any]) -> List[str]:
    return ["HT_TL"]


def _ht_trendline_seed(series: Dict[str, Any], params: Dict[str, Any]) -> HtTrendlineState:
    """Reconstruct HtTrendlineState by replaying or priming history."""
    state = _ht_trendline_init(params)
    if state.use_talib:
        close_s = series.get("close")
        if close_s is not None:
            import pandas as _pd
            for v in close_s:
                if _pd.isna(v):
                    continue
                state.close_hist.append(float(v))
        return state
    return replay_seed("ht_trendline", series, params)


STATEFUL_REGISTRY["ht_trendline"] = StatefulIndicator(
    kind="ht_trendline",
    inputs=("close",),
    init=_ht_trendline_init,
    update=_ht_trendline_update,
    output_names=_ht_trendline_output_names,
)
SEED_REGISTRY["ht_trendline"] = _ht_trendline_seed


# ===========================================================================
# QSTICK  (internal_series)
# ===========================================================================
# State: ma_state
# Default: length=10, mamode="sma"

@dataclass
class QstickState:
    length: int
    mamode: str
    ma_state: Any       # EMAState or deque


def _qstick_init(params: Dict[str, Any]) -> QstickState:
    length = _as_int(_param(params, "length", 10), 10)
    mamode = str(_param(params, "mamode", "sma"))

    return QstickState(
        length=length,
        mamode=mamode,
        ma_state=_ma_state_make(mamode, length),
    )


def _qstick_update(
    state: QstickState, bar: Dict[str, Any], params: Dict[str, Any]
) -> Tuple[List[Optional[float]], QstickState]:
    open_ = bar["open"]
    close = bar["close"]

    diff = close - open_
    qs_val, state.ma_state = _ma_state_update(state.ma_state, diff, state.mamode)

    return [qs_val], state


def _qstick_output_names(params: Dict[str, Any]) -> List[str]:
    length = _as_int(_param(params, "length", 10), 10)
    return [f"QS_{length}"]


def _qstick_seed(series: Dict[str, Any], params: Dict[str, Any]) -> QstickState:
    """Reconstruct QstickState by replaying over the raw input series."""
    return replay_seed("qstick", series, params)


STATEFUL_REGISTRY["qstick"] = StatefulIndicator(
    kind="qstick",
    inputs=("open", "close"),
    init=_qstick_init,
    update=_qstick_update,
    output_names=_qstick_output_names,
)
SEED_REGISTRY["qstick"] = _qstick_seed


# ===========================================================================
# RWI  (internal_series)
# ===========================================================================
# State: atr_state
# Default: length=14, mamode="rma"

@dataclass
class RwiState:
    length: int
    mamode: str
    atr_state: ATRState
    high_buf: deque = field(default_factory=lambda: deque(maxlen=15))
    low_buf: deque = field(default_factory=lambda: deque(maxlen=15))


def _rwi_init(params: Dict[str, Any]) -> RwiState:
    length = _as_int(_param(params, "length", 14), 14)
    mamode = str(_param(params, "mamode", "rma"))

    state = RwiState(
        length=length,
        mamode=mamode,
        atr_state=ATRState(length=length, percent=False),
    )
    state.high_buf = deque(maxlen=length + 1)
    state.low_buf = deque(maxlen=length + 1)

    return state


def _rwi_update(
    state: RwiState, bar: Dict[str, Any], params: Dict[str, Any]
) -> Tuple[List[Optional[float]], RwiState]:
    high, low, close = bar["high"], bar["low"], bar["close"]

    # ATR calculation
    atr_val, state.atr_state = atr_update_raw(state.atr_state, high, low, close)

    state.high_buf.append(high)
    state.low_buf.append(low)

    if atr_val is None or atr_val == 0.0 or len(state.high_buf) <= state.length:
        return [None, None], state

    # RWI calculation
    denom = atr_val * (state.length ** 0.5)

    # RWI High: (high - low[length]) / denom
    rwi_high = (high - state.low_buf[0]) / denom

    # RWI Low: (high[length] - low) / denom
    rwi_low = (state.high_buf[0] - low) / denom

    return [rwi_high, rwi_low], state


def _rwi_output_names(params: Dict[str, Any]) -> List[str]:
    length = _as_int(_param(params, "length", 14), 14)
    return [f"RWIh_{length}", f"RWIl_{length}"]


def _rwi_seed(series: Dict[str, Any], params: Dict[str, Any]) -> RwiState:
    """Reconstruct RwiState by replaying over the raw input series."""
    return replay_seed("rwi", series, params)


STATEFUL_REGISTRY["rwi"] = StatefulIndicator(
    kind="rwi",
    inputs=("high", "low", "close"),
    init=_rwi_init,
    update=_rwi_update,
    output_names=_rwi_output_names,
)
SEED_REGISTRY["rwi"] = _rwi_seed


# ===========================================================================
# TRENDFLEX  (internal_series)
# ===========================================================================
# State: _f, _ms
# Default: length=20, smooth=20, alpha=0.04

@dataclass
class TrendflexState:
    length: int
    smooth: int
    alpha: float
    # SuperSmoother filter state
    _f: deque = field(default_factory=lambda: deque([0.0, 0.0], maxlen=2))
    # Mean square state
    _ms: float = 0.0
    # Close buffer for sum calculation
    _f_buf: deque = field(default_factory=lambda: deque(maxlen=21))
    close_buf: deque = field(default_factory=lambda: deque([0.0, 0.0], maxlen=2))
    warmup: int = 0
    # Filter coefficients (calculated once)
    a: float = 0.0
    b: float = 0.0
    c: float = 0.0


def _trendflex_init(params: Dict[str, Any]) -> TrendflexState:
    length = _as_int(_param(params, "length", 20), 20)
    smooth = _as_int(_param(params, "smooth", 20), 20)
    alpha = _as_float(_param(params, "alpha", 0.04), 0.04)
    pi = _as_float(_param(params, "pi", 3.14159), 3.14159)
    sqrt2 = _as_float(_param(params, "sqrt2", 1.414), 1.414)

    # Calculate filter coefficients
    ratio = 2 * sqrt2 / smooth
    a = math.exp(-pi * ratio)
    b = 2 * a * math.cos(math.radians(180 * ratio))
    c = a * a - b + 1

    state = TrendflexState(length=length, smooth=smooth, alpha=alpha)
    state.a = a
    state.b = b
    state.c = c
    state._f_buf = deque(maxlen=length)

    return state


def _trendflex_update(
    state: TrendflexState, bar: Dict[str, Any], params: Dict[str, Any]
) -> Tuple[List[Optional[float]], TrendflexState]:
    x = bar["close"]
    state.close_buf.append(x)
    state.warmup += 1

    # SuperSmoother filter
    if state.warmup >= 2:
        f_val = (0.5 * state.c * (state.close_buf[-1] + state.close_buf[-2]) +
                 state.b * state._f[-1] - state.a * state.a * state._f[-2])
        state._f.append(f_val)
        state._f_buf.append(f_val)
    else:
        return [None], state

    if len(state._f_buf) < state.length:
        return [None], state

    # Calculate sum of differences
    _sum = 0.0
    for j in range(1, state.length):
        _sum += state._f_buf[-1] - state._f_buf[-1-j]
    _sum /= state.length

    # Update mean square with EMA
    state._ms = state.alpha * _sum * _sum + (1 - state.alpha) * state._ms

    # Calculate result
    if state._ms != 0.0:
        result = _sum / math.sqrt(state._ms)
    else:
        result = 0.0

    return [result], state


def _trendflex_output_names(params: Dict[str, Any]) -> List[str]:
    length = _as_int(_param(params, "length", 20), 20)
    smooth = _as_int(_param(params, "smooth", 20), 20)
    alpha = _as_float(_param(params, "alpha", 0.04), 0.04)
    return [f"TRENDFLEX_{length}_{smooth}_{alpha}"]


def _trendflex_seed(series: Dict[str, Any], params: Dict[str, Any]) -> TrendflexState:
    """Reconstruct TrendflexState by replaying over the raw input series."""
    return replay_seed("trendflex", series, params)


STATEFUL_REGISTRY["trendflex"] = StatefulIndicator(
    kind="trendflex",
    inputs=("close",),
    init=_trendflex_init,
    update=_trendflex_update,
    output_names=_trendflex_output_names,
)
SEED_REGISTRY["trendflex"] = _trendflex_seed


# ===========================================================================
# PSAR  (replay_only)
# ===========================================================================
# State: sar, af, ep, falling
# Default: af0=0.02, af=0.02, max_af=0.2

@dataclass
class PsarState:
    af0: float
    af: float
    max_af: float
    sar: float
    ep: float           # extreme point
    falling: bool
    prev_high: Optional[float] = None
    prev_low: Optional[float] = None
    prev_close: Optional[float] = None
    initialized: bool = False


def _psar_init(params: Dict[str, Any]) -> PsarState:
    af = _as_float(_param(params, "af", 0.02), 0.02)
    af0 = _as_float(_param(params, "af0", af), af)
    max_af = _as_float(_param(params, "max_af", 0.2), 0.2)

    return PsarState(
        af0=af0,
        af=af0,
        max_af=max_af,
        sar=0.0,
        ep=0.0,
        falling=False,
    )


def _psar_update(
    state: PsarState, bar: Dict[str, Any], params: Dict[str, Any]
) -> Tuple[List[Optional[float]], PsarState]:
    high, low, close = bar["high"], bar["low"], bar["close"]

    # First bar initialization: store prevs, return NaN for long/short
    if state.prev_high is None or state.prev_low is None:
        state.prev_high = high
        state.prev_low = low
        state.prev_close = close
        state.sar = close
        return [None, None, state.af0, 0], state

    # Second bar: determine initial trend (matches vectorized _falling)
    if not state.initialized:
        up = high - state.prev_high
        dn = state.prev_low - low
        state.falling = (dn > up and dn > 0)
        state.ep = state.prev_low if state.falling else state.prev_high
        if state.prev_close is not None:
            state.sar = state.prev_close
        state.af = state.af0
        state.initialized = True

    # Calculate new SAR
    sar = state.sar + state.af * (state.ep - state.sar)
    reverse = False

    if state.falling:
        reverse = high > sar
        if low < state.ep:
            state.ep = low
            state.af = min(state.af + state.af0, state.max_af)
        sar = max(state.prev_high, sar)
    else:
        reverse = low < sar
        if high > state.ep:
            state.ep = high
            state.af = min(state.af + state.af0, state.max_af)
        sar = min(state.prev_low, sar)

    if reverse:
        sar = state.ep
        state.af = state.af0
        state.falling = not state.falling
        state.ep = low if state.falling else high

    state.sar = sar
    state.prev_high = high
    state.prev_low = low
    state.prev_close = close

    # Return long/short values
    long_val = state.sar if not state.falling else None
    short_val = state.sar if state.falling else None

    return [long_val, short_val, state.af, int(reverse)], state


def _psar_output_names(params: Dict[str, Any]) -> List[str]:
    af = _as_float(_param(params, "af", 0.02), 0.02)
    af0 = _as_float(_param(params, "af0", af), af)
    max_af = _as_float(_param(params, "max_af", 0.2), 0.2)
    props = f"_{af0}_{max_af}"
    return [f"PSARl{props}", f"PSARs{props}", f"PSARaf{props}", f"PSARr{props}"]


# replay_only: no SEED_REGISTRY entry

STATEFUL_REGISTRY["psar"] = StatefulIndicator(
    kind="psar",
    inputs=("high", "low", "close"),
    init=_psar_init,
    update=_psar_update,
    output_names=_psar_output_names,
)


# ===========================================================================
# ZIGZAG  (replay_only)
# ===========================================================================
# State: pivot_history, zz_state
# Default: legs=10, deviation=5

@dataclass
class ZigzagState:
    legs: int
    deviation: float
    # Pivot detection state
    high_buf: deque = field(default_factory=lambda: deque(maxlen=21))
    low_buf: deque = field(default_factory=lambda: deque(maxlen=21))
    idx: int = 0
    # ZigZag state
    last_zz_swing: int = 0      # 1=top, -1=bottom, 0=none
    last_zz_value: float = 0.0
    last_zz_idx: int = -1
    changes: int = 0
    # Pivot history
    pivot_history: List[Tuple[int, int, float]] = field(default_factory=list)  # (idx, swing, value)


def _zigzag_init(params: Dict[str, Any]) -> ZigzagState:
    legs = _as_int(_param(params, "legs", 10), 10)
    deviation = _as_float(_param(params, "deviation", 5), 5.0)

    state = ZigzagState(legs=legs, deviation=deviation)
    state.high_buf = deque(maxlen=legs)
    state.low_buf = deque(maxlen=legs)

    return state


def _zigzag_update(
    state: ZigzagState, bar: Dict[str, Any], params: Dict[str, Any]
) -> Tuple[List[Optional[float]], ZigzagState]:
    """ZigZag update - simplified live mode.

    Returns [swing, value, dev] where:
    - swing: 1 for high pivot, -1 for low pivot, None otherwise
    - value: price at pivot, None otherwise
    - dev:   deviation from last pivot, None otherwise (not computed here)
    """
    high, low = bar["high"], bar["low"]

    state.high_buf.append(high)
    state.low_buf.append(low)
    state.idx += 1

    # Need full window for pivot detection
    if len(state.high_buf) < state.legs:
        return [None, None, None], state

    # Check if current midpoint is a pivot
    # For simplicity in stateful mode, we detect pivots at center of window
    mid_idx = state.legs // 2

    # Check if it's a low pivot (lowest in window)
    if len(state.low_buf) == state.legs:
        mid_low = state.low_buf[mid_idx]
        is_low_pivot = all(mid_low <= v for v in state.low_buf)

        if is_low_pivot:
            pivot_idx = state.idx - (state.legs - mid_idx)
            state.pivot_history.append((pivot_idx, -1, mid_low))

    # Check if it's a high pivot (highest in window)
    if len(state.high_buf) == state.legs:
        mid_high = state.high_buf[mid_idx]
        is_high_pivot = all(mid_high >= v for v in state.high_buf)

        if is_high_pivot:
            pivot_idx = state.idx - (state.legs - mid_idx)
            state.pivot_history.append((pivot_idx, 1, mid_high))

    # Process latest pivot if any
    if not state.pivot_history:
        return [None, None, None], state

    # For live mode, return None (zigzag requires full history analysis)
    # Full implementation would need backtest mode logic
    return [None, None, None], state


def _zigzag_output_names(params: Dict[str, Any]) -> List[str]:
    deviation = _as_float(_param(params, "deviation", 5), 5.0)
    legs = _as_int(_param(params, "legs", 10), 10)
    props = f"_{deviation}%_{legs}"
    return [f"ZIGZAGs{props}", f"ZIGZAGv{props}", f"ZIGZAGd{props}"]


# replay_only: no SEED_REGISTRY entry

STATEFUL_REGISTRY["zigzag"] = StatefulIndicator(
    kind="zigzag",
    inputs=("high", "low"),
    init=_zigzag_init,
    update=_zigzag_update,
    output_names=_zigzag_output_names,
)
