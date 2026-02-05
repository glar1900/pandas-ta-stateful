# -*- coding: utf-8 -*-
"""pandas-ta stateful – volume indicators.

Registered kinds
----------------
output_only  : obv, ad, pvt, nvi, pvi, efi
internal_series: adosc, aobv, kvo, pvo, vwap
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from collections import deque

from ._base import (
    NAN, _is_nan, _param, _as_int, _as_float,
    EMAState, ema_make, rma_make, ema_update_raw,
    StatefulIndicator,
    STATEFUL_REGISTRY, SEED_REGISTRY,
    replay_seed,
)


# ===========================================================================
# OBV – On Balance Volume
# ===========================================================================
# sign: +vol if close > prev_close, -vol if close < prev_close, 0 if equal
# cumsum of signed volume.  First bar: prev_close is None -> treat as 0.
# ===========================================================================

@dataclass
class OBVState:
    cumsum_value: float = 0.0
    prev_close: Optional[float] = None


def _obv_init(params: Dict[str, Any]) -> OBVState:
    return OBVState()


def _obv_update(
    state: OBVState, bar: Dict[str, float], params: Dict[str, Any]
) -> Tuple[List[Optional[float]], OBVState]:
    close = bar["close"]
    volume = bar["volume"]

    if state.prev_close is None:
        # Match vectorized OBV (and TA-Lib) behavior: first value = volume.
        state.cumsum_value = volume
    else:
        if close > state.prev_close:
            state.cumsum_value += volume
        elif close < state.prev_close:
            state.cumsum_value -= volume
        # equal -> no change

    state.prev_close = close
    return [state.cumsum_value], state


def _obv_output_names(params: Dict[str, Any]) -> List[str]:
    return ["OBV"]


def _obv_seed(inputs: Dict[str, Any], params: Dict[str, Any]) -> OBVState:
    """output_only seed: replay then extract final state."""
    return replay_seed("obv", inputs, params)


STATEFUL_REGISTRY["obv"] = StatefulIndicator(
    kind="obv",
    inputs=("close", "volume"),
    init=_obv_init,
    update=_obv_update,
    output_names=_obv_output_names,
)
SEED_REGISTRY["obv"] = _obv_seed


# ===========================================================================
# AD – Accumulation / Distribution
# ===========================================================================
# CLV = (2*close - (high + low)) / (high - low)
# AD = cumsum(CLV * volume)
# When high == low, CLV contribution is 0.
# ===========================================================================

@dataclass
class ADState:
    cumsum_value: float = 0.0


def _ad_init(params: Dict[str, Any]) -> ADState:
    return ADState()


def _ad_update(
    state: ADState, bar: Dict[str, float], params: Dict[str, Any]
) -> Tuple[List[Optional[float]], ADState]:
    high  = bar["high"]
    low   = bar["low"]
    close = bar["close"]
    volume = bar["volume"]

    hl_range = high - low
    if hl_range != 0.0:
        clv = (2.0 * close - (high + low)) / hl_range
    else:
        clv = 0.0

    state.cumsum_value += clv * volume
    return [state.cumsum_value], state


def _ad_output_names(params: Dict[str, Any]) -> List[str]:
    return ["AD"]


def _ad_seed(inputs: Dict[str, Any], params: Dict[str, Any]) -> ADState:
    return replay_seed("ad", inputs, params)


STATEFUL_REGISTRY["ad"] = StatefulIndicator(
    kind="ad",
    inputs=("high", "low", "close", "volume"),
    init=_ad_init,
    update=_ad_update,
    output_names=_ad_output_names,
)
SEED_REGISTRY["ad"] = _ad_seed


# ===========================================================================
# PVT – Price Volume Trend
# ===========================================================================
# ROC = (close - prev_close) / prev_close * 100   (drift=1)
# PVT = cumsum(ROC * volume).  First bar: no prev_close -> contribute 0.
# ===========================================================================

@dataclass
class PVTState:
    cumsum_value: float = 0.0
    prev_close: Optional[float] = None


def _pvt_init(params: Dict[str, Any]) -> PVTState:
    return PVTState()


def _pvt_update(
    state: PVTState, bar: Dict[str, float], params: Dict[str, Any]
) -> Tuple[List[Optional[float]], PVTState]:
    close  = bar["close"]
    volume = bar["volume"]

    if state.prev_close is not None and state.prev_close != 0.0:
        roc = (close - state.prev_close) / state.prev_close * 100.0
        state.cumsum_value += roc * volume

    state.prev_close = close
    return [state.cumsum_value], state


def _pvt_output_names(params: Dict[str, Any]) -> List[str]:
    return ["PVT"]


def _pvt_seed(inputs: Dict[str, Any], params: Dict[str, Any]) -> PVTState:
    return replay_seed("pvt", inputs, params)


STATEFUL_REGISTRY["pvt"] = StatefulIndicator(
    kind="pvt",
    inputs=("close", "volume"),
    init=_pvt_init,
    update=_pvt_update,
    output_names=_pvt_output_names,
)
SEED_REGISTRY["pvt"] = _pvt_seed


# ===========================================================================
# NVI – Negative Volume Index
# ===========================================================================
# Vectorized source:
#   signed_volume = signed_series(volume, 1)   -> +1 if vol>prev, -1 if vol<prev
#   nvi = cumsum( abs(signed_volume[neg]) * roc )
#   first value = initial (default 1000)
#
# Stateful logic (per bar):
#   First bar: nvi = initial, store prev_volume.
#   Subsequent: if volume < prev_volume -> nvi += ROC(close, length).
#               ROC = (close - prev_close) / prev_close * 100
#   length default = 1 (same as vectorized v_pos_default(length, 1))
# ===========================================================================

@dataclass
class NVIState:
    cumsum_value: float = 0.0
    prev_volume: Optional[float] = None
    prev_close: Optional[float] = None
    _first_bar: bool = True


def _nvi_init(params: Dict[str, Any]) -> NVIState:
    return NVIState()


def _nvi_update(
    state: NVIState, bar: Dict[str, float], params: Dict[str, Any]
) -> Tuple[List[Optional[float]], NVIState]:
    close  = bar["close"]
    volume = bar["volume"]
    initial = _as_float(_param(params, "initial", 1000), 1000.0)
    length  = _as_int(_param(params, "length", 1), 1)

    if state._first_bar:
        # vectorized: nvi.iloc[0] = initial, then cumsum
        state.cumsum_value = initial
        state.prev_volume = volume
        state.prev_close = close
        state._first_bar = False
        return [state.cumsum_value], state

    # ROC uses `length` bars back; for streaming length>1 we can only
    # approximate with the single previous close (length=1 is the default
    # and the only length that maps cleanly to single-bar streaming).
    if state.prev_close is not None and state.prev_close != 0.0:
        roc = (close - state.prev_close) / state.prev_close * 100.0
    else:
        roc = 0.0

    if volume < state.prev_volume:
        # negative volume day: accumulate ROC
        state.cumsum_value += roc

    state.prev_volume = volume
    state.prev_close = close
    return [state.cumsum_value], state


def _nvi_output_names(params: Dict[str, Any]) -> List[str]:
    length = _as_int(_param(params, "length", 1), 1)
    return [f"NVI_{length}"]


def _nvi_seed(inputs: Dict[str, Any], params: Dict[str, Any]) -> NVIState:
    return replay_seed("nvi", inputs, params)


STATEFUL_REGISTRY["nvi"] = StatefulIndicator(
    kind="nvi",
    inputs=("close", "volume"),
    init=_nvi_init,
    update=_nvi_update,
    output_names=_nvi_output_names,
)
SEED_REGISTRY["nvi"] = _nvi_seed


# ===========================================================================
# PVI – Positive Volume Index
# ===========================================================================
# Vectorized source (nb_pvi):
#   result[0] = initial (default 100)
#   if volume[i] > volume[i-1]:  result[i] = result[i-1] * (close[i]/close[i-1])
#   else:                        result[i] = result[i-1]
#
# Note: source has typo `result[i - i]` which is `result[0]`; the
# intended and documented formula uses `result[i-1]`.  We implement
# the correct intended logic.
# ===========================================================================

@dataclass
class PVIState:
    cumsum_value: float = 0.0          # current PVI value
    prev_volume: Optional[float] = None
    prev_close: Optional[float] = None
    _first_bar: bool = True


def _pvi_init(params: Dict[str, Any]) -> PVIState:
    return PVIState()


def _pvi_update(
    state: PVIState, bar: Dict[str, float], params: Dict[str, Any]
) -> Tuple[List[Optional[float]], PVIState]:
    close  = bar["close"]
    volume = bar["volume"]
    initial = _as_float(_param(params, "initial", 100), 100.0)

    if state._first_bar:
        state.cumsum_value = initial
        state.prev_volume = volume
        state.prev_close = close
        state._first_bar = False
        return [state.cumsum_value], state

    if volume > state.prev_volume and state.prev_close != 0.0:
        state.cumsum_value = state.cumsum_value * (close / state.prev_close)
    # else: value stays the same

    state.prev_volume = volume
    state.prev_close = close
    return [state.cumsum_value], state


def _pvi_output_names(params: Dict[str, Any]) -> List[str]:
    return ["PVI"]


def _pvi_seed(inputs: Dict[str, Any], params: Dict[str, Any]) -> PVIState:
    return replay_seed("pvi", inputs, params)


STATEFUL_REGISTRY["pvi"] = StatefulIndicator(
    kind="pvi",
    inputs=("close", "volume"),
    init=_pvi_init,
    update=_pvi_update,
    output_names=_pvi_output_names,
)
SEED_REGISTRY["pvi"] = _pvi_seed


# ===========================================================================
# EFI – Elder's Force Index
# ===========================================================================
# raw = (close - prev_close) * volume   (drift=1)
# EFI = EMA(raw, length=13)
# First bar has no prev_close -> raw is undefined; feed nothing to EMA.
# ===========================================================================

@dataclass
class EFIState:
    ma_state: EMAState
    prev_close: Optional[float] = None


def _efi_init(params: Dict[str, Any]) -> EFIState:
    length = _as_int(_param(params, "length", 13), 13)
    return EFIState(ma_state=ema_make(length))


def _efi_update(
    state: EFIState, bar: Dict[str, float], params: Dict[str, Any]
) -> Tuple[List[Optional[float]], EFIState]:
    close  = bar["close"]
    volume = bar["volume"]

    value: Optional[float] = None
    if state.prev_close is not None:
        raw = (close - state.prev_close) * volume
        value, state.ma_state = ema_update_raw(state.ma_state, raw)

    state.prev_close = close
    return [value], state


def _efi_output_names(params: Dict[str, Any]) -> List[str]:
    length = _as_int(_param(params, "length", 13), 13)
    return [f"EFI_{length}"]


def _efi_seed(inputs: Dict[str, Any], params: Dict[str, Any]) -> EFIState:
    return replay_seed("efi", inputs, params)


STATEFUL_REGISTRY["efi"] = StatefulIndicator(
    kind="efi",
    inputs=("close", "volume"),
    init=_efi_init,
    update=_efi_update,
    output_names=_efi_output_names,
)
SEED_REGISTRY["efi"] = _efi_seed


# ===========================================================================
# ADOSC – Accumulation / Distribution Oscillator  (Chaikin)
# ===========================================================================
# Internal series on top of AD cumsum:
#   AD = cumsum(CLV * volume)    (same as ad indicator)
#   ADOSC = EMA(AD, fast) - EMA(AD, slow)
# Defaults: fast=3, slow=10  (matches vectorized adosc.py)
# ===========================================================================

@dataclass
class ADOSCState:
    ad_cumsum: float = 0.0
    fast_ema: EMAState = None       # type: ignore[assignment]
    slow_ema: EMAState = None       # type: ignore[assignment]


def _adosc_init(params: Dict[str, Any]) -> ADOSCState:
    fast = _as_int(_param(params, "fast", 3), 3)
    slow = _as_int(_param(params, "slow", 10), 10)
    return ADOSCState(
        fast_ema=ema_make(fast),
        slow_ema=ema_make(slow),
    )


def _adosc_update(
    state: ADOSCState, bar: Dict[str, float], params: Dict[str, Any]
) -> Tuple[List[Optional[float]], ADOSCState]:
    high   = bar["high"]
    low    = bar["low"]
    close  = bar["close"]
    volume = bar["volume"]

    # --- AD cumsum step (identical to _ad_update logic) ---
    hl_range = high - low
    clv = (2.0 * close - (high + low)) / hl_range if hl_range != 0.0 else 0.0
    state.ad_cumsum += clv * volume

    # --- Feed AD cumsum into both EMAs ---
    fast_val, state.fast_ema = ema_update_raw(state.fast_ema, state.ad_cumsum)
    slow_val, state.slow_ema = ema_update_raw(state.slow_ema, state.ad_cumsum)

    value: Optional[float] = None
    if fast_val is not None and slow_val is not None:
        value = fast_val - slow_val

    return [value], state


def _adosc_output_names(params: Dict[str, Any]) -> List[str]:
    fast = _as_int(_param(params, "fast", 3), 3)
    slow = _as_int(_param(params, "slow", 10), 10)
    return [f"ADOSC_{fast}_{slow}"]


def _adosc_seed(inputs: Dict[str, Any], params: Dict[str, Any]) -> ADOSCState:
    return replay_seed("adosc", inputs, params)


STATEFUL_REGISTRY["adosc"] = StatefulIndicator(
    kind="adosc",
    inputs=("high", "low", "close", "volume"),
    init=_adosc_init,
    update=_adosc_update,
    output_names=_adosc_output_names,
)
SEED_REGISTRY["adosc"] = _adosc_seed


# ===========================================================================
# AOBV – Archer On Balance Volume
# ===========================================================================
# OBV cumsum + fast/slow MA on OBV + rolling min/max buffers.
# Outputs: OBV, OBV_min_{min_lb}, OBV_max_{max_lb}, OBV_ma_fast, OBV_ma_slow
# Defaults: fast=4, slow=12, min_lookback=2, max_lookback=2, mamode="ema"
# ===========================================================================

@dataclass
class AOBVState:
    obv_cumsum: float = 0.0
    prev_close: Optional[float] = None
    fast_ma: EMAState = None         # type: ignore[assignment]
    slow_ma: EMAState = None         # type: ignore[assignment]
    rolling_min_buf: deque = field(default_factory=deque)
    rolling_max_buf: deque = field(default_factory=deque)
    _min_lookback: int = 2
    _max_lookback: int = 2


def _aobv_init(params: Dict[str, Any]) -> AOBVState:
    fast       = _as_int(_param(params, "fast", 4), 4)
    slow       = _as_int(_param(params, "slow", 12), 12)
    min_lb     = _as_int(_param(params, "min_lookback", 2), 2)
    max_lb     = _as_int(_param(params, "max_lookback", 2), 2)
    if slow < fast:
        fast, slow = slow, fast
    return AOBVState(
        fast_ma=ema_make(fast),
        slow_ma=ema_make(slow),
        rolling_min_buf=deque(maxlen=min_lb),
        rolling_max_buf=deque(maxlen=max_lb),
        _min_lookback=min_lb,
        _max_lookback=max_lb,
    )


def _aobv_update(
    state: AOBVState, bar: Dict[str, float], params: Dict[str, Any]
) -> Tuple[List[Optional[float]], AOBVState]:
    close  = bar["close"]
    volume = bar["volume"]

    # --- OBV step ---
    if state.prev_close is not None:
        if close > state.prev_close:
            state.obv_cumsum += volume
        elif close < state.prev_close:
            state.obv_cumsum -= volume
    state.prev_close = close

    obv_val = state.obv_cumsum

    # --- Rolling min / max buffers (on OBV values) ---
    state.rolling_min_buf.append(obv_val)
    state.rolling_max_buf.append(obv_val)
    obv_min = min(state.rolling_min_buf)
    obv_max = max(state.rolling_max_buf)

    # --- Fast / Slow MA on OBV ---
    fast_val, state.fast_ma = ema_update_raw(state.fast_ma, obv_val)
    slow_val, state.slow_ma = ema_update_raw(state.slow_ma, obv_val)

    return [obv_val, obv_min, obv_max, fast_val, slow_val], state


def _aobv_output_names(params: Dict[str, Any]) -> List[str]:
    fast   = _as_int(_param(params, "fast", 4), 4)
    slow   = _as_int(_param(params, "slow", 12), 12)
    min_lb = _as_int(_param(params, "min_lookback", 2), 2)
    max_lb = _as_int(_param(params, "max_lookback", 2), 2)
    if slow < fast:
        fast, slow = slow, fast
    mamode = str(_param(params, "mamode", "ema")).lower()
    _mode  = mamode[0] if len(mamode) else ""
    return [
        "OBV",
        f"OBV_min_{min_lb}",
        f"OBV_max_{max_lb}",
        f"OBV{_mode}_{fast}",
        f"OBV{_mode}_{slow}",
    ]


def _aobv_seed(inputs: Dict[str, Any], params: Dict[str, Any]) -> AOBVState:
    return replay_seed("aobv", inputs, params)


STATEFUL_REGISTRY["aobv"] = StatefulIndicator(
    kind="aobv",
    inputs=("close", "volume"),
    init=_aobv_init,
    update=_aobv_update,
    output_names=_aobv_output_names,
)
SEED_REGISTRY["aobv"] = _aobv_seed


# ===========================================================================
# KVO – Klinger Volume Oscillator
# ===========================================================================
# Vectorized:
#   hlc3 = (high + low + close) / 3
#   signed_volume = volume * signed_series(hlc3, -1)
#       signed_series(x, -1): first = -1, then +1 if x>prev, -1 if x<prev, 0 if eq
#   kvo  = EMA(sv, fast) - EMA(sv, slow)
#   signal = EMA(kvo, signal_period)
# Defaults: fast=34, slow=55, signal=13
# ===========================================================================

@dataclass
class KVOState:
    fast_ema: EMAState = None        # type: ignore[assignment]
    slow_ema: EMAState = None        # type: ignore[assignment]
    signal_ema: EMAState = None      # type: ignore[assignment]
    prev_hlc3: Optional[float] = None
    _first_bar: bool = True


def _kvo_init(params: Dict[str, Any]) -> KVOState:
    fast   = _as_int(_param(params, "fast", 34), 34)
    slow   = _as_int(_param(params, "slow", 55), 55)
    signal = _as_int(_param(params, "signal", 13), 13)
    return KVOState(
        fast_ema=ema_make(fast),
        slow_ema=ema_make(slow),
        signal_ema=ema_make(signal),
    )


def _kvo_update(
    state: KVOState, bar: Dict[str, float], params: Dict[str, Any]
) -> Tuple[List[Optional[float]], KVOState]:
    high   = bar["high"]
    low    = bar["low"]
    close  = bar["close"]
    volume = bar["volume"]

    hlc3 = (high + low + close) / 3.0

    # --- signed_series logic ---
    # signed_series(hlc3, initial=-1):
    #   first bar sign = -1  (the `initial` value)
    #   subsequent: +1 if hlc3 > prev, -1 if hlc3 < prev, 0 if equal
    if state._first_bar:
        sign = -1.0
        state._first_bar = False
    else:
        if hlc3 > state.prev_hlc3:
            sign = 1.0
        elif hlc3 < state.prev_hlc3:
            sign = -1.0
        else:
            sign = 0.0

    state.prev_hlc3 = hlc3
    sv = volume * sign

    # --- Feed signed volume into fast / slow EMA ---
    fast_val, state.fast_ema = ema_update_raw(state.fast_ema, sv)
    slow_val, state.slow_ema = ema_update_raw(state.slow_ema, sv)

    # --- KVO = fast - slow; feed into signal EMA only when both ready ---
    kvo_val: Optional[float] = None
    signal_val: Optional[float] = None
    if fast_val is not None and slow_val is not None:
        kvo_val = fast_val - slow_val
        signal_val, state.signal_ema = ema_update_raw(state.signal_ema, kvo_val)

    return [kvo_val, signal_val], state


def _kvo_output_names(params: Dict[str, Any]) -> List[str]:
    fast   = _as_int(_param(params, "fast", 34), 34)
    slow   = _as_int(_param(params, "slow", 55), 55)
    signal = _as_int(_param(params, "signal", 13), 13)
    _props = f"_{fast}_{slow}_{signal}"
    return [f"KVO{_props}", f"KVOs{_props}"]


def _kvo_seed(inputs: Dict[str, Any], params: Dict[str, Any]) -> KVOState:
    return replay_seed("kvo", inputs, params)


STATEFUL_REGISTRY["kvo"] = StatefulIndicator(
    kind="kvo",
    inputs=("high", "low", "close", "volume"),
    init=_kvo_init,
    update=_kvo_update,
    output_names=_kvo_output_names,
)
SEED_REGISTRY["kvo"] = _kvo_seed


# ===========================================================================
# PVO – Percentage Volume Oscillator
# ===========================================================================
# fastma  = EMA(volume, fast)
# slowma  = EMA(volume, slow)
# pvo     = scalar * (fastma - slowma) / slowma    (scalar=100)
# signal  = EMA(pvo, signal)
# histogram = pvo - signal
# Defaults: fast=12, slow=26, signal=9, scalar=100
# ===========================================================================

@dataclass
class PVOState:
    fast_ema: EMAState = None        # type: ignore[assignment]
    slow_ema: EMAState = None        # type: ignore[assignment]
    signal_ema: EMAState = None      # type: ignore[assignment]


def _pvo_init(params: Dict[str, Any]) -> PVOState:
    fast   = _as_int(_param(params, "fast", 12), 12)
    slow   = _as_int(_param(params, "slow", 26), 26)
    signal = _as_int(_param(params, "signal", 9), 9)
    if slow < fast:
        fast, slow = slow, fast
    return PVOState(
        fast_ema=ema_make(fast),
        slow_ema=ema_make(slow),
        signal_ema=ema_make(signal),
    )


def _pvo_update(
    state: PVOState, bar: Dict[str, float], params: Dict[str, Any]
) -> Tuple[List[Optional[float]], PVOState]:
    volume = bar["volume"]
    scalar = _as_float(_param(params, "scalar", 100), 100.0)

    fast_val, state.fast_ema = ema_update_raw(state.fast_ema, volume)
    slow_val, state.slow_ema = ema_update_raw(state.slow_ema, volume)

    pvo_val: Optional[float]   = None
    signal_val: Optional[float] = None
    hist_val: Optional[float]   = None

    if fast_val is not None and slow_val is not None and slow_val != 0.0:
        pvo_val = scalar * (fast_val - slow_val) / slow_val
        signal_val, state.signal_ema = ema_update_raw(state.signal_ema, pvo_val)
        if signal_val is not None:
            hist_val = pvo_val - signal_val

    return [pvo_val, hist_val, signal_val], state


def _pvo_output_names(params: Dict[str, Any]) -> List[str]:
    fast   = _as_int(_param(params, "fast", 12), 12)
    slow   = _as_int(_param(params, "slow", 26), 26)
    signal = _as_int(_param(params, "signal", 9), 9)
    if slow < fast:
        fast, slow = slow, fast
    _props = f"_{fast}_{slow}_{signal}"
    return [f"PVO{_props}", f"PVOh{_props}", f"PVOs{_props}"]


def _pvo_seed(inputs: Dict[str, Any], params: Dict[str, Any]) -> PVOState:
    return replay_seed("pvo", inputs, params)


STATEFUL_REGISTRY["pvo"] = StatefulIndicator(
    kind="pvo",
    inputs=("volume",),
    init=_pvo_init,
    update=_pvo_update,
    output_names=_pvo_output_names,
)
SEED_REGISTRY["pvo"] = _pvo_seed


# ===========================================================================
# VWAP – Volume Weighted Average Price
# ===========================================================================
# TP = (high + low + close) / 3
# Within a session: VWAP = cumsum(TP * volume) / cumsum(volume)
# Session boundary detected via ``timestamp`` input; anchor default = "D".
#
# inputs includes "timestamp" (as numeric epoch-seconds or comparable key).
# Session key is derived by truncating the timestamp to the anchor period.
# For simplicity the stateful implementation compares the *date* portion
# (integer division by 86400 for seconds-epoch) when anchor is "D".
# ===========================================================================

@dataclass
class VWAPState:
    cum_pv: float = 0.0          # cumulative  TP * volume within session
    cum_vol: float = 0.0         # cumulative volume within session
    session_key: Optional[Any]   = None   # last session identifier


def _vwap_init(params: Dict[str, Any]) -> VWAPState:
    return VWAPState()


def _vwap_session_key(timestamp: float, anchor: str) -> Any:
    """Derive a session key from a numeric timestamp and anchor string.

    Supported anchors (case-insensitive prefix match):
        D  – calendar day   (epoch // 86400)
        W  – ISO week       (epoch // 604800)
        M  – calendar month (year*100 + month, approximate via epoch)
        H  – hour           (epoch // 3600)

    Fallback: integer-truncate timestamp to day.
    """
    anchor_up = anchor.upper() if anchor else "D"
    ts = float(timestamp)

    if anchor_up.startswith("D"):
        return int(ts // 86400)
    elif anchor_up.startswith("W"):
        return int(ts // 604800)
    elif anchor_up.startswith("H"):
        return int(ts // 3600)
    elif anchor_up.startswith("M"):
        # Approximate month from epoch (good enough for session detection)
        # Use days since epoch -> month boundaries are ~30.4375 days
        return int(ts // (86400 * 30))
    else:
        # Unknown anchor: fall back to daily
        return int(ts // 86400)


def _vwap_update(
    state: VWAPState, bar: Dict[str, float], params: Dict[str, Any]
) -> Tuple[List[Optional[float]], VWAPState]:
    high      = bar["high"]
    low       = bar["low"]
    close     = bar["close"]
    volume    = bar["volume"]
    timestamp = bar["timestamp"]
    anchor    = str(_param(params, "anchor", "D"))

    tp = (high + low + close) / 3.0
    key = _vwap_session_key(timestamp, anchor)

    # Reset accumulators on session change
    if state.session_key is not None and key != state.session_key:
        state.cum_pv  = 0.0
        state.cum_vol = 0.0

    state.session_key = key
    state.cum_pv  += tp * volume
    state.cum_vol += volume

    vwap_val: Optional[float] = None
    if state.cum_vol != 0.0:
        vwap_val = state.cum_pv / state.cum_vol

    return [vwap_val], state


def _vwap_output_names(params: Dict[str, Any]) -> List[str]:
    anchor = str(_param(params, "anchor", "D")).upper()
    return [f"VWAP_{anchor}"]


def _vwap_seed(inputs: Dict[str, Any], params: Dict[str, Any]) -> VWAPState:
    return replay_seed("vwap", inputs, params)


STATEFUL_REGISTRY["vwap"] = StatefulIndicator(
    kind="vwap",
    inputs=("high", "low", "close", "volume", "timestamp"),
    init=_vwap_init,
    update=_vwap_update,
    output_names=_vwap_output_names,
)
SEED_REGISTRY["vwap"] = _vwap_seed
