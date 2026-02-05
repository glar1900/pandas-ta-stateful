# -*- coding: utf-8 -*-
"""pandas-ta stateful -- momentum indicators.

Each section follows the pattern:
  1. State dataclass  (if beyond what _base already provides)
  2. init / update / output_names helpers
  3. STATEFUL_REGISTRY["<kind>"] = StatefulIndicator(...)
  4. SEED_REGISTRY["<kind>"]     = seed_fn          (output_only / internal_series)
     (replay_only indicators omit step 4)

Seed-method legend:
  output_only     -- seed_fn extracts the final output value(s) from the
                     vectorised result and reconstructs minimal state.
  internal_series -- seed_fn additionally captures intermediate series so
                     recursive chains stay numerically identical.
  replay_only     -- no seed_fn; initialise only via replay_seed().
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from collections import deque

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
from pandas_ta_stateful.maps import Imports


# ---------------------------------------------------------------------------
# Internal helper: make an MA state factory that respects mamode param.
# Supported: "ema" -> ema_make, "rma"/"smma"/"wilder" -> rma_make, "sma" -> SMA buffer.
# ---------------------------------------------------------------------------

def _ma_state_for_mode(mamode: str, length: int) -> EMAState:
    """Return an EMAState appropriate for the requested mamode."""
    m = mamode.lower()
    if m in ("rma", "smma", "wilder"):
        return rma_make(length, presma=True)
    # default / "ema"
    return ema_make(length, presma=True)


# ===========================================================================
# MACD  (output_only)
# ===========================================================================
# MACD = EMA(close, fast) - EMA(close, slow)
# Signal = EMA(MACD, signal)   -- warmup starts when first MACD value appears
# Hist = MACD - Signal
# Defaults: fast=12, slow=26, signal=9

@dataclass
class MACDState:
    ema_fast: EMAState
    ema_slow: EMAState
    ema_signal: EMAState


def _macd_init(params: Dict[str, Any]) -> MACDState:
    fast   = _as_int(_param(params, "fast",   12), 12)
    slow   = _as_int(_param(params, "slow",   26), 26)
    signal = _as_int(_param(params, "signal",  9),  9)
    if slow < fast:
        fast, slow = slow, fast
    return MACDState(
        ema_fast=ema_make(fast),
        ema_slow=ema_make(slow),
        ema_signal=ema_make(signal),
    )


def _macd_update(
    state: MACDState, bar: Dict[str, Any], params: Dict[str, Any]
) -> Tuple[List[Optional[float]], MACDState]:
    x = bar["close"]

    fast_val, state.ema_fast = ema_update_raw(state.ema_fast, x)
    slow_val, state.ema_slow = ema_update_raw(state.ema_slow, x)

    macd_val: Optional[float] = None
    sig_val:  Optional[float] = None
    hist_val: Optional[float] = None

    if fast_val is not None and slow_val is not None:
        macd_val = fast_val - slow_val
        sig_val, state.ema_signal = ema_update_raw(state.ema_signal, macd_val)
        if sig_val is not None:
            hist_val = macd_val - sig_val

    return [macd_val, hist_val, sig_val], state


def _macd_output_names(params: Dict[str, Any]) -> List[str]:
    fast   = _as_int(_param(params, "fast",   12), 12)
    slow   = _as_int(_param(params, "slow",   26), 26)
    signal = _as_int(_param(params, "signal",  9),  9)
    if slow < fast:
        fast, slow = slow, fast
    p = f"_{fast}_{slow}_{signal}"
    return [f"MACD{p}", f"MACDh{p}", f"MACDs{p}"]


def _macd_seed(series: Dict[str, Any], params: Dict[str, Any]) -> MACDState:
    fast   = _as_int(_param(params, "fast",   12), 12)
    slow   = _as_int(_param(params, "slow",   26), 26)
    signal = _as_int(_param(params, "signal",  9),  9)
    if slow < fast:
        fast, slow = slow, fast
    state = _macd_init(params)
    names = _macd_output_names(params)
    # MACD line
    s = series.get(names[0])
    if s is not None:
        lv = s.dropna()
        if len(lv) > 0:
            # fast EMA: last = fast_val cannot be recovered independently;
            # use replay for exact numerics.  Here we seed signal from output.
            pass
    # Signal line
    s_sig = series.get(names[2])
    if s_sig is not None:
        lv = s_sig.dropna()
        if len(lv) > 0:
            state.ema_signal.last = float(lv.iloc[-1])
            state.ema_signal._warmup_count = state.ema_signal.length
            state.ema_signal._warmup_sum   = 0.0
    # For fast / slow we fall back to replay_seed for full precision.
    # Seed the sub-EMAs via close series if available.
    close_s = series.get("close")
    if close_s is not None:
        # replay fast and slow over the full close series
        fast_st = ema_make(fast)
        slow_st = ema_make(slow)
        sig_st  = ema_make(signal)
        for i in range(len(close_s)):
            import pandas as _pd
            v = close_s.iloc[i]
            if _pd.isna(v):
                continue
            fv, fast_st = ema_update_raw(fast_st, float(v))
            sv, slow_st = ema_update_raw(slow_st, float(v))
            if fv is not None and sv is not None:
                mv = fv - sv
                _, sig_st = ema_update_raw(sig_st, mv)
        state.ema_fast  = fast_st
        state.ema_slow  = slow_st
        state.ema_signal = sig_st
    return state


STATEFUL_REGISTRY["macd"] = StatefulIndicator(
    kind="macd",
    inputs=("close",),
    init=_macd_init,
    update=_macd_update,
    output_names=_macd_output_names,
)
SEED_REGISTRY["macd"] = _macd_seed


# ===========================================================================
# RSI  (internal_series)
# ===========================================================================
# TA-Lib default: delta = close - prev_close
# gain = max(delta, 0), loss = max(-delta, 0)
# avg_gain / avg_loss -> RMA (alpha=1/n, presma=True  =>  SMA seed)
# RSI = 100 * avg_gain / (avg_gain + avg_loss)
# First bar: store prev_close only; no delta.
# Default length=14

@dataclass
class RSIState:
    length: int
    avg_gain: EMAState     # RMA state for gains
    avg_loss: EMAState     # RMA state for losses
    prev_close: Optional[float] = None


def _rsi_init(params: Dict[str, Any]) -> RSIState:
    length = _as_int(_param(params, "length", 14), 14)
    return RSIState(
        length=length,
        avg_gain=rma_make(length, presma=True),
        avg_loss=rma_make(length, presma=True),
    )


def _rsi_update(
    state: RSIState, bar: Dict[str, Any], params: Dict[str, Any]
) -> Tuple[List[Optional[float]], RSIState]:
    close = bar["close"]

    if state.prev_close is None:
        # Very first bar -- no delta yet, just store close.
        state.prev_close = close
        return [None], state

    delta = close - state.prev_close
    state.prev_close = close

    gain = max(delta, 0.0)
    loss = max(-delta, 0.0)

    g_val, state.avg_gain = ema_update_raw(state.avg_gain, gain)
    l_val, state.avg_loss = ema_update_raw(state.avg_loss, loss)

    if g_val is None or l_val is None:
        return [None], state

    denom = g_val + l_val
    rsi_val = 100.0 * g_val / denom if denom != 0.0 else 50.0
    return [rsi_val], state


def _rsi_output_names(params: Dict[str, Any]) -> List[str]:
    length = _as_int(_param(params, "length", 14), 14)
    return [f"RSI_{length}"]


def _rsi_seed(series: Dict[str, Any], params: Dict[str, Any]) -> RSIState:
    """internal_series seed: replay close to reconstruct exact RMA state."""
    state = _rsi_init(params)
    close_s = series.get("close")
    if close_s is not None:
        import pandas as _pd
        for i in range(len(close_s)):
            v = close_s.iloc[i]
            if _pd.isna(v):
                continue
            _, state = _rsi_update(state, {"close": float(v)}, params)
    return state


STATEFUL_REGISTRY["rsi"] = StatefulIndicator(
    kind="rsi",
    inputs=("close",),
    init=_rsi_init,
    update=_rsi_update,
    output_names=_rsi_output_names,
)
SEED_REGISTRY["rsi"] = _rsi_seed


# ===========================================================================
# APO  (output_only)
# ===========================================================================
# APO = MA(close, fast) - MA(close, slow)
# Vectorised source default mamode="sma" but spec says mamode="ema".
# We honour vectorised source default: mamode="ema" per task spec.
# Defaults: fast=12, slow=26

@dataclass
class APOState:
    fast_ma: EMAState
    slow_ma: EMAState


def _apo_init(params: Dict[str, Any]) -> APOState:
    fast    = _as_int(_param(params, "fast",  12), 12)
    slow    = _as_int(_param(params, "slow",  26), 26)
    mamode  = str(_param(params, "mamode", "ema"))
    if slow < fast:
        fast, slow = slow, fast
    return APOState(
        fast_ma=_ma_state_for_mode(mamode, fast),
        slow_ma=_ma_state_for_mode(mamode, slow),
    )


def _apo_update(
    state: APOState, bar: Dict[str, Any], params: Dict[str, Any]
) -> Tuple[List[Optional[float]], APOState]:
    x = bar["close"]
    fv, state.fast_ma = ema_update_raw(state.fast_ma, x)
    sv, state.slow_ma = ema_update_raw(state.slow_ma, x)

    if fv is not None and sv is not None:
        return [fv - sv], state
    return [None], state


def _apo_output_names(params: Dict[str, Any]) -> List[str]:
    fast = _as_int(_param(params, "fast",  12), 12)
    slow = _as_int(_param(params, "slow",  26), 26)
    if slow < fast:
        fast, slow = slow, fast
    return [f"APO_{fast}_{slow}"]


def _apo_seed(series: Dict[str, Any], params: Dict[str, Any]) -> APOState:
    state = _apo_init(params)
    close_s = series.get("close")
    if close_s is not None:
        import pandas as _pd
        for i in range(len(close_s)):
            v = close_s.iloc[i]
            if _pd.isna(v):
                continue
            _, state = _apo_update(state, {"close": float(v)}, params)
    return state


STATEFUL_REGISTRY["apo"] = StatefulIndicator(
    kind="apo",
    inputs=("close",),
    init=_apo_init,
    update=_apo_update,
    output_names=_apo_output_names,
)
SEED_REGISTRY["apo"] = _apo_seed


# ===========================================================================
# PPO  (output_only)
# ===========================================================================
# PPO  = scalar * (MA(fast) - MA(slow)) / MA(slow)
# Signal = EMA(PPO, signal)
# Hist = PPO - Signal
# Vectorised default mamode="sma"; task spec keeps mamode param.
# Defaults: fast=12, slow=26, signal=9, scalar=100  (signal default=10 per spec)

@dataclass
class PPOState:
    fast_ma: EMAState
    slow_ma: EMAState
    signal_ema: EMAState
    scalar: float


def _ppo_init(params: Dict[str, Any]) -> PPOState:
    fast   = _as_int(_param(params, "fast",   12), 12)
    slow   = _as_int(_param(params, "slow",   26), 26)
    signal = _as_int(_param(params, "signal", 10), 10)
    scalar = _as_float(_param(params, "scalar", 100.0), 100.0)
    mamode = str(_param(params, "mamode", "ema"))
    if slow < fast:
        fast, slow = slow, fast
    return PPOState(
        fast_ma=_ma_state_for_mode(mamode, fast),
        slow_ma=_ma_state_for_mode(mamode, slow),
        signal_ema=ema_make(signal),
        scalar=scalar,
    )


def _ppo_update(
    state: PPOState, bar: Dict[str, Any], params: Dict[str, Any]
) -> Tuple[List[Optional[float]], PPOState]:
    x = bar["close"]
    fv, state.fast_ma  = ema_update_raw(state.fast_ma,  x)
    sv, state.slow_ma  = ema_update_raw(state.slow_ma,  x)

    ppo_val:  Optional[float] = None
    sig_val:  Optional[float] = None
    hist_val: Optional[float] = None

    if fv is not None and sv is not None and sv != 0.0:
        ppo_val = state.scalar * (fv - sv) / sv
        sig_val, state.signal_ema = ema_update_raw(state.signal_ema, ppo_val)
        if sig_val is not None:
            hist_val = ppo_val - sig_val

    return [ppo_val, hist_val, sig_val], state


def _ppo_output_names(params: Dict[str, Any]) -> List[str]:
    fast   = _as_int(_param(params, "fast",   12), 12)
    slow   = _as_int(_param(params, "slow",   26), 26)
    signal = _as_int(_param(params, "signal", 10), 10)
    if slow < fast:
        fast, slow = slow, fast
    p = f"_{fast}_{slow}_{signal}"
    return [f"PPO{p}", f"PPOh{p}", f"PPOs{p}"]


def _ppo_seed(series: Dict[str, Any], params: Dict[str, Any]) -> PPOState:
    state = _ppo_init(params)
    close_s = series.get("close")
    if close_s is not None:
        import pandas as _pd
        for i in range(len(close_s)):
            v = close_s.iloc[i]
            if _pd.isna(v):
                continue
            _, state = _ppo_update(state, {"close": float(v)}, params)
    return state


STATEFUL_REGISTRY["ppo"] = StatefulIndicator(
    kind="ppo",
    inputs=("close",),
    init=_ppo_init,
    update=_ppo_update,
    output_names=_ppo_output_names,
)
SEED_REGISTRY["ppo"] = _ppo_seed


# ===========================================================================
# BIAS  (output_only)
# ===========================================================================
# BIAS = (close / MA(close, length)) - 1
# Defaults: mamode="ema", length=26

@dataclass
class BIASState:
    ma_state: EMAState


def _bias_init(params: Dict[str, Any]) -> BIASState:
    length = _as_int(_param(params, "length", 26), 26)
    mamode = str(_param(params, "mamode", "ema"))
    return BIASState(ma_state=_ma_state_for_mode(mamode, length))


def _bias_update(
    state: BIASState, bar: Dict[str, Any], params: Dict[str, Any]
) -> Tuple[List[Optional[float]], BIASState]:
    x = bar["close"]
    ma_val, state.ma_state = ema_update_raw(state.ma_state, x)
    if ma_val is not None and ma_val != 0.0:
        return [x / ma_val - 1.0], state
    return [None], state


def _bias_output_names(params: Dict[str, Any]) -> List[str]:
    length = _as_int(_param(params, "length", 26), 26)
    mamode = str(_param(params, "mamode", "ema"))
    # match vectorised naming: BIAS_EMA_26
    ma_tag = mamode.upper()
    return [f"BIAS_{ma_tag}_{length}"]


def _bias_seed(series: Dict[str, Any], params: Dict[str, Any]) -> BIASState:
    state = _bias_init(params)
    close_s = series.get("close")
    if close_s is not None:
        import pandas as _pd
        for i in range(len(close_s)):
            v = close_s.iloc[i]
            if _pd.isna(v):
                continue
            _, state = _bias_update(state, {"close": float(v)}, params)
    return state


STATEFUL_REGISTRY["bias"] = StatefulIndicator(
    kind="bias",
    inputs=("close",),
    init=_bias_init,
    update=_bias_update,
    output_names=_bias_output_names,
)
SEED_REGISTRY["bias"] = _bias_seed


# ===========================================================================
# DM  (output_only)  -- Directional Movement
# ===========================================================================
# up   = high - prev_high
# dn   = prev_low - low
# DM+  = up  if (up > dn and up > 0) else 0
# DM-  = dn  if (dn > up and dn > 0) else 0
# Output: MA(DM+, length), MA(DM-, length)
# Defaults: mamode="rma", length=14

@dataclass
class DMState:
    length: int
    mamode: str
    use_talib: bool
    dmp_ma: EMAState
    dmn_ma: EMAState
    prev_high: Optional[float] = None
    prev_low:  Optional[float] = None


def _dm_init(params: Dict[str, Any]) -> DMState:
    length = _as_int(_param(params, "length", 14), 14)
    mamode = str(_param(params, "mamode", "rma"))
    use_talib = bool(_param(params, "talib", True)) and Imports.get("talib", False)
    if use_talib:
        mamode = "rma"
    return DMState(
        length=length,
        mamode=mamode,
        use_talib=use_talib,
        dmp_ma=_ma_state_for_mode(mamode, length),
        dmn_ma=_ma_state_for_mode(mamode, length),
    )


def _dm_update(
    state: DMState, bar: Dict[str, Any], params: Dict[str, Any]
) -> Tuple[List[Optional[float]], DMState]:
    high = bar["high"]
    low  = bar["low"]

    if state.prev_high is None:
        # First bar: no previous high/low available.
        state.prev_high = high
        state.prev_low  = low
        # Feed zeros into MA warmup
        _, state.dmp_ma = ema_update_raw(state.dmp_ma, 0.0)
        _, state.dmn_ma = ema_update_raw(state.dmn_ma, 0.0)
        return [None, None], state

    up = high - state.prev_high
    dn = state.prev_low - low
    state.prev_high = high
    state.prev_low  = low

    dmp = up  if (up > dn and up > 0) else 0.0
    dmn = dn  if (dn > up and dn > 0) else 0.0

    p_val, state.dmp_ma = ema_update_raw(state.dmp_ma, dmp)
    n_val, state.dmn_ma = ema_update_raw(state.dmn_ma, dmn)

    if p_val is not None and n_val is not None:
        if state.use_talib and state.mamode.lower() == "rma":
            # TA-Lib PLUS_DM/MINUS_DM returns Wilder-smoothed SUM (not avg).
            p_val = p_val * state.length
            n_val = n_val * state.length
        return [p_val, n_val], state
    return [None, None], state


def _dm_output_names(params: Dict[str, Any]) -> List[str]:
    length = _as_int(_param(params, "length", 14), 14)
    return [f"DMP_{length}", f"DMN_{length}"]


def _dm_seed(series: Dict[str, Any], params: Dict[str, Any]) -> DMState:
    state = _dm_init(params)
    high_s = series.get("high")
    low_s  = series.get("low")
    if high_s is not None and low_s is not None:
        import pandas as _pd
        for i in range(len(high_s)):
            h = high_s.iloc[i]
            l = low_s.iloc[i]
            if _pd.isna(h) or _pd.isna(l):
                continue
            _, state = _dm_update(state, {"high": float(h), "low": float(l)}, params)
    return state


STATEFUL_REGISTRY["dm"] = StatefulIndicator(
    kind="dm",
    inputs=("high", "low"),
    init=_dm_init,
    update=_dm_update,
    output_names=_dm_output_names,
)
SEED_REGISTRY["dm"] = _dm_seed


# ===========================================================================
# KDJ  (output_only)
# ===========================================================================
# fastk = 100 * (close - lowest_low(length)) / (highest_high(length) - lowest_low(length))
# K = RMA(fastk, signal)     -- pd_rma uses alpha=1/n, presma=True
# D = RMA(K,     signal)
# J = 3*K - 2*D
# Defaults: length=9, signal=3
# Needs: high_buf, low_buf (deque maxlen=length) for rolling min/max

@dataclass
class KDJState:
    length: int
    high_buf: deque
    low_buf:  deque
    k_rma: EMAState
    d_rma: EMAState


def _kdj_init(params: Dict[str, Any]) -> KDJState:
    length = _as_int(_param(params, "length", 9), 9)
    signal = _as_int(_param(params, "signal", 3), 3)
    return KDJState(
        length=length,
        high_buf=deque(maxlen=length),
        low_buf =deque(maxlen=length),
        k_rma=rma_make(signal, presma=True),
        d_rma=rma_make(signal, presma=True),
    )


def _kdj_update(
    state: KDJState, bar: Dict[str, Any], params: Dict[str, Any]
) -> Tuple[List[Optional[float]], KDJState]:
    high  = bar["high"]
    low   = bar["low"]
    close = bar["close"]

    state.high_buf.append(high)
    state.low_buf.append(low)

    if len(state.high_buf) < state.length:
        # Not enough bars for rolling window
        return [None, None, None], state

    hh = max(state.high_buf)
    ll = min(state.low_buf)
    rng = hh - ll
    fastk = 100.0 * (close - ll) / rng if rng != 0.0 else 0.0

    k_val, state.k_rma = ema_update_raw(state.k_rma, fastk)
    if k_val is None:
        return [None, None, None], state

    d_val, state.d_rma = ema_update_raw(state.d_rma, k_val)
    if d_val is None:
        return [k_val, None, None], state

    j_val = 3.0 * k_val - 2.0 * d_val
    return [k_val, d_val, j_val], state


def _kdj_output_names(params: Dict[str, Any]) -> List[str]:
    length = _as_int(_param(params, "length", 9), 9)
    signal = _as_int(_param(params, "signal", 3), 3)
    p = f"_{length}_{signal}"
    return [f"K{p}", f"D{p}", f"J{p}"]


def _kdj_seed(series: Dict[str, Any], params: Dict[str, Any]) -> KDJState:
    state = _kdj_init(params)
    high_s  = series.get("high")
    low_s   = series.get("low")
    close_s = series.get("close")
    if high_s is not None and low_s is not None and close_s is not None:
        import pandas as _pd
        for i in range(len(high_s)):
            h = high_s.iloc[i]
            l = low_s.iloc[i]
            c = close_s.iloc[i]
            if _pd.isna(h) or _pd.isna(l) or _pd.isna(c):
                continue
            _, state = _kdj_update(
                state, {"high": float(h), "low": float(l), "close": float(c)}, params
            )
    return state


STATEFUL_REGISTRY["kdj"] = StatefulIndicator(
    kind="kdj",
    inputs=("high", "low", "close"),
    init=_kdj_init,
    update=_kdj_update,
    output_names=_kdj_output_names,
)
SEED_REGISTRY["kdj"] = _kdj_seed


# ===========================================================================
# ERI  (output_only)  -- Elder Ray Index
# ===========================================================================
# bull_power = high - EMA(close, length)
# bear_power = low  - EMA(close, length)
# Default length=13

@dataclass
class ERIState:
    ema_state: EMAState


def _eri_init(params: Dict[str, Any]) -> ERIState:
    length = _as_int(_param(params, "length", 13), 13)
    return ERIState(ema_state=ema_make(length))


def _eri_update(
    state: ERIState, bar: Dict[str, Any], params: Dict[str, Any]
) -> Tuple[List[Optional[float]], ERIState]:
    high  = bar["high"]
    low   = bar["low"]
    close = bar["close"]

    ema_val, state.ema_state = ema_update_raw(state.ema_state, close)
    if ema_val is None:
        return [None, None], state

    bull = high - ema_val
    bear = low  - ema_val
    return [bull, bear], state


def _eri_output_names(params: Dict[str, Any]) -> List[str]:
    length = _as_int(_param(params, "length", 13), 13)
    return [f"BULLP_{length}", f"BEARP_{length}"]


def _eri_seed(series: Dict[str, Any], params: Dict[str, Any]) -> ERIState:
    state = _eri_init(params)
    close_s = series.get("close")
    if close_s is not None:
        import pandas as _pd
        for i in range(len(close_s)):
            v = close_s.iloc[i]
            if _pd.isna(v):
                continue
            ema_val, state.ema_state = ema_update_raw(state.ema_state, float(v))
    return state


STATEFUL_REGISTRY["eri"] = StatefulIndicator(
    kind="eri",
    inputs=("high", "low", "close"),
    init=_eri_init,
    update=_eri_update,
    output_names=_eri_output_names,
)
SEED_REGISTRY["eri"] = _eri_seed


# ===========================================================================
# EXHC  (output_only)  -- Exhaustion Count
# ===========================================================================
# Vectorised: compares close[i] vs close[i-length].
# diff = close - close[length bars ago].
# Stateful equivalent: keep a deque of last `length` closes.
# diff = close - close_buf[0]  (oldest in buffer of size length).
# up:  consecutive count of diff > 0 (reset on diff <= 0)
# dn:  consecutive count of diff < 0 (reset on diff >= 0)
# diff == 0 -> both counters stay (no reset).
# Defaults: length=4, cap=13

@dataclass
class EXHCState:
    length: int
    cap: int
    close_buf: deque
    prev_up: int = 0
    prev_dn: int = 0


def _exhc_init(params: Dict[str, Any]) -> EXHCState:
    length = _as_int(_param(params, "length", 4), 4)
    cap    = _as_int(_param(params, "cap",    13), 13)
    return EXHCState(
        length=length,
        cap=cap,
        close_buf=deque(maxlen=length + 1),
    )


def _exhc_update(
    state: EXHCState, bar: Dict[str, Any], params: Dict[str, Any]
) -> Tuple[List[Optional[float]], EXHCState]:
    close = bar["close"]
    state.close_buf.append(close)

    if len(state.close_buf) <= state.length:
        # Not enough bars yet
        return [0.0, 0.0], state

    # close_buf[0] is close from `length` bars ago
    diff = close - state.close_buf[0]

    if diff > 0:
        state.prev_up = state.prev_up + 1
        state.prev_dn = 0
    elif diff < 0:
        state.prev_dn = state.prev_dn + 1
        state.prev_up = 0
    # diff == 0: both stay unchanged

    up = state.prev_up
    dn = state.prev_dn
    if state.cap > 0:
        up = min(up, state.cap)
        dn = min(dn, state.cap)

    return [float(dn), float(up)], state


def _exhc_output_names(params: Dict[str, Any]) -> List[str]:
    return ["EXHC_DNa", "EXHC_UPa"]


def _exhc_seed(series: Dict[str, Any], params: Dict[str, Any]) -> EXHCState:
    state = _exhc_init(params)
    close_s = series.get("close")
    if close_s is not None:
        import pandas as _pd
        for i in range(len(close_s)):
            v = close_s.iloc[i]
            if _pd.isna(v):
                continue
            _, state = _exhc_update(state, {"close": float(v)}, params)
    return state


STATEFUL_REGISTRY["exhc"] = StatefulIndicator(
    kind="exhc",
    inputs=("close",),
    init=_exhc_init,
    update=_exhc_update,
    output_names=_exhc_output_names,
)
SEED_REGISTRY["exhc"] = _exhc_seed


# ===========================================================================
# TMO  (output_only)  -- Triple MA Oscillator (simplified stateful)
# ===========================================================================
# Stateful interpretation per spec:
#   delta = close - prev_close
#   ema1  = EMA(delta, length)
#   ema2  = EMA(ema1, length)   -- cascaded
#   ema3  = EMA(ema2, length)   -- cascaded  (= main TMO line)
#   smooth = EMA(ema3, length)  -- 4th EMA for the smooth line
# Outputs: main (ema3), smooth (ema_smooth)
# Defaults: length=5  (calc_length from vectorised source)

@dataclass
class TMOState:
    ema1: EMAState
    ema2: EMAState
    ema3: EMAState       # main
    ema_smooth: EMAState # smooth
    prev_close: Optional[float] = None


def _tmo_init(params: Dict[str, Any]) -> TMOState:
    length = _as_int(_param(params, "length", 5), 5)
    return TMOState(
        ema1=ema_make(length),
        ema2=ema_make(length),
        ema3=ema_make(length),
        ema_smooth=ema_make(length),
    )


def _tmo_update(
    state: TMOState, bar: Dict[str, Any], params: Dict[str, Any]
) -> Tuple[List[Optional[float]], TMOState]:
    close = bar["close"]

    if state.prev_close is None:
        state.prev_close = close
        return [None, None], state

    delta = close - state.prev_close
    state.prev_close = close

    v1, state.ema1 = ema_update_raw(state.ema1, delta)
    if v1 is None:
        return [None, None], state

    v2, state.ema2 = ema_update_raw(state.ema2, v1)
    if v2 is None:
        return [None, None], state

    v3, state.ema3 = ema_update_raw(state.ema3, v2)
    if v3 is None:
        return [None, None], state

    vs, state.ema_smooth = ema_update_raw(state.ema_smooth, v3)
    # main = v3, smooth = vs (may still be None during warmup)
    return [v3, vs], state


def _tmo_output_names(params: Dict[str, Any]) -> List[str]:
    length = _as_int(_param(params, "length", 5), 5)
    return [f"TMO_{length}", f"TMOs_{length}"]


def _tmo_seed(series: Dict[str, Any], params: Dict[str, Any]) -> TMOState:
    state = _tmo_init(params)
    close_s = series.get("close")
    if close_s is not None:
        import pandas as _pd
        for i in range(len(close_s)):
            v = close_s.iloc[i]
            if _pd.isna(v):
                continue
            _, state = _tmo_update(state, {"close": float(v)}, params)
    return state


STATEFUL_REGISTRY["tmo"] = StatefulIndicator(
    kind="tmo",
    inputs=("close",),
    init=_tmo_init,
    update=_tmo_update,
    output_names=_tmo_output_names,
)
SEED_REGISTRY["tmo"] = _tmo_seed


# ===========================================================================
# TSI  (output_only)  -- True Strength Index
# ===========================================================================
# diff = close - prev_close
# slow_ema       = EMA(diff,           slow)
# fast_slow_ema  = EMA(slow_ema,       fast)   -- double smoothed momentum
# abs_slow_ema   = EMA(abs(diff),      slow)
# abs_fast_slow  = EMA(abs_slow_ema,   fast)   -- double smoothed abs momentum
# TSI   = scalar * fast_slow_ema / (abs_fast_slow + 1e-10)
# Signal = EMA(TSI, signal)
# Defaults: slow=25, fast=13, signal=13, scalar=100

@dataclass
class TSIState:
    slow_ema1: EMAState          # EMA(diff, slow)
    fast_slow_ema1: EMAState     # EMA(slow_ema1, fast)
    abs_slow_ema1: EMAState      # EMA(|diff|, slow)
    abs_fast_slow_ema1: EMAState # EMA(abs_slow_ema1, fast)
    signal_ema: EMAState         # EMA(TSI, signal)
    scalar: float
    prev_close: Optional[float] = None


def _tsi_init(params: Dict[str, Any]) -> TSIState:
    fast   = _as_int(_param(params, "fast",   13), 13)
    slow   = _as_int(_param(params, "slow",   25), 25)
    signal = _as_int(_param(params, "signal", 13), 13)
    scalar = _as_float(_param(params, "scalar", 100.0), 100.0)
    if slow < fast:
        fast, slow = slow, fast
    return TSIState(
        slow_ema1=ema_make(slow),
        fast_slow_ema1=ema_make(fast),
        abs_slow_ema1=ema_make(slow),
        abs_fast_slow_ema1=ema_make(fast),
        signal_ema=ema_make(signal),
        scalar=scalar,
    )


def _tsi_update(
    state: TSIState, bar: Dict[str, Any], params: Dict[str, Any]
) -> Tuple[List[Optional[float]], TSIState]:
    close = bar["close"]

    if state.prev_close is None:
        state.prev_close = close
        return [None, None], state

    diff = close - state.prev_close
    state.prev_close = close

    # Double smooth of diff
    v_slow, state.slow_ema1 = ema_update_raw(state.slow_ema1, diff)
    if v_slow is not None:
        v_fs, state.fast_slow_ema1 = ema_update_raw(state.fast_slow_ema1, v_slow)
    else:
        v_fs = None

    # Double smooth of |diff|
    v_abs_slow, state.abs_slow_ema1 = ema_update_raw(state.abs_slow_ema1, abs(diff))
    if v_abs_slow is not None:
        v_afs, state.abs_fast_slow_ema1 = ema_update_raw(state.abs_fast_slow_ema1, v_abs_slow)
    else:
        v_afs = None

    if v_fs is None or v_afs is None:
        return [None, None], state

    tsi_val = state.scalar * v_fs / (v_afs + 1e-10)
    sig_val, state.signal_ema = ema_update_raw(state.signal_ema, tsi_val)

    return [tsi_val, sig_val], state


def _tsi_output_names(params: Dict[str, Any]) -> List[str]:
    fast   = _as_int(_param(params, "fast",   13), 13)
    slow   = _as_int(_param(params, "slow",   25), 25)
    signal = _as_int(_param(params, "signal", 13), 13)
    if slow < fast:
        fast, slow = slow, fast
    p = f"_{fast}_{slow}_{signal}"
    return [f"TSI{p}", f"TSIs{p}"]


def _tsi_seed(series: Dict[str, Any], params: Dict[str, Any]) -> TSIState:
    state = _tsi_init(params)
    close_s = series.get("close")
    if close_s is not None:
        import pandas as _pd
        for i in range(len(close_s)):
            v = close_s.iloc[i]
            if _pd.isna(v):
                continue
            _, state = _tsi_update(state, {"close": float(v)}, params)
    return state


STATEFUL_REGISTRY["tsi"] = StatefulIndicator(
    kind="tsi",
    inputs=("close",),
    init=_tsi_init,
    update=_tsi_update,
    output_names=_tsi_output_names,
)
SEED_REGISTRY["tsi"] = _tsi_seed


# ===========================================================================
# TRIX  (internal_series)
# ===========================================================================
# ema1 = EMA(close, length)
# ema2 = EMA(ema1,  length)
# ema3 = EMA(ema2,  length)
# TRIX = scalar * (ema3 - prev_ema3) / prev_ema3   (= scalar * pct_change(ema3))
# Signal = SMA(TRIX, signal)  -- implemented via deque buffer
# Defaults: length=30, signal=9, scalar=100

@dataclass
class TRIXState:
    ema1: EMAState
    ema2: EMAState
    ema3: EMAState
    scalar: float
    signal: int
    prev_ema3: Optional[float] = None
    sig_buf: deque = field(default_factory=deque)  # maxlen=signal


def _trix_init(params: Dict[str, Any]) -> TRIXState:
    length = _as_int(_param(params, "length", 30), 30)
    signal = _as_int(_param(params, "signal",  9),  9)
    scalar = _as_float(_param(params, "scalar", 100.0), 100.0)
    return TRIXState(
        ema1=ema_make(length),
        ema2=ema_make(length),
        ema3=ema_make(length),
        scalar=scalar,
        signal=signal,
        sig_buf=deque(maxlen=signal),
    )


def _trix_update(
    state: TRIXState, bar: Dict[str, Any], params: Dict[str, Any]
) -> Tuple[List[Optional[float]], TRIXState]:
    x = bar["close"]

    v1, state.ema1 = ema_update_raw(state.ema1, x)
    if v1 is None:
        return [None, None], state

    v2, state.ema2 = ema_update_raw(state.ema2, v1)
    if v2 is None:
        return [None, None], state

    v3, state.ema3 = ema_update_raw(state.ema3, v2)
    if v3 is None:
        return [None, None], state

    trix_val: Optional[float] = None
    if state.prev_ema3 is not None and state.prev_ema3 != 0.0:
        trix_val = state.scalar * (v3 - state.prev_ema3) / state.prev_ema3
    state.prev_ema3 = v3

    if trix_val is None:
        return [None, None], state

    state.sig_buf.append(trix_val)
    sig_val: Optional[float] = None
    if len(state.sig_buf) == state.signal:
        sig_val = sum(state.sig_buf) / state.signal

    return [trix_val, sig_val], state


def _trix_output_names(params: Dict[str, Any]) -> List[str]:
    length = _as_int(_param(params, "length", 30), 30)
    signal = _as_int(_param(params, "signal",  9),  9)
    return [f"TRIX_{length}_{signal}", f"TRIXs_{length}_{signal}"]


def _trix_seed(series: Dict[str, Any], params: Dict[str, Any]) -> TRIXState:
    """internal_series: replay close to reconstruct cascaded EMA + SMA buffer."""
    state = _trix_init(params)
    close_s = series.get("close")
    if close_s is not None:
        import pandas as _pd
        for i in range(len(close_s)):
            v = close_s.iloc[i]
            if _pd.isna(v):
                continue
            _, state = _trix_update(state, {"close": float(v)}, params)
    return state


STATEFUL_REGISTRY["trix"] = StatefulIndicator(
    kind="trix",
    inputs=("close",),
    init=_trix_init,
    update=_trix_update,
    output_names=_trix_output_names,
)
SEED_REGISTRY["trix"] = _trix_seed


# ===========================================================================
# CMO  (internal_series)  -- Chande Momentum Oscillator
# ===========================================================================
# delta = close - prev_close
# pos = max(delta, 0)   ->  RMA smoothed
# neg = max(-delta, 0)  ->  RMA smoothed
# CMO = scalar * (pos_avg - neg_avg) / (pos_avg + neg_avg)
# Defaults: length=14, scalar=100

@dataclass
class CMOState:
    pos_rma: EMAState
    neg_rma: EMAState
    scalar: float
    prev_close: Optional[float] = None


def _cmo_init(params: Dict[str, Any]) -> CMOState:
    length = _as_int(_param(params, "length", 14), 14)
    scalar = _as_float(_param(params, "scalar", 100.0), 100.0)
    return CMOState(
        pos_rma=rma_make(length, presma=True),
        neg_rma=rma_make(length, presma=True),
        scalar=scalar,
    )


def _cmo_update(
    state: CMOState, bar: Dict[str, Any], params: Dict[str, Any]
) -> Tuple[List[Optional[float]], CMOState]:
    close = bar["close"]

    if state.prev_close is None:
        state.prev_close = close
        return [None], state

    delta = close - state.prev_close
    state.prev_close = close

    pos = max(delta, 0.0)
    neg = max(-delta, 0.0)

    p_val, state.pos_rma = ema_update_raw(state.pos_rma, pos)
    n_val, state.neg_rma = ema_update_raw(state.neg_rma, neg)

    if p_val is None or n_val is None:
        return [None], state

    denom = p_val + n_val
    cmo_val = state.scalar * (p_val - n_val) / denom if denom != 0.0 else 0.0
    return [cmo_val], state


def _cmo_output_names(params: Dict[str, Any]) -> List[str]:
    length = _as_int(_param(params, "length", 14), 14)
    return [f"CMO_{length}"]


def _cmo_seed(series: Dict[str, Any], params: Dict[str, Any]) -> CMOState:
    state = _cmo_init(params)
    close_s = series.get("close")
    if close_s is not None:
        import pandas as _pd
        for i in range(len(close_s)):
            v = close_s.iloc[i]
            if _pd.isna(v):
                continue
            _, state = _cmo_update(state, {"close": float(v)}, params)
    return state


STATEFUL_REGISTRY["cmo"] = StatefulIndicator(
    kind="cmo",
    inputs=("close",),
    init=_cmo_init,
    update=_cmo_update,
    output_names=_cmo_output_names,
)
SEED_REGISTRY["cmo"] = _cmo_seed


# ===========================================================================
# CRSI  (internal_series)  -- Connors RSI
# ===========================================================================
# Components:
#   1) RSI(close, rsi_length)                  -> reuse RSIState
#   2) streak RSI: RSI(streak, streak_length)  -> second RSIState
#      streak: +N if N consecutive up bars, -N if N consecutive down bars.
#   3) percent_rank = rank(close, rank_length) * 100 / rank_length
#      -> deque buffer of last rank_length closes for rank calculation.
# CRSI = (rsi + streak_rsi + percent_rank) / 3
# Defaults: rsi_length=3, streak_length=2, rank_length=100

@dataclass
class CRSIState:
    rsi_state: RSIState
    streak_rsi_state: RSIState
    rank_length: int
    pct_buf: deque            # maxlen=rank_length+1, stores pct_change values
    streak: int = 0           # current streak value
    prev_close: Optional[float] = None


def _crsi_init(params: Dict[str, Any]) -> CRSIState:
    rsi_length    = _as_int(_param(params, "rsi_length",    3), 3)
    streak_length = _as_int(_param(params, "streak_length", 2), 2)
    rank_length   = _as_int(_param(params, "rank_length", 100), 100)
    return CRSIState(
        rsi_state=RSIState(
            length=rsi_length,
            avg_gain=rma_make(rsi_length, presma=True),
            avg_loss=rma_make(rsi_length, presma=True),
        ),
        streak_rsi_state=RSIState(
            length=streak_length,
            avg_gain=rma_make(streak_length, presma=True),
            avg_loss=rma_make(streak_length, presma=True),
        ),
        rank_length=rank_length,
        pct_buf=deque(maxlen=rank_length + 1),
    )


def _crsi_update(
    state: CRSIState, bar: Dict[str, Any], params: Dict[str, Any]
) -> Tuple[List[Optional[float]], CRSIState]:
    close = bar["close"]
    prev_close = state.prev_close

    # --- Streak calculation ---
    if prev_close is not None:
        if close > prev_close:
            state.streak = (state.streak + 1) if state.streak >= 0 else 1
        elif close < prev_close:
            state.streak = (state.streak - 1) if state.streak <= 0 else -1
        # close == prev_close: streak stays at 0 if was 0, else resets
        else:
            state.streak = 0
    state.prev_close = close

    # --- RSI of close ---
    rsi_out, state.rsi_state = _rsi_update(state.rsi_state, {"close": close}, params)
    rsi_val = rsi_out[0]

    # --- RSI of streak ---
    streak_out, state.streak_rsi_state = _rsi_update(
        state.streak_rsi_state, {"close": float(state.streak)}, params
    )
    streak_rsi_val = streak_out[0]

    # --- Percent rank ---
    pct_val: Optional[float] = None
    if prev_close is not None and prev_close != 0.0:
        pct_val = (close / prev_close) - 1.0
    elif prev_close is not None:
        pct_val = float("nan")
    if pct_val is not None:
        state.pct_buf.append(pct_val)
    pr_val: Optional[float] = None
    if len(state.pct_buf) >= state.rank_length + 1:
        # percent_rank on pct_change window: count of previous values < current
        last = state.pct_buf[-1]
        count = 0
        for v in list(state.pct_buf)[:-1]:
            if v < last:
                count += 1
        pr_val = 100.0 * count / state.rank_length

    # --- Combine ---
    if rsi_val is None or streak_rsi_val is None or pr_val is None:
        return [None], state

    crsi_val = (rsi_val + streak_rsi_val + pr_val) / 3.0
    return [crsi_val], state


def _crsi_output_names(params: Dict[str, Any]) -> List[str]:
    rsi_length    = _as_int(_param(params, "rsi_length",    3), 3)
    streak_length = _as_int(_param(params, "streak_length", 2), 2)
    rank_length   = _as_int(_param(params, "rank_length", 100), 100)
    return [f"CRSI_{rsi_length}_{streak_length}_{rank_length}"]


def _crsi_seed(series: Dict[str, Any], params: Dict[str, Any]) -> CRSIState:
    state = _crsi_init(params)
    close_s = series.get("close")
    if close_s is not None:
        import pandas as _pd
        for i in range(len(close_s)):
            v = close_s.iloc[i]
            if _pd.isna(v):
                continue
            _, state = _crsi_update(state, {"close": float(v)}, params)
    return state


STATEFUL_REGISTRY["crsi"] = StatefulIndicator(
    kind="crsi",
    inputs=("close",),
    init=_crsi_init,
    update=_crsi_update,
    output_names=_crsi_output_names,
)
SEED_REGISTRY["crsi"] = _crsi_seed


# ===========================================================================
# STOCHRSI  (internal_series)
# ===========================================================================
# 1) Compute RSI(close, rsi_length)
# 2) stoch = 100 * (rsi - lowest_rsi(length)) / (highest_rsi(length) - lowest_rsi(length))
#    -> rsi_buf (deque maxlen=length) for rolling min/max
# 3) K = SMA(stoch, k)  -> deque buffer
# 4) D = SMA(K, d)      -> deque buffer
# Defaults: length=14, rsi_length=14, k=3, d=3

@dataclass
class StochRSIState:
    rsi_state: RSIState
    length: int
    rsi_buf: deque           # maxlen=length, stores RSI values
    k: int
    d: int
    stoch_buf: deque         # maxlen=k, for SMA of stoch -> K
    k_buf: deque             # maxlen=d, for SMA of K     -> D


def _stochrsi_init(params: Dict[str, Any]) -> StochRSIState:
    length     = _as_int(_param(params, "length",     14), 14)
    rsi_length = _as_int(_param(params, "rsi_length", 14), 14)
    k          = _as_int(_param(params, "k",           3),  3)
    d          = _as_int(_param(params, "d",           3),  3)
    return StochRSIState(
        rsi_state=RSIState(
            length=rsi_length,
            avg_gain=rma_make(rsi_length, presma=True),
            avg_loss=rma_make(rsi_length, presma=True),
        ),
        length=length,
        rsi_buf=deque(maxlen=length),
        k=k,
        d=d,
        stoch_buf=deque(maxlen=k),
        k_buf=deque(maxlen=d),
    )


def _stochrsi_update(
    state: StochRSIState, bar: Dict[str, Any], params: Dict[str, Any]
) -> Tuple[List[Optional[float]], StochRSIState]:
    close = bar["close"]

    rsi_out, state.rsi_state = _rsi_update(state.rsi_state, {"close": close}, params)
    rsi_val = rsi_out[0]

    if rsi_val is None:
        return [None, None], state

    state.rsi_buf.append(rsi_val)

    if len(state.rsi_buf) < state.length:
        return [None, None], state

    lo = min(state.rsi_buf)
    hi = max(state.rsi_buf)
    rng = hi - lo
    stoch = 100.0 * (rsi_val - lo) / rng if rng != 0.0 else 0.0

    # K = SMA(stoch, k)
    state.stoch_buf.append(stoch)
    if len(state.stoch_buf) < state.k:
        return [None, None], state
    k_val = sum(state.stoch_buf) / state.k

    # D = SMA(K, d)
    state.k_buf.append(k_val)
    if len(state.k_buf) < state.d:
        return [k_val, None], state
    d_val = sum(state.k_buf) / state.d

    return [k_val, d_val], state


def _stochrsi_output_names(params: Dict[str, Any]) -> List[str]:
    length     = _as_int(_param(params, "length",     14), 14)
    rsi_length = _as_int(_param(params, "rsi_length", 14), 14)
    k          = _as_int(_param(params, "k",           3),  3)
    d          = _as_int(_param(params, "d",           3),  3)
    p = f"_{length}_{rsi_length}_{k}_{d}"
    return [f"STOCHRSIk{p}", f"STOCHRSId{p}"]


def _stochrsi_seed(series: Dict[str, Any], params: Dict[str, Any]) -> StochRSIState:
    state = _stochrsi_init(params)
    close_s = series.get("close")
    if close_s is not None:
        import pandas as _pd
        for i in range(len(close_s)):
            v = close_s.iloc[i]
            if _pd.isna(v):
                continue
            _, state = _stochrsi_update(state, {"close": float(v)}, params)
    return state


STATEFUL_REGISTRY["stochrsi"] = StatefulIndicator(
    kind="stochrsi",
    inputs=("close",),
    init=_stochrsi_init,
    update=_stochrsi_update,
    output_names=_stochrsi_output_names,
)
SEED_REGISTRY["stochrsi"] = _stochrsi_seed


# ===========================================================================
# SMI  (internal_series)  -- SMI Ergodic Indicator
# ===========================================================================
# SMI reuses TSI internals.  Outputs: SMI (=TSI value), Signal, Oscillator (SMI - Signal)
# Defaults: fast=5, slow=20, signal=5, scalar=1

@dataclass
class SMIState:
    tsi_state: TSIState


def _smi_init(params: Dict[str, Any]) -> SMIState:
    # Map SMI params into TSI params
    fast   = _as_int(_param(params, "fast",   5), 5)
    slow   = _as_int(_param(params, "slow",  20), 20)
    signal = _as_int(_param(params, "signal", 5), 5)
    scalar = _as_float(_param(params, "scalar", 1.0), 1.0)
    tsi_params: Dict[str, Any] = {
        "fast": fast, "slow": slow, "signal": signal, "scalar": scalar
    }
    return SMIState(tsi_state=_tsi_init(tsi_params))


def _smi_update(
    state: SMIState, bar: Dict[str, Any], params: Dict[str, Any]
) -> Tuple[List[Optional[float]], SMIState]:
    fast   = _as_int(_param(params, "fast",   5), 5)
    slow   = _as_int(_param(params, "slow",  20), 20)
    signal = _as_int(_param(params, "signal", 5), 5)
    scalar = _as_float(_param(params, "scalar", 1.0), 1.0)
    tsi_params: Dict[str, Any] = {
        "fast": fast, "slow": slow, "signal": signal, "scalar": scalar
    }

    tsi_out, state.tsi_state = _tsi_update(state.tsi_state, bar, tsi_params)
    smi_val  = tsi_out[0]   # TSI value
    sig_val  = tsi_out[1]   # Signal

    osc_val: Optional[float] = None
    if smi_val is not None and sig_val is not None:
        osc_val = smi_val - sig_val

    return [smi_val, sig_val, osc_val], state


def _smi_output_names(params: Dict[str, Any]) -> List[str]:
    fast   = _as_int(_param(params, "fast",   5), 5)
    slow   = _as_int(_param(params, "slow",  20), 20)
    signal = _as_int(_param(params, "signal", 5), 5)
    scalar = _as_float(_param(params, "scalar", 1.0), 1.0)
    p = f"_{fast}_{slow}_{signal}_{scalar}"
    return [f"SMI{p}", f"SMIs{p}", f"SMIo{p}"]


def _smi_seed(series: Dict[str, Any], params: Dict[str, Any]) -> SMIState:
    state = _smi_init(params)
    close_s = series.get("close")
    if close_s is not None:
        import pandas as _pd
        for i in range(len(close_s)):
            v = close_s.iloc[i]
            if _pd.isna(v):
                continue
            _, state = _smi_update(state, {"close": float(v)}, params)
    return state


STATEFUL_REGISTRY["smi"] = StatefulIndicator(
    kind="smi",
    inputs=("close",),
    init=_smi_init,
    update=_smi_update,
    output_names=_smi_output_names,
)
SEED_REGISTRY["smi"] = _smi_seed


# ===========================================================================
# INERTIA  (internal_series)
# ===========================================================================
# Inertia = linreg(RVI, length)
# RVI (basic mode): uses EMA of stdev-based up/dn signals.
#   Simplified stateful RVI:
#     diff = close - prev_close
#     if diff > 0: up = stdev(close, rvi_length), dn = 0
#     if diff < 0: up = 0, dn = stdev(close, rvi_length)
#     if diff == 0: up = 0, dn = 0
#     pos_ema = EMA(up, rvi_length)
#     neg_ema = EMA(dn, rvi_length)
#     RVI = 100 * pos_ema / (pos_ema + neg_ema)
#   stdev needs a close buffer of size rvi_length.
# linreg: linear regression over last `length` RVI values.
#   -> rvi_buf (deque maxlen=length), numpy-free manual linreg.
# Defaults: length=20, rvi_length=14

@dataclass
class InertiaState:
    rvi_length: int
    length: int
    close_buf: deque         # maxlen=rvi_length for stdev
    rvi_pos_ema: EMAState
    rvi_neg_ema: EMAState
    rvi_prev_close: Optional[float] = None
    rvi_buf: deque = field(default_factory=deque)  # maxlen=length for linreg


def _inertia_init(params: Dict[str, Any]) -> InertiaState:
    length     = _as_int(_param(params, "length",     20), 20)
    rvi_length = _as_int(_param(params, "rvi_length", 14), 14)
    return InertiaState(
        rvi_length=rvi_length,
        length=length,
        close_buf=deque(maxlen=rvi_length),
        rvi_pos_ema=ema_make(rvi_length),
        rvi_neg_ema=ema_make(rvi_length),
        rvi_buf=deque(maxlen=length),
    )


def _stdev_from_buf(buf: deque) -> float:
    """Population stdev from a deque of floats."""
    n = len(buf)
    if n < 2:
        return 0.0
    mean = sum(buf) / n
    variance = sum((x - mean) ** 2 for x in buf) / n
    return math.sqrt(variance)


def _linreg_from_buf(buf: deque) -> Optional[float]:
    """Linear regression value (last point on fitted line) from a deque.

    Uses the closed-form OLS slope/intercept with x = 0, 1, ..., n-1.
    Returns the fitted value at x = n-1 (the last point).
    """
    n = len(buf)
    if n < 2:
        return float(buf[-1]) if n == 1 else None
    # sum_x = n*(n-1)/2,  sum_x2 = n*(n-1)*(2n-1)/6
    sum_x  = n * (n - 1) / 2.0
    sum_x2 = n * (n - 1) * (2 * n - 1) / 6.0
    sum_y  = 0.0
    sum_xy = 0.0
    for i, y in enumerate(buf):
        sum_y  += y
        sum_xy += i * y
    denom = n * sum_x2 - sum_x * sum_x
    if denom == 0.0:
        return sum_y / n
    slope     = (n * sum_xy - sum_x * sum_y) / denom
    intercept = (sum_y - slope * sum_x) / n
    return intercept + slope * (n - 1)


def _inertia_update(
    state: InertiaState, bar: Dict[str, Any], params: Dict[str, Any]
) -> Tuple[List[Optional[float]], InertiaState]:
    close = bar["close"]
    state.close_buf.append(close)

    # --- RVI calculation ---
    up_val = 0.0
    dn_val = 0.0
    if state.rvi_prev_close is not None:
        diff = close - state.rvi_prev_close
        sd = _stdev_from_buf(state.close_buf)
        if diff > 0:
            up_val = sd
        elif diff < 0:
            dn_val = sd
    state.rvi_prev_close = close

    pos_v, state.rvi_pos_ema = ema_update_raw(state.rvi_pos_ema, up_val)
    neg_v, state.rvi_neg_ema = ema_update_raw(state.rvi_neg_ema, dn_val)

    rvi_val: Optional[float] = None
    if pos_v is not None and neg_v is not None:
        denom = pos_v + neg_v
        rvi_val = 100.0 * pos_v / denom if denom != 0.0 else 50.0

    if rvi_val is None:
        return [None], state

    state.rvi_buf.append(rvi_val)

    # --- Linear regression ---
    if len(state.rvi_buf) < state.length:
        return [None], state

    lr_val = _linreg_from_buf(state.rvi_buf)
    return [lr_val], state


def _inertia_output_names(params: Dict[str, Any]) -> List[str]:
    length     = _as_int(_param(params, "length",     20), 20)
    rvi_length = _as_int(_param(params, "rvi_length", 14), 14)
    return [f"INERTIA_{length}_{rvi_length}"]


def _inertia_seed(series: Dict[str, Any], params: Dict[str, Any]) -> InertiaState:
    state = _inertia_init(params)
    close_s = series.get("close")
    if close_s is not None:
        import pandas as _pd
        for i in range(len(close_s)):
            v = close_s.iloc[i]
            if _pd.isna(v):
                continue
            _, state = _inertia_update(state, {"close": float(v)}, params)
    return state


STATEFUL_REGISTRY["inertia"] = StatefulIndicator(
    kind="inertia",
    inputs=("close",),
    init=_inertia_init,
    update=_inertia_update,
    output_names=_inertia_output_names,
)
SEED_REGISTRY["inertia"] = _inertia_seed


# ===========================================================================
# PGO  (internal_series)  -- Pretty Good Oscillator
# ===========================================================================
# PGO = (close - SMA(close, length)) / EMA(ATR(high, low, close, length), length)
# SMA: deque buffer of last `length` closes.
# ATR: ATRState (Wilder, SMA seed).
# EMA of ATR: EMAState wrapping the ATR output.
# Defaults: length=14

@dataclass
class PGOState:
    length: int
    sma_buf: deque           # maxlen=length
    atr_state: ATRState
    atr_ema: EMAState        # EMA of ATR values


def _pgo_init(params: Dict[str, Any]) -> PGOState:
    length = _as_int(_param(params, "length", 14), 14)
    return PGOState(
        length=length,
        sma_buf=deque(maxlen=length),
        atr_state=ATRState(length=length),
        atr_ema=ema_make(length),
    )


def _pgo_update(
    state: PGOState, bar: Dict[str, Any], params: Dict[str, Any]
) -> Tuple[List[Optional[float]], PGOState]:
    high  = bar["high"]
    low   = bar["low"]
    close = bar["close"]

    # SMA
    state.sma_buf.append(close)
    sma_val: Optional[float] = None
    if len(state.sma_buf) == state.length:
        sma_val = sum(state.sma_buf) / state.length

    # ATR
    atr_val, state.atr_state = atr_update_raw(state.atr_state, high, low, close)

    # EMA of ATR
    atr_ema_val: Optional[float] = None
    if atr_val is not None:
        atr_ema_val, state.atr_ema = ema_update_raw(state.atr_ema, atr_val)

    if sma_val is None or atr_ema_val is None or atr_ema_val == 0.0:
        return [None], state

    pgo_val = (close - sma_val) / atr_ema_val
    return [pgo_val], state


def _pgo_output_names(params: Dict[str, Any]) -> List[str]:
    length = _as_int(_param(params, "length", 14), 14)
    return [f"PGO_{length}"]


def _pgo_seed(series: Dict[str, Any], params: Dict[str, Any]) -> PGOState:
    state = _pgo_init(params)
    high_s  = series.get("high")
    low_s   = series.get("low")
    close_s = series.get("close")
    if high_s is not None and low_s is not None and close_s is not None:
        import pandas as _pd
        for i in range(len(close_s)):
            h = high_s.iloc[i]
            l = low_s.iloc[i]
            c = close_s.iloc[i]
            if _pd.isna(h) or _pd.isna(l) or _pd.isna(c):
                continue
            _, state = _pgo_update(
                state, {"high": float(h), "low": float(l), "close": float(c)}, params
            )
    return state


STATEFUL_REGISTRY["pgo"] = StatefulIndicator(
    kind="pgo",
    inputs=("high", "low", "close"),
    init=_pgo_init,
    update=_pgo_update,
    output_names=_pgo_output_names,
)
SEED_REGISTRY["pgo"] = _pgo_seed


# ===========================================================================
# FISHER  (replay_only)  -- Fisher Transform
# ===========================================================================
# hl2 = (high + low) / 2
# highest_hl2 / lowest_hl2 over rolling window of `length` -> deque buffers
# position = (hl2 - lowest) / max(highest - lowest, 0.001) - 0.5
# v = 0.66 * position + 0.67 * v_prev
# v clamped to (-0.999, 0.999)
# fisher = 0.5 * (ln((1+v)/(1-v)) + fisher_prev)
# signal = fisher shifted by `signal` bars -> keep last `signal` fisher values
# Defaults: length=9, signal=1

@dataclass
class FisherState:
    length: int
    signal: int
    hl2_buf: deque           # maxlen=length for rolling min/max of hl2
    v_prev: float = 0.0
    fisher_prev: float = 0.0
    fisher_buf: deque = field(default_factory=deque)  # maxlen=signal for signal line


def _fisher_init(params: Dict[str, Any]) -> FisherState:
    length = _as_int(_param(params, "length", 9), 9)
    signal = _as_int(_param(params, "signal", 1), 1)
    return FisherState(
        length=length,
        signal=signal,
        hl2_buf=deque(maxlen=length),
        fisher_buf=deque(maxlen=signal),
    )


def _fisher_update(
    state: FisherState, bar: Dict[str, Any], params: Dict[str, Any]
) -> Tuple[List[Optional[float]], FisherState]:
    high = bar["high"]
    low  = bar["low"]
    hl2  = (high + low) / 2.0

    state.hl2_buf.append(hl2)

    if len(state.hl2_buf) < state.length:
        state.fisher_buf.append(state.fisher_prev)
        return [None, None], state

    highest = max(state.hl2_buf)
    lowest  = min(state.hl2_buf)
    hlr = highest - lowest
    if hlr < 0.001:
        hlr = 0.001

    position = (hl2 - lowest) / hlr - 0.5

    v = 0.66 * position + 0.67 * state.v_prev
    if v < -0.99:
        v = -0.999
    if v > 0.99:
        v = 0.999
    state.v_prev = v

    fisher_val = 0.5 * (math.log((1.0 + v) / (1.0 - v)) + state.fisher_prev)
    state.fisher_prev = fisher_val

    # Signal = fisher[i - signal]  (simple shift)
    # fisher_buf stores previous fisher values; signal line = fisher_buf[0] when full
    sig_val: Optional[float] = None
    if len(state.fisher_buf) >= state.signal:
        sig_val = state.fisher_buf[0]  # oldest = signal bars ago
    state.fisher_buf.append(fisher_val)

    return [fisher_val, sig_val], state


def _fisher_output_names(params: Dict[str, Any]) -> List[str]:
    length = _as_int(_param(params, "length", 9), 9)
    signal = _as_int(_param(params, "signal", 1), 1)
    p = f"_{length}_{signal}"
    return [f"FISHERT{p}", f"FISHERTs{p}"]


STATEFUL_REGISTRY["fisher"] = StatefulIndicator(
    kind="fisher",
    inputs=("high", "low"),
    init=_fisher_init,
    update=_fisher_update,
    output_names=_fisher_output_names,
)
# replay_only: no SEED_REGISTRY entry


# ===========================================================================
# QQE  (replay_only)  -- Quantitative Qualitative Estimation
# ===========================================================================
# 1) RSI(close, length)
# 2) rsi_ma = EMA(RSI, smooth)
# 3) rsi_ma_tr = |rsi_ma - prev_rsi_ma|  (abs true range of rsi_ma)
# 4) smoothed_rsi_tr_ma = EMA(rsi_ma_tr, wilders_length)   wilders_length = 2*length-1
# 5) dar = factor * EMA(smoothed_rsi_tr_ma, wilders_length)
# 6) upperband = rsi_ma + dar,  lowerband = rsi_ma - dar
# 7) Loop logic for long/short/trend/qqe lines.
# Outputs: qqe, rsi_ma, qqe_long, qqe_short
# Defaults: length=14, smooth=5, factor=4.236

@dataclass
class QQEState:
    rsi_state: RSIState
    rsi_ma_ema: EMAState           # EMA(RSI, smooth)
    tr_ema1: EMAState              # EMA(|rsi_ma_tr|, wilders_length)
    tr_ema2: EMAState              # EMA(tr_ema1, wilders_length)
    factor: float
    prev_rsi_ma: Optional[float] = None
    # Loop state
    prev_long: float = 0.0
    prev_short: float = 0.0
    prev_prev_long: float = 0.0    # long[i-2]
    prev_prev_short: float = 0.0   # short[i-2]
    prev_rsi_ma_val: Optional[float] = None   # rsi_ma[i-1]
    prev_prev_rsi_ma: Optional[float] = None  # rsi_ma[i-2]
    trend: int = 1


def _qqe_init(params: Dict[str, Any]) -> QQEState:
    length  = _as_int(_param(params, "length",  14), 14)
    smooth  = _as_int(_param(params, "smooth",   5),  5)
    factor  = _as_float(_param(params, "factor", 4.236), 4.236)
    wilders_length = 2 * length - 1
    return QQEState(
        rsi_state=RSIState(
            length=length,
            avg_gain=rma_make(length, presma=True),
            avg_loss=rma_make(length, presma=True),
        ),
        rsi_ma_ema=ema_make(smooth),
        tr_ema1=ema_make(wilders_length),
        tr_ema2=ema_make(wilders_length),
        factor=factor,
    )


def _qqe_update(
    state: QQEState, bar: Dict[str, Any], params: Dict[str, Any]
) -> Tuple[List[Optional[float]], QQEState]:
    close = bar["close"]

    # RSI
    rsi_out, state.rsi_state = _rsi_update(state.rsi_state, {"close": close}, params)
    rsi_val = rsi_out[0]
    if rsi_val is None:
        return [None, None, None, None], state

    # RSI MA
    rsi_ma_val, state.rsi_ma_ema = ema_update_raw(state.rsi_ma_ema, rsi_val)
    if rsi_ma_val is None:
        return [None, None, None, None], state

    # RSI MA True Range
    rsi_ma_tr: Optional[float] = None
    if state.prev_rsi_ma is not None:
        rsi_ma_tr = abs(rsi_ma_val - state.prev_rsi_ma)
    state.prev_rsi_ma = rsi_ma_val

    if rsi_ma_tr is None:
        return [None, None, None, None], state

    # Double-smooth TR
    v1, state.tr_ema1 = ema_update_raw(state.tr_ema1, rsi_ma_tr)
    if v1 is None:
        return [None, None, None, None], state
    v2, state.tr_ema2 = ema_update_raw(state.tr_ema2, v1)
    if v2 is None:
        return [None, None, None, None], state

    dar = state.factor * v2
    upperband = rsi_ma_val + dar
    lowerband = rsi_ma_val - dar

    # --- Loop logic (mirrors vectorised qqe.py lines 104-137) ---
    c_rsi  = rsi_ma_val
    p_rsi  = state.prev_rsi_ma_val if state.prev_rsi_ma_val is not None else rsi_ma_val
    c_long = state.prev_long
    p_long = state.prev_prev_long
    c_short = state.prev_short
    p_short = state.prev_prev_short

    # Long Line
    if p_rsi > c_long and c_rsi > c_long:
        new_long = max(c_long, lowerband)
    else:
        new_long = lowerband

    # Short Line
    if p_rsi < c_short and c_rsi < c_short:
        new_short = min(c_short, upperband)
    else:
        new_short = upperband

    # Trend & QQE
    qqe_val: Optional[float] = None
    qqe_long: Optional[float] = None
    qqe_short: Optional[float] = None

    if (c_rsi > c_short and p_rsi < p_short) or \
       (c_rsi <= c_short and p_rsi >= p_short):
        state.trend = 1
        qqe_val = new_long
        qqe_long = new_long
    elif (c_rsi > c_long and p_rsi < p_long) or \
         (c_rsi <= c_long and p_rsi >= p_long):
        state.trend = -1
        qqe_val = new_short
        qqe_short = new_short
    else:
        if state.trend == 1:
            qqe_val = new_long
            qqe_long = new_long
        else:
            qqe_val = new_short
            qqe_short = new_short

    # Shift history
    state.prev_prev_long  = state.prev_long
    state.prev_prev_short = state.prev_short
    state.prev_long  = new_long
    state.prev_short = new_short
    state.prev_prev_rsi_ma = state.prev_rsi_ma_val
    state.prev_rsi_ma_val  = rsi_ma_val

    return [qqe_val, rsi_ma_val, qqe_long, qqe_short], state


def _qqe_output_names(params: Dict[str, Any]) -> List[str]:
    length = _as_int(_param(params, "length",  14), 14)
    smooth = _as_int(_param(params, "smooth",   5),  5)
    factor = _as_float(_param(params, "factor", 4.236), 4.236)
    p = f"_{length}_{smooth}_{factor}"
    return [f"QQE{p}", f"QQE{p}_RSIMA", f"QQEl{p}", f"QQEs{p}"]


STATEFUL_REGISTRY["qqe"] = StatefulIndicator(
    kind="qqe",
    inputs=("close",),
    init=_qqe_init,
    update=_qqe_update,
    output_names=_qqe_output_names,
)
# replay_only: no SEED_REGISTRY entry


# ===========================================================================
# RSX  (replay_only)  -- Jurik Relative Strength Xtra
# ===========================================================================
# Extremely stateful Jurik filter. All internal variables are scalar accumulators.
# Faithfully translated from the vectorised loop in rsx.py.
# Defaults: length=14

@dataclass
class RSXState:
    length: int
    # Jurik internal scalars
    vC: float = 0.0
    v1C: float = 0.0
    v4: float = 0.0
    v8: float = 0.0
    v10: float = 0.0
    v14: float = 0.0
    v18: float = 0.0
    v20: float = 0.0
    f0: float = 0.0
    f8: float = 0.0
    f10: float = 0.0
    f18: float = 0.0
    f20: float = 0.0
    f28: float = 0.0
    f30: float = 0.0
    f38: float = 0.0
    f40: float = 0.0
    f48: float = 0.0
    f50: float = 0.0
    f58: float = 0.0
    f60: float = 0.0
    f68: float = 0.0
    f70: float = 0.0
    f78: float = 0.0
    f80: float = 0.0
    f88: float = 0.0
    f90: float = 0.0
    _bar_count: int = 0      # total bars seen (0-based index equivalent)


def _rsx_init(params: Dict[str, Any]) -> RSXState:
    length = _as_int(_param(params, "length", 14), 14)
    return RSXState(length=length)


def _rsx_update(
    state: RSXState, bar: Dict[str, Any], params: Dict[str, Any]
) -> Tuple[List[Optional[float]], RSXState]:
    close = bar["close"]
    length = state.length

    # The vectorised source emits nan for indices 0..length-2, then 50 at index length-1.
    # The loop body runs from index=length onwards.
    # We map: _bar_count tracks how many bars we have seen (0-indexed).
    i = state._bar_count
    state._bar_count += 1

    if i < length - 1:
        # Pre-warmup: no output
        return [None], state

    if i == length - 1:
        # Seed bar: emit 50, initialise f90=0 so next bar triggers the f90==0 branch
        return [50.0], state

    # i >= length  ->  main loop body
    if state.f90 == 0:
        state.f90 = 1.0
        state.f0  = 0.0
        if length - 1.0 >= 5:
            state.f88 = float(length - 1)
        else:
            state.f88 = 5.0
        state.f8  = 100.0 * close
        state.f18 = 3.0 / (length + 2.0)
        state.f20 = 1.0 - state.f18
    else:
        if state.f88 <= state.f90:
            state.f90 = state.f88 + 1
        else:
            state.f90 = state.f90 + 1

        state.f10 = state.f8
        state.f8  = 100.0 * close
        state.v8  = state.f8 - state.f10

        state.f28 = state.f20 * state.f28 + state.f18 * state.v8
        state.f30 = state.f18 * state.f28 + state.f20 * state.f30
        state.vC  = 1.5 * state.f28 - 0.5 * state.f30

        state.f38 = state.f20 * state.f38 + state.f18 * state.vC
        state.f40 = state.f18 * state.f38 + state.f20 * state.f40
        state.v10 = 1.5 * state.f38 - 0.5 * state.f40

        state.f48 = state.f20 * state.f48 + state.f18 * state.v10
        state.f50 = state.f18 * state.f48 + state.f20 * state.f50
        state.v14 = 1.5 * state.f48 - 0.5 * state.f50

        state.f58 = state.f20 * state.f58 + state.f18 * abs(state.v8)
        state.f60 = state.f18 * state.f58 + state.f20 * state.f60
        state.v18 = 1.5 * state.f58 - 0.5 * state.f60

        state.f68 = state.f20 * state.f68 + state.f18 * state.v18
        state.f70 = state.f18 * state.f68 + state.f20 * state.f70
        state.v1C = 1.5 * state.f68 - 0.5 * state.f70

        state.f78 = state.f20 * state.f78 + state.f18 * state.v1C
        state.f80 = state.f18 * state.f78 + state.f20 * state.f80
        state.v20 = 1.5 * state.f78 - 0.5 * state.f80

        if state.f88 >= state.f90 and state.f8 != state.f10:
            state.f0 = 1.0
        if state.f88 == state.f90 and state.f0 == 0.0:
            state.f90 = 0.0

    # Output
    if state.f88 < state.f90 and state.v20 > 0.0000000001:
        v4 = (state.v14 / state.v20 + 1.0) * 50.0
        if v4 > 100.0:
            v4 = 100.0
        if v4 < 0.0:
            v4 = 0.0
    else:
        v4 = 50.0
    state.v4 = v4

    return [v4], state


def _rsx_output_names(params: Dict[str, Any]) -> List[str]:
    length = _as_int(_param(params, "length", 14), 14)
    return [f"RSX_{length}"]


STATEFUL_REGISTRY["rsx"] = StatefulIndicator(
    kind="rsx",
    inputs=("close",),
    init=_rsx_init,
    update=_rsx_update,
    output_names=_rsx_output_names,
)
# replay_only: no SEED_REGISTRY entry


# ===========================================================================
# STC  (replay_only)  -- Schaff Trend Cycle
# ===========================================================================
# 1) seed = EMA(close, fast) - EMA(close, slow)  (= MACD line)
# 2) schaff_tc loop:
#      stoch1[i] = 100*(seed[i]-lowest_seed(tc_length)) / range   (or prev if lowest<=0)
#      pf[i]     = pf[i-1] + factor*(stoch1[i]-pf[i-1])
#      stoch2[i] = 100*(pf[i]-lowest_pf(tc_length)) / pf_range
#      pff[i]    = pff[i-1] + factor*(stoch2[i]-pff[i-1])
# Outputs: STC (=pff), STCmacd (=seed), STCstoch (=pf)
# Defaults: tc_length=10, fast=12, slow=26, factor=0.5

@dataclass
class STCState:
    ema_fast: EMAState
    ema_slow: EMAState
    tc_length: int
    factor: float
    # Buffers for rolling min/max
    seed_buf: deque          # maxlen=tc_length for seed (MACD) values
    pf_buf: deque            # maxlen=tc_length for pf values
    # Loop accumulators
    stoch1_prev: float = 0.0
    pf_prev: float = 0.0
    stoch2_prev: float = 0.0
    pff_prev: float = 0.0


def _stc_init(params: Dict[str, Any]) -> STCState:
    fast      = _as_int(_param(params, "fast",      12), 12)
    slow      = _as_int(_param(params, "slow",      26), 26)
    tc_length = _as_int(_param(params, "tc_length", 10), 10)
    factor    = _as_float(_param(params, "factor",  0.5), 0.5)
    if slow < fast:
        fast, slow = slow, fast
    return STCState(
        ema_fast=ema_make(fast),
        ema_slow=ema_make(slow),
        tc_length=tc_length,
        factor=factor,
        seed_buf=deque(maxlen=tc_length),
        pf_buf=deque(maxlen=tc_length),
    )


def _stc_update(
    state: STCState, bar: Dict[str, Any], params: Dict[str, Any]
) -> Tuple[List[Optional[float]], STCState]:
    close = bar["close"]

    fv, state.ema_fast = ema_update_raw(state.ema_fast, close)
    sv, state.ema_slow = ema_update_raw(state.ema_slow, close)

    if fv is None or sv is None:
        return [None, None, None], state

    seed_val = fv - sv  # MACD line
    state.seed_buf.append(seed_val)

    # stoch1 calculation
    if len(state.seed_buf) < state.tc_length:
        # Not enough data for full window; use available
        lowest_seed = min(state.seed_buf)
        highest_seed = max(state.seed_buf)
    else:
        lowest_seed  = min(state.seed_buf)
        highest_seed = max(state.seed_buf)

    seed_range = highest_seed - lowest_seed
    if lowest_seed > 0 and seed_range != 0.0:
        stoch1 = 100.0 * (seed_val - lowest_seed) / seed_range
    else:
        stoch1 = state.stoch1_prev
    state.stoch1_prev = stoch1

    # pf (smoothed stoch1)
    pf = round(state.pf_prev + state.factor * (stoch1 - state.pf_prev), 8)
    state.pf_prev = pf
    state.pf_buf.append(pf)

    # stoch2 calculation
    if len(state.pf_buf) < state.tc_length:
        lowest_pf  = min(state.pf_buf)
        highest_pf = max(state.pf_buf)
    else:
        lowest_pf  = min(state.pf_buf)
        highest_pf = max(state.pf_buf)

    pf_range = highest_pf - lowest_pf
    if pf_range <= 0:
        pf_range = 1.0

    if pf_range > 0:
        stoch2 = 100.0 * (pf - lowest_pf) / pf_range
    else:
        stoch2 = state.stoch2_prev
    state.stoch2_prev = stoch2

    # pff (smoothed stoch2) = STC output
    pff = round(state.pff_prev + state.factor * (stoch2 - state.pff_prev), 8)
    state.pff_prev = pff

    return [pff, seed_val, pf], state


def _stc_output_names(params: Dict[str, Any]) -> List[str]:
    tc_length = _as_int(_param(params, "tc_length", 10), 10)
    fast      = _as_int(_param(params, "fast",      12), 12)
    slow      = _as_int(_param(params, "slow",      26), 26)
    factor    = _as_float(_param(params, "factor",  0.5), 0.5)
    if slow < fast:
        fast, slow = slow, fast
    p = f"_{tc_length}_{fast}_{slow}_{factor}"
    return [f"STC{p}", f"STCmacd{p}", f"STCstoch{p}"]


STATEFUL_REGISTRY["stc"] = StatefulIndicator(
    kind="stc",
    inputs=("close",),
    init=_stc_init,
    update=_stc_update,
    output_names=_stc_output_names,
)
# replay_only: no SEED_REGISTRY entry
