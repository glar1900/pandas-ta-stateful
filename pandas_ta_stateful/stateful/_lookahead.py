# -*- coding: utf-8 -*-
"""pandas-ta stateful -- lookahead indicators (conditional / approximate).

This module implements indicators that typically use future data (lookahead)
but are made streaming-compatible through restrictions or approximations:

  1. dpo        -- CONDITIONAL: centered=False (no lookahead shift)
  2. ichimoku   -- CONDITIONAL: lookahead=False (exclude chikou span)
  3. tos_stdevall -- APPROXIMATE: expanding window with runtime warning
  4. vp         -- APPROXIMATE: expanding window Volume Profile with warning

Each section follows the pattern:
  1. State dataclass
  2. init / update / output_names helpers
  3. LOOKAHEAD_REGISTRY["<kind>"] = StatefulIndicator(...)
  4. SEED_REGISTRY["<kind>"] = seed_fn (where applicable)
"""
from __future__ import annotations

import warnings
from collections import deque
from dataclasses import dataclass, field
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
    LOOKAHEAD_REGISTRY,
    replay_seed,
)


# ===========================================================================
# DPO  (CONDITIONAL lookahead: centered=False ONLY)
# ===========================================================================
# Detrended Price Oscillator: close - SMA(close, length).shift(t)
# where t = int(0.5 * length) + 1.
# The vectorized version allows centered=True (uses future data via shift(-t)).
# Stateful version FIXES centered=False (no lookahead).

@dataclass
class DPOState:
    """DPO state with fixed centered=False.

    RESTRICTION: centered=False is FIXED in stateful mode to avoid lookahead.
    The centered=True variant uses close.shift(t) - ma which looks into the future.
    Only the centered=False variant (close - ma.shift(t)) is streaming-compatible.
    """
    length: int
    t: int                           # shift amount = int(0.5*length)+1
    sma_sum: float = 0.0
    sma_count: int = 0
    sma_buffer: deque = field(default_factory=deque)  # rolling window for SMA
    ma_buffer: deque = field(default_factory=deque)   # shifted MA values


def _dpo_init(params: Dict[str, Any]) -> DPOState:
    """Initialize DPO state. ALWAYS uses centered=False."""
    length = _as_int(_param(params, "length", 20), 20)
    t = int(0.5 * length) + 1
    return DPOState(
        length=length,
        t=t,
        sma_buffer=deque(maxlen=length),
        ma_buffer=deque(maxlen=t),
    )


def _dpo_update(
    state: DPOState, bar: Dict[str, Any], params: Dict[str, Any]
) -> Tuple[List[Optional[float]], DPOState]:
    """Update DPO: close - SMA(close, length).shift(t).

    Note: centered=False is FIXED in stateful mode to avoid lookahead.
    """
    close = bar["close"]

    # Update SMA rolling window
    # Track oldest value before appending (deque with maxlen auto-removes)
    oldest: Optional[float] = None
    if len(state.sma_buffer) >= state.length:
        oldest = state.sma_buffer[0]

    state.sma_buffer.append(close)
    state.sma_sum += close
    state.sma_count += 1

    # Remove oldest value from sum if buffer was full
    if oldest is not None:
        state.sma_sum -= oldest
        state.sma_count = state.length

    # Calculate current SMA
    if state.sma_count >= state.length:
        current_ma = state.sma_sum / state.length
    else:
        current_ma = None

    # Add MA to shift buffer
    if current_ma is not None:
        state.ma_buffer.append(current_ma)

    # DPO = close - shifted_ma
    # We need MA from t bars ago
    dpo_val: Optional[float] = None
    if len(state.ma_buffer) >= state.t:
        shifted_ma = state.ma_buffer[0]  # oldest value in the t-length buffer
        dpo_val = close - shifted_ma

    return [dpo_val], state


def _dpo_output_names(params: Dict[str, Any]) -> List[str]:
    length = _as_int(_param(params, "length", 20), 20)
    return [f"DPO_{length}"]


def _dpo_seed(series: Dict[str, Any], params: Dict[str, Any]) -> DPOState:
    """Seed DPO from historical data.

    For precise state, we need the SMA buffer and the shifted MA buffer.
    Since this is complex, we use replay_seed approach.
    """
    # DPO is effectively replay_only for accurate seeding
    # because we need both the SMA window and the shift buffer
    return replay_seed("dpo", series, params)


LOOKAHEAD_REGISTRY["dpo"] = StatefulIndicator(
    kind="dpo",
    inputs=("close",),
    init=_dpo_init,
    update=_dpo_update,
    output_names=_dpo_output_names,
)
SEED_REGISTRY["dpo"] = _dpo_seed


# ===========================================================================
# ICHIMOKU  (CONDITIONAL lookahead: lookahead=False, exclude chikou)
# ===========================================================================
# Ichimoku Kinko Hyo with forward-looking spans.
# The chikou span uses close.shift(-kijun+1) which is lookahead.
# Stateful version FIXES lookahead=False (exclude chikou span).
# Outputs: tenkan_sen, kijun_sen, span_a, span_b (4 outputs)

@dataclass
class IchimokuState:
    """Ichimoku state without chikou span (lookahead=False).

    RESTRICTION: lookahead=False is FIXED in stateful mode to avoid lookahead.
    The chikou span uses close.shift(-kijun+1) which looks into the future.
    Only the 4 components (tenkan, kijun, span_a, span_b) are streaming-compatible.
    The chikou span is EXCLUDED from stateful output.
    """
    tenkan: int
    kijun: int
    senkou: int
    high_tenkan: deque = field(default_factory=deque)
    low_tenkan: deque = field(default_factory=deque)
    high_kijun: deque = field(default_factory=deque)
    low_kijun: deque = field(default_factory=deque)
    high_senkou: deque = field(default_factory=deque)
    low_senkou: deque = field(default_factory=deque)
    tenkan_buffer: deque = field(default_factory=deque)  # for span_a shift
    kijun_buffer: deque = field(default_factory=deque)   # for span_a shift
    span_b_buffer: deque = field(default_factory=deque)  # for span_b shift


def _ichimoku_init(params: Dict[str, Any]) -> IchimokuState:
    """Initialize Ichimoku state. ALWAYS uses lookahead=False (no chikou)."""
    tenkan = _as_int(_param(params, "tenkan", 9), 9)
    kijun = _as_int(_param(params, "kijun", 26), 26)
    senkou = _as_int(_param(params, "senkou", 52), 52)

    return IchimokuState(
        tenkan=tenkan,
        kijun=kijun,
        senkou=senkou,
        high_tenkan=deque(maxlen=tenkan),
        low_tenkan=deque(maxlen=tenkan),
        high_kijun=deque(maxlen=kijun),
        low_kijun=deque(maxlen=kijun),
        high_senkou=deque(maxlen=senkou),
        low_senkou=deque(maxlen=senkou),
        tenkan_buffer=deque(maxlen=kijun - 1),
        kijun_buffer=deque(maxlen=kijun - 1),
        span_b_buffer=deque(maxlen=kijun - 1),
    )


def _ichimoku_update(
    state: IchimokuState, bar: Dict[str, Any], params: Dict[str, Any]
) -> Tuple[List[Optional[float]], IchimokuState]:
    """Update Ichimoku without chikou span (lookahead=False).

    Returns: [tenkan_sen, kijun_sen, span_a, span_b]

    Note: lookahead=False is FIXED in stateful mode to avoid future data.
          Chikou span is excluded.
    """
    high = bar["high"]
    low = bar["low"]

    # Update rolling windows
    state.high_tenkan.append(high)
    state.low_tenkan.append(low)
    state.high_kijun.append(high)
    state.low_kijun.append(low)
    state.high_senkou.append(high)
    state.low_senkou.append(low)

    # Calculate midprice for each period
    tenkan_sen: Optional[float] = None
    if len(state.high_tenkan) >= state.tenkan:
        tenkan_sen = 0.5 * (max(state.high_tenkan) + min(state.low_tenkan))

    kijun_sen: Optional[float] = None
    if len(state.high_kijun) >= state.kijun:
        kijun_sen = 0.5 * (max(state.high_kijun) + min(state.low_kijun))

    span_b_current: Optional[float] = None
    if len(state.high_senkou) >= state.senkou:
        span_b_current = 0.5 * (max(state.high_senkou) + min(state.low_senkou))

    # Span A = average of tenkan and kijun (unshifted)
    span_a_current: Optional[float] = None
    if tenkan_sen is not None and kijun_sen is not None:
        span_a_current = 0.5 * (tenkan_sen + kijun_sen)

    # Store current values for shifting
    if tenkan_sen is not None:
        state.tenkan_buffer.append(tenkan_sen)
    if kijun_sen is not None:
        state.kijun_buffer.append(kijun_sen)
    if span_b_current is not None:
        state.span_b_buffer.append(span_b_current)

    # Output shifted spans (shift forward by kijun-1)
    # This means we output values calculated kijun-1 bars ago
    span_a: Optional[float] = None
    span_b: Optional[float] = None

    if len(state.tenkan_buffer) >= state.kijun - 1 and len(state.kijun_buffer) >= state.kijun - 1:
        # Get values from kijun-1 bars ago
        old_tenkan = state.tenkan_buffer[0]
        old_kijun = state.kijun_buffer[0]
        span_a = 0.5 * (old_tenkan + old_kijun)

    if len(state.span_b_buffer) >= state.kijun - 1:
        span_b = state.span_b_buffer[0]

    return [tenkan_sen, kijun_sen, span_a, span_b], state


def _ichimoku_output_names(params: Dict[str, Any]) -> List[str]:
    tenkan = _as_int(_param(params, "tenkan", 9), 9)
    kijun = _as_int(_param(params, "kijun", 26), 26)
    return [
        f"ITS_{tenkan}",      # Tenkan Sen
        f"IKS_{kijun}",       # Kijun Sen
        f"ISA_{tenkan}",      # Span A (shifted)
        f"ISB_{kijun}",       # Span B (shifted)
    ]


def _ichimoku_seed(series: Dict[str, Any], params: Dict[str, Any]) -> IchimokuState:
    """Seed Ichimoku from historical data.

    Requires replay to build all rolling windows and shift buffers.
    """
    return replay_seed("ichimoku", series, params)


LOOKAHEAD_REGISTRY["ichimoku"] = StatefulIndicator(
    kind="ichimoku",
    inputs=("high", "low", "close"),
    init=_ichimoku_init,
    update=_ichimoku_update,
    output_names=_ichimoku_output_names,
)
SEED_REGISTRY["ichimoku"] = _ichimoku_seed


# ===========================================================================
# TOS_STDEVALL  (APPROXIMATE: online regression with warning)
# ===========================================================================
# TD Ameritrade's Think or Swim Standard Deviation All
# Original uses all bars or last N bars for linear regression + std bands.
# Stateful version uses ONLINE regression/std (expanding or rolling).

@dataclass
class TosStdevallState:
    """TOS_STDEVALL state using O(1) online regression/std.

    Uses expanding window by default. If length is provided, uses a rolling
    window of that size. This still differs from vectorized snapshot behavior
    but is suitable for streaming.
    """
    stds: Tuple[int, ...]
    ddof: int
    length: Optional[int] = None
    # running sums
    n: int = 0
    sum_x: float = 0.0
    sum_xx: float = 0.0
    sum_y: float = 0.0
    sum_y2: float = 0.0
    sum_xy: float = 0.0
    # rolling buffer (only when length is set)
    y_buf: deque = field(default_factory=deque)
    warned: bool = False


def _tos_stdevall_init(params: Dict[str, Any]) -> TosStdevallState:
    """Initialize TOS_STDEVALL with online regression."""
    stds_param = params.get("stds", [1, 2, 3])
    if not isinstance(stds_param, (list, tuple)):
        stds_param = [1, 2, 3]
    stds = tuple(sorted(stds_param))
    ddof = _as_int(_param(params, "ddof", 1), 1)
    length = _param(params, "length", None)
    length = _as_int(length, 0) if length is not None else None
    if length is not None and length <= 1:
        length = None

    state = TosStdevallState(stds=stds, ddof=ddof, length=length)
    if length:
        state.y_buf = deque(maxlen=length)
    return state


def _tos_stdevall_update(
    state: TosStdevallState, bar: Dict[str, Any], params: Dict[str, Any]
) -> Tuple[List[Optional[float]], TosStdevallState]:
    """Update TOS_STDEVALL using O(1) online regression/std."""
    close = float(bar["close"])

    # Emit warning once per state instance
    if not state.warned:
        warnings.warn(
            "TOS_STDEVALL in stateful mode uses online regression (expanding/rolling). "
            "Results may differ from vectorized snapshot behavior.",
            UserWarning,
            stacklevel=2
        )
        state.warned = True

    # Update running sums (expanding or rolling)
    if state.length is None:
        # expanding
        x = state.n  # zero-based
        state.n += 1
        state.sum_x += x
        state.sum_xx += x * x
        state.sum_y += close
        state.sum_y2 += close * close
        state.sum_xy += x * close
        n = state.n
    else:
        # rolling
        L = state.length
        if len(state.y_buf) < L:
            x = len(state.y_buf)
            state.y_buf.append(close)
            state.n = len(state.y_buf)
            state.sum_x += x
            state.sum_xx += x * x
            state.sum_y += close
            state.sum_y2 += close * close
            state.sum_xy += x * close
        else:
            y0 = state.y_buf[0]
            state.y_buf.append(close)
            # update sums for sliding window
            old_sum_y = state.sum_y
            state.sum_y = state.sum_y - y0 + close
            state.sum_y2 = state.sum_y2 - y0 * y0 + close * close
            state.sum_xy = state.sum_xy - old_sum_y + y0 + (L - 1) * close
            state.n = L
        n = state.n

    # Need at least 2 points for regression/std
    if n < 2:
        num_outputs = 1 + 2 * len(state.stds)
        return [None] * num_outputs, state

    # Compute regression coefficients
    if state.length is None:
        sum_x = state.sum_x
        sum_xx = state.sum_xx
    else:
        sum_x = n * (n - 1) / 2.0
        sum_xx = (n - 1) * n * (2 * n - 1) / 6.0

    denom = n * sum_xx - sum_x * sum_x
    if denom == 0.0:
        m = 0.0
    else:
        m = (n * state.sum_xy - sum_x * state.sum_y) / denom
    b = (state.sum_y - m * sum_x) / n

    # Regression value at latest x
    x_cur = n - 1
    lr = m * x_cur + b

    # Standard deviation of y
    if n <= state.ddof:
        stdev = 0.0
    else:
        var = (state.sum_y2 - (state.sum_y * state.sum_y) / n) / (n - state.ddof)
        if var < 0.0:
            var = 0.0
        stdev = var ** 0.5

    outputs: List[Optional[float]] = [lr]
    for std_mult in state.stds:
        outputs.append(lr - std_mult * stdev)
        outputs.append(lr + std_mult * stdev)

    return outputs, state


def _tos_stdevall_output_names(params: Dict[str, Any]) -> List[str]:
    stds_param = params.get("stds", [1, 2, 3])
    if not isinstance(stds_param, (list, tuple)):
        stds_param = [1, 2, 3]
    stds = sorted(stds_param)

    names = ["TOS_STDEVALL_LR"]
    for std_mult in stds:
        names.append(f"TOS_STDEVALL_L_{std_mult}")
        names.append(f"TOS_STDEVALL_U_{std_mult}")
    return names


def _tos_stdevall_seed(series: Dict[str, Any], params: Dict[str, Any]) -> TosStdevallState:
    """Seed TOS_STDEVALL from historical data.

    We need to replay all historical prices to build the expanding window.
    """
    return replay_seed("tos_stdevall", series, params)


LOOKAHEAD_REGISTRY["tos_stdevall"] = StatefulIndicator(
    kind="tos_stdevall",
    inputs=("close",),
    init=_tos_stdevall_init,
    update=_tos_stdevall_update,
    output_names=_tos_stdevall_output_names,
)
SEED_REGISTRY["tos_stdevall"] = _tos_stdevall_seed


# ===========================================================================
# VP  (APPROXIMATE: online histogram with warning)
# ===========================================================================
# Volume Profile bins price/volume into ranges.
# Original is a distribution snapshot, not a time series.
# Stateful version uses a fixed-width histogram (approximate).

@dataclass
class VPState:
    """VP state using fixed-width histogram (approximate)."""
    width: int
    min_price: Optional[float] = None
    max_price: Optional[float] = None
    pos_bins: List[float] = field(default_factory=list)
    neg_bins: List[float] = field(default_factory=list)
    neut_bins: List[float] = field(default_factory=list)
    sum_bins: List[float] = field(default_factory=list)
    count_bins: List[float] = field(default_factory=list)
    last_close: Optional[float] = None
    warned: bool = False


def _vp_init(params: Dict[str, Any]) -> VPState:
    """Initialize VP with expanding window."""
    width = _as_int(_param(params, "width", 10), 10)
    state = VPState(width=width)
    state.pos_bins = [0.0] * width
    state.neg_bins = [0.0] * width
    state.neut_bins = [0.0] * width
    state.sum_bins = [0.0] * width
    state.count_bins = [0.0] * width
    return state


def _vp_bin_index(price: float, min_p: float, max_p: float, width: int) -> int:
    if max_p <= min_p:
        return width // 2
    pos = (price - min_p) / (max_p - min_p)
    idx = int(pos * width)
    if idx < 0:
        return 0
    if idx >= width:
        return width - 1
    return idx


def _vp_rebin(state: VPState, new_min: float, new_max: float) -> None:
    if state.min_price is None or state.max_price is None:
        state.min_price = new_min
        state.max_price = new_max
        return
    old_min = state.min_price
    old_max = state.max_price
    width = state.width
    if old_max <= old_min:
        state.min_price = new_min
        state.max_price = new_max
        return

    pos_new = [0.0] * width
    neg_new = [0.0] * width
    neut_new = [0.0] * width
    sum_new = [0.0] * width
    count_new = [0.0] * width

    old_w = (old_max - old_min) / width
    for i in range(width):
        center = old_min + (i + 0.5) * old_w
        j = _vp_bin_index(center, new_min, new_max, width)
        pos_new[j] += state.pos_bins[i]
        neg_new[j] += state.neg_bins[i]
        neut_new[j] += state.neut_bins[i]
        sum_new[j] += state.sum_bins[i]
        count_new[j] += state.count_bins[i]

    state.pos_bins = pos_new
    state.neg_bins = neg_new
    state.neut_bins = neut_new
    state.sum_bins = sum_new
    state.count_bins = count_new
    state.min_price = new_min
    state.max_price = new_max


def _vp_update(
    state: VPState, bar: Dict[str, Any], params: Dict[str, Any]
) -> Tuple[List[Optional[float]], VPState]:
    """Update VP using fixed-width histogram."""
    close = float(bar["close"])
    volume = float(bar["volume"])

    # Emit warning once per state instance
    if not state.warned:
        warnings.warn(
            "VP in stateful mode uses EXPANDING window (approximate). "
            "Volume Profile is not a time series indicator. "
            "Results will differ significantly from vectorized training data.",
            UserWarning,
            stacklevel=2
        )
        state.warned = True

    # Update min/max and re-bin if needed
    if state.min_price is None or state.max_price is None:
        state.min_price = close
        state.max_price = close
    else:
        new_min = min(state.min_price, close)
        new_max = max(state.max_price, close)
        if new_min != state.min_price or new_max != state.max_price:
            _vp_rebin(state, new_min, new_max)

    # Determine sign (signed_series with initial=1)
    if state.last_close is None:
        sign = 1.0
    else:
        if close > state.last_close:
            sign = 1.0
        elif close < state.last_close:
            sign = -1.0
        else:
            sign = 0.0
    state.last_close = close

    idx = _vp_bin_index(close, state.min_price, state.max_price, state.width)
    if sign > 0:
        state.pos_bins[idx] += volume
    elif sign < 0:
        state.neg_bins[idx] += volume
    else:
        state.neut_bins[idx] += volume
    state.sum_bins[idx] += close
    state.count_bins[idx] += 1.0

    # Find max-volume bin
    max_idx = None
    max_total = None
    for i in range(state.width):
        total = state.pos_bins[i] + state.neg_bins[i] + state.neut_bins[i]
        if max_total is None or total > max_total:
            max_total = total
            max_idx = i

    if max_idx is None:
        return [None] * 7, state

    # Compute bin boundaries and stats
    if state.max_price <= state.min_price:
        low_p = high_p = state.min_price
    else:
        bin_w = (state.max_price - state.min_price) / state.width
        low_p = state.min_price + max_idx * bin_w
        high_p = low_p + bin_w

    if state.count_bins[max_idx] > 0:
        mean_price = state.sum_bins[max_idx] / state.count_bins[max_idx]
    else:
        mean_price = 0.5 * (low_p + high_p)

    pos_vol = state.pos_bins[max_idx]
    neg_vol = state.neg_bins[max_idx]
    neut_vol = state.neut_bins[max_idx]
    total_volume = pos_vol + neg_vol

    return [
        low_p,
        mean_price,
        high_p,
        pos_vol,
        neg_vol,
        neut_vol,
        total_volume,
    ], state


def _vp_output_names(params: Dict[str, Any]) -> List[str]:
    width = _as_int(_param(params, "width", 10), 10)
    # Using generic close/volume names since input names vary
    return [
        f"VP_{width}_low_price",
        f"VP_{width}_mean_price",
        f"VP_{width}_high_price",
        f"VP_{width}_pos_volume",
        f"VP_{width}_neg_volume",
        f"VP_{width}_neut_volume",
        f"VP_{width}_total_volume",
    ]


def _vp_seed(series: Dict[str, Any], params: Dict[str, Any]) -> VPState:
    """Seed VP from historical data.

    We need to replay all historical prices/volumes to build the expanding window.
    """
    return replay_seed("vp", series, params)


LOOKAHEAD_REGISTRY["vp"] = StatefulIndicator(
    kind="vp",
    inputs=("close", "volume"),
    init=_vp_init,
    update=_vp_update,
    output_names=_vp_output_names,
)
SEED_REGISTRY["vp"] = _vp_seed
